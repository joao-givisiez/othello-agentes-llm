"""Interface gráfica em Tkinter.

* Tabuleiro clicável com coordenadas, jogadas válidas destacadas e marcação
  da última jogada.
* Escolha de quem joga com as pretas e com as brancas (Humano, agentes
  clássicos ou agentes LLM): Humano x Agente, Agente x Agente ou Humano x
  Humano.
* Painel de log com cada jogada, a justificativa do agente, as ferramentas
  chamadas pelo LLM e os tempos de resposta.
* Os agentes jogam numa thread separada para a janela não congelar enquanto
  o LLM "pensa"; o resultado volta para a thread da interface por uma fila.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from ..agents.base import Agent
from ..agents.registry import create_agent
from ..game import BLACK, EMPTY, PLAYER_NAMES, WHITE, IllegalMoveError, OthelloGame
from ..game.board import COLUMNS, parse_square
from .common import AGENT_OPTIONS, describe_turn, is_llm_spec, llm_warning

CELL = 64
MARGIN = 28
BOARD_PX = MARGIN + 8 * CELL + 8

BOARD_COLOR = "#1f7a3a"
LINE_COLOR = "#0d3b1c"
HINT_COLOR = "#5fae78"
LAST_MOVE_COLOR = "#e53935"
DISC_COLORS = {BLACK: ("#111111", "#000000"), WHITE: ("#f4f4f4", "#9e9e9e")}
SYMBOLS = {BLACK: "●", WHITE: "○"}


class OthelloApp:
    def __init__(
        self,
        root: tk.Tk,
        model: str | None = None,
        black: str = "Humano",
        white: str = "Minimax (prof. 4)",
    ) -> None:
        from ..llm.config import resolve_model_spec

        self.root = root
        self.root.title("Othello - Agentes Inteligentes")
        self.root.resizable(False, False)

        self.black_var = tk.StringVar(value=black)
        self.white_var = tk.StringVar(value=white)
        self.model_var = tk.StringVar(value=resolve_model_spec(model))
        self.delay_var = tk.IntVar(value=600)
        self.hints_var = tk.BooleanVar(value=True)

        self.queue: queue.Queue = queue.Queue()
        self.generation = 0
        self.worker_busy = False
        self.paused = False
        self.game = OthelloGame()
        self.players: dict[int, Agent | None] = {BLACK: None, WHITE: None}

        self._build_widgets()
        self.new_game()
        self.root.after(100, self._poll_queue)

    # ------------------------------------------------------------------
    # Construção da janela
    # ------------------------------------------------------------------
    def _build_widgets(self) -> None:
        main = ttk.Frame(self.root, padding=8)
        main.grid(row=0, column=0, sticky="nsew")

        self.canvas = tk.Canvas(main, width=BOARD_PX, height=BOARD_PX, bg="#e8e8e8", highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=2, sticky="n")
        self.canvas.bind("<Button-1>", self._on_click)

        panel = ttk.Frame(main, padding=(12, 0, 0, 0))
        panel.grid(row=0, column=1, sticky="nsew")

        players = ttk.LabelFrame(panel, text="Jogadores", padding=8)
        players.grid(row=0, column=0, sticky="ew")
        ttk.Label(players, text="Pretas ●").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            players, textvariable=self.black_var, values=list(AGENT_OPTIONS), state="readonly", width=22
        ).grid(row=0, column=1, padx=4, pady=2)
        ttk.Label(players, text="Brancas ○").grid(row=1, column=0, sticky="w")
        ttk.Combobox(
            players, textvariable=self.white_var, values=list(AGENT_OPTIONS), state="readonly", width=22
        ).grid(row=1, column=1, padx=4, pady=2)
        ttk.Label(players, text="Modelo LLM").grid(row=2, column=0, sticky="w")
        ttk.Entry(players, textvariable=self.model_var, width=25).grid(row=2, column=1, padx=4, pady=2)

        options = ttk.LabelFrame(panel, text="Opções", padding=8)
        options.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        ttk.Label(options, text="Atraso entre jogadas dos agentes (ms)").grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Scale(options, from_=0, to=2000, variable=self.delay_var, orient="horizontal", length=240).grid(
            row=1, column=0, columnspan=2, sticky="ew"
        )
        ttk.Checkbutton(
            options, text="Mostrar jogadas válidas", variable=self.hints_var, command=self._redraw
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(4, 0))

        buttons = ttk.Frame(panel)
        buttons.grid(row=2, column=0, sticky="ew", pady=8)
        ttk.Button(buttons, text="Nova partida", command=self.new_game).grid(row=0, column=0, padx=(0, 4))
        self.pause_button = ttk.Button(buttons, text="Pausar", command=self.toggle_pause)
        self.pause_button.grid(row=0, column=1, padx=4)
        self.step_button = ttk.Button(buttons, text="Próxima jogada", command=self.step, state="disabled")
        self.step_button.grid(row=0, column=2, padx=4)

        self.score_label = ttk.Label(panel, font=("Segoe UI", 16, "bold"))
        self.score_label.grid(row=3, column=0, sticky="w")
        self.turn_label = ttk.Label(panel, font=("Segoe UI", 11))
        self.turn_label.grid(row=4, column=0, sticky="w")
        self.status_label = ttk.Label(panel, foreground="#555555", wraplength=330)
        self.status_label.grid(row=5, column=0, sticky="w", pady=(2, 6))

        self.log = ScrolledText(main, width=62, height=14, wrap="word", font=("Consolas", 9), state="disabled")
        self.log.grid(row=1, column=1, sticky="nsew", padx=(12, 0))
        self.log.tag_configure("header", font=("Consolas", 9, "bold"))
        self.log.tag_configure("error", foreground="#c62828")
        self.log.tag_configure("final", font=("Consolas", 10, "bold"), foreground="#1b5e20")
        self.log.tag_configure("pass", foreground="#8d6e63")

    # ------------------------------------------------------------------
    # Controle da partida
    # ------------------------------------------------------------------
    def new_game(self) -> None:
        self.generation += 1
        self.worker_busy = False
        self.paused = False
        self.pause_button.configure(text="Pausar")
        self.step_button.configure(state="disabled")
        self.game = OthelloGame()
        self._clear_log()

        model = self.model_var.get().strip() or None
        self.players = {
            BLACK: self._create_player(self.black_var.get(), model),
            WHITE: self._create_player(self.white_var.get(), model),
        }
        names = {p: (a.name if a else "Humano") for p, a in self.players.items()}
        self._log(f"Nova partida: Pretas = {names[BLACK]}  x  Brancas = {names[WHITE]}", "header")
        if any(is_llm_spec(AGENT_OPTIONS[v.get()]) for v in (self.black_var, self.white_var)):
            self._log(f"Modelo LLM: {model}")
            warning = llm_warning(model)
            if warning:
                self._log(f"AVISO: {warning}", "error")
        self._redraw()
        self._schedule_next()

    def _create_player(self, label: str, model: str | None) -> Agent | None:
        spec = AGENT_OPTIONS.get(label)
        if spec is None:
            return None
        try:
            return create_agent(spec, model=model)
        except Exception as exc:  # ex.: pacote do provedor não instalado
            messagebox.showerror("Erro ao criar agente", f"{label}: {exc}\nUsando jogador humano.")
            return None

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.pause_button.configure(text="Continuar" if self.paused else "Pausar")
        self.step_button.configure(state="normal" if self.paused else "disabled")
        if not self.paused:
            self._schedule_next()
        else:
            self._set_status("Pausado. Use 'Próxima jogada' para avançar um lance.")

    def step(self) -> None:
        if self.paused:
            self._start_agent(self.generation, force=True)

    def _current_agent(self) -> Agent | None:
        return self.players.get(self.game.current_player)

    def _schedule_next(self) -> None:
        self._update_labels()
        if self.game.game_over:
            self._announce_result()
            return
        agent = self._current_agent()
        if agent is None:
            self._set_status(f"Sua vez ({PLAYER_NAMES[self.game.current_player]}): clique em uma casa destacada.")
            return
        if self.paused or self.worker_busy:
            return
        both_agents = all(self.players.values())
        delay = int(self.delay_var.get()) if both_agents else 250
        self.root.after(delay, self._start_agent, self.generation)

    def _start_agent(self, generation: int, force: bool = False) -> None:
        if generation != self.generation or self.worker_busy or self.game.game_over:
            return
        if self.paused and not force:
            return
        agent = self._current_agent()
        if agent is None:
            return
        self.worker_busy = True
        self._set_status(f"{agent.name} ({PLAYER_NAMES[self.game.current_player]}) está pensando...")
        game = self.game
        history_size = len(game.history)

        def work() -> None:
            try:
                result = agent.play_turn(game)
                self.queue.put(("ok", generation, agent, (result, history_size)))
            except Exception as exc:  # erro inesperado do agente
                self.queue.put(("error", generation, agent, exc))

        threading.Thread(target=work, daemon=True).start()

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, generation, agent, payload = self.queue.get_nowait()
                if generation != self.generation:
                    continue  # resultado de uma partida já descartada
                self.worker_busy = False
                if kind == "ok":
                    result, history_size = payload
                    color = PLAYER_NAMES[self.game.history[history_size].player]
                    self._log(f"[{color}] {describe_turn(agent, result)}")
                    self._log_passes(history_size + 1)
                    self._redraw()
                    self._schedule_next()
                else:
                    self._log(f"Erro no agente {agent.name}: {payload}", "error")
                    messagebox.showerror("Erro no agente", f"{agent.name}: {payload}")
                    if not self.paused:
                        self.toggle_pause()
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _on_click(self, event: tk.Event) -> None:
        if self.game.game_over or self.worker_busy or self._current_agent() is not None:
            return
        col = (event.x - MARGIN) // CELL
        row = (event.y - MARGIN) // CELL
        if not (0 <= row < 8 and 0 <= col < 8):
            return
        player = self.game.current_player
        history_size = len(self.game.history)
        try:
            record = self.game.make_move((row, col), actor="Humano", player=player)
        except IllegalMoveError as exc:
            self._set_status(f"Jogada inválida: {exc}")
            return
        self._log(f"[{PLAYER_NAMES[player]}] Humano jogou {record.square} e virou {len(record.flipped)} peça(s)")
        self._log_passes(history_size + 1)
        self._redraw()
        self._schedule_next()

    def _log_passes(self, start: int) -> None:
        for record in self.game.history[start:]:
            if record.is_pass:
                self._log(f"    {PLAYER_NAMES[record.player]} sem jogadas válidas: passam a vez.", "pass")

    def _announce_result(self) -> None:
        score = self.game.score()
        winner = self.game.winner()
        if winner == EMPTY:
            text = f"Fim de jogo: empate ({score['pretas']} x {score['brancas']})."
        else:
            agent = self.players[winner]
            who = agent.name if agent else "Humano"
            text = (
                f"Fim de jogo: vitória das {PLAYER_NAMES[winner]} ({who}) por "
                f"{score['pretas']} x {score['brancas']}."
            )
        self._set_status(text)
        self._log(text, "final")

    # ------------------------------------------------------------------
    # Desenho
    # ------------------------------------------------------------------
    def _redraw(self) -> None:
        c = self.canvas
        c.delete("all")
        x0 = y0 = MARGIN
        size = 8 * CELL
        c.create_rectangle(x0, y0, x0 + size, y0 + size, fill=BOARD_COLOR, outline=LINE_COLOR, width=2)
        for i in range(9):
            c.create_line(x0 + i * CELL, y0, x0 + i * CELL, y0 + size, fill=LINE_COLOR)
            c.create_line(x0, y0 + i * CELL, x0 + size, y0 + i * CELL, fill=LINE_COLOR)
        for i in range(8):
            c.create_text(x0 + i * CELL + CELL / 2, y0 - 12, text=COLUMNS[i], font=("Segoe UI", 11, "bold"))
            c.create_text(x0 - 13, y0 + i * CELL + CELL / 2, text=str(i + 1), font=("Segoe UI", 11, "bold"))
        for square in ("c3", "f3", "c6", "f6"):  # marcações tradicionais do tabuleiro
            r, col = parse_square(square)
            cx, cy = x0 + col * CELL, y0 + r * CELL
            c.create_oval(cx + CELL - 4, cy + CELL - 4, cx + CELL + 4, cy + CELL + 4, fill=LINE_COLOR, outline="")

        board = self.game.board
        pad = 6
        for r in range(8):
            for col in range(8):
                piece = board.get((r, col))
                if piece == EMPTY:
                    continue
                fill, outline = DISC_COLORS[piece]
                cx, cy = x0 + col * CELL, y0 + r * CELL
                c.create_oval(cx + pad, cy + pad, cx + CELL - pad, cy + CELL - pad, fill=fill, outline=outline, width=2)

        human_turn = self._current_agent() is None and not self.game.game_over
        if self.hints_var.get() and (human_turn or all(self.players.values())):
            for r, col in self.game.valid_moves():
                cx, cy = x0 + col * CELL + CELL / 2, y0 + r * CELL + CELL / 2
                c.create_oval(cx - 7, cy - 7, cx + 7, cy + 7, fill=HINT_COLOR, outline="")

        last = self.game.last_move()
        if last is not None:
            r, col = parse_square(last.square)
            cx, cy = x0 + col * CELL + CELL / 2, y0 + r * CELL + CELL / 2
            c.create_oval(cx - 5, cy - 5, cx + 5, cy + 5, fill=LAST_MOVE_COLOR, outline="")
        self._update_labels()

    def _update_labels(self) -> None:
        score = self.game.score()
        self.score_label.configure(text=f"● {score['pretas']}   x   {score['brancas']} ○")
        if self.game.game_over:
            self.turn_label.configure(text="Partida encerrada")
        else:
            player = self.game.current_player
            agent = self.players.get(player)
            who = agent.name if agent else "Humano"
            self.turn_label.configure(
                text=f"Vez das {PLAYER_NAMES[player]} {SYMBOLS[player]} ({who}) - jogada {self.game.moves_played() + 1}"
            )

    # ------------------------------------------------------------------
    # Log
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self.status_label.configure(text=text)

    def _log(self, text: str, tag: str | None = None) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n", tag or ())
        self.log.see("end")
        self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")


def run_gui(model: str | None = None, black: str = "Humano", white: str = "Minimax (prof. 4)") -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista" if "vista" in ttk.Style().theme_names() else "clam")
    except tk.TclError:
        pass
    OthelloApp(root, model=model, black=black, white=white)
    root.mainloop()
