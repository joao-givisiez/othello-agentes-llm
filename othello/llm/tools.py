"""Ferramentas (tools) LangChain com as quais o agente LLM percebe e age no jogo.

Ferramentas básicas (exigidas no enunciado) - disponíveis nos dois modos:

* ``get_board_state``  - observa o tabuleiro (ASCII com coordenadas).
* ``get_valid_moves``  - lista as jogadas válidas e as peças que cada uma vira.
* ``make_move``        - executa a jogada no ambiente (única ação do agente).
* ``get_game_status``  - placar, vez, casas vazias, fim de jogo.
* ``get_move_history`` - histórico recente da partida (memória).

Ferramentas de análise e planejamento - apenas no modo ``planner``:

* ``analyze_moves``    - características estratégicas de cada jogada.
* ``simulate_move``    - simula uma jogada (sem executá-la) e mostra as
                         melhores respostas do adversário.
* ``minimax_search``   - busca alfa-beta que devolve o valor de cada jogada.

As ferramentas são funções fechadas sobre um ``ToolContext`` que aponta para
a partida e o jogador do turno atual, e registram cada chamada para as
métricas dos experimentos.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

from langchain_core.tools import BaseTool, tool

from ..agents.evaluation import analyze_move
from ..agents.minimax import score_moves
from ..game import (
    PLAYER_NAMES,
    IllegalMoveError,
    OthelloGame,
    opponent,
    parse_square,
    to_notation,
)
from ..game.board import PLAYER_SYMBOLS

BASIC_TOOL_NAMES = (
    "get_board_state",
    "get_valid_moves",
    "make_move",
    "get_game_status",
    "get_move_history",
)
PLANNER_TOOL_NAMES = ("analyze_moves", "simulate_move", "minimax_search")


def _json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=1)


@dataclass
class ToolContext:
    """Estado compartilhado entre as ferramentas durante um turno."""

    game: OthelloGame | None = None
    player: int = 1
    actor: str = "LLM"
    move_made: str | None = None
    explanation: str = ""
    tool_calls: int = 0
    invalid_attempts: int = 0
    tool_log: list[str] = field(default_factory=list)
    lock: threading.RLock = field(default_factory=threading.RLock)

    def new_turn(self, game: OthelloGame, player: int, actor: str) -> None:
        with self.lock:
            self.game = game
            self.player = player
            self.actor = actor
            self.move_made = None
            self.explanation = ""
            self.tool_calls = 0
            self.invalid_attempts = 0
            self.tool_log = []

    def record(self, name: str) -> OthelloGame:
        with self.lock:
            self.tool_calls += 1
            self.tool_log.append(name)
        if self.game is None:
            raise RuntimeError("Nenhuma partida associada ao agente")
        return self.game

    @property
    def color(self) -> str:
        return f"{PLAYER_NAMES[self.player]} ({PLAYER_SYMBOLS[self.player]})"


def build_tools(ctx: ToolContext, mode: str = "basic") -> list[BaseTool]:
    """Cria as ferramentas do agente. ``mode`` = "basic" ou "planner"."""

    # ------------------------------------------------------------------
    # Ferramentas básicas
    # ------------------------------------------------------------------
    @tool
    def get_board_state() -> str:
        """Mostra o tabuleiro atual 8x8 com coordenadas (colunas a-h, linhas 1-8).
        X = Pretas, O = Brancas, . = vazio, * = jogada válida para quem está na vez.
        Use para observar a posição antes de decidir."""
        game = ctx.record("get_board_state")
        score = game.score()
        header = "Tabuleiro (X = Pretas, O = Brancas, . = vazio, * = jogada válida da vez):"
        footer = (
            f"Você joga com: {ctx.color}. Vez de: "
            f"{PLAYER_NAMES[game.current_player] if not game.game_over else 'ninguém (fim)'}. "
            f"Placar: Pretas {score['pretas']} x {score['brancas']} Brancas. "
            f"Casas vazias: {game.board.empty_count()}."
        )
        return f"{header}\n{game.render(show_moves=True)}\n{footer}"

    @tool
    def get_valid_moves() -> str:
        """Lista as jogadas válidas do jogador da vez e quais peças cada uma vira.
        Só é permitido jogar em uma das casas desta lista."""
        game = ctx.record("get_valid_moves")
        moves = []
        for move in game.valid_moves():
            flips = game.board.flips(game.current_player, move)
            moves.append(
                {
                    "jogada": to_notation(move),
                    "pecas_viradas": len(flips),
                    "vira": [to_notation(f) for f in flips],
                }
            )
        return _json({"vez_de": PLAYER_NAMES[game.current_player], "jogadas_validas": moves})

    @tool(parse_docstring=True)
    def make_move(position: str, reason: str) -> str:
        """Executa a sua jogada no tabuleiro. Esta é a ÚNICA forma de agir no jogo
        e encerra o seu turno.

        Args:
            position: casa escolhida em notação coluna+linha, por exemplo "d3".
                Deve ser uma das jogadas retornadas por get_valid_moves.
            reason: justificativa curta (1 ou 2 frases, em português) explicando
                por que esta jogada foi escolhida.
        """
        game = ctx.record("make_move")
        with ctx.lock:
            if ctx.move_made:
                return f"Você já jogou neste turno ({ctx.move_made}). Não chame make_move de novo."
            if game.game_over:
                return "A partida já terminou; não há jogada a fazer."
            if game.current_player != ctx.player:
                return f"Não é a sua vez: agora jogam as {PLAYER_NAMES[game.current_player]}."
            try:
                record = game.make_move(
                    position, actor=ctx.actor, explanation=reason.strip(), player=ctx.player
                )
            except IllegalMoveError as exc:
                ctx.invalid_attempts += 1
                valid = ", ".join(to_notation(m) for m in game.valid_moves()) or "nenhuma"
                return (
                    f"ERRO: jogada inválida ({exc}). Nada foi executado. "
                    f"Jogadas válidas: {valid}. Escolha uma delas e chame make_move novamente."
                )
            ctx.move_made = record.square
            ctx.explanation = reason.strip()
        return (
            f"Jogada executada: {record.square}, virando {len(record.flipped)} peça(s) "
            f"({', '.join(record.flipped)}). Placar: Pretas {record.black_count} x "
            f"{record.white_count} Brancas. Seu turno terminou."
        )

    @tool
    def get_game_status() -> str:
        """Resumo da partida: se terminou, de quem é a vez, placar, número de
        jogadas realizadas, casas vazias e última jogada."""
        game = ctx.record("get_game_status")
        status = game.status()
        status["voce_joga_com"] = ctx.color
        return _json(status)

    @tool(parse_docstring=True)
    def get_move_history(last_n: int = 10) -> str:
        """Histórico das últimas jogadas da partida (memória do jogo), da mais
        antiga para a mais recente, incluindo passes.

        Args:
            last_n: quantas entradas recentes retornar (1 a 60).
        """
        game = ctx.record("get_move_history")
        last_n = max(1, min(int(last_n), 60))
        lines = game.history_lines(last_n=last_n)
        return "\n".join(lines) if lines else "Nenhuma jogada foi feita ainda."

    tools: list[BaseTool] = [
        get_board_state,
        get_valid_moves,
        make_move,
        get_game_status,
        get_move_history,
    ]
    if mode == "basic":
        return tools

    # ------------------------------------------------------------------
    # Ferramentas de análise e planejamento (modo planner)
    # ------------------------------------------------------------------
    @tool
    def analyze_moves() -> str:
        """Analisa estrategicamente cada jogada válida: peças viradas, tipo de casa
        (canto, borda, casa X/C perigosa), se entrega canto ao adversário,
        mobilidade do adversário depois, peças de fronteira e uma avaliação
        heurística (maior = melhor para você). Lista ordenada da melhor para a pior."""
        game = ctx.record("analyze_moves")
        player = game.current_player
        analyses = [analyze_move(game.board, player, m) for m in game.valid_moves()]
        analyses.sort(key=lambda a: a["avaliacao_heuristica"], reverse=True)
        return _json(analyses)

    @tool(parse_docstring=True)
    def simulate_move(position: str) -> str:
        """Simula uma jogada SEM executá-la: mostra o tabuleiro resultante e as
        melhores respostas do adversário, para planejar um lance à frente.

        Args:
            position: casa a simular, por exemplo "c4".
        """
        game = ctx.record("simulate_move")
        player = game.current_player
        try:
            move = parse_square(position)
            if not game.board.is_legal(player, move):
                raise ValueError(f"{position} não é uma jogada válida agora")
        except ValueError as exc:
            valid = ", ".join(to_notation(m) for m in game.valid_moves())
            return f"ERRO: {exc}. Jogadas válidas: {valid}"
        after = game.board.apply_move(player, move)
        rival = opponent(player)
        replies = [analyze_move(after, rival, m) for m in after.legal_moves(rival)]
        replies.sort(key=lambda a: a["avaliacao_heuristica"], reverse=True)
        result = {
            "jogada_simulada": to_notation(move),
            "tabuleiro_depois": after.to_ascii(set(after.legal_moves(rival))).split("\n"),
            "placar_depois": {"pretas": after.count(1), "brancas": after.count(-1)},
            "respostas_do_adversario": len(replies),
            "melhores_respostas_do_adversario": [
                {
                    "jogada": r["jogada"],
                    "tipo_de_casa": r["tipo_de_casa"],
                    "pecas_viradas": r["pecas_viradas"],
                    "avaliacao_para_o_adversario": r["avaliacao_heuristica"],
                }
                for r in replies[:3]
            ],
            "adversario_pode_pegar_canto": any(r["tipo_de_casa"] == "canto" for r in replies),
        }
        return _json(result)

    @tool(parse_docstring=True)
    def minimax_search(depth: int = 4) -> str:
        """Planejamento por busca Minimax com poda alfa-beta: simula `depth`
        lances alternados (você e o adversário jogando da melhor forma) e devolve
        o valor de cada jogada válida, da melhor para a pior. Valores maiores são
        melhores para você; "vitória garantida" indica vitória forçada.

        Args:
            depth: profundidade da busca em lances (1 a 6). Padrão 4.
        """
        game = ctx.record("minimax_search")
        depth = max(1, min(int(depth), 6))
        if game.board.empty_count() <= 10:
            depth = max(depth, game.board.empty_count() + 2)  # fim de jogo exato
        ranking = score_moves(game.board, game.current_player, depth)
        return _json({"profundidade": depth, "ranking": ranking})

    return tools + [analyze_moves, simulate_move, minimax_search]
