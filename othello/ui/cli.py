"""Interface de terminal: Humano x Agente, Agente x Agente e Humano x Humano."""

from __future__ import annotations

import time

from ..agents.base import Agent, TurnResult
from ..agents.registry import create_agent
from ..game import BLACK, EMPTY, PLAYER_NAMES, WHITE, IllegalMoveError, OthelloGame, to_notation
from .common import describe_turn, is_llm_spec, llm_warning


class HumanCLIAgent(Agent):
    """Jogador humano que digita as jogadas no terminal."""

    is_human = True

    def __init__(self, name: str = "Humano") -> None:
        self.name = name

    def play_turn(self, game: OthelloGame) -> TurnResult:
        valid = ", ".join(to_notation(m) for m in game.valid_moves())
        while True:
            text = input(f"{self.name} ({PLAYER_NAMES[game.current_player]}) - sua jogada [{valid}]: ")
            command = text.strip().lower()
            if command in ("sair", "quit", "exit"):
                raise KeyboardInterrupt
            if command in ("historico", "histórico", "h"):
                print("\n".join(game.history_lines()) or "Sem jogadas ainda.")
                continue
            if command in ("?", "ajuda"):
                print("Digite a casa (ex.: d3), 'historico' ou 'sair'.")
                continue
            start = time.perf_counter()
            try:
                record = game.make_move(command, actor=self.name)
            except IllegalMoveError as exc:
                print(f"  Jogada inválida: {exc}")
                continue
            return TurnResult(move=record.square, elapsed=time.perf_counter() - start)


def _make_player(spec: str, model: str | None, seed: int | None) -> Agent:
    if spec == "human":
        return HumanCLIAgent()
    return create_agent(spec, seed=seed, model=model)


def run_cli(black: str, white: str, model: str | None = None, delay: float = 0.0, seed: int | None = None) -> None:
    for spec in (black, white):
        if is_llm_spec(spec) and (warning := llm_warning(model)):
            print(f"AVISO: {warning}\n")
            break
    players = {BLACK: _make_player(black, model, seed), WHITE: _make_player(white, model, seed)}
    game = OthelloGame()
    print(f"Othello - Pretas (X): {players[BLACK].name}  x  Brancas (O): {players[WHITE].name}\n")
    try:
        while not game.game_over:
            player = game.current_player
            agent = players[player]
            score = game.score()
            print(game.render())
            print(f"Placar: Pretas {score['pretas']} x {score['brancas']} Brancas | vez das {PLAYER_NAMES[player]}\n")
            if not agent.is_human:
                print(f"{agent.name} está pensando...")
            history_size = len(game.history)
            result = agent.play_turn(game)
            if not agent.is_human:
                print(describe_turn(agent, result))
            for record in game.history[history_size + 1 :]:
                if record.is_pass:
                    print(f"  >> {PLAYER_NAMES[record.player]} sem jogadas: passam a vez.")
            print()
            if delay and not agent.is_human:
                time.sleep(delay)
    except KeyboardInterrupt:
        print("\nPartida interrompida.")
        return

    print(game.render(show_moves=False))
    score = game.score()
    winner = game.winner()
    result = "Empate!" if winner == EMPTY else f"Vitória das {PLAYER_NAMES[winner]} ({players[winner].name})!"
    print(f"\nFim de jogo - Pretas {score['pretas']} x {score['brancas']} Brancas. {result}")
