"""Interface comum dos agentes.

Todo agente implementa ``play_turn(game)``: ele observa o ambiente, decide e
EXECUTA a jogada no próprio ambiente (``game.make_move``), devolvendo um
``TurnResult`` com a jogada, a justificativa e métricas de custo. Isso deixa
o laço da partida idêntico para humanos, agentes clássicos e agentes LLM - no
caso do LLM quem executa a jogada é a ferramenta ``make_move``.
"""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from ..game import Board, Move, OthelloGame, to_notation


@dataclass
class TurnResult:
    """Resultado de um turno jogado por um agente."""

    move: str  # notação da jogada executada ("d3")
    explanation: str = ""
    elapsed: float = 0.0  # tempo total de decisão (s)
    llm_calls: int = 0
    llm_time: float = 0.0  # tempo gasto esperando o LLM (s)
    tool_calls: int = 0
    invalid_attempts: int = 0  # jogadas inválidas tentadas antes da válida
    fallback: bool = False  # True se o agente LLM precisou de jogada de segurança
    details: dict[str, Any] = field(default_factory=dict)


class Agent(ABC):
    """Agente que joga Othello."""

    name: str = "agente"
    is_human: bool = False

    def reset(self) -> None:
        """Chamado no início de cada partida (limpa memória/estado interno)."""

    @abstractmethod
    def play_turn(self, game: OthelloGame) -> TurnResult:
        """Decide e executa a jogada do jogador da vez em ``game``."""

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.name!r}>"


class SearchAgent(Agent):
    """Base dos agentes clássicos: basta implementar ``select_move``."""

    def __init__(self, name: str, seed: int | None = None) -> None:
        self.name = name
        self.rng = random.Random(seed)

    @abstractmethod
    def select_move(self, board: Board, player: int) -> tuple[Move, str, dict[str, Any]]:
        """Devolve (jogada, justificativa, detalhes)."""

    def play_turn(self, game: OthelloGame) -> TurnResult:
        player = game.current_player
        start = time.perf_counter()
        move, explanation, details = self.select_move(game.board, player)
        elapsed = time.perf_counter() - start
        game.make_move(move, actor=self.name, explanation=explanation, player=player)
        return TurnResult(
            move=to_notation(move),
            explanation=explanation,
            elapsed=elapsed,
            details=details,
        )

    def _pick_best(self, scored: list[tuple[float, Move]]) -> tuple[float, Move]:
        """Maior pontuação; empates são decididos aleatoriamente (semente fixa)."""
        best = max(score for score, _ in scored)
        candidates = [m for score, m in scored if score == best]
        return best, self.rng.choice(candidates)
