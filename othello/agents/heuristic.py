"""Agente heurístico: avalia cada jogada com a função de avaliação (1 lance)."""

from __future__ import annotations

from typing import Any

from ..game import Board, Move
from .base import SearchAgent
from .evaluation import analyze_move, describe_analysis, evaluate


class HeuristicAgent(SearchAgent):
    """Escolhe a jogada cuja posição resultante tem a melhor avaliação heurística.

    Considera cantos, casas perigosas perto de cantos, mobilidade, fronteira,
    tabela posicional e paridade, mas não antecipa a resposta do adversário.
    """

    def __init__(self, seed: int | None = None, name: str = "Heurístico") -> None:
        super().__init__(name=name, seed=seed)

    def select_move(self, board: Board, player: int) -> tuple[Move, str, dict[str, Any]]:
        scored = [
            (evaluate(board.apply_move(player, m), player), m) for m in board.legal_moves(player)
        ]
        best, move = self._pick_best(scored)
        info = analyze_move(board, player, move)
        explanation = f"Melhor avaliação heurística ({best:.0f}): {describe_analysis(info)}."
        return move, explanation, {"avaliacao": best}
