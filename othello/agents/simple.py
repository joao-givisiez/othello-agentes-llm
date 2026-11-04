"""Agentes de referência (baselines): aleatório e guloso."""

from __future__ import annotations

from typing import Any

from ..game import Board, Move, to_notation
from .base import SearchAgent


class RandomAgent(SearchAgent):
    """Escolhe uniformemente uma jogada válida. Serve de linha de base."""

    def __init__(self, seed: int | None = None, name: str = "Aleatório") -> None:
        super().__init__(name=name, seed=seed)

    def select_move(self, board: Board, player: int) -> tuple[Move, str, dict[str, Any]]:
        moves = board.legal_moves(player)
        move = self.rng.choice(moves)
        return move, f"Jogada aleatória entre {len(moves)} opções.", {"opcoes": len(moves)}


class GreedyAgent(SearchAgent):
    """Maximiza o número de peças viradas na jogada atual (visão de 1 lance)."""

    def __init__(self, seed: int | None = None, name: str = "Guloso") -> None:
        super().__init__(name=name, seed=seed)

    def select_move(self, board: Board, player: int) -> tuple[Move, str, dict[str, Any]]:
        scored = [(float(len(board.flips(player, m))), m) for m in board.legal_moves(player)]
        best, move = self._pick_best(scored)
        explanation = f"{to_notation(move)} vira o maior número de peças ({int(best)})."
        return move, explanation, {"pecas_viradas": int(best)}
