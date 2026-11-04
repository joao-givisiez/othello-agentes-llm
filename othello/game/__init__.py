"""Regras do Othello e ambiente de jogo."""

from .board import (
    BLACK,
    COLUMNS,
    EMPTY,
    PLAYER_NAMES,
    WHITE,
    Board,
    IllegalMoveError,
    Move,
    opponent,
    parse_square,
    to_notation,
)
from .environment import MoveRecord, OthelloGame

__all__ = [
    "BLACK",
    "WHITE",
    "EMPTY",
    "COLUMNS",
    "PLAYER_NAMES",
    "Board",
    "IllegalMoveError",
    "Move",
    "MoveRecord",
    "OthelloGame",
    "opponent",
    "parse_square",
    "to_notation",
]
