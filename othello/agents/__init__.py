"""Agentes clássicos de Othello e interface comum."""

from .base import Agent, SearchAgent, TurnResult
from .evaluation import analyze_move, evaluate
from .heuristic import HeuristicAgent
from .minimax import MinimaxAgent, score_moves
from .simple import GreedyAgent, RandomAgent

__all__ = [
    "Agent",
    "SearchAgent",
    "TurnResult",
    "RandomAgent",
    "GreedyAgent",
    "HeuristicAgent",
    "MinimaxAgent",
    "analyze_move",
    "evaluate",
    "score_moves",
]
