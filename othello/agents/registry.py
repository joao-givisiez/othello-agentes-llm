"""Criação de agentes a partir de uma especificação textual.

Especificações aceitas (usadas na linha de comando, na interface e nos
experimentos):

==================  ==========================================================
``random``          Agente aleatório
``greedy``          Agente guloso (maximiza peças viradas)
``heuristic``       Agente heurístico de 1 lance
``minimax[:N]``     Minimax alfa-beta com profundidade N (padrão 4)
``llm-basic``       Agente LLM com as ferramentas básicas do jogo
``llm-planner``     Agente LLM com ferramentas de análise e planejamento
==================  ==========================================================
"""

from __future__ import annotations

from .base import Agent
from .heuristic import HeuristicAgent
from .minimax import MinimaxAgent
from .simple import GreedyAgent, RandomAgent

AGENT_CHOICES: dict[str, str] = {
    "random": "Aleatório",
    "greedy": "Guloso",
    "heuristic": "Heurístico",
    "minimax": "Minimax alfa-beta",
    "llm-basic": "LLM básico (ferramentas do jogo)",
    "llm-planner": "LLM planejador (ferramentas + análise + minimax)",
}


def create_agent(spec: str, seed: int | None = None, model: str | None = None) -> Agent:
    """Cria um agente a partir de ``spec`` (ver tabela no topo do módulo)."""
    kind, _, arg = spec.strip().lower().partition(":")
    if kind == "random":
        return RandomAgent(seed=seed)
    if kind == "greedy":
        return GreedyAgent(seed=seed)
    if kind == "heuristic":
        return HeuristicAgent(seed=seed)
    if kind == "minimax":
        depth = int(arg) if arg else 4
        return MinimaxAgent(depth=depth, endgame_empties=10, seed=seed)
    if kind in ("llm-basic", "llm-planner"):
        # Import tardio: os agentes clássicos funcionam sem LangChain instalado.
        from ..llm.agent import LLMAgent

        mode = "planner" if kind == "llm-planner" else "basic"
        return LLMAgent(mode=mode, model=model or (arg or None))
    raise ValueError(f"Agente desconhecido: {spec!r}. Opções: {', '.join(AGENT_CHOICES)}")
