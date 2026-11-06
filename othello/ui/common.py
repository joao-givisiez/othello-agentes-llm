"""Utilidades compartilhadas pelas interfaces."""

from __future__ import annotations

import os

from ..agents.base import Agent, TurnResult

# Opções exibidas na interface: rótulo -> especificação do registro de agentes
AGENT_OPTIONS: dict[str, str | None] = {
    "Humano": None,
    "Aleatório": "random",
    "Guloso": "greedy",
    "Heurístico": "heuristic",
    "Minimax (prof. 2)": "minimax:2",
    "Minimax (prof. 4)": "minimax:4",
    "Minimax (prof. 6)": "minimax:6",
    "LLM básico": "llm-basic",
    "LLM planejador": "llm-planner",
}


def is_llm_spec(spec: str | None) -> bool:
    return bool(spec) and spec.startswith("llm")  # type: ignore[union-attr]


def llm_warning(model_spec: str | None) -> str | None:
    """Aviso amigável quando o provedor do LLM provavelmente não tem credencial."""
    from ..llm.config import resolve_model_spec

    spec = resolve_model_spec(model_spec)
    if spec == "mock":
        return None
    provider = spec.split(":", 1)[0]
    env_var = {
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google_genai": "GOOGLE_API_KEY",
        "groq": "GROQ_API_KEY",
    }.get(provider)
    if env_var and not os.getenv(env_var):
        return (
            f"A variável {env_var} não está definida (modelo {spec}); configure-a no arquivo .env. "
            "Sem ela o agente LLM não consegue usar o modelo (conforme o provedor, ele faz jogadas "
            "de segurança heurísticas ou a partida é interrompida com um erro). Para demonstrar o "
            "fluxo sem chave, use o modelo 'mock' (simulado, não é um LLM)."
        )
    return None


def describe_turn(agent: Agent, result: TurnResult) -> str:
    """Linha de log descrevendo a jogada de um agente."""
    text = f"{agent.name} jogou {result.move} ({result.elapsed:.2f}s)"
    if result.llm_calls:
        text += f" | LLM: {result.llm_calls} chamada(s), {result.llm_time:.1f}s"
    if result.tool_calls:
        tools = " -> ".join(result.details.get("ferramentas", []))
        text += f" | ferramentas: {tools}"
    if result.invalid_attempts:
        text += f" | {result.invalid_attempts} tentativa(s) inválida(s)"
    if result.explanation:
        text += f"\n    Motivo: {result.explanation}"
    return text
