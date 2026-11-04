"""Modelos de chat SIMULADOS para testes e demonstração sem chave de API.

Atenção: nada aqui é um LLM. ``MockOthelloChatModel`` apenas imita o protocolo
de chamadas de ferramentas (observa, consulta, analisa e joga), escolhendo a
jogada mais bem avaliada pelas ferramentas. Ele permite exercitar todo o
grafo LangGraph (nós, roteamento, ferramentas, memória) sem rede. Os
resultados obtidos com ele NÃO devem ser apresentados como resultados de LLM.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:10]}", "type": "tool_call"}


class _ToolAwareFake(BaseChatModel):
    """Base: aceita bind_tools e guarda os nomes das ferramentas disponíveis."""

    tool_names: list[str] = []

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # type: ignore[override]
        names = [getattr(t, "name", str(t)) for t in tools]
        return self.model_copy(update={"tool_names": names})


class MockOthelloChatModel(_ToolAwareFake):
    """Política determinística que segue o protocolo de ferramentas do agente."""

    model: str = "mock"

    @property
    def _llm_type(self) -> str:
        return "mock-othello"

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        called = [
            call["name"]
            for msg in messages
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]
        results = {m.name: m.content for m in messages if isinstance(m, ToolMessage)}

        if "get_board_state" not in called:
            message = AIMessage(
                content="Vou observar o tabuleiro e listar as jogadas válidas.",
                tool_calls=[_call("get_board_state"), _call("get_valid_moves")],
            )
        elif "analyze_moves" in self.tool_names and "analyze_moves" not in called:
            message = AIMessage(content="Vou analisar as jogadas.", tool_calls=[_call("analyze_moves")])
        else:
            position, reason = self._choose(results, called)
            message = AIMessage(content="", tool_calls=[_call("make_move", position=position, reason=reason)])
        return ChatResult(generations=[ChatGeneration(message=message)])

    @staticmethod
    def _choose(results: dict[str, Any], called: list[str]) -> tuple[str, str]:
        attempts = called.count("make_move")
        if "analyze_moves" in results:
            ranking = json.loads(results["analyze_moves"])
            pick = ranking[min(attempts, len(ranking) - 1)]
            return pick["jogada"], (
                f"[modelo simulado] {pick['jogada']} tem a melhor avaliação heurística "
                f"({pick['avaliacao_heuristica']})."
            )
        moves = json.loads(results["get_valid_moves"])["jogadas_validas"]
        moves.sort(key=lambda m: m["pecas_viradas"], reverse=True)
        pick = moves[min(attempts, len(moves) - 1)]
        return pick["jogada"], f"[modelo simulado] {pick['jogada']} vira {pick['pecas_viradas']} peça(s)."


class ScriptedChatModel(_ToolAwareFake):
    """Devolve uma sequência fixa de mensagens (usado nos testes do grafo)."""

    responses: list[AIMessage]
    _index: int = PrivateAttr(default=0)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # type: ignore[override]
        return self  # mantém o mesmo contador de respostas

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self._index >= len(self.responses):
            message = AIMessage(content="(sem mais respostas roteirizadas)")
        else:
            message = self.responses[self._index]
            self._index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])


__all__ = ["MockOthelloChatModel", "ScriptedChatModel"]
