"""Agente LLM para Othello estruturado como um grafo LangGraph.

Fluxo de um turno::

    START -> perceive -> reason --(chamou ferramentas)--> tools --(jogou)--> finalize -> END
                           ^  |                             |
                           |  +--(não chamou ferramenta)    +--(ainda não jogou)--> reason
                           |  |       -> nudge -> reason
                           |  +--(erro / limite)-> fallback -> finalize
                           +-----------------------------------------------+

* ``perceive``  monta a percepção do turno: cor, última jogada do adversário,
  memória das decisões anteriores e histórico recente. Zera as mensagens do
  turno anterior (a memória de longo prazo fica em ``memory``).
* ``reason``    chama o LLM com as ferramentas vinculadas (``bind_tools``).
* ``tools``     executa as ferramentas pedidas (``ToolNode``). A jogada só
  acontece quando o LLM chama ``make_move``.
* ``nudge``     se o LLM responder sem jogar, lembra-o de chamar make_move.
* ``fallback``  jogada de segurança (melhor avaliação heurística) se o LLM
  falhar repetidamente, estourar o limite de passos ou a API der erro duas
  vezes seguidas (após um erro o ``reason`` é repetido uma vez) - o jogo
  nunca trava.
* ``finalize``  grava a decisão e a justificativa na memória do agente.

O estado do grafo é persistido por partida com um checkpointer
(``InMemorySaver``, ``thread_id`` = id da partida + cor), de modo que a
memória de decisões acumula ao longo da partida.
"""

from __future__ import annotations

import operator
import time
from typing import Annotated, Any, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import REMOVE_ALL_MESSAGES, add_messages
from langgraph.prebuilt import ToolNode

from ..agents.base import Agent, TurnResult
from ..agents.evaluation import analyze_move, describe_analysis, evaluate
from ..game import PLAYER_NAMES, OthelloGame, opponent, to_notation
from ..game.board import PLAYER_SYMBOLS
from .config import create_chat_model, model_display_name, resolve_model_spec
from .prompts import NUDGE_PROMPT, system_prompt, turn_prompt
from .tools import ToolContext, build_tools

MAX_LLM_CALLS_PER_TURN = 10  # modelos sem chamadas paralelas fazem 1 ferramenta por chamada
MAX_NUDGES = 2
MAX_API_RETRIES = 1  # nova tentativa após erro da API (ex.: tool_use_failed do Groq)
MEMORY_IN_PROMPT = 6
HISTORY_IN_PROMPT = 6


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # conversa do turno
    memory: Annotated[list[str], operator.add]  # memória de longo prazo (partida)
    turn: int
    llm_calls: int
    llm_time: float
    nudges: int
    api_errors: int
    error: str | None
    fallback: bool
    move: str | None
    explanation: str


class LLMAgent(Agent):
    """Agente que joga Othello com um LLM, ferramentas e um grafo LangGraph."""

    def __init__(
        self,
        mode: str = "basic",
        model: str | BaseChatModel | None = None,
        name: str | None = None,
        max_llm_calls: int = MAX_LLM_CALLS_PER_TURN,
    ) -> None:
        if mode not in ("basic", "planner"):
            raise ValueError("mode deve ser 'basic' ou 'planner'")
        self.mode = mode
        self.name = name or ("LLM-Planejador" if mode == "planner" else "LLM-Básico")
        self.max_llm_calls = max_llm_calls
        self._model_arg = model
        self._llm: BaseChatModel | None = model if isinstance(model, BaseChatModel) else None
        self.model_name = (
            model_display_name(self._llm) if self._llm else resolve_model_spec(model)  # type: ignore[arg-type]
        )
        self.ctx = ToolContext()
        self.tools = build_tools(self.ctx, mode)
        self._graph = None
        self._thread_id: str | None = None
        self._turn = 0
        self.last_trace: list[str] = []

    # ------------------------------------------------------------------
    # Construção preguiçosa: o modelo só é criado quando o agente joga, então
    # a interface pode listar o agente mesmo sem chave de API configurada.
    # ------------------------------------------------------------------
    def _ensure_graph(self):
        if self._graph is None:
            if self._llm is None:
                try:
                    self._llm = create_chat_model(self._model_arg)  # type: ignore[arg-type]
                except Exception as exc:  # ex.: chave ausente ou pacote do provedor não instalado
                    raise RuntimeError(
                        f"Não foi possível criar o modelo '{self.model_name}': {exc}. "
                        "Verifique a chave de API no arquivo .env e se o pacote do provedor "
                        "(ex.: langchain-groq) está instalado."
                    ) from exc
                self.model_name = model_display_name(self._llm, self.model_name)
            self._graph = self._build_graph(self._llm.bind_tools(self.tools))
        return self._graph

    def reset(self) -> None:
        self._thread_id = None
        self._turn = 0

    # ------------------------------------------------------------------
    # Grafo
    # ------------------------------------------------------------------
    def _build_graph(self, llm_with_tools):
        ctx = self.ctx

        def color() -> str:
            return f"{PLAYER_NAMES[ctx.player]} ({PLAYER_SYMBOLS[ctx.player]})"

        def perceive(state: AgentState) -> dict[str, Any]:
            game = ctx.game
            assert game is not None
            last = game.last_move()
            last_opp = (
                f"{last.square} (virou {len(last.flipped)} peça(s))"
                if last is not None and last.player == opponent(ctx.player)
                else None
            )
            prompt = turn_prompt(
                turn=state.get("turn", 1),
                color=color(),
                last_opponent_move=last_opp,
                memory=state.get("memory", [])[-MEMORY_IN_PROMPT:],
                recent_history=game.history_lines(last_n=HISTORY_IN_PROMPT),
            )
            return {
                "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), HumanMessage(prompt)],
                "llm_calls": 0,
                "llm_time": 0.0,
                "nudges": 0,
                "api_errors": 0,
                "error": None,
                "fallback": False,
                "move": None,
                "explanation": "",
            }

        def reason(state: AgentState) -> dict[str, Any]:
            messages = [SystemMessage(system_prompt(self.mode, color()))] + state["messages"]
            start = time.perf_counter()
            try:
                response = llm_with_tools.invoke(messages)
                error = None
            except Exception as exc:  # erro de rede/API: o jogo não pode travar
                response, error = None, f"{type(exc).__name__}: {exc}"
            update: dict[str, Any] = {
                "llm_calls": state.get("llm_calls", 0) + 1,
                "llm_time": state.get("llm_time", 0.0) + time.perf_counter() - start,
                "error": error,
            }
            if response is not None:
                update["messages"] = [response]
            else:
                update["api_errors"] = state.get("api_errors", 0) + 1
            return update

        def route_after_reason(state: AgentState) -> str:
            if state.get("error"):
                can_retry = (
                    state.get("api_errors", 0) <= MAX_API_RETRIES
                    and state.get("llm_calls", 0) < self.max_llm_calls
                )
                return "reason" if can_retry else "fallback"
            last = state["messages"][-1]
            if isinstance(last, AIMessage) and last.tool_calls:
                return "tools"
            if ctx.move_made:
                return "finalize"
            if state.get("nudges", 0) < MAX_NUDGES and state.get("llm_calls", 0) < self.max_llm_calls:
                return "nudge"
            return "fallback"

        def route_after_tools(state: AgentState) -> str:
            if ctx.move_made:
                return "finalize"
            if state.get("llm_calls", 0) >= self.max_llm_calls:
                return "fallback"
            return "reason"

        def nudge(state: AgentState) -> dict[str, Any]:
            valid = ", ".join(to_notation(m) for m in ctx.game.valid_moves())  # type: ignore[union-attr]
            return {
                "messages": [HumanMessage(NUDGE_PROMPT.format(valid=valid))],
                "nudges": state.get("nudges", 0) + 1,
            }

        def fallback(state: AgentState) -> dict[str, Any]:
            game = ctx.game
            assert game is not None
            if not ctx.move_made and not game.game_over and game.current_player == ctx.player:
                board, player = game.board, ctx.player
                move = max(board.legal_moves(player), key=lambda m: evaluate(board.apply_move(player, m), player))
                motive = state.get("error") or "o LLM não executou uma jogada válida"
                explanation = (
                    f"[jogada de segurança: {motive}] "
                    f"{describe_analysis(analyze_move(board, player, move))}."
                )
                record = game.make_move(move, actor=self.name, explanation=explanation, player=player)
                ctx.move_made = record.square
                ctx.explanation = explanation
            return {"fallback": True}

        def finalize(state: AgentState) -> dict[str, Any]:
            note = f"Turno {state.get('turn', 0)}: joguei {ctx.move_made}. {ctx.explanation}".strip()
            if state.get("fallback"):
                note += " (jogada de segurança)"
            return {"memory": [note], "move": ctx.move_made, "explanation": ctx.explanation}

        builder = StateGraph(AgentState)
        builder.add_node("perceive", perceive)
        builder.add_node("reason", reason)
        builder.add_node("tools", ToolNode(self.tools, handle_tool_errors=True))
        builder.add_node("nudge", nudge)
        builder.add_node("fallback", fallback)
        builder.add_node("finalize", finalize)

        builder.add_edge(START, "perceive")
        builder.add_edge("perceive", "reason")
        builder.add_conditional_edges(
            "reason",
            route_after_reason,
            {
                "tools": "tools",
                "nudge": "nudge",
                "fallback": "fallback",
                "finalize": "finalize",
                "reason": "reason",  # nova tentativa após erro da API
            },
        )
        builder.add_conditional_edges(
            "tools",
            route_after_tools,
            {"finalize": "finalize", "fallback": "fallback", "reason": "reason"},
        )
        builder.add_edge("nudge", "reason")
        builder.add_edge("fallback", "finalize")
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=InMemorySaver())

    # ------------------------------------------------------------------
    # Interface Agent
    # ------------------------------------------------------------------
    def play_turn(self, game: OthelloGame) -> TurnResult:
        graph = self._ensure_graph()
        player = game.current_player
        thread_id = f"{game.game_id}-{player}"
        if thread_id != self._thread_id:  # nova partida: nova memória
            self._thread_id = thread_id
            self._turn = 0
        self._turn += 1
        self.ctx.new_turn(game, player, self.name)

        start = time.perf_counter()
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 4 * self.max_llm_calls + 10,
        }
        state = graph.invoke({"turn": self._turn}, config)
        elapsed = time.perf_counter() - start

        self.last_trace = list(self.ctx.tool_log)
        return TurnResult(
            move=self.ctx.move_made or "?",
            explanation=self.ctx.explanation,
            elapsed=elapsed,
            llm_calls=state.get("llm_calls", 0),
            llm_time=state.get("llm_time", 0.0),
            tool_calls=self.ctx.tool_calls,
            invalid_attempts=self.ctx.invalid_attempts,
            fallback=bool(state.get("fallback")),
            details={
                "modelo": self.model_name,
                "ferramentas": list(self.ctx.tool_log),
                "erro": state.get("error"),
                "memoria": len(state.get("memory", [])),
            },
        )

    def memory(self) -> list[str]:
        """Memória acumulada da partida atual (lida do checkpointer)."""
        if self._graph is None or self._thread_id is None:
            return []
        snapshot = self._graph.get_state({"configurable": {"thread_id": self._thread_id}})
        return list(snapshot.values.get("memory", []))

    def graph_mermaid(self) -> str:
        """Diagrama Mermaid do grafo (usado na documentação)."""
        return self._ensure_graph().get_graph().draw_mermaid()
