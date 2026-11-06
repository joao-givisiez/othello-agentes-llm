"""Testes do agente LLM (grafo LangGraph + ferramentas) sem acesso à rede.

Os modelos usados aqui são roteirizados/simulados: o objetivo é verificar o
fluxo do agente (percepção, chamadas de ferramentas, execução da jogada,
cobrança, jogada de segurança e memória), não a qualidade de um LLM real.
"""

import json

import pytest
from langchain_core.messages import AIMessage

from othello.agents.registry import create_agent
from othello.game import BLACK, WHITE, OthelloGame
from othello.llm import LLMAgent, ToolContext, build_tools
from othello.llm.mock import ScriptedChatModel, _ToolAwareFake


def call(name, **args):
    return {"name": name, "args": args, "id": f"id_{name}_{len(args)}_{id(args)}", "type": "tool_call"}


def scripted(*responses):
    return ScriptedChatModel(responses=list(responses))


# ---------------------------------------------------------------------------
# Ferramentas
# ---------------------------------------------------------------------------
def make_tools(mode="planner"):
    ctx = ToolContext()
    game = OthelloGame()
    ctx.new_turn(game, BLACK, "teste")
    return ctx, game, {t.name: t for t in build_tools(ctx, mode)}


def test_tool_sets_per_mode():
    _, _, basic = make_tools("basic")
    _, _, planner = make_tools("planner")
    required = {"get_board_state", "get_valid_moves", "make_move", "get_game_status"}
    assert required <= set(basic)
    assert set(planner) - set(basic) == {"analyze_moves", "simulate_move", "minimax_search"}


def test_observation_tools():
    ctx, game, tools = make_tools()
    board = tools["get_board_state"].invoke({})
    assert "a b c d e f g h" in board and "Pretas (X)" in board
    moves = json.loads(tools["get_valid_moves"].invoke({}))
    assert {m["jogada"] for m in moves["jogadas_validas"]} == {"d3", "c4", "f5", "e6"}
    status = json.loads(tools["get_game_status"].invoke({}))
    assert status["partida_encerrada"] is False
    assert tools["get_move_history"].invoke({"last_n": 5}) == "Nenhuma jogada foi feita ainda."
    assert ctx.tool_calls == 4


def test_planning_tools_do_not_change_the_game():
    _, game, tools = make_tools()
    analysis = json.loads(tools["analyze_moves"].invoke({}))
    assert len(analysis) == 4 and "avaliacao_heuristica" in analysis[0]
    sim = json.loads(tools["simulate_move"].invoke({"position": "d3"}))
    assert sim["jogada_simulada"] == "d3" and sim["respostas_do_adversario"] > 0
    assert "ERRO" in tools["simulate_move"].invoke({"position": "a1"})
    ranking = json.loads(tools["minimax_search"].invoke({"depth": 2}))["ranking"]
    assert len(ranking) == 4
    assert game.history == []  # nada foi executado


def test_make_move_tool_validates_and_executes_once():
    ctx, game, tools = make_tools()
    error = tools["make_move"].invoke({"position": "a1", "reason": "teste"})
    assert error.startswith("ERRO") and ctx.invalid_attempts == 1 and game.history == []
    ok = tools["make_move"].invoke({"position": "d3", "reason": "abre o jogo"})
    assert "Jogada executada: d3" in ok
    assert ctx.move_made == "d3" and game.history[-1].explanation == "abre o jogo"
    again = tools["make_move"].invoke({"position": "c3", "reason": "de novo"})
    assert "já jogou" in again and len(game.history) == 1


# ---------------------------------------------------------------------------
# Grafo do agente
# ---------------------------------------------------------------------------
def test_agent_observes_then_acts():
    model = scripted(
        AIMessage(content="", tool_calls=[call("get_board_state"), call("get_valid_moves")]),
        AIMessage(content="", tool_calls=[call("make_move", position="d3", reason="Mantém a mobilidade.")]),
    )
    agent = LLMAgent(mode="basic", model=model)
    game = OthelloGame()
    result = agent.play_turn(game)
    assert result.move == "d3"
    assert result.explanation == "Mantém a mobilidade."
    assert result.llm_calls == 2
    assert result.tool_calls == 3
    assert result.details["ferramentas"] == ["get_board_state", "get_valid_moves", "make_move"]
    assert not result.fallback
    assert game.current_player == WHITE
    assert game.history[-1].actor == "LLM-Básico"
    assert agent.memory() == ["Turno 1: joguei d3. Mantém a mobilidade."]


def test_agent_recovers_from_invalid_move():
    model = scripted(
        AIMessage(content="", tool_calls=[call("make_move", position="a1", reason="canto!")]),
        AIMessage(content="", tool_calls=[call("make_move", position="f5", reason="corrigindo")]),
    )
    result = LLMAgent(model=model).play_turn(OthelloGame())
    assert result.move == "f5"
    assert result.invalid_attempts == 1
    assert result.llm_calls == 2 and not result.fallback


def test_agent_is_nudged_and_then_falls_back():
    model = scripted(
        AIMessage(content="Acho que d3 é boa."),
        AIMessage(content="Ainda pensando..."),
        AIMessage(content="Não sei."),
    )
    game = OthelloGame()
    result = LLMAgent(model=model).play_turn(game)
    assert result.fallback
    assert result.llm_calls == 3  # resposta inicial + 2 cobranças
    assert result.move in {"d3", "c4", "f5", "e6"}
    assert "jogada de segurança" in result.explanation
    assert game.current_player == WHITE


class _BrokenModel(_ToolAwareFake):
    @property
    def _llm_type(self):
        return "broken"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        raise ConnectionError("API indisponível")


def test_agent_retries_once_then_falls_back_on_api_error():
    result = LLMAgent(model=_BrokenModel()).play_turn(OthelloGame())
    assert result.fallback
    assert result.llm_calls == 2  # tentativa original + 1 nova tentativa
    assert "API indisponível" in result.explanation
    assert result.details["erro"].startswith("ConnectionError")


class _FlakyModel(ScriptedChatModel):
    """Falha na primeira chamada (como um tool_use_failed do Groq) e depois segue o roteiro."""

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        if not getattr(self, "_failed", False):
            object.__setattr__(self, "_failed", True)
            raise ValueError("tool_use_failed")
        return super()._generate(messages, stop, run_manager, **kwargs)


def test_agent_recovers_from_transient_api_error():
    model = _FlakyModel(
        responses=[AIMessage(content="", tool_calls=[call("make_move", position="c4", reason="ok")])]
    )
    result = LLMAgent(model=model).play_turn(OthelloGame())
    assert result.move == "c4"
    assert not result.fallback
    assert result.llm_calls == 2


def test_memory_accumulates_during_game_and_resets_on_new_game():
    agent = LLMAgent(mode="planner", model="mock")
    opponent = create_agent("greedy", seed=0)
    game = OthelloGame()
    for _ in range(3):
        result = agent.play_turn(game)
        assert result.details["ferramentas"][-1] == "make_move"
        assert "analyze_moves" in result.details["ferramentas"]
        opponent.play_turn(game)
    memory = agent.memory()
    assert len(memory) == 3 and memory[0].startswith("Turno 1")
    agent.play_turn(OthelloGame())  # nova partida
    assert len(agent.memory()) == 1


def test_mock_llm_agent_plays_full_game():
    agent = create_agent("llm-basic", model="mock")
    rival = create_agent("random", seed=3)
    game = OthelloGame()
    agents = {BLACK: agent, WHITE: rival}
    while not game.game_over:
        agents[game.current_player].play_turn(game)
    assert game.game_over


def test_invalid_mode_rejected():
    with pytest.raises(ValueError):
        LLMAgent(mode="xadrez")
