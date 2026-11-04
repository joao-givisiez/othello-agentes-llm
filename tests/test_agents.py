"""Testes dos agentes clássicos e da função de avaliação."""

import pytest

from othello.agents import (
    GreedyAgent,
    HeuristicAgent,
    MinimaxAgent,
    RandomAgent,
    analyze_move,
    evaluate,
    score_moves,
)
from othello.agents.registry import create_agent
from othello.game import BLACK, WHITE, Board, OthelloGame, parse_square, to_notation


def play_full_game(black, white):
    game = OthelloGame()
    agents = {BLACK: black, WHITE: white}
    while not game.game_over:
        player = game.current_player
        result = agents[player].play_turn(game)
        assert result.move != "passe"
        assert game.history[-1].square == result.move or game.history[-1].is_pass
    return game


@pytest.mark.parametrize(
    "agent_factory",
    [
        lambda: RandomAgent(seed=1),
        lambda: GreedyAgent(seed=1),
        lambda: HeuristicAgent(seed=1),
        lambda: MinimaxAgent(depth=2, seed=1),
    ],
)
def test_agents_play_complete_legal_games(agent_factory):
    game = play_full_game(agent_factory(), RandomAgent(seed=99))
    assert game.game_over
    assert sum(game.score().values()) <= 64


def test_greedy_maximizes_flips():
    board = Board.initial().apply_move(BLACK, parse_square("d3"))
    move, _, details = GreedyAgent(seed=0).select_move(board, WHITE)
    best = max(len(board.flips(WHITE, m)) for m in board.legal_moves(WHITE))
    assert len(board.flips(WHITE, move)) == best == details["pecas_viradas"]


# Posição em que as pretas podem pegar o canto a1 (capturando b2) ou jogar
# no centro (c4, d3, f5, e6).
CORNER_POSITION = Board.from_rows(
    [
        "........",
        ".O......",
        "..X.....",
        "...OX...",
        "...XO...",
        "........",
        "........",
        "........",
    ]
)


def test_corner_is_preferred_by_heuristic_and_minimax():
    moves = {to_notation(m) for m in CORNER_POSITION.legal_moves(BLACK)}
    assert "a1" in moves and len(moves) > 1
    for agent in (HeuristicAgent(seed=0), MinimaxAgent(depth=3, seed=0)):
        move, explanation, _ = agent.select_move(CORNER_POSITION, BLACK)
        assert to_notation(move) == "a1", agent.name
        assert explanation


def test_minimax_solves_endgame_exactly():
    # Restam 3 casas (h6, h7, h8); qualquer jogada das pretas vence, mas h7
    # e h6 deixam as brancas sem resposta e permitem capturar tudo depois.
    board = Board.from_rows(
        [
            "XXXXXXXX",
            "XXXXXXXX",
            "XXXXXXXX",
            "XXXXXXXX",
            "XXXXXXXX",
            "XXXXXXX.",
            "XXXXXXO.",
            "XXXXXXO.",
        ]
    )
    assert {to_notation(m) for m in board.legal_moves(BLACK)} == {"h6", "h7", "h8"}
    move, explanation, details = MinimaxAgent(depth=4, seed=0).select_move(board, BLACK)
    assert details["fim_de_jogo_exato"]
    assert "vitória garantida por 63" in explanation
    assert to_notation(move) in {"h6", "h7"}


def test_minimax_beats_random_agent():
    wins = 0
    for seed in range(4):
        game = OthelloGame()
        agents = {BLACK: MinimaxAgent(depth=3, seed=seed), WHITE: RandomAgent(seed=seed)}
        while not game.game_over:
            agents[game.current_player].play_turn(game)
        wins += game.winner() == BLACK
    assert wins >= 3


def test_evaluation_is_antisymmetric_in_middle_game():
    board = Board.initial().apply_move(BLACK, parse_square("d3"))
    assert evaluate(board, BLACK) == pytest.approx(-evaluate(board, WHITE))


def test_analyze_move_reports_features():
    info = analyze_move(CORNER_POSITION, BLACK, parse_square("a1"))
    assert info["jogada"] == "a1"
    assert info["tipo_de_casa"] == "canto"
    assert info["pecas_viradas"] >= 1
    assert isinstance(info["avaliacao_heuristica"], float)


def test_score_moves_ranks_all_moves():
    ranking = score_moves(CORNER_POSITION, BLACK, depth=2)
    assert len(ranking) == len(CORNER_POSITION.legal_moves(BLACK))
    assert ranking[0]["jogada"] == "a1"


def test_registry_creates_classical_agents():
    assert isinstance(create_agent("random"), RandomAgent)
    assert isinstance(create_agent("greedy"), GreedyAgent)
    assert isinstance(create_agent("heuristic"), HeuristicAgent)
    agent = create_agent("minimax:3")
    assert isinstance(agent, MinimaxAgent) and agent.depth == 3
    with pytest.raises(ValueError):
        create_agent("xadrez")
