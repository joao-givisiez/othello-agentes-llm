"""Testes das regras do Othello (bitboards) e do ambiente de jogo."""

import random

import pytest

from othello.game import (
    BLACK,
    EMPTY,
    WHITE,
    Board,
    IllegalMoveError,
    OthelloGame,
    parse_square,
    to_notation,
)

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# ---------------------------------------------------------------------------
# Implementação de referência, propositalmente ingênua (matriz 8x8), usada
# para validar a implementação com bitboards.
# ---------------------------------------------------------------------------
def naive_flips(matrix, player, r, c):
    if matrix[r][c] != EMPTY:
        return []
    flips = []
    for dr, dc in DIRECTIONS:
        line = []
        rr, cc = r + dr, c + dc
        while 0 <= rr < 8 and 0 <= cc < 8 and matrix[rr][cc] == -player:
            line.append((rr, cc))
            rr += dr
            cc += dc
        if line and 0 <= rr < 8 and 0 <= cc < 8 and matrix[rr][cc] == player:
            flips.extend(line)
    return flips


def naive_moves(matrix, player):
    return sorted(
        (r, c) for r in range(8) for c in range(8) if naive_flips(matrix, player, r, c)
    )


# ---------------------------------------------------------------------------
# Notação
# ---------------------------------------------------------------------------
def test_notation_roundtrip():
    for r in range(8):
        for c in range(8):
            assert parse_square(to_notation((r, c))) == (r, c)
    assert parse_square(" D3 ") == (2, 3)
    with pytest.raises(ValueError):
        parse_square("z9")
    with pytest.raises(ValueError):
        parse_square("d")


# ---------------------------------------------------------------------------
# Posição inicial e jogadas
# ---------------------------------------------------------------------------
def test_initial_position():
    board = Board.initial()
    assert board.count(BLACK) == 2
    assert board.count(WHITE) == 2
    assert board.get(parse_square("d4")) == WHITE
    assert board.get(parse_square("e5")) == WHITE
    assert board.get(parse_square("d5")) == BLACK
    assert board.get(parse_square("e4")) == BLACK


def test_initial_legal_moves_for_black():
    board = Board.initial()
    moves = {to_notation(m) for m in board.legal_moves(BLACK)}
    assert moves == {"d3", "c4", "f5", "e6"}


def test_apply_move_flips_pieces():
    board = Board.initial().apply_move(BLACK, parse_square("d3"))
    assert board.get(parse_square("d3")) == BLACK
    assert board.get(parse_square("d4")) == BLACK  # peça capturada
    assert board.count(BLACK) == 4
    assert board.count(WHITE) == 1


def test_illegal_moves_raise():
    board = Board.initial()
    with pytest.raises(IllegalMoveError):
        board.apply_move(BLACK, parse_square("a1"))  # não captura nada
    with pytest.raises(IllegalMoveError):
        board.apply_move(BLACK, parse_square("d4"))  # casa ocupada


def test_multi_direction_capture():
    # Pretas jogando em d4 capturam em três direções ao mesmo tempo.
    board = Board.from_rows(
        [
            "........",
            ".X.X.X..",
            "..OOO...",
            ".XO.OX..",
            "..OOO...",
            ".X.X.X..",
            "........",
            "........",
        ]
    )
    flips = {to_notation(m) for m in board.flips(BLACK, parse_square("d4"))}
    assert flips == {"c3", "d3", "e3", "c4", "e4", "c5", "d5", "e5"}


def test_no_wraparound_between_edges():
    # Uma peça branca em h1 e uma preta em a2 não podem formar captura "dando a volta".
    board = Board.from_rows(
        [
            "......XO",
            "X.......",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
        ]
    )
    assert board.legal_moves(BLACK) == []


def test_bitboard_matches_naive_reference_in_random_games():
    rng = random.Random(1234)
    for _ in range(60):
        board = Board.initial()
        player = BLACK
        while not board.is_terminal():
            matrix = board.to_matrix()
            expected = naive_moves(matrix, player)
            assert sorted(board.legal_moves(player)) == expected
            if not expected:
                player = -player
                continue
            move = rng.choice(expected)
            assert sorted(board.flips(player, move)) == sorted(
                naive_flips(matrix, player, *move)
            )
            board = board.apply_move(player, move)
            player = -player
        assert board.count(BLACK) + board.count(WHITE) + board.empty_count() == 64


# ---------------------------------------------------------------------------
# Ambiente de jogo: turnos, passes, término e histórico
# ---------------------------------------------------------------------------
def test_game_turns_and_history():
    game = OthelloGame()
    assert game.current_player == BLACK
    record = game.make_move("d3", actor="teste", explanation="abertura")
    assert record.square == "d3"
    assert record.flipped == ["d4"]
    assert record.explanation == "abertura"
    assert game.current_player == WHITE
    assert game.moves_played() == 1
    assert "d3" in game.history_lines()[0]


def test_game_rejects_invalid_and_out_of_turn_moves():
    game = OthelloGame()
    with pytest.raises(IllegalMoveError):
        game.make_move("a1")
    with pytest.raises(IllegalMoveError):
        game.make_move("d3", player=WHITE)
    with pytest.raises(IllegalMoveError):
        game.make_move("k9")
    assert game.history == []


def test_automatic_pass_and_game_over():
    # Brancas jogam em c1 e as pretas (só g8) ficam sem jogadas -> passe automático.
    board = Board.from_rows(
        [
            "OX......",
            "........",
            "........",
            "........",
            "........",
            "........",
            "........",
            "......XO",
        ]
    )
    game = OthelloGame(board=board, current_player=WHITE)
    assert {to_notation(m) for m in game.valid_moves()} == {"c1", "f8"}
    game.make_move("c1")
    assert game.history[-1].is_pass
    assert game.history[-1].player == BLACK
    assert game.current_player == WHITE
    # Brancas capturam a última peça preta: ninguém mais pode jogar.
    game.make_move("f8")
    assert game.game_over
    assert game.winner() == WHITE
    assert game.status()["resultado"] == "vitória das Brancas"


def test_full_random_game_ends_with_winner():
    rng = random.Random(7)
    game = OthelloGame()
    while not game.game_over:
        game.make_move(rng.choice(game.valid_moves()))
    status = game.status()
    assert status["partida_encerrada"] is True
    assert game.winner() in (BLACK, WHITE, EMPTY)
    assert game.valid_moves() == []
    with pytest.raises(IllegalMoveError):
        game.make_move("a1")


def test_copy_is_independent():
    game = OthelloGame()
    clone = game.copy()
    clone.make_move("d3")
    assert game.history == []
    assert game.board == Board.initial()
