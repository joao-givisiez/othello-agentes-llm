"""Função de avaliação heurística e análise de jogadas.

A avaliação combina seis critérios clássicos de Othello, todos medidos do
ponto de vista de um jogador ("own") contra o adversário ("opp"):

=================  ==========================================================
Critério           Ideia
=================  ==========================================================
Cantos             Cantos nunca podem ser recapturados: são as casas mais
                   valiosas do tabuleiro.
Proximidade de     Casas X (diagonal do canto) e C (vizinhas ortogonais) com o
canto              canto ainda vazio costumam entregar o canto ao adversário.
Mobilidade         Ter mais jogadas que o adversário mantém opções e pode
                   forçar o adversário a jogar mal (ou passar).
Fronteira          Peças vizinhas de casas vazias dão jogadas ao adversário;
                   quanto menos, melhor.
Posicional         Tabela de pesos por casa (cantos e bordas bons, casas X
                   ruins).
Paridade de peças  Quantidade de peças. Pesa pouco no meio de jogo (ter
                   muitas peças cedo costuma ser ruim), mas decide o final.
=================  ==========================================================

Os pesos de cada termo seguem a heurística "dinâmica" descrita por
Sannidhanam & Annamalai (*An Analysis of Heuristics in Othello*), com um
reforço da paridade de peças nas últimas jogadas.
"""

from __future__ import annotations

from typing import Any

from ..game.board import (
    FULL_MASK,
    SHIFTS,
    Board,
    Move,
    flips_mask,
    index_to_move,
    iter_bits,
    legal_moves_mask,
    popcount,
    square_bit,
    to_notation,
)

WEIGHTS = [
    [100, -20, 10, 5, 5, 10, -20, 100],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [5, -2, -1, -1, -1, -1, -2, 5],
    [10, -2, -1, -1, -1, -1, -2, 10],
    [-20, -50, -2, -2, -2, -2, -50, -20],
    [100, -20, 10, 5, 5, 10, -20, 100],
]

# Máscara de bits por valor de peso: soma posicional com poucos popcounts.
_WEIGHT_MASKS: dict[int, int] = {}
for _r in range(8):
    for _c in range(8):
        _WEIGHT_MASKS[WEIGHTS[_r][_c]] = _WEIGHT_MASKS.get(WEIGHTS[_r][_c], 0) | square_bit(
            (_r, _c)
        )

CORNERS: tuple[Move, ...] = ((0, 0), (0, 7), (7, 0), (7, 7))
CORNER_MASK = 0
for _m in CORNERS:
    CORNER_MASK |= square_bit(_m)

# Para cada canto: (canto, casa X, casas C)
CORNER_REGIONS: tuple[tuple[int, int, int], ...] = tuple(
    (
        square_bit(corner),
        square_bit((1 if corner[0] == 0 else 6, 1 if corner[1] == 0 else 6)),
        square_bit((corner[0], 1 if corner[1] == 0 else 6))
        | square_bit((1 if corner[0] == 0 else 6, corner[1])),
    )
    for corner in CORNERS
)
X_SQUARES_MASK = 0
C_SQUARES_MASK = 0
for _corner, _x, _cs in CORNER_REGIONS:
    X_SQUARES_MASK |= _x
    C_SQUARES_MASK |= _cs

EDGE_MASK = 0
for _i in range(8):
    EDGE_MASK |= square_bit((0, _i)) | square_bit((7, _i)) | square_bit((_i, 0)) | square_bit((_i, 7))

WIN_SCORE = 1_000_000.0

# Pesos dos termos (ver docstring do módulo)
W_PARITY = 10.0
W_PARITY_ENDGAME = 120.0
W_CORNERS = 801.724
W_CLOSENESS = 382.026
W_MOBILITY = 78.922
W_FRONTIER = 74.396
W_POSITIONAL = 10.0
ENDGAME_EMPTIES = 14


def positional_score(own: int, opp: int) -> int:
    total = 0
    for weight, mask in _WEIGHT_MASKS.items():
        total += weight * (popcount(own & mask) - popcount(opp & mask))
    return total


def frontier_count(pieces: int, empty: int) -> int:
    """Peças vizinhas (8 direções) de pelo menos uma casa vazia."""
    adjacent_to_empty = 0
    for shift in SHIFTS:
        adjacent_to_empty |= shift(empty)
    return popcount(pieces & adjacent_to_empty)


def _ratio(a: int, b: int) -> float:
    return 100.0 * (a - b) / (a + b) if a + b else 0.0


def terminal_score(own: int, opp: int) -> float:
    """Valor de uma posição final: vitória/derrota dominam qualquer heurística."""
    diff = popcount(own) - popcount(opp)
    if diff > 0:
        return WIN_SCORE + diff
    if diff < 0:
        return -WIN_SCORE + diff
    return 0.0


def evaluate_bits(own: int, opp: int) -> float:
    """Avaliação heurística da posição para o lado ``own``."""
    own_moves = legal_moves_mask(own, opp)
    opp_moves = legal_moves_mask(opp, own)
    if not own_moves and not opp_moves:
        return terminal_score(own, opp)

    empty = ~(own | opp) & FULL_MASK
    empties = popcount(empty)

    parity = _ratio(popcount(own), popcount(opp))
    mobility = _ratio(popcount(own_moves), popcount(opp_moves))
    corners = 25.0 * (popcount(own & CORNER_MASK) - popcount(opp & CORNER_MASK))

    closeness = 0.0
    for corner, x_sq, c_sq in CORNER_REGIONS:
        if empty & corner:
            region = x_sq | c_sq
            closeness += popcount(own & region) - popcount(opp & region)
    closeness *= -12.5

    frontier = -_ratio(frontier_count(own, empty), frontier_count(opp, empty))
    positional = positional_score(own, opp)

    w_parity = W_PARITY_ENDGAME if empties <= ENDGAME_EMPTIES else W_PARITY
    return (
        w_parity * parity
        + W_CORNERS * corners
        + W_CLOSENESS * closeness
        + W_MOBILITY * mobility
        + W_FRONTIER * frontier
        + W_POSITIONAL * positional
    )


def evaluate(board: Board, player: int) -> float:
    own, opp = board.bitboards(player)
    return evaluate_bits(own, opp)


def square_kind(move: Move) -> str:
    bit = square_bit(move)
    if bit & CORNER_MASK:
        return "canto"
    if bit & X_SQUARES_MASK:
        return "casa X (diagonal de canto)"
    if bit & C_SQUARES_MASK:
        return "casa C (vizinha de canto)"
    if bit & EDGE_MASK:
        return "borda"
    return "interior"


def analyze_move(board: Board, player: int, move: Move) -> dict[str, Any]:
    """Características estratégicas de uma jogada (usado por agentes e ferramentas)."""
    own, opp = board.bitboards(player)
    bit = square_bit(move)
    flipped = flips_mask(own, opp, bit)
    new_own = own | bit | flipped
    new_opp = opp & ~flipped
    opp_moves = legal_moves_mask(new_opp, new_own)
    own_moves = legal_moves_mask(new_own, new_opp)
    corners_given = [to_notation(index_to_move(i)) for i in iter_bits(opp_moves & CORNER_MASK)]

    # Casa X/C só é perigosa se o canto correspondente ainda estiver vazio.
    risky = False
    empty_after = ~(new_own | new_opp) & FULL_MASK
    for corner, x_sq, c_sq in CORNER_REGIONS:
        if (bit & (x_sq | c_sq)) and (empty_after & corner):
            risky = True

    return {
        "jogada": to_notation(move),
        "pecas_viradas": popcount(flipped),
        "tipo_de_casa": square_kind(move),
        "arrisca_canto_vazio": risky,
        "valor_posicional": WEIGHTS[move[0]][move[1]],
        "mobilidade_adversario_depois": popcount(opp_moves),
        "minha_mobilidade_depois": popcount(own_moves),
        "cantos_liberados_ao_adversario": corners_given,
        "pecas_de_fronteira_depois": frontier_count(new_own, empty_after),
        "avaliacao_heuristica": round(evaluate_bits(new_own, new_opp), 1),
    }


def describe_analysis(info: dict[str, Any]) -> str:
    """Resumo em português de ``analyze_move`` para justificar uma jogada."""
    parts = [f"{info['jogada']} ({info['tipo_de_casa']}) vira {info['pecas_viradas']} peça(s)"]
    parts.append(f"deixa o adversário com {info['mobilidade_adversario_depois']} jogada(s)")
    if info["cantos_liberados_ao_adversario"]:
        parts.append("mas libera canto(s) " + ", ".join(info["cantos_liberados_ao_adversario"]))
    elif info["arrisca_canto_vazio"]:
        parts.append("mas fica ao lado de um canto vazio")
    return "; ".join(parts)
