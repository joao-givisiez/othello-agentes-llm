"""Regras do Othello (Reversi) implementadas com bitboards.

Representação
-------------
O tabuleiro 8x8 é guardado em dois inteiros de 64 bits: um com as peças
pretas e outro com as brancas. A casa (linha, coluna) corresponde ao bit
``linha * 8 + coluna``. A linha 0 é a linha "1" da notação e a coluna 0 é a
coluna "a", de modo que ``a1`` é o bit 0 e ``h8`` é o bit 63.

Com bitboards a geração de jogadas e o cálculo das peças capturadas são
feitos com deslocamentos de bits nas 8 direções, o que é dezenas de vezes
mais rápido que percorrer uma matriz em Python. Isso permite que o Minimax
pesquise mais fundo dentro do mesmo tempo.

Regras implementadas
--------------------
* Posição inicial: brancas em d4 e e5, pretas em d5 e e4. Pretas começam.
* Uma jogada é válida se a peça colocada "cerca" uma ou mais sequências
  contínuas de peças adversárias em qualquer das 8 direções.
* Todas as sequências cercadas são invertidas (capturadas).
* Se um jogador não tem jogada válida ele passa a vez.
* A partida termina quando nenhum dos dois jogadores pode jogar; vence quem
  tiver mais peças.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

BLACK = 1
WHITE = -1
EMPTY = 0

PLAYER_NAMES = {BLACK: "Pretas", WHITE: "Brancas"}
PLAYER_SYMBOLS = {BLACK: "X", WHITE: "O", EMPTY: "."}

COLUMNS = "abcdefgh"

FULL_MASK = 0xFFFFFFFFFFFFFFFF
NOT_A_FILE = 0xFEFEFEFEFEFEFEFE  # todas as casas exceto a coluna "a"
NOT_H_FILE = 0x7F7F7F7F7F7F7F7F  # todas as casas exceto a coluna "h"

Move = tuple[int, int]  # (linha, coluna), ambos de 0 a 7


class IllegalMoveError(ValueError):
    """Jogada que não respeita as regras do Othello."""


# ---------------------------------------------------------------------------
# Deslocamentos nas 8 direções. As máscaras evitam que um bit "dê a volta"
# de uma borda do tabuleiro para a borda oposta.
# ---------------------------------------------------------------------------
def _north(b: int) -> int:  # linha + 1
    return (b << 8) & FULL_MASK


def _south(b: int) -> int:  # linha - 1
    return b >> 8


def _east(b: int) -> int:  # coluna + 1
    return (b << 1) & NOT_A_FILE & FULL_MASK


def _west(b: int) -> int:  # coluna - 1
    return (b >> 1) & NOT_H_FILE


def _north_east(b: int) -> int:
    return (b << 9) & NOT_A_FILE & FULL_MASK


def _north_west(b: int) -> int:
    return (b << 7) & NOT_H_FILE & FULL_MASK


def _south_east(b: int) -> int:
    return (b >> 7) & NOT_A_FILE


def _south_west(b: int) -> int:
    return (b >> 9) & NOT_H_FILE


SHIFTS = (
    _north,
    _south,
    _east,
    _west,
    _north_east,
    _north_west,
    _south_east,
    _south_west,
)


# ---------------------------------------------------------------------------
# Funções de baixo nível sobre bitboards
# ---------------------------------------------------------------------------
def popcount(b: int) -> int:
    """Número de bits ligados (quantidade de peças)."""
    return b.bit_count()


def iter_bits(b: int) -> Iterator[int]:
    """Itera sobre os índices dos bits ligados, do menor para o maior."""
    while b:
        lsb = b & -b
        yield lsb.bit_length() - 1
        b ^= lsb


def square_bit(move: Move) -> int:
    row, col = move
    return 1 << (row * 8 + col)


def index_to_move(index: int) -> Move:
    return divmod(index, 8)


def legal_moves_mask(own: int, opp: int) -> int:
    """Bitboard com todas as casas onde ``own`` pode jogar.

    Para cada direção, propaga a partir das peças próprias por cima de peças
    adversárias contíguas (no máximo 6 em sequência num tabuleiro 8x8); a
    casa vazia logo depois dessa sequência é uma jogada válida.
    """
    empty = ~(own | opp) & FULL_MASK
    moves = 0
    for shift in SHIFTS:
        t = shift(own) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        t |= shift(t) & opp
        moves |= shift(t) & empty
    return moves


def flips_mask(own: int, opp: int, move_bit: int) -> int:
    """Bitboard das peças adversárias invertidas ao jogar em ``move_bit``."""
    flips = 0
    for shift in SHIFTS:
        captured = 0
        x = shift(move_bit)
        while x & opp:
            captured |= x
            x = shift(x)
        if x & own:
            flips |= captured
    return flips


# ---------------------------------------------------------------------------
# Notação algébrica (ex.: "d3")
# ---------------------------------------------------------------------------
def to_notation(move: Move | None) -> str:
    """Converte (linha, coluna) para a notação usada no Othello, ex. "d3"."""
    if move is None:
        return "passe"
    row, col = move
    return f"{COLUMNS[col]}{row + 1}"


def parse_square(text: str) -> Move:
    """Converte uma casa em notação ("d3", "D3", " d3 ") para (linha, coluna)."""
    s = text.strip().lower()
    if len(s) != 2 or s[0] not in COLUMNS or s[1] not in "12345678":
        raise ValueError(f"Casa inválida: {text!r}. Use o formato coluna+linha, ex.: d3")
    return int(s[1]) - 1, COLUMNS.index(s[0])


# ---------------------------------------------------------------------------
# Tabuleiro imutável
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Board:
    """Estado do tabuleiro. Imutável: ``apply_move`` devolve um novo tabuleiro."""

    black: int
    white: int

    # --- construção -------------------------------------------------------
    @classmethod
    def initial(cls) -> "Board":
        d4, e4, d5, e5 = (3, 3), (3, 4), (4, 3), (4, 4)
        white = square_bit(d4) | square_bit(e5)
        black = square_bit(e4) | square_bit(d5)
        return cls(black=black, white=white)

    @classmethod
    def from_rows(cls, rows: list[str]) -> "Board":
        """Cria um tabuleiro a partir de 8 strings (linha 1 primeiro).

        ``X`` = pretas, ``O`` = brancas, qualquer outro caractere = vazio.
        Útil em testes e para montar posições específicas.
        """
        if len(rows) != 8 or any(len(r) != 8 for r in rows):
            raise ValueError("São necessárias 8 linhas com 8 caracteres cada")
        black = white = 0
        for r, line in enumerate(rows):
            for c, ch in enumerate(line):
                if ch.upper() == "X":
                    black |= square_bit((r, c))
                elif ch.upper() == "O":
                    white |= square_bit((r, c))
        return cls(black=black, white=white)

    # --- consultas --------------------------------------------------------
    def bitboards(self, player: int) -> tuple[int, int]:
        """Devolve (peças do jogador, peças do adversário)."""
        return (self.black, self.white) if player == BLACK else (self.white, self.black)

    def get(self, move: Move) -> int:
        bit = square_bit(move)
        if self.black & bit:
            return BLACK
        if self.white & bit:
            return WHITE
        return EMPTY

    def count(self, player: int) -> int:
        return popcount(self.black if player == BLACK else self.white)

    def empty_count(self) -> int:
        return 64 - popcount(self.black | self.white)

    def legal_mask(self, player: int) -> int:
        own, opp = self.bitboards(player)
        return legal_moves_mask(own, opp)

    def legal_moves(self, player: int) -> list[Move]:
        return [index_to_move(i) for i in iter_bits(self.legal_mask(player))]

    def has_moves(self, player: int) -> bool:
        return self.legal_mask(player) != 0

    def is_legal(self, player: int, move: Move) -> bool:
        return bool(self.legal_mask(player) & square_bit(move))

    def flips(self, player: int, move: Move) -> list[Move]:
        """Casas que seriam invertidas se ``player`` jogasse em ``move``."""
        own, opp = self.bitboards(player)
        bit = square_bit(move)
        if (own | opp) & bit:
            return []
        return [index_to_move(i) for i in iter_bits(flips_mask(own, opp, bit))]

    def is_terminal(self) -> bool:
        return not self.has_moves(BLACK) and not self.has_moves(WHITE)

    def winner(self) -> int | None:
        """BLACK, WHITE ou EMPTY (empate). ``None`` se a partida não acabou."""
        if not self.is_terminal():
            return None
        diff = self.count(BLACK) - self.count(WHITE)
        if diff > 0:
            return BLACK
        if diff < 0:
            return WHITE
        return EMPTY

    # --- transição --------------------------------------------------------
    def apply_move(self, player: int, move: Move) -> "Board":
        """Coloca a peça de ``player`` em ``move`` e inverte as capturadas."""
        row, col = move
        if not (0 <= row < 8 and 0 <= col < 8):
            raise IllegalMoveError(f"Casa fora do tabuleiro: {move}")
        own, opp = self.bitboards(player)
        bit = square_bit(move)
        if (own | opp) & bit:
            raise IllegalMoveError(f"A casa {to_notation(move)} já está ocupada")
        flipped = flips_mask(own, opp, bit)
        if not flipped:
            raise IllegalMoveError(
                f"Jogada {to_notation(move)} não captura nenhuma peça adversária"
            )
        own |= bit | flipped
        opp &= ~flipped
        if player == BLACK:
            return Board(black=own, white=opp)
        return Board(black=opp, white=own)

    # --- representação ----------------------------------------------------
    def to_matrix(self) -> list[list[int]]:
        return [[self.get((r, c)) for c in range(8)] for r in range(8)]

    def to_ascii(self, highlight: set[Move] | None = None) -> str:
        """Desenha o tabuleiro em texto. Casas em ``highlight`` aparecem como '*'."""
        highlight = highlight or set()
        lines = ["  " + " ".join(COLUMNS)]
        for r in range(8):
            cells = []
            for c in range(8):
                piece = self.get((r, c))
                if piece == EMPTY and (r, c) in highlight:
                    cells.append("*")
                else:
                    cells.append(PLAYER_SYMBOLS[piece])
            lines.append(f"{r + 1} " + " ".join(cells))
        return "\n".join(lines)

    def __str__(self) -> str:  # pragma: no cover - conveniência
        return self.to_ascii()


def opponent(player: int) -> int:
    return -player
