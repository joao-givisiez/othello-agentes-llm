"""Ambiente de jogo: controla turnos, passes, término e histórico da partida.

O ``OthelloGame`` é o "mundo" com o qual os agentes interagem. Ele expõe as
operações que as ferramentas do agente LLM usam (estado do tabuleiro, jogadas
válidas, executar jogada e status) e mantém o histórico completo da partida,
que serve de memória para os agentes.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

from .board import (
    BLACK,
    EMPTY,
    PLAYER_NAMES,
    WHITE,
    Board,
    IllegalMoveError,
    Move,
    opponent,
    parse_square,
    to_notation,
)


@dataclass
class MoveRecord:
    """Uma entrada do histórico da partida (jogada ou passe)."""

    number: int  # número sequencial da entrada no histórico (1, 2, ...)
    player: int
    square: str  # notação ("d3") ou "passe"
    flipped: list[str] = field(default_factory=list)
    black_count: int = 0
    white_count: int = 0
    actor: str = ""  # nome do agente/humano que jogou
    explanation: str = ""  # justificativa da jogada (quando houver)

    @property
    def is_pass(self) -> bool:
        return self.square == "passe"

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["player"] = PLAYER_NAMES[self.player]
        return data

    def describe(self) -> str:
        who = PLAYER_NAMES[self.player]
        if self.is_pass:
            return f"{self.number}. {who} passaram a vez (sem jogadas válidas)"
        text = (
            f"{self.number}. {who} jogaram {self.square} e viraram "
            f"{len(self.flipped)} peça(s) -> placar P {self.black_count} x {self.white_count} B"
        )
        return text


class OthelloGame:
    """Partida de Othello com controle de turno e histórico.

    Os passes são automáticos: depois de cada jogada, se o adversário não tem
    jogada válida mas o jogador atual tem, registra-se um passe e o turno
    volta para o jogador atual. Se nenhum dos dois pode jogar a partida acaba.
    """

    def __init__(self, board: Board | None = None, current_player: int = BLACK) -> None:
        self.game_id = uuid.uuid4().hex[:12]
        self.board = board or Board.initial()
        self.current_player = current_player
        self.history: list[MoveRecord] = []
        self._lock = threading.RLock()
        # Caso a posição inicial (personalizada) já obrigue um passe.
        self._resolve_passes()

    # ------------------------------------------------------------------
    # Consultas
    # ------------------------------------------------------------------
    @property
    def game_over(self) -> bool:
        return self.board.is_terminal()

    def valid_moves(self, player: int | None = None) -> list[Move]:
        player = self.current_player if player is None else player
        if self.game_over:
            return []
        return self.board.legal_moves(player)

    def score(self) -> dict[str, int]:
        return {"pretas": self.board.count(BLACK), "brancas": self.board.count(WHITE)}

    def winner(self) -> int | None:
        return self.board.winner()

    def moves_played(self) -> int:
        """Quantidade de jogadas efetivas (passes não contam)."""
        return sum(1 for rec in self.history if not rec.is_pass)

    def last_move(self) -> MoveRecord | None:
        for rec in reversed(self.history):
            if not rec.is_pass:
                return rec
        return None

    def status(self) -> dict[str, Any]:
        """Resumo do estado da partida (usado pela ferramenta get_game_status)."""
        winner = self.winner()
        if winner is None:
            result = "em andamento"
        elif winner == EMPTY:
            result = "empate"
        else:
            result = f"vitória das {PLAYER_NAMES[winner]}"
        last = self.last_move()
        return {
            "partida_encerrada": self.game_over,
            "resultado": result,
            "vez_de": None if self.game_over else PLAYER_NAMES[self.current_player],
            "placar": self.score(),
            "jogadas_realizadas": self.moves_played(),
            "casas_vazias": self.board.empty_count(),
            "jogadas_validas_do_jogador_da_vez": len(self.valid_moves()),
            "ultima_jogada": None if last is None else f"{PLAYER_NAMES[last.player]} em {last.square}",
        }

    def history_lines(self, last_n: int | None = None) -> list[str]:
        records = self.history if last_n is None else self.history[-last_n:]
        return [rec.describe() for rec in records]

    # ------------------------------------------------------------------
    # Ações
    # ------------------------------------------------------------------
    def make_move(
        self,
        move: Move | str,
        *,
        actor: str = "",
        explanation: str = "",
        player: int | None = None,
    ) -> MoveRecord:
        """Executa a jogada do jogador da vez.

        Aceita (linha, coluna) ou notação ("d3"). Levanta ``IllegalMoveError``
        se a partida acabou, se não é a vez de ``player`` ou se a jogada é
        inválida. Depois da jogada resolve passes automáticos.
        """
        with self._lock:
            if self.game_over:
                raise IllegalMoveError("A partida já terminou")
            if player is not None and player != self.current_player:
                raise IllegalMoveError(
                    f"Não é a vez das {PLAYER_NAMES[player]}; é a vez das "
                    f"{PLAYER_NAMES[self.current_player]}"
                )
            if isinstance(move, str):
                try:
                    move = parse_square(move)
                except ValueError as exc:
                    raise IllegalMoveError(str(exc)) from exc

            mover = self.current_player
            flipped = self.board.flips(mover, move)
            self.board = self.board.apply_move(mover, move)  # valida a jogada
            record = MoveRecord(
                number=len(self.history) + 1,
                player=mover,
                square=to_notation(move),
                flipped=[to_notation(m) for m in flipped],
                black_count=self.board.count(BLACK),
                white_count=self.board.count(WHITE),
                actor=actor,
                explanation=explanation,
            )
            self.history.append(record)
            self.current_player = opponent(mover)
            self._resolve_passes()
            return record

    def _resolve_passes(self) -> None:
        """Registra passes enquanto o jogador da vez não tiver jogadas."""
        if self.game_over:
            return
        if not self.board.has_moves(self.current_player):
            self.history.append(
                MoveRecord(
                    number=len(self.history) + 1,
                    player=self.current_player,
                    square="passe",
                    black_count=self.board.count(BLACK),
                    white_count=self.board.count(WHITE),
                )
            )
            self.current_player = opponent(self.current_player)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------
    def copy(self) -> "OthelloGame":
        clone = OthelloGame.__new__(OthelloGame)
        clone.game_id = self.game_id
        clone.board = self.board
        clone.current_player = self.current_player
        clone.history = [MoveRecord(**asdict(r)) for r in self.history]
        clone._lock = threading.RLock()
        return clone

    def render(self, show_moves: bool = True) -> str:
        highlight = set(self.valid_moves()) if show_moves else set()
        return self.board.to_ascii(highlight)


__all__ = ["OthelloGame", "MoveRecord", "BLACK", "WHITE", "EMPTY"]
