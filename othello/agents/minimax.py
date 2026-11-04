"""Agente Minimax com poda alfa-beta (formulação negamax).

* Busca até ``depth`` lances à frente e avalia as folhas com a heurística de
  ``evaluation.py``; posições finais recebem valor exato (vitória/derrota).
* Poda alfa-beta descarta ramos que não podem alterar a decisão.
* Ordenação de jogadas (cantos primeiro, casas X por último, e nos níveis
  mais altos a jogada que mais reduz a mobilidade do adversário) aumenta
  o número de podas.
* Fim de jogo exato: quando restam poucas casas vazias a busca vai até o
  final da partida, jogando de forma perfeita.
* Aprofundamento iterativo opcional com limite de tempo por jogada.
"""

from __future__ import annotations

import math
import time
from typing import Any

from ..game import Board, Move, to_notation
from ..game.board import flips_mask, index_to_move, iter_bits, legal_moves_mask, popcount
from .base import SearchAgent
from .evaluation import WEIGHTS, WIN_SCORE, analyze_move, describe_analysis, evaluate_bits, terminal_score

_WEIGHT_BY_INDEX = [WEIGHTS[i // 8][i % 8] for i in range(64)]


class _Timeout(Exception):
    pass


class AlphaBetaSearch:
    """Motor de busca reutilizável (agente Minimax e ferramentas do LLM)."""

    def __init__(self, deadline: float | None = None) -> None:
        self.nodes = 0
        self.deadline = deadline

    # ------------------------------------------------------------------
    def ordered_children(self, own: int, opp: int, moves: int, depth: int) -> list[tuple[int, int, int]]:
        """Filhos (bit, novo_own, novo_opp) em ordem promissora."""
        children = []
        for idx in iter_bits(moves):
            bit = 1 << idx
            flipped = flips_mask(own, opp, bit)
            new_own = own | bit | flipped
            new_opp = opp & ~flipped
            key = _WEIGHT_BY_INDEX[idx] * 10
            if depth >= 3:
                key -= 25 * popcount(legal_moves_mask(new_opp, new_own))
            children.append((key, idx, new_own, new_opp))
        children.sort(key=lambda c: c[0], reverse=True)
        return [(idx, new_own, new_opp) for _, idx, new_own, new_opp in children]

    def negamax(self, own: int, opp: int, depth: int, alpha: float, beta: float) -> float:
        self.nodes += 1
        if self.deadline is not None and (self.nodes & 511) == 0 and time.perf_counter() > self.deadline:
            raise _Timeout
        if depth <= 0:
            return evaluate_bits(own, opp)
        moves = legal_moves_mask(own, opp)
        if not moves:
            if not legal_moves_mask(opp, own):
                return terminal_score(own, opp)
            return -self.negamax(opp, own, depth - 1, -beta, -alpha)  # passe
        best = -math.inf
        for _, new_own, new_opp in self.ordered_children(own, opp, moves, depth):
            score = -self.negamax(new_opp, new_own, depth - 1, -beta, -alpha)
            if score > best:
                best = score
                if best > alpha:
                    alpha = best
                    if alpha >= beta:
                        break
        return best

    def search_root(
        self, own: int, opp: int, depth: int, order: list[int] | None = None, exact: bool = False
    ) -> list[tuple[int, float]]:
        """Avalia as jogadas da raiz. Com ``exact=True`` cada jogada recebe seu
        valor exato (janela cheia); senão apenas a melhor tem valor exato."""
        moves = legal_moves_mask(own, opp)
        children = {idx: (o, p) for idx, o, p in self.ordered_children(own, opp, moves, depth)}
        indices = order if order is not None else list(children)
        results = []
        alpha = -math.inf
        for idx in indices:
            new_own, new_opp = children[idx]
            lower = -math.inf if exact else alpha
            score = -self.negamax(new_opp, new_own, depth - 1, -math.inf, -lower)
            results.append((idx, score))
            alpha = max(alpha, score)
        return results


def score_moves(board: Board, player: int, depth: int) -> list[dict[str, Any]]:
    """Valor minimax exato de cada jogada válida (usado pela ferramenta do LLM)."""
    own, opp = board.bitboards(player)
    engine = AlphaBetaSearch()
    results = engine.search_root(own, opp, depth, exact=True)
    results.sort(key=lambda r: r[1], reverse=True)
    return [
        {"jogada": to_notation(index_to_move(idx)), "valor_minimax": _format_score(score)}
        for idx, score in results
    ]


def _format_score(score: float) -> float | str:
    if score >= WIN_SCORE:
        return f"vitória garantida por {int(score - WIN_SCORE)} peça(s)"
    if score <= -WIN_SCORE:
        return f"derrota forçada por {int(-WIN_SCORE - score)} peça(s)"
    return round(score, 1)


class MinimaxAgent(SearchAgent):
    """Minimax com poda alfa-beta, ordenação de jogadas e fim de jogo exato."""

    def __init__(
        self,
        depth: int = 4,
        time_limit: float | None = None,
        endgame_empties: int = 8,
        seed: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or f"Minimax-{depth}", seed=seed)
        self.depth = depth
        self.time_limit = time_limit
        self.endgame_empties = endgame_empties

    def select_move(self, board: Board, player: int) -> tuple[Move, str, dict[str, Any]]:
        own, opp = board.bitboards(player)
        indices = list(iter_bits(legal_moves_mask(own, opp)))
        self.rng.shuffle(indices)  # desempate aleatório (reprodutível pela semente)

        empties = board.empty_count()
        exact_endgame = empties <= self.endgame_empties
        max_depth = empties + 2 if exact_endgame else self.depth
        deadline = time.perf_counter() + self.time_limit if self.time_limit else None
        engine = AlphaBetaSearch(deadline=deadline)

        if len(indices) == 1:
            best_idx, best_score, reached = indices[0], math.nan, 0
        else:
            # Ordem inicial: heurística da raiz, mantendo o embaralhamento nos empates.
            ranked = engine.ordered_children(own, opp, legal_moves_mask(own, opp), max_depth)
            rank = {idx: i for i, (idx, _, _) in enumerate(ranked)}
            order = sorted(indices, key=lambda i: rank[i])
            best_idx, best_score, reached = order[0], -math.inf, 0
            depths = range(1, max_depth + 1) if deadline else [max_depth]
            for depth in depths:
                try:
                    results = engine.search_root(own, opp, depth, order=order)
                except _Timeout:
                    break
                best_idx, best_score = max(results, key=lambda r: r[1])
                reached = depth
                # Próxima iteração começa pela melhor jogada encontrada.
                order = [best_idx] + [i for i in order if i != best_idx]

        move = index_to_move(best_idx)
        info = analyze_move(board, player, move)
        if len(indices) == 1:
            summary = "única jogada possível"
        elif best_score >= WIN_SCORE:
            summary = f"vitória garantida por {int(best_score - WIN_SCORE)} peça(s) com jogo perfeito"
        elif best_score <= -WIN_SCORE:
            summary = "derrota inevitável; escolhe a jogada que perde por menos"
        else:
            kind = "fim de jogo exato" if exact_endgame else f"profundidade {reached}"
            summary = (
                f"melhor valor esperado ({best_score:.0f}) supondo a melhor resposta "
                f"do adversário ({kind}, {engine.nodes} nós)"
            )
        explanation = f"Minimax alfa-beta: {summary}. {describe_analysis(info)}."
        details = {
            "profundidade": reached,
            "nos": engine.nodes,
            "valor": None if math.isnan(best_score) else best_score,
            "fim_de_jogo_exato": exact_endgame,
        }
        return move, explanation, details
