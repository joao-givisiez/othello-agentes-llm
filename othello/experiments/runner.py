"""Partidas em lote entre agentes e coleta de métricas.

Protocolo
---------
* Torneio todos-contra-todos: cada par de agentes joga ``games_per_pair``
  partidas, alternando as cores a cada partida.
* Aberturas aleatórias: cada par de partidas (A de pretas / B de pretas) começa
  da mesma sequência de ``opening_plies`` lances aleatórios. Isso evita que
  agentes determinísticos repitam sempre a mesma partida e mantém a
  comparação justa (os dois lados jogam a mesma abertura com as duas cores).
* Pontuação do torneio: vitória = 3, empate = 1, derrota = 0.

Saídas (em ``out_dir``)
-----------------------
* ``games.csv``    uma linha por partida (gravada a cada partida, então uma
                   execução interrompida não perde o que já foi jogado);
* ``moves.jsonl``  uma linha por jogada, com justificativa e ferramentas
                   usadas (base da análise qualitativa);
* ``summary.json`` métricas agregadas;
* ``report.md``    tabelas prontas para o relatório e gráficos ``*.png``.
"""

from __future__ import annotations

import csv
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from ..agents.base import Agent
from ..agents.registry import create_agent
from ..game import BLACK, EMPTY, WHITE, OthelloGame, to_notation


@dataclass
class SideStats:
    """Métricas de um agente em uma partida."""

    spec: str
    agent: str
    moves: int = 0
    time: float = 0.0
    llm_calls: int = 0
    llm_time: float = 0.0
    tool_calls: int = 0
    invalid_attempts: int = 0
    fallbacks: int = 0
    max_move_time: float = 0.0


@dataclass
class GameResult:
    index: int
    black: SideStats
    white: SideStats
    black_discs: int
    white_discs: int
    winner: str  # "black", "white" ou "draw"
    total_moves: int
    passes: int
    opening: list[str] = field(default_factory=list)
    duration: float = 0.0

    def winner_spec(self) -> str | None:
        if self.winner == "draw":
            return None
        return self.black.spec if self.winner == "black" else self.white.spec

    def row(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "partida": self.index,
            "pretas": self.black.spec,
            "brancas": self.white.spec,
            "vencedor": {"black": self.black.spec, "white": self.white.spec}.get(self.winner, "empate"),
            "pecas_pretas": self.black_discs,
            "pecas_brancas": self.white_discs,
            "jogadas": self.total_moves,
            "passes": self.passes,
            "abertura": " ".join(self.opening),
            "duracao_s": round(self.duration, 3),
        }
        for color, side in (("pretas", self.black), ("brancas", self.white)):
            data[f"{color}_tempo_medio_s"] = round(side.time / side.moves, 4) if side.moves else 0
            data[f"{color}_chamadas_llm"] = side.llm_calls
            data[f"{color}_tempo_llm_s"] = round(side.llm_time, 3)
            data[f"{color}_chamadas_ferramentas"] = side.tool_calls
            data[f"{color}_jogadas_invalidas"] = side.invalid_attempts
            data[f"{color}_jogadas_seguranca"] = side.fallbacks
        return data


def random_opening(rng: random.Random, plies: int) -> list[str]:
    game = OthelloGame()
    sequence: list[str] = []
    for _ in range(plies):
        if game.game_over:
            break
        move = rng.choice(game.valid_moves())
        game.make_move(move)
        sequence.append(to_notation(move))
    return sequence


def play_game(
    black: Agent,
    white: Agent,
    black_spec: str,
    white_spec: str,
    opening: list[str] | None = None,
    index: int = 0,
    move_log: Callable[[dict[str, Any]], None] | None = None,
) -> GameResult:
    """Joga uma partida completa e devolve o resultado com as métricas."""
    game = OthelloGame()
    for square in opening or []:
        game.make_move(square, actor="abertura")
    for agent in (black, white):
        agent.reset()

    sides = {BLACK: SideStats(black_spec, black.name), WHITE: SideStats(white_spec, white.name)}
    agents = {BLACK: black, WHITE: white}
    start = time.perf_counter()
    while not game.game_over:
        player = game.current_player
        result = agents[player].play_turn(game)
        side = sides[player]
        side.moves += 1
        side.time += result.elapsed
        side.max_move_time = max(side.max_move_time, result.elapsed)
        side.llm_calls += result.llm_calls
        side.llm_time += result.llm_time
        side.tool_calls += result.tool_calls
        side.invalid_attempts += result.invalid_attempts
        side.fallbacks += int(result.fallback)
        if move_log is not None:
            move_log(
                {
                    "partida": index,
                    "lance": game.moves_played(),
                    "cor": "pretas" if player == BLACK else "brancas",
                    "agente": agents[player].name,
                    "spec": side.spec,
                    "jogada": result.move,
                    "tempo_s": round(result.elapsed, 4),
                    "chamadas_llm": result.llm_calls,
                    "chamadas_ferramentas": result.tool_calls,
                    "ferramentas": result.details.get("ferramentas"),
                    "jogadas_invalidas": result.invalid_attempts,
                    "jogada_seguranca": result.fallback,
                    "justificativa": result.explanation,
                }
            )
    duration = time.perf_counter() - start

    winner = game.winner()
    return GameResult(
        index=index,
        black=sides[BLACK],
        white=sides[WHITE],
        black_discs=game.board.count(BLACK),
        white_discs=game.board.count(WHITE),
        winner={BLACK: "black", WHITE: "white", EMPTY: "draw"}[winner],  # type: ignore[index]
        total_moves=game.moves_played(),
        passes=sum(1 for r in game.history if r.is_pass),
        opening=list(opening or []),
        duration=duration,
    )


# ---------------------------------------------------------------------------
# Agregação
# ---------------------------------------------------------------------------
def summarize(results: list[GameResult], specs: list[str], names: dict[str, str]) -> dict[str, Any]:
    agents: dict[str, dict[str, Any]] = {
        spec: {
            "spec": spec,
            "nome": names.get(spec, spec),
            "partidas": 0,
            "vitorias": 0,
            "empates": 0,
            "derrotas": 0,
            "saldo_pecas": 0,
            "jogadas": 0,
            "tempo_total_s": 0.0,
            "tempo_max_jogada_s": 0.0,
            "chamadas_llm": 0,
            "tempo_llm_s": 0.0,
            "chamadas_ferramentas": 0,
            "jogadas_invalidas": 0,
            "jogadas_seguranca": 0,
        }
        for spec in specs
    }
    pairs: dict[str, dict[str, Any]] = {}

    for res in results:
        for side, own, opp in (
            (res.black, res.black_discs, res.white_discs),
            (res.white, res.white_discs, res.black_discs),
        ):
            a = agents[side.spec]
            a["partidas"] += 1
            a["saldo_pecas"] += own - opp
            a["jogadas"] += side.moves
            a["tempo_total_s"] += side.time
            a["tempo_max_jogada_s"] = max(a["tempo_max_jogada_s"], side.max_move_time)
            a["chamadas_llm"] += side.llm_calls
            a["tempo_llm_s"] += side.llm_time
            a["chamadas_ferramentas"] += side.tool_calls
            a["jogadas_invalidas"] += side.invalid_attempts
            a["jogadas_seguranca"] += side.fallbacks
        win = res.winner_spec()
        if win is None:
            agents[res.black.spec]["empates"] += 1
            agents[res.white.spec]["empates"] += 1
        else:
            loser = res.white.spec if win == res.black.spec else res.black.spec
            agents[win]["vitorias"] += 1
            agents[loser]["derrotas"] += 1

        a_spec, b_spec = sorted((res.black.spec, res.white.spec), key=specs.index)
        key = f"{a_spec} x {b_spec}"
        pair = pairs.setdefault(
            key, {"a": a_spec, "b": b_spec, "partidas": 0, "vitorias_a": 0, "empates": 0, "vitorias_b": 0, "saldo_a": 0}
        )
        pair["partidas"] += 1
        if win is None:
            pair["empates"] += 1
        elif win == a_spec:
            pair["vitorias_a"] += 1
        else:
            pair["vitorias_b"] += 1
        a_discs = res.black_discs if res.black.spec == a_spec else res.white_discs
        b_discs = res.white_discs if res.black.spec == a_spec else res.black_discs
        pair["saldo_a"] += a_discs - b_discs

    for a in agents.values():
        games = a["partidas"] or 1
        moves = a["jogadas"] or 1
        a["pontos"] = 3 * a["vitorias"] + a["empates"]
        a["taxa_vitoria"] = a["vitorias"] / games
        a["saldo_medio_pecas"] = a["saldo_pecas"] / games
        a["tempo_medio_jogada_s"] = a["tempo_total_s"] / moves
        a["chamadas_llm_por_jogada"] = a["chamadas_llm"] / moves
        a["tempo_medio_chamada_llm_s"] = a["tempo_llm_s"] / a["chamadas_llm"] if a["chamadas_llm"] else 0.0
        a["ferramentas_por_jogada"] = a["chamadas_ferramentas"] / moves

    standings = sorted(
        agents.values(), key=lambda a: (a["pontos"], a["saldo_pecas"]), reverse=True
    )
    total = len(results) or 1
    return {
        "classificacao": standings,
        "confrontos": list(pairs.values()),
        "partidas": len(results),
        "media_jogadas_por_partida": sum(r.total_moves for r in results) / total,
        "vitorias_pretas": sum(r.winner == "black" for r in results),
        "vitorias_brancas": sum(r.winner == "white" for r in results),
        "empates": sum(r.winner == "draw" for r in results),
    }


# ---------------------------------------------------------------------------
# Torneio
# ---------------------------------------------------------------------------
def run_tournament(
    specs: list[str],
    games_per_pair: int = 10,
    opening_plies: int = 4,
    seed: int = 42,
    model: str | None = None,
    out_dir: str | Path = "results/experimento",
    pairs: list[tuple[str, str]] | None = None,
    progress: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Executa o torneio e grava os arquivos de resultado em ``out_dir``."""
    from .report import write_report

    if games_per_pair % 2:
        games_per_pair += 1  # número par: cada agente joga metade com cada cor
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    pairs = pairs or list(itertools.combinations(specs, 2))
    results: list[GameResult] = []
    names: dict[str, str] = {}
    started = time.strftime("%Y-%m-%d %H:%M:%S")

    with open(out / "games.csv", "w", newline="", encoding="utf-8") as csv_file, open(
        out / "moves.jsonl", "w", encoding="utf-8"
    ) as moves_file:
        writer: csv.DictWriter | None = None

        def log_move(entry: dict[str, Any]) -> None:
            moves_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

        index = 0
        for a_spec, b_spec in pairs:
            agent_a = create_agent(a_spec, seed=rng.randrange(1 << 30), model=model)
            agent_b = create_agent(b_spec, seed=rng.randrange(1 << 30), model=model)
            names[a_spec], names[b_spec] = agent_a.name, agent_b.name
            opening: list[str] = []
            for g in range(games_per_pair):
                if g % 2 == 0:
                    opening = random_opening(rng, opening_plies)
                black, white = (agent_a, agent_b) if g % 2 == 0 else (agent_b, agent_a)
                black_spec, white_spec = (a_spec, b_spec) if g % 2 == 0 else (b_spec, a_spec)
                index += 1
                res = play_game(black, white, black_spec, white_spec, opening, index, log_move)
                results.append(res)
                row = res.row()
                if writer is None:
                    writer = csv.DictWriter(csv_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                csv_file.flush()
                moves_file.flush()
                outcome = row["vencedor"]
                progress(
                    f"[{index}] {black_spec} (P) {res.black_discs} x {res.white_discs} {white_spec} (B)"
                    f" -> {outcome} ({res.duration:.1f}s)"
                )

    summary = summarize(results, specs, names)
    summary["configuracao"] = {
        "agentes": specs,
        "partidas_por_par": games_per_pair,
        "lances_de_abertura_aleatorios": opening_plies,
        "semente": seed,
        "modelo_llm": model,
        "inicio": started,
        "fim": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    write_report(summary, out)
    return summary


__all__ = ["GameResult", "SideStats", "play_game", "random_opening", "run_tournament", "summarize"]
