"""Testes do módulo de experimentos (torneio, métricas e relatório)."""

import csv
import json
import random

from othello.agents import GreedyAgent, RandomAgent
from othello.experiments import play_game, random_opening, run_tournament


def test_random_opening_is_reproducible():
    assert random_opening(random.Random(5), 4) == random_opening(random.Random(5), 4)
    assert len(random_opening(random.Random(5), 4)) == 4


def test_play_game_collects_metrics():
    moves = []
    result = play_game(
        GreedyAgent(seed=1), RandomAgent(seed=2), "greedy", "random", ["d3", "c3"], index=1, move_log=moves.append
    )
    assert result.black_discs + result.white_discs <= 64
    assert result.total_moves == result.black.moves + result.white.moves + 2  # +2 da abertura
    assert result.winner in ("black", "white", "draw")
    assert len(moves) == result.black.moves + result.white.moves
    assert moves[0]["spec"] == "greedy"


def test_tournament_writes_outputs_and_scores(tmp_path):
    lines = []
    summary = run_tournament(
        ["random", "greedy", "heuristic"],
        games_per_pair=3,  # arredondado para 4
        opening_plies=2,
        seed=7,
        out_dir=tmp_path,
        progress=lines.append,
    )
    assert summary["partidas"] == 12 and len(lines) == 12
    standings = summary["classificacao"]
    assert sum(a["partidas"] for a in standings) == 24
    for a in standings:
        assert a["pontos"] == 3 * a["vitorias"] + a["empates"]
        assert a["vitorias"] + a["empates"] + a["derrotas"] == a["partidas"] == 8
    with open(tmp_path / "games.csv", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 12
    assert json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))["partidas"] == 12
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Classificação" in report and "Confrontos diretos" in report
    assert "Uso de casas estratégicas" in report
    assert (tmp_path / "taxa_vitoria.png").exists()


def test_tournament_with_mock_llm_reports_tool_usage(tmp_path):
    summary = run_tournament(
        ["llm-basic", "random"], games_per_pair=2, opening_plies=0, seed=1, model="mock", out_dir=tmp_path,
        progress=lambda _: None,
    )
    llm = next(a for a in summary["classificacao"] if a["spec"] == "llm-basic")
    assert llm["chamadas_llm_por_jogada"] >= 2
    assert llm["ferramentas_por_jogada"] >= 3
    assert "Custo dos agentes LLM" in (tmp_path / "report.md").read_text(encoding="utf-8")
