"""Ponto de entrada: ``python -m othello <comando>``.

Comandos:
    gui         interface gráfica (padrão quando nenhum comando é dado)
    play        partida no terminal
    experiment  torneio entre agentes com métricas e relatório
    report      regenera o relatório de uma pasta de resultados
    graph      imprime o grafo LangGraph do agente LLM (Mermaid)
"""

from __future__ import annotations

import argparse
import sys

from .agents.registry import AGENT_CHOICES


def _add_model_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        default=None,
        help="modelo dos agentes LLM no formato provedor:modelo (ex.: anthropic:claude-opus-5-5) "
        "ou 'mock' para o modelo simulado; padrão: OTHELLO_LLM_MODEL ou anthropic:claude-opus-5-5",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m othello", description="Othello com agentes inteligentes")
    sub = parser.add_subparsers(dest="command")

    gui = sub.add_parser("gui", help="abre a interface gráfica")
    _add_model_arg(gui)

    agents_help = "humano ('human') ou agente: " + ", ".join(f"{k}" for k in AGENT_CHOICES) + " (minimax:N)"
    play = sub.add_parser("play", help="partida no terminal")
    play.add_argument("--black", default="human", help=agents_help)
    play.add_argument("--white", default="minimax:4", help=agents_help)
    play.add_argument("--delay", type=float, default=0.0, help="pausa (s) após cada jogada de agente")
    play.add_argument("--seed", type=int, default=None, help="semente dos agentes (reprodutibilidade)")
    _add_model_arg(play)

    exp = sub.add_parser("experiment", help="torneio entre agentes com métricas (vitória=3, empate=1)")
    exp.add_argument(
        "--agents",
        nargs="+",
        default=["random", "greedy", "heuristic", "minimax:2", "minimax:4"],
        help="agentes participantes (padrão: os clássicos)",
    )
    exp.add_argument("--games", type=int, default=10, help="partidas por confronto (arredondado para par)")
    exp.add_argument("--opening", type=int, default=4, help="lances aleatórios de abertura por par de partidas")
    exp.add_argument("--seed", type=int, default=42)
    exp.add_argument(
        "--vs",
        default=None,
        help="em vez de todos contra todos, cada agente de --agents enfrenta apenas este agente",
    )
    exp.add_argument("--out", default=None, help="pasta de saída (padrão: results/<data-hora>)")
    _add_model_arg(exp)

    report = sub.add_parser("report", help="regenera report.md e gráficos de uma pasta de resultados")
    report.add_argument("folder", help="pasta com summary.json (e moves.jsonl)")

    graph = sub.add_parser("graph", help="mostra o grafo LangGraph do agente LLM em Mermaid")
    graph.add_argument("--mode", choices=["basic", "planner"], default="planner")

    return parser


def main(argv: list[str] | None = None) -> None:
    try:
        if sys.stdout.isatty():
            sys.stdout.reconfigure(errors="replace")  # type: ignore[attr-defined]
        else:  # saída redirecionada para arquivo/pipe: UTF-8
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    command = args.command or "gui"

    if command == "gui":
        from .ui.gui import run_gui

        run_gui(model=getattr(args, "model", None))
    elif command == "play":
        from .ui.cli import run_cli

        run_cli(args.black, args.white, model=args.model, delay=args.delay, seed=args.seed)
    elif command == "experiment":
        import time

        from .experiments import run_tournament

        specs = list(dict.fromkeys(args.agents))
        pairs = None
        if args.vs:
            pairs = [(spec, args.vs) for spec in specs if spec != args.vs]
            specs = specs + [args.vs] if args.vs not in specs else specs
        out = args.out or f"results/{time.strftime('%Y%m%d-%H%M%S')}"
        summary = run_tournament(
            specs,
            games_per_pair=args.games,
            opening_plies=args.opening,
            seed=args.seed,
            model=args.model,
            out_dir=out,
            pairs=pairs,
        )
        print(f"\nClassificação (resultados completos em {out}/report.md):")
        for i, a in enumerate(summary["classificacao"], 1):
            print(
                f"{i}. {a['nome']:<16} {a['pontos']:>4} pts  V{a['vitorias']} E{a['empates']} D{a['derrotas']}"
                f"  vitória {100 * a['taxa_vitoria']:.0f}%  tempo/jogada {1000 * a['tempo_medio_jogada_s']:.1f} ms"
            )
    elif command == "report":
        import json
        from pathlib import Path

        from .experiments.report import write_report

        folder = Path(args.folder)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        write_report(summary, folder)
        print(f"Relatório regenerado em {folder / 'report.md'}")
    elif command == "graph":
        from .llm.agent import LLMAgent

        print(LLMAgent(mode=args.mode, model="mock").graph_mermaid())


if __name__ == "__main__":
    main()
