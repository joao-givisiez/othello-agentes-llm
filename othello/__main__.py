"""Ponto de entrada: ``python -m othello <comando>``.

Comandos:
    gui         interface gráfica (padrão quando nenhum comando é dado)
    play        partida no terminal
    graph       imprime o grafo LangGraph do agente LLM (Mermaid)
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
    elif command == "graph":
        from .llm.agent import LLMAgent

        print(LLMAgent(mode=args.mode, model="mock").graph_mermaid())


if __name__ == "__main__":
    main()
