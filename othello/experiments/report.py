"""Relatório dos experimentos: tabelas em Markdown e gráficos."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Paleta: uma única série por gráfico -> um único tom (azul da paleta de
# referência), tinta de texto neutra e grade discreta.
SERIES_COLOR = "#2a78d6"
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _time(seconds: float) -> str:
    if seconds < 1:
        return f"{1000 * seconds:.1f} ms"
    return f"{seconds:.2f} s"


def standings_table(summary: dict[str, Any]) -> str:
    lines = [
        "| # | Agente | J | V | E | D | Pontos | Taxa de vitória | Saldo médio de peças | Tempo médio/jogada |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, a in enumerate(summary["classificacao"], 1):
        lines.append(
            f"| {i} | {a['nome']} (`{a['spec']}`) | {a['partidas']} | {a['vitorias']} | {a['empates']} | "
            f"{a['derrotas']} | **{a['pontos']}** | {_pct(a['taxa_vitoria'])} | "
            f"{a['saldo_medio_pecas']:+.1f} | {_time(a['tempo_medio_jogada_s'])} |"
        )
    return "\n".join(lines)


def head_to_head_table(summary: dict[str, Any]) -> str:
    lines = [
        "| Confronto (A x B) | Partidas | Vitórias A | Empates | Vitórias B | Saldo médio de A |",
        "|---|---|---|---|---|---|",
    ]
    for p in summary["confrontos"]:
        lines.append(
            f"| `{p['a']}` x `{p['b']}` | {p['partidas']} | {p['vitorias_a']} | {p['empates']} | "
            f"{p['vitorias_b']} | {p['saldo_a'] / max(p['partidas'], 1):+.1f} |"
        )
    return "\n".join(lines)


def llm_cost_table(summary: dict[str, Any]) -> str | None:
    rows = [a for a in summary["classificacao"] if a["chamadas_llm"] or a["chamadas_ferramentas"]]
    if not rows:
        return None
    lines = [
        "| Agente | Jogadas | Chamadas ao LLM/jogada | Tempo médio por chamada | Ferramentas/jogada | "
        "Jogadas inválidas | Jogadas de segurança | Maior tempo de jogada |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for a in rows:
        lines.append(
            f"| {a['nome']} (`{a['spec']}`) | {a['jogadas']} | {a['chamadas_llm_por_jogada']:.2f} | "
            f"{_time(a['tempo_medio_chamada_llm_s'])} | {a['ferramentas_por_jogada']:.2f} | "
            f"{a['jogadas_invalidas']} | {a['jogadas_seguranca']} | {_time(a['tempo_max_jogada_s'])} |"
        )
    return "\n".join(lines)


def square_usage(out: Path) -> dict[str, dict[str, Any]]:
    """Frequência de jogadas por tipo de casa para cada agente (de moves.jsonl).

    Casas X/C só contam como "arriscadas" pelo tipo da casa; como o log não
    guarda o tabuleiro, não se sabe se o canto vizinho estava vazio.
    """
    import json

    from ..agents.evaluation import square_kind
    from ..game import parse_square

    path = out / "moves.jsonl"
    if not path.exists():
        return {}
    usage: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if not entry.get("jogada") or entry["jogada"] in ("?", "passe"):
                continue
            kind = square_kind(parse_square(entry["jogada"])).split(" (")[0]
            agent = usage.setdefault(
                entry["spec"],
                {"nome": entry["agente"], "total": 0, "canto": 0, "casa X": 0, "casa C": 0, "borda": 0, "interior": 0},
            )
            agent["total"] += 1
            agent[kind] += 1
    return usage


def square_usage_table(usage: dict[str, dict[str, Any]], order: list[str]) -> str | None:
    if not usage:
        return None
    lines = [
        "| Agente | Jogadas | Cantos | Casas X | Casas C | Outras bordas | Interior |",
        "|---|---|---|---|---|---|---|",
    ]
    for spec in [s for s in order if s in usage]:
        u = usage[spec]
        total = u["total"] or 1
        lines.append(
            f"| {u['nome']} | {u['total']} | "
            + " | ".join(_pct(u[k] / total) for k in ("canto", "casa X", "casa C", "borda", "interior"))
            + " |"
        )
    return "\n".join(lines)


def _bar_chart(labels: list[str], values: list[float], title: str, xlabel: str, path: Path, log: bool = False,
               percent: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(labels) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    positions = list(range(len(labels)))[::-1]  # primeiro rótulo no topo
    ax.barh(positions, values, height=0.55, color=SERIES_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks(positions, labels, color=TEXT_PRIMARY, fontsize=9)
    if log:
        ax.set_xscale("log")
    if percent:
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{100 * v:.0f}%"))
    ax.set_xlabel(xlabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left", pad=10)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(GRID)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write_charts(summary: dict[str, Any], out: Path) -> list[str]:
    try:
        import matplotlib  # noqa: F401
    except ImportError:  # pragma: no cover
        return []
    standings = summary["classificacao"]
    labels = [a["nome"] for a in standings]
    _bar_chart(
        labels,
        [a["taxa_vitoria"] for a in standings],
        "Taxa de vitória no torneio",
        "partidas vencidas",
        out / "taxa_vitoria.png",
        percent=True,
    )
    by_time = sorted(standings, key=lambda a: a["tempo_medio_jogada_s"], reverse=True)
    _bar_chart(
        [a["nome"] for a in by_time],
        [max(a["tempo_medio_jogada_s"] * 1000, 1e-3) for a in by_time],
        "Tempo médio por jogada",
        "milissegundos por jogada (escala logarítmica)",
        out / "tempo_por_jogada.png",
        log=True,
    )
    return ["taxa_vitoria.png", "tempo_por_jogada.png"]


def write_report(summary: dict[str, Any], out: Path) -> None:
    cfg = summary.get("configuracao", {})
    charts = write_charts(summary, out)
    parts = [
        "# Resultados do experimento",
        "",
        f"- Agentes: {', '.join(f'`{s}`' for s in cfg.get('agentes', []))}",
        f"- Partidas por confronto: {cfg.get('partidas_por_par')} (cores alternadas; cada par de partidas "
        f"parte da mesma abertura aleatória de {cfg.get('lances_de_abertura_aleatorios')} lances)",
        f"- Semente: {cfg.get('semente')}"
        + (f" | Modelo LLM: `{cfg['modelo_llm']}`" if cfg.get("modelo_llm") else ""),
        f"- Execução: {cfg.get('inicio')} a {cfg.get('fim')}",
        f"- Total: {summary['partidas']} partidas, média de {summary['media_jogadas_por_partida']:.1f} jogadas "
        f"por partida; vitórias das pretas: {summary['vitorias_pretas']}, das brancas: "
        f"{summary['vitorias_brancas']}, empates: {summary['empates']}",
        "",
        "## Classificação (vitória = 3, empate = 1, derrota = 0)",
        "",
        standings_table(summary),
        "",
        "## Confrontos diretos",
        "",
        head_to_head_table(summary),
    ]
    cost = llm_cost_table(summary)
    if cost:
        parts += ["", "## Custo dos agentes LLM", "", cost]
    usage = square_usage_table(square_usage(out), [a["spec"] for a in summary["classificacao"]])
    if usage:
        parts += [
            "",
            "## Uso de casas estratégicas",
            "",
            "Porcentagem das jogadas de cada agente por tipo de casa (cantos = a1, h1, a8, h8; "
            "casas X = diagonais vizinhas dos cantos; casas C = vizinhas ortogonais dos cantos).",
            "",
            usage,
        ]
    if charts:
        parts += ["", "## Gráficos", ""] + [f"![{c}]({c})" for c in charts]
    (out / "report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
