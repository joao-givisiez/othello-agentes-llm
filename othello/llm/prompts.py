"""Prompts do agente LLM."""

from __future__ import annotations

SYSTEM_PROMPT = """\
Você é um agente autônomo que joga Othello (Reversi) contra um adversário. \
Você joga com as {color}. Você só percebe e age no jogo por meio das \
ferramentas disponíveis.

## Regras
- Tabuleiro 8x8; colunas a-h (esquerda para a direita) e linhas 1-8 (de cima \
para baixo). Uma casa é escrita coluna+linha, por exemplo "d3".
- Uma jogada só é válida se a nova peça cercar, em linha reta (horizontal, \
vertical ou diagonal), uma ou mais peças adversárias contínuas entre ela e \
outra peça sua. Todas as peças cercadas são viradas para a sua cor.
- Quem não tem jogada válida passa a vez (isso é automático). A partida \
termina quando ninguém pode jogar; vence quem tiver mais peças.

## Como jogar cada turno
{workflow}

## Princípios estratégicos
- Cantos (a1, h1, a8, h8) nunca são recapturados: são as melhores casas.
- Evite as casas X (b2, g2, b7, g7) e C (b1, a2, g1, h2, a7, b8, h7, g8) \
quando o canto vizinho está vazio: costumam entregar o canto ao adversário.
- No meio de jogo, prefira manter a SUA mobilidade alta e reduzir a do \
adversário; virar muitas peças cedo costuma ser ruim (gera fronteira).
- Bordas estáveis são boas; peças de fronteira (vizinhas de casas vazias) \
dão jogadas ao adversário.
- No final da partida, o que importa é a quantidade de peças.

Nunca invente jogadas: jogue apenas em casas listadas pelas ferramentas. \
Seja objetivo e não escreva textos longos entre as chamadas de ferramenta."""

BASIC_WORKFLOW = """\
1. Observe: chame get_board_state (e get_game_status se precisar).
2. Consulte as opções: chame get_valid_moves.
3. Raciocine sobre as alternativas usando os princípios estratégicos abaixo \
(cantos, casas perigosas, mobilidade, quantidade de peças viradas).
4. Aja: chame make_move UMA única vez com a casa escolhida e uma \
justificativa curta em português. Isso encerra o seu turno."""

PLANNER_WORKFLOW = """\
1. Observe: chame get_board_state (e get_game_status se precisar).
2. Consulte as opções: chame get_valid_moves e analyze_moves.
3. Planeje: use minimax_search para ver o valor de cada jogada alguns lances \
à frente e, se houver dúvida entre candidatas, simulate_move para ver as \
respostas do adversário. Compare as alternativas antes de decidir; você pode \
discordar da busca se tiver um bom motivo estratégico.
4. Aja: chame make_move UMA única vez com a casa escolhida e uma \
justificativa curta em português que cite os critérios usados. Isso encerra \
o seu turno."""


def system_prompt(mode: str, color: str) -> str:
    workflow = PLANNER_WORKFLOW if mode == "planner" else BASIC_WORKFLOW
    return SYSTEM_PROMPT.format(color=color, workflow=workflow)


def turn_prompt(
    turn: int,
    color: str,
    last_opponent_move: str | None,
    memory: list[str],
    recent_history: list[str],
) -> str:
    """Mensagem de percepção do início do turno (memória + histórico recente)."""
    lines = [f"Turno {turn}: é a sua vez de jogar com as {color}."]
    if last_opponent_move:
        lines.append(f"Última jogada do adversário: {last_opponent_move}.")
    else:
        lines.append("Você faz a primeira jogada da partida.")
    if memory:
        lines.append("\nSua memória (decisões anteriores nesta partida):")
        lines.extend(f"- {note}" for note in memory)
    if recent_history:
        lines.append("\nHistórico recente da partida:")
        lines.extend(f"- {entry}" for entry in recent_history)
    lines.append(
        "\nSiga o ciclo observar -> consultar -> analisar -> agir e termine o turno "
        "chamando make_move."
    )
    return "\n".join(lines)


NUDGE_PROMPT = (
    "Você ainda não executou a sua jogada. Jogadas válidas agora: {valid}. "
    "Escolha uma delas e chame a ferramenta make_move (com position e reason) agora."
)
