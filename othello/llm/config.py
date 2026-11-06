"""Criação do modelo de linguagem usado pelos agentes LLM.

O modelo é escolhido por uma especificação "provedor:modelo" no formato do
``init_chat_model`` do LangChain, lida (em ordem de prioridade) do argumento
``--model``, da variável de ambiente ``OTHELLO_LLM_MODEL`` (ou do arquivo
``.env``) ou do padrão ``anthropic:claude-opus-5-5``. Outros provedores
suportados pelo LangChain funcionam instalando o pacote correspondente, por
exemplo ``groq:openai/gpt-oss-120b`` com ``langchain-groq`` e ``GROQ_API_KEY``.

A especificação especial ``mock`` usa um modelo simulado, sem LLM real, que
segue o mesmo protocolo de ferramentas - útil para testes e para demonstrar o
fluxo do agente sem chave de API.
"""

from __future__ import annotations

import os

from langchain_core.language_models import BaseChatModel

DEFAULT_MODEL = "anthropic:claude-opus-5-5"


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - python-dotenv é opcional
        return
    load_dotenv()


def resolve_model_spec(spec: str | None = None) -> str:
    _load_dotenv()
    return (spec or os.getenv("OTHELLO_LLM_MODEL") or DEFAULT_MODEL).strip()


def create_chat_model(spec: str | None = None) -> BaseChatModel:
    """Instancia o modelo de chat a partir da especificação."""
    spec = resolve_model_spec(spec)
    if spec == "mock":
        from .mock import MockOthelloChatModel

        return MockOthelloChatModel()

    from langchain.chat_models import init_chat_model

    provider = spec.split(":", 1)[0] if ":" in spec else ""
    kwargs: dict = {"max_retries": 3}
    if provider == "anthropic":
        # Com raciocínio adaptativo, o raciocínio interno conta em max_tokens.
        kwargs["max_tokens"] = 16000
        kwargs["timeout"] = 300
        effort = os.getenv("OTHELLO_LLM_EFFORT")
        if effort:
            kwargs["effort"] = effort.strip()
    elif provider == "groq":
        kwargs["timeout"] = 300
        effort = os.getenv("OTHELLO_LLM_EFFORT")
        if effort:  # modelos de raciocínio do Groq (ex.: openai/gpt-oss-*)
            kwargs["reasoning_effort"] = effort.strip()
    elif provider in ("openai", "google_genai"):
        kwargs["timeout"] = 300
    return init_chat_model(spec, **kwargs)


def model_display_name(model: BaseChatModel, spec: str | None = None) -> str:
    for attr in ("model", "model_name"):
        value = getattr(model, attr, None)
        if isinstance(value, str) and value:
            return value
    return spec or type(model).__name__
