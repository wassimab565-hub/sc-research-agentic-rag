import argparse
import sys
import time
from collections.abc import Sequence

from sc_research_agentic_rag.llm.base import ChatMessage
from sc_research_agentic_rag.llm.config import OllamaSettings
from sc_research_agentic_rag.llm.errors import LLMError
from sc_research_agentic_rag.llm.ollama_client import OllamaClient

DEFAULT_PROMPT = "Réponds en une phrase : c'est quoi le RAG ?"


def build_client(settings: OllamaSettings) -> OllamaClient:
    return OllamaClient(settings)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check that the local LLM answers.")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("--model", help="override OLLAMA_MODEL")
    args = parser.parse_args(argv)

    settings = OllamaSettings()
    if args.model:
        settings = settings.model_copy(update={"model": args.model})
    client = build_client(settings)

    try:
        installed = client.list_models()
        if settings.model not in installed and f"{settings.model}:latest" not in installed:
            print(
                f"error: model '{settings.model}' is not installed. Installed: {', '.join(installed) or 'none'}. "
                f"Run: ollama pull {settings.model}",
                file=sys.stderr,
            )
            return 1
        started = time.perf_counter()
        answer = client.chat([ChatMessage("user", args.prompt)])
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(answer)
    print(f"\n({time.perf_counter() - started:.1f}s, model {settings.model})")
    return 0