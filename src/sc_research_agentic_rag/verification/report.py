# src\sc_research_agentic_rag\verification\
from collections import Counter
from collections.abc import Callable, Sequence

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError, EmbeddingSettings
from sc_research_agentic_rag.verification.citations import format_report, verify_citations
from sc_research_agentic_rag.verification.support import (
    DEFAULT_THRESHOLD,
    STATUS_NO_VALID_SOURCE,
    STATUS_NUMBER_MISSING,
    STATUS_SUPPORTED,
    STATUS_UNCHECKED,
    STATUS_WEAK,
    ClaimCheck,
    PassageLike,
    SupportEmbedder,
    check_claims,
)

_ORDER = (STATUS_SUPPORTED, STATUS_WEAK, STATUS_NUMBER_MISSING, STATUS_NO_VALID_SOURCE, STATUS_UNCHECKED)
_SHOWN = 10


def build_support_embedder() -> Embedder:
    """CPU on purpose: the retrieval embedder already sits on the GPU next to Ollama (CUDA out of memory, I-28)."""
    return Embedder.from_settings(EmbeddingSettings(device="cpu"))


def format_claims(checks: Sequence[ClaimCheck]) -> str:
    if not checks:
        return "Claims: none with a citation"
    counts = Counter(check.status for check in checks)
    summary = ", ".join(f"{counts[status]} {status}" for status in _ORDER if counts[status])
    lines = [f"Claims ({len(checks)}): {summary}"]
    problems = [check for check in checks if check.status != STATUS_SUPPORTED]
    for check in problems[:_SHOWN]:
        score = f"{check.support:.2f} " if check.support is not None else ""
        text = check.text if len(check.text) <= 100 else check.text[:97].rstrip() + "..."
        line = f'  [{check.status}] {score}"{text}"'
        if check.missing_numbers:
            line += f" (not in the cited sources: {', '.join(check.missing_numbers)})"
        lines.append(line)
    if len(problems) > _SHOWN:
        lines.append(f"  ... and {len(problems) - _SHOWN} more")
    return "\n".join(lines)


def build_report(
    answer: str,
    passages: Sequence[PassageLike],
    has_notes: bool = False,
    threshold: float = DEFAULT_THRESHOLD,
    embedder_factory: Callable[[], SupportEmbedder] = build_support_embedder,
) -> str:
    """Citation report (7.1) followed by the claim checks (7.2). The embedder is only used if a claim needs it."""
    citations = verify_citations(answer, [passage.number for passage in passages], has_notes)
    note = ""
    try:
        checks = check_claims(answer, passages, embedder_factory(), threshold)
    except EmbeddingError as exc:
        checks = check_claims(answer, passages, None, threshold)
        note = f"\n  support score unavailable: {exc}"
    return f"{format_report(citations)}\n{format_claims(checks)}{note}"
