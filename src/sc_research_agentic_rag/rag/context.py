from collections.abc import Sequence
from dataclasses import dataclass, field

from sc_research_agentic_rag.retrieval import RetrievedChunk

DEFAULT_MAX_CONTEXT_CHARS = 12000


@dataclass(frozen=True)
class SourcePassage:
    number: int
    paper_uid: str
    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str | None
    url: str | None
    source: str
    page: int
    text: str
    chunk_ids: tuple[str, ...]
    score: float


@dataclass
class _Span:
    first: RetrievedChunk
    text: str
    char_end: int
    chunk_ids: list[str] = field(default_factory=list)
    score: float = 0.0


def build_passages(
    chunks: Sequence[RetrievedChunk], max_chars: int = DEFAULT_MAX_CONTEXT_CHARS
) -> list[SourcePassage]:
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")

    selected: list[_Span] = []
    used = 0
    for span in sorted(_merge(chunks), key=lambda s: s.score, reverse=True):
        if not selected and len(span.text) > max_chars:
            span.text = span.text[:max_chars].rstrip()
        elif used + len(span.text) > max_chars:
            continue
        selected.append(span)
        used += len(span.text)

    return [
        SourcePassage(
            number=number,
            paper_uid=span.first.paper_uid,
            title=span.first.title,
            authors=tuple(span.first.authors),
            year=span.first.year,
            doi=span.first.doi,
            url=span.first.url,
            source=span.first.source,
            page=span.first.page,
            text=span.text,
            chunk_ids=tuple(span.chunk_ids),
            score=span.score,
        )
        for number, span in enumerate(selected, start=1)
    ]


def _merge(chunks: Sequence[RetrievedChunk]) -> list[_Span]:
    groups: dict[tuple[str, int], list[RetrievedChunk]] = {}
    for chunk in chunks:
        groups.setdefault((chunk.paper_uid, chunk.page), []).append(chunk)

    spans: list[_Span] = []
    for group in groups.values():
        current: _Span | None = None
        for chunk in sorted(group, key=lambda c: (c.char_start, c.char_end)):
            if current is not None and _overlaps(current, chunk):
                if chunk.char_end > current.char_end:
                    current.text += chunk.text[current.char_end - chunk.char_start :]
                    current.char_end = chunk.char_end
                current.chunk_ids.append(chunk.chunk_id)
                current.score = max(current.score, chunk.score)
            else:
                current = _Span(chunk, chunk.text, chunk.char_end, [chunk.chunk_id], chunk.score)
                spans.append(current)
    return spans


def _overlaps(span: _Span, chunk: RetrievedChunk) -> bool:
    consistent = len(chunk.text) == chunk.char_end - chunk.char_start and len(span.text) == (
        span.char_end - span.first.char_start
    )
    return consistent and chunk.char_start < span.char_end