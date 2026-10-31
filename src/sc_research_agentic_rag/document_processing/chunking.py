from collections import defaultdict
from collections.abc import Iterator

from sc_research_agentic_rag.document_processing.models import Chunk, Page

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 150
_SENTENCE_ENDS = (". ", ".\n", "? ", "?\n", "! ", "!\n")


def chunk_pages(
    pages: list[Page],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[Chunk]:
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    next_index: dict[str, int] = defaultdict(int)
    chunks: list[Chunk] = []
    for page in pages:
        for start, end in _spans(page.text, chunk_size, overlap):
            index = next_index[page.paper_uid]
            next_index[page.paper_uid] += 1
            chunks.append(
                Chunk(
                    paper_uid=page.paper_uid,
                    index=index,
                    page=page.page,
                    text=page.text[start:end],
                    char_start=start,
                    char_end=end,
                )
            )
    return chunks


def _spans(text: str, size: int, overlap: int) -> Iterator[tuple[int, int]]:
    length = len(text)
    position = _skip_whitespace(text, 0)
    while position < length:
        limit = min(position + size, length)
        end = limit if limit == length else _break_point(text, position, limit)

        trimmed_end = end
        while trimmed_end > position and text[trimmed_end - 1].isspace():
            trimmed_end -= 1
        yield position, trimmed_end

        if end >= length:
            return
        candidate = _word_start(text, max(end - overlap, position + 1))
        position = _skip_whitespace(text, candidate if candidate <= end else end)


def _break_point(text: str, start: int, limit: int) -> int:
    floor = start + (limit - start) // 2
    index = text.rfind("\n\n", floor, limit)
    if index != -1:
        return index + 2
    sentence = max(text.rfind(marker, floor, limit) for marker in _SENTENCE_ENDS)
    if sentence != -1:
        return sentence + 2
    for separator in ("\n", " "):
        index = text.rfind(separator, floor, limit)
        if index != -1:
            return index + 1
    return limit


def _word_start(text: str, position: int) -> int:
    while 0 < position < len(text) and not text[position - 1].isspace() and not text[position].isspace():
        position += 1
    return position


def _skip_whitespace(text: str, position: int) -> int:
    while position < len(text) and text[position].isspace():
        position += 1
    return position