import re
import unicodedata
from collections import Counter

from sc_research_agentic_rag.document_processing.models import Page

_MIN_PAGES_FOR_EDGE_REMOVAL = 3
_EDGE_LINES = 2
_MIN_LINES_FOR_EDGES = 6
_EDGE_MIN_RATIO = 0.4

_NORMALIZE = str.maketrans(
    {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        "\ufb05": "st",
        "\ufb06": "st",
        "\u00a0": " ",
        "\u2009": " ",
        "\u202f": " ",
        "\u00ad": None,
        "\u200b": None,
        "\u200c": None,
        "\u200d": None,
        "\ufeff": None,
        "\x00": None,
    }
)
_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])-[ \t]*\n[ \t]*(?=[a-zà-öø-ÿ])")


def clean_pages(pages: list[Page]) -> list[Page]:
    texts = [_tidy(_normalize_chars(page.text)) for page in pages]
    texts = _remove_repeated_edges(texts)
    texts = [_tidy(_HYPHEN_BREAK.sub("", text)) for text in texts]
    return [
        Page(paper_uid=page.paper_uid, page=page.page, text=text)
        for page, text in zip(pages, texts, strict=True)
    ]


def _normalize_chars(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\u00ad\n", "-\n")
    return text.translate(_NORMALIZE)


def _tidy(text: str) -> str:
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _edge_key(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _edge_indexes(lines: list[str]) -> set[int]:
    non_empty = [index for index, line in enumerate(lines) if line.strip()]
    if len(non_empty) < _MIN_LINES_FOR_EDGES:
        return set()
    return set(non_empty[:_EDGE_LINES] + non_empty[-_EDGE_LINES:])


def _remove_repeated_edges(texts: list[str]) -> list[str]:
    if len(texts) < _MIN_PAGES_FOR_EDGE_REMOVAL:
        return texts

    split = [text.splitlines() for text in texts]
    counts: Counter[str] = Counter()
    for lines in split:
        counts.update({_edge_key(lines[i]) for i in _edge_indexes(lines)})

    threshold = max(_MIN_PAGES_FOR_EDGE_REMOVAL, _EDGE_MIN_RATIO * len(texts))
    repeated = {key for key, count in counts.items() if count >= threshold}

    cleaned = []
    for lines in split:
        drop = {i for i in _edge_indexes(lines) if _edge_key(lines[i]) in repeated}
        cleaned.append("\n".join(line for i, line in enumerate(lines) if i not in drop))
    return cleaned