# src\sc_research_agentic_rag\verification\
import re
from collections.abc import Collection
from dataclasses import dataclass

from sc_research_agentic_rag.rag.pipeline import extract_citations

MIN_WORDS = 4
SHOWN_UNCITED = 3

STATUS_OK = "ok"
STATUS_NO_SOURCES = "no_sources"
STATUS_NO_CITATION = "no_citation"
STATUS_INVALID = "invalid"
STATUS_ON_NOTES = "citation_on_notes"

_DOT = "\x00"
_MARK = r"\[[\d\s,\-\u2013]+\]"
_ABBREVIATIONS = re.compile(r"\b(e\.g|i\.e|et al|fig|vs|cf)\.", re.IGNORECASE)
_MARK_AFTER_STOP = re.compile(rf"([.!?])\s+({_MARK}(?:\s*{_MARK})*)(?=\s|$)")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s+")


@dataclass(frozen=True)
class CitationReport:
    """status: ok | no_sources | no_citation | invalid | citation_on_notes."""

    status: str
    valid: tuple[int, ...] = ()
    invalid: tuple[int, ...] = ()
    sentences: int = 0
    uncited_sentences: tuple[str, ...] = ()


def split_sentences(text: str) -> list[str]:
    """Plain sentence split (no LLM). A [n] written after the full stop goes back to its sentence."""
    protected = _ABBREVIATIONS.sub(lambda match: match.group(0).replace(".", _DOT), text)
    protected = _MARK_AFTER_STOP.sub(r" \2\1", protected)
    sentences: list[str] = []
    for line in protected.splitlines():
        line = _BULLET.sub("", line).strip()
        for piece in _SENTENCE_END.split(line):
            piece = piece.strip().replace(_DOT, ".")
            if piece:
                sentences.append(piece)
    return sentences


def verify_citations(answer: str, source_numbers: Collection[int], has_notes: bool = False) -> CitationReport:
    """Checks the [n] of an answer against the numbered sources that were given to the LLM.

    A [n] written on database notes (graph, paper search) cannot be seen when numbered sources exist too:
    it is only detected when there is no numbered source at all (status citation_on_notes).
    """
    known = set(source_numbers)
    cited = extract_citations(answer)
    valid = tuple(n for n in cited if n in known)
    invalid = tuple(n for n in cited if n not in known)

    sentences = [s for s in split_sentences(answer) if len(s.split()) >= MIN_WORDS] if known else []
    uncited = tuple(s for s in sentences if not extract_citations(s))

    if invalid:
        status = STATUS_ON_NOTES if has_notes and not known else STATUS_INVALID
    elif not known:
        status = STATUS_NO_SOURCES
    elif not valid:
        status = STATUS_NO_CITATION
    else:
        status = STATUS_OK
    return CitationReport(status, valid, invalid, len(sentences), uncited)


def format_report(report: CitationReport) -> str:
    def numbers(values: tuple[int, ...]) -> str:
        return "".join(f"[{n}]" for n in values) or "none"

    lines = [
        f"Verification: {report.status}",
        f"  valid citations: {numbers(report.valid)}; invalid: {numbers(report.invalid)}",
        f"  uncited sentences: {len(report.uncited_sentences)}/{report.sentences}",
    ]
    for sentence in report.uncited_sentences[:SHOWN_UNCITED]:
        short = sentence if len(sentence) <= 100 else sentence[:97].rstrip() + "..."
        lines.append(f"    - {short}")
    return "\n".join(lines)
