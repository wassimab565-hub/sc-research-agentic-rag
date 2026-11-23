# src\sc_research_agentic_rag\verification\
import re
from collections.abc import Sequence
from dataclasses import dataclass

from sc_research_agentic_rag.rag.pipeline import extract_citations
from sc_research_agentic_rag.verification.citations import MIN_WORDS, split_sentences

_MARKERS = re.compile(r"\[[\d\s,\-\u2013]+\]")
# 14 781 255 | 1,000.5 | 66,4 | 332. Not inside words or identifiers (BM25, arxiv:2305.06983).
_NUMBER = re.compile(
    r"(?<![\w:/.])"
    r"(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+(?:[.,]\d+)?|\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:[.,]\d+)?)"
    r"(?!\w)"
)
_COMMA_GROUPS = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?")


@dataclass(frozen=True)
class Claim:
    text: str
    citations: tuple[int, ...]


def strip_citations(text: str) -> str:
    cleaned = " ".join(_MARKERS.sub("", text).split())
    return re.sub(r"\s+([.,;:!?])", r"\1", cleaned)


def extract_claims(answer: str) -> list[Claim]:
    """A claim is a sentence of the answer that carries at least one [n]."""
    claims = []
    for sentence in split_sentences(answer):
        citations = extract_citations(sentence)
        if citations and len(sentence.split()) >= MIN_WORDS:
            claims.append(Claim(sentence, tuple(citations)))
    return claims


def extract_numbers(text: str) -> list[str]:
    """Numbers of a text in one form: '14 781 255' -> '14781255', '66,4' -> '66.4', '66.40' -> '66.4'."""
    found: dict[str, None] = {}
    for match in _NUMBER.finditer(_MARKERS.sub(" ", text)):
        found.setdefault(_normalize(match.group(0)))
    return list(found)


def missing_numbers(claim_text: str, passage_texts: Sequence[str]) -> tuple[str, ...]:
    """Numbers written in the claim that none of the cited passages contains."""
    available = {number for text in passage_texts for number in extract_numbers(text)}
    return tuple(number for number in extract_numbers(claim_text) if number not in available)


def _normalize(raw: str) -> str:
    value = re.sub(r"[ \u00a0\u202f]", "", raw)
    value = value.replace(",", "") if _COMMA_GROUPS.fullmatch(value) else value.replace(",", ".")
    if "." in value:
        value = value.rstrip("0").rstrip(".")
    return value
