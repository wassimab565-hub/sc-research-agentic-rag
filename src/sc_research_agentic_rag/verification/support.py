# src\sc_research_agentic_rag\verification\
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sc_research_agentic_rag.verification.citations import split_sentences
from sc_research_agentic_rag.verification.claims import (
    extract_claims,
    extract_numbers,
    missing_numbers,
    strip_citations,
)

# Provisional: not calibrated yet (the same warning as RAG_SCORE_THRESHOLD, I-4). Calibrate in 7.3.
DEFAULT_THRESHOLD = 0.5

STATUS_SUPPORTED = "supported"
STATUS_WEAK = "weak"
STATUS_NUMBER_MISSING = "number_missing"
STATUS_MISATTRIBUTED = "misattributed"
STATUS_NO_VALID_SOURCE = "no_valid_source"
STATUS_UNCHECKED = "unchecked"


class SupportEmbedder(Protocol):
    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class PassageLike(Protocol):
    number: int
    text: str


@dataclass(frozen=True)
class ClaimCheck:
    """status: supported | weak | misattributed | number_missing | no_valid_source | unchecked (no embedder).

    misattributed: the numbers are not in the cited sources but are in other sources (found_in).
    """

    text: str
    citations: tuple[int, ...]
    status: str
    support: float | None = None
    missing_numbers: tuple[str, ...] = ()
    found_in: tuple[int, ...] = ()


def passage_windows(text: str) -> list[str]:
    """Each sentence of a passage, and each pair of neighbouring sentences."""
    sentences = split_sentences(" ".join(text.split()))
    pairs = [f"{a} {b}" for a, b in zip(sentences, sentences[1:])]
    return [*sentences, *pairs]


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return sum(x * y for x, y in zip(a, b)) / norm if norm else 0.0


def check_claims(
    answer: str,
    passages: Sequence[PassageLike],
    embedder: SupportEmbedder | None,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[ClaimCheck]:
    """Number check (exact) and support score (bge-m3 cosine, a weak signal) for each claim of the answer.

    Without an embedder only the number check runs and the other claims stay `unchecked`.
    """
    claims = extract_claims(answer)
    by_number = {passage.number: passage for passage in passages}
    cited = [[by_number[n] for n in claim.citations if n in by_number] for claim in claims]
    scores = _support_scores([strip_citations(c.text) for c in claims], cited, embedder) if embedder else {}

    checks = []
    for index, claim in enumerate(claims):
        if not cited[index]:
            checks.append(ClaimCheck(claim.text, claim.citations, STATUS_NO_VALID_SOURCE))
            continue
        support = scores.get(index)
        missing = missing_numbers(claim.text, [p.text for p in cited[index]])
        if missing:
            found_in = _found_elsewhere(missing, passages, claim.citations)
            status = STATUS_MISATTRIBUTED if found_in else STATUS_NUMBER_MISSING
            checks.append(ClaimCheck(claim.text, claim.citations, status, support, missing, found_in))
            continue
        if support is None:
            status = STATUS_UNCHECKED
        else:
            status = STATUS_SUPPORTED if support >= threshold else STATUS_WEAK
        checks.append(ClaimCheck(claim.text, claim.citations, status, support, missing))
    return checks

def _found_elsewhere(
    missing: Sequence[str], passages: Sequence[PassageLike], cited: Sequence[int]
) -> tuple[int, ...]:
    """Numbers of the other sources that contain the missing numbers; empty unless ALL of them are found there."""
    others = {p.number: set(extract_numbers(p.text)) for p in passages if p.number not in cited}
    if not set(missing) <= set().union(*others.values()):
        return ()
    return tuple(number for number, numbers in others.items() if numbers & set(missing))

def _support_scores(
    texts: Sequence[str], cited: Sequence[Sequence[PassageLike]], embedder: SupportEmbedder
) -> dict[int, float]:
    """Best cosine between a claim and any window of the passages it cites. One batch for claims, one for windows."""
    scorable = [index for index, passages in enumerate(cited) if passages]
    if not scorable:
        return {}
    windows: dict[int, list[str]] = {}
    for index in scorable:
        for passage in cited[index]:
            windows.setdefault(passage.number, passage_windows(passage.text))

    flat = [window for number in windows for window in windows[number]]
    claim_vectors = embedder.embed_passages([texts[index] for index in scorable])
    window_vectors = embedder.embed_passages(flat) if flat else []

    vectors: dict[int, list[list[float]]] = {}
    start = 0
    for number, items in windows.items():
        vectors[number] = window_vectors[start : start + len(items)]
        start += len(items)

    scores = {}
    for index, claim_vector in zip(scorable, claim_vectors):
        candidates = [cosine(claim_vector, v) for passage in cited[index] for v in vectors[passage.number]]
        if candidates:
            scores[index] = max(candidates)
    return scores
