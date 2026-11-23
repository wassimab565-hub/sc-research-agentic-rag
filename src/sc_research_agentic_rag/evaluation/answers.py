# src\sc_research_agentic_rag\evaluation\
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from sc_research_agentic_rag.evaluation.metrics import normalize
from sc_research_agentic_rag.evaluation.models import EvalQuestion, Evidence
from sc_research_agentic_rag.verification import verify_citations


class PassageLike(Protocol):
    number: int
    paper_uid: str
    page: int
    text: str


class AnswerLike(Protocol):
    """RagAnswer and AgentAnswer both fit."""

    answer: str
    sources: Sequence[PassageLike]
    cited_sources: Sequence[PassageLike]


@dataclass(frozen=True)
class AnswerCheck:
    question_id: str
    status: str
    has_sources: bool
    valid: int = 0
    invalid: int = 0
    sentences: int = 0
    uncited: int = 0
    gold_retrieved: bool | None = None
    gold_cited: bool | None = None


def evidence_in_passage(evidence: Evidence, passage: PassageLike) -> bool:
    """Same rules as `matches`, but passages have no char offsets: only paper, page and `contains`."""
    if passage.paper_uid != evidence.paper_uid:
        return False
    if evidence.page is not None and passage.page != evidence.page:
        return False
    return evidence.contains is None or normalize(evidence.contains) in normalize(passage.text)


def _hits(evidence: Sequence[Evidence], passages: Sequence[PassageLike]) -> bool:
    return any(evidence_in_passage(e, p) for e in evidence for p in passages)


def evaluate_answer(question: EvalQuestion, answer: AnswerLike, has_notes: bool = False) -> AnswerCheck:
    report = verify_citations(answer.answer, [passage.number for passage in answer.sources], has_notes)
    gold_retrieved = gold_cited = None
    if all(e.char_start is None for e in question.evidence):  # offsets cannot be compared with passages
        gold_retrieved = _hits(question.evidence, answer.sources)
        gold_cited = _hits(question.evidence, answer.cited_sources)
    return AnswerCheck(
        question_id=question.id,
        status=report.status,
        has_sources=bool(answer.sources),
        valid=len(report.valid),
        invalid=len(report.invalid),
        sentences=report.sentences,
        uncited=len(report.uncited_sentences),
        gold_retrieved=gold_retrieved,
        gold_cited=gold_cited,
    )


def aggregate_answers(checks: Sequence[AnswerCheck]) -> dict[str, float]:
    if not checks:
        return {}
    sourced = [c for c in checks if c.has_sources]
    metrics = {"no_sources_rate": (len(checks) - len(sourced)) / len(checks)}
    if sourced:
        metrics["valid_citation_rate"] = sum(c.invalid == 0 for c in sourced) / len(sourced)
        metrics["cited_rate"] = sum(c.valid > 0 for c in sourced) / len(sourced)
        sentences = sum(c.sentences for c in sourced)
        if sentences:
            metrics["uncited_sentence_rate"] = sum(c.uncited for c in sourced) / sentences
    retrieved = [c.gold_retrieved for c in checks if c.gold_retrieved is not None]
    if retrieved:
        metrics["gold_retrieved_rate"] = sum(retrieved) / len(retrieved)
    cited = [c.gold_cited for c in checks if c.gold_cited is not None]
    if cited:
        metrics["gold_cited_rate"] = sum(cited) / len(cited)
    return metrics
