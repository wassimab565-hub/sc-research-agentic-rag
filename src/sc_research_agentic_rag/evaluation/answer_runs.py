# src\sc_research_agentic_rag\evaluation\
import json
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sc_research_agentic_rag.evaluation.answers import AnswerCheck, aggregate_answers, evaluate_answer
from sc_research_agentic_rag.evaluation.models import EvalQuestion
from sc_research_agentic_rag.verification import ClaimCheck, check_claims
from sc_research_agentic_rag.verification.support import (
    STATUS_NO_VALID_SOURCE,
    STATUS_NUMBER_MISSING,
    STATUS_SUPPORTED,
    STATUS_WEAK,
    SupportEmbedder,
)

AskFn = Callable[[EvalQuestion], "SavedAnswer"]


@dataclass(frozen=True)
class SavedPassage:
    number: int
    paper_uid: str
    page: int
    text: str


@dataclass(frozen=True)
class SavedAnswer:
    """What an answer run keeps: enough to verify it again later without calling the LLM."""

    question_id: str
    question: str
    answer: str
    sources: tuple[SavedPassage, ...] = ()
    cited_numbers: tuple[int, ...] = ()
    has_notes: bool = False
    stop_reason: str = ""

    @property
    def cited_sources(self) -> tuple[SavedPassage, ...]:
        return tuple(passage for passage in self.sources if passage.number in self.cited_numbers)


@dataclass(frozen=True)
class RunEvaluation:
    checks: list[AnswerCheck] = field(default_factory=list)
    claims: dict[str, list[ClaimCheck]] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


def saved_answer(question_id: str, question: str, result: Any) -> SavedAnswer:
    """From a RagAnswer or an AgentAnswer."""
    trace = getattr(result, "trace", ())
    return SavedAnswer(
        question_id=question_id,
        question=question,
        answer=result.answer,
        sources=tuple(SavedPassage(p.number, p.paper_uid, p.page, p.text) for p in result.sources),
        cited_numbers=tuple(p.number for p in result.cited_sources),
        has_notes=any(step.found and step.decision.action != "retrieve_evidence" for step in trace),
        stop_reason=getattr(result, "stop_reason", ""),
    )


def _to_line(saved: SavedAnswer) -> str:
    data = {
        "question_id": saved.question_id,
        "question": saved.question,
        "answer": saved.answer,
        "sources": [vars(passage) for passage in saved.sources],
        "cited_numbers": list(saved.cited_numbers),
        "has_notes": saved.has_notes,
        "stop_reason": saved.stop_reason,
    }
    return json.dumps(data, ensure_ascii=False)


def _from_dict(data: dict[str, Any]) -> SavedAnswer:
    return SavedAnswer(
        question_id=data["question_id"],
        question=data.get("question", ""),
        answer=data["answer"],
        sources=tuple(SavedPassage(**item) for item in data.get("sources", [])),
        cited_numbers=tuple(data.get("cited_numbers", [])),
        has_notes=bool(data.get("has_notes", False)),
        stop_reason=data.get("stop_reason", ""),
    )


def load_answers(path: Path) -> dict[str, SavedAnswer]:
    """Saved answers by question id (a later line replaces an earlier one; unreadable lines are skipped)."""
    answers: dict[str, SavedAnswer] = {}
    if not path.exists():
        return answers
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            saved = _from_dict(json.loads(line))
        except (ValueError, KeyError, TypeError):
            continue
        answers[saved.question_id] = saved
    return answers


def append_answer(path: Path, saved: SavedAnswer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_to_line(saved) + "\n")


def collect_answers(
    questions: Sequence[EvalQuestion],
    ask: AskFn,
    cache_path: Path,
    limit: int | None = None,
    on_progress: Callable[[int], None] | None = None,
) -> dict[str, SavedAnswer]:
    """Ask the questions that have no saved answer yet. Each answer is saved at once, so a rerun resumes."""
    answers = load_answers(cache_path)
    asked = 0
    for question in questions:
        if question.id in answers:
            continue
        if limit is not None and asked >= limit:
            break
        saved = ask(question)
        append_answer(cache_path, saved)
        answers[question.id] = saved
        asked += 1
        if on_progress:
            on_progress(asked)
    return answers


def aggregate_claims(claims: Sequence[ClaimCheck]) -> dict[str, float]:
    if not claims:
        return {}
    counts = Counter(claim.status for claim in claims)
    names = {
        STATUS_SUPPORTED: "supported_claim_rate",
        STATUS_WEAK: "weak_claim_rate",
        STATUS_NUMBER_MISSING: "number_missing_claim_rate",
        STATUS_NO_VALID_SOURCE: "no_valid_source_claim_rate",
    }
    return {name: counts[status] / len(claims) for status, name in names.items()}


def support_curve(claims: Sequence[ClaimCheck], thresholds: Sequence[float]) -> dict[float, float]:
    """Share of the scored claims (numbers found, valid source) whose support reaches each threshold."""
    scores = [c.support for c in claims if c.support is not None and c.status in (STATUS_SUPPORTED, STATUS_WEAK)]
    if not scores:
        return {}
    return {t: sum(score >= t for score in scores) / len(scores) for t in thresholds}


def evaluate_run(
    questions: Sequence[EvalQuestion],
    answers: dict[str, SavedAnswer],
    embedder: SupportEmbedder | None,
    support_threshold: float,
) -> RunEvaluation:
    checks: list[AnswerCheck] = []
    claims: dict[str, list[ClaimCheck]] = {}
    for question in questions:
        saved = answers.get(question.id)
        if saved is None:
            continue
        checks.append(evaluate_answer(question, saved, saved.has_notes))
        claims[question.id] = check_claims(saved.answer, saved.sources, embedder, support_threshold)
    metrics = aggregate_answers(checks)
    metrics.update(aggregate_claims([claim for items in claims.values() for claim in items]))
    return RunEvaluation(checks, claims, metrics)
