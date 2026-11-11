import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sc_research_agentic_rag.evaluation.models import EvalQuestion, Evidence


class ChunkLike(Protocol):
    paper_uid: str
    page: int
    text: str
    char_start: int
    char_end: int


@dataclass(frozen=True)
class QuestionResult:
    question_id: str
    n_evidence: int
    first_rank: int | None
    matched_ranks: list[int] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)


def normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def matches(evidence: Evidence, chunk: ChunkLike) -> bool:
    if chunk.paper_uid != evidence.paper_uid:
        return False
    if evidence.page is not None and chunk.page != evidence.page:
        return False
    if evidence.char_start is not None and not (
        chunk.char_start < evidence.char_end and evidence.char_start < chunk.char_end
    ):
        return False
    if evidence.contains is not None and normalize(evidence.contains) not in normalize(chunk.text):
        return False
    return True


def evaluate_question(
    question: EvalQuestion, retrieved: Sequence[ChunkLike], chunk_ids: Sequence[str]
) -> QuestionResult:
    unmatched = list(question.evidence)
    first_rank: int | None = None
    matched_ranks: list[int] = []
    for rank, chunk in enumerate(retrieved, start=1):
        if first_rank is None and any(matches(e, chunk) for e in question.evidence):
            first_rank = rank
        newly = [e for e in unmatched if matches(e, chunk)]
        if newly:
            matched_ranks.append(rank)
            unmatched = [e for e in unmatched if e not in newly]
    return QuestionResult(
        question_id=question.id,
        n_evidence=len(question.evidence),
        first_rank=first_rank,
        matched_ranks=matched_ranks,
        retrieved_ids=list(chunk_ids),
    )


def _recall(result: QuestionResult, k: int) -> float:
    found = sum(1 for rank in result.matched_ranks if rank <= k)
    return min(found, result.n_evidence) / result.n_evidence


def _ndcg(result: QuestionResult, k: int) -> float:
    dcg = sum(1 / math.log2(rank + 1) for rank in result.matched_ranks if rank <= k)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, result.n_evidence) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate(results: Sequence[QuestionResult], ks: Sequence[int]) -> dict[str, float]:
    if not results:
        return {}
    count = len(results)
    metrics = {f"recall@{k}": sum(_recall(r, k) for r in results) / count for k in ks}
    metrics["mrr"] = sum(1 / r.first_rank if r.first_rank else 0.0 for r in results) / count
    metrics.update({f"ndcg@{k}": sum(_ndcg(r, k) for r in results) / count for k in ks})
    return metrics