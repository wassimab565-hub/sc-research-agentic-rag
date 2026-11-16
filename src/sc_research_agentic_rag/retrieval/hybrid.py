from collections.abc import Sequence
from dataclasses import replace
from typing import Protocol

from sc_research_agentic_rag.retrieval.retriever import RetrievedChunk
from sc_research_agentic_rag.storage.vector_store import SearchFilter

RRF_K = 60
DEFAULT_CANDIDATES = 50


class Ranker(Protocol):
    def retrieve(
        self, question: str, top_k: int = ..., filters: SearchFilter | None = ...
    ) -> list[RetrievedChunk]: ...


def rrf_fuse(rankings: Sequence[Sequence[RetrievedChunk]], k: int = RRF_K) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    chunks: dict[str, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1 / (k + rank)
            chunks.setdefault(chunk.chunk_id, chunk)
    order = sorted(scores, key=lambda chunk_id: scores[chunk_id], reverse=True)
    return [replace(chunks[chunk_id], score=scores[chunk_id]) for chunk_id in order]


def _limit_per_paper(chunks: Sequence[RetrievedChunk], limit: int | None) -> list[RetrievedChunk]:
    if limit is None:
        return list(chunks)
    counts: dict[str, int] = {}
    kept: list[RetrievedChunk] = []
    for chunk in chunks:
        if counts.get(chunk.paper_uid, 0) < limit:
            counts[chunk.paper_uid] = counts.get(chunk.paper_uid, 0) + 1
            kept.append(chunk)
    return kept


class HybridRetriever:
    def __init__(self, dense: Ranker, bm25: Ranker, candidates: int = DEFAULT_CANDIDATES) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._dense = dense
        self._bm25 = bm25
        self._candidates = candidates

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: SearchFilter | None = None,
        score_threshold: float | None = None,
        max_per_paper: int | None = None,
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if max_per_paper is not None and max_per_paper < 1:
            raise ValueError("max_per_paper must be >= 1")

        depth = max(self._candidates, top_k)
        dense_hits = self._dense.retrieve(question, top_k=depth, filters=filters)
        # The threshold is a gate on the dense (cosine) score: RRF scores are not comparable to it.
        if score_threshold is not None and max((c.score for c in dense_hits), default=0.0) < score_threshold:
            return []
        bm25_hits = self._bm25.retrieve(question, top_k=depth, filters=filters)
        fused = rrf_fuse([dense_hits, bm25_hits])
        return _limit_per_paper(fused, max_per_paper)[:top_k]
