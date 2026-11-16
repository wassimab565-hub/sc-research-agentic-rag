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


class HybridRetriever:
    def __init__(self, dense: Ranker, bm25: Ranker, candidates: int = DEFAULT_CANDIDATES) -> None:
        if candidates < 1:
            raise ValueError("candidates must be >= 1")
        self._dense = dense
        self._bm25 = bm25
        self._candidates = candidates

    def retrieve(
        self, question: str, top_k: int = 5, filters: SearchFilter | None = None
    ) -> list[RetrievedChunk]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        depth = max(self._candidates, top_k)
        rankings = [
            self._dense.retrieve(question, top_k=depth, filters=filters),
            self._bm25.retrieve(question, top_k=depth, filters=filters),
        ]
        return rrf_fuse(rankings)[:top_k]
