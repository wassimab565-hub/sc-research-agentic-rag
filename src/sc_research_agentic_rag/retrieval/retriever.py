from dataclasses import dataclass, field
from typing import Any, Protocol

from sc_research_agentic_rag.storage.vector_store import SearchFilter, SearchHit, VectorStore


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    score: float
    text: str
    paper_uid: str
    page: int
    char_start: int
    char_end: int
    title: str
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    url: str | None = None
    source: str = ""

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "RetrievedChunk":
        payload: dict[str, Any] = hit.payload
        paper_uid = payload.get("paper_uid", "")
        return cls(
            chunk_id=hit.chunk_id,
            score=hit.score,
            text=payload.get("text", ""),
            paper_uid=paper_uid,
            page=payload.get("page", 0),
            char_start=payload.get("char_start", 0),
            char_end=payload.get("char_end", 0),
            title=payload.get("title") or paper_uid,
            authors=list(payload.get("authors") or []),
            year=payload.get("year"),
            doi=payload.get("doi"),
            url=payload.get("url"),
            source=payload.get("source", ""),
        )


class Retriever:
    def __init__(self, embedder: QueryEmbedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
        filters: SearchFilter | None = None,
        score_threshold: float | None = None,
        max_per_paper: int | None = None,
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if max_per_paper is not None and max_per_paper < 1:
            raise ValueError("max_per_paper must be >= 1")

        hits = self._store.search(
            self._embedder.embed_query(question),
            limit=top_k,
            filters=filters,
            score_threshold=score_threshold,
            max_per_paper=max_per_paper,
        )
        return [RetrievedChunk.from_hit(hit) for hit in hits]