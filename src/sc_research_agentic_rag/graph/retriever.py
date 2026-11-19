import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from sc_research_agentic_rag.retrieval.bm25 import tokenize
from sc_research_agentic_rag.retrieval.retriever import RetrievedChunk
from sc_research_agentic_rag.storage.vector_store import SearchFilter, SearchHit

CANDIDATE_FACTOR = 5
MIN_CANDIDATES = 50


class ConceptSource(Protocol):
    def concept_keys(self) -> list[str]: ...
    def chunk_count(self) -> int: ...
    def concept_chunks(self, keys: Sequence[str]) -> list[tuple[str, str]]: ...


class PayloadSource(Protocol):
    def get_payloads(self, chunk_ids: Sequence[str]) -> dict[str, dict[str, Any]]: ...


def build_index(keys: Iterable[str]) -> dict[tuple[str, ...], list[str]]:
    """Concept keys indexed by their tokens, so the question and the concepts are compared the same way."""
    index: dict[tuple[str, ...], list[str]] = {}
    for key in keys:
        tokens = tuple(tokenize(key))
        if tokens:
            index.setdefault(tokens, []).append(key)
    return index


def match_concepts(question: str, index: dict[tuple[str, ...], list[str]]) -> list[str]:
    tokens = tokenize(question)
    longest = max((len(key) for key in index), default=0)
    found: dict[str, None] = {}
    for start in range(len(tokens)):
        for size in range(1, min(longest, len(tokens) - start) + 1):
            for key in index.get(tuple(tokens[start : start + size]), ()):
                found.setdefault(key)
    return list(found)


def idf(total: int, frequency: int) -> float:
    return math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))


class GraphRetriever:
    """Finds the concepts named in the question and returns the chunks that mention them, rare concepts first."""

    def __init__(self, graph: ConceptSource, store: PayloadSource) -> None:
        self._graph = graph
        self._store = store
        self._index = build_index(graph.concept_keys())
        self._total = graph.chunk_count()

    def retrieve(
        self, question: str, top_k: int = 5, filters: SearchFilter | None = None
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        keys = match_concepts(question, self._index)
        if not keys:
            return []
        pairs = self._graph.concept_chunks(keys)
        frequency = Counter(key for key, _ in pairs)
        scores: dict[str, float] = defaultdict(float)
        for key, chunk_id in pairs:
            scores[chunk_id] += idf(self._total, frequency[key])

        ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        payloads = self._store.get_payloads(ranked[: max(top_k * CANDIDATE_FACTOR, MIN_CANDIDATES)])
        chunks: list[RetrievedChunk] = []
        for chunk_id in ranked:
            payload = payloads.get(chunk_id)
            if payload is None or (filters is not None and not filters.accepts(payload)):
                continue
            chunks.append(RetrievedChunk.from_hit(SearchHit(chunk_id, scores[chunk_id], payload)))
            if len(chunks) == top_k:
                break
        return chunks