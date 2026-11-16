import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

from sc_research_agentic_rag.retrieval.retriever import RetrievedChunk
from sc_research_agentic_rag.storage.vector_store import SearchFilter, SearchHit, VectorStore

_TOKEN = re.compile(r"[^\W_]+")


def tokenize(text: str) -> list[str]:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return _TOKEN.findall(plain)


class Bm25Index:
    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._lengths: list[int] = []
        for doc_id, text in enumerate(documents):
            counts = Counter(tokenize(text))
            self._lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self._postings.setdefault(term, []).append((doc_id, count))
        total = len(self._lengths)
        self._avg_length = (sum(self._lengths) / total) if total else 0.0
        self._idf = {
            term: math.log(1 + (total - len(docs) + 0.5) / (len(docs) + 0.5))
            for term, docs in self._postings.items()
        }

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * len(self._lengths)
        average = self._avg_length or 1.0
        for term in set(tokenize(query)):
            for doc_id, count in self._postings.get(term, ()):
                norm = 1 - self._b + self._b * self._lengths[doc_id] / average
                scores[doc_id] += self._idf[term] * count * (self._k1 + 1) / (count + self._k1 * norm)
        return scores


class Bm25Retriever:
    def __init__(self, store: VectorStore) -> None:
        self._payloads = list(store.scroll_payloads())
        self._index = Bm25Index([payload.get("text", "") for payload in self._payloads])

    def retrieve(
        self, question: str, top_k: int = 5, filters: SearchFilter | None = None
    ) -> list[RetrievedChunk]:
        if not question.strip():
            raise ValueError("question must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be >= 1")

        scores = self._index.scores(question)
        ranked = sorted(
            (
                i
                for i, score in enumerate(scores)
                if score > 0 and (filters is None or filters.accepts(self._payloads[i]))
            ),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]
        return [
            RetrievedChunk.from_hit(
                SearchHit(self._payloads[i].get("chunk_id", ""), scores[i], self._payloads[i])
            )
            for i in ranked
        ]
