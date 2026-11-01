import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

from sc_research_agentic_rag.llm import LLMClient
from sc_research_agentic_rag.rag.context import DEFAULT_MAX_CONTEXT_CHARS, SourcePassage, build_passages
from sc_research_agentic_rag.rag.prompts import build_messages
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter

NO_SOURCES_MESSAGE = "No relevant sources were found in the indexed documents."

_CITATION = re.compile(r"\[\s*(\d+(?:\s*[-\u2013]\s*\d+)?(?:\s*,\s*\d+(?:\s*[-\u2013]\s*\d+)?)*)\s*\]")
_MAX_CITATION_NUMBER = 99


class ChunkRetriever(Protocol):
    def retrieve(
        self,
        question: str,
        top_k: int = ...,
        filters: SearchFilter | None = ...,
        score_threshold: float | None = ...,
        max_per_paper: int | None = ...,
    ) -> list[RetrievedChunk]: ...


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    sources: list[SourcePassage] = field(default_factory=list)
    cited_sources: list[SourcePassage] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_sources)


def extract_citations(text: str) -> list[int]:
    numbers: set[int] = set()
    for match in _CITATION.finditer(text):
        for part in match.group(1).split(","):
            bounds = [int(value) for value in re.split(r"[-\u2013]", part)]
            low, high = bounds[0], bounds[-1]
            if high < low or high > _MAX_CITATION_NUMBER:
                continue
            numbers.update(range(low, high + 1))
    return sorted(numbers)


class RagPipeline:
    def __init__(
        self,
        retriever: ChunkRetriever,
        llm: LLMClient,
        top_k: int = 5,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
        score_threshold: float | None = None,
        max_per_paper: int | None = None,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        self._score_threshold = score_threshold
        self._max_per_paper = max_per_paper

    def ask(self, question: str, filters: SearchFilter | None = None) -> RagAnswer:
        chunks = self._retriever.retrieve(
            question,
            top_k=self._top_k,
            filters=filters,
            score_threshold=self._score_threshold,
            max_per_paper=self._max_per_paper,
        )
        passages = build_passages(chunks, self._max_context_chars)
        if not passages:
            return RagAnswer(question=question, answer=NO_SOURCES_MESSAGE)

        answer = self._llm.chat(build_messages(question, passages))
        return _with_citations(question, answer, passages)


def _with_citations(question: str, answer: str, passages: Sequence[SourcePassage]) -> RagAnswer:
    by_number = {passage.number: passage for passage in passages}
    cited_numbers = extract_citations(answer)
    return RagAnswer(
        question=question,
        answer=answer,
        sources=list(passages),
        cited_sources=[by_number[n] for n in cited_numbers if n in by_number],
        invalid_citations=[n for n in cited_numbers if n not in by_number],
    )