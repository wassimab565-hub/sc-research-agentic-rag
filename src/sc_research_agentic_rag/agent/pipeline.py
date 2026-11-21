# src\sc_research_agentic_rag\agent\
from dataclasses import dataclass, field

from sc_research_agentic_rag.agent.decisions import MAX_ARG_CHARS, Decision, Step, parse_decision
from sc_research_agentic_rag.agent.prompts import RETRY_PROMPT, build_answer_messages, build_decision_messages
from sc_research_agentic_rag.agent.tools import AgentTools
from sc_research_agentic_rag.llm import ChatMessage, LLMClient
from sc_research_agentic_rag.rag import (
    DEFAULT_MAX_CONTEXT_CHARS,
    NO_SOURCES_MESSAGE,
    RagAnswer,
    SourcePassage,
    build_passages,
    extract_citations,
)
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter

STOP_ANSWERED = "answered"
STOP_MAX_STEPS = "max_steps"
STOP_REPEATED = "repeated_action"
STOP_RETRIEVAL_LIMIT = "retrieval_limit"
STOP_EMPTY = "empty_results"
STOP_INVALID = "invalid_decision"

MAX_EMPTY_STREAK = 2


@dataclass(frozen=True)
class AgentAnswer:
    question: str
    answer: str
    sources: list[SourcePassage] = field(default_factory=list)
    cited_sources: list[SourcePassage] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)
    trace: list[Step] = field(default_factory=list)
    stop_reason: str = ""

    @property
    def has_citations(self) -> bool:
        return bool(self.cited_sources)

    def as_rag_answer(self) -> RagAnswer:
        return RagAnswer(
            question=self.question,
            answer=self.answer,
            sources=self.sources,
            cited_sources=self.cited_sources,
            invalid_citations=self.invalid_citations,
        )


class AgentPipeline:
    """A bounded loop: the LLM picks one action per step, the tools run it, then one grounded answer is written."""

    def __init__(
        self,
        llm: LLMClient,
        tools: AgentTools,
        max_steps: int = 4,
        max_retrievals: int = 3,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if max_retrievals < 1:
            raise ValueError("max_retrievals must be >= 1")
        self._llm = llm
        self._tools = tools
        self._max_steps = max_steps
        self._max_retrievals = max_retrievals
        self._max_context_chars = max_context_chars

    def ask(self, question: str, filters: SearchFilter | None = None) -> AgentAnswer:
        if not question.strip():
            raise ValueError("question must not be empty")

        steps: list[Step] = []
        chunks: dict[str, RetrievedChunk] = {}
        notes: list[str] = []
        seen: set[str] = set()
        retrievals = empty_streak = 0
        stop = STOP_MAX_STEPS

        while len(steps) < self._max_steps:
            decision, valid = self._decide(question, steps)
            if decision.action == "answer":
                stop = STOP_ANSWERED if valid else STOP_INVALID
                break
            if decision.key in seen:
                stop = STOP_REPEATED
                break
            if decision.action == "retrieve_evidence" and retrievals >= self._max_retrievals:
                stop = STOP_RETRIEVAL_LIMIT
                break

            seen.add(decision.key)
            result = self._tools.run(decision, filters)
            steps.append(Step(len(steps) + 1, decision, result.summary, result.found))
            if decision.action == "retrieve_evidence":
                retrievals += 1
            for chunk in result.chunks:
                known = chunks.get(chunk.chunk_id)
                if known is None or chunk.score > known.score:
                    chunks[chunk.chunk_id] = chunk
            if result.note and result.found:
                notes.append(result.note)

            empty_streak = 0 if result.found else empty_streak + 1
            if empty_streak >= MAX_EMPTY_STREAK:
                stop = STOP_EMPTY
                break

        return self._finish(question, steps, list(chunks.values()), notes, stop)

    def _decide(self, question: str, steps: list[Step]) -> tuple[Decision, bool]:
        """The next decision and whether it came from the LLM (False = the fallback was used)."""
        messages = build_decision_messages(question, steps, self._max_steps - len(steps))
        raw = self._llm.chat(messages)
        decision = parse_decision(raw)
        if decision is None:
            retry = [*messages, ChatMessage("assistant", raw), ChatMessage("user", RETRY_PROMPT)]
            decision = parse_decision(self._llm.chat(retry))
        if decision is not None:
            return decision, True
        return _fallback(question, steps), False

    def _finish(
        self, question: str, steps: list[Step], chunks: list[RetrievedChunk], notes: list[str], stop: str
    ) -> AgentAnswer:
        passages = build_passages(chunks, self._max_context_chars)
        if not passages and not notes:
            return AgentAnswer(question=question, answer=NO_SOURCES_MESSAGE, trace=steps, stop_reason=stop)

        answer = self._llm.chat(build_answer_messages(question, passages, notes))
        by_number = {passage.number: passage for passage in passages}
        numbers = extract_citations(answer)
        return AgentAnswer(
            question=question,
            answer=answer,
            sources=passages,
            cited_sources=[by_number[n] for n in numbers if n in by_number],
            invalid_citations=[n for n in numbers if n not in by_number],
            trace=steps,
            stop_reason=stop,
        )


def _fallback(question: str, steps: list[Step]) -> Decision:
    """The LLM gave no usable decision twice: plain RAG on the question first, then stop."""
    if any(step.decision.action == "retrieve_evidence" for step in steps):
        return Decision("answer")
    return Decision("retrieve_evidence", {"query": " ".join(question.split())[:MAX_ARG_CHARS]})
