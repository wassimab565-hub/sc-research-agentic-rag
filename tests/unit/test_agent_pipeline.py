import json

import pytest

from sc_research_agentic_rag.agent import AgentPipeline, AgentTools
from sc_research_agentic_rag.graph.queries import ConceptPaper
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.rag import NO_SOURCES_MESSAGE
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter


def chunk(n: int) -> RetrievedChunk:
    text = f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"p#{n}", score=1.0 - n / 10, text=text, paper_uid=f"p{n}", page=1,
        char_start=0, char_end=len(text), title=f"Paper {n}",
    )


def act(action: str, **args) -> str:
    return json.dumps({"action": action, "args": args})


class ScriptedLLM:
    """Returns the scripted replies in order (a reply that is an exception is raised)."""

    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls: list[list] = []

    def chat(self, messages):
        self.calls.append(list(messages))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeRetriever:
    def __init__(self, by_query: dict[str, list[RetrievedChunk]] | None = None) -> None:
        self.by_query = by_query or {}
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        self.calls.append({"query": question, "filters": filters})
        return self.by_query.get(question, [])


class FakeGraph:
    def papers_for_concept(self, name, limit=10):
        return [ConceptPaper("arxiv:1", "Paper One", 2020, (), (3,), 2)]

    def similar_concepts(self, name, limit=5):
        return []


def make(llm, by_query=None, graph=None, **kwargs):
    retriever = FakeRetriever(by_query)
    return AgentPipeline(llm, AgentTools(retriever, graph), **kwargs), retriever


def test_evidence_then_answer_keeps_citations_and_trace() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="dpr"), act("answer"), "Because of source one [1].")
    pipeline, retriever = make(llm, {"dpr": [chunk(1), chunk(2)]})
    result = pipeline.ask("What is DPR?")

    assert result.answer == "Because of source one [1]."
    assert [s.number for s in result.sources] == [1, 2]
    assert [s.title for s in result.cited_sources] == ["Paper 1"]
    assert result.invalid_citations == []
    assert result.has_citations
    assert result.stop_reason == "answered"
    assert [(s.number, s.decision.action, s.found) for s in result.trace] == [(1, "retrieve_evidence", 2)]
    assert retriever.calls[0]["query"] == "dpr"
    assert '1. retrieve_evidence {"query": "dpr"} -> 2 passages' in llm.calls[1][1].content
    answer_prompt = llm.calls[2][1].content
    assert "Text of source number 1." in answer_prompt
    assert "Question: What is DPR?" in answer_prompt
    assert result.as_rag_answer().answer == result.answer


def test_relational_question_answers_from_graph_notes_without_citations() -> None:
    llm = ScriptedLLM(act("graph_lookup", kind="concept", value="BM25"), act("answer"), "arxiv:1 mentions it.")
    pipeline, _ = make(llm, graph=FakeGraph())
    result = pipeline.ask("Which papers mention BM25?")

    assert result.answer == "arxiv:1 mentions it."
    assert result.sources == []
    assert not result.has_citations
    prompt = llm.calls[2][1].content
    assert "Database notes (not citable)" in prompt
    assert "[arxiv:1] Paper One (2020)" in prompt
    assert "Sources:" not in prompt


def test_loop_stops_at_max_steps() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"), "Answer [1].")
    pipeline, retriever = make(llm, {"a": [chunk(1)], "b": [chunk(2)]}, max_steps=2)
    result = pipeline.ask("q?")
    assert result.stop_reason == "max_steps"
    assert len(result.trace) == 2
    assert len(retriever.calls) == 2
    assert len(llm.calls) == 3


def test_repeated_action_stops_the_loop() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="DPR"), act("retrieve_evidence", query="dpr"), "Answer [1].")
    pipeline, retriever = make(llm, {"DPR": [chunk(1)]})
    result = pipeline.ask("q?")
    assert result.stop_reason == "repeated_action"
    assert len(retriever.calls) == 1


def test_retrieval_limit_stops_the_loop() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"), "Answer [1].")
    pipeline, retriever = make(llm, {"a": [chunk(1)], "b": [chunk(2)]}, max_retrievals=1)
    result = pipeline.ask("q?")
    assert result.stop_reason == "retrieval_limit"
    assert len(retriever.calls) == 1


def test_two_empty_results_in_a_row_stop_without_calling_the_answer_llm() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"))
    pipeline, _ = make(llm)
    result = pipeline.ask("q?")
    assert result.stop_reason == "empty_results"
    assert result.answer == NO_SOURCES_MESSAGE
    assert result.sources == []
    assert len(llm.calls) == 2


def test_an_empty_result_followed_by_a_hit_does_not_stop() -> None:
    llm = ScriptedLLM(
        act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"), act("answer"), "Final [1]."
    )
    pipeline, _ = make(llm, {"b": [chunk(1)]})
    result = pipeline.ask("q?")
    assert result.stop_reason == "answered"
    assert [s.found for s in result.trace] == [0, 1]
    assert result.has_citations


def test_invalid_json_is_retried_once_with_the_bad_reply_echoed() -> None:
    llm = ScriptedLLM("not json", act("retrieve_evidence", query="a"), act("answer"), "Answer [1].")
    pipeline, _ = make(llm, {"a": [chunk(1)]})
    result = pipeline.ask("q?")
    retry = llm.calls[1]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == "not json"
    assert result.stop_reason == "answered"
    assert len(llm.calls) == 4


def test_unusable_decisions_fall_back_to_plain_rag() -> None:
    llm = ScriptedLLM("x", "y", "x", "y", "Answer [1].")
    pipeline, retriever = make(llm, {"What is DPR?": [chunk(1)]})
    result = pipeline.ask("What is DPR?")
    assert retriever.calls[0]["query"] == "What is DPR?"
    assert result.stop_reason == "invalid_decision"
    assert [s.decision.action for s in result.trace] == ["retrieve_evidence"]
    assert result.answer == "Answer [1]."


def test_answering_at_once_without_evidence_skips_the_answer_llm() -> None:
    llm = ScriptedLLM(act("answer"))
    pipeline, _ = make(llm)
    result = pipeline.ask("What is the capital of France?")
    assert result.answer == NO_SOURCES_MESSAGE
    assert result.trace == []
    assert len(llm.calls) == 1


def test_unknown_citations_are_reported() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("answer"), "A [1] and C [7].")
    pipeline, _ = make(llm, {"a": [chunk(1)]})
    result = pipeline.ask("q?")
    assert [s.number for s in result.cited_sources] == [1]
    assert result.invalid_citations == [7]


def test_passages_from_several_searches_are_combined_without_duplicates() -> None:
    llm = ScriptedLLM(
        act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"), act("answer"), "Answer [1]."
    )
    pipeline, _ = make(llm, {"a": [chunk(1), chunk(2)], "b": [chunk(2), chunk(3)]})
    result = pipeline.ask("q?")
    assert sorted(s.paper_uid for s in result.sources) == ["p1", "p2", "p3"]


def test_filters_reach_the_retriever() -> None:
    flt = SearchFilter(sources=["arxiv"])
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("answer"), "Answer [1].")
    pipeline, retriever = make(llm, {"a": [chunk(1)]})
    pipeline.ask("q?", flt)
    assert retriever.calls[0]["filters"] is flt


def test_llm_errors_propagate() -> None:
    pipeline, _ = make(ScriptedLLM(LLMError("Ollama is down")))
    with pytest.raises(LLMError):
        pipeline.ask("q?")


def test_invalid_arguments() -> None:
    pipeline, _ = make(ScriptedLLM())
    with pytest.raises(ValueError):
        pipeline.ask("  ")
    with pytest.raises(ValueError):
        make(ScriptedLLM(), max_steps=0)
    with pytest.raises(ValueError):
        make(ScriptedLLM(), max_retrievals=0)
