import json

import pytest

from sc_research_agentic_rag.agent import AgentPipeline, AgentTools
from sc_research_agentic_rag.agent import cli as agent_cli
from sc_research_agentic_rag.graph.queries import ConceptPaper
from sc_research_agentic_rag.rag import RagPipeline
from sc_research_agentic_rag.rag import cli as rag_cli
from sc_research_agentic_rag.retrieval import RetrievedChunk


def chunk(n: int) -> RetrievedChunk:
    text = f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"p#{n}", score=0.9, text=text, paper_uid=f"p{n}", page=1,
        char_start=0, char_end=len(text), title=f"Paper {n}",
    )


class FakeRetriever:
    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        return [chunk(1)]


class FakeLLM:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)

    def chat(self, messages):
        return self.replies.pop(0)


class FakeGraph:
    def papers_for_concept(self, name, limit=10):
        return [ConceptPaper("arxiv:1", "Paper One", 2020, (), (3,), 2)]

    def similar_concepts(self, name, limit=5):
        return []


class Store:
    def close(self) -> None:
        pass


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("AGENT_MAX_STEPS", "AGENT_MAX_RETRIEVALS", "AGENT_TOP_K", "RETRIEVAL_MODE", "RAG_SCORE_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)


def test_rag_cli_verify(monkeypatch, clean_env, capsys) -> None:
    pipeline = RagPipeline(FakeRetriever(), FakeLLM("The first source says something useful [1]. It also adds more."))
    monkeypatch.setattr(rag_cli, "build_pipeline", lambda *args: (pipeline, Store()))
    assert rag_cli.main(["What?", "--verify"]) == 0
    out = capsys.readouterr().out
    assert "Verification: ok" in out
    assert "uncited sentences: 1/2" in out


def test_rag_cli_without_verify_prints_no_report(monkeypatch, clean_env, capsys) -> None:
    pipeline = RagPipeline(FakeRetriever(), FakeLLM("The first source says something useful [1]."))
    monkeypatch.setattr(rag_cli, "build_pipeline", lambda *args: (pipeline, Store()))
    rag_cli.main(["What?"])
    assert "Verification" not in capsys.readouterr().out


def test_agent_cli_verify_detects_citation_on_graph_notes(monkeypatch, clean_env, capsys) -> None:
    act = lambda action, **args: json.dumps({"action": action, "args": args})  # noqa: E731
    llm = FakeLLM(act("graph_lookup", kind="concept", value="BM25"), act("answer"), "The paper arxiv:1 mentions it [1].")
    pipeline = AgentPipeline(llm, AgentTools(FakeRetriever(), FakeGraph()))
    monkeypatch.setattr(agent_cli, "build_pipeline", lambda *args: (pipeline, lambda: None))
    assert agent_cli.main(["Which papers mention BM25?", "--verify"]) == 0
    assert "Verification: citation_on_notes" in capsys.readouterr().out
