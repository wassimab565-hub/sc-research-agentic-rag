import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sc_research_agentic_rag.api import AppServices, create_app
from sc_research_agentic_rag.embeddings import EmbeddingError
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.rag import NO_SOURCES_MESSAGE
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter

TEXT = "Dense retrieval uses a dot product of vectors."
ANSWER = "Dense retrieval uses a dot product of vectors [1]."


def chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="p1#0", score=0.9, text=TEXT, paper_uid="p1", page=2, char_start=0, char_end=len(TEXT),
        title="Paper 1", authors=["Ann", "Bob"], year=2024, url="https://x/1", source="arxiv",
    )


def act(action: str, **args) -> str:
    return json.dumps({"action": action, "args": args})


class ScriptedLLM:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.calls = 0

    def chat(self, messages):
        self.calls += 1
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeRetriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        self.calls.append(
            {"query": question, "top_k": top_k, "filters": filters,
             "score_threshold": score_threshold, "max_per_paper": max_per_paper}
        )
        return self.chunks


class FakeSupportEmbedder:
    def embed_passages(self, texts):
        return [[1.0, 0.0] for _ in texts]


class BrokenSupportEmbedder:
    def embed_passages(self, texts):
        raise EmbeddingError("no model")


class FakeReader:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def read(self, query, **params):
        return self.answers.pop(0)


def make(llm, chunks=None, graph=None, support=FakeSupportEmbedder):
    dense = FakeRetriever([chunk()] if chunks is None else chunks)
    hybrid = FakeRetriever([chunk()] if chunks is None else chunks)
    services = AppServices(
        retrievers={"dense": dense, "hybrid": hybrid}, llm=llm, store=None, graph=graph,
        default_mode="dense", score_threshold=0.4, support_embedder_factory=support,
    )
    return TestClient(create_app(services)), dense, hybrid


def test_ask_returns_answer_sources_and_citations() -> None:
    client, dense, _ = make(ScriptedLLM(ANSWER))
    response = client.post("/ask", json={"question": "What?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == ANSWER
    assert data["cited_numbers"] == [1] and data["invalid_citations"] == []
    source = data["sources"][0]
    assert (source["number"], source["title"], source["page"], source["cited"]) == (1, "Paper 1", 2, True)
    assert source["authors"] == ["Ann", "Bob"] and source["text"] == TEXT
    assert data["verification"] is None
    assert dense.calls[0]["top_k"] == 5 and dense.calls[0]["score_threshold"] == 0.4


def test_ask_options_reach_the_retriever() -> None:
    client, dense, hybrid = make(ScriptedLLM(ANSWER))
    body = {
        "question": "What?", "mode": "hybrid", "top_k": 3, "threshold": 0, "max_per_paper": 2,
        "filters": {"sources": ["arxiv"], "year_min": 2020},
    }
    assert client.post("/ask", json=body).status_code == 200
    assert dense.calls == []
    call = hybrid.calls[0]
    assert (call["top_k"], call["score_threshold"], call["max_per_paper"]) == (3, 0.0, 2)
    assert call["filters"] == SearchFilter(sources=["arxiv"], year_min=2020)


def test_ask_without_sources_does_not_call_the_llm() -> None:
    llm = ScriptedLLM()
    client, _, _ = make(llm, chunks=[])
    data = client.post("/ask", json={"question": "What?"}).json()
    assert data["answer"] == NO_SOURCES_MESSAGE and data["sources"] == []
    assert llm.calls == 0


def test_ask_with_verification() -> None:
    client, _, _ = make(ScriptedLLM(ANSWER))
    verification = client.post("/ask", json={"question": "What?", "verify": True}).json()["verification"]
    assert verification["status"] == "ok"
    assert verification["note"] is None
    assert [c["status"] for c in verification["claims"]] == ["supported"]
    assert verification["claims"][0]["citations"] == [1]


def test_verification_survives_a_support_model_that_cannot_load() -> None:
    client, _, _ = make(ScriptedLLM(ANSWER), support=BrokenSupportEmbedder)
    verification = client.post("/ask", json={"question": "What?", "verify": True}).json()["verification"]
    assert "support score unavailable: no model" in verification["note"]
    assert [c["status"] for c in verification["claims"]] == ["unchecked"]


def test_misattributed_number_is_reported() -> None:
    answer = "Dense retrieval uses a dot product of 332 vectors [1]."
    second = RetrievedChunk("p2#0", 0.8, "There are 332 vectors here.", "p2", 1, 0, 27, "Paper 2")
    client, _, _ = make(ScriptedLLM(answer), chunks=[chunk(), second])
    claim = client.post("/ask", json={"question": "What?", "verify": True}).json()["verification"]["claims"][0]
    assert (claim["status"], claim["missing_numbers"], claim["found_in"]) == ("misattributed", ["332"], [2])


def test_errors_are_mapped_to_http_status() -> None:
    client, _, _ = make(ScriptedLLM(LLMError("Ollama is down")))
    response = client.post("/ask", json={"question": "What?"})
    assert response.status_code == 503 and "Ollama is down" in response.json()["detail"]


@pytest.mark.parametrize(
    "body",
    [
        {"question": "  "},
        {},
        {"question": "q", "top_k": 0},
        {"question": "q", "mode": "bm25"},
        {"question": "q", "support_threshold": 2},
        {"question": "q", "filters": {"sources": ["web"]}},
        {"question": "q", "filters": {"year_min": 2024, "year_max": 2020}},
    ],
)
def test_invalid_requests_are_422(body) -> None:
    client, _, _ = make(ScriptedLLM(ANSWER))
    assert client.post("/ask", json=body).status_code == 422


def test_agent_returns_the_trace_and_the_stop_reason() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="dpr"), act("answer"), ANSWER)
    client, dense, _ = make(llm)
    data = client.post("/agent/ask", json={"question": "What is DPR?"}).json()
    assert data["answer"] == ANSWER and data["cited_numbers"] == [1]
    assert data["stop_reason"] == "answered"
    step = data["trace"][0]
    assert (step["number"], step["action"], step["args"], step["found"]) == (1, "retrieve_evidence", {"query": "dpr"}, 1)
    assert dense.calls[0]["query"] == "dpr"
    assert data["verification"] is None


def test_agent_limits_and_filters_are_applied() -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="a"), act("retrieve_evidence", query="b"), ANSWER)
    client, _, hybrid = make(llm)
    body = {"question": "q?", "mode": "hybrid", "max_steps": 1, "filters": {"paper_uids": ["p1"]}}
    data = client.post("/agent/ask", json=body).json()
    assert data["stop_reason"] == "max_steps" and len(data["trace"]) == 1
    assert hybrid.calls[0]["filters"] == SearchFilter(paper_uids=["p1"])


def test_agent_verification_detects_a_citation_on_graph_notes() -> None:
    row = {"uid": "arxiv:1", "title": "Paper One", "year": 2020, "chunks": 2, "pages": [3], "authors": []}
    llm = ScriptedLLM(act("graph_lookup", kind="concept", value="BM25"), act("answer"), "The paper arxiv:1 mentions it [1].")
    client, _, _ = make(llm, chunks=[], graph=FakeReader([row]))
    data = client.post("/agent/ask", json={"question": "Which papers mention BM25?", "verify": True}).json()
    assert data["sources"] == []
    assert data["verification"]["status"] == "citation_on_notes"


def test_agent_llm_errors_are_503() -> None:
    client, _, _ = make(ScriptedLLM(LLMError("boom")))
    assert client.post("/agent/ask", json={"question": "q?"}).status_code == 503
