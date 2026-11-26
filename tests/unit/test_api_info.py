import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from sc_research_agentic_rag.api import AppServices, create_app
from sc_research_agentic_rag.api import cli
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.storage import PointRecord, VectorStore


class FakeLLM:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def list_models(self):
        if self.error:
            raise self.error
        return ["qwen3:8b"]


class FakeGraphStore:
    def counts(self):
        return {"Paper": 2}


class FakeReader:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)

    def read(self, query, **params):
        return self.answers.pop(0)


def make_store() -> VectorStore:
    store = VectorStore(QdrantClient(":memory:"), "chunks_test")
    store.ensure_collection(2)
    rows = [
        ("a#0", {"title": "Paper A", "authors": ["Ann"], "year": 2020, "source": "arxiv", "doi": "10.1/a"}),
        ("a#1", {"title": "Paper A", "authors": ["Ann"], "year": 2020, "source": "arxiv", "doi": "10.1/a"}),
        ("upload:r#0", {"title": "r", "source": "upload"}),
    ]
    for chunk_id, payload in rows:
        store.upsert([PointRecord(chunk_id, [1.0, 0.0], {"paper_uid": chunk_id.split("#")[0], "text": "t", **payload})])
    return store


def client_for(**kwargs) -> tuple[TestClient, AppServices]:
    defaults = dict(retrievers={}, llm=FakeLLM(), store=make_store(), graph=FakeGraphStore())
    services = AppServices(**{**defaults, **kwargs})
    return TestClient(create_app(services)), services


def test_health_ok() -> None:
    client, _ = client_for()
    data = client.get("/health").json()
    assert data["status"] == "ok" and data["default_mode"] == "dense"
    assert data["qdrant"] == {"ok": True, "chunks": 3}
    assert data["ollama"] == {"ok": True, "models": ["qwen3:8b"]}
    assert data["graph"] == {"ok": True, "counts": {"Paper": 2}}


def test_health_degraded_and_graph_not_configured() -> None:
    client, _ = client_for(llm=FakeLLM(LLMError("cannot reach Ollama")), graph=None)
    data = client.get("/health").json()
    assert data["status"] == "degraded"
    assert data["ollama"] == {"ok": False, "error": "cannot reach Ollama"}
    assert data["graph"] == {"ok": False, "error": "not configured"}


def test_health_does_not_wait_behind_a_running_request() -> None:
    client, services = client_for()
    with services.lock:
        data = client.get("/health").json()
    assert data["qdrant"] == {"ok": True, "busy": True}


def test_papers_are_grouped_by_paper() -> None:
    client, _ = client_for()
    papers = client.get("/papers").json()
    assert [p["paper_uid"] for p in papers] == ["a", "upload:r"]
    first = papers[0]
    assert (first["title"], first["chunks"], first["year"], first["authors"], first["doi"]) == ("Paper A", 2, 2020, ["Ann"], "10.1/a")
    assert (papers[1]["source"], papers[1]["chunks"], papers[1]["year"]) == ("upload", 1, None)


CONCEPT_ROW = {
    "uid": "arxiv:1", "title": "Paper One", "year": 2020, "chunks": 2,
    "pages": [7, 3], "authors": ["Ann", "Bob"],
}


def test_graph_concept() -> None:
    client, _ = client_for(graph=FakeReader([CONCEPT_ROW]))
    data = client.get("/graph/concept", params={"name": "BM25"}).json()
    assert data["papers"] == [
        {"uid": "arxiv:1", "title": "Paper One", "year": 2020, "authors": ["Ann", "Bob"], "pages": [3, 7], "chunks": 2}
    ]
    assert data["similar_concepts"] == []


def test_graph_concept_without_paper_suggests_similar_concepts() -> None:
    client, _ = client_for(graph=FakeReader([], [{"name": "BM25", "chunks": 25}]))
    data = client.get("/graph/concept", params={"name": "bm"}).json()
    assert data["papers"] == []
    assert data["similar_concepts"] == [{"name": "BM25", "count": 25}]


def test_graph_author() -> None:
    reader = FakeReader(
        [{"uid": "arxiv:2", "title": None, "year": None, "matched": ["Patrick Lewis"]}],
        [{"name": "RAG", "chunks": 9}],
    )
    data = client_for(graph=reader)[0].get("/graph/author", params={"name": "lewis"}).json()
    assert data["matched_authors"] == ["Patrick Lewis"]
    assert data["papers"][0]["uid"] == "arxiv:2"
    assert data["concepts"] == [{"name": "RAG", "count": 9}]


def test_graph_related() -> None:
    row = {"uid": "q", "title": "Q", "year": None, "score": 0.5, "shared": [{"name": "BM25", "df": 2}]}
    data = client_for(graph=FakeReader([row]))[0].get("/graph/related", params={"uid": "arxiv:1", "limit": 3}).json()
    assert data["papers"] == [{"uid": "q", "title": "Q", "year": None, "score": 0.5, "shared": ["BM25"]}]


def test_graph_missing_is_503_and_bad_input_is_422() -> None:
    client, _ = client_for(graph=None)
    response = client.get("/graph/concept", params={"name": "x"})
    assert response.status_code == 503 and "not available" in response.json()["detail"]
    client, _ = client_for(graph=FakeReader())
    assert client.get("/graph/concept", params={"name": " "}).status_code == 422
    assert client.get("/graph/concept").status_code == 422
    assert client.get("/graph/concept", params={"name": "x", "limit": 0}).status_code == 422


def test_cli_starts_uvicorn_with_one_app(monkeypatch) -> None:
    for key in ("API_HOST", "API_PORT"):
        monkeypatch.delenv(key, raising=False)
    received: dict = {}
    monkeypatch.setattr(cli.uvicorn, "run", lambda app, **kwargs: received.update(app=app, **kwargs))
    assert cli.main([]) == 0
    assert (received["host"], received["port"]) == ("127.0.0.1", 8000)
    cli.main(["--port", "9000"])
    assert received["port"] == 9000


def test_cli_rejects_invalid_port() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--port", "0"])
    assert exc.value.code == 2
