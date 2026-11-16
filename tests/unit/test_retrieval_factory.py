import pytest
from pydantic import ValidationError
from qdrant_client import QdrantClient

from sc_research_agentic_rag.retrieval import (
    MODES,
    HybridRetriever,
    RetrievalSettings,
    Retriever,
    build_retriever,
)
from sc_research_agentic_rag.retrieval import cli
from sc_research_agentic_rag.storage import PointRecord, VectorStore, VectorStoreError


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_MODE", raising=False)


@pytest.fixture
def store() -> VectorStore:
    store = VectorStore(QdrantClient(":memory:"), "chunks_test")
    store.ensure_collection(2)
    rows = [
        ("arxiv:1#0", [1.0, 0.0], "alpha text about HAM500"),
        ("arxiv:1#1", [1.0, 0.3], "beta text"),
        ("arxiv:2#0", [0.0, 1.0], "gamma HAM500 text"),
    ]
    for chunk_id, vector, text in rows:
        payload = {"paper_uid": chunk_id.split("#")[0], "text": text, "page": 1, "source": "arxiv", "title": "T"}
        store.upsert([PointRecord(chunk_id, vector, payload)])
    return store


def test_settings_defaults_and_modes(clean_env) -> None:
    assert RetrievalSettings(_env_file=None).mode == "dense"
    assert MODES == ("dense", "hybrid")


def test_settings_from_environment(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    assert RetrievalSettings(_env_file=None).mode == "hybrid"


def test_blank_mode_falls_back_to_dense(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", " ")
    assert RetrievalSettings(_env_file=None).mode == "dense"


def test_unknown_mode_in_settings_is_rejected(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "bm25")
    with pytest.raises(ValidationError):
        RetrievalSettings(_env_file=None)


def test_env_file_ignores_unrelated_keys(clean_env, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENALEX_API_KEY=abc\nRETRIEVAL_MODE=hybrid\n", encoding="utf-8")
    assert RetrievalSettings(_env_file=env_file).mode == "hybrid"


def test_factory_builds_each_mode(store) -> None:
    assert isinstance(build_retriever("dense", FakeEmbedder(), store), Retriever)
    assert isinstance(build_retriever("hybrid", FakeEmbedder(), store), HybridRetriever)


def test_factory_rejects_unknown_mode(store) -> None:
    with pytest.raises(ValueError, match="bm25"):
        build_retriever("bm25", FakeEmbedder(), store)


def test_hybrid_needs_an_existing_collection() -> None:
    empty = VectorStore(QdrantClient(":memory:"), "nothing")
    with pytest.raises(VectorStoreError):
        build_retriever("hybrid", FakeEmbedder(), empty)


def test_hybrid_retrieves_through_the_real_store(store) -> None:
    chunks = build_retriever("hybrid", FakeEmbedder(), store).retrieve("HAM500", top_k=3)
    assert {c.chunk_id for c in chunks} == {"arxiv:1#0", "arxiv:1#1", "arxiv:2#0"}
    assert chunks[0].chunk_id == "arxiv:1#0"


def test_hybrid_threshold_gate_with_the_real_store(store) -> None:
    retriever = build_retriever("hybrid", FakeEmbedder(), store)
    assert retriever.retrieve("HAM500", score_threshold=2.0) == []
    assert retriever.retrieve("HAM500", score_threshold=0.5)


def test_cli_mode_option_selects_hybrid(clean_env, monkeypatch, store, capsys) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), store))
    assert cli.main(["HAM500", "--mode", "hybrid", "--top-k", "3"]) == 0
    assert "arxiv:2#0" in capsys.readouterr().out


def test_cli_mode_defaults_to_the_setting(monkeypatch, store, capsys) -> None:
    monkeypatch.setenv("RETRIEVAL_MODE", "hybrid")
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), store))
    cli.main(["HAM500", "--top-k", "1"])
    assert "[0.0" in capsys.readouterr().out


def test_cli_rejects_unknown_mode() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["q", "--mode", "bm25"])
    assert exc.value.code == 2
