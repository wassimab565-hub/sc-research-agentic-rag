import pytest
from qdrant_client import QdrantClient

from sc_research_agentic_rag.retrieval import RetrievedChunk, Retriever
from sc_research_agentic_rag.retrieval import cli
from sc_research_agentic_rag.storage import (
    PointRecord,
    SearchFilter,
    VectorStore,
    VectorStoreError,
)


class FakeEmbedder:
    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]


def make_store() -> VectorStore:
    store = VectorStore(QdrantClient(":memory:"), "chunks_test")
    store.ensure_collection(2)
    return store


def add(store: VectorStore, chunk_id: str, vector: list[float], **payload) -> None:
    base = {"paper_uid": chunk_id.split("#")[0], "text": f"text of {chunk_id}", "page": 1}
    store.upsert([PointRecord(chunk_id, vector, {**base, **payload})])


@pytest.fixture
def store() -> VectorStore:
    store = make_store()
    add(store, "arxiv:1#0", [1, 0.0], source="arxiv", year=2020, title="Paper One", page=3,
        authors=["Ann", "Bob"], doi="10.1/x", url="https://arxiv.org/abs/1", char_start=10, char_end=50)
    add(store, "arxiv:1#1", [1, 0.2], source="arxiv", year=2020, title="Paper One")
    add(store, "arxiv:1#2", [1, 0.4], source="arxiv", year=2020, title="Paper One")
    add(store, "openalex:W2#0", [1, 0.6], source="openalex", year=2024, title="Paper Two")
    add(store, "upload:rapport#0", [1, 1.0], source="upload", year=None, title="rapport")
    add(store, "arxiv:3#0", [0, 1.0], source="arxiv", year=2022, title="Paper Three")
    return store


def ids(chunks: list[RetrievedChunk]) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]


def test_returns_best_chunks_first_with_provenance(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve("question", top_k=2)
    assert ids(chunks) == ["arxiv:1#0", "arxiv:1#1"]
    first = chunks[0]
    assert first.score > chunks[1].score
    assert first.text == "text of arxiv:1#0"
    assert (first.paper_uid, first.page, first.char_start, first.char_end) == ("arxiv:1", 3, 10, 50)
    assert (first.title, first.authors, first.year) == ("Paper One", ["Ann", "Bob"], 2020)
    assert (first.doi, first.url, first.source) == ("10.1/x", "https://arxiv.org/abs/1", "arxiv")


def test_top_k_limits_results(store) -> None:
    assert len(Retriever(FakeEmbedder(), store).retrieve("q", top_k=3)) == 3


def test_filter_by_paper(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve(
        "q", top_k=10, filters=SearchFilter(paper_uids=["arxiv:3", "openalex:W2"])
    )
    assert ids(chunks) == ["openalex:W2#0", "arxiv:3#0"]


def test_filter_by_source(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve(
        "q", top_k=10, filters=SearchFilter(sources=["upload", "openalex"])
    )
    assert ids(chunks) == ["openalex:W2#0", "upload:rapport#0"]


def test_filter_by_year_excludes_papers_without_year(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve(
        "q", top_k=10, filters=SearchFilter(year_min=2021)
    )
    assert set(ids(chunks)) == {"openalex:W2#0", "arxiv:3#0"}


def test_filters_combine(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve(
        "q", top_k=10, filters=SearchFilter(sources=["arxiv"], year_max=2020)
    )
    assert set(ids(chunks)) == {"arxiv:1#0", "arxiv:1#1", "arxiv:1#2"}


def test_score_threshold_drops_weak_matches(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve("q", top_k=10, score_threshold=0.9)
    assert set(ids(chunks)) == {"arxiv:1#0", "arxiv:1#1", "arxiv:1#2"}
    assert all(chunk.score >= 0.9 for chunk in chunks)


def test_max_per_paper_spreads_results(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve("q", top_k=3, max_per_paper=1)
    assert ids(chunks) == ["arxiv:1#0", "openalex:W2#0", "upload:rapport#0"]


def test_max_per_paper_may_return_fewer_than_top_k(store) -> None:
    chunks = Retriever(FakeEmbedder(), store).retrieve("q", top_k=10, max_per_paper=1)
    assert len(chunks) == 4


def test_missing_payload_fields_use_defaults() -> None:
    store = make_store()
    store.upsert([PointRecord("x#0", [1.0, 0.0], {"paper_uid": "x"})])
    [chunk] = Retriever(FakeEmbedder(), store).retrieve("q")
    assert (chunk.text, chunk.page, chunk.title, chunk.authors, chunk.year) == ("", 0, "x", [], None)


def test_no_results_returns_empty_list() -> None:
    assert Retriever(FakeEmbedder(), make_store()).retrieve("q") == []


@pytest.mark.parametrize(
    "kwargs",
    [{"question": "  "}, {"question": "q", "top_k": 0}, {"question": "q", "max_per_paper": 0}],
)
def test_invalid_arguments(store, kwargs) -> None:
    with pytest.raises(ValueError):
        Retriever(FakeEmbedder(), store).retrieve(**kwargs)


def test_missing_collection_raises_clear_error() -> None:
    empty = VectorStore(QdrantClient(":memory:"), "nothing")
    with pytest.raises(VectorStoreError):
        Retriever(FakeEmbedder(), empty).retrieve("q")


def test_search_filter_validation_and_empty_filter() -> None:
    with pytest.raises(ValueError):
        SearchFilter(year_min=2024, year_max=2020)
    assert SearchFilter().to_qdrant() is None
    assert SearchFilter(paper_uids=[], sources=None).to_qdrant() is None


def test_cli_prints_results_with_provenance(monkeypatch, store, capsys) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), store))
    code = cli.main(["what is x", "--top-k", "1"])
    out = capsys.readouterr().out
    assert code == 0
    assert "1. [1.000] Paper One | p. 3 | arxiv:1#0" in out
    assert "Ann, Bob | 2020 | https://arxiv.org/abs/1" in out
    assert "text of arxiv:1#0" in out


def test_cli_applies_filters(monkeypatch, store, capsys) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), store))
    cli.main(["q", "--source", "upload", "--top-k", "5"])
    out = capsys.readouterr().out
    assert "upload:rapport#0" in out
    assert "arxiv" not in out


def test_cli_no_results(monkeypatch, store, capsys) -> None:
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), store))
    assert cli.main(["q", "--threshold", "2.0"]) == 0
    assert "no results" in capsys.readouterr().out


def test_cli_reports_store_errors(monkeypatch, capsys) -> None:
    empty = VectorStore(QdrantClient(":memory:"), "nothing")
    monkeypatch.setattr(cli, "build_services", lambda: (FakeEmbedder(), empty))
    assert cli.main(["q"]) == 1
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [["  "], ["q", "--top-k", "0"], ["q", "--max-per-paper", "0"], ["q", "--year-min", "2024", "--year-max", "2020"], ["q", "--source", "web"]],
)
def test_cli_rejects_invalid_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2