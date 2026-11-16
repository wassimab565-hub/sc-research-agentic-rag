import pytest
from qdrant_client import QdrantClient

from sc_research_agentic_rag.retrieval import Bm25Index, Bm25Retriever, tokenize
from sc_research_agentic_rag.storage import PointRecord, SearchFilter, VectorStore, VectorStoreError


@pytest.mark.parametrize(
    "text,expected",
    [
        ("TP DE DEEP LEARNING - HAM500", ["tp", "de", "deep", "learning", "ham500"]),
        ("Développement à l'école", ["developpement", "a", "l", "ecole"]),
        ("snake_case and state-of-the-art", ["snake", "case", "and", "state", "of", "the", "art"]),
        ("", []),
    ],
)
def test_tokenize(text: str, expected: list[str]) -> None:
    assert tokenize(text) == expected


def test_accents_and_case_do_not_matter() -> None:
    index = Bm25Index(["Le développement des modèles", "Autre sujet"])
    scores = index.scores("DEVELOPPEMENT")
    assert scores[0] > 0
    assert scores[1] == 0


def test_rare_term_scores_higher_than_common_term() -> None:
    index = Bm25Index(["common rare", "common word", "common other", "common more"])
    assert index.scores("rare")[0] > index.scores("common")[0] > 0


def test_term_frequency_helps_but_saturates() -> None:
    index = Bm25Index(["needle " * 1 + "x " * 9, "needle " * 2 + "x " * 8, "needle " * 8 + "x " * 2, "other " * 10])
    scores = index.scores("needle")
    assert scores[0] < scores[1] < scores[2]
    assert scores[2] - scores[1] < (scores[1] - scores[0]) * 6
    assert scores[3] == 0


def test_shorter_document_wins_for_equal_term_count() -> None:
    index = Bm25Index(["needle " + "x " * 5, "needle " + "x " * 50, "filler " * 20])
    scores = index.scores("needle")
    assert scores[0] > scores[1]


def test_unknown_query_and_empty_index() -> None:
    assert Bm25Index(["a b c"]).scores("zzz") == [0.0]
    assert Bm25Index([]).scores("a") == []
    assert Bm25Index(["a"]).scores("") == [0.0]


def test_exact_value_for_a_simple_case() -> None:
    index = Bm25Index(["a b", "c d"])
    # N=2, df=1: idf = ln(1 + 1.5/1.5) = ln 2; tf=1, doc length == average -> (k1+1)/(1+k1) = 1
    assert index.scores("a") == [pytest.approx(0.6931471805599453), 0.0]


def make_store() -> VectorStore:
    store = VectorStore(QdrantClient(":memory:"), "chunks_test")
    store.ensure_collection(2)
    return store


def add(store: VectorStore, chunk_id: str, text: str, **payload) -> None:
    base = {"paper_uid": chunk_id.split("#")[0], "text": text, "page": 1}
    store.upsert([PointRecord(chunk_id, [1.0, 0.0], {**base, **payload})])


@pytest.fixture
def store() -> VectorStore:
    store = make_store()
    add(store, "arxiv:1#0", "Le rapport HAM500 décrit le dataset", source="arxiv", year=2020, title="One", page=3)
    add(store, "arxiv:1#1", "Les résultats du rapport", source="arxiv", year=2020, title="One")
    add(store, "openalex:W2#0", "Dense retrieval with vectors", source="openalex", year=2024, title="Two")
    add(store, "upload:r#0", "HAM500 appears here too", source="upload", year=None, title="r")
    return store


def test_returns_best_matches_with_provenance(store) -> None:
    chunks = Bm25Retriever(store).retrieve("HAM500", top_k=5)
    assert {c.chunk_id for c in chunks} == {"arxiv:1#0", "upload:r#0"}
    assert chunks[0].score >= chunks[1].score > 0
    first = next(c for c in chunks if c.chunk_id == "arxiv:1#0")
    assert (first.paper_uid, first.page, first.title, first.year) == ("arxiv:1", 3, "One", 2020)


def test_chunks_without_a_match_are_not_returned(store) -> None:
    assert Bm25Retriever(store).retrieve("zzzz") == []


def test_top_k_limits_results(store) -> None:
    assert len(Bm25Retriever(store).retrieve("rapport HAM500", top_k=1)) == 1


def test_filters_are_applied(store) -> None:
    retriever = Bm25Retriever(store)
    assert [c.chunk_id for c in retriever.retrieve("HAM500", filters=SearchFilter(sources=["upload"]))] == ["upload:r#0"]
    assert [c.chunk_id for c in retriever.retrieve("HAM500", filters=SearchFilter(year_min=2000))] == ["arxiv:1#0"]
    assert retriever.retrieve("HAM500", filters=SearchFilter(paper_uids=["nothing"])) == []


@pytest.mark.parametrize("kwargs", [{"question": "  "}, {"question": "q", "top_k": 0}])
def test_invalid_arguments(store, kwargs) -> None:
    with pytest.raises(ValueError):
        Bm25Retriever(store).retrieve(**kwargs)


def test_missing_collection_raises_clear_error() -> None:
    with pytest.raises(VectorStoreError):
        Bm25Retriever(VectorStore(QdrantClient(":memory:"), "nothing"))
