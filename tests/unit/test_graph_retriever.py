import pytest

from sc_research_agentic_rag.graph import GraphRetriever
from sc_research_agentic_rag.graph.retriever import build_index, match_concepts
from sc_research_agentic_rag.storage import SearchFilter


class FakeGraph:
    def __init__(self, links: dict[str, list[str]], total: int = 10) -> None:
        self.links, self.total = links, total

    def concept_keys(self):
        return list(self.links)

    def chunk_count(self):
        return self.total

    def concept_chunks(self, keys):
        return [(key, chunk_id) for key in keys for chunk_id in self.links[key]]


class FakeStore:
    def __init__(self, years: dict[str, int | None] | None = None, missing: tuple[str, ...] = ()) -> None:
        self.years, self.missing = years or {}, missing

    def get_payloads(self, chunk_ids):
        return {
            chunk_id: {
                "chunk_id": chunk_id, "paper_uid": chunk_id.split("#")[0], "text": f"text {chunk_id}",
                "page": 2, "title": "T", "source": "arxiv", "year": self.years.get(chunk_id, 2020),
            }
            for chunk_id in chunk_ids
            if chunk_id not in self.missing
        }


def ids(chunks) -> list[str]:
    return [chunk.chunk_id for chunk in chunks]


def test_match_concepts_uses_tokens_phrases_and_ignores_case_and_accents() -> None:
    index = build_index(["dense retrieval", "retrieval", "bm25", "retrieve-and-generate", "unrelated"])
    found = match_concepts("How does Dense Retrieval compare to BM25?", index)
    assert sorted(found) == ["bm25", "dense retrieval", "retrieval"]
    assert match_concepts("What is retrieve and generate?", index) == ["retrieve-and-generate"]
    assert match_concepts("nothing here", index) == []


def test_rare_concepts_rank_first_with_provenance() -> None:
    graph = FakeGraph({"common": [f"a#{i}" for i in range(6)], "rare": ["a#3"]})
    chunks = GraphRetriever(graph, FakeStore()).retrieve("common rare", top_k=3)
    assert chunks[0].chunk_id == "a#3"
    assert chunks[0].score > chunks[1].score > 0
    first = chunks[0]
    assert (first.paper_uid, first.page, first.text, first.title) == ("a", 2, "text a#3", "T")


def test_no_matching_concept_returns_nothing() -> None:
    assert GraphRetriever(FakeGraph({"bm25": ["a#0"]}), FakeStore()).retrieve("unknown words") == []


def test_top_k_limits_results() -> None:
    graph = FakeGraph({"bm25": [f"a#{i}" for i in range(6)]})
    assert len(GraphRetriever(graph, FakeStore()).retrieve("bm25", top_k=2)) == 2


def test_filters_and_chunks_missing_from_the_store_are_skipped() -> None:
    graph = FakeGraph({"bm25": ["a#0", "a#1", "a#2"]})
    store = FakeStore(years={"a#0": 2010}, missing=("a#1",))
    retriever = GraphRetriever(graph, store)
    assert ids(retriever.retrieve("bm25", filters=SearchFilter(year_min=2015))) == ["a#2"]
    assert ids(retriever.retrieve("bm25")) == ["a#0", "a#2"]


@pytest.mark.parametrize("kwargs", [{"question": "  "}, {"question": "bm25", "top_k": 0}])
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        GraphRetriever(FakeGraph({"bm25": ["a#0"]}), FakeStore()).retrieve(**kwargs)