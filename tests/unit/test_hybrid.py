import pytest

from sc_research_agentic_rag.retrieval import HybridRetriever, RetrievedChunk, rrf_fuse
from sc_research_agentic_rag.storage import SearchFilter


def chunk(chunk_id: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(chunk_id, score, f"text {chunk_id}", chunk_id.split("#")[0], 1, 0, 10, "T")


def ids(chunks) -> list[str]:
    return [c.chunk_id for c in chunks]


def test_rrf_scores_and_order() -> None:
    fused = rrf_fuse([[chunk("a#0"), chunk("b#0"), chunk("c#0")], [chunk("b#0"), chunk("d#0")]], k=60)
    assert ids(fused) == ["b#0", "a#0", "d#0", "c#0"]
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[1].score == pytest.approx(1 / 61)
    assert fused[2].score == pytest.approx(1 / 62)
    assert fused[3].score == pytest.approx(1 / 63)


def test_rrf_keeps_the_first_list_data_and_does_not_mutate_inputs() -> None:
    first, second = chunk("a#0", score=0.9), chunk("a#0", score=0.1)
    [fused] = rrf_fuse([[first], [second]])
    assert fused.text == first.text
    assert first.score == 0.9


def test_rrf_ties_keep_first_seen_order() -> None:
    assert ids(rrf_fuse([[chunk("a#0")], [chunk("b#0")]])) == ["a#0", "b#0"]


def test_rrf_empty() -> None:
    assert rrf_fuse([]) == []
    assert rrf_fuse([[], []]) == []


class FakeRanker:
    def __init__(self, ranking) -> None:
        self.ranking = ranking
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=5, filters=None):
        self.calls.append({"question": question, "top_k": top_k, "filters": filters})
        return self.ranking[:top_k]


def test_hybrid_fuses_both_rankers_and_cuts_to_top_k() -> None:
    dense = FakeRanker([chunk("a#0"), chunk("b#0"), chunk("c#0")])
    bm25 = FakeRanker([chunk("b#0"), chunk("d#0")])
    result = HybridRetriever(dense, bm25).retrieve("q?", top_k=2)
    assert ids(result) == ["b#0", "a#0"]


def test_hybrid_asks_each_ranker_for_candidates_and_forwards_filters() -> None:
    dense, bm25 = FakeRanker([]), FakeRanker([])
    flt = SearchFilter(sources=["arxiv"])
    HybridRetriever(dense, bm25, candidates=20).retrieve("q?", top_k=5, filters=flt)
    expected = {"question": "q?", "top_k": 20, "filters": flt}
    assert dense.calls == [expected]
    assert bm25.calls == [expected]
    HybridRetriever(dense, bm25, candidates=3).retrieve("q?", top_k=8)
    assert dense.calls[-1]["top_k"] == 8


def test_hybrid_works_when_one_side_is_empty() -> None:
    result = HybridRetriever(FakeRanker([chunk("a#0")]), FakeRanker([])).retrieve("q?")
    assert ids(result) == ["a#0"]


@pytest.mark.parametrize("kwargs", [{"top_k": 0}])
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        HybridRetriever(FakeRanker([]), FakeRanker([])).retrieve("q?", **kwargs)
    with pytest.raises(ValueError):
        HybridRetriever(FakeRanker([]), FakeRanker([]), candidates=0)
