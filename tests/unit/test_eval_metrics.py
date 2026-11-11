import math

import pytest

from sc_research_agentic_rag.evaluation import EvalQuestion, Evidence, aggregate, evaluate_question, matches
from sc_research_agentic_rag.retrieval import RetrievedChunk


def chunk(chunk_id: str, paper: str = "p", page: int = 1, start: int = 0, text: str = "some text") -> RetrievedChunk:
    return RetrievedChunk(chunk_id, 0.5, text, paper, page, start, start + len(text), "T")


def question(*evidence: Evidence) -> EvalQuestion:
    return EvalQuestion(id="q", question="What?", evidence=list(evidence))


def test_paper_page_and_contains_must_all_match() -> None:
    c = chunk("a", paper="p", page=2, text="The  dot   Product of vectors")
    assert matches(Evidence(paper_uid="p"), c)
    assert matches(Evidence(paper_uid="p", page=2, contains="dot product"), c)
    assert not matches(Evidence(paper_uid="other"), c)
    assert not matches(Evidence(paper_uid="p", page=3), c)
    assert not matches(Evidence(paper_uid="p", contains="missing"), c)


def test_offsets_must_overlap() -> None:
    c = chunk("a", start=100, text="x" * 50)
    assert matches(Evidence(paper_uid="p", page=1, char_start=120, char_end=200), c)
    assert matches(Evidence(paper_uid="p", page=1, char_start=0, char_end=101), c)
    assert not matches(Evidence(paper_uid="p", page=1, char_start=150, char_end=200), c)
    assert not matches(Evidence(paper_uid="p", page=1, char_start=0, char_end=100), c)


def result(retrieved: list[RetrievedChunk], *evidence: Evidence):
    return evaluate_question(question(*evidence), retrieved, [c.chunk_id for c in retrieved])


def test_first_rank_and_matched_ranks_for_single_evidence() -> None:
    target = Evidence(paper_uid="p", contains="needle")
    res = result([chunk("a", text="x"), chunk("b", text="x"), chunk("c", text="a needle")], target)
    assert (res.first_rank, res.matched_ranks, res.n_evidence) == (3, [3], 1)
    assert res.retrieved_ids == ["a", "b", "c"]


def test_not_found_has_no_rank() -> None:
    res = result([chunk("a", text="x")], Evidence(paper_uid="p", contains="needle"))
    assert res.first_rank is None
    assert res.matched_ranks == []


def test_overlapping_neighbours_do_not_count_twice() -> None:
    target = Evidence(paper_uid="p", page=1, char_start=0, char_end=100)
    res = result([chunk("a", start=0, text="x" * 60), chunk("b", start=50, text="x" * 60)], target)
    assert res.first_rank == 1
    assert res.matched_ranks == [1]


def test_multiple_evidence_items_are_matched_independently() -> None:
    first = Evidence(paper_uid="p", contains="alpha")
    second = Evidence(paper_uid="p", contains="beta")
    res = result([chunk("a", text="alpha"), chunk("b", text="x"), chunk("c", text="x"), chunk("d", text="beta")], first, second)
    assert res.matched_ranks == [1, 4]
    assert res.first_rank == 1


def results_for(ranks: list[int | None]):
    out = []
    for index, rank in enumerate(ranks):
        retrieved = [chunk(f"{index}-{i}", text="needle" if i == rank else "x") for i in range(1, 11)]
        out.append(result(retrieved, Evidence(paper_uid="p", contains="needle")))
    return out


def test_aggregate_single_evidence_values() -> None:
    metrics = aggregate(results_for([1, 3, None]), [1, 3, 5])
    assert metrics["recall@1"] == pytest.approx(1 / 3)
    assert metrics["recall@3"] == pytest.approx(2 / 3)
    assert metrics["recall@5"] == pytest.approx(2 / 3)
    assert metrics["mrr"] == pytest.approx((1 + 1 / 3) / 3)
    assert metrics["ndcg@3"] == pytest.approx((1 + 1 / math.log2(4)) / 3)


def test_aggregate_multi_evidence_recall_and_ndcg() -> None:
    first = Evidence(paper_uid="p", contains="alpha")
    second = Evidence(paper_uid="p", contains="beta")
    retrieved = [chunk("a", text="alpha"), chunk("b", text="x"), chunk("c", text="x"), chunk("d", text="beta")]
    res = result(retrieved, first, second)
    metrics = aggregate([res], [3, 5])
    assert metrics["recall@3"] == 0.5
    assert metrics["recall@5"] == 1.0
    dcg = 1 + 1 / math.log2(5)
    ideal = 1 + 1 / math.log2(3)
    assert metrics["ndcg@5"] == pytest.approx(dcg / ideal)


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate([], [1, 3]) == {}