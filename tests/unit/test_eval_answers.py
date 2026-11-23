import pytest

from sc_research_agentic_rag.evaluation import EvalQuestion, Evidence
from sc_research_agentic_rag.evaluation.answers import AnswerCheck, aggregate_answers, evaluate_answer
from sc_research_agentic_rag.rag import NO_SOURCES_MESSAGE, RagAnswer, SourcePassage


def passage(number: int, text: str, uid: str = "a", page: int = 1) -> SourcePassage:
    return SourcePassage(
        number=number, paper_uid=uid, title="T", authors=(), year=None, doi=None, url=None, source="arxiv",
        page=page, text=text, chunk_ids=(f"{uid}#0",), score=0.9,
    )


def question(**evidence) -> EvalQuestion:
    return EvalQuestion(id="q", question="What?", evidence=[Evidence(paper_uid="a", **evidence)])


P1 = passage(1, "The similarity is the dot  product of their vectors.")
P2 = passage(2, "Something else.", uid="b")


def test_good_answer_with_gold_evidence_cited() -> None:
    answer = RagAnswer("What?", "DPR uses a dot product of vectors [1].", [P1, P2], [P1], [])
    check = evaluate_answer(question(contains="Dot Product"), answer)
    assert (check.status, check.has_sources, check.valid, check.invalid) == ("ok", True, 1, 0)
    assert (check.gold_retrieved, check.gold_cited) == (True, True)


def test_gold_retrieved_but_not_cited() -> None:
    answer = RagAnswer("What?", "Something else entirely happens here [2].", [P1, P2], [P2], [])
    check = evaluate_answer(question(contains="dot product"), answer)
    assert (check.gold_retrieved, check.gold_cited) == (True, False)


def test_page_must_match() -> None:
    answer = RagAnswer("What?", "DPR uses a dot product of vectors [1].", [P1], [P1], [])
    assert evaluate_answer(question(page=2, contains="dot product"), answer).gold_retrieved is False


def test_offsets_cannot_be_checked() -> None:
    answer = RagAnswer("What?", "DPR uses a dot product of vectors [1].", [P1], [P1], [])
    check = evaluate_answer(question(page=1, char_start=0, char_end=10), answer)
    assert (check.gold_retrieved, check.gold_cited) == (None, None)


def test_no_sources_answer() -> None:
    check = evaluate_answer(question(contains="x"), RagAnswer("What?", NO_SOURCES_MESSAGE))
    assert (check.status, check.has_sources) == ("no_sources", False)
    assert (check.gold_retrieved, check.gold_cited) == (False, False)


def test_invalid_citation_counts() -> None:
    answer = RagAnswer("What?", "DPR uses a dot product of vectors [1][5].", [P1], [P1], [5])
    check = evaluate_answer(question(contains="dot product"), answer)
    assert (check.status, check.valid, check.invalid) == ("invalid", 1, 1)


def test_aggregate() -> None:
    checks = [
        AnswerCheck("a", "ok", True, valid=2, sentences=4, uncited=1, gold_retrieved=True, gold_cited=True),
        AnswerCheck("b", "invalid", True, invalid=1, sentences=2, uncited=2, gold_retrieved=True, gold_cited=False),
        AnswerCheck("c", "no_sources", False, gold_retrieved=False, gold_cited=False),
    ]
    metrics = aggregate_answers(checks)
    assert metrics["no_sources_rate"] == pytest.approx(1 / 3)
    assert metrics["valid_citation_rate"] == pytest.approx(0.5)
    assert metrics["cited_rate"] == pytest.approx(0.5)
    assert metrics["uncited_sentence_rate"] == pytest.approx(0.5)
    assert metrics["gold_retrieved_rate"] == pytest.approx(2 / 3)
    assert metrics["gold_cited_rate"] == pytest.approx(1 / 3)


def test_aggregate_skips_unavailable_metrics() -> None:
    metrics = aggregate_answers([AnswerCheck("c", "no_sources", False)])
    assert metrics == {"no_sources_rate": 1.0}


def test_aggregate_of_nothing_is_empty() -> None:
    assert aggregate_answers([]) == {}
