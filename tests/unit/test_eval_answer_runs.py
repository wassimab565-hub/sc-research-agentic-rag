from types import SimpleNamespace

import pytest

from sc_research_agentic_rag.evaluation import EvalQuestion, Evidence
from sc_research_agentic_rag.evaluation.answer_runs import (
    SavedAnswer,
    SavedPassage,
    aggregate_claims,
    append_answer,
    collect_answers,
    evaluate_run,
    load_answers,
    saved_answer,
    support_curve,
)
from sc_research_agentic_rag.verification import ClaimCheck

DOT = "The similarity is the dot product of vectors."


def question(qid: str = "q1") -> EvalQuestion:
    return EvalQuestion(id=qid, question=f"What about {qid}?", evidence=[Evidence(paper_uid="a", contains="dot product")])


def saved(qid: str = "q1", answer: str = "Dense retrieval uses a dot product [1].", **kwargs) -> SavedAnswer:
    sources = (SavedPassage(1, "a", 1, DOT),)
    defaults = dict(question="What?", sources=sources, cited_numbers=(1,))
    return SavedAnswer(qid, **{**defaults, **kwargs, "answer": answer})


class FakeEmbedder:
    def embed_passages(self, texts):
        return [[1.0, 0.0, 0.1] if "dot" in text else [0.0, 1.0, 0.1] for text in texts]


def test_saved_answer_from_a_rag_result() -> None:
    passage = SimpleNamespace(number=1, paper_uid="a", page=3, text=DOT)
    result = SimpleNamespace(answer="A [1].", sources=[passage], cited_sources=[passage])
    out = saved_answer("q1", "What?", result)
    assert out.sources == (SavedPassage(1, "a", 3, DOT),)
    assert (out.cited_numbers, out.has_notes, out.stop_reason) == ((1,), False, "")


def test_saved_answer_from_an_agent_result_detects_notes() -> None:
    def step(action: str, found: int):
        return SimpleNamespace(found=found, decision=SimpleNamespace(action=action))

    result = SimpleNamespace(
        answer="x", sources=[], cited_sources=[], stop_reason="answered",
        trace=[step("retrieve_evidence", 2), step("graph_lookup", 3)],
    )
    out = saved_answer("q1", "What?", result)
    assert (out.has_notes, out.stop_reason) == (True, "answered")
    only_evidence = SimpleNamespace(answer="x", sources=[], cited_sources=[], trace=[step("retrieve_evidence", 2)])
    assert saved_answer("q1", "?", only_evidence).has_notes is False


def test_cache_roundtrip_skips_bad_lines_and_last_line_wins(tmp_path) -> None:
    path = tmp_path / "nested" / "answers.jsonl"
    append_answer(path, saved("q1", "Réponse avec accents é à [1]."))
    with path.open("a", encoding="utf-8") as handle:
        handle.write("not json\n{}\n")
    append_answer(path, saved("q1", "Newer answer about the dot product [1]."))
    append_answer(path, saved("q2"))
    answers = load_answers(path)
    assert set(answers) == {"q1", "q2"}
    assert answers["q1"].answer == "Newer answer about the dot product [1]."
    assert answers["q2"].sources == (SavedPassage(1, "a", 1, DOT),)
    assert answers["q2"].cited_sources == answers["q2"].sources
    assert load_answers(tmp_path / "missing.jsonl") == {}


def test_collect_asks_only_missing_questions_and_resumes(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    asked: list[str] = []

    def ask(q: EvalQuestion) -> SavedAnswer:
        asked.append(q.id)
        if q.id == "q3" and "fail" in asked:
            raise RuntimeError("boom")
        return saved(q.id)

    questions = [question("q1"), question("q2"), question("q3")]
    collect_answers(questions[:1], ask, path)
    asked.append("fail")
    with pytest.raises(RuntimeError):
        collect_answers(questions, ask, path)
    assert set(load_answers(path)) == {"q1", "q2"}
    asked.clear()
    answers = collect_answers(questions, lambda q: saved(q.id), path)
    assert set(answers) == {"q1", "q2", "q3"}
    assert asked == []


def test_collect_limit_counts_new_answers_only(tmp_path) -> None:
    path = tmp_path / "a.jsonl"
    questions = [question(f"q{n}") for n in range(4)]
    collect_answers(questions, lambda q: saved(q.id), path, limit=1)
    progress: list[int] = []
    answers = collect_answers(questions, lambda q: saved(q.id), path, limit=2, on_progress=progress.append)
    assert len(answers) == 3
    assert progress == [1, 2]


def test_aggregate_claims_and_support_curve() -> None:
    claims = [
        ClaimCheck("a", (1,), "supported", 0.9),
        ClaimCheck("b", (1,), "weak", 0.4),
        ClaimCheck("c", (1,), "number_missing", 0.95, ("5",)),
        ClaimCheck("d", (9,), "no_valid_source"),
    ]
    metrics = aggregate_claims(claims)
    assert metrics == {
        "supported_claim_rate": 0.25, "weak_claim_rate": 0.25,
        "number_missing_claim_rate": 0.25, "no_valid_source_claim_rate": 0.25,
    }
    assert support_curve(claims, [0.3, 0.5, 0.95]) == {0.3: 1.0, 0.5: 0.5, 0.95: 0.0}
    assert aggregate_claims([]) == {} and support_curve([], [0.5]) == {}


def test_evaluate_run_combines_answer_and_claim_metrics() -> None:
    questions = [question("q1"), question("q2"), question("q3")]
    answers = {
        "q1": saved("q1"),
        "q2": saved("q2", "BM25 beats everything else [1]."),
    }
    result = evaluate_run(questions, answers, FakeEmbedder(), 0.5)
    assert [c.question_id for c in result.checks] == ["q1", "q2"]
    assert result.metrics["valid_citation_rate"] == 1.0
    assert result.metrics["gold_cited_rate"] == 1.0
    assert result.metrics["supported_claim_rate"] == 0.5
    assert result.metrics["weak_claim_rate"] == 0.5
    assert [c.status for c in result.claims["q2"]] == ["weak"]


def test_evaluate_run_without_embedder_leaves_claims_unchecked() -> None:
    result = evaluate_run([question("q1")], {"q1": saved("q1")}, None, 0.5)
    assert [c.status for c in result.claims["q1"]] == ["unchecked"]
    assert "supported_claim_rate" in result.metrics and result.metrics["supported_claim_rate"] == 0.0
