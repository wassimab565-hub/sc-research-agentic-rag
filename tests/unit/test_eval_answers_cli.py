import json

import pytest

from sc_research_agentic_rag.evaluation import EvalQuestion, Evidence, save_dataset
from sc_research_agentic_rag.evaluation import answers_cli as cli
from sc_research_agentic_rag.evaluation.answer_runs import SavedAnswer, SavedPassage
from sc_research_agentic_rag.llm import LLMError

DOT = "The similarity is the dot product of vectors."


class FakeEmbedder:
    def embed_passages(self, texts):
        return [[1.0, 0.0] for _ in texts]


@pytest.fixture(autouse=True)
def fakes(monkeypatch):
    for key in ("RETRIEVAL_MODE", "RAG_SCORE_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(cli, "build_support_embedder", FakeEmbedder)


@pytest.fixture
def dataset(tmp_path):
    questions = [
        EvalQuestion(id=f"q{n}", question=f"Question {n}?", evidence=[Evidence(paper_uid="a", contains="dot product")])
        for n in range(1, 4)
    ]
    path = tmp_path / "set.jsonl"
    save_dataset(questions, path)
    return path


def install_asker(monkeypatch, closed: list, calls: list, fail_on: str | None = None):
    def build(agent, mode, top_k, max_steps, max_retrievals):
        calls.append((agent, mode))

        def ask(question):
            if question.id == fail_on:
                raise LLMError("Ollama is down")
            return SavedAnswer(
                question.id, question.question, "Dense retrieval uses a dot product [1].",
                sources=(SavedPassage(1, "a", 1, DOT),), cited_numbers=(1,),
            )

        return ask, lambda: closed.append(True)

    monkeypatch.setattr(cli, "build_asker", build)


def run(tmp_path, dataset, *extra) -> int:
    return cli.main(["--dataset", str(dataset), "--answers-dir", str(tmp_path / "out"), *extra])


def test_run_prints_metrics_saves_cache_and_results(monkeypatch, tmp_path, dataset, capsys) -> None:
    closed, calls = [], []
    install_asker(monkeypatch, closed, calls)
    assert run(tmp_path, dataset, "--mode", "hybrid", "--show-claims") == 0
    out = capsys.readouterr().out
    assert "rag-hybrid: 3 answers (of 3 questions)" in out
    assert "statuses: ok 3" in out
    assert "valid_citation_rate" in out and "supported_claim_rate" in out
    assert "support >= t: 0.3: 1.00" in out
    assert 'q1 [supported 1.00] "Dense retrieval uses a dot product [1]."' in out
    assert closed == [True] and calls == [(False, "hybrid")]
    assert len((tmp_path / "out" / "answers_rag-hybrid.jsonl").read_text(encoding="utf-8").splitlines()) == 3
    [saved_json] = (tmp_path / "out").glob("answers_2*_rag-hybrid.json")
    data = json.loads(saved_json.read_text(encoding="utf-8"))
    assert data["label"] == "rag-hybrid"
    assert data["metrics"]["valid_citation_rate"] == 1.0
    assert data["questions"][0]["claims"][0]["status"] == "supported"


def test_second_run_uses_the_cache_and_builds_nothing(monkeypatch, tmp_path, dataset, capsys) -> None:
    install_asker(monkeypatch, [], [])
    run(tmp_path, dataset)
    capsys.readouterr()
    monkeypatch.setattr(cli, "build_asker", lambda *args: pytest.fail("nothing to ask"))
    assert run(tmp_path, dataset) == 0
    assert "3 answers" in capsys.readouterr().out


def test_limit_and_agent_label(monkeypatch, tmp_path, dataset, capsys) -> None:
    calls: list = []
    install_asker(monkeypatch, [], calls)
    assert run(tmp_path, dataset, "--agent", "--limit", "2") == 0
    out = capsys.readouterr().out
    assert "agent-dense: 2 answers (of 3 questions)" in out
    assert calls == [(True, "dense")]


def test_llm_error_keeps_progress_and_stops(monkeypatch, tmp_path, dataset, capsys) -> None:
    closed: list = []
    install_asker(monkeypatch, closed, [], fail_on="q2")
    assert run(tmp_path, dataset) == 1
    assert "resume" in capsys.readouterr().err
    assert closed == [True]
    assert len((tmp_path / "out" / "answers_rag-dense.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_no_claims_skips_the_embedder(monkeypatch, tmp_path, dataset, capsys) -> None:
    install_asker(monkeypatch, [], [])
    monkeypatch.setattr(cli, "build_support_embedder", lambda: pytest.fail("no embedder expected"))
    assert run(tmp_path, dataset, "--no-claims") == 0
    assert "statuses: ok 3" in capsys.readouterr().out


def test_missing_dataset(tmp_path, capsys) -> None:
    assert cli.main(["--dataset", str(tmp_path / "none.jsonl"), "--answers-dir", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv", [["--limit", "0"], ["--top-k", "0"], ["--support-threshold", "1.5"], ["--mode", "bm25"]]
)
def test_invalid_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
