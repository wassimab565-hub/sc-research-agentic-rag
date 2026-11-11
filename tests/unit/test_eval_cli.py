import json

import pytest

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation import EvalQuestion, Evidence, save_dataset
from sc_research_agentic_rag.evaluation import cli
from sc_research_agentic_rag.retrieval import RetrievedChunk


def make_chunk(paper: str, index: int, text: str, page: int = 1) -> Chunk:
    start = index * 1000
    return Chunk(paper_uid=paper, index=index, page=page, text=text, char_start=start, char_end=start + len(text))


@pytest.fixture
def chunks_dir(tmp_path):
    directory = tmp_path / "chunks"
    directory.mkdir()
    chunks = [make_chunk("p", 0, "alpha text"), make_chunk("p", 1, "beta text"), make_chunk("p", 2, "gamma text")]
    payload = {"paper_uid": "p", "paper": None, "chunks": [{**c.model_dump(), "chunk_id": c.chunk_id} for c in chunks]}
    (directory / "p.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory


def dataset(tmp_path, name: str = "set"):
    questions = [
        EvalQuestion(id="q-alpha", question="Which one is alpha?", evidence=[Evidence(paper_uid="p", contains="alpha")]),
        EvalQuestion(id="q-beta", question="Which one is beta?", evidence=[Evidence(paper_uid="p", contains="beta")]),
        EvalQuestion(id="q-gamma", question="Which one is gamma?", evidence=[Evidence(paper_uid="p", contains="gamma")]),
    ]
    path = tmp_path / f"{name}.jsonl"
    save_dataset(questions, path)
    return path


def retrieved(chunk: Chunk, rank: int) -> RetrievedChunk:
    return RetrievedChunk(chunk.chunk_id, 1 / rank, chunk.text, chunk.paper_uid, chunk.page, chunk.char_start, chunk.char_end, "T")


def fake_services(order_by_question: dict[str, list[str]], closed: list):
    corpus = {
        "alpha": make_chunk("p", 0, "alpha text"),
        "beta": make_chunk("p", 1, "beta text"),
        "gamma": make_chunk("p", 2, "gamma text"),
    }

    def retrieve(question: str, k: int):
        names = order_by_question[question]
        return [retrieved(corpus[name], rank) for rank, name in enumerate(names, start=1)][:k]

    return lambda: (retrieve, "fake/model", lambda: closed.append(True))


PERFECT = {"Which one is alpha?": ["alpha", "beta"], "Which one is beta?": ["beta", "alpha"], "Which one is gamma?": ["gamma", "beta"]}


def run(monkeypatch, tmp_path, chunks_dir, order, *extra):
    closed: list = []
    monkeypatch.setattr(cli, "build_retrieve_fn", fake_services(order, closed))
    argv = ["--dataset", str(dataset(tmp_path)), "--chunks-dir", str(chunks_dir), "--output-dir", str(tmp_path / "results"), *extra]
    return cli.main(argv), closed


def test_reports_metrics_and_saves_results(monkeypatch, tmp_path, chunks_dir, capsys) -> None:
    code, closed = run(monkeypatch, tmp_path, chunks_dir, PERFECT, "--k", "1", "3", "--label", "dense")
    out = capsys.readouterr().out
    assert code == 0
    assert closed == [True]
    assert "set (3 questions)" in out
    assert "Recall@1   1.000" in out
    assert "MRR        1.000" in out
    saved = list((tmp_path / "results").glob("*_dense.json"))
    assert len(saved) == 1
    data = json.loads(saved[0].read_text(encoding="utf-8"))
    assert data["label"] == "dense"
    assert data["embedding_model"] == "fake/model"
    assert data["datasets"]["set"]["metrics"]["recall@1"] == 1.0
    assert data["datasets"]["set"]["results"][0]["id"] == "q-alpha"


def test_partial_quality_is_computed(monkeypatch, tmp_path, chunks_dir, capsys) -> None:
    order = {**PERFECT, "Which one is gamma?": ["beta", "alpha", "gamma"]}
    run(monkeypatch, tmp_path, chunks_dir, order, "--k", "1", "3", "--show-misses", "1")
    out = capsys.readouterr().out
    assert "Recall@1   0.667" in out
    assert "Recall@3   1.000" in out
    assert "q-gamma (rank 3)" in out
    assert "expected: p page None; got: p#1, p#0, p#2" in out


def test_missed_questions_are_listed_first(monkeypatch, tmp_path, chunks_dir, capsys) -> None:
    order = {**PERFECT, "Which one is gamma?": ["beta", "alpha"]}
    run(monkeypatch, tmp_path, chunks_dir, order, "--show-misses", "1")
    assert "q-gamma (not found)" in capsys.readouterr().out


def test_invalid_dataset_stops_before_retrieval(monkeypatch, tmp_path, chunks_dir, capsys) -> None:
    path = tmp_path / "bad.jsonl"
    save_dataset([EvalQuestion(id="x", question="Q?", evidence=[Evidence(paper_uid="p", contains="absent")])], path)
    monkeypatch.setattr(cli, "build_retrieve_fn", lambda: pytest.fail("retrieval must not start"))
    assert cli.main(["--dataset", str(path), "--chunks-dir", str(chunks_dir)]) == 1
    assert "matches no chunk" in capsys.readouterr().err


def test_validate_only(monkeypatch, tmp_path, chunks_dir, capsys) -> None:
    monkeypatch.setattr(cli, "build_retrieve_fn", lambda: pytest.fail("retrieval must not start"))
    code = cli.main(["--dataset", str(dataset(tmp_path)), "--chunks-dir", str(chunks_dir), "--validate"])
    assert code == 0
    assert "datasets are valid: set (3 questions)" in capsys.readouterr().out


def test_missing_inputs(tmp_path, capsys) -> None:
    assert cli.main(["--dataset", str(tmp_path / "none.jsonl"), "--chunks-dir", str(tmp_path)]) == 1
    assert "cannot read" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["--dataset", str(dataset(tmp_path)), "--chunks-dir", str(empty)]) == 1
    assert "no chunks" in capsys.readouterr().err


def test_invalid_k() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--dataset", "x.jsonl", "--k", "0"])
    assert exc.value.code == 2


def test_manual_dataset_in_the_repo_is_well_formed() -> None:
    from pathlib import Path

    from sc_research_agentic_rag.evaluation import load_dataset

    manual = load_dataset(Path("data/evaluation/manual.jsonl"))
    assert 10 <= len(manual) <= 15
    assert len({q.id for q in manual}) == len(manual)