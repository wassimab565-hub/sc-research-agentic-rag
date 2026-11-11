import json

import pytest
from pydantic import ValidationError

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation import (
    DatasetError,
    EvalQuestion,
    Evidence,
    load_chunk_corpus,
    load_dataset,
    save_dataset,
    validate_dataset,
)


def make_question(qid: str = "q1", **evidence) -> EvalQuestion:
    return EvalQuestion(id=qid, question="What is x?", evidence=[Evidence(paper_uid="p", **evidence)])


def make_chunk(index: int = 0, text: str = "alpha beta", paper: str = "p", page: int = 1) -> Chunk:
    return Chunk(paper_uid=paper, index=index, page=page, text=text, char_start=index * 100, char_end=index * 100 + len(text))


def test_save_and_load_roundtrip(tmp_path) -> None:
    questions = [make_question("a", contains="é à"), make_question("b", page=3)]
    path = tmp_path / "nested" / "set.jsonl"
    save_dataset(questions, path)
    assert "é à" in path.read_text(encoding="utf-8")
    assert load_dataset(path) == questions


def test_blank_lines_are_ignored(tmp_path) -> None:
    path = tmp_path / "set.jsonl"
    save_dataset([make_question("a")], path)
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert len(load_dataset(path)) == 1


@pytest.mark.parametrize(
    "content,message",
    [
        ("not json\n", "set.jsonl:1"),
        ('{"id": "a", "question": "q?"}\n', "set.jsonl:1"),
        ('{"id": "a", "question": " ", "evidence": [{"paper_uid": "p"}]}\n', "set.jsonl:1"),
        ('{"id": "a", "question": "q?", "evidence": []}\n', "set.jsonl:1"),
    ],
)
def test_invalid_lines_report_the_line_number(tmp_path, content, message) -> None:
    path = tmp_path / "set.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset(path)


def test_duplicate_ids_are_rejected(tmp_path) -> None:
    path = tmp_path / "set.jsonl"
    save_dataset([make_question("a"), make_question("a")], path)
    with pytest.raises(DatasetError, match="duplicate question id 'a'"):
        load_dataset(path)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(DatasetError):
        load_dataset(tmp_path / "missing.jsonl")


@pytest.mark.parametrize(
    "kwargs",
    [{"contains": "  "}, {"char_start": 5}, {"char_start": 10, "char_end": 10}, {"page": 0}],
)
def test_invalid_evidence(kwargs) -> None:
    with pytest.raises(ValidationError):
        Evidence(paper_uid="p", **kwargs)


def test_validation_flags_unmatched_and_broad_evidence() -> None:
    corpus = [make_chunk(i, text="common word") for i in range(12)] + [make_chunk(20, text="rare needle")]
    questions = [
        make_question("ok", contains="needle"),
        make_question("missing", contains="absent"),
        make_question("broad", contains="common"),
    ]
    issues = validate_dataset(questions, corpus)
    assert [(i.question_id, i.level) for i in issues] == [("missing", "error"), ("broad", "warning")]
    assert "matches no chunk" in issues[0].message
    assert "12 chunks" in issues[1].message


def test_load_chunk_corpus(tmp_path) -> None:
    chunk = make_chunk()
    payload = {"paper_uid": "p", "paper": None, "chunks": [{**chunk.model_dump(), "chunk_id": chunk.chunk_id}]}
    (tmp_path / "p.json").write_text(json.dumps(payload), encoding="utf-8")
    assert load_chunk_corpus(tmp_path) == [chunk]
    assert load_chunk_corpus(tmp_path / "none") == []


def test_load_chunk_corpus_rejects_bad_files(tmp_path) -> None:
    (tmp_path / "bad.json").write_text("nope", encoding="utf-8")
    with pytest.raises(DatasetError, match="bad.json"):
        load_chunk_corpus(tmp_path)