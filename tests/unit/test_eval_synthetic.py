import json

import pytest

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation import generate_questions, load_dataset
from sc_research_agentic_rag.evaluation import generate_cli
from sc_research_agentic_rag.evaluation.synthetic import clean_question, copies_passage, is_candidate
from sc_research_agentic_rag.llm import LLMError

PROSE = (
    "Dense retrieval encodes questions and passages into vectors. The similarity is computed with a dot "
    "product, and the nearest passages are returned to the reader model for answer extraction. "
) * 4


def make_chunk(paper: str, index: int, text: str = PROSE, page: int = 1) -> Chunk:
    start = index * 1000
    return Chunk(paper_uid=paper, index=index, page=page, text=text, char_start=start, char_end=start + len(text))


class FakeLLM:
    def __init__(self, answers=None) -> None:
        self.answers = answers
        self.calls: list[list] = []

    def chat(self, messages):
        self.calls.append(messages)
        if self.answers is None:
            return f"What does topic number {len(self.calls)} describe in detail?"
        return self.answers[(len(self.calls) - 1) % len(self.answers)]


def test_candidate_filter() -> None:
    assert is_candidate(make_chunk("a", 0))
    assert not is_candidate(make_chunk("a", 0, "short text"))
    references = "Smith et al. 2019. Title. Proceedings of X, 2020. " * 12
    assert not is_candidate(make_chunk("a", 0, references))
    assert not is_candidate(make_chunk("a", 0, "contact: a@b.com and c@d.org " + PROSE))
    assert not is_candidate(make_chunk("a", 0, "1 2 3 4 5 6 7 8 9 0 " * 40))


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("What is the goal of the system?", "What is the goal of the system?"),
        ('Question: "What is the goal of the system?"', "What is the goal of the system?"),
        ("\n\n**What is the goal of the system?**\nExplanation follows.", "What is the goal of the system?"),
        ("What is the goal of the system? And what is its cost?", "What is the goal of the system?"),
        ("Statement without question mark", None),
        ("Why?", None),
        ("", None),
        ("x" * 400 + "?", None),
    ],
)
def test_clean_question(raw: str, expected) -> None:
    assert clean_question(raw) == expected


def test_verbatim_copies_are_detected() -> None:
    passage = "the similarity is computed with a dot product between vectors"
    assert copies_passage("What is computed with a dot product between vectors?", passage)
    assert not copies_passage("How is the similarity computed with a dot product?", passage)
    assert not copies_passage("Which operation compares two embeddings?", passage)
    assert not copies_passage("Short?", passage)


def test_generation_is_balanced_across_papers_and_stops_at_count() -> None:
    chunks = [make_chunk("a", i) for i in range(10)] + [make_chunk("b", i) for i in range(10)]
    questions = list(generate_questions(chunks, FakeLLM(), count=6))
    assert len(questions) == 6
    assert [q.id for q in questions] == [f"syn-{n:04d}" for n in range(1, 7)]
    papers = [q.evidence[0].paper_uid for q in questions]
    assert papers.count("a") == papers.count("b") == 3
    assert all(q.source == "synthetic" for q in questions)


def test_evidence_points_to_the_source_chunk() -> None:
    chunk = make_chunk("a", 3, page=7)
    [question] = list(generate_questions([chunk], FakeLLM(), count=1))
    evidence = question.evidence[0]
    assert (evidence.paper_uid, evidence.page) == ("a", 7)
    assert (evidence.char_start, evidence.char_end) == (chunk.char_start, chunk.char_end)


def test_prompt_contains_the_passage_and_rules() -> None:
    llm = FakeLLM()
    list(generate_questions([make_chunk("a", 0)], llm, count=1))
    system, user = llm.calls[0]
    assert system.role == "system"
    assert PROSE[:60] in user.content
    assert "same language" in user.content


def test_same_seed_gives_the_same_questions() -> None:
    chunks = [make_chunk("a", i) for i in range(8)]
    first = [q.evidence[0].char_start for q in generate_questions(chunks, FakeLLM(), count=4, seed=1)]
    again = [q.evidence[0].char_start for q in generate_questions(chunks, FakeLLM(), count=4, seed=1)]
    other = [q.evidence[0].char_start for q in generate_questions(chunks, FakeLLM(), count=4, seed=2)]
    assert first == again
    assert first != other


def test_bad_answers_are_skipped_and_attempts_are_bounded() -> None:
    chunks = [make_chunk("a", i) for i in range(20)]
    copied = "Is it true that the nearest passages are returned to the reader model for answer extraction?"
    llm = FakeLLM(["no question here", copied])
    assert list(generate_questions(chunks, llm, count=3)) == []
    assert len(llm.calls) == 9


def test_fewer_candidates_than_requested() -> None:
    chunks = [make_chunk("a", 0), make_chunk("a", 1, "too short")]
    assert len(list(generate_questions(chunks, FakeLLM(), count=5))) == 1


def test_invalid_count_and_llm_errors_propagate() -> None:
    with pytest.raises(ValueError):
        list(generate_questions([make_chunk("a", 0)], FakeLLM(), count=0))

    class Broken:
        def chat(self, messages):
            raise LLMError("down")

    with pytest.raises(LLMError):
        list(generate_questions([make_chunk("a", 0)], Broken(), count=1))


def write_chunks(directory, paper: str, count: int) -> None:
    chunks = [make_chunk(paper, i) for i in range(count)]
    payload = {"paper_uid": paper, "paper": None, "chunks": [{**c.model_dump(), "chunk_id": c.chunk_id} for c in chunks]}
    (directory / f"{paper}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cli_writes_a_loadable_dataset(monkeypatch, tmp_path, capsys) -> None:
    write_chunks(tmp_path, "a", 5)
    monkeypatch.setattr(generate_cli, "build_client", lambda: FakeLLM())
    output = tmp_path / "out" / "synthetic.jsonl"
    assert generate_cli.main(["--chunks-dir", str(tmp_path), "--count", "3", "--output", str(output)]) == 0
    assert "3 questions saved" in capsys.readouterr().out
    assert len(load_dataset(output)) == 3


def test_cli_keeps_partial_output_when_the_llm_fails(monkeypatch, tmp_path, capsys) -> None:
    write_chunks(tmp_path, "a", 5)

    class FailsSecond(FakeLLM):
        def chat(self, messages):
            if len(self.calls) == 1:
                raise LLMError("Ollama stopped")
            return super().chat(messages)

    monkeypatch.setattr(generate_cli, "build_client", lambda: FailsSecond())
    output = tmp_path / "synthetic.jsonl"
    assert generate_cli.main(["--chunks-dir", str(tmp_path), "--count", "5", "--output", str(output)]) == 1
    assert "1 questions were saved" in capsys.readouterr().err
    assert len(load_dataset(output)) == 1


def test_cli_errors(tmp_path, capsys) -> None:
    assert generate_cli.main(["--chunks-dir", str(tmp_path)]) == 1
    assert "no chunks" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        generate_cli.main(["--count", "0"])