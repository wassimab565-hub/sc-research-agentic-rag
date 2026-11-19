import json

import pytest

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.graph import concept_cli as cli
from sc_research_agentic_rag.graph.extractor import cache_path, extract_concepts, load_concepts, read_cache
from sc_research_agentic_rag.llm import LLMError

PROSE = (
    "Dense retrieval encodes questions and passages into vectors. The similarity is computed with a dot "
    "product, and the nearest passages are returned to the reader model for answer extraction. "
) * 4
ANSWER = '["dense retrieval", "dot product", "unicorn"]'


def make_chunk(paper: str, index: int, text: str = PROSE, page: int = 1) -> Chunk:
    start = index * 1000
    return Chunk(paper_uid=paper, index=index, page=page, text=text, char_start=start, char_end=start + len(text))


class FakeLLM:
    def __init__(self, answer: str = ANSWER, fail_on: int | None = None) -> None:
        self.answer, self.fail_on, self.calls = answer, fail_on, 0

    def chat(self, messages):
        self.calls += 1
        if self.fail_on == self.calls:
            raise LLMError("down")
        return self.answer


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list = []
        self.closed = False

    def ensure_schema(self) -> None:
        pass

    def upsert_chunk_concepts(self, paper_uid, chunk_id, page, concepts) -> None:
        self.calls.append((paper_uid, chunk_id, page, concepts))

    def counts(self) -> dict[str, int]:
        return {"Concept": 2}

    def close(self) -> None:
        self.closed = True


def test_extraction_saves_validated_concepts(tmp_path) -> None:
    llm = FakeLLM()
    stats = extract_concepts([make_chunk("a", 0), make_chunk("a", 1)], llm, tmp_path)
    assert (stats.extracted, stats.cached, llm.calls) == (2, 0, 2)
    cache = read_cache(cache_path(tmp_path, "a"))
    assert cache["a#1"]["concepts"] == ["dense retrieval", "dot product"]


def test_rerun_uses_the_cache_and_changed_text_is_extracted_again(tmp_path) -> None:
    chunks = [make_chunk("a", 0)]
    extract_concepts(chunks, FakeLLM(), tmp_path)
    llm = FakeLLM()
    assert extract_concepts(chunks, llm, tmp_path).cached == 1
    assert llm.calls == 0
    changed = [make_chunk("a", 0, text=PROSE + " Extra sentence.")]
    assert extract_concepts(changed, llm, tmp_path).extracted == 1


def test_short_chunks_are_filtered_without_calling_the_llm(tmp_path) -> None:
    llm = FakeLLM()
    stats = extract_concepts([make_chunk("a", 0, "too short")], llm, tmp_path)
    assert (stats.filtered, llm.calls) == (1, 0)


def test_limit_bounds_the_number_of_llm_calls(tmp_path) -> None:
    llm = FakeLLM()
    stats = extract_concepts([make_chunk("a", i) for i in range(3)], llm, tmp_path, limit=2)
    assert (stats.extracted, llm.calls) == (2, 2)


def test_unparseable_answer_counts_as_failed_and_is_not_cached(tmp_path) -> None:
    stats = extract_concepts([make_chunk("a", 0)], FakeLLM("not json"), tmp_path)
    assert stats.failed == 1
    assert not cache_path(tmp_path, "a").exists()


def test_llm_error_keeps_earlier_progress(tmp_path) -> None:
    with pytest.raises(LLMError):
        extract_concepts([make_chunk("a", 0), make_chunk("a", 1)], FakeLLM(fail_on=2), tmp_path)
    assert list(read_cache(cache_path(tmp_path, "a"))) == ["a#0"]


def test_load_only_sends_cached_chunks_with_matching_text(tmp_path) -> None:
    chunks = [make_chunk("a", 0), make_chunk("a", 1)]
    extract_concepts(chunks[:1], FakeLLM(), tmp_path)
    graph = FakeGraph()
    assert load_concepts(chunks, tmp_path, graph) == 1
    assert graph.calls == [("a", "a#0", 1, ["dense retrieval", "dot product"])]


def write_chunks(directory, paper: str, count: int) -> None:
    chunks = [make_chunk(paper, i) for i in range(count)]
    payload = {"paper_uid": paper, "paper": None, "chunks": [{**c.model_dump(), "chunk_id": c.chunk_id} for c in chunks]}
    (directory / f"{paper}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_cli_extracts_then_loads(monkeypatch, tmp_path, capsys) -> None:
    write_chunks(tmp_path, "a", 3)
    graph = FakeGraph()
    monkeypatch.setattr(cli, "build_llm", lambda: FakeLLM())
    monkeypatch.setattr(cli, "build_store", lambda: graph)
    base = ["--chunks-dir", str(tmp_path), "--cache-dir", str(tmp_path / "cache")]

    assert cli.main([*base, "--limit", "2"]) == 0
    assert "2 extracted" in capsys.readouterr().out
    assert cli.main([*base, "--load"]) == 0
    assert "2 chunks loaded" in capsys.readouterr().out
    assert graph.closed


def test_cli_reports_llm_errors_and_empty_corpus(monkeypatch, tmp_path, capsys) -> None:
    write_chunks(tmp_path, "a", 2)
    monkeypatch.setattr(cli, "build_llm", lambda: FakeLLM(fail_on=1))
    args = ["--chunks-dir", str(tmp_path), "--cache-dir", str(tmp_path / "cache")]
    assert cli.main(args) == 1
    assert "resume" in capsys.readouterr().err
    assert cli.main([*args, "--paper", "zzz"]) == 1
    assert "no chunks" in capsys.readouterr().err


def test_cli_rejects_invalid_limit() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["--limit", "0"])
    assert exc.value.code == 2