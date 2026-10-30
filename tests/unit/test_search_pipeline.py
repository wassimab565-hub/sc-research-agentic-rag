import json

import pytest

from sc_research_agentic_rag.ingestion import IngestionError, Paper, PaperSource, cli
from sc_research_agentic_rag.ingestion.service import SearchResult, collect_papers, save_result


def make_paper(source: PaperSource, source_id: str, title: str, **kwargs) -> Paper:
    return Paper(source=source, source_id=source_id, title=title, url=f"https://x/{source_id}", **kwargs)


class FakeSource:
    def __init__(self, papers: list[Paper] | None = None, error: str | None = None) -> None:
        self.papers = papers or []
        self.error = error
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, max_results: int) -> list[Paper]:
        self.calls.append((query, max_results))
        if self.error:
            raise IngestionError(self.error)
        return self.papers


def test_collect_merges_sources_and_passes_arguments() -> None:
    arxiv = FakeSource([make_paper(PaperSource.ARXIV, "1", "Same Title", abstract="abs")])
    openalex = FakeSource([make_paper(PaperSource.OPENALEX, "W1", "Same Title", doi="10.1/x")])
    result = collect_papers("graphrag", {"arxiv": arxiv, "openalex": openalex}, max_results=7)
    assert arxiv.calls == [("graphrag", 7)]
    assert openalex.calls == [("graphrag", 7)]
    assert len(result.papers) == 1
    assert result.papers[0].doi == "10.1/x"
    assert result.papers[0].also_found_as == ["openalex:W1"]
    assert result.errors == {}


def test_collect_records_failing_source_and_keeps_the_rest() -> None:
    ok = FakeSource([make_paper(PaperSource.ARXIV, "1", "A")])
    bad = FakeSource(error="boom")
    result = collect_papers("x", {"arxiv": ok, "openalex": bad})
    assert [p.source_id for p in result.papers] == ["1"]
    assert result.errors == {"openalex": "boom"}


def test_save_result_writes_json(tmp_path) -> None:
    paper = make_paper(PaperSource.ARXIV, "1", "A", doi="10.1/x")
    path = save_result(SearchResult("Graph RAG?", [paper], {"openalex": "boom"}), tmp_path / "nested")
    assert path.name.startswith("graph-rag_")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["query"] == "Graph RAG?"
    assert data["errors"] == {"openalex": "boom"}
    assert data["papers"][0]["source_id"] == "1"
    assert data["papers"][0]["doi"] == "10.1/x"


def test_save_result_slug_fallback(tmp_path) -> None:
    path = save_result(SearchResult("???", []), tmp_path)
    assert path.name.startswith("query_")


def test_cli_prints_and_saves(monkeypatch, tmp_path, capsys) -> None:
    source = FakeSource([make_paper(PaperSource.ARXIV, "1", "A Paper", authors=["Ann"])])
    monkeypatch.setattr(cli, "build_sources", lambda names, key: {"arxiv": source})
    code = cli.main(["graph rag", "--source", "arxiv", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "[arxiv:1]" in out
    assert "A Paper" in out
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cli_returns_1_when_all_sources_fail(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr(cli, "build_sources", lambda names, key: {"arxiv": FakeSource(error="boom")})
    code = cli.main(["x", "--source", "arxiv", "--output-dir", str(tmp_path)])
    assert code == 1
    assert "boom" in capsys.readouterr().err
    assert list(tmp_path.glob("*.json")) == []


def test_cli_rejects_invalid_max_results() -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(["x", "--max-results", "0"])
    assert exc.value.code == 2


def test_build_sources_creates_requested_clients() -> None:
    assert set(cli.build_sources(["arxiv", "openalex"], "key")) == {"arxiv", "openalex"}
    assert set(cli.build_sources(["arxiv"], None)) == {"arxiv"}