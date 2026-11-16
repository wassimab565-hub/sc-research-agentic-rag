import json

from sc_research_agentic_rag.graph import build_graph, read_chunk_papers
from sc_research_agentic_rag.graph import cli
from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource


def make_paper(source_id: str) -> Paper:
    return Paper(
        source=PaperSource.ARXIV, source_id=source_id, title=f"Paper {source_id}",
        url=f"https://x/{source_id}", authors=["Ann"], categories=["cs.IR"],
    )


def write_chunk_file(directory, name: str, paper: Paper | None) -> None:
    payload = {"paper_uid": name, "paper": paper.model_dump(mode="json") if paper else None, "chunks": []}
    (directory / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")


class FakeGraph:
    def __init__(self, error=None) -> None:
        self.calls: list = []
        self.error = error
        self.closed = False

    def ensure_schema(self) -> None:
        self.calls.append("schema")

    def clear(self) -> None:
        self.calls.append("clear")

    def upsert_paper(self, paper) -> None:
        if self.error:
            raise self.error
        self.calls.append(paper.uid)

    def counts(self) -> dict[str, int]:
        return {"Paper": 2}

    def close(self) -> None:
        self.closed = True


def make_dir(tmp_path):
    write_chunk_file(tmp_path, "a", make_paper("1"))
    write_chunk_file(tmp_path, "b", make_paper("2"))
    write_chunk_file(tmp_path, "upload", None)
    (tmp_path / "bad.json").write_text("nope", encoding="utf-8")
    (tmp_path / "list.json").write_text("[1]", encoding="utf-8")
    return tmp_path


def test_read_chunk_papers_skips_uploads_and_reports_bad_files(tmp_path) -> None:
    papers, uploads, failed = read_chunk_papers(make_dir(tmp_path))
    assert [p.uid for p in papers] == ["arxiv:1", "arxiv:2"]
    assert uploads == 1
    assert set(failed) == {"bad.json", "list.json"}


def test_build_graph_writes_schema_then_papers(tmp_path) -> None:
    graph = FakeGraph()
    result = build_graph(make_dir(tmp_path), graph)
    assert graph.calls == ["schema", "arxiv:1", "arxiv:2"]
    assert (result.papers, result.uploads_skipped, len(result.failed)) == (2, 1, 2)


def test_force_clears_before_writing(tmp_path) -> None:
    graph = FakeGraph()
    build_graph(make_dir(tmp_path), graph, force=True)
    assert graph.calls[:2] == ["schema", "clear"]


def test_cli_prints_summary(monkeypatch, tmp_path, capsys) -> None:
    graph = FakeGraph()
    monkeypatch.setattr(cli, "build_store", lambda: graph)
    assert cli.main(["--chunks-dir", str(make_dir(tmp_path))]) == 0
    captured = capsys.readouterr()
    assert "2 papers written, 1 uploads skipped" in captured.out
    assert "Paper: 2" in captured.out
    assert "bad.json" in captured.err
    assert graph.closed


def test_cli_reports_graph_errors(monkeypatch, tmp_path, capsys) -> None:
    graph = FakeGraph(error=GraphError("down"))
    monkeypatch.setattr(cli, "build_store", lambda: graph)
    assert cli.main(["--chunks-dir", str(make_dir(tmp_path))]) == 1
    assert "down" in capsys.readouterr().err
    assert graph.closed


def test_cli_without_chunk_files(tmp_path, capsys) -> None:
    assert cli.main(["--chunks-dir", str(tmp_path)]) == 1
    assert "no chunk files" in capsys.readouterr().err