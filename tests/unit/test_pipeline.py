import json

import httpx
import pytest

from sc_research_agentic_rag.document_processing import (
    PdfDownloader,
    ProcessStatus,
    load_papers,
    process_papers,
    process_uploads,
)
from sc_research_agentic_rag.document_processing import cli
from sc_research_agentic_rag.document_processing.pipeline import chunks_path
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource
from sc_research_agentic_rag.ingestion.service import SearchResult, save_result

LONG_TEXT = " ".join(f"word{n}" for n in range(300))


def make_paper(source_id: str = "2401.00001", pdf_url: str | None = "https://x/p.pdf") -> Paper:
    return Paper(
        source=PaperSource.ARXIV,
        source_id=source_id,
        title=f"Paper {source_id}",
        url=f"https://arxiv.org/abs/{source_id}",
        pdf_url=pdf_url,
        doi="10.1/x",
    )


def make_downloader(tmp_path, handler) -> PdfDownloader:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return PdfDownloader(output_dir=tmp_path / "pdfs", client=client)


def serve(pdf_bytes: bytes):
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=pdf_bytes)

    handler.calls = calls
    return handler


def test_process_papers_end_to_end(tmp_path, write_pdf) -> None:
    pdf_bytes = write_pdf([LONG_TEXT, "Second page text"], "source.pdf").read_bytes()
    downloader = make_downloader(tmp_path, serve(pdf_bytes))
    papers = [make_paper("1"), make_paper("2", pdf_url=None)]

    results = process_papers(papers, downloader, tmp_path / "out", chunk_size=300, overlap=50)

    assert [r.status for r in results] == [ProcessStatus.PROCESSED, ProcessStatus.NO_PDF]
    assert results[0].chunks > 2
    data = json.loads(chunks_path(tmp_path / "out", "arxiv:1").read_text(encoding="utf-8"))
    assert data["paper_uid"] == "arxiv:1"
    assert data["paper"]["doi"] == "10.1/x"
    assert data["paper"]["url"] == "https://arxiv.org/abs/1"
    assert (data["chunk_size"], data["overlap"]) == (300, 50)
    assert len(data["chunks"]) == results[0].chunks
    first, last = data["chunks"][0], data["chunks"][-1]
    assert first["chunk_id"] == "arxiv:1#0"
    assert first["page"] == 1
    assert last["page"] == 2
    assert [c["index"] for c in data["chunks"]] == list(range(len(data["chunks"])))
    assert not chunks_path(tmp_path / "out", "arxiv:2").exists()


def test_existing_output_is_skipped_unless_forced(tmp_path, write_pdf) -> None:
    handler = serve(write_pdf([LONG_TEXT], "source.pdf").read_bytes())
    downloader = make_downloader(tmp_path, handler)
    out = tmp_path / "out"

    process_papers([make_paper("1")], downloader, out)
    calls_after_first = len(handler.calls)
    again = process_papers([make_paper("1")], downloader, out)
    assert again[0].status == ProcessStatus.EXISTS
    assert len(handler.calls) == calls_after_first

    forced = process_papers([make_paper("1")], downloader, out, force=True)
    assert forced[0].status == ProcessStatus.PROCESSED


def test_download_failure_does_not_stop_the_batch(tmp_path, write_pdf) -> None:
    pdf_bytes = write_pdf([LONG_TEXT], "source.pdf").read_bytes()

    def handler(request: httpx.Request) -> httpx.Response:
        if "bad" in str(request.url):
            return httpx.Response(404)
        return httpx.Response(200, content=pdf_bytes)

    papers = [make_paper("1", pdf_url="https://x/bad.pdf"), make_paper("2", pdf_url="https://x/ok.pdf")]
    results = process_papers(papers, make_downloader(tmp_path, handler), tmp_path / "out")
    assert [r.status for r in results] == [ProcessStatus.FAILED, ProcessStatus.PROCESSED]
    assert results[0].message


def test_pdf_without_text_fails_with_message(tmp_path, write_pdf) -> None:
    handler = serve(write_pdf(["", ""], "blank.pdf").read_bytes())
    results = process_papers([make_paper("1")], make_downloader(tmp_path, handler), tmp_path / "out")
    assert results[0].status == ProcessStatus.FAILED
    assert "no extractable text" in results[0].message


def test_process_uploads(tmp_path, write_pdf) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "good.pdf").write_bytes(write_pdf([LONG_TEXT], "tmp.pdf").read_bytes())
    (uploads / "broken.pdf").write_bytes(b"not a pdf")
    (uploads / "notes.txt").write_text("ignored")

    results = process_uploads(uploads, tmp_path / "out", chunk_size=300, overlap=50)

    assert [(r.paper_uid, r.status) for r in results] == [
        ("upload:broken", ProcessStatus.FAILED),
        ("upload:good", ProcessStatus.PROCESSED),
    ]
    data = json.loads(chunks_path(tmp_path / "out", "upload:good").read_text(encoding="utf-8"))
    assert data["paper"] is None
    assert data["chunks"][0]["paper_uid"] == "upload:good"


def test_process_uploads_missing_directory(tmp_path) -> None:
    assert process_uploads(tmp_path / "missing", tmp_path / "out") == []


def test_load_papers_roundtrip(tmp_path) -> None:
    papers = [make_paper("1"), make_paper("2", pdf_url=None)]
    path = save_result(SearchResult("q", papers), tmp_path)
    assert load_papers(path) == papers


@pytest.mark.parametrize("content", [None, "not json", '{"query": "x"}', "[1, 2]"])
def test_load_papers_invalid_file(tmp_path, content) -> None:
    path = tmp_path / "search.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError):
        load_papers(path)


def test_cli_from_search(monkeypatch, tmp_path, write_pdf, capsys) -> None:
    handler = serve(write_pdf([LONG_TEXT], "source.pdf").read_bytes())
    monkeypatch.setattr(cli, "build_downloader", lambda pdf_dir: make_downloader(tmp_path, handler))
    search = save_result(SearchResult("q", [make_paper("1"), make_paper("2"), make_paper("3")]), tmp_path)
    out = tmp_path / "out"

    code = cli.main(["--from-search", str(search), "--limit", "2", "--output-dir", str(out)])

    printed = capsys.readouterr().out
    assert code == 0
    assert "[processed] arxiv:1" in printed
    assert "arxiv:3" not in printed
    assert "processed: 2" in printed
    assert len(list(out.glob("*.json"))) == 2


def test_cli_uploads(tmp_path, write_pdf, capsys) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "doc.pdf").write_bytes(write_pdf([LONG_TEXT], "tmp.pdf").read_bytes())
    code = cli.main(["--uploads", "--uploads-dir", str(uploads), "--output-dir", str(tmp_path / "out")])
    assert code == 0
    assert "upload:doc" in capsys.readouterr().out


def test_cli_returns_1_when_everything_fails(tmp_path, capsys) -> None:
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    (uploads / "broken.pdf").write_bytes(b"nope")
    code = cli.main(["--uploads", "--uploads-dir", str(uploads), "--output-dir", str(tmp_path / "out")])
    assert code == 1
    assert "[failed" in capsys.readouterr().out


def test_cli_nothing_to_process(tmp_path, capsys) -> None:
    code = cli.main(["--uploads", "--uploads-dir", str(tmp_path / "none"), "--output-dir", str(tmp_path)])
    assert code == 1
    assert "nothing to process" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--uploads", "--from-search", "x.json"],
        ["--uploads", "--limit", "0"],
        ["--uploads", "--chunk-size", "0"],
        ["--uploads", "--overlap", "2000"],
        ["--from-search", "does-not-exist.json"],
    ],
)
def test_cli_rejects_invalid_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2