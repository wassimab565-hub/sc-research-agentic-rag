import httpx
import pytest

from sc_research_agentic_rag.document_processing import DownloadStatus, PdfDownloader
from sc_research_agentic_rag.document_processing.downloader import pdf_filename
from sc_research_agentic_rag.ingestion.http import USER_AGENT, RateLimiter
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource

PDF_BYTES = b"%PDF-1.4\n%fake pdf body\n"


def make_paper(source_id: str = "2401.00001", pdf_url: str | None = "https://x/p.pdf") -> Paper:
    return Paper(
        source=PaperSource.ARXIV,
        source_id=source_id,
        title="T",
        url=f"https://arxiv.org/abs/{source_id}",
        pdf_url=pdf_url,
    )


def make_downloader(handler, tmp_path, **kwargs) -> PdfDownloader:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return PdfDownloader(output_dir=tmp_path, client=client, **kwargs)


def pdf_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, content=PDF_BYTES)


@pytest.mark.parametrize(
    "uid,expected",
    [
        ("arxiv:2401.00001", "arxiv_2401.00001.pdf"),
        ("arxiv:hep-th/9901001", "arxiv_hep-th_9901001.pdf"),
        ("openalex:W111", "openalex_W111.pdf"),
    ],
)
def test_pdf_filename(uid: str, expected: str) -> None:
    assert pdf_filename(uid) == expected


def test_downloads_and_writes_file(tmp_path) -> None:
    result = make_downloader(pdf_ok, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.DOWNLOADED
    assert result.path == tmp_path / "arxiv_2401.00001.pdf"
    assert result.path.read_bytes() == PDF_BYTES
    assert list(tmp_path.glob("*.part")) == []


def test_creates_output_dir(tmp_path) -> None:
    target = tmp_path / "nested" / "pdfs"
    result = make_downloader(pdf_ok, target).download(make_paper())
    assert result.path.exists()


def test_existing_file_is_not_downloaded_again(tmp_path) -> None:
    (tmp_path / "arxiv_2401.00001.pdf").write_bytes(PDF_BYTES)
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=PDF_BYTES)

    result = make_downloader(handler, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.EXISTS
    assert calls == []


def test_empty_existing_file_is_replaced(tmp_path) -> None:
    (tmp_path / "arxiv_2401.00001.pdf").write_bytes(b"")
    result = make_downloader(pdf_ok, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.DOWNLOADED
    assert result.path.read_bytes() == PDF_BYTES


def test_paper_without_pdf_url_is_skipped(tmp_path) -> None:
    calls: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(200, content=PDF_BYTES)

    result = make_downloader(handler, tmp_path).download(make_paper(pdf_url=None))
    assert result.status == DownloadStatus.NO_PDF_URL
    assert calls == []


def test_non_pdf_response_is_rejected(tmp_path) -> None:
    handler = lambda request: httpx.Response(200, content=b"<html>paywall</html>")  # noqa: E731
    result = make_downloader(handler, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.FAILED
    assert "not a PDF" in result.message
    assert list(tmp_path.iterdir()) == []


def test_oversized_pdf_is_rejected(tmp_path) -> None:
    result = make_downloader(pdf_ok, tmp_path, max_bytes=10).download(make_paper())
    assert result.status == DownloadStatus.FAILED
    assert list(tmp_path.iterdir()) == []


def test_http_error_is_reported_not_raised(tmp_path) -> None:
    handler = lambda request: httpx.Response(404)  # noqa: E731
    result = make_downloader(handler, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.FAILED
    assert result.path is None


def test_network_error_is_reported_not_raised(tmp_path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    result = make_downloader(handler, tmp_path).download(make_paper())
    assert result.status == DownloadStatus.FAILED


def test_retries_on_503_then_succeeds(tmp_path) -> None:
    responses = [httpx.Response(503), httpx.Response(200, content=PDF_BYTES)]
    result = make_downloader(lambda request: responses.pop(0), tmp_path).download(make_paper())
    assert result.status == DownloadStatus.DOWNLOADED


def test_url_query_string_and_user_agent_are_preserved(tmp_path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PDF_BYTES)

    make_downloader(handler, tmp_path).download(make_paper(pdf_url="https://x/p.pdf?download=1"))
    assert str(seen[0].url) == "https://x/p.pdf?download=1"
    assert seen[0].headers["user-agent"] == USER_AGENT


def test_default_client_follows_redirects(tmp_path) -> None:
    assert PdfDownloader(output_dir=tmp_path)._client.follow_redirects is True


def test_download_many_keeps_order_and_waits_between_downloads(tmp_path) -> None:
    now = [0.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=fake_sleep)
    papers = [make_paper("1"), make_paper("2", pdf_url=None), make_paper("3")]
    results = make_downloader(pdf_ok, tmp_path, rate_limiter=limiter).download_many(papers)
    assert [r.status for r in results] == [
        DownloadStatus.DOWNLOADED,
        DownloadStatus.NO_PDF_URL,
        DownloadStatus.DOWNLOADED,
    ]
    assert sleeps == [3.0]