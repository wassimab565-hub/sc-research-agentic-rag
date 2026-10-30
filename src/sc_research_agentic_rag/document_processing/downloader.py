import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from sc_research_agentic_rag.ingestion.http import (
    USER_AGENT,
    RateLimiter,
    default_sleep,
    get_with_retry,
)
from sc_research_agentic_rag.ingestion.models import Paper

DEFAULT_PDF_DIR = Path("data/raw/downloaded_papers")
MIN_INTERVAL = 3.0
MAX_PDF_BYTES = 50 * 1024 * 1024
_PDF_MAGIC = b"%PDF-"


class DownloadStatus(StrEnum):
    DOWNLOADED = "downloaded"
    EXISTS = "exists"
    NO_PDF_URL = "no_pdf_url"
    FAILED = "failed"


@dataclass(frozen=True)
class DownloadResult:
    paper_uid: str
    status: DownloadStatus
    path: Path | None = None
    message: str | None = None


def pdf_filename(paper_uid: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", paper_uid) + ".pdf"


class PdfDownloader:
    def __init__(
        self,
        output_dir: Path = DEFAULT_PDF_DIR,
        client: httpx.Client | None = None,
        rate_limiter: RateLimiter | None = None,
        max_retries: int = 3,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = default_sleep,
        max_bytes: int = MAX_PDF_BYTES,
    ) -> None:
        self._output_dir = output_dir
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True)
        self._rate_limiter = rate_limiter or RateLimiter(MIN_INTERVAL, sleep=sleep)
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep
        self._max_bytes = max_bytes

    def download(self, paper: Paper) -> DownloadResult:
        if not paper.pdf_url:
            return DownloadResult(paper.uid, DownloadStatus.NO_PDF_URL)

        path = self._output_dir / pdf_filename(paper.uid)
        if path.exists() and path.stat().st_size > 0:
            return DownloadResult(paper.uid, DownloadStatus.EXISTS, path)

        try:
            response = get_with_retry(
                self._client,
                paper.pdf_url,
                headers={"User-Agent": USER_AGENT},
                max_retries=self._max_retries,
                backoff=self._backoff,
                sleep=self._sleep,
                before_request=self._rate_limiter.wait,
            )
        except httpx.HTTPError as exc:
            return self._failed(paper, f"download failed: {exc}")

        content = response.content
        if not content.startswith(_PDF_MAGIC):
            return self._failed(paper, "response is not a PDF")
        if len(content) > self._max_bytes:
            return self._failed(paper, f"PDF larger than {self._max_bytes} bytes")

        try:
            self._write(path, content)
        except OSError as exc:
            return self._failed(paper, f"could not write file: {exc}")
        return DownloadResult(paper.uid, DownloadStatus.DOWNLOADED, path)

    def download_many(self, papers: Iterable[Paper]) -> list[DownloadResult]:
        return [self.download(paper) for paper in papers]

    @staticmethod
    def _failed(paper: Paper, message: str) -> DownloadResult:
        return DownloadResult(paper.uid, DownloadStatus.FAILED, message=message)

    @staticmethod
    def _write(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(".pdf.part")
        partial.write_bytes(content)
        partial.replace(path)