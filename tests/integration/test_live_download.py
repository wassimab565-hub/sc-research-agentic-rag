import os

import pytest

from sc_research_agentic_rag.document_processing import DownloadStatus, PdfDownloader
from sc_research_agentic_rag.ingestion import ArxivClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call the real APIs",
)


def test_download_one_arxiv_pdf_live(tmp_path) -> None:
    paper = ArxivClient().search("retrieval augmented generation", max_results=1)[0]
    result = PdfDownloader(output_dir=tmp_path).download(paper)
    assert result.status == DownloadStatus.DOWNLOADED
    assert result.path.read_bytes().startswith(b"%PDF-")