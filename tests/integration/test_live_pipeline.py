import json
import os

import pytest

from sc_research_agentic_rag.document_processing import PdfDownloader, ProcessStatus, process_papers
from sc_research_agentic_rag.document_processing.pipeline import chunks_path
from sc_research_agentic_rag.ingestion import ArxivClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call the real APIs",
)


def test_pipeline_on_real_arxiv_papers(tmp_path) -> None:
    papers = ArxivClient().search("dense passage retrieval open-domain question answering", max_results=2)
    downloader = PdfDownloader(output_dir=tmp_path / "pdfs")
    results = process_papers(papers, downloader, tmp_path / "chunks")

    processed = [r for r in results if r.status == ProcessStatus.PROCESSED]
    assert processed, [r.message for r in results]
    data = json.loads(chunks_path(tmp_path / "chunks", processed[0].paper_uid).read_text(encoding="utf-8"))
    assert data["paper"]["url"].startswith("https://arxiv.org/abs/")
    assert data["chunks"] and all(c["text"].strip() for c in data["chunks"])