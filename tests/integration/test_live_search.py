import os

import pytest
from dotenv import load_dotenv

from sc_research_agentic_rag.ingestion import ArxivClient, OpenAlexClient

load_dotenv()

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TESTS") != "1",
    reason="set RUN_LIVE_TESTS=1 to call the real APIs",
)


def test_arxiv_live() -> None:
    papers = ArxivClient().search("retrieval augmented generation", max_results=3)
    assert 1 <= len(papers) <= 3
    assert all(p.title and p.url.startswith("https://arxiv.org/abs/") for p in papers)


@pytest.mark.skipif(not os.environ.get("OPENALEX_API_KEY"), reason="OPENALEX_API_KEY not set")
def test_openalex_live() -> None:
    papers = OpenAlexClient(api_key=os.environ["OPENALEX_API_KEY"]).search(
        "retrieval augmented generation", max_results=3
    )
    assert 1 <= len(papers) <= 3
    assert all(p.title and p.url.startswith("https://openalex.org/W") for p in papers)