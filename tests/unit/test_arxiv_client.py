from datetime import date
from pathlib import Path

import httpx
import pytest

from sc_research_agentic_rag.ingestion.arxiv_client import ArxivClient
from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.models import PaperSource

FIXTURE = Path(__file__).parent.parent / "fixtures" / "arxiv_sample.xml"
EMPTY_FEED = '<feed xmlns="http://www.w3.org/2005/Atom"></feed>'
ERROR_FEED = """<feed xmlns="http://www.w3.org/2005/Atom"><entry>
<id>http://arxiv.org/api/errors#bad_query</id><title>Error</title>
<summary>malformed query</summary></entry></feed>"""


def make_client(handler) -> ArxivClient:
    return ArxivClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def ok(text: str):
    return lambda request: httpx.Response(200, text=text)


def test_parses_new_style_entry() -> None:
    papers = make_client(ok(FIXTURE.read_text(encoding="utf-8"))).search("graphrag")
    paper = papers[0]
    assert paper.source == PaperSource.ARXIV
    assert paper.source_id == "2401.00001"
    assert paper.title == "Improving Retrieval Quality in GraphRAG"
    assert paper.abstract == "We study retrieval. It works."
    assert paper.authors == ["Alice Martin", "Bob Karim"]
    assert paper.published == date(2024, 1, 2)
    assert paper.doi == "10.1000/abc.123"
    assert paper.url == "https://arxiv.org/abs/2401.00001"
    assert paper.pdf_url == "http://arxiv.org/pdf/2401.00001v2"
    assert paper.categories == ["cs.IR", "cs.CL"]


def test_parses_old_style_identifier_without_optional_fields() -> None:
    papers = make_client(ok(FIXTURE.read_text(encoding="utf-8"))).search("x")
    paper = papers[1]
    assert paper.source_id == "hep-th/9901001"
    assert paper.doi is None
    assert paper.pdf_url is None


def test_sends_expected_query_params() -> None:
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.url.params)
        return httpx.Response(200, text=EMPTY_FEED)

    make_client(handler).search("all:graphrag", max_results=5, start=10)
    assert seen["search_query"] == "all:graphrag"
    assert seen["max_results"] == "5"
    assert seen["start"] == "10"


def test_empty_feed_returns_empty_list() -> None:
    assert make_client(ok(EMPTY_FEED)).search("nothing") == []


def test_http_error_raises_ingestion_error() -> None:
    client = make_client(lambda request: httpx.Response(500))
    with pytest.raises(IngestionError):
        client.search("x")


def test_network_error_raises_ingestion_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(IngestionError):
        make_client(handler).search("x")


def test_invalid_xml_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError):
        make_client(ok("not xml")).search("x")


def test_api_error_entry_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError, match="malformed query"):
        make_client(ok(ERROR_FEED)).search("x")


@pytest.mark.parametrize("kwargs", [{"query": "  "}, {"query": "x", "max_results": 0}, {"query": "x", "max_results": 101}])
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        make_client(ok(EMPTY_FEED)).search(**kwargs)


def test_sends_user_agent_and_waits_between_requests() -> None:
    from sc_research_agentic_rag.ingestion.http import USER_AGENT, RateLimiter

    now = [0.0]
    sleeps: list[float] = []

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    agents: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        agents.append(request.headers["user-agent"])
        return httpx.Response(200, text=EMPTY_FEED)

    limiter = RateLimiter(3.0, clock=lambda: now[0], sleep=fake_sleep)
    client = ArxivClient(client=httpx.Client(transport=httpx.MockTransport(handler)), rate_limiter=limiter)
    client.search("a")
    client.search("b")
    assert agents == [USER_AGENT, USER_AGENT]
    assert sleeps == [3.0]


def test_retries_on_503_then_succeeds() -> None:
    responses = [httpx.Response(503), httpx.Response(200, text=EMPTY_FEED)]
    client = make_client(lambda request: responses.pop(0))
    assert client.search("x") == []