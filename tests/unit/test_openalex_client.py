import json
from datetime import date
from pathlib import Path

import httpx
import pytest

from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.models import PaperSource
from sc_research_agentic_rag.ingestion.openalex_client import OpenAlexClient, rebuild_abstract

FIXTURE = Path(__file__).parent.parent / "fixtures" / "openalex_sample.json"


def make_client(handler, api_key: str | None = None) -> OpenAlexClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OpenAlexClient(api_key=api_key, client=http)


def ok(payload):
    return lambda request: httpx.Response(200, json=payload)


@pytest.fixture
def sample() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_parses_full_work(sample) -> None:
    papers = make_client(ok(sample)).search("graphrag")
    paper = papers[0]
    assert paper.source == PaperSource.OPENALEX
    assert paper.source_id == "W111"
    assert paper.title == "Improving Retrieval Quality in GraphRAG"
    assert paper.authors == ["Alice Martin", "Bob Karim"]
    assert paper.abstract == "We study retrieval. It works."
    assert paper.published == date(2024, 1, 2)
    assert paper.doi == "10.1000/abc.123"
    assert paper.url == "https://openalex.org/W111"
    assert paper.pdf_url == "https://example.org/paper.pdf"
    assert paper.categories == ["Information Retrieval", "Knowledge Graphs"]


def test_sparse_work_uses_fallbacks_and_untitled_is_skipped(sample) -> None:
    papers = make_client(ok(sample)).search("x")
    assert len(papers) == 2
    sparse = papers[1]
    assert sparse.title == "Fallback Display Name"
    assert sparse.abstract is None
    assert sparse.doi is None
    assert sparse.published is None
    assert sparse.pdf_url is None


def test_sends_params_and_bearer_header() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["params"] = dict(request.url.params)
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    make_client(handler, api_key="secret").search("graphrag", max_results=5, page=2)
    assert seen["params"] == {"search": "graphrag", "per_page": "5", "page": "2"}
    assert seen["auth"] == "Bearer secret"
    assert "secret" not in str(seen["params"])


def test_no_auth_header_without_key() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"results": []})

    make_client(handler).search("x")
    assert seen["auth"] is None


def test_http_error_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError):
        make_client(lambda request: httpx.Response(429)).search("x")


def test_network_error_raises_ingestion_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(IngestionError):
        make_client(handler).search("x")


@pytest.mark.parametrize("payload", [{"meta": {}}, ["not", "a", "dict"]])
def test_unexpected_payload_raises_ingestion_error(payload) -> None:
    with pytest.raises(IngestionError):
        make_client(ok(payload)).search("x")


def test_invalid_json_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError):
        make_client(lambda request: httpx.Response(200, text="not json")).search("x")


def test_rebuild_abstract_orders_by_position() -> None:
    assert rebuild_abstract({"b": [1], "a": [0], "c": [2, 3]}) == "a b c c"
    assert rebuild_abstract(None) is None
    assert rebuild_abstract({}) is None


@pytest.mark.parametrize(
    "kwargs",
    [{"query": " "}, {"query": "x", "max_results": 0}, {"query": "x", "max_results": 101}, {"query": "x", "page": 0}],
)
def test_invalid_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        make_client(ok({"results": []})).search(**kwargs)