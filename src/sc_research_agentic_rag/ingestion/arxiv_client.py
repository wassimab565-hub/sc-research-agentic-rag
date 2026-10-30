import re
import xml.etree.ElementTree as ET
from datetime import date

import httpx

from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource

ARXIV_API_URL = "https://export.arxiv.org/api/query"
_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
_VERSION_SUFFIX = re.compile(r"v\d+$")


class ArxivClient:
    def __init__(
        self,
        client: httpx.Client | None = None,
        base_url: str = ARXIV_API_URL,
        timeout: float = 30.0,
    ) -> None:
        self._client = client or httpx.Client(timeout=timeout)
        self._base_url = base_url

    def search(self, query: str, max_results: int = 10, start: int = 0) -> list[Paper]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

        params = {
            "search_query": query,
            "start": start,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        try:
            response = self._client.get(self._base_url, params=params)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise IngestionError(f"arXiv request failed: {exc}") from exc

        return self._parse_feed(response.text)

    def _parse_feed(self, xml_text: str) -> list[Paper]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            raise IngestionError(f"invalid arXiv response: {exc}") from exc

        papers: list[Paper] = []
        for entry in root.findall("atom:entry", _NS):
            entry_id = _text(entry, "atom:id")
            if "/api/errors" in entry_id:
                raise IngestionError(f"arXiv API error: {_text(entry, 'atom:summary')}")
            papers.append(self._parse_entry(entry, entry_id))
        return papers

    @staticmethod
    def _parse_entry(entry: ET.Element, entry_id: str) -> Paper:
        source_id = _VERSION_SUFFIX.sub("", entry_id.split("/abs/", 1)[-1])
        published = _text(entry, "atom:published")[:10]

        pdf_url = None
        for link in entry.findall("atom:link", _NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")

        return Paper(
            source=PaperSource.ARXIV,
            source_id=source_id,
            title=_text(entry, "atom:title"),
            authors=[_text(a, "atom:name") for a in entry.findall("atom:author", _NS)],
            abstract=_text(entry, "atom:summary") or None,
            published=date.fromisoformat(published) if published else None,
            doi=_text(entry, "arxiv:doi") or None,
            url=f"https://arxiv.org/abs/{source_id}",
            pdf_url=pdf_url,
            categories=[c.get("term", "") for c in entry.findall("atom:category", _NS)],
        )


def _text(element: ET.Element, path: str) -> str:
    node = element.find(path, _NS)
    return (node.text or "").strip() if node is not None else ""