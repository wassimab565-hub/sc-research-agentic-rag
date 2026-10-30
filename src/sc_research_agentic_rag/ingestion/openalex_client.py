from collections.abc import Callable
from datetime import date
from typing import Any

import httpx

from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.http import USER_AGENT, default_sleep, get_with_retry
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource

OPENALEX_API_URL = "https://api.openalex.org/works"


class OpenAlexClient:
    def __init__(
        self,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        base_url: str = OPENALEX_API_URL,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 1.0,
        sleep: Callable[[float], None] = default_sleep,
    ) -> None:
        self._api_key = api_key
        self._client = client or httpx.Client(timeout=timeout)
        self._base_url = base_url
        self._max_retries = max_retries
        self._backoff = backoff
        self._sleep = sleep

    def search(self, query: str, max_results: int = 10, page: int = 1) -> list[Paper]:
        if not query.strip():
            raise ValueError("query must not be empty")
        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")
        if page < 1:
            raise ValueError("page must be >= 1")

        headers = {"User-Agent": USER_AGENT}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        params = {"search": query, "per_page": max_results, "page": page}
        try:
            response = get_with_retry(
                self._client,
                self._base_url,
                params=params,
                headers=headers,
                max_retries=self._max_retries,
                backoff=self._backoff,
                sleep=self._sleep,
            )
            payload = response.json()
        except httpx.HTTPError as exc:
            raise IngestionError(f"OpenAlex request failed: {exc}") from exc
        except ValueError as exc:
            raise IngestionError(f"invalid OpenAlex response: {exc}") from exc

        results = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(results, list):
            raise IngestionError("invalid OpenAlex response: missing 'results'")

        papers = (self._parse_work(work) for work in results)
        return [paper for paper in papers if paper is not None]

    @staticmethod
    def _parse_work(work: dict[str, Any]) -> Paper | None:
        title = work.get("title") or work.get("display_name")
        openalex_url = work.get("id")
        if not title or not openalex_url:
            return None

        primary = work.get("primary_location") or {}
        best_oa = work.get("best_oa_location") or {}
        published = work.get("publication_date")

        return Paper(
            source=PaperSource.OPENALEX,
            source_id=openalex_url.rsplit("/", 1)[-1],
            title=title,
            authors=[
                name
                for a in work.get("authorships") or []
                if (name := (a.get("author") or {}).get("display_name"))
            ],
            abstract=rebuild_abstract(work.get("abstract_inverted_index")),
            published=date.fromisoformat(published) if published else None,
            doi=work.get("doi"),
            url=openalex_url,
            pdf_url=best_oa.get("pdf_url") or primary.get("pdf_url"),
            categories=[
                name for t in work.get("topics") or [] if (name := t.get("display_name"))
            ],
        )


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    if not inverted_index:
        return None
    positions = {pos: word for word, indexes in inverted_index.items() for pos in indexes}
    return " ".join(positions[i] for i in sorted(positions))