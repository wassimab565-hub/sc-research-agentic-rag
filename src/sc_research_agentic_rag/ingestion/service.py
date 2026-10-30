import json
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from sc_research_agentic_rag.ingestion.dedup import deduplicate
from sc_research_agentic_rag.ingestion.errors import IngestionError
from sc_research_agentic_rag.ingestion.models import Paper


class PaperSearcher(Protocol):
    def search(self, query: str, max_results: int) -> list[Paper]: ...


@dataclass
class SearchResult:
    query: str
    papers: list[Paper]
    errors: dict[str, str] = field(default_factory=dict)


def collect_papers(
    query: str, sources: dict[str, PaperSearcher], max_results: int = 10
) -> SearchResult:
    papers: list[Paper] = []
    errors: dict[str, str] = {}
    for name, searcher in sources.items():
        try:
            papers.extend(searcher.search(query, max_results))
        except IngestionError as exc:
            errors[name] = str(exc)
    return SearchResult(query=query, papers=deduplicate(papers), errors=errors)


def save_result(result: SearchResult, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(UTC)
    slug = re.sub(r"[^a-z0-9]+", "-", result.query.lower()).strip("-")[:50] or "query"
    path = output_dir / f"{slug}_{now:%Y%m%dT%H%M%SZ}.json"
    payload = {
        "query": result.query,
        "retrieved_at": now.isoformat(),
        "errors": result.errors,
        "papers": [paper.model_dump(mode="json") for paper in result.papers],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path