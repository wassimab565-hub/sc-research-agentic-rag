import re
from collections.abc import Iterable

from sc_research_agentic_rag.ingestion.models import Paper


def deduplicate(papers: Iterable[Paper]) -> list[Paper]:
    unique: list[Paper] = []
    by_doi: dict[str, int] = {}
    by_title: dict[str, int] = {}

    for paper in papers:
        title_key = _title_key(paper.title)
        index = by_doi.get(paper.doi) if paper.doi else None
        if index is None and title_key:
            index = by_title.get(title_key)

        if index is None:
            unique.append(paper)
            index = len(unique) - 1
        else:
            unique[index] = _merge(unique[index], paper)

        merged = unique[index]
        if merged.doi:
            by_doi.setdefault(merged.doi, index)
        if title_key:
            by_title.setdefault(title_key, index)

    return unique


def _title_key(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def _merge(base: Paper, other: Paper) -> Paper:
    found_as = [other.uid, *other.also_found_as]
    return base.model_copy(
        update={
            "abstract": base.abstract or other.abstract,
            "published": base.published or other.published,
            "doi": base.doi or other.doi,
            "pdf_url": base.pdf_url or other.pdf_url,
            "authors": base.authors or other.authors,
            "categories": list(dict.fromkeys([*base.categories, *other.categories])),
            "also_found_as": list(
                dict.fromkeys(uid for uid in [*base.also_found_as, *found_as] if uid != base.uid)
            ),
        }
    )