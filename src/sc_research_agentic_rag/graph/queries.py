from dataclasses import dataclass
from typing import Any, Protocol

from sc_research_agentic_rag.graph.store import author_key

SHARED_SHOWN = 5

PAPERS_FOR_CONCEPT = """
MATCH (k:Concept {key: $key})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(p:Paper)
WITH p, count(DISTINCT c) AS chunks, collect(DISTINCT c.page) AS pages
OPTIONAL MATCH (a:Author)-[:AUTHORED]->(p)
RETURN p.uid AS uid, p.title AS title, p.year AS year, chunks, pages,
       collect(DISTINCT a.name) AS authors
ORDER BY chunks DESC, uid LIMIT $limit
"""
SIMILAR_CONCEPTS = """
MATCH (k:Concept)<-[:MENTIONS]-(c:Chunk)
WHERE k.key CONTAINS $key
RETURN k.name AS name, count(c) AS chunks
ORDER BY chunks DESC, name LIMIT $limit
"""
PAPERS_FOR_AUTHOR = """
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
WHERE a.key CONTAINS $key
RETURN p.uid AS uid, p.title AS title, p.year AS year, collect(DISTINCT a.name) AS matched
ORDER BY uid
"""
AUTHOR_CONCEPTS = """
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
WHERE a.key CONTAINS $key
WITH DISTINCT p
MATCH (p)-[:HAS_CHUNK]->(c:Chunk)-[:MENTIONS]->(k:Concept)
RETURN k.name AS name, count(DISTINCT c) AS chunks
ORDER BY chunks DESC, name LIMIT $limit
"""
RELATED_PAPERS = """
MATCH (p:Paper {uid: $uid})-[:HAS_CHUNK]->(:Chunk)-[:MENTIONS]->(k:Concept)
WITH DISTINCT k
MATCH (k)<-[:MENTIONS]-(:Chunk)<-[:HAS_CHUNK]-(q:Paper)
WITH k, collect(DISTINCT q) AS papers
UNWIND papers AS q
WITH q, k, size(papers) AS df
WHERE q.uid <> $uid
WITH q, collect({name: k.name, df: df}) AS shared, sum(1.0 / df) AS score
RETURN q.uid AS uid, q.title AS title, q.year AS year, score, shared
ORDER BY score DESC, uid LIMIT $limit
"""


class Reader(Protocol):
    def read(self, query: str, **params: Any) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class ConceptPaper:
    uid: str
    title: str | None
    year: int | None
    authors: tuple[str, ...]
    pages: tuple[int, ...]
    chunks: int


@dataclass(frozen=True)
class AuthorPaper:
    uid: str
    title: str | None
    year: int | None
    matched_authors: tuple[str, ...]


@dataclass(frozen=True)
class RelatedPaper:
    uid: str
    title: str | None
    year: int | None
    score: float
    shared: tuple[str, ...]


def _key(text: str) -> str:
    key = author_key(text)
    if not key:
        raise ValueError("name must not be empty")
    return key


def _check_limit(limit: int) -> None:
    if limit < 1:
        raise ValueError("limit must be >= 1")


class GraphQueries:
    """Relational questions answered straight from the graph: no LLM, nothing invented."""

    def __init__(self, graph: Reader) -> None:
        self._graph = graph

    def papers_for_concept(self, name: str, limit: int = 10) -> list[ConceptPaper]:
        _check_limit(limit)
        rows = self._graph.read(PAPERS_FOR_CONCEPT, key=_key(name), limit=limit)
        return [
            ConceptPaper(
                row["uid"], row["title"], row["year"], tuple(row["authors"]),
                tuple(sorted(row["pages"])), row["chunks"],
            )
            for row in rows
        ]

    def similar_concepts(self, name: str, limit: int = 5) -> list[tuple[str, int]]:
        _check_limit(limit)
        rows = self._graph.read(SIMILAR_CONCEPTS, key=_key(name), limit=limit)
        return [(row["name"], row["chunks"]) for row in rows]

    def papers_for_author(self, name: str, limit: int = 10) -> list[AuthorPaper]:
        _check_limit(limit)
        rows = self._graph.read(PAPERS_FOR_AUTHOR, key=_key(name))
        papers = [AuthorPaper(row["uid"], row["title"], row["year"], tuple(sorted(row["matched"]))) for row in rows]
        papers.sort(key=lambda paper: (-(paper.year or 0), paper.uid))
        return papers[:limit]

    def author_concepts(self, name: str, limit: int = 10) -> list[tuple[str, int]]:
        _check_limit(limit)
        rows = self._graph.read(AUTHOR_CONCEPTS, key=_key(name), limit=limit)
        return [(row["name"], row["chunks"]) for row in rows]

    def related_papers(self, paper_uid: str, limit: int = 5) -> list[RelatedPaper]:
        _check_limit(limit)
        if not paper_uid.strip():
            raise ValueError("paper uid must not be empty")
        rows = self._graph.read(RELATED_PAPERS, uid=paper_uid.strip(), limit=limit)
        related = []
        for row in rows:
            rarest_first = sorted(row["shared"], key=lambda item: (item["df"], item["name"]))
            related.append(
                RelatedPaper(
                    row["uid"], row["title"], row["year"], float(row["score"]),
                    tuple(item["name"] for item in rarest_first[:SHARED_SHOWN]),
                )
            )
        return related