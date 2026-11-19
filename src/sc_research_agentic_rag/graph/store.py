import re
import unicodedata
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from sc_research_agentic_rag.graph.config import Neo4jSettings
from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.ingestion.models import Paper

from collections.abc import Sequence

SCHEMA = (
    "CREATE CONSTRAINT paper_uid IF NOT EXISTS FOR (p:Paper) REQUIRE p.uid IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
    "CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE",
)

UPSERT_PAPER = """
MERGE (p:Paper {uid: $uid})
SET p.title = $title, p.year = $year, p.source = $source, p.doi = $doi, p.url = $url
WITH p
CALL (p) {
  UNWIND $authors AS author
  MERGE (a:Author {key: author.key})
  ON CREATE SET a.name = author.name
  MERGE (a)-[:AUTHORED]->(p)
}
CALL (p) {
  UNWIND $categories AS category
  MERGE (c:Category {name: category})
  MERGE (p)-[:HAS_CATEGORY]->(c)
}
"""

UPSERT_CHUNK = """
MERGE (p:Paper {uid: $paper_uid})
MERGE (c:Chunk {chunk_id: $chunk_id})
SET c.page = $page, c.paper_uid = $paper_uid
MERGE (p)-[:HAS_CHUNK]->(c)
WITH c
UNWIND $concepts AS concept
MERGE (k:Concept {key: concept.key})
ON CREATE SET k.name = concept.name
MERGE (c)-[:MENTIONS]->(k)
"""

LABELS = "n:Paper OR n:Author OR n:Category OR n:Chunk OR n:Concept"
CLEAR = f"MATCH (n) WHERE {LABELS} DETACH DELETE n"
COUNT_NODES = f"MATCH (n) WHERE {LABELS} RETURN labels(n)[0] AS name, count(n) AS count"
COUNT_RELATIONSHIPS = (
    "MATCH ()-[r:AUTHORED|HAS_CATEGORY|HAS_CHUNK|MENTIONS]->() RETURN type(r) AS name, count(r) AS count"
)

COUNT_CHUNKS = "MATCH (c:Chunk) RETURN count(c) AS count"
LIST_CONCEPT_KEYS = "MATCH (k:Concept) RETURN k.key AS key"
CONCEPT_CHUNKS = (
    "MATCH (k:Concept)<-[:MENTIONS]-(c:Chunk) WHERE k.key IN $keys "
    "RETURN k.key AS key, c.chunk_id AS chunk_id"
)

def author_key(name: str) -> str:
    """Case, accent and spacing insensitive key. 'A. Martin' and 'Alice Martin' stay different."""
    decomposed = unicodedata.normalize("NFKD", name.casefold())
    plain = "".join(char for char in decomposed if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", plain).strip()


class GraphStore:
    def __init__(self, driver: Any, database: str = "neo4j") -> None:
        self._driver = driver
        self._database = database

    @classmethod
    def from_settings(cls, settings: Neo4jSettings | None = None) -> "GraphStore":
        settings = settings or Neo4jSettings()
        if not settings.password:
            raise GraphError("NEO4J_PASSWORD is not set (add it to .env)")
        driver = GraphDatabase.driver(settings.uri, auth=(settings.user, settings.password))
        return cls(driver, settings.database)

    def ensure_schema(self) -> None:
        for statement in SCHEMA:
            self._write(statement)

    def upsert_paper(self, paper: Paper) -> None:
        authors = {}
        for name in paper.authors:
            key = author_key(name)
            if key:
                authors.setdefault(key, " ".join(name.split()))
        categories = list(dict.fromkeys(c.strip() for c in paper.categories if c.strip()))
        self._write(
            UPSERT_PAPER,
            uid=paper.uid,
            title=paper.title,
            year=paper.year,
            source=paper.source.value,
            doi=paper.doi,
            url=paper.url,
            authors=[{"key": key, "name": name} for key, name in authors.items()],
            categories=categories,
        )
        
    def upsert_chunk_concepts(self, paper_uid: str, chunk_id: str, page: int, concepts: list[str]) -> None:
        unique: dict[str, str] = {}
        for name in concepts:
            key = author_key(name)  # same normalization (case, accents, spaces)
            if key:
                unique.setdefault(key, " ".join(name.split()))
        self._write(
            UPSERT_CHUNK,
            paper_uid=paper_uid,
            chunk_id=chunk_id,
            page=page,
            concepts=[{"key": key, "name": name} for key, name in unique.items()],
        )

    def clear(self) -> None:
        self._write(CLEAR)

    def counts(self) -> dict[str, int]:
        rows = self._read(COUNT_NODES) + self._read(COUNT_RELATIONSHIPS)
        return {row["name"]: row["count"] for row in rows}

    def chunk_count(self) -> int:
        rows = self._read(COUNT_CHUNKS)
        return rows[0]["count"] if rows else 0

    def concept_keys(self) -> list[str]:
        return [row["key"] for row in self._read(LIST_CONCEPT_KEYS)]

    def concept_chunks(self, keys: Sequence[str]) -> list[tuple[str, str]]:
        rows = self._read(CONCEPT_CHUNKS, keys=list(keys))
        return [(row["key"], row["chunk_id"]) for row in rows]

    def close(self) -> None:
        self._driver.close()

    def _write(self, query: str, **params: Any) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(lambda tx: tx.run(query, **params).consume())
        except (Neo4jError, DriverError) as exc:
            raise GraphError(f"Neo4j error: {exc}") from exc

    def _read(self, query: str, **params: Any) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                return session.run(query, **params).data()
        except (Neo4jError, DriverError) as exc:
            raise GraphError(f"Neo4j error: {exc}") from exc