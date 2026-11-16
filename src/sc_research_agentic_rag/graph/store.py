import re
import unicodedata
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import DriverError, Neo4jError

from sc_research_agentic_rag.graph.config import Neo4jSettings
from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.ingestion.models import Paper

SCHEMA = (
    "CREATE CONSTRAINT paper_uid IF NOT EXISTS FOR (p:Paper) REQUIRE p.uid IS UNIQUE",
    "CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE",
    "CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
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

CLEAR = "MATCH (n) WHERE n:Paper OR n:Author OR n:Category DETACH DELETE n"
COUNT_NODES = (
    "MATCH (n) WHERE n:Paper OR n:Author OR n:Category "
    "RETURN labels(n)[0] AS name, count(n) AS count"
)
COUNT_RELATIONSHIPS = (
    "MATCH ()-[r:AUTHORED|HAS_CATEGORY]->() RETURN type(r) AS name, count(r) AS count"
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

    def clear(self) -> None:
        self._write(CLEAR)

    def counts(self) -> dict[str, int]:
        rows = self._read(COUNT_NODES) + self._read(COUNT_RELATIONSHIPS)
        return {row["name"]: row["count"] for row in rows}

    def close(self) -> None:
        self._driver.close()

    def _write(self, query: str, **params: Any) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(lambda tx: tx.run(query, **params).consume())
        except (Neo4jError, DriverError) as exc:
            raise GraphError(f"Neo4j error: {exc}") from exc

    def _read(self, query: str) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                return session.run(query).data()
        except (Neo4jError, DriverError) as exc:
            raise GraphError(f"Neo4j error: {exc}") from exc