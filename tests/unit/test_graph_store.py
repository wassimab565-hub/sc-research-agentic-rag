import pytest
from neo4j.exceptions import ServiceUnavailable

from sc_research_agentic_rag.graph import GraphError, GraphStore, Neo4jSettings, author_key
from sc_research_agentic_rag.graph import store as store_module
from sc_research_agentic_rag.graph.store import UPSERT_PAPER
from sc_research_agentic_rag.ingestion.models import Paper, PaperSource


class FakeSession:
    def __init__(self, driver) -> None:
        self.driver = driver

    def __enter__(self):
        if self.driver.error:
            raise self.driver.error
        return self

    def __exit__(self, *args) -> None:
        return None

    def execute_write(self, fn):
        return fn(self)

    def run(self, query, **params):
        self.driver.calls.append((query, params))
        return self

    def consume(self):
        return None

    def data(self):
        return self.driver.rows.pop(0)


class FakeDriver:
    def __init__(self, rows=None, error=None) -> None:
        self.calls: list = []
        self.rows = list(rows or [])
        self.error = error
        self.databases: list = []
        self.closed = False

    def session(self, database):
        self.databases.append(database)
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True


def make_paper(**kwargs) -> Paper:
    data = {"source": PaperSource.ARXIV, "source_id": "1", "title": "T", "url": "https://x/1"}
    data.update(kwargs)
    return Paper(**data)


@pytest.mark.parametrize(
    "name,expected",
    [("  Alice   MARTIN ", "alice martin"), ("José Álvarez", "jose alvarez"), ("A. Martin", "a. martin"), ("  ", "")],
)
def test_author_key(name, expected) -> None:
    assert author_key(name) == expected


def test_ensure_schema_creates_three_constraints() -> None:
    driver = FakeDriver()
    GraphStore(driver, "db").ensure_schema()
    assert len(driver.calls) == 5
    assert all("CREATE CONSTRAINT" in query for query, _ in driver.calls)
    assert driver.databases == ["db"] * 5

def test_upsert_chunk_concepts_normalizes_and_deduplicates() -> None:
    driver = FakeDriver()
    GraphStore(driver).upsert_chunk_concepts("a", "a#0", 3, ["Dense  Retrieval", "dense retrieval", "BM25", " "])
    _, params = driver.calls[0]
    assert (params["paper_uid"], params["chunk_id"], params["page"]) == ("a", "a#0", 3)
    assert params["concepts"] == [
        {"key": "dense retrieval", "name": "Dense Retrieval"},
        {"key": "bm25", "name": "BM25"},
    ]

def test_upsert_paper_sends_metadata_authors_and_categories() -> None:
    driver = FakeDriver()
    paper = make_paper(
        authors=["Alice Martin", "alice  martin", "Bob Karim", " "],
        categories=["cs.IR", "cs.IR", "cs.CL"],
        doi="10.1/x",
    )
    GraphStore(driver).upsert_paper(paper)
    query, params = driver.calls[0]
    assert query == UPSERT_PAPER
    assert params["uid"] == "arxiv:1"
    assert params["doi"] == "10.1/x"
    assert params["year"] is None
    assert params["source"] == "arxiv"
    assert params["authors"] == [
        {"key": "alice martin", "name": "Alice Martin"},
        {"key": "bob karim", "name": "Bob Karim"},
    ]
    assert params["categories"] == ["cs.IR", "cs.CL"]


def test_clear_and_counts() -> None:
    driver = FakeDriver(rows=[[{"name": "Paper", "count": 2}], [{"name": "AUTHORED", "count": 5}]])
    store = GraphStore(driver)
    store.clear()
    assert "DETACH DELETE" in driver.calls[0][0]
    assert store.counts() == {"Paper": 2, "AUTHORED": 5}


def test_driver_errors_become_graph_errors() -> None:
    store = GraphStore(FakeDriver(error=ServiceUnavailable("down")))
    with pytest.raises(GraphError, match="down"):
        store.upsert_paper(make_paper())
    with pytest.raises(GraphError):
        store.counts()


def test_close_closes_the_driver() -> None:
    driver = FakeDriver()
    GraphStore(driver).close()
    assert driver.closed


def test_from_settings_requires_a_password() -> None:
    with pytest.raises(GraphError, match="NEO4J_PASSWORD"):
        GraphStore.from_settings(Neo4jSettings(_env_file=None, password=""))


def test_from_settings_builds_the_driver(monkeypatch) -> None:
    received: dict = {}

    def fake_driver(uri, auth):
        received.update(uri=uri, auth=auth)
        return FakeDriver()

    monkeypatch.setattr(store_module.GraphDatabase, "driver", fake_driver)
    GraphStore.from_settings(Neo4jSettings(_env_file=None, uri="bolt://h:1", user="u", password="p"))
    assert received == {"uri": "bolt://h:1", "auth": ("u", "p")}