from datetime import date

import pytest

from sc_research_agentic_rag.agent import AgentTools, Decision
from sc_research_agentic_rag.graph import GraphError
from sc_research_agentic_rag.graph.queries import AuthorPaper, ConceptPaper, RelatedPaper
from sc_research_agentic_rag.ingestion import IngestionError, Paper, PaperSource
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter


def chunk(n: int, text: str | None = None) -> RetrievedChunk:
    text = text or f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"p#{n}", score=1.0 - n / 10, text=text, paper_uid=f"p{n}", page=n,
        char_start=0, char_end=len(text), title=f"Paper {n}",
    )


class FakeRetriever:
    def __init__(self, chunks=()) -> None:
        self.chunks = list(chunks)
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        self.calls.append(
            {"question": question, "top_k": top_k, "filters": filters,
             "score_threshold": score_threshold, "max_per_paper": max_per_paper}
        )
        return self.chunks


class FakeGraph:
    def __init__(self, error: Exception | None = None, **rows) -> None:
        self.error = error
        self.rows = rows
        self.calls: list[tuple] = []

    def _answer(self, name: str, *args):
        self.calls.append((name, *args))
        if self.error:
            raise self.error
        return self.rows.get(name, [])

    def papers_for_concept(self, name, limit=10):
        return self._answer("papers_for_concept", name, limit)

    def similar_concepts(self, name, limit=5):
        return self._answer("similar_concepts", name)

    def papers_for_author(self, name, limit=10):
        return self._answer("papers_for_author", name, limit)

    def author_concepts(self, name, limit=10):
        return self._answer("author_concepts", name, limit)

    def related_papers(self, uid, limit=5):
        return self._answer("related_papers", uid, limit)


class FakeSearcher:
    def __init__(self, papers=(), error: str | None = None) -> None:
        self.papers, self.error, self.calls = list(papers), error, []

    def search(self, query, max_results):
        self.calls.append((query, max_results))
        if self.error:
            raise IngestionError(self.error)
        return self.papers


def make_paper(source_id: str, title: str, **kwargs) -> Paper:
    return Paper(source=PaperSource.ARXIV, source_id=source_id, title=title, url=f"https://x/{source_id}", **kwargs)


def test_retrieve_evidence_forwards_parameters_and_returns_the_chunks() -> None:
    retriever = FakeRetriever([chunk(1), chunk(2)])
    flt = SearchFilter(sources=["arxiv"])
    tools = AgentTools(retriever, top_k=7, score_threshold=0.4, max_per_paper=2)
    result = tools.run(Decision("retrieve_evidence", {"query": "dpr"}), flt)

    assert retriever.calls == [
        {"question": "dpr", "top_k": 7, "filters": flt, "score_threshold": 0.4, "max_per_paper": 2}
    ]
    assert result.found == 2
    assert [c.chunk_id for c in result.chunks] == ["p#1", "p#2"]
    assert "2 passages" in result.summary
    assert "Paper 1 p.1" in result.summary
    assert "Text of source number 1." in result.summary
    assert result.note == ""


def test_retrieve_evidence_without_results() -> None:
    result = AgentTools(FakeRetriever([])).retrieve_evidence("x")
    assert (result.found, result.chunks, result.summary) == (0, (), "no passage found")


def test_long_passage_is_shortened_in_the_summary() -> None:
    result = AgentTools(FakeRetriever([chunk(1, "word " * 200)])).retrieve_evidence("x")
    assert len(result.summary) < 400
    assert "..." in result.summary


def test_graph_concept() -> None:
    graph = FakeGraph(papers_for_concept=[
        ConceptPaper("arxiv:1", "Paper One", 2020, ("Ann",), (3, 7), 2),
        ConceptPaper("arxiv:2", None, None, (), (1,), 1),
    ])
    result = AgentTools(FakeRetriever(), graph).graph_lookup("concept", "BM25")
    assert graph.calls == [("papers_for_concept", "BM25", 5)]
    assert result.found == 2
    assert result.summary == "2 papers mention 'BM25': arxiv:1, arxiv:2"
    assert "- [arxiv:1] Paper One (2020): 2 chunks, pages 3, 7" in result.note
    assert "- [arxiv:2] arxiv:2: 1 chunks, pages 1" in result.note
    assert result.chunks == ()


def test_graph_concept_without_paper_suggests_similar_concepts() -> None:
    graph = FakeGraph(similar_concepts=[("BM25", 25), ("BM25F", 2)])
    result = AgentTools(FakeRetriever(), graph).graph_lookup("concept", "bm")
    assert result.found == 0
    assert result.summary == "no paper mentions 'bm'; similar concepts: BM25, BM25F"
    assert result.note == ""
    assert AgentTools(FakeRetriever(), FakeGraph()).graph_lookup("concept", "zzz").summary == "no paper mentions 'zzz'"


def test_graph_author() -> None:
    graph = FakeGraph(
        papers_for_author=[AuthorPaper("arxiv:2", "Paper Two", 2023, ("Patrick Lewis",))],
        author_concepts=[("RAG", 9)],
    )
    result = AgentTools(FakeRetriever(), graph).graph_lookup("author", "lewis")
    assert result.found == 1
    assert result.summary == "1 papers by Patrick Lewis"
    assert "- [arxiv:2] Paper Two (2023)" in result.note
    assert "Main concepts: RAG (9)" in result.note
    assert AgentTools(FakeRetriever(), FakeGraph()).graph_lookup("author", "x").summary == "no author matches 'x'"


def test_graph_related() -> None:
    graph = FakeGraph(related_papers=[RelatedPaper("q", "Paper Q", 2021, 1.5, ("BM25", "DPR"))])
    result = AgentTools(FakeRetriever(), graph).graph_lookup("related", "arxiv:1")
    assert graph.calls == [("related_papers", "arxiv:1", 5)]
    assert result.summary == "1 related papers: q"
    assert "- [q] Paper Q (2021), shared concepts: BM25, DPR" in result.note
    assert "no related paper" in AgentTools(FakeRetriever(), FakeGraph()).graph_lookup("related", "x").summary


def test_graph_missing_or_failing_does_not_stop_the_agent() -> None:
    assert AgentTools(FakeRetriever()).graph_lookup("concept", "x").summary == "knowledge graph not available"
    down = AgentTools(FakeRetriever(), FakeGraph(error=GraphError("Neo4j error: down")))
    result = down.graph_lookup("concept", "x")
    assert result.found == 0
    assert "knowledge graph unavailable: Neo4j error: down" in result.summary
    invalid = AgentTools(FakeRetriever(), FakeGraph(error=ValueError("name must not be empty")))
    assert "invalid request" in invalid.graph_lookup("author", "x").summary
    assert "unknown graph kind" in AgentTools(FakeRetriever(), FakeGraph()).graph_lookup("paper", "x").summary


def test_search_papers_lists_titles_without_downloading() -> None:
    searcher = FakeSearcher([
        make_paper("1", "First Paper", authors=["Ann", "Bob", "Cy", "Dee"], published=date(2024, 1, 2)),
        make_paper("2", "Second Paper"),
    ])
    result = AgentTools(FakeRetriever(), searchers={"arxiv": searcher}, list_size=3).search_papers("rag survey")
    assert searcher.calls == [("rag survey", 3)]
    assert result.found == 2
    assert result.summary == "2 papers: First Paper; Second Paper"
    assert "- [arxiv:1] 2024 | First Paper | Ann, Bob, Cy et al." in result.note
    assert "- [arxiv:2] n/a | Second Paper" in result.note
    assert result.chunks == ()


def test_search_papers_failures() -> None:
    assert AgentTools(FakeRetriever()).search_papers("x").summary == "paper search not available"
    failing = AgentTools(FakeRetriever(), searchers={"arxiv": FakeSearcher(error="boom")})
    result = failing.search_papers("x")
    assert result.found == 0
    assert result.summary == "no paper found (arxiv: boom)"
    mixed = AgentTools(
        FakeRetriever(), searchers={"arxiv": FakeSearcher(error="boom"), "openalex": FakeSearcher([make_paper("1", "A")])}
    )
    assert mixed.search_papers("x").found == 1


def test_run_dispatches_and_rejects_unknown_actions() -> None:
    tools = AgentTools(FakeRetriever([chunk(1)]), FakeGraph(), {"arxiv": FakeSearcher()})
    assert tools.run(Decision("retrieve_evidence", {"query": "x"})).found == 1
    assert tools.run(Decision("graph_lookup", {"kind": "concept", "value": "x"})).found == 0
    assert tools.run(Decision("search_papers", {"query": "x"})).found == 0
    with pytest.raises(ValueError):
        tools.run(Decision("answer"))


@pytest.mark.parametrize("kwargs", [{"top_k": 0}, {"list_size": 0}, {"list_size": 101}])
def test_invalid_constructor_arguments(kwargs) -> None:
    with pytest.raises(ValueError):
        AgentTools(FakeRetriever(), **kwargs)
