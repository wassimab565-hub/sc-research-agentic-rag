import pytest

from sc_research_agentic_rag.graph import GraphQueries
from sc_research_agentic_rag.graph import query_cli as cli
from sc_research_agentic_rag.graph.errors import GraphError


class FakeReader:
    def __init__(self, *answers) -> None:
        self.answers = list(answers)
        self.calls: list = []
        self.closed = False

    def read(self, query, **params):
        self.calls.append((query, params))
        if isinstance(self.answers[0], Exception):
            raise self.answers.pop(0)
        return self.answers.pop(0)

    def close(self) -> None:
        self.closed = True


CONCEPT_ROW = {
    "uid": "arxiv:1", "title": "Paper One", "year": 2020, "chunks": 2,
    "pages": [7, 3], "authors": ["Ann", "Bob", "Cy", "Dee"],
}


def test_papers_for_concept_normalizes_the_name_and_sorts_pages() -> None:
    reader = FakeReader([CONCEPT_ROW])
    [paper] = GraphQueries(reader).papers_for_concept("  Dense   RETRIEVAL ", limit=3)
    assert reader.calls[0][1] == {"key": "dense retrieval", "limit": 3}
    assert (paper.uid, paper.chunks, paper.pages, paper.authors[0]) == ("arxiv:1", 2, (3, 7), "Ann")


def test_similar_concepts() -> None:
    reader = FakeReader([{"name": "BM25", "chunks": 25}])
    assert GraphQueries(reader).similar_concepts("bm") == [("BM25", 25)]


def test_papers_for_author_sorts_by_year_with_unknown_year_last_and_applies_limit() -> None:
    reader = FakeReader([
        {"uid": "u", "title": None, "year": None, "matched": ["P. Lewis"]},
        {"uid": "a", "title": "A", "year": 2020, "matched": ["P. Lewis"]},
        {"uid": "b", "title": "B", "year": 2023, "matched": ["Patrick Lewis", "P. Lewis"]},
    ])
    papers = GraphQueries(reader).papers_for_author("Lewis", limit=2)
    assert [p.uid for p in papers] == ["b", "a"]
    assert papers[0].matched_authors == ("P. Lewis", "Patrick Lewis")
    assert reader.calls[0][1] == {"key": "lewis"}


def test_related_papers_show_the_rarest_shared_concepts_first() -> None:
    shared = [{"name": f"c{n}", "df": n} for n in (9, 2, 5, 3, 4, 6, 7)]
    reader = FakeReader([{"uid": "q", "title": "Q", "year": 2021, "score": 1.5, "shared": shared}])
    [paper] = GraphQueries(reader).related_papers(" arxiv:1 ")
    assert paper.shared == ("c2", "c3", "c4", "c5", "c6")
    assert paper.score == 1.5
    assert reader.calls[0][1] == {"uid": "arxiv:1", "limit": 5}


@pytest.mark.parametrize(
    "call",
    [
        lambda q: q.papers_for_concept("  "),
        lambda q: q.papers_for_author(""),
        lambda q: q.related_papers(" "),
        lambda q: q.papers_for_concept("x", limit=0),
        lambda q: q.similar_concepts("x", limit=0),
    ],
)
def test_invalid_arguments(call) -> None:
    with pytest.raises(ValueError):
        call(GraphQueries(FakeReader()))


def run(monkeypatch, reader, *argv) -> int:
    monkeypatch.setattr(cli, "build_store", lambda: reader)
    return cli.main(list(argv))


def test_cli_concept(monkeypatch, capsys) -> None:
    reader = FakeReader([CONCEPT_ROW])
    assert run(monkeypatch, reader, "--concept", "BM25") == 0
    out = capsys.readouterr().out
    assert "1. [arxiv:1] Paper One (2020)" in out
    assert "2 chunks, pages 3, 7 | Ann, Bob, Cy et al." in out
    assert reader.closed


def test_cli_concept_without_result_suggests_similar_ones(monkeypatch, capsys) -> None:
    reader = FakeReader([], [{"name": "BM25", "chunks": 25}])
    assert run(monkeypatch, reader, "--concept", "bm") == 0
    out = capsys.readouterr().out
    assert "no paper mentions 'bm'" in out
    assert "BM25 (25 chunks)" in out


def test_cli_author(monkeypatch, capsys) -> None:
    reader = FakeReader(
        [{"uid": "arxiv:2", "title": None, "year": None, "matched": ["Patrick Lewis"]}],
        [{"name": "RAG", "chunks": 9}],
    )
    assert run(monkeypatch, reader, "--author", "lewis") == 0
    out = capsys.readouterr().out
    assert "matched authors: Patrick Lewis" in out
    assert "1. [arxiv:2] arxiv:2" in out
    assert "main concepts: RAG (9)" in out


def test_cli_related(monkeypatch, capsys) -> None:
    row = {"uid": "q", "title": "Q", "year": None, "score": 0.5, "shared": [{"name": "BM25", "df": 2}]}
    assert run(monkeypatch, FakeReader([row]), "--related", "arxiv:1") == 0
    out = capsys.readouterr().out
    assert "1. [q] Q (score 0.50)" in out
    assert "shared: BM25" in out


def test_cli_related_unknown_paper(monkeypatch, capsys) -> None:
    assert run(monkeypatch, FakeReader([]), "--related", "nope") == 0
    assert "no related paper" in capsys.readouterr().out


def test_cli_reports_graph_errors_and_closes(monkeypatch, capsys) -> None:
    reader = FakeReader(GraphError("down"))
    assert run(monkeypatch, reader, "--concept", "x") == 1
    assert "down" in capsys.readouterr().err
    assert reader.closed


@pytest.mark.parametrize("argv", [[], ["--concept", "x", "--author", "y"], ["--concept", " "], ["--concept", "x", "--limit", "0"]])
def test_cli_rejects_invalid_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2