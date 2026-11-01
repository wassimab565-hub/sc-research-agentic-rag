import pytest

from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.rag import NO_SOURCES_MESSAGE, RagPipeline, extract_citations
from sc_research_agentic_rag.rag import cli
from sc_research_agentic_rag.retrieval import RetrievedChunk
from sc_research_agentic_rag.storage import SearchFilter


def chunk(n: int, paper: str = "p", page: int = 1, **kwargs) -> RetrievedChunk:
    text = f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"{paper}#{n}", score=1.0 - n / 10, text=text, paper_uid=f"{paper}{n}", page=page,
        char_start=0, char_end=len(text), title=f"Paper {n}", **kwargs,
    )


class FakeRetriever:
    def __init__(self, chunks) -> None:
        self.chunks = chunks
        self.calls: list[dict] = []

    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        self.calls.append(
            {"question": question, "top_k": top_k, "filters": filters,
             "score_threshold": score_threshold, "max_per_paper": max_per_paper}
        )
        return self.chunks


class FakeLLM:
    def __init__(self, answer: str = "An answer [1].", error: Exception | None = None) -> None:
        self.answer, self.error, self.messages = answer, error, None

    def chat(self, messages):
        if self.error:
            raise self.error
        self.messages = messages
        return self.answer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("A [1] and B [2][3].", [1, 2, 3]),
        ("Both [1, 2] and [ 4 ,5 ]", [1, 2, 4, 5]),
        ("Range [1-3] and [5\u20137]", [1, 2, 3, 5, 6, 7]),
        ("Mixed [1, 3-4]", [1, 3, 4]),
        ("Repeated [2] [2] [1]", [1, 2]),
        ("No citations here", []),
        ("Not citations [a] [] [1.5] [x1] (2020)", []),
        ("A year in brackets [2020] is ignored", []),
        ("Backwards range [4-2]", []),
        ("Markdown link [3](http://x)", [3]),
    ],
)
def test_extract_citations(text: str, expected: list[int]) -> None:
    assert extract_citations(text) == expected


def test_answer_is_generated_from_retrieved_sources() -> None:
    retriever, llm = FakeRetriever([chunk(1), chunk(2)]), FakeLLM("Because of source one [1].")
    result = RagPipeline(retriever, llm).ask("Why?")

    assert result.answer == "Because of source one [1]."
    assert [s.number for s in result.sources] == [1, 2]
    assert [s.title for s in result.cited_sources] == ["Paper 1"]
    assert result.invalid_citations == []
    assert result.has_citations
    assert [m.role for m in llm.messages] == ["system", "user"]
    assert "Text of source number 1." in llm.messages[1].content
    assert "Question: Why?" in llm.messages[1].content


def test_retrieval_parameters_are_forwarded() -> None:
    retriever = FakeRetriever([chunk(1)])
    flt = SearchFilter(sources=["arxiv"])
    pipeline = RagPipeline(retriever, FakeLLM(), top_k=7, score_threshold=0.4, max_per_paper=2)
    pipeline.ask("q?", flt)
    assert retriever.calls == [
        {"question": "q?", "top_k": 7, "filters": flt, "score_threshold": 0.4, "max_per_paper": 2}
    ]


def test_unknown_citations_are_reported_not_hidden() -> None:
    result = RagPipeline(FakeRetriever([chunk(1), chunk(2)]), FakeLLM("A [1], B [2] and C [7].")).ask("q?")
    assert [s.number for s in result.cited_sources] == [1, 2]
    assert result.invalid_citations == [7]


def test_answer_without_citations_is_flagged() -> None:
    result = RagPipeline(FakeRetriever([chunk(1)]), FakeLLM("I cannot say.")).ask("q?")
    assert not result.has_citations
    assert result.cited_sources == []
    assert len(result.sources) == 1


def test_no_retrieved_chunks_skips_the_llm() -> None:
    llm = FakeLLM()
    result = RagPipeline(FakeRetriever([]), llm).ask("q?")
    assert result.answer == NO_SOURCES_MESSAGE
    assert result.sources == []
    assert llm.messages is None


def test_llm_errors_propagate() -> None:
    with pytest.raises(LLMError):
        RagPipeline(FakeRetriever([chunk(1)]), FakeLLM(error=LLMError("boom"))).ask("q?")


def fake_pipeline(answer: str = "Answer [1].", chunks=None, error=None):
    retriever = FakeRetriever([chunk(1, authors=["Ann", "Bob"], year=2024, url="https://x/1")] if chunks is None else chunks)
    pipeline = RagPipeline(retriever, FakeLLM(answer, error))

    class Store:
        closed = False

        def close(self) -> None:
            self.closed = True

    store = Store()
    return pipeline, store


def test_cli_prints_answer_and_cited_sources(monkeypatch, capsys) -> None:
    pipeline, store = fake_pipeline()
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    assert cli.main(["What?"]) == 0
    out = capsys.readouterr().out
    assert "Answer [1]." in out
    assert "Sources:" in out
    assert "[1] Paper 1 (Ann et al., 2024), p. 1 - https://x/1" in out
    assert "Warning" not in out
    assert store.closed


def test_cli_warns_about_missing_and_unknown_citations(monkeypatch, capsys) -> None:
    pipeline, store = fake_pipeline("No citation at all.")
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    cli.main(["What?"])
    assert "cites no source" in capsys.readouterr().out

    pipeline, store = fake_pipeline("Odd claim [9].")
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    cli.main(["What?"])
    assert "unknown sources: [9]" in capsys.readouterr().out


def test_cli_show_sources(monkeypatch, capsys) -> None:
    pipeline, store = fake_pipeline("Answer [1].", chunks=[chunk(1), chunk(2)])
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    cli.main(["What?", "--show-sources"])
    out = capsys.readouterr().out
    assert "Retrieved passages:" in out
    assert "[2] Paper 2" in out
    assert "score" in out


def test_cli_no_results(monkeypatch, capsys) -> None:
    pipeline, store = fake_pipeline(chunks=[])
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    assert cli.main(["What?"]) == 0
    assert NO_SOURCES_MESSAGE in capsys.readouterr().out


def test_cli_reports_errors_and_closes_the_store(monkeypatch, capsys) -> None:
    pipeline, store = fake_pipeline(error=LLMError("Ollama is down"))
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (pipeline, store))
    assert cli.main(["What?"]) == 1
    assert "Ollama is down" in capsys.readouterr().err
    assert store.closed


def test_cli_passes_options_to_the_pipeline_builder(monkeypatch) -> None:
    received: list = []
    pipeline, store = fake_pipeline()

    def build(*args):
        received.extend(args)
        return pipeline, store

    monkeypatch.setattr(cli, "build_pipeline", build)
    cli.main(["q?", "--top-k", "3", "--threshold", "0.5", "--max-per-paper", "1", "--max-context-chars", "5000"])
    assert received == [3, 5000, 0.5, 1]


@pytest.mark.parametrize(
    "argv",
    [["  "], ["q", "--top-k", "0"], ["q", "--max-per-paper", "0"], ["q", "--max-context-chars", "0"],
     ["q", "--year-min", "2024", "--year-max", "2020"], ["q", "--source", "web"]],
)
def test_cli_rejects_invalid_arguments(argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2