import pytest

from sc_research_agentic_rag.document_processing import Page, chunk_pages
from sc_research_agentic_rag.rag import SYSTEM_PROMPT, SourcePassage, build_messages, build_passages
from sc_research_agentic_rag.retrieval import RetrievedChunk


def make_chunk(
    chunk_id: str = "a#0",
    text: str = "some text",
    paper_uid: str = "a",
    page: int = 1,
    char_start: int = 0,
    score: float = 0.5,
    **kwargs,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        score=score,
        text=text,
        paper_uid=paper_uid,
        page=page,
        char_start=char_start,
        char_end=char_start + len(text),
        title=kwargs.pop("title", f"Title {paper_uid}"),
        **kwargs,
    )


def to_retrieved(chunks, score: float = 0.5) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            chunk_id=c.chunk_id, score=score, text=c.text, paper_uid=c.paper_uid, page=c.page,
            char_start=c.char_start, char_end=c.char_end, title="T",
        )
        for c in chunks
    ]


def test_overlapping_chunks_are_merged_into_the_original_text() -> None:
    page_text = " ".join(f"word{n}" for n in range(300))
    chunks = chunk_pages([Page(paper_uid="a", page=1, text=page_text)], chunk_size=200, overlap=50)
    assert len(chunks) > 5

    [passage] = build_passages(to_retrieved(chunks), max_chars=10_000)

    assert passage.text == page_text
    assert len(passage.chunk_ids) == len(chunks)


def test_merge_does_not_depend_on_input_order() -> None:
    page_text = " ".join(f"w{n}" for n in range(200))
    chunks = to_retrieved(chunk_pages([Page(paper_uid="a", page=1, text=page_text)], chunk_size=150, overlap=40))
    [passage] = build_passages(list(reversed(chunks)), max_chars=10_000)
    assert passage.text == page_text


def test_disjoint_chunks_stay_separate() -> None:
    first = make_chunk("a#0", "first part", char_start=0, score=0.9)
    second = make_chunk("a#5", "far away part", char_start=500, score=0.8)
    passages = build_passages([second, first])
    assert [p.text for p in passages] == ["first part", "far away part"]


def test_chunk_contained_in_another_is_absorbed() -> None:
    big = make_chunk("a#0", "0123456789", char_start=0, score=0.4)
    inner = make_chunk("a#1", "345", char_start=3, score=0.9)
    [passage] = build_passages([big, inner])
    assert passage.text == "0123456789"
    assert passage.score == 0.9
    assert set(passage.chunk_ids) == {"a#0", "a#1"}


def test_different_pages_and_papers_are_never_merged() -> None:
    chunks = [
        make_chunk("a#0", "same", "a", page=1, score=0.9),
        make_chunk("a#1", "same", "a", page=2, score=0.8),
        make_chunk("b#0", "same", "b", page=1, score=0.7),
    ]
    assert len(build_passages(chunks)) == 3


def test_passages_are_numbered_by_relevance_and_keep_metadata() -> None:
    low = make_chunk("a#0", "low", "a", score=0.2)
    high = make_chunk(
        "b#3", "high", "b", page=4, score=0.9, title="Best paper", authors=["Ann", "Bob"],
        year=2024, doi="10.1/x", url="https://x/b", source="arxiv",
    )
    passages = build_passages([low, high])
    assert [p.number for p in passages] == [1, 2]
    first = passages[0]
    assert (first.paper_uid, first.title, first.page, first.score) == ("b", "Best paper", 4, 0.9)
    assert first.authors == ("Ann", "Bob")
    assert (first.year, first.doi, first.url, first.source) == (2024, "10.1/x", "https://x/b", "arxiv")
    assert first.chunk_ids == ("b#3",)


def test_inconsistent_offsets_are_not_merged() -> None:
    first = make_chunk("a#0", "abcdef", char_start=0)
    broken = RetrievedChunk("a#1", 0.5, "xyz", "a", 1, 3, 99, "T")
    assert len(build_passages([first, broken])) == 2


def test_budget_skips_passages_that_do_not_fit() -> None:
    chunks = [
        make_chunk("a#0", "x" * 60, "a", score=0.9),
        make_chunk("b#0", "y" * 60, "b", score=0.8),
        make_chunk("c#0", "z" * 30, "c", score=0.7),
    ]
    passages = build_passages(chunks, max_chars=100)
    assert [p.paper_uid for p in passages] == ["a", "c"]
    assert [p.number for p in passages] == [1, 2]


def test_first_passage_is_truncated_when_it_alone_exceeds_the_budget() -> None:
    [passage] = build_passages([make_chunk("a#0", "word " * 100, score=0.9)], max_chars=50)
    assert 0 < len(passage.text) <= 50


def test_empty_input_and_invalid_budget() -> None:
    assert build_passages([]) == []
    with pytest.raises(ValueError):
        build_passages([make_chunk()], max_chars=0)


def passage(number: int = 1, **kwargs) -> SourcePassage:
    defaults = dict(
        number=number, paper_uid="a", title="Dense Retrieval", authors=("Ann", "Bob"), year=2020,
        doi=None, url=None, source="arxiv", page=3, text="Passage text.", chunk_ids=("a#0",), score=0.9,
    )
    return SourcePassage(**{**defaults, **kwargs})


def test_messages_have_system_prompt_then_sources_and_question() -> None:
    messages = build_messages("What is DPR?", [passage(1), passage(2, title="Other", text="More text.")])
    assert [m.role for m in messages] == ["system", "user"]
    assert messages[0].content == SYSTEM_PROMPT
    user = messages[1].content
    assert "[1] Dense Retrieval (Ann et al., 2020) - page 3\nPassage text." in user
    assert "[2] Other (Ann et al., 2020) - page 3\nMore text." in user
    assert user.index("[1]") < user.index("[2]") < user.index("Question: What is DPR?")


@pytest.mark.parametrize(
    "kwargs,header",
    [
        ({"authors": ("Solo",), "year": 2021}, "[1] Dense Retrieval (Solo, 2021) - page 3"),
        ({"authors": (), "year": 2021}, "[1] Dense Retrieval (2021) - page 3"),
        ({"authors": ("Solo",), "year": None}, "[1] Dense Retrieval (Solo) - page 3"),
        ({"authors": (), "year": None}, "[1] Dense Retrieval - page 3"),
    ],
)
def test_header_formats(kwargs, header) -> None:
    assert header in build_messages("q?", [passage(**kwargs)])[1].content


def test_system_prompt_contains_the_grounding_rules() -> None:
    for expected in ("ONLY", "[1]", "do not contain enough information", "untrusted", "same language"):
        assert expected in SYSTEM_PROMPT


@pytest.mark.parametrize("question,passages", [("  ", [passage()]), ("q?", [])])
def test_messages_reject_invalid_input(question, passages) -> None:
    with pytest.raises(ValueError):
        build_messages(question, passages)