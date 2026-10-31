import pytest

from sc_research_agentic_rag.document_processing import Page, chunk_pages


def page(text: str, number: int = 1, uid: str = "arxiv:1") -> Page:
    return Page(paper_uid=uid, page=number, text=text)


def words(count: int) -> str:
    return " ".join(f"word{n}" for n in range(count))


def covered_non_whitespace(text: str, chunks) -> bool:
    covered = set()
    for chunk in chunks:
        covered.update(range(chunk.char_start, chunk.char_end))
    return all(i in covered for i, char in enumerate(text) if not char.isspace())


def test_short_page_gives_one_chunk() -> None:
    chunks = chunk_pages([page("Short text.", number=4)])
    assert len(chunks) == 1
    chunk = chunks[0]
    assert (chunk.paper_uid, chunk.index, chunk.page) == ("arxiv:1", 0, 4)
    assert (chunk.char_start, chunk.char_end, chunk.text) == (0, 11, "Short text.")


def test_offsets_always_match_page_text_and_size_is_respected() -> None:
    text = words(400)
    chunks = chunk_pages([page(text)], chunk_size=200, overlap=40)
    assert len(chunks) > 5
    for chunk in chunks:
        assert text[chunk.char_start : chunk.char_end] == chunk.text
        assert len(chunk.text) <= 200
        assert chunk.text == chunk.text.strip()


def test_consecutive_chunks_overlap_without_exceeding_overlap() -> None:
    text = words(400)
    chunks = chunk_pages([page(text)], chunk_size=200, overlap=40)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        shared = previous.char_end - current.char_start
        assert 0 < shared <= 40


def test_zero_overlap_gives_disjoint_chunks() -> None:
    chunks = chunk_pages([page(words(300))], chunk_size=200, overlap=0)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.char_start >= previous.char_end


def test_words_are_never_cut() -> None:
    text = words(500)
    vocabulary = set(text.split())
    for chunk in chunk_pages([page(text)], chunk_size=150, overlap=30):
        tokens = chunk.text.split()
        assert tokens[0] in vocabulary
        assert tokens[-1] in vocabulary


def test_paragraph_break_is_preferred() -> None:
    first = "A" * 3 + " " + "alpha " * 100
    second = "beta " * 100
    text = first.strip() + "\n\n" + second.strip()
    chunks = chunk_pages([page(text)], chunk_size=1000, overlap=100)
    assert chunks[0].text == first.strip()


def test_sentence_end_is_preferred_over_line_break() -> None:
    text = "one two three four five six seven\neight nine ten. Eleven twelve thirteen\nfourteen fifteen"
    chunks = chunk_pages([page(text)], chunk_size=60, overlap=0)
    assert chunks[0].text.endswith("ten.")


def test_indexes_continue_across_pages_and_are_per_paper() -> None:
    pages = [
        page(words(100), 1, "arxiv:1"),
        page(words(100), 2, "arxiv:1"),
        page(words(100), 1, "arxiv:2"),
    ]
    chunks = chunk_pages(pages, chunk_size=200, overlap=20)
    first = [c for c in chunks if c.paper_uid == "arxiv:1"]
    second = [c for c in chunks if c.paper_uid == "arxiv:2"]
    assert [c.index for c in first] == list(range(len(first)))
    assert [c.index for c in second] == list(range(len(second)))
    assert {c.page for c in first} == {1, 2}
    assert first[0].chunk_id == "arxiv:1#0"


@pytest.mark.parametrize(
    "text",
    [words(300), "x" * 3500, "intro\n\n" + words(200) + "\n\n" + "y" * 1500 + "\n\nend", "é à ü " * 200],
)
def test_no_content_is_lost(text: str) -> None:
    chunks = chunk_pages([page(text)], chunk_size=200, overlap=40)
    assert covered_non_whitespace(text, chunks)
    assert all(len(c.text) <= 200 for c in chunks)


def test_very_long_token_is_split() -> None:
    chunks = chunk_pages([page("x" * 3500)], chunk_size=1000, overlap=100)
    assert len(chunks) >= 4


def test_blank_pages_produce_no_chunks() -> None:
    assert chunk_pages([page("   \n\n  "), page("")]) == []
    assert chunk_pages([]) == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, -1), (100, 100), (100, 150)])
def test_invalid_parameters(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_pages([page("text")], chunk_size=size, overlap=overlap)