import pytest
from pydantic import ValidationError

from sc_research_agentic_rag.document_processing import Chunk, Page


def make_chunk(**overrides) -> Chunk:
    data = {
        "paper_uid": "arxiv:2401.00001",
        "index": 3,
        "page": 2,
        "text": "Some text.",
        "char_start": 0,
        "char_end": 10,
    }
    data.update(overrides)
    return Chunk(**data)


def test_page_valid() -> None:
    page = Page(paper_uid="arxiv:1", page=1, text="hello")
    assert page.page == 1


@pytest.mark.parametrize("page", [0, -1])
def test_page_number_must_be_positive(page: int) -> None:
    with pytest.raises(ValidationError):
        Page(paper_uid="arxiv:1", page=page, text="x")


def test_chunk_id_is_deterministic() -> None:
    assert make_chunk().chunk_id == "arxiv:2401.00001#3"


@pytest.mark.parametrize("text", ["", "   \n"])
def test_blank_text_is_rejected(text: str) -> None:
    with pytest.raises(ValidationError):
        make_chunk(text=text)


@pytest.mark.parametrize("overrides", [{"page": 0}, {"index": -1}, {"char_start": -1}])
def test_out_of_range_values_are_rejected(overrides) -> None:
    with pytest.raises(ValidationError):
        make_chunk(**overrides)


@pytest.mark.parametrize("start,end", [(5, 5), (5, 4)])
def test_offsets_must_be_ordered(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        make_chunk(char_start=start, char_end=end)