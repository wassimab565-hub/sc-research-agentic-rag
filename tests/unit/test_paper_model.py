from datetime import date

import pytest
from pydantic import ValidationError

from sc_research_agentic_rag.ingestion import Paper, PaperSource


def make_paper(**overrides) -> Paper:
    data = {
        "source": PaperSource.ARXIV,
        "source_id": "2401.00001",
        "title": "A Test Paper",
        "url": "https://arxiv.org/abs/2401.00001",
    }
    data.update(overrides)
    return Paper(**data)


def test_minimal_paper_defaults() -> None:
    paper = make_paper()
    assert paper.authors == []
    assert paper.abstract is None
    assert paper.doi is None
    assert paper.year is None
    assert paper.retrieved_at.tzinfo is not None


def test_uid_and_year() -> None:
    paper = make_paper(published=date(2024, 1, 2))
    assert paper.uid == "arxiv:2401.00001"
    assert paper.year == 2024


@pytest.mark.parametrize(
    "raw",
    ["10.1000/XYZ123", "https://doi.org/10.1000/XYZ123", "doi:10.1000/xyz123", " HTTP://doi.org/10.1000/xyz123 "],
)
def test_doi_is_normalized(raw: str) -> None:
    assert make_paper(doi=raw).doi == "10.1000/xyz123"


def test_empty_doi_becomes_none() -> None:
    assert make_paper(doi="").doi is None


def test_whitespace_is_cleaned() -> None:
    paper = make_paper(title="  A   Test\n Paper ", abstract="line one\n   line two")
    assert paper.title == "A Test Paper"
    assert paper.abstract == "line one line two"


def test_missing_required_fields_fail() -> None:
    with pytest.raises(ValidationError):
        Paper(source=PaperSource.ARXIV, source_id="x")