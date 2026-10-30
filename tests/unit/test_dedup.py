from datetime import date

from sc_research_agentic_rag.ingestion import Paper, PaperSource, deduplicate


def paper(source: PaperSource, source_id: str, title: str, **kwargs) -> Paper:
    return Paper(source=source, source_id=source_id, title=title, url=f"https://x/{source_id}", **kwargs)


def test_merges_by_doi_and_fills_missing_fields() -> None:
    a = paper(PaperSource.ARXIV, "2401.1", "Title A", doi="10.1/x", abstract="abs", categories=["cs.IR"])
    b = paper(
        PaperSource.OPENALEX, "W1", "Different Title", doi="https://doi.org/10.1/X",
        published=date(2024, 1, 2), pdf_url="https://x/p.pdf", categories=["IR", "cs.IR"],
    )
    result = deduplicate([a, b])
    assert len(result) == 1
    merged = result[0]
    assert merged.uid == "arxiv:2401.1"
    assert merged.abstract == "abs"
    assert merged.published == date(2024, 1, 2)
    assert merged.pdf_url == "https://x/p.pdf"
    assert merged.categories == ["cs.IR", "IR"]
    assert merged.also_found_as == ["openalex:W1"]


def test_merges_by_normalized_title_and_adopts_missing_doi() -> None:
    a = paper(PaperSource.ARXIV, "2401.1", "Improving Retrieval: A Study!")
    b = paper(PaperSource.OPENALEX, "W1", "improving retrieval - a study", doi="10.1/x")
    result = deduplicate([a, b])
    assert len(result) == 1
    assert result[0].doi == "10.1/x"


def test_doi_found_by_merge_matches_later_papers() -> None:
    a = paper(PaperSource.ARXIV, "1", "Shared Title")
    b = paper(PaperSource.OPENALEX, "W1", "Shared Title", doi="10.1/x")
    c = paper(PaperSource.OPENALEX, "W2", "Other Title", doi="10.1/x")
    result = deduplicate([a, b, c])
    assert len(result) == 1
    assert result[0].also_found_as == ["openalex:W1", "openalex:W2"]


def test_distinct_papers_are_kept_in_order() -> None:
    a = paper(PaperSource.ARXIV, "1", "First", doi="10.1/a")
    b = paper(PaperSource.ARXIV, "2", "Second", doi="10.1/b")
    assert [p.source_id for p in deduplicate([a, b])] == ["1", "2"]


def test_same_paper_twice_does_not_list_itself() -> None:
    a = paper(PaperSource.ARXIV, "1", "Same")
    assert deduplicate([a, a])[0].also_found_as == []


def test_empty_input() -> None:
    assert deduplicate([]) == []