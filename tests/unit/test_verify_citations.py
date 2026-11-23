from sc_research_agentic_rag.verification import format_report, split_sentences, verify_citations


def test_valid_citations_are_ok() -> None:
    report = verify_citations("Dense retrieval uses a dot product [1]. BM25 is a sparse method [2].", [1, 2])
    assert report.status == "ok"
    assert (report.valid, report.invalid) == ((1, 2), ())
    assert (report.sentences, report.uncited_sentences) == (2, ())


def test_unknown_number_is_invalid() -> None:
    report = verify_citations("Dense retrieval uses a dot product [1] and more [7].", [1, 2])
    assert report.status == "invalid"
    assert (report.valid, report.invalid) == ((1,), (7,))


def test_ranges_are_expanded() -> None:
    report = verify_citations("Both papers describe the same retrieval method [1-2].", [1, 2])
    assert report.valid == (1, 2)


def test_answer_without_citation() -> None:
    report = verify_citations("I cannot say anything from the sources provided here.", [1])
    assert report.status == "no_citation"
    assert report.sentences == 1
    assert len(report.uncited_sentences) == 1


def test_no_sources_and_no_citation() -> None:
    report = verify_citations("No relevant sources were found in the indexed documents.", [])
    assert report.status == "no_sources"
    assert (report.sentences, report.uncited_sentences) == (0, ())


def test_citation_on_notes_when_there_is_no_numbered_source() -> None:
    answer = "The papers arxiv:1 and arxiv:2 mention BM25 in their text [1]."
    assert verify_citations(answer, [], has_notes=True).status == "citation_on_notes"
    assert verify_citations(answer, [], has_notes=False).status == "invalid"
    assert verify_citations(answer, [], has_notes=True).invalid == (1,)


def test_uncited_sentence_is_listed() -> None:
    report = verify_citations("Dense retrieval uses a dot product [1]. It is also very fast in practice.", [1])
    assert report.status == "ok"
    assert report.uncited_sentences == ("It is also very fast in practice.",)
    assert report.sentences == 2


def test_short_sentences_are_ignored() -> None:
    report = verify_citations("Yes. Dense retrieval uses a dot product [1].", [1])
    assert report.sentences == 1


def test_marker_after_the_full_stop_belongs_to_its_sentence() -> None:
    report = verify_citations("Dense retrieval uses a dot product. [1] BM25 is sparse and lexical. [2]", [1, 2])
    assert report.sentences == 2
    assert report.uncited_sentences == ()


def test_abbreviations_do_not_split_sentences() -> None:
    sentences = split_sentences("Methods such as BM25, e.g. TF-IDF, are sparse [1]. See Fig. 3 for details [2].")
    assert sentences == [
        "Methods such as BM25, e.g. TF-IDF, are sparse [1].",
        "See Fig. 3 for details [2].",
    ]


def test_bullets_and_lines() -> None:
    answer = "- Dense retrieval uses vectors [1]\n- BM25 uses term counts [2]\n1. Hybrid fuses both rankings [1]"
    report = verify_citations(answer, [1, 2])
    assert report.sentences == 3
    assert report.uncited_sentences == ()


def test_french_answer_with_decimals() -> None:
    report = verify_citations("La classe majoritaire représente 66,4 % des images [1].", [1])
    assert (report.status, report.sentences) == ("ok", 1)


def test_format_report() -> None:
    report = verify_citations("Dense retrieval uses a dot product [1]. It is also very fast in practice [9]. Cheap too, they say.", [1])
    text = format_report(report)
    assert "Verification: invalid" in text
    assert "valid citations: [1]; invalid: [9]" in text
    assert "uncited sentences: 1/3" in text
    assert "- Cheap too, they say." in text
