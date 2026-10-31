import pytest

from sc_research_agentic_rag.document_processing import Page, clean_pages


def make_pages(*texts: str, uid: str = "arxiv:1") -> list[Page]:
    return [Page(paper_uid=uid, page=i, text=text) for i, text in enumerate(texts, start=1)]


def clean_one(text: str) -> str:
    return clean_pages(make_pages(text))[0].text


def test_ligatures_are_expanded() -> None:
    assert clean_one("We de\ufb01ne \ufb02ow in the o\ufb03ce") == "We define flow in the office"


def test_special_spaces_and_invisible_characters() -> None:
    assert clean_one("a\u00a0b\u2009c\u200bd\ufeffe") == "a b cde"


def test_soft_hyphen_at_line_end_is_joined() -> None:
    assert clean_one("the measur\u00ad\ning tool") == "the measuring tool"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("the dis-\ntance is", "the distance is"),
        ("the dis-  \n  tance is", "the distance is"),
        ("rapport d\u00e9ve-\nlopp\u00e9", "rapport d\u00e9veloppé"),
        ("state-of-the-art model", "state-of-the-art model"),
        ("by Smith-\nJones", "by Smith-\nJones"),
        ("COVID-\n19 data", "COVID-\n19 data"),
    ],
)
def test_hyphenation(raw: str, expected: str) -> None:
    assert clean_one(raw) == expected


def test_whitespace_is_normalized() -> None:
    assert clean_one("  a \t b  \n\n\n\n  c   d  \n") == "a b\n\nc d"


WORDS = ["apple", "banana", "cherry", "damson", "elder", "fig", "grape", "hazel"]


def body(i: int) -> list[str]:
    word = WORDS[i % len(WORDS)]
    return [f"{word} sentence {n} about {word}s" for n in range(5)]


def test_repeated_header_and_page_numbers_are_removed() -> None:
    pages = make_pages(*["\n".join(["TP DE DEEP LEARNING - HAM500", *body(i), str(i + 1)]) for i in range(5)])
    cleaned = clean_pages(pages)
    assert [p.text for p in cleaned] == ["\n".join(body(i)) for i in range(5)]


def test_header_missing_on_first_page_is_still_removed() -> None:
    cover = "\n".join(["Cover title", "Author", *body(0)])
    texts = [cover] + ["\n".join(["RUNNING HEADER", *body(i)]) for i in range(1, 7)]
    cleaned = clean_pages(make_pages(*texts))
    assert cleaned[0].text == cover
    assert cleaned[3].text == "\n".join(body(3))


def test_line_repeated_in_the_middle_of_pages_is_kept() -> None:
    texts = ["\n".join([*body(i)[:2], "Figure", *body(i)[2:]]) for i in range(5)]
    for page in clean_pages(make_pages(*texts)):
        assert "Figure" in page.text


def test_line_below_threshold_is_kept() -> None:
    texts = ["\n".join(["Seen twice", *body(i)]) for i in range(2)] + ["\n".join(body(i)) for i in range(2, 8)]
    cleaned = clean_pages(make_pages(*texts))
    assert cleaned[0].text.startswith("Seen twice")


def test_short_pages_are_left_alone() -> None:
    texts = [f"Same header\nshort body {WORDS[i]}\nfooter" for i in range(5)]
    cleaned = clean_pages(make_pages(*texts))
    assert [p.text for p in cleaned] == texts


def test_edge_removal_is_skipped_for_short_documents() -> None:
    texts = ["\n".join(["Same header", *body(i)]) for i in range(2)]
    cleaned = clean_pages(make_pages(*texts))
    assert [p.text for p in cleaned] == texts


def test_page_numbers_and_uid_are_preserved_and_input_untouched() -> None:
    pages = make_pages("a\ufb01", "", "c", uid="upload:x")
    cleaned = clean_pages(pages)
    assert [p.page for p in cleaned] == [1, 2, 3]
    assert {p.paper_uid for p in cleaned} == {"upload:x"}
    assert cleaned[1].text == ""
    assert pages[0].text == "a\ufb01"


def test_empty_input() -> None:
    assert clean_pages([]) == []