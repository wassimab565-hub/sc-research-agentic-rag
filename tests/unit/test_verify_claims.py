import pytest

from sc_research_agentic_rag.verification import extract_claims, extract_numbers, missing_numbers, strip_citations


def test_only_cited_sentences_are_claims() -> None:
    claims = extract_claims("Dense retrieval uses a dot product [1]. It is also very fast in practice. BM25 is sparse [2][3].")
    assert [c.text for c in claims] == [
        "Dense retrieval uses a dot product [1].",
        "BM25 is sparse [2][3].",
    ]
    assert [c.citations for c in claims] == [(1,), (2, 3)]


def test_short_sentences_are_not_claims() -> None:
    assert extract_claims("Yes [1].") == []
    assert extract_claims("") == []


def test_strip_citations() -> None:
    assert strip_citations("BM25 is  sparse [2][3]. And [1, 4] more [5-6].") == "BM25 is sparse. And more."


@pytest.mark.parametrize(
    "text,expected",
    [
        ("332 paramètres", ["332"]),
        ("66,4 % des images", ["66.4"]),
        ("66.40 percent", ["66.4"]),
        ("14 781 255 pixels", ["14781255"]),
        ("14781255 pixels", ["14781255"]),
        ("about 1,000.5 items", ["1000.5"]),
        ("in 2020 and 2021", ["2020", "2021"]),
        ("100.0 and 10.00", ["100", "10"]),
        ("repeated 5 and 5", ["5"]),
        ("citation [1] and [2-3] only", []),
        ("BM25 and arxiv:2305.06983 and hep-th/9901001", []),
        ("no digits here", []),
    ],
)
def test_extract_numbers(text: str, expected: list[str]) -> None:
    assert extract_numbers(text) == expected


def test_missing_numbers_ignores_formatting_differences() -> None:
    assert missing_numbers("Le jeu contient 14 781 255 pixels [1].", ["Il y a 14781255 pixels."]) == ()
    assert missing_numbers("La classe pèse 66,4 % [1].", ["La classe pèse 66.4 % des images."]) == ()


def test_missing_numbers_lists_the_absent_ones() -> None:
    assert missing_numbers("Il y a 332 images et 66,4 % [1].", ["Il y a 331 images, soit 66.4 %."]) == ("332",)


def test_numbers_may_come_from_any_cited_passage() -> None:
    assert missing_numbers("Values 12 and 34 [1][2].", ["twelve is 12", "and 34 here"]) == ()


def test_claim_without_number_has_nothing_missing() -> None:
    assert missing_numbers("Dense retrieval uses vectors [1].", ["anything"]) == ()
