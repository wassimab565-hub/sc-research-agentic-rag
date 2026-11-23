from types import SimpleNamespace

import pytest

from sc_research_agentic_rag.embeddings import EmbeddingError
from sc_research_agentic_rag.verification import build_report, check_claims, format_claims
from sc_research_agentic_rag.verification.support import cosine, passage_windows

DOT = "The similarity is the dot product of vectors."


def passage(number: int, text: str) -> SimpleNamespace:
    return SimpleNamespace(number=number, text=text)


def vec(text: str) -> list[float]:
    return [1.0 if "dot" in text else 0.0, 1.0 if "bm25" in text.lower() else 0.0, 0.1]


class FakeEmbedder:
    def __init__(self, error: Exception | None = None) -> None:
        self.calls: list[list[str]] = []
        self.error = error

    def embed_passages(self, texts):
        if self.error:
            raise self.error
        self.calls.append(list(texts))
        return [vec(text) for text in texts]


def test_cosine() -> None:
    assert cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0


def test_windows_are_sentences_and_neighbour_pairs() -> None:
    assert passage_windows("One two.\nThree four. Five six.") == [
        "One two.", "Three four.", "Five six.", "One two. Three four.", "Three four. Five six.",
    ]
    assert passage_windows("Single sentence only") == ["Single sentence only"]
    assert passage_windows("") == []


def test_supported_claim() -> None:
    embedder = FakeEmbedder()
    [check] = check_claims("Dense retrieval uses a dot product [1].", [passage(1, DOT + " BM25 counts terms.")], embedder)
    assert check.status == "supported"
    assert check.support == pytest.approx(1.0)
    assert check.missing_numbers == ()
    assert embedder.calls[0] == ["Dense retrieval uses a dot product."]
    assert len(embedder.calls) == 2


def test_weak_claim() -> None:
    [check] = check_claims("BM25 beats everything else [1].", [passage(1, DOT)], FakeEmbedder())
    assert check.status == "weak"
    assert check.support < 0.5


def test_threshold_is_a_parameter() -> None:
    [check] = check_claims("BM25 beats everything else [1].", [passage(1, DOT)], FakeEmbedder(), threshold=0.0)
    assert check.status == "supported"


def test_missing_number_wins_over_a_good_score() -> None:
    [check] = check_claims("The dot product has 332 dimensions [1].", [passage(1, DOT)], FakeEmbedder())
    assert check.status == "number_missing"
    assert check.missing_numbers == ("332",)
    assert check.support is not None


def test_claim_citing_an_unknown_source() -> None:
    embedder = FakeEmbedder()
    [check] = check_claims("The dot product is used here [9].", [passage(1, DOT)], embedder)
    assert (check.status, check.support) == ("no_valid_source", None)
    assert embedder.calls == []


def test_support_is_the_best_over_all_cited_passages() -> None:
    answer = "Dense retrieval uses a dot product [1][2]."
    [check] = check_claims(answer, [passage(1, "Nothing relevant here."), passage(2, DOT)], FakeEmbedder())
    assert check.status == "supported"


def test_without_embedder_only_numbers_are_checked() -> None:
    answer = "The dot product is used here [1]. It has 332 dimensions in total [1]."
    checks = check_claims(answer, [passage(1, DOT)], None)
    assert [c.status for c in checks] == ["unchecked", "number_missing"]


def test_answer_without_claims_never_calls_the_embedder() -> None:
    embedder = FakeEmbedder()
    assert check_claims("No citation anywhere in this answer.", [passage(1, DOT)], embedder) == []
    assert embedder.calls == []


def test_format_claims() -> None:
    answer = "Dense retrieval uses a dot product [1]. BM25 beats everything else [1]. It has 332 dimensions in total [1]."
    text = format_claims(check_claims(answer, [passage(1, DOT)], FakeEmbedder()))
    assert text.splitlines()[0] == "Claims (3): 1 supported, 1 weak, 1 number_missing"
    assert '[weak] 0.01 "BM25 beats everything else [1]."' in text
    assert "(not in the cited sources: 332)" in text
    assert "Dense retrieval" not in text
    assert format_claims([]) == "Claims: none with a citation"


def test_build_report_joins_citation_and_claim_parts() -> None:
    text = build_report("Dense retrieval uses a dot product [1].", [passage(1, DOT)], embedder_factory=FakeEmbedder)
    assert "Verification: ok" in text
    assert "Claims (1): 1 supported" in text


def test_build_report_survives_a_model_that_cannot_load() -> None:
    answer = "The dot product is used here [1]. It has 332 dimensions in total [1]."
    text = build_report(answer, [passage(1, DOT)], embedder_factory=lambda: FakeEmbedder(EmbeddingError("no model")))
    assert "Verification: ok" in text
    assert "1 number_missing" in text and "1 unchecked" in text
    assert "support score unavailable: no model" in text
