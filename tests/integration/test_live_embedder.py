import math
import os

import pytest

from sc_research_agentic_rag.embeddings import Embedder

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="set RUN_MODEL_TESTS=1 to load the real embedding model (first run downloads ~2.3 GB)",
)


def dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_real_model_dimension_normalization_and_semantics() -> None:
    embedder = Embedder.from_settings()
    relevant = "Dense retrieval encodes questions and passages into vectors and compares them with a dot product."
    unrelated = "The recipe calls for two cups of flour, three eggs and a pinch of salt."
    passages = embedder.embed_passages([relevant, unrelated])

    assert embedder.dimension == 1024
    assert all(math.isclose(math.sqrt(dot(v, v)), 1.0, abs_tol=1e-3) for v in passages)

    english = embedder.embed_query("How does dense retrieval compare questions with passages?")
    assert dot(english, passages[0]) > dot(english, passages[1])

    french = embedder.embed_query("Comment la recherche dense compare-t-elle les questions et les passages ?")
    assert dot(french, passages[0]) > dot(french, passages[1])