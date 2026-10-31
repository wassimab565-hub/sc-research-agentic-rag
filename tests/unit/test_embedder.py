import sys
import types

import numpy as np
import pytest

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError, EmbeddingSettings
from sc_research_agentic_rag.embeddings.config import DEFAULT_MODEL, DEFAULT_REVISION


class FakeModel:
    def __init__(self, dimension: int = 4) -> None:
        self.dimension = dimension
        self.calls: list[tuple[list[str], dict]] = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return np.array([[float(len(text))] * self.dimension for text in texts], dtype=np.float32)


def make_embedder(model: FakeModel | None = None, **kwargs) -> tuple[Embedder, FakeModel, list[int]]:
    fake = model or FakeModel()
    loads: list[int] = []

    def loader() -> FakeModel:
        loads.append(1)
        return fake

    return Embedder("fake/model", model_loader=loader, **kwargs), fake, loads


def test_embed_passages_applies_prefix_and_encode_options() -> None:
    embedder, fake, _ = make_embedder(batch_size=8, passage_prefix="passage: ")
    vectors = embedder.embed_passages(["ab", "abcd"])
    texts, kwargs = fake.calls[0]
    assert texts == ["passage: ab", "passage: abcd"]
    assert kwargs["batch_size"] == 8
    assert kwargs["normalize_embeddings"] is True
    assert kwargs["convert_to_numpy"] is True
    assert vectors == [[11.0] * 4, [13.0] * 4]
    assert all(isinstance(value, float) for value in vectors[0])


def test_embed_query_applies_query_prefix() -> None:
    embedder, fake, _ = make_embedder(query_prefix="query: ")
    vector = embedder.embed_query("hi")
    assert fake.calls[0][0] == ["query: hi"]
    assert len(vector) == 4


@pytest.mark.parametrize("query", ["", "   \n"])
def test_blank_query_is_rejected(query: str) -> None:
    embedder, fake, _ = make_embedder()
    with pytest.raises(ValueError):
        embedder.embed_query(query)
    assert fake.calls == []


def test_empty_input_does_not_load_the_model() -> None:
    embedder, _, loads = make_embedder()
    assert embedder.embed_passages([]) == []
    assert loads == []


def test_model_is_loaded_lazily_and_only_once() -> None:
    embedder, _, loads = make_embedder()
    assert loads == []
    embedder.embed_query("a")
    embedder.embed_passages(["b"])
    assert loads == [1]


def test_dimension_is_probed_once_then_cached() -> None:
    embedder, fake, _ = make_embedder(FakeModel(dimension=7))
    assert embedder.dimension == 7
    assert embedder.dimension == 7
    assert len(fake.calls) == 1


def test_dimension_is_learned_from_real_encoding() -> None:
    embedder, fake, _ = make_embedder(FakeModel(dimension=3))
    embedder.embed_passages(["x"])
    assert embedder.dimension == 3
    assert len(fake.calls) == 1


def test_invalid_batch_size() -> None:
    with pytest.raises(ValueError):
        Embedder("m", batch_size=0)


@pytest.mark.parametrize("error", [OSError("offline"), ValueError("bad"), ImportError("missing")])
def test_load_failure_becomes_embedding_error(error) -> None:
    def loader():
        raise error

    with pytest.raises(EmbeddingError, match="fake/model"):
        Embedder("fake/model", model_loader=loader).embed_query("x")


def test_missing_sentence_transformers_gives_clear_error(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(EmbeddingError, match="fake/model"):
        Embedder("fake/model").embed_query("x")


def test_default_loader_passes_model_revision_and_device(monkeypatch) -> None:
    received: dict = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs) -> None:
            received["name"] = name
            received.update(kwargs)

        def encode(self, texts, **kwargs):
            return np.zeros((len(texts), 2), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    Embedder("org/name", revision="abc123", device="cpu").embed_query("x")
    assert received == {"name": "org/name", "revision": "abc123", "device": "cpu"}


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("MODEL", "REVISION", "BATCH_SIZE", "QUERY_PREFIX", "PASSAGE_PREFIX", "DEVICE"):
        monkeypatch.delenv(f"EMBEDDING_{key}", raising=False)


def test_settings_defaults(clean_env) -> None:
    settings = EmbeddingSettings(_env_file=None)
    assert settings.model == DEFAULT_MODEL == "BAAI/bge-m3"
    assert settings.revision == DEFAULT_REVISION
    assert settings.batch_size == 16
    assert (settings.query_prefix, settings.passage_prefix, settings.device) == ("", "", None)


def test_settings_from_environment(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "4")
    monkeypatch.setenv("EMBEDDING_QUERY_PREFIX", "query: ")
    settings = EmbeddingSettings(_env_file=None)
    assert settings.model == "intfloat/multilingual-e5-base"
    assert settings.batch_size == 4
    assert settings.query_prefix == "query: "


def test_blank_revision_and_device_become_none(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("EMBEDDING_REVISION", "")
    monkeypatch.setenv("EMBEDDING_DEVICE", " ")
    settings = EmbeddingSettings(_env_file=None)
    assert settings.revision is None
    assert settings.device is None


def test_env_file_ignores_unrelated_keys(clean_env, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENALEX_API_KEY=abc\nAPP_ENV=development\nEMBEDDING_BATCH_SIZE=2\n", encoding="utf-8")
    assert EmbeddingSettings(_env_file=env_file).batch_size == 2


def test_from_settings_builds_embedder(clean_env, monkeypatch) -> None:
    received: dict = {}

    class FakeSentenceTransformer:
        def __init__(self, name, **kwargs) -> None:
            received["name"] = name
            received.update(kwargs)

        def encode(self, texts, **kwargs):
            received["batch_size"] = kwargs["batch_size"]
            return np.zeros((len(texts), 2), dtype=np.float32)

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    settings = EmbeddingSettings(_env_file=None, batch_size=3, device="cpu")
    embedder = Embedder.from_settings(settings)
    embedder.embed_query("x")
    assert embedder.model_name == "BAAI/bge-m3"
    assert received == {"name": "BAAI/bge-m3", "revision": DEFAULT_REVISION, "device": "cpu", "batch_size": 3}