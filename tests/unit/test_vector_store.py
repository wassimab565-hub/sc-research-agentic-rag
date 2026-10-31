import uuid

import pytest
from qdrant_client import QdrantClient

from sc_research_agentic_rag.storage import (
    PointRecord,
    QdrantSettings,
    VectorStore,
    VectorStoreError,
    default_collection_name,
    point_id,
)
from sc_research_agentic_rag.storage import vector_store as vector_store_module


def make_store(collection: str = "chunks_test") -> VectorStore:
    return VectorStore(QdrantClient(":memory:"), collection)


def record(chunk_id: str, vector: list[float], **payload) -> PointRecord:
    return PointRecord(chunk_id=chunk_id, vector=vector, payload=payload)


@pytest.mark.parametrize(
    "model,expected",
    [
        ("BAAI/bge-m3", "chunks_baai_bge_m3"),
        ("intfloat/multilingual-e5-base", "chunks_intfloat_multilingual_e5_base"),
    ],
)
def test_default_collection_name(model: str, expected: str) -> None:
    assert default_collection_name(model) == expected


def test_point_id_is_a_deterministic_uuid() -> None:
    assert point_id("arxiv:1#0") == point_id("arxiv:1#0")
    assert point_id("arxiv:1#0") != point_id("arxiv:1#1")
    uuid.UUID(point_id("arxiv:1#0"))


def test_ensure_collection_creates_and_is_idempotent() -> None:
    store = make_store()
    store.ensure_collection(3)
    store.ensure_collection(3)
    assert store.count() == 0


def test_dimension_mismatch_is_rejected() -> None:
    store = make_store()
    store.ensure_collection(3)
    with pytest.raises(VectorStoreError, match="3.*4|4.*3"):
        store.ensure_collection(4)


def test_search_returns_closest_first_with_payload() -> None:
    store = make_store()
    store.ensure_collection(3)
    store.upsert(
        [
            record("a#0", [1, 0, 0], paper_uid="a", page=2),
            record("b#0", [0, 1, 0], paper_uid="b", page=1),
            record("c#0", [0.7, 0.7, 0], paper_uid="c", page=5),
        ]
    )
    hits = store.search([1, 0.1, 0], limit=2)
    assert [h.chunk_id for h in hits] == ["a#0", "c#0"]
    assert hits[0].score > hits[1].score
    assert hits[0].payload["paper_uid"] == "a"
    assert hits[0].payload["page"] == 2


def test_upsert_is_idempotent_and_updates_payload() -> None:
    store = make_store()
    store.ensure_collection(2)
    store.upsert([record("a#0", [1, 0], note="old")])
    store.upsert([record("a#0", [1, 0], note="new")])
    assert store.count() == 1
    assert store.search([1, 0])[0].payload["note"] == "new"


def test_upsert_in_batches() -> None:
    store = make_store()
    store.ensure_collection(2)
    records = [record(f"p#{i}", [1, i / 10], paper_uid="p") for i in range(10)]
    assert store.upsert(records, batch_size=3) == 10
    assert store.count() == 10


def test_delete_paper_only_removes_that_paper() -> None:
    store = make_store()
    store.ensure_collection(2)
    store.upsert(
        [
            record("a#0", [1, 0], paper_uid="a"),
            record("a#1", [1, 1], paper_uid="a"),
            record("b#0", [0, 1], paper_uid="b"),
        ]
    )
    store.delete_paper("a")
    assert store.count() == 1
    assert store.search([0, 1])[0].chunk_id == "b#0"


def test_operations_on_missing_collection_fail_clearly() -> None:
    store = make_store("missing")
    with pytest.raises(VectorStoreError, match="missing"):
        store.count()
    with pytest.raises(VectorStoreError):
        store.search([1.0])
    with pytest.raises(VectorStoreError):
        store.upsert([record("a#0", [1.0])])
    with pytest.raises(VectorStoreError):
        store.delete_paper("a")


def test_invalid_arguments() -> None:
    store = make_store()
    store.ensure_collection(2)
    with pytest.raises(ValueError):
        store.search([1, 0], limit=0)
    with pytest.raises(ValueError):
        store.upsert([], batch_size=0)


def test_local_storage_persists_between_sessions(tmp_path) -> None:
    path = str(tmp_path / "qdrant")
    first = VectorStore(QdrantClient(path=path), "chunks_test")
    first.ensure_collection(2)
    first.upsert([record("a#0", [1, 0], paper_uid="a")])
    first.close()

    second = VectorStore(QdrantClient(path=path), "chunks_test")
    assert second.count() == 1
    second.close()


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("PATH", "URL", "API_KEY", "COLLECTION"):
        monkeypatch.delenv(f"QDRANT_{key}", raising=False)


def test_settings_defaults(clean_env) -> None:
    settings = QdrantSettings(_env_file=None)
    assert settings.path == "data/qdrant"
    assert (settings.url, settings.api_key, settings.collection) == (None, None, None)


def test_blank_values_become_none(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "")
    monkeypatch.setenv("QDRANT_API_KEY", " ")
    monkeypatch.setenv("QDRANT_COLLECTION", "")
    settings = QdrantSettings(_env_file=None)
    assert (settings.url, settings.api_key, settings.collection) == (None, None, None)


def test_env_file_ignores_unrelated_keys(clean_env, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENALEX_API_KEY=abc\nQDRANT_PATH=somewhere\n", encoding="utf-8")
    assert QdrantSettings(_env_file=env_file).path == "somewhere"


def test_from_settings_local_mode_uses_derived_collection(clean_env, tmp_path) -> None:
    settings = QdrantSettings(_env_file=None, path=str(tmp_path / "q"))
    store = VectorStore.from_settings("BAAI/bge-m3", settings)
    assert store.collection == "chunks_baai_bge_m3"
    store.ensure_collection(2)
    assert store.count() == 0
    store.close()


def test_from_settings_collection_override(clean_env, tmp_path) -> None:
    settings = QdrantSettings(_env_file=None, path=str(tmp_path / "q"), collection="custom")
    store = VectorStore.from_settings("BAAI/bge-m3", settings)
    assert store.collection == "custom"
    store.close()


def test_from_settings_server_mode_passes_url_and_key(clean_env, monkeypatch) -> None:
    received: dict = {}

    def fake_client(**kwargs):
        received.update(kwargs)
        return QdrantClient(":memory:")

    monkeypatch.setattr(vector_store_module, "QdrantClient", fake_client)
    settings = QdrantSettings(_env_file=None, url="http://localhost:6333", api_key="secret")
    VectorStore.from_settings("BAAI/bge-m3", settings)
    assert received == {"url": "http://localhost:6333", "api_key": "secret"}


def test_count_paper_counts_only_that_paper() -> None:
    store = make_store()
    store.ensure_collection(2)
    store.upsert(
        [
            record("a#0", [1, 0], paper_uid="a"),
            record("a#1", [1, 1], paper_uid="a"),
            record("b#0", [0, 1], paper_uid="b"),
        ]
    )
    assert store.count_paper("a") == 2
    assert store.count_paper("b") == 1
    assert store.count_paper("missing") == 0