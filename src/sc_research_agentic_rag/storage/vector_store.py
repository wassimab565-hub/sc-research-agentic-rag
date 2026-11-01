import re
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from sc_research_agentic_rag.storage.config import QdrantSettings
from sc_research_agentic_rag.storage.errors import VectorStoreError


def default_collection_name(model_name: str) -> str:
    return "chunks_" + re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")


def point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _paper_filter(paper_uid: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key="paper_uid", match=models.MatchValue(value=paper_uid))]
    )


@dataclass(frozen=True)
class PointRecord:
    chunk_id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass(frozen=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: dict[str, Any]


@dataclass(frozen=True)
class SearchFilter:
    paper_uids: Sequence[str] | None = None
    sources: Sequence[str] | None = None
    year_min: int | None = None
    year_max: int | None = None

    def __post_init__(self) -> None:
        if self.year_min is not None and self.year_max is not None and self.year_min > self.year_max:
            raise ValueError("year_min must be <= year_max")

    def to_qdrant(self) -> models.Filter | None:
        must: list[models.Condition] = []
        if self.paper_uids:
            must.append(_match_any("paper_uid", self.paper_uids))
        if self.sources:
            must.append(_match_any("source", self.sources))
        if self.year_min is not None or self.year_max is not None:
            must.append(
                models.FieldCondition(key="year", range=models.Range(gte=self.year_min, lte=self.year_max))
            )
        return models.Filter(must=must) if must else None


def _match_any(key: str, values: Sequence[str]) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchAny(any=list(values)))


class VectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    @classmethod
    def from_settings(cls, model_name: str, settings: QdrantSettings | None = None) -> "VectorStore":
        settings = settings or QdrantSettings()
        if settings.url:
            client = QdrantClient(url=settings.url, api_key=settings.api_key)
        else:
            client = QdrantClient(path=settings.path)
        return cls(client, settings.collection or default_collection_name(model_name))

    @property
    def collection(self) -> str:
        return self._collection

    def ensure_collection(self, dimension: int) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                self._collection,
                vectors_config=models.VectorParams(size=dimension, distance=models.Distance.COSINE),
            )
            return

        existing = self._client.get_collection(self._collection).config.params.vectors
        size = existing.size if isinstance(existing, models.VectorParams) else None
        if size != dimension:
            raise VectorStoreError(
                f"collection '{self._collection}' has vector size {size}, expected {dimension}; "
                "use another collection or delete this one"
            )

    def upsert(self, records: Sequence[PointRecord], batch_size: int = 64) -> int:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._require_collection()
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            self._client.upsert(
                self._collection,
                points=[
                    models.PointStruct(
                        id=point_id(record.chunk_id),
                        vector=record.vector,
                        payload={**record.payload, "chunk_id": record.chunk_id},
                    )
                    for record in batch
                ],
            )
        return len(records)

    def count(self) -> int:
        self._require_collection()
        return self._client.count(self._collection, exact=True).count

    def count_paper(self, paper_uid: str) -> int:
        self._require_collection()
        return self._client.count(
            self._collection, count_filter=_paper_filter(paper_uid), exact=True
        ).count

    def delete_paper(self, paper_uid: str) -> None:
        self._require_collection()
        self._client.delete(
            self._collection,
            points_selector=models.FilterSelector(filter=_paper_filter(paper_uid)),
        )

    def search(
        self,
        vector: list[float],
        limit: int = 5,
        filters: SearchFilter | None = None,
        score_threshold: float | None = None,
        max_per_paper: int | None = None,
    ) -> list[SearchHit]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if max_per_paper is not None and max_per_paper < 1:
            raise ValueError("max_per_paper must be >= 1")
        self._require_collection()

        query_filter = filters.to_qdrant() if filters else None
        if max_per_paper is None:
            points = self._client.query_points(
                self._collection,
                query=vector,
                limit=limit,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).points
        else:
            groups = self._client.query_points_groups(
                self._collection,
                group_by="paper_uid",
                query=vector,
                limit=limit,
                group_size=max_per_paper,
                query_filter=query_filter,
                score_threshold=score_threshold,
                with_payload=True,
            ).groups
            points = sorted(
                (point for group in groups for point in group.hits),
                key=lambda point: point.score,
                reverse=True,
            )[:limit]

        return [
            SearchHit(
                chunk_id=(point.payload or {}).get("chunk_id", ""),
                score=point.score,
                payload=point.payload or {},
            )
            for point in points
        ]

    def close(self) -> None:
        self._client.close()

    def _require_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            raise VectorStoreError(
                f"collection '{self._collection}' does not exist; index some papers first"
            )