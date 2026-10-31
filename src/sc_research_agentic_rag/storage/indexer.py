import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pydantic import ValidationError

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.ingestion.models import Paper
from sc_research_agentic_rag.storage.vector_store import PointRecord, VectorStore


class PassageEmbedder(Protocol):
    model_name: str

    @property
    def dimension(self) -> int: ...

    def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...


class IndexStatus(StrEnum):
    INDEXED = "indexed"
    EXISTS = "exists"
    FAILED = "failed"


@dataclass(frozen=True)
class IndexResult:
    paper_uid: str
    status: IndexStatus
    chunks: int = 0
    message: str | None = None


class Indexer:
    def __init__(self, embedder: PassageEmbedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def prepare(self) -> None:
        self._store.ensure_collection(self._embedder.dimension)

    def index_files(self, paths: Sequence[Path], force: bool = False) -> list[IndexResult]:
        self.prepare()
        return [self.index_file(path, force) for path in paths]

    def index_file(self, path: Path, force: bool = False) -> IndexResult:
        try:
            paper_uid, paper, chunks = _load_chunk_file(path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return IndexResult(path.stem, IndexStatus.FAILED, message=f"invalid chunk file: {exc}")
        if not chunks:
            return IndexResult(paper_uid, IndexStatus.FAILED, message="file contains no chunks")

        if not force and self._store.count_paper(paper_uid) == len(chunks):
            return IndexResult(paper_uid, IndexStatus.EXISTS, chunks=len(chunks))

        vectors = self._embedder.embed_passages([chunk.text for chunk in chunks])
        records = [
            PointRecord(chunk.chunk_id, vector, _payload(chunk, paper, self._embedder.model_name))
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self._store.delete_paper(paper_uid)
        self._store.upsert(records)
        return IndexResult(paper_uid, IndexStatus.INDEXED, chunks=len(chunks))


def _load_chunk_file(path: Path) -> tuple[str, Paper | None, list[Chunk]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    paper = Paper.model_validate(data["paper"]) if data.get("paper") else None
    try:
        chunks = [Chunk.model_validate(item) for item in data["chunks"]]
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc
    return data["paper_uid"], paper, chunks


def _payload(chunk: Chunk, paper: Paper | None, model_name: str) -> dict[str, Any]:
    return {
        "paper_uid": chunk.paper_uid,
        "chunk_index": chunk.index,
        "page": chunk.page,
        "char_start": chunk.char_start,
        "char_end": chunk.char_end,
        "text": chunk.text,
        "title": paper.title if paper else chunk.paper_uid.removeprefix("upload:"),
        "authors": paper.authors if paper else [],
        "year": paper.year if paper else None,
        "doi": paper.doi if paper else None,
        "url": paper.url if paper else None,
        "pdf_url": paper.pdf_url if paper else None,
        "source": paper.source.value if paper else "upload",
        "embedding_model": model_name,
    }