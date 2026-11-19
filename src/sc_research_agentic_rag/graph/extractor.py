import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from sc_research_agentic_rag.document_processing.downloader import pdf_filename
from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation.synthetic import is_candidate
from sc_research_agentic_rag.graph.concepts import build_messages, parse_concepts
from sc_research_agentic_rag.llm import LLMClient

DEFAULT_CACHE_DIR = Path("data/graph/concepts")


class ConceptGraph(Protocol):
    def ensure_schema(self) -> None: ...
    def upsert_chunk_concepts(self, paper_uid: str, chunk_id: str, page: int, concepts: list[str]) -> None: ...


@dataclass
class ExtractStats:
    extracted: int = 0
    cached: int = 0
    filtered: int = 0
    failed: int = 0


def cache_path(cache_dir: Path, paper_uid: str) -> Path:
    return cache_dir / Path(pdf_filename(paper_uid)).with_suffix(".jsonl")


def text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def read_cache(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
            records[record["chunk_id"]] = record
        except (ValueError, KeyError, TypeError):
            continue
    return records


def _is_cached(cache: dict[str, dict[str, Any]], chunk: Chunk) -> bool:
    record = cache.get(chunk.chunk_id)
    return bool(record) and record.get("text_hash") == text_hash(chunk.text)


def _append(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_concepts(
    chunks: Sequence[Chunk],
    llm: LLMClient,
    cache_dir: Path,
    limit: int | None = None,
    on_progress: Callable[[ExtractStats], None] | None = None,
) -> ExtractStats:
    """Ask the LLM for the concepts of each chunk. Each answer is saved at once, so a rerun resumes."""
    stats = ExtractStats()
    caches: dict[str, dict[str, dict[str, Any]]] = {}
    for chunk in chunks:
        if not is_candidate(chunk):
            stats.filtered += 1
            continue
        path = cache_path(cache_dir, chunk.paper_uid)
        if chunk.paper_uid not in caches:
            caches[chunk.paper_uid] = read_cache(path)
        cache = caches[chunk.paper_uid]
        if _is_cached(cache, chunk):
            stats.cached += 1
            continue
        if limit is not None and stats.extracted + stats.failed >= limit:
            break

        concepts = parse_concepts(llm.chat(build_messages(chunk.text)), chunk.text)
        if concepts is None:
            stats.failed += 1
            continue
        record = {
            "chunk_id": chunk.chunk_id,
            "paper_uid": chunk.paper_uid,
            "page": chunk.page,
            "text_hash": text_hash(chunk.text),
            "concepts": concepts,
        }
        _append(path, record)
        cache[chunk.chunk_id] = record
        stats.extracted += 1
        if on_progress:
            on_progress(stats)
    return stats


def load_concepts(chunks: Sequence[Chunk], cache_dir: Path, graph: ConceptGraph) -> int:
    """Write cached concepts to the graph (no LLM). Chunks whose text changed since extraction are skipped."""
    graph.ensure_schema()
    caches: dict[str, dict[str, dict[str, Any]]] = {}
    loaded = 0
    for chunk in chunks:
        if chunk.paper_uid not in caches:
            caches[chunk.paper_uid] = read_cache(cache_path(cache_dir, chunk.paper_uid))
        cache = caches[chunk.paper_uid]
        if not _is_cached(cache, chunk):
            continue
        graph.upsert_chunk_concepts(chunk.paper_uid, chunk.chunk_id, chunk.page, cache[chunk.chunk_id]["concepts"])
        loaded += 1
    return loaded