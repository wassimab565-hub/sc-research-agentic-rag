import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError
from sc_research_agentic_rag.storage.errors import VectorStoreError
from sc_research_agentic_rag.storage.indexer import Indexer, IndexStatus
from sc_research_agentic_rag.storage.vector_store import VectorStore

DEFAULT_CHUNKS_DIR = Path("data/chunks")


def build_services() -> tuple[Embedder, VectorStore]:
    embedder = Embedder.from_settings()
    return embedder, VectorStore.from_settings(embedder.model_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Embed processed chunks and index them in Qdrant.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--force", action="store_true", help="re-embed papers already indexed")
    args = parser.parse_args(argv)

    paths = sorted(args.chunks_dir.glob("*.json"))
    if not paths:
        print(f"no chunk files found in {args.chunks_dir}", file=sys.stderr)
        return 1

    embedder, store = build_services()
    try:
        indexer = Indexer(embedder, store)
        indexer.prepare()
        results = []
        for path in paths:
            result = indexer.index_file(path, force=args.force)
            results.append(result)
            _print_result(result)
        total = store.count()
        collection = store.collection
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    counts = Counter(result.status for result in results)
    print(f"\n{', '.join(f'{s.value}: {counts[s]}' for s in IndexStatus if counts[s])}")
    print(f"collection '{collection}' now holds {total} chunks")
    succeeded = counts[IndexStatus.INDEXED] + counts[IndexStatus.EXISTS]
    return 0 if succeeded else 1


def _print_result(result) -> None:
    line = f"[{result.status.value:<7}] {result.paper_uid}"
    if result.status != IndexStatus.FAILED:
        line += f"  ({result.chunks} chunks)"
    if result.message:
        line += f"  - {result.message}"
    print(line)