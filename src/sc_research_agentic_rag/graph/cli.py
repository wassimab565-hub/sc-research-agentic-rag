# src\sc_research_agentic_rag\graph\
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sc_research_agentic_rag.graph.builder import build_graph
from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.graph.store import GraphStore

DEFAULT_CHUNKS_DIR = Path("data/chunks")


def build_store() -> GraphStore:
    return GraphStore.from_settings()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the metadata graph (papers, authors, categories) in Neo4j.")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--force", action="store_true", help="clear the metadata graph before rebuilding")
    args = parser.parse_args(argv)

    if not any(args.chunks_dir.glob("*.json")):
        print(f"error: no chunk files found in {args.chunks_dir}", file=sys.stderr)
        return 1

    store = None
    try:
        store = build_store()
        result = build_graph(args.chunks_dir, store, force=args.force)
        counts = store.counts()
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()

    for name, message in result.failed.items():
        print(f"failed: {name}: {message}", file=sys.stderr)
    print(f"{result.papers} papers written, {result.uploads_skipped} uploads skipped (no metadata)")
    print(", ".join(f"{name}: {count}" for name, count in sorted(counts.items())))
    return 1 if result.failed and not result.papers else 0