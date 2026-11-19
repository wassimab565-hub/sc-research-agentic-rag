# src\sc_research_agentic_rag\graph\
import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation.dataset import DatasetError, load_chunk_corpus
from sc_research_agentic_rag.graph.errors import GraphError
from sc_research_agentic_rag.graph.extractor import DEFAULT_CACHE_DIR, ExtractStats, extract_concepts, load_concepts
from sc_research_agentic_rag.graph.store import GraphStore
from sc_research_agentic_rag.llm import LLMError, OllamaClient

DEFAULT_CHUNKS_DIR = Path("data/chunks")


def build_llm() -> OllamaClient:
    return OllamaClient()


def build_store() -> GraphStore:
    return GraphStore.from_settings()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract concepts from chunks (Ollama) and load them into Neo4j.")
    parser.add_argument("--load", action="store_true", help="load the extracted concepts into Neo4j (no LLM)")
    parser.add_argument("--limit", type=int, help="maximum number of new LLM calls")
    parser.add_argument("--paper", action="append", metavar="UID", help="restrict to a paper (repeatable)")
    parser.add_argument("--chunks-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")

    try:
        corpus = load_chunk_corpus(args.chunks_dir)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.paper:
        corpus = [chunk for chunk in corpus if chunk.paper_uid in args.paper]
    if not corpus:
        print(f"error: no chunks found in {args.chunks_dir}", file=sys.stderr)
        return 1
    return _load(args, corpus) if args.load else _extract(args, corpus)


def _progress(stats: ExtractStats) -> None:
    if stats.extracted % 10 == 0:
        print(f"  {stats.extracted} chunks done", flush=True)


def _extract(args: argparse.Namespace, corpus: list[Chunk]) -> int:
    try:
        stats = extract_concepts(corpus, build_llm(), args.cache_dir, args.limit, _progress)
    except LLMError as exc:
        print(f"error: {exc} (progress is saved: run the command again to resume)", file=sys.stderr)
        return 1
    print(
        f"{stats.extracted} extracted, {stats.cached} already cached, "
        f"{stats.filtered} filtered out, {stats.failed} failed"
    )
    return 0 if stats.extracted or stats.cached else 1


def _load(args: argparse.Namespace, corpus: list[Chunk]) -> int:
    store = None
    try:
        store = build_store()
        loaded = load_concepts(corpus, args.cache_dir, store)
        counts = store.counts()
    except GraphError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        if store is not None:
            store.close()
    print(f"{loaded} chunks loaded into the graph")
    print(", ".join(f"{name}: {count}" for name, count in sorted(counts.items())))
    return 0