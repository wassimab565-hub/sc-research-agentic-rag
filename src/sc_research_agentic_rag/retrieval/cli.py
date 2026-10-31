import argparse
import sys
from collections.abc import Sequence

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError
from sc_research_agentic_rag.retrieval.retriever import RetrievedChunk, Retriever
from sc_research_agentic_rag.storage import SearchFilter, VectorStore, VectorStoreError


def build_services() -> tuple[Embedder, VectorStore]:
    embedder = Embedder.from_settings()
    return embedder, VectorStore.from_settings(embedder.model_name)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search the indexed chunks.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper", action="append", metavar="UID", help="restrict to a paper (repeatable)")
    parser.add_argument("--source", action="append", choices=["arxiv", "openalex", "upload"])
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--threshold", type=float, help="minimum similarity score")
    parser.add_argument("--max-per-paper", type=int)
    parser.add_argument("--chars", type=int, default=300, help="characters of text to print")
    args = parser.parse_args(argv)

    if not args.question.strip():
        parser.error("question must not be empty")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_per_paper is not None and args.max_per_paper < 1:
        parser.error("--max-per-paper must be >= 1")
    try:
        filters = SearchFilter(
            paper_uids=args.paper, sources=args.source, year_min=args.year_min, year_max=args.year_max
        )
    except ValueError as exc:
        parser.error(str(exc))

    embedder, store = build_services()
    try:
        chunks = Retriever(embedder, store).retrieve(
            args.question,
            top_k=args.top_k,
            filters=filters,
            score_threshold=args.threshold,
            max_per_paper=args.max_per_paper,
        )
    except (EmbeddingError, VectorStoreError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    if not chunks:
        print("no results")
        return 0
    for rank, chunk in enumerate(chunks, start=1):
        _print_chunk(rank, chunk, args.chars)
    return 0


def _print_chunk(rank: int, chunk: RetrievedChunk, chars: int) -> None:
    print(f"{rank}. [{chunk.score:.3f}] {chunk.title} | p. {chunk.page} | {chunk.chunk_id}")
    authors = ", ".join(chunk.authors[:3]) + (" et al." if len(chunk.authors) > 3 else "")
    details = " | ".join(part for part in (authors, str(chunk.year) if chunk.year else "", chunk.url or "") if part)
    if details:
        print(f"   {details}")
    print(f"   {chunk.text[:chars].replace(chr(10), ' ')}")
    print()