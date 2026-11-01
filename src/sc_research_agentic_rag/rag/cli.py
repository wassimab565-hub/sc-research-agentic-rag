import argparse
import sys
from collections.abc import Sequence

from sc_research_agentic_rag.embeddings import Embedder, EmbeddingError
from sc_research_agentic_rag.llm import LLMError, OllamaClient
from sc_research_agentic_rag.rag.context import DEFAULT_MAX_CONTEXT_CHARS, SourcePassage
from sc_research_agentic_rag.rag.pipeline import RagAnswer, RagPipeline
from sc_research_agentic_rag.retrieval import Retriever
from sc_research_agentic_rag.storage import SearchFilter, VectorStore, VectorStoreError


def build_pipeline(
    top_k: int, max_context_chars: int, score_threshold: float | None, max_per_paper: int | None
) -> tuple[RagPipeline, VectorStore]:
    embedder = Embedder.from_settings()
    store = VectorStore.from_settings(embedder.model_name)
    pipeline = RagPipeline(
        Retriever(embedder, store),
        OllamaClient(),
        top_k=top_k,
        max_context_chars=max_context_chars,
        score_threshold=score_threshold,
        max_per_paper=max_per_paper,
    )
    return pipeline, store


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ask a question about the indexed papers.")
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--paper", action="append", metavar="UID", help="restrict to a paper (repeatable)")
    parser.add_argument("--source", action="append", choices=["arxiv", "openalex", "upload"])
    parser.add_argument("--year-min", type=int)
    parser.add_argument("--year-max", type=int)
    parser.add_argument("--threshold", type=float, help="minimum similarity score")
    parser.add_argument("--max-per-paper", type=int)
    parser.add_argument("--max-context-chars", type=int, default=DEFAULT_MAX_CONTEXT_CHARS)
    parser.add_argument("--show-sources", action="store_true", help="print every retrieved passage")
    args = parser.parse_args(argv)

    if not args.question.strip():
        parser.error("question must not be empty")
    if args.top_k < 1:
        parser.error("--top-k must be >= 1")
    if args.max_per_paper is not None and args.max_per_paper < 1:
        parser.error("--max-per-paper must be >= 1")
    if args.max_context_chars < 1:
        parser.error("--max-context-chars must be >= 1")
    try:
        filters = SearchFilter(
            paper_uids=args.paper, sources=args.source, year_min=args.year_min, year_max=args.year_max
        )
    except ValueError as exc:
        parser.error(str(exc))

    pipeline, store = build_pipeline(args.top_k, args.max_context_chars, args.threshold, args.max_per_paper)
    try:
        result = pipeline.ask(args.question, filters)
    except (EmbeddingError, VectorStoreError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    _print_result(result, args.show_sources)
    return 0


def _print_result(result: RagAnswer, show_sources: bool) -> None:
    print(result.answer)
    if not result.sources:
        return
    if result.cited_sources:
        print("\nSources:")
        for passage in result.cited_sources:
            print(f"  {_format_source(passage)}")
    else:
        print("\nWarning: the answer cites no source, treat it with caution.")
    if result.invalid_citations:
        numbers = ", ".join(f"[{n}]" for n in result.invalid_citations)
        print(f"Warning: the answer cites unknown sources: {numbers}")
    if show_sources:
        print("\nRetrieved passages:")
        for passage in result.sources:
            print(f"  {_format_source(passage)} (score {passage.score:.3f})")
            print(f"      {passage.text[:200].replace(chr(10), ' ')}")


def _format_source(passage: SourcePassage) -> str:
    authors = ""
    if passage.authors:
        authors = passage.authors[0] + (" et al." if len(passage.authors) > 1 else "")
    byline = ", ".join(part for part in (authors, str(passage.year) if passage.year else "") if part)
    link = passage.url or (f"https://doi.org/{passage.doi}" if passage.doi else "")
    line = f"[{passage.number}] {passage.title}"
    if byline:
        line += f" ({byline})"
    line += f", p. {passage.page}"
    return f"{line} - {link}" if link else line