import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from dotenv import load_dotenv

from sc_research_agentic_rag.ingestion.arxiv_client import ArxivClient
from sc_research_agentic_rag.ingestion.openalex_client import OpenAlexClient
from sc_research_agentic_rag.ingestion.service import PaperSearcher, collect_papers, save_result

DEFAULT_OUTPUT_DIR = Path("data/raw/search_results")


def build_sources(names: Sequence[str], openalex_api_key: str | None) -> dict[str, PaperSearcher]:
    factories = {
        "arxiv": lambda: ArxivClient(),
        "openalex": lambda: OpenAlexClient(api_key=openalex_api_key),
    }
    return {name: factories[name]() for name in names}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Search arXiv and OpenAlex and save the results.")
    parser.add_argument("query")
    parser.add_argument("--max-results", type=int, default=10)
    parser.add_argument("--source", choices=["arxiv", "openalex", "all"], default="all")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    if not 1 <= args.max_results <= 100:
        parser.error("--max-results must be between 1 and 100")

    load_dotenv()
    api_key = os.environ.get("OPENALEX_API_KEY") or None
    names = ["arxiv", "openalex"] if args.source == "all" else [args.source]
    if "openalex" in names and api_key is None:
        print("warning: OPENALEX_API_KEY is not set (very small daily budget)", file=sys.stderr)

    sources = build_sources(names, api_key)
    result = collect_papers(args.query, sources, args.max_results)

    for name, message in result.errors.items():
        print(f"error [{name}]: {message}", file=sys.stderr)
    if len(result.errors) == len(sources):
        return 1

    for paper in result.papers:
        authors = ", ".join(paper.authors[:3]) + (" et al." if len(paper.authors) > 3 else "")
        print(f"[{paper.uid}] {paper.year or 'n/a'} | {paper.title}")
        if authors:
            print(f"    {authors}")

    path = save_result(result, args.output_dir)
    print(f"\n{len(result.papers)} papers saved to {path}")
    return 0