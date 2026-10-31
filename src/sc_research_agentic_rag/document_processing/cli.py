import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from sc_research_agentic_rag.document_processing.chunking import DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from sc_research_agentic_rag.document_processing.downloader import DEFAULT_PDF_DIR, PdfDownloader
from sc_research_agentic_rag.document_processing.pipeline import (
    ProcessResult,
    ProcessStatus,
    load_papers,
    process_papers,
    process_uploads,
)

DEFAULT_CHUNKS_DIR = Path("data/chunks")
DEFAULT_UPLOADS_DIR = Path("data/raw/uploaded_papers")


def build_downloader(pdf_dir: Path) -> PdfDownloader:
    return PdfDownloader(output_dir=pdf_dir)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download, clean and chunk papers.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-search", type=Path, metavar="JSON", help="search results file")
    mode.add_argument("--uploads", action="store_true", help="process PDFs in the uploads folder")
    parser.add_argument("--limit", type=int, help="only process the first N papers")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--force", action="store_true", help="reprocess existing outputs")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_CHUNKS_DIR)
    parser.add_argument("--pdf-dir", type=Path, default=DEFAULT_PDF_DIR)
    parser.add_argument("--uploads-dir", type=Path, default=DEFAULT_UPLOADS_DIR)
    args = parser.parse_args(argv)

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be >= 1")
    if not 0 <= args.overlap < args.chunk_size:
        parser.error("--overlap must be >= 0 and smaller than --chunk-size")

    options = {
        "output_dir": args.output_dir,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "force": args.force,
    }
    if args.from_search:
        try:
            papers = load_papers(args.from_search)
        except ValueError as exc:
            parser.error(str(exc))
        results = process_papers(papers[: args.limit], build_downloader(args.pdf_dir), **options)
    else:
        results = process_uploads(args.uploads_dir, **options)

    if not results:
        print("nothing to process", file=sys.stderr)
        return 1
    _report(results)

    counts = Counter(result.status for result in results)
    succeeded = counts[ProcessStatus.PROCESSED] + counts[ProcessStatus.EXISTS]
    return 1 if counts[ProcessStatus.FAILED] and not succeeded else 0


def _report(results: list[ProcessResult]) -> None:
    for result in results:
        line = f"[{result.status.value:<9}] {result.paper_uid}"
        if result.status == ProcessStatus.PROCESSED:
            line += f"  ({result.chunks} chunks)"
        if result.message:
            line += f"  - {result.message}"
        print(line)
    counts = Counter(result.status for result in results)
    summary = ", ".join(f"{status.value}: {counts[status]}" for status in ProcessStatus if counts[status])
    print(f"\n{summary}")