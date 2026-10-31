import json
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from sc_research_agentic_rag.document_processing.chunking import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_OVERLAP,
    chunk_pages,
)
from sc_research_agentic_rag.document_processing.cleaning import clean_pages
from sc_research_agentic_rag.document_processing.downloader import (
    DownloadStatus,
    PdfDownloader,
    pdf_filename,
)
from sc_research_agentic_rag.document_processing.errors import ExtractionError
from sc_research_agentic_rag.document_processing.extractor import extract_pages, upload_uid
from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.ingestion.models import Paper


class ProcessStatus(StrEnum):
    PROCESSED = "processed"
    EXISTS = "exists"
    NO_PDF = "no_pdf"
    FAILED = "failed"


@dataclass(frozen=True)
class ProcessResult:
    paper_uid: str
    status: ProcessStatus
    chunks: int = 0
    path: Path | None = None
    message: str | None = None


def load_papers(path: Path) -> list[Paper]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return [Paper.model_validate(item) for item in payload["papers"]]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"cannot load papers from {path}: {exc}") from exc


def chunks_path(output_dir: Path, paper_uid: str) -> Path:
    return output_dir / Path(pdf_filename(paper_uid)).with_suffix(".json")


def process_papers(
    papers: list[Paper],
    downloader: PdfDownloader,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    force: bool = False,
) -> list[ProcessResult]:
    return [
        _process_paper(paper, downloader, output_dir, chunk_size, overlap, force)
        for paper in papers
    ]


def process_uploads(
    pdf_dir: Path,
    output_dir: Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    force: bool = False,
) -> list[ProcessResult]:
    if not pdf_dir.is_dir():
        return []
    results = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        uid = upload_uid(pdf_path)
        target = chunks_path(output_dir, uid)
        if target.exists() and not force:
            results.append(ProcessResult(uid, ProcessStatus.EXISTS, path=target))
        else:
            results.append(_chunk_pdf(pdf_path, uid, None, output_dir, chunk_size, overlap))
    return results


def _process_paper(
    paper: Paper,
    downloader: PdfDownloader,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
    force: bool,
) -> ProcessResult:
    target = chunks_path(output_dir, paper.uid)
    if target.exists() and not force:
        return ProcessResult(paper.uid, ProcessStatus.EXISTS, path=target)

    download = downloader.download(paper)
    if download.status == DownloadStatus.NO_PDF_URL:
        return ProcessResult(paper.uid, ProcessStatus.NO_PDF)
    if download.path is None:
        return ProcessResult(paper.uid, ProcessStatus.FAILED, message=download.message)
    return _chunk_pdf(download.path, paper.uid, paper, output_dir, chunk_size, overlap)


def _chunk_pdf(
    pdf_path: Path,
    uid: str,
    paper: Paper | None,
    output_dir: Path,
    chunk_size: int,
    overlap: int,
) -> ProcessResult:
    try:
        chunks = chunk_pages(clean_pages(extract_pages(pdf_path, uid)), chunk_size, overlap)
        path = _save_chunks(output_dir, uid, paper, chunks, chunk_size, overlap)
    except (ExtractionError, OSError) as exc:
        return ProcessResult(uid, ProcessStatus.FAILED, message=str(exc))
    return ProcessResult(uid, ProcessStatus.PROCESSED, chunks=len(chunks), path=path)


def _save_chunks(
    output_dir: Path,
    uid: str,
    paper: Paper | None,
    chunks: list[Chunk],
    chunk_size: int,
    overlap: int,
) -> Path:
    payload = {
        "paper_uid": uid,
        "paper": paper.model_dump(mode="json") if paper else None,
        "processed_at": datetime.now(UTC).isoformat(),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "chunks": [{**chunk.model_dump(mode="json"), "chunk_id": chunk.chunk_id} for chunk in chunks],
    }
    path = chunks_path(output_dir, uid)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".json.part")
    partial.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    partial.replace(path)
    return path