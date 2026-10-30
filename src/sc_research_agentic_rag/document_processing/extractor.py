import logging
from pathlib import Path

from pypdf import PageObject, PdfReader
from pypdf.errors import DependencyError, PyPdfError

from sc_research_agentic_rag.document_processing.errors import ExtractionError
from sc_research_agentic_rag.document_processing.models import Page

logger = logging.getLogger(__name__)


def upload_uid(pdf_path: Path) -> str:
    return f"upload:{pdf_path.stem}"


def extract_pages(pdf_path: Path, paper_uid: str) -> list[Page]:
    try:
        reader = PdfReader(pdf_path)
        if reader.is_encrypted and not reader.decrypt(""):
            raise ExtractionError(f"{pdf_path.name} is password-protected")
        pages = [
            Page(paper_uid=paper_uid, page=number, text=_page_text(page, pdf_path.name, number))
            for number, page in enumerate(reader.pages, start=1)
        ]
    except (PyPdfError, DependencyError, OSError) as exc:
        raise ExtractionError(f"cannot read {pdf_path.name}: {exc}") from exc

    if not any(page.text.strip() for page in pages):
        raise ExtractionError(
            f"no extractable text in {pdf_path.name} (scanned PDF? OCR is not supported)"
        )
    return pages


def _page_text(page: PageObject, pdf_name: str, number: int) -> str:
    try:
        return page.extract_text() or ""
    except Exception:  # noqa: BLE001 - pypdf raises many error types on malformed pages
        logger.warning("could not extract text from %s, page %d", pdf_name, number)
        return ""