import logging
import re
from pathlib import Path

from pypdf import PageObject, PdfReader
from pypdf.errors import DependencyError, PyPdfError

from sc_research_agentic_rag.document_processing.errors import ExtractionError
from sc_research_agentic_rag.document_processing.models import Page

logger = logging.getLogger(__name__)

_MIN_LINES_FOR_DETECTION = 5
_FRAGMENTED_RATIO = 0.6


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


def looks_fragmented(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) < _MIN_LINES_FOR_DETECTION:
        return False
    single_word = sum(1 for line in lines if len(line.split()) == 1)
    return single_word / len(lines) > _FRAGMENTED_RATIO


def _page_text(page: PageObject, pdf_name: str, number: int) -> str:
    try:
        text = page.extract_text() or ""
        if looks_fragmented(text):
            logger.info("%s page %d: fragmented text, using layout mode", pdf_name, number)
            text = _collapse_spaces(page.extract_text(extraction_mode="layout") or "")
        return text
    except Exception:  # noqa: BLE001 - pypdf raises many error types on malformed pages
        logger.warning("could not extract text from %s, page %d", pdf_name, number)
        return ""


def _collapse_spaces(text: str) -> str:
    return "\n".join(re.sub(r"[ \t]{2,}", " ", line).strip() for line in text.splitlines())