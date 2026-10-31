import logging

import pytest
from pypdf import PageObject, PdfReader, PdfWriter

from sc_research_agentic_rag.document_processing import ExtractionError, extract_pages, upload_uid
from sc_research_agentic_rag.document_processing.extractor import looks_fragmented


def test_extracts_pages_in_order_with_numbers(write_pdf) -> None:
    path = write_pdf(["Hello page one", "Second page text"])
    pages = extract_pages(path, "arxiv:1")
    assert [p.page for p in pages] == [1, 2]
    assert [p.paper_uid for p in pages] == ["arxiv:1", "arxiv:1"]
    assert pages[0].text.strip() == "Hello page one"
    assert pages[1].text.strip() == "Second page text"


def test_empty_page_is_kept_to_preserve_numbering(write_pdf) -> None:
    pages = extract_pages(write_pdf(["First", "", "Third"]), "arxiv:1")
    assert [p.page for p in pages] == [1, 2, 3]
    assert pages[1].text.strip() == ""
    assert pages[2].text.strip() == "Third"


def test_pdf_without_any_text_is_rejected(write_pdf) -> None:
    with pytest.raises(ExtractionError, match="no extractable text"):
        extract_pages(write_pdf(["", ""]), "arxiv:1")


def test_invalid_pdf_is_rejected(tmp_path) -> None:
    path = tmp_path / "bad.pdf"
    path.write_bytes(b"this is not a pdf")
    with pytest.raises(ExtractionError):
        extract_pages(path, "arxiv:1")


def test_empty_file_is_rejected(tmp_path) -> None:
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    with pytest.raises(ExtractionError):
        extract_pages(path, "arxiv:1")


def test_missing_file_is_rejected(tmp_path) -> None:
    with pytest.raises(ExtractionError):
        extract_pages(tmp_path / "missing.pdf", "arxiv:1")


def test_password_protected_pdf_is_rejected(write_pdf, tmp_path) -> None:
    writer = PdfWriter()
    for page in PdfReader(write_pdf(["secret"])).pages:
        writer.add_page(page)
    writer.encrypt("pw", algorithm="RC4-128")
    encrypted = tmp_path / "locked.pdf"
    with encrypted.open("wb") as handle:
        writer.write(handle)
    with pytest.raises(ExtractionError, match="password-protected"):
        extract_pages(encrypted, "arxiv:1")


def test_failing_page_is_logged_and_left_empty(write_pdf, monkeypatch, caplog) -> None:
    original = PageObject.extract_text

    def flaky(self, *args, **kwargs):
        if self.page_number == 1:
            raise ValueError("broken page")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(PageObject, "extract_text", flaky)
    with caplog.at_level(logging.WARNING):
        pages = extract_pages(write_pdf(["One", "Two"]), "arxiv:1")
    assert [p.text.strip() for p in pages] == ["One", ""]
    assert "page 2" in caplog.text


def test_upload_uid_uses_file_stem(tmp_path) -> None:
    assert upload_uid(tmp_path / "My Paper.pdf") == "upload:My Paper"


FRAGMENTED = "\n\n".join(["RAPPORT", "DE", "TRAVAUX", "PRATIQUES", "IMAGE", "MINING"])
PROSE = "\n".join(["This is a normal line of text."] * 6)


@pytest.mark.parametrize(
    "text,expected",
    [
        (FRAGMENTED, True),
        (PROSE, False),
        ("One\nTwo\nThree", False),
        ("", False),
        ("\n".join(["Heading", "A normal sentence here.", "Another normal sentence."] * 3), False),
    ],
)
def test_looks_fragmented(text: str, expected: bool) -> None:
    assert looks_fragmented(text) is expected


def test_fragmented_text_falls_back_to_layout_mode(write_pdf, monkeypatch) -> None:
    def fake_extract(self, *args, **kwargs):
        if kwargs.get("extraction_mode") == "layout":
            return "   RAPPORT DE TRAVAUX      PRATIQUES\n       IMAGE MINING\n"
        return FRAGMENTED

    monkeypatch.setattr(PageObject, "extract_text", fake_extract)
    pages = extract_pages(write_pdf(["x"]), "arxiv:1")
    assert pages[0].text == "RAPPORT DE TRAVAUX PRATIQUES\nIMAGE MINING"


def test_normal_text_does_not_use_layout_mode(write_pdf, monkeypatch) -> None:
    def fake_extract(self, *args, **kwargs):
        if kwargs.get("extraction_mode") == "layout":
            raise AssertionError("layout mode must not be used")
        return PROSE

    monkeypatch.setattr(PageObject, "extract_text", fake_extract)
    assert extract_pages(write_pdf(["x"]), "arxiv:1")[0].text == PROSE