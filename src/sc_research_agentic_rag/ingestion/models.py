from datetime import UTC, date, datetime
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator

_DOI_PREFIXES = ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:")


class PaperSource(StrEnum):
    ARXIV = "arxiv"
    OPENALEX = "openalex"


class Paper(BaseModel):
    source: PaperSource
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    abstract: str | None = None
    published: date | None = None
    doi: str | None = None
    url: str
    pdf_url: str | None = None
    categories: list[str] = Field(default_factory=list)
    also_found_as: list[str] = Field(default_factory=list)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("title", "abstract")
    @classmethod
    def _clean_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return " ".join(value.split()) or None

    @field_validator("doi")
    @classmethod
    def _normalize_doi(cls, value: str | None) -> str | None:
        if not value:
            return None
        doi = value.strip().lower()
        for prefix in _DOI_PREFIXES:
            if doi.startswith(prefix):
                doi = doi[len(prefix):]
                break
        return doi or None

    @property
    def year(self) -> int | None:
        return self.published.year if self.published else None

    @property
    def uid(self) -> str:
        return f"{self.source.value}:{self.source_id}"