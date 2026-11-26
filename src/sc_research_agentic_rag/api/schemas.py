# src\sc_research_agentic_rag\api\
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from sc_research_agentic_rag.storage import SearchFilter
from sc_research_agentic_rag.verification import DEFAULT_THRESHOLD


class FilterIn(BaseModel):
    paper_uids: list[str] | None = None
    sources: list[Literal["arxiv", "openalex", "upload"]] | None = None
    year_min: int | None = None
    year_max: int | None = None

    def to_search_filter(self) -> SearchFilter:
        """Raises ValueError when year_min > year_max (the API answers 422)."""
        return SearchFilter(
            paper_uids=self.paper_uids, sources=self.sources, year_min=self.year_min, year_max=self.year_max
        )


class QuestionIn(BaseModel):
    question: str
    mode: Literal["dense", "hybrid"] | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    threshold: float | None = None
    filters: FilterIn | None = None
    verify: bool = False
    support_threshold: float = Field(default=DEFAULT_THRESHOLD, ge=0, le=1)

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be empty")
        return value


class AskRequest(QuestionIn):
    max_per_paper: int | None = Field(default=None, ge=1)


class AgentRequest(QuestionIn):
    max_steps: int | None = Field(default=None, ge=1, le=10)
    max_retrievals: int | None = Field(default=None, ge=1, le=10)


class PassageOut(BaseModel):
    number: int
    paper_uid: str
    title: str
    authors: list[str]
    year: int | None
    doi: str | None
    url: str | None
    source: str
    page: int
    text: str
    score: float
    cited: bool


class ClaimOut(BaseModel):
    text: str
    citations: list[int]
    status: str
    support: float | None
    missing_numbers: list[str]
    found_in: list[int]


class VerificationOut(BaseModel):
    status: str
    valid: list[int]
    invalid: list[int]
    sentences: int
    uncited_sentences: list[str]
    claims: list[ClaimOut]
    note: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[PassageOut]
    cited_numbers: list[int]
    invalid_citations: list[int]
    verification: VerificationOut | None = None


class StepOut(BaseModel):
    number: int
    action: str
    args: dict[str, str]
    summary: str
    found: int


class AgentResponse(AskResponse):
    stop_reason: str
    trace: list[StepOut]


class PaperOut(BaseModel):
    paper_uid: str
    title: str
    authors: list[str]
    year: int | None
    source: str
    doi: str | None
    url: str | None
    chunks: int


class NameCountOut(BaseModel):
    name: str
    count: int


class ConceptPaperOut(BaseModel):
    uid: str
    title: str | None
    year: int | None
    authors: list[str]
    pages: list[int]
    chunks: int


class ConceptResponse(BaseModel):
    name: str
    papers: list[ConceptPaperOut]
    similar_concepts: list[NameCountOut]


class AuthorPaperOut(BaseModel):
    uid: str
    title: str | None
    year: int | None
    matched_authors: list[str]


class AuthorResponse(BaseModel):
    name: str
    matched_authors: list[str]
    papers: list[AuthorPaperOut]
    concepts: list[NameCountOut]


class RelatedPaperOut(BaseModel):
    uid: str
    title: str | None
    year: int | None
    score: float
    shared: list[str]


class RelatedResponse(BaseModel):
    uid: str
    papers: list[RelatedPaperOut]
