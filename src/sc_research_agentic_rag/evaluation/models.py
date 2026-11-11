from pydantic import BaseModel, Field, field_validator, model_validator


class Evidence(BaseModel):
    paper_uid: str
    page: int | None = Field(default=None, ge=1)
    contains: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = None

    @field_validator("contains")
    @classmethod
    def _contains_not_blank(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("contains must not be blank")
        return value

    @model_validator(mode="after")
    def _offsets_go_together(self) -> "Evidence":
        if (self.char_start is None) != (self.char_end is None):
            raise ValueError("char_start and char_end must be given together")
        if self.char_start is not None and self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self


class EvalQuestion(BaseModel):
    id: str
    question: str
    evidence: list[Evidence] = Field(min_length=1)
    source: str = "manual"

    @field_validator("question")
    @classmethod
    def _question_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value