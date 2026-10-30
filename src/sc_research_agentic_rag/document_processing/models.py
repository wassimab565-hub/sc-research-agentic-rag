from pydantic import BaseModel, Field, field_validator, model_validator


class Page(BaseModel):
    paper_uid: str
    page: int = Field(ge=1)
    text: str


class Chunk(BaseModel):
    paper_uid: str
    index: int = Field(ge=0)
    page: int = Field(ge=1)
    text: str
    char_start: int = Field(ge=0)
    char_end: int

    @field_validator("text")
    @classmethod
    def _text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chunk text must not be blank")
        return value

    @model_validator(mode="after")
    def _offsets_are_ordered(self) -> "Chunk":
        if self.char_end <= self.char_start:
            raise ValueError("char_end must be greater than char_start")
        return self

    @property
    def chunk_id(self) -> str:
        return f"{self.paper_uid}#{self.index}"