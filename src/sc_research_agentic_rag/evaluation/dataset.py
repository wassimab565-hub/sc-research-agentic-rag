import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation.metrics import matches
from sc_research_agentic_rag.evaluation.models import EvalQuestion

BROAD_EVIDENCE_THRESHOLD = 10


class DatasetError(Exception):
    """Raised when an evaluation dataset or chunk file cannot be read."""


@dataclass(frozen=True)
class ValidationIssue:
    question_id: str
    level: str
    message: str


def load_dataset(path: Path) -> list[EvalQuestion]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise DatasetError(f"cannot read {path}: {exc}") from exc

    questions: list[EvalQuestion] = []
    seen: set[str] = set()
    for number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            question = EvalQuestion.model_validate(json.loads(line))
        except (ValueError, ValidationError) as exc:
            raise DatasetError(f"{path}:{number}: {exc}") from exc
        if question.id in seen:
            raise DatasetError(f"{path}:{number}: duplicate question id '{question.id}'")
        seen.add(question.id)
        questions.append(question)
    return questions


def question_to_line(question: EvalQuestion) -> str:
    return json.dumps(question.model_dump(exclude_none=True), ensure_ascii=False)


def save_dataset(questions: Iterable[EvalQuestion], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(question_to_line(q) + "\n" for q in questions), encoding="utf-8")


def load_chunk_corpus(chunks_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(chunks_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            chunks.extend(Chunk.model_validate(item) for item in data["chunks"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise DatasetError(f"invalid chunk file {path}: {exc}") from exc
    return chunks


def validate_dataset(questions: Sequence[EvalQuestion], corpus: Sequence[Chunk]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for question in questions:
        for index, evidence in enumerate(question.evidence, start=1):
            count = sum(1 for chunk in corpus if matches(evidence, chunk))
            if count == 0:
                issues.append(ValidationIssue(question.id, "error", f"evidence #{index} matches no chunk"))
            elif count > BROAD_EVIDENCE_THRESHOLD:
                issues.append(
                    ValidationIssue(question.id, "warning", f"evidence #{index} matches {count} chunks (too broad?)")
                )
    return issues