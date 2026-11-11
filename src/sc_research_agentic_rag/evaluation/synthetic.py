import random
import re
from collections import defaultdict
from collections.abc import Iterator, Sequence

from sc_research_agentic_rag.document_processing.models import Chunk
from sc_research_agentic_rag.evaluation.models import EvalQuestion, Evidence
from sc_research_agentic_rag.llm import ChatMessage, LLMClient

MIN_CHUNK_CHARS = 500
MAX_ATTEMPT_FACTOR = 3
NGRAM = 6

SYSTEM_PROMPT = "You write evaluation questions for a document retrieval system."
USER_TEMPLATE = (
    "Passage:\n{passage}\n\n"
    "Write ONE question that this passage answers. Rules: the question must make sense on its own, "
    "without the passage (never say 'this passage', 'the text' or 'the authors' without naming the subject); "
    "do not copy phrases of more than four words from the passage; "
    "write the question in the same language as the passage; output only the question."
)

_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_PREFIX = re.compile(r"^(?:question|q)\s*[:\-]\s*", re.IGNORECASE)
_WORD = re.compile(r"\w+", re.UNICODE)


def is_candidate(chunk: Chunk) -> bool:
    text, lower = chunk.text, chunk.text.lower()
    if len(text) < MIN_CHUNK_CHARS:
        return False
    if len(_YEAR.findall(text)) >= 6 or "arxiv preprint" in lower:
        return False
    if lower.count("et al.") >= 4 or lower.count("proceedings") >= 2 or text.count("@") >= 2:
        return False
    digits = sum(character.isdigit() for character in text)
    return digits / len(text) <= 0.3


def clean_question(raw: str) -> str | None:
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        return None
    question = _PREFIX.sub("", lines[0]).strip().strip('"\u201c\u201d*').strip()
    if "?" not in question:
        return None
    question = question[: question.index("?") + 1]
    question = " ".join(question.split())
    return question if 15 <= len(question) <= 300 else None


def copies_passage(question: str, passage: str, n: int = NGRAM) -> bool:
    words = [word.casefold() for word in _WORD.findall(question)]
    source = [word.casefold() for word in _WORD.findall(passage)]
    grams = {tuple(source[i : i + n]) for i in range(len(source) - n + 1)}
    return any(tuple(words[i : i + n]) in grams for i in range(len(words) - n + 1))


def generate_questions(
    chunks: Sequence[Chunk], llm: LLMClient, count: int, seed: int = 42
) -> Iterator[EvalQuestion]:
    if count < 1:
        raise ValueError("count must be >= 1")

    by_paper: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        if is_candidate(chunk):
            by_paper[chunk.paper_uid].append(chunk)
    rng = random.Random(seed)
    queues = [rng.sample(group, len(group)) for _, group in sorted(by_paper.items())]

    produced = attempts = 0
    max_attempts = count * MAX_ATTEMPT_FACTOR
    while produced < count and attempts < max_attempts and any(queues):
        for queue in queues:
            if not queue or produced >= count or attempts >= max_attempts:
                continue
            chunk = queue.pop()
            attempts += 1
            raw = llm.chat(
                [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", USER_TEMPLATE.format(passage=chunk.text))]
            )
            question = clean_question(raw)
            if question is None or copies_passage(question, chunk.text):
                continue
            produced += 1
            yield EvalQuestion(
                id=f"syn-{produced:04d}",
                question=question,
                source="synthetic",
                evidence=[
                    Evidence(
                        paper_uid=chunk.paper_uid, page=chunk.page,
                        char_start=chunk.char_start, char_end=chunk.char_end,
                    )
                ],
            )