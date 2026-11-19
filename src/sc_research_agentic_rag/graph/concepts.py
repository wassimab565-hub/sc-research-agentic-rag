# src\sc_research_agentic_rag\graph\
import json
import re

from sc_research_agentic_rag.graph.store import author_key as normalize_key
from sc_research_agentic_rag.llm import ChatMessage

MAX_CONCEPTS = 6
MIN_CHARS = 3
MAX_CHARS = 60
MAX_WORDS = 6
STOPWORDS = frozenset(
    {
        "model", "models", "method", "methods", "result", "results", "data", "paper", "approach",
        "system", "task", "work", "figure", "table", "performance", "training", "experiment",
        "modele", "modeles", "methode", "methodes", "resultat", "resultats", "donnees", "approche",
        "systeme", "tache", "travail", "figure", "tableau",
    }
)

GENERIC = frozenset(
    {"answer", "answers", "query", "queries", "span", "mask", "knowledge", "user input",
     "iteratively", "concatenation", "input", "output", "question", "questions", "document", "documents"}
)
_ARTICLES = frozenset({"a", "an", "the"})
_BAD_CHARS = "ˆ<>=_{}\\"

SYSTEM_PROMPT = "You extract key concepts from scientific passages for a knowledge graph."
USER_TEMPLATE = (
    "Passage:\n{passage}\n\n"
    f"List up to {MAX_CONCEPTS} key scientific concepts explicitly mentioned in the passage "
    "(methods, models, datasets, metrics, technical terms). Rules: copy each concept exactly as written "
    "in the passage; each concept must be a noun phrase or a name, never a verb phrase or a sentence; "
    "no generic words (like 'model' or 'results'); no math symbols or variables; "
    "no author names, no references; "
    'output ONLY a JSON array of strings, for example ["BM25", "dense retrieval"]; output [] if none.'
)

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


def build_messages(passage: str) -> list[ChatMessage]:
    return [ChatMessage("system", SYSTEM_PROMPT), ChatMessage("user", USER_TEMPLATE.format(passage=passage))]


def parse_concepts(raw: str, passage: str) -> list[str] | None:
    """Valid concepts found in the passage, or None when the answer is not a JSON array."""
    text = _FENCE.sub("", raw)
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end < start:
        return None
    try:
        items = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(items, list):
        return None

    haystack = normalize_key(passage)
    concepts: list[str] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, str):
            continue
        name = " ".join(item.split())
        key = normalize_key(name)
        if not _acceptable(name, key) or key in seen or key not in haystack:
            continue
        seen.add(key)
        concepts.append(name)
        if len(concepts) == MAX_CONCEPTS:
            break
    return concepts


def _acceptable(name: str, key: str) -> bool:
    return (
        MIN_CHARS <= len(name) <= MAX_CHARS
        and len(name.split()) <= MAX_WORDS
        and any(char.isalpha() for char in name)
        and key not in STOPWORDS
        and key not in GENERIC
        and not _ARTICLES & set(key.split())
        and not any(char in name for char in _BAD_CHARS)
    )