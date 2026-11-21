# src\sc_research_agentic_rag\agent\
import json
import re
from dataclasses import dataclass, field

ACTIONS = ("retrieve_evidence", "graph_lookup", "search_papers", "answer")
GRAPH_KINDS = ("concept", "author", "related")
MAX_ARG_CHARS = 300

_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


@dataclass(frozen=True)
class Decision:
    action: str
    args: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        """Same action with the same arguments (case ignored) gives the same key."""
        return json.dumps([self.action, self.args], sort_keys=True, ensure_ascii=False).casefold()


@dataclass(frozen=True)
class Step:
    number: int
    decision: Decision
    summary: str
    found: int


def parse_decision(raw: str) -> Decision | None:
    """The decision in the LLM answer, or None when it is not one JSON object with sound arguments."""
    text = _FENCE.sub("", raw)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    if not isinstance(data, dict):
        return None

    action = data.get("action")
    if action not in ACTIONS:
        return None
    args = data["args"] if "args" in data else {key: value for key, value in data.items() if key != "action"}
    if not isinstance(args, dict):
        return None
    clean = _clean_args(action, args)
    return None if clean is None else Decision(action, clean)


def _clean_args(action: str, args: dict) -> dict[str, str] | None:
    if action == "answer":
        return {}
    if action == "graph_lookup":
        kind, value = _text(args.get("kind")), _text(args.get("value"))
        if kind is None or kind.casefold() not in GRAPH_KINDS or value is None:
            return None
        return {"kind": kind.casefold(), "value": value}
    query = _text(args.get("query"))
    return {"query": query} if query else None


def _text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = " ".join(value.split())[:MAX_ARG_CHARS].strip()
    return cleaned or None
