import json

import pytest

from sc_research_agentic_rag.agent import Decision, Step, parse_decision
from sc_research_agentic_rag.agent.decisions import MAX_ARG_CHARS
from sc_research_agentic_rag.agent.prompts import (
    ANSWER_SYSTEM_PROMPT,
    DECISION_SYSTEM_PROMPT,
    build_answer_messages,
    build_decision_messages,
)
from sc_research_agentic_rag.rag import SourcePassage


def test_parses_each_action() -> None:
    assert parse_decision('{"action": "retrieve_evidence", "args": {"query": "dense retrieval"}}') == Decision(
        "retrieve_evidence", {"query": "dense retrieval"}
    )
    assert parse_decision('{"action": "search_papers", "args": {"query": "RAG surveys"}}') == Decision(
        "search_papers", {"query": "RAG surveys"}
    )
    assert parse_decision('{"action": "graph_lookup", "args": {"kind": "Concept", "value": " BM25 "}}') == Decision(
        "graph_lookup", {"kind": "concept", "value": "BM25"}
    )
    assert parse_decision('{"action": "answer", "args": {"ignored": 1}}') == Decision("answer", {})


def test_accepts_code_fences_text_around_and_flat_arguments() -> None:
    assert parse_decision('Sure:\n```json\n{"action": "answer"}\n```') == Decision("answer")
    assert parse_decision('{"action": "retrieve_evidence", "query": "x  y"}') == Decision(
        "retrieve_evidence", {"query": "x y"}
    )


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "no json at all",
        "[1, 2]",
        '{"action": "retrieve_evidence"',
        '{"args": {"query": "x"}}',
        '{"action": "delete_everything", "args": {}}',
        '{"action": ["answer"], "args": {}}',
        '{"action": "retrieve_evidence", "args": "x"}',
        '{"action": "retrieve_evidence", "args": {}}',
        '{"action": "retrieve_evidence", "args": {"query": "   "}}',
        '{"action": "retrieve_evidence", "args": {"query": 5}}',
        '{"action": "graph_lookup", "args": {"kind": "paper", "value": "x"}}',
        '{"action": "graph_lookup", "args": {"kind": "concept"}}',
        '{"action": "graph_lookup", "args": {"value": "BM25"}}',
    ],
)
def test_invalid_decisions_return_none(raw: str) -> None:
    assert parse_decision(raw) is None


def test_long_arguments_are_cut() -> None:
    decision = parse_decision(json.dumps({"action": "search_papers", "args": {"query": "a" * 1000}}))
    assert len(decision.args["query"]) == MAX_ARG_CHARS


def test_key_ignores_case_and_argument_order() -> None:
    a = Decision("graph_lookup", {"kind": "concept", "value": "BM25"})
    b = Decision("graph_lookup", {"value": "bm25", "kind": "concept"})
    assert a.key == b.key
    assert a.key != Decision("graph_lookup", {"kind": "author", "value": "BM25"}).key
    assert Decision("retrieve_evidence", {"query": "x"}).key != Decision("search_papers", {"query": "x"}).key


def test_decision_messages_show_question_history_and_budget() -> None:
    step = Step(1, Decision("retrieve_evidence", {"query": "dpr"}), "2 passages", 2)
    system, user = build_decision_messages("What is DPR?", [step], steps_left=3)
    assert system.role == "system"
    assert system.content == DECISION_SYSTEM_PROMPT
    assert "Question: What is DPR?" in user.content
    assert '1. retrieve_evidence {"query": "dpr"} -> 2 passages' in user.content
    assert "Actions left: 3" in user.content
    assert "none yet" in build_decision_messages("q?", [], 4)[1].content


def test_decision_prompt_lists_every_action() -> None:
    for action in ("retrieve_evidence", "graph_lookup", "search_papers", "answer"):
        assert action in DECISION_SYSTEM_PROMPT


def passage(number: int = 1, **kwargs) -> SourcePassage:
    defaults = dict(
        number=number, paper_uid="a", title="Dense Retrieval", authors=("Ann", "Bob"), year=2020,
        doi=None, url=None, source="arxiv", page=3, text="Passage text.", chunk_ids=("a#0",), score=0.9,
    )
    return SourcePassage(**{**defaults, **kwargs})


def test_answer_messages_with_sources_and_notes() -> None:
    system, user = build_answer_messages("What?", [passage()], ["Papers that mention X:\n- [arxiv:1] T"])
    assert system.content == ANSWER_SYSTEM_PROMPT
    text = user.content
    assert "[1] Dense Retrieval (Ann et al., 2020) - page 3\nPassage text." in text
    assert "Database notes (not citable):\n\nPapers that mention X:" in text
    assert text.index("Sources:") < text.index("Database notes") < text.index("Question: What?")


def test_answer_messages_with_notes_only_or_sources_only() -> None:
    assert "Sources:" not in build_answer_messages("q?", [], ["note"])[1].content
    assert "Database notes" not in build_answer_messages("q?", [passage()], [])[1].content


@pytest.mark.parametrize("question,passages,notes", [("  ", [passage()], []), ("q?", [], [])])
def test_answer_messages_reject_invalid_input(question, passages, notes) -> None:
    with pytest.raises(ValueError):
        build_answer_messages(question, passages, notes)


def test_answer_prompt_keeps_the_grounding_rules() -> None:
    for expected in ("ONLY", "never cite them with a number", "do not answer from memory", "untrusted", "same language"):
        assert expected in ANSWER_SYSTEM_PROMPT


def test_blank_question_is_rejected_for_decisions() -> None:
    with pytest.raises(ValueError):
        build_decision_messages(" ", [], 3)
