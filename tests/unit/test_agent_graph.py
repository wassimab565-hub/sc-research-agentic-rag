import pytest

from sc_research_agentic_rag.agent import Decision, Step, ToolResult
from sc_research_agentic_rag.agent.graph import AgentGraph
from sc_research_agentic_rag.retrieval import RetrievedChunk


def chunk(n: int, score: float | None = None) -> RetrievedChunk:
    text = f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"p#{n}", score=score if score is not None else 1.0 - n / 10, text=text,
        paper_uid=f"p{n}", page=1, char_start=0, char_end=len(text), title=f"Paper {n}",
    )


def retrieve(query: str) -> Decision:
    return Decision("retrieve_evidence", {"query": query})


ANSWER = Decision("answer")


class Harness:
    """Scripted decisions and tool results; finish returns a dict describing what it received."""

    def __init__(self, decisions, results=None, valid: bool = True, error: Exception | None = None, **kwargs) -> None:
        self.decisions = list(decisions)
        self.results = results or {}
        self.valid = valid
        self.error = error
        self.decide_calls: list[int] = []
        self.tool_calls: list[tuple] = []
        self.graph = AgentGraph(self.decide, self.run_tool, self.finish, **{"max_steps": 4, "max_retrievals": 3, **kwargs})

    def decide(self, question, steps):
        self.decide_calls.append(len(steps))
        return self.decisions.pop(0), self.valid

    def run_tool(self, decision, filters):
        self.tool_calls.append((decision, filters))
        if self.error:
            raise self.error
        return self.results.get(decision.args.get("query"), ToolResult("no passage found"))

    def finish(self, question, steps, chunks, notes, stop):
        return {"question": question, "steps": steps, "chunks": chunks, "notes": notes, "stop": stop}


def found(*chunks: RetrievedChunk, note: str = "") -> ToolResult:
    return ToolResult(f"{len(chunks)} passages", len(chunks), tuple(chunks), note)


def test_one_tool_call_then_answer() -> None:
    harness = Harness([retrieve("a"), ANSWER], {"a": found(chunk(1), chunk(2))})
    result = harness.graph.run("What?", None)
    assert result["question"] == "What?"
    assert result["stop"] == "answered"
    assert [(s.number, s.decision.args["query"], s.found) for s in result["steps"]] == [(1, "a", 2)]
    assert [c.chunk_id for c in result["chunks"]] == ["p#1", "p#2"]
    assert harness.decide_calls == [0, 1]


def test_answering_at_once_never_runs_a_tool() -> None:
    harness = Harness([ANSWER])
    result = harness.graph.run("q?", None)
    assert result["steps"] == []
    assert harness.tool_calls == []


def test_invalid_decision_is_reported_as_the_stop_reason() -> None:
    harness = Harness([ANSWER], valid=False)
    assert harness.graph.run("q?", None)["stop"] == "invalid_decision"


def test_repeated_action_stops_before_running_the_tool_again() -> None:
    harness = Harness([retrieve("DPR"), retrieve("dpr")], {"DPR": found(chunk(1))})
    result = harness.graph.run("q?", None)
    assert result["stop"] == "repeated_action"
    assert len(harness.tool_calls) == 1


def test_retrieval_limit() -> None:
    harness = Harness([retrieve("a"), retrieve("b")], {"a": found(chunk(1)), "b": found(chunk(2))}, max_retrievals=1)
    result = harness.graph.run("q?", None)
    assert result["stop"] == "retrieval_limit"
    assert len(harness.tool_calls) == 1


def test_max_steps() -> None:
    harness = Harness(
        [retrieve("a"), retrieve("b")], {"a": found(chunk(1)), "b": found(chunk(2))}, max_steps=2
    )
    result = harness.graph.run("q?", None)
    assert result["stop"] == "max_steps"
    assert len(result["steps"]) == 2
    assert harness.decide_calls == [0, 1]


def test_two_empty_results_in_a_row_stop() -> None:
    harness = Harness([retrieve("a"), retrieve("b")])
    result = harness.graph.run("q?", None)
    assert result["stop"] == "empty_results"
    assert len(result["steps"]) == 2


def test_an_empty_result_followed_by_a_hit_resets_the_streak() -> None:
    harness = Harness([retrieve("a"), retrieve("b"), retrieve("c"), ANSWER], {"b": found(chunk(1))})
    result = harness.graph.run("q?", None)
    assert result["stop"] == "answered"
    assert [s.found for s in result["steps"]] == [0, 1, 0]


def test_chunks_keep_the_best_score_and_notes_are_kept_only_with_a_hit() -> None:
    harness = Harness(
        [retrieve("a"), retrieve("b"), retrieve("c"), ANSWER],
        {
            "a": found(chunk(1, 0.2), chunk(2), note="note A"),
            "b": found(chunk(1, 0.9), note="note B"),
            "c": ToolResult("nothing", 0, note="ignored"),
        },
    )
    result = harness.graph.run("q?", None)
    assert sorted(c.chunk_id for c in result["chunks"]) == ["p#1", "p#2"]
    assert {c.chunk_id: c.score for c in result["chunks"]}["p#1"] == 0.9
    assert result["notes"] == ["note A", "note B"]


def test_filters_reach_the_tool() -> None:
    harness = Harness([retrieve("a"), ANSWER], {"a": found(chunk(1))})
    harness.graph.run("q?", "the-filters")
    assert harness.tool_calls[0][1] == "the-filters"


def test_tool_errors_propagate() -> None:
    harness = Harness([retrieve("a")], error=RuntimeError("store down"))
    with pytest.raises(RuntimeError, match="store down"):
        harness.graph.run("q?", None)


def test_a_long_run_stays_inside_the_recursion_limit() -> None:
    decisions = [retrieve(f"q{n}") for n in range(30)]
    results = {f"q{n}": found(chunk(n)) for n in range(30)}
    harness = Harness(decisions, results, max_steps=30, max_retrievals=30)
    result = harness.graph.run("q?", None)
    assert result["stop"] == "max_steps"
    assert len(result["steps"]) == 30
