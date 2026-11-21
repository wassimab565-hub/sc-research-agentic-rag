# src\sc_research_agentic_rag\agent\
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypedDict

from langgraph.graph import END, START, StateGraph

from sc_research_agentic_rag.agent.decisions import Decision, Step
from sc_research_agentic_rag.agent.tools import ToolResult
from sc_research_agentic_rag.retrieval import RetrievedChunk

if TYPE_CHECKING:
    from sc_research_agentic_rag.agent.pipeline import AgentAnswer

STOP_ANSWERED = "answered"
STOP_MAX_STEPS = "max_steps"
STOP_REPEATED = "repeated_action"
STOP_RETRIEVAL_LIMIT = "retrieval_limit"
STOP_EMPTY = "empty_results"
STOP_INVALID = "invalid_decision"

MAX_EMPTY_STREAK = 2
RECURSION_MARGIN = 5

DecideFn = Callable[[str, list[Step]], tuple[Decision, bool]]
RunToolFn = Callable[[Decision, Any], ToolResult]
FinishFn = Callable[[str, list[Step], list[RetrievedChunk], list[str], str], "AgentAnswer"]


class AgentState(TypedDict):
    question: str
    filters: Any
    steps: list[Step]
    chunks: dict[str, RetrievedChunk]
    notes: list[str]
    seen: list[str]
    retrievals: int
    empty_streak: int
    decision: Decision | None
    stop: str
    answer: Any


class AgentGraph:
    """decide -> act -> decide ... -> finish. `stop` stays empty until the loop must end."""

    def __init__(
        self, decide: DecideFn, run_tool: RunToolFn, finish: FinishFn, max_steps: int, max_retrievals: int
    ) -> None:
        self._decide = decide
        self._run_tool = run_tool
        self._finish = finish
        self._max_steps = max_steps
        self._max_retrievals = max_retrievals

        graph = StateGraph(AgentState)
        graph.add_node("decide", self._decide_node)
        graph.add_node("act", self._act_node)
        graph.add_node("finish", self._finish_node)
        graph.add_edge(START, "decide")
        graph.add_conditional_edges("decide", _next_after_decide, {"act": "act", "finish": "finish"})
        graph.add_conditional_edges("act", _next_after_act, {"decide": "decide", "finish": "finish"})
        graph.add_edge("finish", END)
        self._graph = graph.compile()

    def run(self, question: str, filters: Any = None) -> "AgentAnswer":
        state: AgentState = {
            "question": question,
            "filters": filters,
            "steps": [],
            "chunks": {},
            "notes": [],
            "seen": [],
            "retrievals": 0,
            "empty_streak": 0,
            "decision": None,
            "stop": "",
            "answer": None,
        }
        limit = 2 * self._max_steps + RECURSION_MARGIN
        final = self._graph.invoke(state, config={"recursion_limit": limit})
        return final["answer"]

    def _decide_node(self, state: AgentState) -> dict[str, Any]:
        decision, valid = self._decide(state["question"], state["steps"])
        update: dict[str, Any] = {"decision": decision}
        if decision.action == "answer":
            update["stop"] = STOP_ANSWERED if valid else STOP_INVALID
        elif decision.key in state["seen"]:
            update["stop"] = STOP_REPEATED
        elif decision.action == "retrieve_evidence" and state["retrievals"] >= self._max_retrievals:
            update["stop"] = STOP_RETRIEVAL_LIMIT
        return update

    def _act_node(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        result = self._run_tool(decision, state["filters"])

        steps = [*state["steps"], Step(len(state["steps"]) + 1, decision, result.summary, result.found)]
        chunks = dict(state["chunks"])
        for chunk in result.chunks:
            known = chunks.get(chunk.chunk_id)
            if known is None or chunk.score > known.score:
                chunks[chunk.chunk_id] = chunk
        notes = [*state["notes"], result.note] if result.note and result.found else state["notes"]
        empty_streak = 0 if result.found else state["empty_streak"] + 1

        stop = ""
        if empty_streak >= MAX_EMPTY_STREAK:
            stop = STOP_EMPTY
        elif len(steps) >= self._max_steps:
            stop = STOP_MAX_STEPS
        return {
            "steps": steps,
            "chunks": chunks,
            "notes": notes,
            "seen": [*state["seen"], decision.key],
            "retrievals": state["retrievals"] + (1 if decision.action == "retrieve_evidence" else 0),
            "empty_streak": empty_streak,
            "stop": stop,
        }

    def _finish_node(self, state: AgentState) -> dict[str, Any]:
        answer = self._finish(
            state["question"], state["steps"], list(state["chunks"].values()), state["notes"], state["stop"]
        )
        return {"answer": answer}


def _next_after_decide(state: AgentState) -> str:
    return "finish" if state["stop"] else "act"


def _next_after_act(state: AgentState) -> str:
    return "finish" if state["stop"] else "decide"
