import json

import pytest

from sc_research_agentic_rag.agent import AgentPipeline, AgentTools
from sc_research_agentic_rag.agent import cli
from sc_research_agentic_rag.llm import LLMError
from sc_research_agentic_rag.retrieval import RetrievedChunk


def chunk(n: int, **kwargs) -> RetrievedChunk:
    text = f"Text of source number {n}."
    return RetrievedChunk(
        chunk_id=f"p#{n}", score=1.0 - n / 10, text=text, paper_uid=f"p{n}", page=1,
        char_start=0, char_end=len(text), title=f"Paper {n}", **kwargs,
    )


def act(action: str, **args) -> str:
    return json.dumps({"action": action, "args": args})


class ScriptedLLM:
    def __init__(self, *replies) -> None:
        self.replies = list(replies)

    def chat(self, messages):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeRetriever:
    def retrieve(self, question, top_k=5, filters=None, score_threshold=None, max_per_paper=None):
        return [chunk(1, authors=["Ann", "Bob"], year=2024, url="https://x/1")]


def fake_build(llm, closed: list, received: list | None = None):
    def build(*args):
        if received is not None:
            received.extend(args)
        return AgentPipeline(llm, AgentTools(FakeRetriever())), lambda: closed.append(True)

    return build


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("AGENT_MAX_STEPS", "AGENT_MAX_RETRIEVALS", "AGENT_TOP_K", "RETRIEVAL_MODE", "RAG_SCORE_THRESHOLD"):
        monkeypatch.delenv(key, raising=False)


def test_cli_prints_answer_cited_sources_and_closes(monkeypatch, clean_env, capsys) -> None:
    closed: list = []
    llm = ScriptedLLM(act("retrieve_evidence", query="x"), act("answer"), "Answer [1].")
    monkeypatch.setattr(cli, "build_pipeline", fake_build(llm, closed))
    assert cli.main(["What?"]) == 0
    out = capsys.readouterr().out
    assert "Answer [1]." in out
    assert "[1] Paper 1 (Ann et al., 2024), p. 1 - https://x/1" in out
    assert "Agent steps" not in out
    assert closed == [True]


def test_cli_show_trace(monkeypatch, clean_env, capsys) -> None:
    llm = ScriptedLLM(act("retrieve_evidence", query="x"), act("answer"), "Answer [1].")
    monkeypatch.setattr(cli, "build_pipeline", fake_build(llm, []))
    cli.main(["What?", "--show-trace"])
    out = capsys.readouterr().out
    assert "Agent steps (stopped: answered):" in out
    assert "1. retrieve_evidence" in out


def test_cli_trace_without_steps(monkeypatch, clean_env, capsys) -> None:
    monkeypatch.setattr(cli, "build_pipeline", fake_build(ScriptedLLM(act("answer")), []))
    cli.main(["What?", "--show-trace"])
    out = capsys.readouterr().out
    assert "No relevant sources were found" in out
    assert "  none" in out


def test_cli_passes_options_to_the_builder(monkeypatch, clean_env) -> None:
    received: list = []
    monkeypatch.setattr(cli, "build_pipeline", fake_build(ScriptedLLM(act("answer")), [], received))
    cli.main(["q?", "--max-steps", "3", "--max-retrievals", "2", "--top-k", "4", "--threshold", "0.5", "--mode", "hybrid"])
    assert received == [3, 2, 4, 0.5, "hybrid"]


def test_cli_uses_the_agent_settings_by_default(monkeypatch, clean_env) -> None:
    received: list = []
    monkeypatch.setenv("AGENT_MAX_STEPS", "6")
    monkeypatch.setattr(cli, "build_pipeline", fake_build(ScriptedLLM(act("answer")), [], received))
    cli.main(["q?", "--threshold", "0.4", "--mode", "dense"])
    assert received[:3] == [6, 3, 5]


def test_cli_reports_errors_and_closes(monkeypatch, clean_env, capsys) -> None:
    closed: list = []
    monkeypatch.setattr(cli, "build_pipeline", fake_build(ScriptedLLM(LLMError("Ollama is down")), closed))
    assert cli.main(["What?"]) == 1
    assert "Ollama is down" in capsys.readouterr().err
    assert closed == [True]


@pytest.mark.parametrize(
    "argv",
    [
        ["  "],
        ["q", "--max-steps", "0"],
        ["q", "--max-retrievals", "0"],
        ["q", "--top-k", "0"],
        ["q", "--year-min", "2024", "--year-max", "2020"],
        ["q", "--source", "web"],
        ["q", "--mode", "bm25"],
    ],
)
def test_cli_rejects_invalid_arguments(clean_env, argv) -> None:
    with pytest.raises(SystemExit) as exc:
        cli.main(argv)
    assert exc.value.code == 2
