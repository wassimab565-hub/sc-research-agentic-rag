import json

import httpx
import pytest

from sc_research_agentic_rag.llm import ChatMessage, LLMError, OllamaClient, OllamaSettings
from sc_research_agentic_rag.llm import cli


def make_client(handler, **settings) -> OllamaClient:
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaClient(OllamaSettings(_env_file=None, **settings), client=http)


def reply(content: str, status: int = 200):
    return lambda request: httpx.Response(status, json={"message": {"role": "assistant", "content": content}, "done": True})


MESSAGES = [ChatMessage("system", "Be brief."), ChatMessage("user", "Hello?")]


def test_sends_expected_chat_payload() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "Hi"}})

    answer = make_client(handler, model="qwen3:8b", num_ctx=4096, temperature=0.2).chat(MESSAGES)

    assert answer == "Hi"
    assert seen["url"] == "http://localhost:11434/api/chat"
    body = seen["body"]
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["messages"] == [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hello?"},
    ]
    assert body["options"] == {"temperature": 0.2, "num_ctx": 4096, "num_predict": 1024}
    assert body["keep_alive"] == "5m"


def test_think_is_omitted_when_none() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    make_client(handler, think=None).chat(MESSAGES)
    assert "think" not in seen["body"]


def test_custom_base_url_with_trailing_slash() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"message": {"content": "ok"}})

    make_client(handler, base_url="http://box:9999/").chat(MESSAGES)
    assert seen["url"] == "http://box:9999/api/chat"


def test_inline_think_blocks_are_removed() -> None:
    answer = make_client(reply("<think>\nlet me think\n</think>\n\nThe answer [1].")).chat(MESSAGES)
    assert answer == "The answer [1]."


@pytest.mark.parametrize("content", ["", "   ", "<think>only thoughts</think>"])
def test_empty_answers_are_errors(content: str) -> None:
    with pytest.raises(LLMError, match="empty answer"):
        make_client(reply(content)).chat(MESSAGES)


def test_missing_message_is_an_error() -> None:
    with pytest.raises(LLMError, match="empty answer"):
        make_client(lambda request: httpx.Response(200, json={"done": True})).chat(MESSAGES)


def test_connection_error_explains_the_problem() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError, match="is it running"):
        make_client(handler).chat(MESSAGES)


def test_timeout_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow")

    with pytest.raises(LLMError, match="in time"):
        make_client(handler, timeout=5).chat(MESSAGES)


def test_unknown_model_suggests_pull() -> None:
    handler = lambda request: httpx.Response(404, json={"error": "model 'nope' not found"})  # noqa: E731
    with pytest.raises(LLMError, match="ollama pull nope"):
        make_client(handler, model="nope").chat(MESSAGES)


def test_server_error_includes_message() -> None:
    handler = lambda request: httpx.Response(500, json={"error": "out of memory"})  # noqa: E731
    with pytest.raises(LLMError, match="500.*out of memory"):
        make_client(handler).chat(MESSAGES)


def test_non_json_error_body() -> None:
    handler = lambda request: httpx.Response(502, text="bad gateway")  # noqa: E731
    with pytest.raises(LLMError, match="bad gateway"):
        make_client(handler).chat(MESSAGES)


@pytest.mark.parametrize("response", [httpx.Response(200, text="not json"), httpx.Response(200, json=["x"])])
def test_invalid_success_body(response) -> None:
    with pytest.raises(LLMError, match="invalid response"):
        make_client(lambda request: response).chat(MESSAGES)


def test_empty_messages_are_rejected() -> None:
    with pytest.raises(ValueError):
        make_client(reply("x")).chat([])


def test_list_models() -> None:
    handler = lambda request: httpx.Response(200, json={"models": [{"name": "qwen3:8b"}, {"name": "llama3.2:latest"}]})  # noqa: E731
    assert make_client(handler).list_models() == ["qwen3:8b", "llama3.2:latest"]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ("BASE_URL", "MODEL", "NUM_CTX", "NUM_PREDICT", "TEMPERATURE", "THINK", "KEEP_ALIVE", "TIMEOUT", "HOST"):
        monkeypatch.delenv(f"OLLAMA_{key}", raising=False)


def test_settings_defaults(clean_env) -> None:
    s = OllamaSettings(_env_file=None)
    assert (s.base_url, s.model, s.num_ctx, s.think) == ("http://localhost:11434", "qwen3:8b", 8192, False)


def test_settings_from_env_and_blank_think(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_THINK", "")
    s = OllamaSettings(_env_file=None)
    assert s.model == "llama3.2:3b"
    assert s.think is None


def test_server_side_ollama_host_variable_does_not_break_the_client(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    assert OllamaSettings(_env_file=None).base_url == "http://localhost:11434"


class FakeClient:
    def __init__(self, installed=("qwen3:8b",), answer="A short answer.", error=None) -> None:
        self.installed, self.answer, self.error = list(installed), answer, error

    def list_models(self):
        return self.installed

    def chat(self, messages):
        if self.error:
            raise self.error
        self.last = messages
        return self.answer


def test_cli_prints_answer_and_timing(monkeypatch, clean_env, capsys) -> None:
    fake = FakeClient()
    monkeypatch.setattr(cli, "build_client", lambda settings: fake)
    assert cli.main(["Hello?"]) == 0
    out = capsys.readouterr().out
    assert "A short answer." in out
    assert "model qwen3:8b" in out
    assert fake.last == [ChatMessage("user", "Hello?")]


def test_cli_reports_missing_model(monkeypatch, clean_env, capsys) -> None:
    monkeypatch.setattr(cli, "build_client", lambda settings: FakeClient(installed=["llama3.2:3b"]))
    assert cli.main([]) == 1
    err = capsys.readouterr().err
    assert "ollama pull qwen3:8b" in err
    assert "llama3.2:3b" in err


def test_cli_accepts_latest_tag_and_model_override(monkeypatch, clean_env, capsys) -> None:
    monkeypatch.setattr(cli, "build_client", lambda settings: FakeClient(installed=["qwen3:latest"]))
    assert cli.main(["--model", "qwen3"]) == 0
    assert "model qwen3" in capsys.readouterr().out


def test_cli_reports_llm_errors(monkeypatch, clean_env, capsys) -> None:
    monkeypatch.setattr(cli, "build_client", lambda settings: FakeClient(error=LLMError("boom")))
    assert cli.main([]) == 1
    assert "boom" in capsys.readouterr().err