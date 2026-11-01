import re
from collections.abc import Sequence

import httpx

from sc_research_agentic_rag.llm.base import ChatMessage
from sc_research_agentic_rag.llm.config import OllamaSettings
from sc_research_agentic_rag.llm.errors import LLMError

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)


class OllamaClient:
    def __init__(self, settings: OllamaSettings | None = None, client: httpx.Client | None = None) -> None:
        self._settings = settings or OllamaSettings()
        self._client = client or httpx.Client(timeout=self._settings.timeout)
        self._base_url = self._settings.base_url.rstrip("/")

    @property
    def model(self) -> str:
        return self._settings.model

    def chat(self, messages: Sequence[ChatMessage]) -> str:
        if not messages:
            raise ValueError("messages must not be empty")
        data = self._request("POST", "/api/chat", json=self._chat_payload(messages))
        content = (data.get("message") or {}).get("content") or ""
        answer = _THINK_BLOCK.sub("", content).strip()
        if not answer:
            raise LLMError(
                f"empty answer from '{self.model}' (it may have used all its tokens thinking; "
                "increase OLLAMA_NUM_PREDICT or set OLLAMA_THINK=false)"
            )
        return answer

    def list_models(self) -> list[str]:
        data = self._request("GET", "/api/tags")
        return [item["name"] for item in data.get("models", []) if "name" in item]

    def _chat_payload(self, messages: Sequence[ChatMessage]) -> dict:
        settings = self._settings
        payload: dict = {
            "model": settings.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": False,
            "keep_alive": settings.keep_alive,
            "options": {
                "temperature": settings.temperature,
                "num_ctx": settings.num_ctx,
                "num_predict": settings.num_predict,
            },
        }
        if settings.think is not None:
            payload["think"] = settings.think
        return payload

    def _request(self, method: str, path: str, **kwargs) -> dict:
        url = self._base_url + path
        try:
            response = self._client.request(method, url, **kwargs)
        except httpx.ConnectError as exc:
            raise LLMError(f"cannot reach Ollama at {self._base_url}; is it running? ({exc})") from exc
        except httpx.TimeoutException as exc:
            raise LLMError(f"Ollama did not answer in time ({self._settings.timeout:.0f}s)") from exc
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc

        if response.status_code >= 400:
            raise LLMError(f"Ollama error {response.status_code}: {self._error_message(response)}")
        try:
            data = response.json()
        except ValueError as exc:
            raise LLMError(f"invalid response from Ollama: {exc}") from exc
        if not isinstance(data, dict):
            raise LLMError("invalid response from Ollama: expected a JSON object")
        return data

    def _error_message(self, response: httpx.Response) -> str:
        try:
            message = response.json().get("error")
        except (ValueError, AttributeError):
            message = None
        message = message or response.text or "no details"
        if response.status_code == 404 and "not found" in message:
            message += f" (download it with: ollama pull {self.model})"
        return message