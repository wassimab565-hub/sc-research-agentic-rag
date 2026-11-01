from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    def chat(self, messages: Sequence[ChatMessage]) -> str: ...