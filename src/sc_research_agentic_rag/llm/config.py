from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:11434"
    model: str = "qwen3:8b"
    num_ctx: int = 8192
    num_predict: int = 1024
    temperature: float = 0.1
    think: bool | None = False
    keep_alive: str = "5m"
    timeout: float = 300.0

    @field_validator("think", mode="before")
    @classmethod
    def _blank_think_is_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value