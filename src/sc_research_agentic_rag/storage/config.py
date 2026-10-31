from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class QdrantSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QDRANT_", env_file=".env", extra="ignore")

    path: str = "data/qdrant"
    url: str | None = None
    api_key: str | None = None
    collection: str | None = None

    @field_validator("url", "api_key", "collection", mode="before")
    @classmethod
    def _blank_to_none(cls, value: str | None) -> str | None:
        if isinstance(value, str) and not value.strip():
            return None
        return value