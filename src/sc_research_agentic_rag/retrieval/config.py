from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrievalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    mode: Literal["dense", "hybrid"] = "dense"

    @field_validator("mode", mode="before")
    @classmethod
    def _blank_to_default(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return "dense"
        return value
