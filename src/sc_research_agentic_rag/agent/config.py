# src\sc_research_agentic_rag\agent\
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_", env_file=".env", extra="ignore")

    max_steps: int = 4
    max_retrievals: int = 3
    top_k: int = 5
    list_size: int = 5
