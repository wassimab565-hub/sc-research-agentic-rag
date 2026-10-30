from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = PROJECT_ROOT / "data"
    openalex_api_key: str | None = None


settings = Settings()