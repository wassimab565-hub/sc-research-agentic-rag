import sc_research_agentic_rag
from config.settings import Settings
from sc_research_agentic_rag.main import main


def test_package_version() -> None:
    assert sc_research_agentic_rag.__version__ == "0.1.0"


def test_settings_defaults() -> None:
    s = Settings(_env_file=None)
    assert s.app_env == "development"
    assert s.log_level == "INFO"


def test_main_runs(capsys) -> None:
    main()
    assert "SC-Research-Agentic-RAG" in capsys.readouterr().out


def test_openalex_key_is_read_from_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENALEX_API_KEY", "abc")
    assert Settings(_env_file=None).openalex_api_key == "abc"