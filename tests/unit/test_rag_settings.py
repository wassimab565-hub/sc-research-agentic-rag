import pytest

from sc_research_agentic_rag.rag import RagSettings, cli


@pytest.fixture
def clean_env(monkeypatch):
    monkeypatch.delenv("RAG_SCORE_THRESHOLD", raising=False)


def test_default_threshold_separates_the_observed_scores(clean_env) -> None:
    threshold = RagSettings(_env_file=None).score_threshold
    assert threshold == 0.4
    assert 0.348 < threshold < 0.444


def test_threshold_from_environment(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.55")
    assert RagSettings(_env_file=None).score_threshold == 0.55


@pytest.mark.parametrize("value", ["", "  "])
def test_blank_threshold_disables_it(clean_env, monkeypatch, value) -> None:
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", value)
    assert RagSettings(_env_file=None).score_threshold is None


def test_env_file_ignores_unrelated_keys(clean_env, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENALEX_API_KEY=abc\nRAG_SCORE_THRESHOLD=0.3\n", encoding="utf-8")
    assert RagSettings(_env_file=env_file).score_threshold == 0.3


class _Pipeline:
    def ask(self, question, filters=None):
        from sc_research_agentic_rag.rag import RagAnswer

        return RagAnswer(question=question, answer="ok")


class _Store:
    def close(self) -> None:
        pass


def run_cli(monkeypatch, argv: list[str]) -> list:
    received: list = []

    def build(*args):
        received.extend(args)
        return _Pipeline(), _Store()

    monkeypatch.setattr(cli, "build_pipeline", build)
    cli.main(argv)
    return received


def test_cli_uses_the_configured_threshold_by_default(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.42")
    assert run_cli(monkeypatch, ["q?"])[2] == 0.42


def test_cli_threshold_option_overrides_the_setting(clean_env, monkeypatch) -> None:
    monkeypatch.setenv("RAG_SCORE_THRESHOLD", "0.42")
    assert run_cli(monkeypatch, ["q?", "--threshold", "0.1"])[2] == 0.1


def test_cli_threshold_zero_is_respected(clean_env, monkeypatch) -> None:
    assert run_cli(monkeypatch, ["q?", "--threshold", "0"])[2] == 0.0