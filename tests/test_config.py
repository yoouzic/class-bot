import pytest

from classbot.config import Settings


@pytest.fixture(autouse=True)
def clear_classbot_environment(monkeypatch):
    for name in (
        "DEEPSEEK_API_KEY",
        "CLASSBOT_SUBJECT",
        "CLASSBOT_MODEL",
        "CLASSBOT_BASE_URL",
        "CLASSBOT_ASR_DEVICE",
        "CLASSBOT_THINKING",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr("classbot.config.load_dotenv", lambda: None)


def test_from_env_reads_config_values(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("CLASSBOT_SUBJECT", "历史")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "test-key"
    assert settings.subject == "历史"
    assert settings.model == "deepseek-flash"
    assert settings.base_url == "https://api.deepseek.com"
    assert settings.thinking_effort == "none"


def test_from_env_returns_none_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.deepseek_api_key is None


def test_from_env_reads_low_thinking_effort(monkeypatch):
    monkeypatch.setenv("CLASSBOT_THINKING", "low")

    settings = Settings.from_env()

    assert settings.thinking_effort == "low"
