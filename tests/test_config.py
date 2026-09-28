from classbot.config import Settings


def test_from_env_reads_config_values(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("CLASSBOT_SUBJECT", "历史")

    settings = Settings.from_env()

    assert settings.deepseek_api_key == "test-key"
    assert settings.subject == "历史"
    assert settings.model == "deepseek-flash"
    assert settings.base_url == "https://api.deepseek.com"


def test_from_env_returns_none_when_api_key_is_missing(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    settings = Settings.from_env()

    assert settings.deepseek_api_key is None
