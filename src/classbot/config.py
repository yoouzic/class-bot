import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_THINKING_EFFORTS = {"none", "low", "high", "max"}
_DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "classbot"


@dataclass(frozen=True, slots=True)
class Settings:
    deepseek_api_key: str | None
    subject: str
    model: str
    base_url: str
    asr_device: str
    thinking_effort: str = "none"
    audio_source: str = "microphone"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    cache_dir: Path = _DEFAULT_CACHE_DIR
    startup_mode: str = "defense"
    defense_thinking_effort: str = "none"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        thinking_effort = os.getenv("CLASSBOT_THINKING", "none").strip().lower()
        if thinking_effort == "disabled":
            thinking_effort = "none"
        if thinking_effort not in _THINKING_EFFORTS:
            thinking_effort = "none"
        defense_thinking_effort = os.getenv("CLASSBOT_DEFENSE_THINKING", "none").strip().lower()
        if defense_thinking_effort == "disabled" or defense_thinking_effort not in _THINKING_EFFORTS:
            defense_thinking_effort = "none"
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            subject=os.getenv("CLASSBOT_SUBJECT", "历史"),
            model=os.getenv("CLASSBOT_MODEL", "deepseek-flash"),
            base_url=os.getenv("CLASSBOT_BASE_URL", "https://api.deepseek.com").rstrip(
                "/"
            ),
            asr_device=os.getenv("CLASSBOT_ASR_DEVICE", "cpu"),
            thinking_effort=thinking_effort,
            embedding_model=os.getenv(
                "CLASSBOT_EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"
            ).strip(),
            cache_dir=Path(os.getenv("CLASSBOT_CACHE_DIR") or str(_DEFAULT_CACHE_DIR))
            .expanduser()
            .resolve(),
            startup_mode=os.getenv("CLASSBOT_MODE", "defense").strip().lower(),
            defense_thinking_effort=defense_thinking_effort,
        )
