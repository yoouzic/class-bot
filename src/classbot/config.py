import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class Settings:
    deepseek_api_key: str | None
    subject: str
    model: str
    base_url: str
    asr_device: str

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY") or None,
            subject=os.getenv("CLASSBOT_SUBJECT", "历史"),
            model=os.getenv("CLASSBOT_MODEL", "deepseek-flash"),
            base_url=os.getenv(
                "CLASSBOT_BASE_URL", "https://api.deepseek.com"
            ).rstrip("/"),
            asr_device=os.getenv("CLASSBOT_ASR_DEVICE", "cpu"),
        )
