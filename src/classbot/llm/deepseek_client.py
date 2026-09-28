from __future__ import annotations

import json
from collections.abc import AsyncIterator

import httpx

from classbot.config import Settings

_SYSTEM_PROMPT = (
    "你是一个课堂问答助手。只输出直接、准确、简短的中文答案（1～3句），"
    "不输出推理、寒暄、标题或代码块；不确定时明确说不确定。"
)


class DeepSeekClient:
    def __init__(
        self,
        settings: Settings,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._settings = settings
        if http_client is None:
            self._http_client = httpx.AsyncClient(
                base_url=settings.base_url,
                timeout=httpx.Timeout(30.0, connect=5.0),
            )
            self._owns_http_client = True
        else:
            self._http_client = http_client
            self._owns_http_client = False

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def stream_answer(
        self, question: str, subject: str
    ) -> AsyncIterator[str]:
        api_key = self._settings.deepseek_api_key
        if api_key is None or not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is required")

        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"学科：{subject}\n问题：{question}",
                },
            ],
            "stream": True,
            "temperature": 0.2,
            "max_tokens": 120,
        }
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        }

        async with self._http_client.stream(
            "POST",
            "/chat/completions",
            headers=headers,
            json=payload,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue

                data = line[5:].strip()
                if not data:
                    continue
                if data == "[DONE]":
                    return

                event = json.loads(data)
                if not isinstance(event, dict):
                    continue
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue

                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue

                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content
