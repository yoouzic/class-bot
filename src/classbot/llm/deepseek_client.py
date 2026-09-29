from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from math import isfinite

import httpx

from classbot.config import Settings
from classbot.course import CourseProfile, load_course_profile

_SYSTEM_PROMPT = (
    "你是一个课堂问答助手。只输出直接、准确、简短的中文答案（1～3句），"
    "不输出推理、寒暄、标题或代码块；不确定时明确说不确定。"
    "问题可能来自语音识别，结合课程领域做最小纠错；不要跨学科联想，"
    "不要把猜测写成事实。优先先给教材中的结论，再补一句必要解释。"
)
_DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS = 1.0
_MAX_RATE_LIMIT_COOLDOWN_SECONDS = 30.0


class DeepSeekStreamError(RuntimeError):
    """Raised when DeepSeek returns an error or an incomplete SSE stream."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


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
        self._rate_limit_until = 0.0
        try:
            self._course_profile = load_course_profile(
                os.getenv("CLASSBOT_COURSE", "")
            )
        except (FileNotFoundError, ValueError):
            # ASR profile loading remains strict in ``main``.  The answer
            # client should still work without optional textbook context.
            self._course_profile = CourseProfile("")

    async def aclose(self) -> None:
        if self._owns_http_client:
            await self._http_client.aclose()

    async def stream_answer(
        self, question: str, subject: str
    ) -> AsyncIterator[str]:
        api_key = self._settings.deepseek_api_key
        if api_key is None or not api_key.strip():
            raise ValueError("DEEPSEEK_API_KEY is required")
        remaining_cooldown = self._rate_limit_until - time.monotonic()
        if remaining_cooldown > 0:
            raise DeepSeekStreamError(
                "DeepSeek API rate limit is active; "
                f"retry in {remaining_cooldown:.1f}s"
            )

        thinking_enabled = self._settings.thinking_effort != "none"
        references = self._course_profile.find_reference(question, limit=3)
        user_content = f"学科：{subject}\n问题：{question}"
        if references:
            reference_text = "\n".join(references)
            user_content += (
                "\n\n教材参考（优先依据其中的术语和结论；"
                "只保留必要的原文表述，不要整段照抄，也不要伪造教材未提供的引文）：\n"
                + reference_text[:4000]
            )

        payload = {
            "model": self._settings.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "stream": True,
            "thinking": {
                "type": "enabled" if thinking_enabled else "disabled"
            },
            "max_tokens": 512 if thinking_enabled else 120,
        }
        if thinking_enabled:
            payload["reasoning_effort"] = self._settings.thinking_effort
        else:
            payload["temperature"] = 0.2
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream",
        }

        for attempt in range(2):
            yielded_content = False
            try:
                done_received = False
                async with self._http_client.stream(
                    "POST",
                    "/chat/completions",
                    headers=headers,
                    json=payload,
                ) as response:
                    if response.status_code == 429:
                        cooldown = self._rate_limit_cooldown(response)
                        self._rate_limit_until = max(
                            self._rate_limit_until,
                            time.monotonic() + cooldown,
                        )
                        raise DeepSeekStreamError(
                            "DeepSeek API rate limit exceeded; "
                            f"retry in {cooldown:.1f}s"
                        )
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data:"):
                            continue

                        data = line[5:].strip()
                        if not data:
                            continue
                        if data == "[DONE]":
                            done_received = True
                            break

                        event = json.loads(data)
                        if not isinstance(event, dict):
                            continue

                        error_payload = event.get("error")
                        if error_payload is not None:
                            if isinstance(error_payload, dict):
                                message = error_payload.get("message")
                            else:
                                message = error_payload
                            if not isinstance(message, str) or not message:
                                message = "DeepSeek API returned an error"
                            safe_message = message.replace(api_key, "[REDACTED]")
                            raise DeepSeekStreamError(
                                f"DeepSeek API error: {safe_message}"
                            )

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
                            yielded_content = True
                            yield content

                if not done_received:
                    raise DeepSeekStreamError(
                        "DeepSeek stream ended before data: [DONE]",
                        retryable=True,
                    )
                if not yielded_content and attempt == 0:
                    continue
                return
            except httpx.TransportError:
                if attempt == 0 and not yielded_content:
                    continue
                raise
            except DeepSeekStreamError as error:
                if attempt == 0 and error.retryable and not yielded_content:
                    continue
                raise

    @staticmethod
    def _rate_limit_cooldown(response: httpx.Response) -> float:
        value = response.headers.get("retry-after")
        seconds: float | None = None
        if value:
            try:
                parsed_seconds = float(value)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(value)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=UTC)
                    seconds = (
                        retry_at - datetime.now(UTC)
                    ).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    seconds = None
            else:
                seconds = parsed_seconds

        if seconds is None or not isfinite(seconds) or seconds <= 0:
            return _DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
        return min(seconds, _MAX_RATE_LIMIT_COOLDOWN_SECONDS)
