import asyncio
import json

import httpx
import pytest

import classbot.llm.deepseek_client as deepseek_module
from classbot.config import Settings
from classbot.course import CourseProfile
from classbot.llm.deepseek_client import DeepSeekClient, DeepSeekStreamError


def _settings(api_key: str | None = "test-key") -> Settings:
    return Settings(
        deepseek_api_key=api_key,
        subject="历史",
        model="deepseek-flash",
        base_url="https://api.deepseek.example",
        asr_device="cpu",
    )


async def _answer_chunks(client: DeepSeekClient, question: str, subject: str):
    return [
        chunk
        async for chunk in client.stream_answer(question=question, subject=subject)
    ]


@pytest.mark.asyncio
async def test_stream_answer_sends_request_contract_and_yields_sse_content():
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.path == "/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"
        assert request.headers["accept"] == "text/event-stream"

        payload = json.loads(request.content)
        assert payload["model"] == "deepseek-flash"
        assert payload["stream"] is True
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 120
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
        ]
        assert "1～3句" in payload["messages"][0]["content"]
        assert "不输出推理" in payload["messages"][0]["content"]
        assert "结合课程领域做最小纠错" in payload["messages"][0]["content"]
        assert "不要跨学科联想" in payload["messages"][0]["content"]
        assert "学科：历史" in payload["messages"][1]["content"]
        assert "马克思出生于哪一年？" in payload["messages"][1]["content"]

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream; charset=utf-8"},
            content=(
                "event: message\n"
                "\n"
                "data: {\"choices\": [{\"delta\": {\"role\": \"assistant\"}}]}\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"马克思\"}}]}\n"
                "data: {\"choices\": []}\n"
                "data: {\"usage\": {\"prompt_tokens\": 1}}\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"出生于\"}}]}\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"\"}}]}\n"
                "not-data: ignored\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"1818 年。\"}}]}\n"
                "data: [DONE]\n"
                "data: {\"choices\": [{\"delta\": {\"content\": \"不应出现\"}}]}\n"
            ),
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "马克思出生于哪一年？", "历史") == [
            "马克思",
            "出生于",
            "1818 年。",
        ]

    assert len(requests) == 1


@pytest.mark.asyncio
async def test_stream_answer_can_enable_low_thinking_effort():
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "low"
        assert "temperature" not in payload
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "简短回答"}}]}\n'
            "data: [DONE]\n",
        )

    settings = Settings(
        deepseek_api_key="test-key",
        subject="马克思主义基本原理",
        model="deepseek-flash",
        base_url="https://api.deepseek.example",
        asr_device="cpu",
        thinking_effort="low",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "物质和意识的关系是什么？", settings.subject) == [
            "简短回答"
        ]


@pytest.mark.asyncio
async def test_stream_answer_includes_relevant_textbook_reference(monkeypatch):
    profile = CourseProfile(
        course_id="demo",
        reference_chunks=(
            "[第2页] 物质决定意识，意识对物质具有能动的反作用。",
            "[第8页] 商品的价值由社会必要劳动时间决定。",
        ),
    )
    monkeypatch.setenv("CLASSBOT_COURSE", "demo")
    monkeypatch.setattr(
        deepseek_module, "load_course_profile", lambda _course_id: profile
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["thinking"] == {"type": "enabled"}
        assert payload["reasoning_effort"] == "high"
        assert payload["max_tokens"] == 1024
        user_message = payload["messages"][1]["content"]
        assert "教材参考" in user_message
        assert "物质决定意识，意识对物质具有能动的反作用" in user_message
        assert "商品的价值由社会必要劳动时间决定" not in user_message
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "简短回答"}}]}\n'
            "data: [DONE]\n",
        )

    settings = Settings(
        deepseek_api_key="test-key",
        subject="马克思主义基本原理",
        model="deepseek-flash",
        base_url="https://api.deepseek.example",
        asr_device="cpu",
        thinking_effort="high",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "物质和意识的关系是什么？", settings.subject) == [
            "简短回答"
        ]


@pytest.mark.asyncio
async def test_stream_answer_prioritizes_original_textbook_wording(
    monkeypatch,
):
    profile = CourseProfile(
        course_id="demo",
        reference_chunks=(
            "[第7页] 哲学的基本问题是思维和存在的关系问题。",
        ),
    )
    monkeypatch.setenv("CLASSBOT_COURSE", "demo")
    monkeypatch.setattr(
        deepseek_module, "load_course_profile", lambda _course_id: profile
    )

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        system_message = payload["messages"][0]["content"]
        user_message = payload["messages"][1]["content"]
        assert "教材原文优先" in system_message
        assert "不得伪造教材原文或页码" in system_message
        assert "教材原文候选" in user_message
        assert "[第7页] 哲学的基本问题是思维和存在的关系问题" in user_message
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "教材原文回答"}}]}\n'
            "data: [DONE]\n",
        )

    settings = Settings(
        deepseek_api_key="test-key",
        subject="马克思主义基本原理",
        model="deepseek-flash",
        base_url="https://api.deepseek.example",
        asr_device="cpu",
        thinking_effort="high",
    )
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(
            client, "哲学的基本问题是什么？", settings.subject
        ) == ["教材原文回答"]


@pytest.mark.asyncio
async def test_missing_api_key_fails_before_transport_is_called():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, content="data: [DONE]\n")

    settings = _settings(api_key=None)
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
            await _answer_chunks(client, "问题", "历史")

    assert calls == 0


@pytest.mark.asyncio
async def test_stream_raises_when_response_ends_before_done():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "半截"}}]}\n',
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(DeepSeekStreamError, match=r"\[DONE\]"):
            await _answer_chunks(client, "问题", "历史")


@pytest.mark.asyncio
async def test_stream_retries_once_after_network_error_before_any_content():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("temporary network failure", request=request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "重试成功"}}]}\n'
            "data: [DONE]\n",
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "问题", "历史") == ["重试成功"]

    assert calls == 2


@pytest.mark.asyncio
async def test_stream_retries_once_after_incomplete_stream_without_content():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: {\"choices\": []}\n",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "补发"}}]}\n'
            "data: [DONE]\n",
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "问题", "历史") == ["补发"]

    assert calls == 2


@pytest.mark.asyncio
async def test_incomplete_stream_after_content_is_not_retried_or_duplicated():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content='data: {"choices": [{"delta": {"content": "半截"}}]}\n',
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        chunks: list[str] = []
        with pytest.raises(DeepSeekStreamError, match=r"\[DONE\]"):
            async for chunk in client.stream_answer("问题", "历史"):
                chunks.append(chunk)

    assert chunks == ["半截"]
    assert calls == 1


@pytest.mark.asyncio
async def test_stream_retries_once_after_empty_completed_stream():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content="data: [DONE]\n",
            )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"choices": [{"delta": {"content": "重试后回答"}}]}\n'
                "data: [DONE]\n"
            ),
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "问题", "历史") == ["重试后回答"]

    assert calls == 2


@pytest.mark.asyncio
async def test_empty_completed_stream_retries_only_once():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: [DONE]\n",
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        assert await _answer_chunks(client, "问题", "历史") == []

    assert calls == 2


@pytest.mark.asyncio
async def test_rate_limit_uses_retry_after_and_skips_followup_request():
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            429,
            headers={"Retry-After": "2"},
            content=b"rate limited",
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(
            DeepSeekStreamError,
            match=r"rate limit.*retry in 2\.0s",
        ):
            await _answer_chunks(client, "问题", "历史")
        with pytest.raises(DeepSeekStreamError, match="rate limit"):
            await _answer_chunks(client, "另一个问题", "历史")

    assert calls == 1


@pytest.mark.parametrize("retry_after", [None, "not-a-duration"])
@pytest.mark.asyncio
async def test_rate_limit_uses_short_default_for_invalid_or_missing_retry_after(
    retry_after,
):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        headers = {} if retry_after is None else {"Retry-After": retry_after}
        return httpx.Response(429, headers=headers, content=b"rate limited")

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(
            DeepSeekStreamError,
            match=r"rate limit.*retry in",
        ) as error:
            await _answer_chunks(client, "问题", "历史")
        with pytest.raises(DeepSeekStreamError, match="rate limit"):
            await _answer_chunks(client, "另一个问题", "历史")

    assert "test-key" not in str(error.value)
    assert calls == 1


@pytest.mark.asyncio
async def test_stream_error_payload_raises_without_exposing_api_key():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                'data: {"error": '
                '{"message": "upstream failed: test-key"}}\n'
            ),
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(DeepSeekStreamError, match="upstream failed") as error:
            await _answer_chunks(client, "问题", "历史")

    assert "[REDACTED]" in str(error.value)
    assert "test-key" not in str(error.value)


class _BlockingStream(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.closed = False

    async def __aiter__(self):
        self.started.set()
        await asyncio.Event().wait()
        yield b"data: [DONE]\n\n"

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_cancelling_stream_propagates_and_closes_response_stream():
    response_stream = _BlockingStream()

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=response_stream,
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        task = asyncio.create_task(
            _answer_chunks(client, "问题", "历史")
        )
        await response_stream.started.wait()
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        assert response_stream.closed is True


@pytest.mark.parametrize("status_code", [401, 500])
@pytest.mark.asyncio
async def test_http_errors_are_raised(status_code: int):
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(
            status_code,
            headers={"content-type": "text/event-stream"},
            content=b"error",
        )

    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)
        with pytest.raises(httpx.HTTPStatusError) as error:
            await _answer_chunks(client, "问题", "历史")

    assert error.value.response.status_code == status_code
    assert calls == 1


@pytest.mark.asyncio
async def test_injected_client_is_not_closed_by_aclose():
    settings = _settings()
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content="data: [DONE]\n")
        ),
        base_url=settings.base_url,
    ) as http_client:
        client = DeepSeekClient(settings, http_client=http_client)

        await client.aclose()

        assert http_client.is_closed is False


@pytest.mark.asyncio
async def test_owned_client_exposes_base_url_timeout_and_can_close():
    settings = _settings()
    client = DeepSeekClient(settings)
    owned_http_client = client._http_client

    assert owned_http_client.base_url == httpx.URL(settings.base_url)
    assert owned_http_client.timeout.connect == 5.0
    assert owned_http_client.timeout.read == 30.0

    await client.aclose()

    assert owned_http_client.is_closed is True
