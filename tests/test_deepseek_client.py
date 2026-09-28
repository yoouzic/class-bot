import json

import httpx
import pytest

from classbot.config import Settings
from classbot.llm.deepseek_client import DeepSeekClient


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
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 120
        assert [message["role"] for message in payload["messages"]] == [
            "system",
            "user",
        ]
        assert "1～3句" in payload["messages"][0]["content"]
        assert "不输出推理" in payload["messages"][0]["content"]
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


@pytest.mark.parametrize("status_code", [401, 500])
@pytest.mark.asyncio
async def test_http_errors_are_raised(status_code: int):
    def handler(request: httpx.Request) -> httpx.Response:
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


@pytest.mark.asyncio
async def test_aclose_is_available_for_an_owned_client():
    client = DeepSeekClient(_settings())

    await client.aclose()
