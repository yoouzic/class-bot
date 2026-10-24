import asyncio
import json
import logging
from contextlib import aclosing
from types import SimpleNamespace

import httpx
import pytest

from classbot.config import Settings
from classbot.domain import CardStatus, QuestionCandidate
from classbot.llm.deepseek_client import DeepSeekClient, DeepSeekStreamError
from classbot.pipeline import LivePipeline


def event(*, content=None, reasoning=None, finish=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return "data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}, ensure_ascii=False) + "\n\n"


def length_response(content=None, *, finish="length"):
    return httpx.Response(200, content=event(reasoning="private reasoning") + event(content=content, finish=finish) + "data: [DONE]\n")


def answer_response(text="当前资料规定了接口。[S1]"):
    return httpx.Response(200, content=event(content=text, finish="stop") + "data: [DONE]\n")


def settings(*, thinking="high"):
    return Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu",
                    thinking_effort=thinking, defense_thinking_effort=thinking)


def index(queries):
    def search(query, limit=5):
        queries.append(query)
        return (SimpleNamespace(source="规范.md · lines 1–2", text="接口规范只描述约定。"),)
    return SimpleNamespace(search=search)


def client_for(http, *, configuration=None, defense=True, queries=None):
    return DeepSeekClient(
        configuration or settings(), http_client=http,
        scenario=SimpleNamespace(name="答辩", kind="competition") if defense else None,
        knowledge_index=index(queries if queries is not None else []) if defense else None,
    )


@pytest.mark.asyncio
async def test_reasoning_only_length_retries_once_without_thinking_or_retrieval(caplog):
    requests, queries = [], []

    def handler(request):
        requests.append(json.loads(request.content))
        return length_response() if len(requests) == 1 else answer_response()

    configuration = settings()
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=configuration.base_url) as http:
        client = client_for(http, configuration=configuration, queries=queries)
        prepared = await client.prepare_answer_with_local_context("QUESTION_SENTINEL", "答辩", local_context="LOCAL_CONTEXT_SENTINEL")
        with caplog.at_level(logging.WARNING, logger="classbot.llm.deepseek_client"):
            answer = [part async for part in client.stream_answer("QUESTION_SENTINEL", "答辩", prepared=prepared)]
        await client.aclose()
    assert answer == ["当前资料规定了接口。[S1]"]
    assert len(requests) == 2 and len(queries) == 1
    assert "LOCAL_CONTEXT_SENTINEL" in queries[0]
    assert requests[0]["messages"] == requests[1]["messages"] == prepared.messages
    assert "LOCAL_CONTEXT_SENTINEL" not in json.dumps(requests[1]["messages"])
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "high" and requests[0]["max_tokens"] == 2048
    assert requests[1]["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in requests[1]
    assert requests[1]["temperature"] == 0.2 and requests[1]["max_tokens"] == 1000
    assert configuration.thinking_effort == "high"
    assert configuration.defense_thinking_effort == "high"
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1 and "fallback" in warnings[0] and "length" in warnings[0]
    assert all(secret not in warnings[0] for secret in ("QUESTION_SENTINEL", "LOCAL_CONTEXT_SENTINEL", "test-key", "规范.md"))


@pytest.mark.asyncio
async def test_partial_answer_length_does_not_retry_or_publish_unverified_card_text():
    requests, deltas = [], []

    def handler(request):
        requests.append(request)
        return length_response("已收到的正文 [S1]")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda _, answer: deltas.append(answer), scenario_name="答辩")
        await pipeline._handle_candidate(QuestionCandidate("q1", "接口有什么约定？", "答辩", 1, True))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    card = pipeline._states["q1"].card
    assert len(requests) == 1
    assert card.status is CardStatus.ERROR and card.answer == "" and deltas == []
    assert "finish_reason=length" in card.error


@pytest.mark.asyncio
@pytest.mark.parametrize("defense,thinking,finish", [(True, "none", "length"), (False, "high", "length"), (True, "high", "content_filter")])
async def test_fallback_is_limited_to_defense_thinking_length(defense, thinking, finish):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return length_response(finish=finish)

    configuration = settings(thinking=thinking)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=configuration.base_url) as http:
        client = client_for(http, configuration=configuration, defense=defense)
        with pytest.raises(DeepSeekStreamError, match=f"finish_reason={finish}"):
            _ = [part async for part in client.stream_answer("问题", "学科")]
        await client.aclose()
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "enabled" if thinking != "none" else "disabled"}


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["length", "transport", "empty"])
async def test_existing_two_attempt_limit_includes_fallback(first):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 and first == "transport":
            raise httpx.ConnectError("synthetic transport failure", request=request)
        if len(requests) == 1 and first == "empty":
            return httpx.Response(200, content="data: [DONE]\n")
        return length_response()

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)
        with pytest.raises(DeepSeekStreamError, match="finish_reason=length"):
            _ = [part async for part in client.stream_answer("问题", "答辩")]
        await client.aclose()
    assert len(requests) == 2
    assert requests[1]["thinking"] == {"type": "disabled" if first == "length" else "enabled"}


@pytest.mark.asyncio
async def test_cancellation_at_fallback_boundary_never_starts_second_request():
    requests = []

    class CancelOnClose(httpx.AsyncByteStream):
        def __init__(self):
            self.closed = False

        async def __aiter__(self):
            yield event(reasoning="reasoning only", finish="length").encode()
            await asyncio.Event().wait()

        async def aclose(self):
            if not self.closed:
                self.closed = True
                asyncio.current_task().cancel()

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=CancelOnClose())

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)

        async def consume():
            return [part async for part in client.stream_answer("问题", "答辩")]

        task = asyncio.create_task(consume())
        with pytest.raises(asyncio.CancelledError):
            await task
        await client.aclose()
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_fallback_still_rejects_unknown_source_ids_and_preserves_received_text():
    requests, chunks = [], []

    def handler(request):
        requests.append(json.loads(request.content))
        return length_response() if len(requests) == 1 else answer_response("未知引用 [S1,S99]")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)
        with pytest.raises(DeepSeekStreamError, match="S99"):
            async for part in client.stream_answer("问题", "答辩"):
                chunks.append(part)
        await client.aclose()
    assert len(requests) == 2 and chunks == ["未知引用 [S1,S99]"]


@pytest.mark.asyncio
async def test_prompt_preserves_subject_configuration_version_and_time_for_claims():
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: pytest.fail("no remote request expected"))) as http:
        client = client_for(http)
        prepared = await client.prepare_answer("某实现有什么限制？", "答辩")
        await client.aclose()
    system = prepared.messages[0]["content"]
    assert all(term in system for term in ("主语", "配置", "版本", "时间范围", "否定结论", "实验结论"))
    assert "转移" in system and "泛化" in system


@pytest.mark.asyncio
@pytest.mark.parametrize("first", ["length", "missing_done", "success"])
async def test_line_iterators_close_in_request_task_before_retry_or_return(first):
    closed, requests = [], []
    owner = asyncio.current_task()

    class TrackedResponse(httpx.Response):
        def __init__(self, label, content):
            super().__init__(200, content=content)
            self.label = label

        async def aiter_lines(self):
            async with aclosing(super().aiter_lines()) as lines:
                try:
                    async for line in lines:
                        yield line
                finally:
                    closed.append((self.label, asyncio.current_task()))

    def handler(request):
        requests.append(request)
        if len(requests) == 2:
            assert closed == [("first", owner)], "first line iterator outlived its request task boundary"
            return TrackedResponse("second", event(content="回答[S1]") + "data: [DONE]\n")
        content = {
            "length": event(reasoning="reasoning only", finish="length") + "data: [DONE]\n",
            "missing_done": event(reasoning="reasoning only"),
            "success": event(content="回答[S1]") + "data: [DONE]\n",
        }[first]
        return TrackedResponse("first", content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)
        result = [part async for part in client.stream_answer("问题", "答辩")]
        expected = ["first"] if first == "success" else ["first", "second"]
        assert [label for label, _ in closed] == expected
        assert all(task is owner for _, task in closed)
        await client.aclose()
    assert result == ["回答[S1]"]


@pytest.mark.asyncio
async def test_caller_closing_partial_stream_synchronously_closes_line_iterator():
    closed = []
    owner = asyncio.current_task()

    class TrackedResponse(httpx.Response):
        async def aiter_lines(self):
            try:
                yield event(content="已收到的部分正文").strip()
                await asyncio.Event().wait()
            finally:
                closed.append(asyncio.current_task())

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: TrackedResponse(200)), base_url=settings().base_url,
    ) as http:
        client = client_for(http)
        stream = client.stream_answer("问题", "答辩")
        assert await anext(stream) == "已收到的部分正文"
        await stream.aclose()
        assert closed == [owner]
        await client.aclose()


@pytest.mark.asyncio
async def test_cancelling_blocked_stream_closes_lines_in_cancelled_request_task():
    reading, closed, requests, received = asyncio.Event(), [], [], []

    class TrackedResponse(httpx.Response):
        async def aiter_lines(self):
            try:
                yield event(content="已收到的正文").strip()
                reading.set()
                await asyncio.Event().wait()
            finally:
                closed.append(asyncio.current_task())

    def handler(request):
        requests.append(request)
        return TrackedResponse(200)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings().base_url) as http:
        client = client_for(http)

        async def consume():
            async for part in client.stream_answer("问题", "答辩"):
                received.append(part)

        task = asyncio.create_task(consume())
        await asyncio.wait_for(reading.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [task]
        await client.aclose()
    assert received == ["已收到的正文"] and len(requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("legacy,override,defense,expected_effort,expected_budget", [
    ("high", None, True, "none", 1000),
    ("none", "high", True, "high", 2048),
    ("high", "low", True, "low", 2048),
    ("low", "none", False, "low", 1024),
    ("none", "high", False, "none", 120),
])
async def test_request_uses_independent_mode_specific_thinking(legacy, override, defense, expected_effort, expected_budget):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return answer_response("当前资料不足。")

    arguments = {} if override is None else {"defense_thinking_effort": override}
    configuration = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu", thinking_effort=legacy, **arguments)
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=configuration.base_url) as http:
        client = client_for(http, configuration=configuration, defense=defense)
        _ = [part async for part in client.stream_answer("问题", "学科")]
        await client.aclose()
    payload = requests[0]
    assert len(requests) == 1 and payload["max_tokens"] == expected_budget
    assert payload["thinking"] == {"type": "disabled" if expected_effort == "none" else "enabled"}
    if expected_effort == "none":
        assert "reasoning_effort" not in payload and payload["temperature"] == 0.2
    else:
        assert payload["reasoning_effort"] == expected_effort and "temperature" not in payload
    assert configuration.thinking_effort == legacy
