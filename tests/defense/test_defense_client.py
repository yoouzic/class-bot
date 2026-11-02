import asyncio
import json
import threading
from types import SimpleNamespace

import httpx
import pytest

from classbot.config import Settings
from classbot.llm.deepseek_client import DeepSeekClient, DeepSeekStreamError


class FakeIndex:
    def __init__(self):
        self.thread_id = None

    def search(self, query, limit=5):
        self.thread_id = threading.get_ident()
        return (
            SimpleNamespace(
                source="model.py:L1-L8", text="def solve_model(): return 42"
            ),
        )


@pytest.mark.asyncio
async def test_defense_request_has_local_sources_and_distinguishes_inference():
    index = FakeIndex()
    profile = SimpleNamespace(name="国赛答辩", kind="competition")
    settings = Settings(
        "test-key", "竞赛", "deepseek-flash", "https://api.example", "cpu"
    )
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(
            200,
            content='data: {"choices":[{"delta":{"content":"回答 [S1]"}}]}\n\ndata: [DONE]\n',
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url=settings.base_url
    ) as http:
        client = DeepSeekClient(
            settings, http_client=http, scenario=profile, knowledge_index=index
        )
        prepared = await client.prepare_answer("为什么返回42？", "竞赛")
        assert prepared.sources == ("[S1] model.py:L1-L8",)
        assert index.thread_id != threading.get_ident()
        answer = [
            part
            async for part in client.stream_answer(
                "为什么返回42？", "竞赛", prepared=prepared
            )
        ]
    assert answer == ["回答 [S1]"]
    system, user = captured[0]["messages"]
    assert "推断" in system["content"]
    assert "不可信" in system["content"]
    assert "model.py:L1-L8" in user["content"]
    assert "def solve_model" in user["content"]
    assert "教材" not in user["content"]
    assert captured[0]["max_tokens"] >= 800


@pytest.mark.asyncio
async def test_no_hits_is_explicit_without_invented_sources():
    index = SimpleNamespace(search=lambda *_args, **_kwargs: ())
    settings = Settings(
        "test-key", "组会", "deepseek-flash", "https://api.example", "cpu"
    )
    client = DeepSeekClient(
        settings,
        scenario=SimpleNamespace(name="组会", kind="group_meeting"),
        knowledge_index=index,
    )
    try:
        prepared = await client.prepare_answer("完成了什么实验？", "组会")
        assert not prepared.sources
        assert "未检索到" in prepared.messages[1]["content"]
    finally:
        await client.aclose()


def _client_for_index(index):
    return DeepSeekClient(
        Settings("test-key", "组会", "deepseek-flash", "https://api.example", "cpu"),
        scenario=SimpleNamespace(name="组会", kind="group_meeting"),
        knowledge_index=index,
    )


async def _wait_until(predicate):
    async with asyncio.timeout(3):
        while not predicate():
            await asyncio.sleep(0.001)


class _BlockingIndex:
    def __init__(self):
        self.release = threading.Event()
        self.queries = []
        self.active = 0
        self.maximum_active = 0
        self.lock = threading.Lock()

    def search(self, query, limit=5):
        with self.lock:
            self.queries.append(query)
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
        try:
            assert self.release.wait(5), "test failed to release retrieval thread"
            return ()
        finally:
            with self.lock:
                self.active -= 1


@pytest.mark.asyncio
async def test_cancellation_storm_never_queues_or_overlaps_stale_searches():
    index = _BlockingIndex()
    client = _client_for_index(index)
    latest = None
    first = asyncio.create_task(client.prepare_answer("first", "组会"))
    try:
        await _wait_until(lambda: index.queries)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        for revision in range(40):
            stale = asyncio.create_task(
                client.prepare_answer(f"stale-{revision}", "组会")
            )
            await asyncio.sleep(0)
            stale.cancel()
            with pytest.raises(asyncio.CancelledError):
                await stale
        latest = asyncio.create_task(client.prepare_answer("latest", "组会"))
        await asyncio.sleep(0.02)
        assert index.queries == ["first"]
        index.release.set()
        await latest
        assert index.queries == ["first", "latest"]
        assert index.maximum_active == 1
    finally:
        index.release.set()
        if not first.done():
            first.cancel()
        if latest is not None and not latest.done():
            latest.cancel()
        await asyncio.gather(
            first, *([latest] if latest is not None else []), return_exceptions=True
        )
        await client.aclose()


@pytest.mark.asyncio
async def test_prepare_answer_uses_index_async_search_lifecycle():
    calls = []

    async def search_async(question, limit=5):
        calls.append((question, limit))
        return ()

    def search(*_args, **_kwargs):
        pytest.fail("async-capable index must own retrieval lifecycle")

    client = _client_for_index(
        SimpleNamespace(search=search, search_async=search_async)
    )
    try:
        prepared = await client.prepare_answer("资料问题", "组会")
        assert not prepared.sources
        assert calls == [("资料问题", 5)]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_close_does_not_wait_for_running_search_and_prevents_new_searches():
    index = _BlockingIndex()
    client = _client_for_index(index)
    pending = asyncio.create_task(client.prepare_answer("first", "组会"))
    try:
        await _wait_until(lambda: index.queries)
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(client.aclose(), timeout=0.25)
        with pytest.raises(DeepSeekStreamError, match="closed"):
            await client.prepare_answer("after-close", "组会")
        assert index.queries == ["first"]
    finally:
        index.release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await _wait_until(lambda: index.active == 0)
        await client.aclose()


@pytest.mark.asyncio
async def test_cancelled_workers_failure_releases_next_search_and_is_observed():
    release = threading.Event()
    started = threading.Event()
    queries = []

    def search(query, limit=5):
        queries.append(query)
        if query == "first":
            started.set()
            assert release.wait(5), "test failed to release worker"
            raise RuntimeError("late worker failure")
        return ()

    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop_errors = []
    loop.set_exception_handler(lambda _loop, context: loop_errors.append(context))
    client = _client_for_index(SimpleNamespace(search=search))
    first = asyncio.create_task(client.prepare_answer("first", "组会"))
    try:
        await _wait_until(started.is_set)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        next_search = asyncio.create_task(client.prepare_answer("next", "组会"))
        release.set()
        await next_search
        assert queries == ["first", "next"]
        await asyncio.sleep(0)
        assert not loop_errors
    finally:
        release.set()
        await asyncio.gather(first, return_exceptions=True)
        await client.aclose()
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_close_prevents_an_already_waiting_search_from_running():
    index = _BlockingIndex()
    client = _client_for_index(index)
    first = asyncio.create_task(client.prepare_answer("first", "组会"))
    waiting = None
    try:
        await _wait_until(lambda: index.queries)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        waiting = asyncio.create_task(client.prepare_answer("waiting", "组会"))
        await asyncio.sleep(0)
        await client.aclose()
        index.release.set()
        with pytest.raises(DeepSeekStreamError, match="closed"):
            await waiting
        assert index.queries == ["first"]
    finally:
        index.release.set()
        if waiting is not None and not waiting.done():
            waiting.cancel()
        await asyncio.gather(
            first, *([waiting] if waiting is not None else []), return_exceptions=True
        )
        await client.aclose()
