"""Real installed HTTP/1.1 generator stack, with an in-memory network backend."""
import asyncio
import gc
import json
import sys
from contextlib import aclosing, contextmanager
from types import SimpleNamespace

import httpcore
import httpx
import pytest
from httpcore._backends.mock import AsyncMockBackend, AsyncMockStream


def _wire():
    events = [b'data: first\n\n', b'data: [DONE]\n\n', b': discarded tail\n\n']
    return [b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Type: text/event-stream\r\n\r\n'] + [
        f'{len(event):x}\r\n'.encode() + event + b'\r\n' for event in events
    ]


class Backend(AsyncMockBackend):
    def __init__(self, *, stall=False):
        super().__init__(_wire() + ([] if stall else [b'0\r\n\r\n']))
        self.stall = stall
        self.streams = []

    async def connect_tcp(self, *args, **kwargs):
        backend = self

        class Stream(AsyncMockStream):
            async def read(self, *args, **kwargs):
                if backend.stall and not self._buffer:
                    await asyncio.Event().wait()
                return await super().read(*args, **kwargs)

        stream = Stream(list(self._buffer))
        self.streams.append(stream)
        return stream


async def _transport(backend):
    transport = httpx.AsyncHTTPTransport()
    await transport._pool.aclose()
    transport._pool = httpcore.AsyncConnectionPool(network_backend=backend)
    return transport


@contextmanager
def _finalizers():
    hooks = sys.get_asyncgen_hooks()
    observed = []

    def finalize(generator):
        observed.append(generator.ag_code.co_qualname)
        hooks.finalizer(generator)

    sys.set_asyncgen_hooks(firstiter=hooks.firstiter, finalizer=finalize)
    try:
        yield observed
    finally:
        sys.set_asyncgen_hooks(*hooks)


async def _settle():
    gc.collect()
    for _ in range(12):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_outer_aclosing_alone_leaves_actual_http11_generator_to_gc():
    backend = Backend()
    with _finalizers() as finalized:
        async with (
            httpx.AsyncClient(transport=await _transport(backend)) as client,
            client.stream("GET", "http://synthetic.invalid") as response,
            aclosing(response.aiter_lines()) as lines,
        ):
            async for _ in lines:
                break
        await _settle()
    assert "HTTP11ConnectionByteStream.__aiter__" in finalized


@pytest.mark.asyncio
@pytest.mark.parametrize("stall", [False, True])
async def test_terminal_reads_to_eof_or_bounded_timeout_without_http11_gc(tmp_path, monkeypatch, stall):
    from classbot.llm import sse

    monkeypatch.setattr(sse, "_TAIL_TIMEOUT_SECONDS", 0.02)
    backend = Backend(stall=stall)
    started = asyncio.get_running_loop().time()
    with _finalizers() as finalized:
        async with (
            httpx.AsyncClient(transport=await _transport(backend)) as client,
            client.stream("GET", "http://synthetic.invalid") as response,
            sse.owned_sse_lines(response) as lines,
        ):
            terminal = False
            async for line in lines:
                if not terminal and "[DONE]" in line:
                    lines.mark_terminal()
                    terminal = True
        await _settle()
    assert terminal
    assert not finalized
    assert backend.streams and all(stream._closed for stream in backend.streams)
    assert asyncio.get_running_loop().time() - started < 1


def _event(*, content=None, finish=None):
    delta = {} if content is None else {"content": content}
    return ("data: " + json.dumps({"choices": [{"delta": delta, "finish_reason": finish}]}) + "\n\n").encode()


class SequenceBackend(AsyncMockBackend):
    def __init__(self, plans):
        super().__init__([])
        self.plans = plans
        self.streams = []
        self.requests = []

    async def connect_tcp(self, *args, **kwargs):
        backend = self
        events, ending = self.plans[len(self.streams)]
        buffer = [b'HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\nContent-Type: text/event-stream\r\n\r\n']
        buffer += [f'{len(event):x}\r\n'.encode() + event + b'\r\n' for event in events]
        if ending == "eof":
            buffer.append(b'0\r\n\r\n')

        class Stream(AsyncMockStream):
            async def write(self, buffer, *args, **kwargs):
                if buffer.startswith(b"{"):
                    backend.requests.append(json.loads(buffer))
                await super().write(buffer, *args, **kwargs)

            async def read(self, *args, **kwargs):
                if not self._buffer:
                    if ending == "transport":
                        raise httpcore.ReadError("synthetic discarded tail failure")
                    if ending == "program":
                        raise RuntimeError("synthetic unknown program error")
                    if ending == "cancel":
                        asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
                        await asyncio.Event().wait()
                return await super().read(*args, **kwargs)

        stream = Stream(buffer)
        self.streams.append(stream)
        return stream


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["done", "fallback", "done_tail_error", "fallback_tail_error", "partial_error", "before_terminal_error", "program_error", "cancel"])
async def test_production_client_real_http11_terminal_semantics(monkeypatch, case):
    from classbot.config import Settings
    from classbot.llm import sse
    from classbot.llm.deepseek_client import (
        DeepSeekClient,
        DeepSeekStreamError,
        PreparedAnswer,
    )

    monkeypatch.setattr(sse, "_TAIL_TIMEOUT_SECONDS", 0.02)
    answer = [_event(content="synthetic answer [S1]", finish="stop"), b"data: [DONE]\n\n"]
    first = [_event(finish="length")] if case.startswith("fallback") else answer
    if case == "partial_error":
        first = [_event(content="partial [S1]", finish="length")]
    if case == "before_terminal_error":
        first = [_event(content="partial [S1]")]
    ending = "transport" if "tail_error" in case or case == "before_terminal_error" else "eof"
    if case in ("program_error", "cancel"):
        ending = "program" if case == "program_error" else "cancel"
    plans = [(first, ending)] + ([(answer, "eof")] if case.startswith("fallback") else [])
    backend = SequenceBackend(plans)
    settings = Settings("synthetic-key", "答辩", "deepseek-flash", "http://synthetic.invalid", "cpu",
                        thinking_effort="high", defense_thinking_effort="high")
    prepared = PreparedAnswer([{"role": "user", "content": "synthetic question"}], ("[S1] source.md · line 1",))
    observed = []
    with _finalizers() as finalized:
        async with httpx.AsyncClient(transport=await _transport(backend), base_url=settings.base_url) as http:
            client = DeepSeekClient(settings, http_client=http,
                                    scenario=SimpleNamespace(name="synthetic", kind="competition"),
                                    knowledge_index=SimpleNamespace())

            async def consume():
                async with aclosing(client.stream_answer("synthetic question", "答辩", prepared=prepared)) as stream:
                    async for part in stream:
                        observed.append(part)

            expected = {"partial_error": DeepSeekStreamError, "before_terminal_error": httpx.ReadError,
                        "program_error": RuntimeError, "cancel": asyncio.CancelledError}.get(case)
            if expected is None:
                await consume()
            else:
                task = asyncio.create_task(consume())
                with pytest.raises(expected):
                    await task
            await client.aclose()
        await _settle()
    assert len(backend.streams) == (2 if case.startswith("fallback") else 1)
    if case.startswith("fallback"):
        assert backend.requests[0]["thinking"] == {"type": "enabled"}
        assert backend.requests[1]["thinking"] == {"type": "disabled"}
        assert backend.requests[1]["max_tokens"] == 1000
        assert backend.requests[0]["messages"] == backend.requests[1]["messages"] == prepared.messages
    expected_answer = "partial [S1]" if case in ("partial_error", "before_terminal_error") else "synthetic answer [S1]"
    assert observed == [expected_answer]
    assert not finalized
    assert all(stream._closed for stream in backend.streams)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["callback", "cancel", "partial", "generator_close"])
async def test_aborted_consumer_preserves_original_and_closes_real_chain(monkeypatch, failure):
    from classbot.llm import sse

    monkeypatch.setattr(sse, "_TAIL_TIMEOUT_SECONDS", 0.02)
    backend = Backend(stall=True)
    expected = ValueError("callback failure") if failure == "callback" else RuntimeError("partial answer length")
    with _finalizers() as finalized:
        async with httpx.AsyncClient(transport=await _transport(backend)) as client:
            async def consume():
                async with (
                    client.stream("GET", "http://synthetic.invalid") as response,
                    sse.owned_sse_lines(response) as lines,
                ):
                    async for line in lines:
                        if "first" not in line:
                            continue
                        if failure == "cancel":
                            asyncio.get_running_loop().call_soon(asyncio.current_task().cancel)
                            await asyncio.sleep(0)
                        if failure == "generator_close":
                            yield line
                        else:
                            raise expected

            async def drive():
                async with aclosing(consume()) as stream:
                    async for _ in stream:
                        break

            if failure == "generator_close":
                await drive()
            else:
                task = asyncio.create_task(drive())
                error_type = asyncio.CancelledError if failure == "cancel" else type(expected)
                with pytest.raises(error_type) as caught:
                    await task
                if failure != "cancel":
                    assert caught.value is expected
        await _settle()
    assert not finalized
    assert backend.streams and all(stream._closed for stream in backend.streams)
