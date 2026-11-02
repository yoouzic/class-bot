import asyncio
from types import SimpleNamespace

import pytest

from classbot.domain import CardStatus, QuestionCandidate
from classbot.pipeline import LivePipeline


@pytest.mark.asyncio
async def test_cancel_in_delta_callback_closes_answer_stream_in_its_own_task():
    entered = asyncio.Event()
    closed = []
    owner = []

    async def answer():
        owner.append(asyncio.current_task())
        try:
            yield "已收到的正文"
            await asyncio.Event().wait()
        finally:
            closed.append(asyncio.current_task())

    stream = answer()  # Keep a strong reference: GC must not own cleanup.

    async def on_delta(*_args):
        entered.set()
        await asyncio.Event().wait()

    pipeline = LivePipeline(None, None, SimpleNamespace(stream_answer=lambda *_: stream),
                            lambda *_: None, on_delta)
    try:
        await pipeline._handle_candidate(QuestionCandidate("q1", "为什么", "测试", 1, True))
        await asyncio.wait_for(entered.wait(), 1)
        task = pipeline._states["q1"].task
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert closed == owner == [task]
        assert pipeline._states["q1"].card.status is CardStatus.CANCELLED
        assert pipeline._states["q1"].card.answer == "已收到的正文"
    finally:
        await stream.aclose()


@pytest.mark.asyncio
async def test_revision_return_closes_owned_stream_before_answer_task_finishes():
    closed = []

    async def answer():
        try:
            yield "旧正文"
            yield "过期正文"
        finally:
            closed.append(asyncio.current_task())

    stream = answer()

    def on_delta(*_args):
        pipeline._states["q1"].revision += 1

    pipeline = LivePipeline(None, None, SimpleNamespace(stream_answer=lambda *_: stream),
                            lambda *_: None, on_delta)
    try:
        await pipeline._handle_candidate(QuestionCandidate("q1", "为什么", "测试", 1, True))
        task = pipeline._states["q1"].task
        await task

        assert closed == [task]
        assert pipeline._states["q1"].card.answer == "旧正文"
        assert pipeline._states["q1"].card.status is not CardStatus.ANSWERED
    finally:
        await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_stream_cleanup_error_preserves_primary_failure_and_records_diagnostic(cancel):
    entered = asyncio.Event()

    class Stream:
        def __init__(self):
            self.sent = False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.sent:
                self.sent = True
                return "部分正文"
            raise ValueError("原始协议错误")

        async def aclose(self):
            raise RuntimeError("关闭流失败")

    async def on_delta(*_args):
        if cancel:
            entered.set()
            await asyncio.Event().wait()

    pipeline = LivePipeline(None, None, SimpleNamespace(stream_answer=lambda *_: Stream()),
                            lambda *_: None, on_delta)
    await pipeline._handle_candidate(QuestionCandidate("q1", "为什么", "测试", 1, True))
    task = pipeline._states["q1"].task
    if cancel:
        await asyncio.wait_for(entered.wait(), 1)
        task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    card = pipeline._states["q1"].card
    assert card.answer == "部分正文"
    if cancel:
        assert task.cancelled() and card.status is CardStatus.CANCELLED
    else:
        assert card.status is CardStatus.ERROR and card.error == "原始协议错误"
    assert any("关闭流失败" in message for message in pipeline._diagnostics)
