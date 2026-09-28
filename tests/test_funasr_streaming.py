import asyncio
import threading

import numpy as np
import pytest

from classbot.asr import funasr_streaming
from classbot.asr.funasr_streaming import FunAsrStreamingRecognizer


class FakeModel:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = self.outputs.pop(0) if self.outputs else ""
        return [{"text": text}]


class PhaseModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        text = "final" if kwargs["is_final"] else ""
        return [{"text": text}]


class BlockingInferenceModel:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.finished = threading.Event()

    def generate(self, **kwargs):
        self.started.set()
        self.release.wait(timeout=3)
        self.finished.set()
        return [{"text": "完成"}]


class RaisingModel:
    def generate(self, **kwargs):
        raise RuntimeError("fake inference failure")


def _pcm(samples: int, value: int = 1000) -> bytes:
    return (np.full(samples, value, dtype="<i2")).tobytes()


async def _next_event(events):
    return await asyncio.wait_for(events.__anext__(), timeout=1)


async def _wait_until(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0)

    await asyncio.wait_for(poll(), timeout=1)


@pytest.mark.asyncio
async def test_partial_and_final_events_keep_segment_id_and_use_streaming_parameters(
    monkeypatch,
):
    model = FakeModel(["你好", "你好啊"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)

    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert partial.text == "你好"
    assert partial.is_final is False
    assert final.text == "你好啊"
    assert final.is_final is True
    assert partial.segment_id == final.segment_id == 0
    assert model.calls[0]["is_final"] is False
    assert model.calls[1]["is_final"] is True
    assert model.calls[0]["chunk_size"] == [0, 10, 5]
    assert model.calls[0]["encoder_chunk_look_back"] == 4
    assert model.calls[0]["decoder_chunk_look_back"] == 1
    assert model.calls[0]["cache"] is model.calls[1]["cache"]
    assert model.calls[0]["input"].dtype == np.float32
    assert np.allclose(model.calls[0]["input"], 1000 / 32768)

    await recognizer.stop()


@pytest.mark.asyncio
async def test_finish_increments_segment_id_and_new_segment_emits_new_event(monkeypatch):
    model = FakeModel(["第一段", "第一段", "第二段"])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    first = await _next_event(events)
    recognizer.finish_segment()
    first_final = await _next_event(events)

    recognizer.push_audio(_pcm(160, value=-2000))
    second = await _next_event(events)

    assert first.segment_id == first_final.segment_id == 0
    assert first_final.is_final is True
    assert model.calls[1]["is_final"] is True
    assert model.calls[1]["input"].size == 0
    assert second.text == "第二段"
    assert second.segment_id == 1
    assert second.segment_id != first.segment_id

    await recognizer.stop()


@pytest.mark.asyncio
async def test_empty_final_inference_replays_last_partial_text(monkeypatch):
    model = FakeModel(["保留文本", ""])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    partial = await _next_event(events)
    recognizer.push_audio(_pcm(80))
    recognizer.finish_segment()
    final = await _next_event(events)

    assert partial.text == final.text == "保留文本"
    assert partial.is_final is False
    assert final.is_final is True
    assert final.segment_id == 0

    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_reliably_ends_events(monkeypatch):
    model = FakeModel([])
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    await recognizer.stop()

    with pytest.raises(StopAsyncIteration):
        await _next_event(events)


@pytest.mark.asyncio
async def test_stop_during_model_load_invalidates_old_start_and_allows_restart(
    monkeypatch,
):
    load_started = threading.Event()
    release_load = threading.Event()
    models = []

    def build(device):
        load_started.set()
        release_load.wait(timeout=3)
        model = FakeModel(["重新开始"])
        models.append(model)
        return model

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    start_task = asyncio.create_task(recognizer.start())
    await asyncio.wait_for(asyncio.to_thread(load_started.wait, 1), timeout=1)

    stop_task = asyncio.create_task(recognizer.stop())
    await asyncio.sleep(0)
    release_load.set()
    await asyncio.wait_for(stop_task, timeout=2)
    await asyncio.wait_for(start_task, timeout=2)

    assert recognizer._worker is None
    assert recognizer._stopped is True

    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    event = await _next_event(events)
    assert event.text == "重新开始"
    assert event.segment_id == 0
    await recognizer.stop()
    assert len(models) == 2


@pytest.mark.asyncio
async def test_cancelled_start_then_immediate_restart_does_not_return_empty_worker(
    monkeypatch,
):
    load_started = threading.Event()
    release_load = threading.Event()
    calls = 0

    def build(device):
        nonlocal calls
        calls += 1
        if calls == 1:
            load_started.set()
            release_load.wait(timeout=3)
        return FakeModel(["新会话"])

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    old_start = asyncio.create_task(recognizer.start())
    await asyncio.wait_for(asyncio.to_thread(load_started.wait, 1), timeout=1)

    old_start.cancel()
    with pytest.raises(asyncio.CancelledError):
        await old_start

    new_start = asyncio.create_task(recognizer.start())
    await asyncio.sleep(0)
    release_load.set()
    await asyncio.wait_for(new_start, timeout=2)

    assert calls == 2
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    assert (await _next_event(events)).text == "新会话"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_cleans_session_state_and_is_idempotent(monkeypatch):
    models = []

    def build(device):
        model = FakeModel([])
        models.append(model)
        return model

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)

    await recognizer.start()
    recognizer.push_audio(_pcm(80))
    await _wait_until(lambda: recognizer._residual.size == 80)
    old_cache = recognizer._cache
    recognizer._last_text = "旧文本"
    recognizer._worker_error = RuntimeError("old error")
    events = recognizer.events()
    await recognizer.stop()
    with pytest.raises(RuntimeError, match="old error"):
        await _next_event(events)
    second_events = recognizer.events()
    await recognizer.stop()
    with pytest.raises(StopAsyncIteration):
        await _next_event(second_events)

    assert recognizer._residual.size == 0
    assert recognizer._cache == {}
    assert recognizer._cache is not old_cache
    assert recognizer._last_text == ""
    assert recognizer._segment_samples == 0
    assert recognizer._worker is None
    assert recognizer._worker_error is None
    assert recognizer._input_queue.empty()
    assert recognizer._events_queue.empty()

    await recognizer.start()
    assert recognizer._segment_id == 0
    await recognizer.stop()
    assert len(models) == 2


@pytest.mark.asyncio
async def test_worker_failure_stops_accepting_audio_and_can_restart(monkeypatch):
    models = [RaisingModel(), FakeModel(["恢复"])]

    def build(device):
        return models.pop(0)

    monkeypatch.setattr(funasr_streaming, "build_model", build)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))

    with pytest.raises(RuntimeError, match="fake inference failure"):
        await _next_event(events)

    assert recognizer._stopped is True
    assert recognizer._worker is None
    with pytest.raises(RuntimeError, match="not running"):
        recognizer.push_audio(_pcm(160))

    await recognizer.start()
    events = recognizer.events()
    recognizer.push_audio(_pcm(160))
    assert (await _next_event(events)).text == "恢复"
    await recognizer.stop()


@pytest.mark.asyncio
async def test_stop_waits_for_inference_thread_before_session_cleanup(monkeypatch):
    model = BlockingInferenceModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)

    await recognizer.start()
    recognizer.push_audio(_pcm(160))
    await asyncio.wait_for(asyncio.to_thread(model.started.wait, 1), timeout=1)

    stop_task = asyncio.create_task(recognizer.stop())
    await asyncio.sleep(0.05)
    stop_finished_early = stop_task.done()

    model.release.set()
    await asyncio.wait_for(stop_task, timeout=2)
    assert stop_finished_early is False
    assert model.finished.is_set()
    assert recognizer._inference_tasks == set()


@pytest.mark.asyncio
async def test_finish_marker_survives_full_queue_and_drops_later_audio(monkeypatch):
    model = PhaseModel()
    monkeypatch.setattr(funasr_streaming, "build_model", lambda device: model)
    recognizer = FunAsrStreamingRecognizer(chunk_samples=160)
    events = recognizer.events()

    await recognizer.start()
    for _ in range(32):
        recognizer.push_audio(_pcm(160))
    recognizer.finish_segment()
    for _ in range(32):
        recognizer.push_audio(_pcm(160, value=-1000))

    final = await _next_event(events)
    await _wait_until(lambda: recognizer._segment_id == 1)

    assert final.text == "final"
    assert final.is_final is True
    assert final.segment_id == 0
    assert len([call for call in model.calls if call["is_final"]]) == 1
    assert recognizer._closing is False
    await recognizer.stop()
