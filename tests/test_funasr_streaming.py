import asyncio

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


def _pcm(samples: int, value: int = 1000) -> bytes:
    return (np.full(samples, value, dtype="<i2")).tobytes()


async def _next_event(events):
    return await asyncio.wait_for(events.__anext__(), timeout=1)


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
    model = FakeModel(["第一段", "第二段"])
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
