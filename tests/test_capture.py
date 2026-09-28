import asyncio
import sys
import types

import pytest

from classbot.audio.capture import AudioCapture


class FakeStream:
    instance = None

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.closed = False
        FakeStream.instance = self

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_fake_stream_callback_and_stop_wake_frames(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawInputStream=FakeStream),
    )
    capture = AudioCapture(device=3)
    frames = capture.frames()

    capture.start()
    stream = FakeStream.instance
    assert stream.kwargs["samplerate"] == 16000
    assert stream.kwargs["blocksize"] == 320
    assert stream.kwargs["channels"] == 1
    assert stream.kwargs["dtype"] == "int16"
    assert stream.kwargs["device"] == 3
    assert stream.started is True

    stream.kwargs["callback"](b"frame", 320, None, None)
    assert await asyncio.wait_for(frames.__anext__(), timeout=1) == b"frame"

    capture.stop()
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(frames.__anext__(), timeout=1)
    assert stream.stopped is True
    assert stream.closed is True


@pytest.mark.asyncio
async def test_capture_queue_drops_oldest_frame_when_full(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawInputStream=FakeStream),
    )
    capture = AudioCapture()
    capture.start()

    for value in range(51):
        capture._enqueue_frame(bytes([value]))

    assert await asyncio.wait_for(capture.frames().__anext__(), timeout=1) == b"\x01"
    capture.stop()
