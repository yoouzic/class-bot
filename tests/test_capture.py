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
    capture.start()
    frames = capture.frames()
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
async def test_frames_bind_to_session_queue_across_restart(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawInputStream=FakeStream),
    )
    capture = AudioCapture()

    capture.start()
    old_frames = capture.frames()
    old_callback = FakeStream.instance.kwargs["callback"]
    capture.stop()

    capture.start()
    new_frames = capture.frames()
    new_callback = FakeStream.instance.kwargs["callback"]
    old_callback(b"old-session", 320, None, None)
    new_callback(b"new-session", 320, None, None)

    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(old_frames.__anext__(), timeout=1)
    assert await asyncio.wait_for(new_frames.__anext__(), timeout=1) == b"new-session"
    capture.stop()


@pytest.mark.asyncio
async def test_capture_queue_drops_oldest_frame_when_full(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "sounddevice",
        types.SimpleNamespace(RawInputStream=FakeStream),
    )
    capture = AudioCapture()
    capture.start()
    callback = FakeStream.instance.kwargs["callback"]

    for value in range(51):
        callback(bytes([value]), 320, None, None)

    assert await asyncio.wait_for(capture.frames().__anext__(), timeout=1) == b"\x01"
    capture.stop()


def test_audio_capture_rejects_non_16khz_sample_rate():
    with pytest.raises(ValueError, match="16000"):
        AudioCapture(sample_rate=44100)
