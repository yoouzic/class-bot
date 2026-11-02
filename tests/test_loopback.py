import asyncio
import importlib
import importlib.util
import sys
import threading
import types

import numpy as np
import pytest

from classbot.audio.capture import AudioCaptureError


def _load_loopback():
    assert importlib.util.find_spec("classbot.audio.loopback") is not None, (
        "Windows playback loopback capture is not implemented"
    )
    return importlib.import_module("classbot.audio.loopback")


@pytest.fixture
def loopback():
    return _load_loopback()


@pytest.fixture
def hardware(monkeypatch):
    class Stream:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.active = False
            self.stopped = False
            self.closed = False

        def start_stream(self):
            if backend.start_error:
                raise backend.start_error
            self.active = True

        def stop_stream(self):
            self.stopped = True
            self.active = False

        def close(self):
            self.closed = True

        def is_active(self):
            if backend.health_error:
                raise backend.health_error
            return self.active

        def emit(self, data, status=0):
            data = np.asarray(data, dtype=np.float32)
            return self.kwargs["stream_callback"](data.tobytes(), len(data), {}, status)

    class PyAudio:
        def __init__(self):
            self.terminated = False
            backend.instances.append(self)

        def get_default_wasapi_loopback(self):
            if backend.default_error:
                raise backend.default_error
            return backend.devices[backend.default_index]

        def get_loopback_device_info_generator(self):
            return iter(backend.devices.values())

        def get_device_info_by_index(self, index):
            return backend.devices[index]

        def open(self, **kwargs):
            if backend.open_error:
                raise backend.open_error
            stream = Stream(**kwargs)
            backend.streams.append(stream)
            return stream

        def terminate(self):
            self.terminated = True

    backend = types.SimpleNamespace(
        PyAudio=PyAudio,
        paFloat32=1,
        paContinue=0,
        paComplete=1,
        paAbort=2,
        instances=[],
        streams=[],
        default_index=12,
        default_error=None,
        open_error=None,
        start_error=None,
        health_error=None,
        devices={
            2: {
                "index": 2,
                "name": "Microphone",
                "isLoopbackDevice": False,
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 1,
            },
            12: {
                "index": 12,
                "name": "HUAWEI FreeBuds Pro 4 [Loopback]",
                "isLoopbackDevice": True,
                "defaultSampleRate": 44100.0,
                "maxInputChannels": 2,
            },
            13: {
                "index": 13,
                "name": "Speakers [Loopback]",
                "isLoopbackDevice": True,
                "defaultSampleRate": 48000.0,
                "maxInputChannels": 2,
            },
        },
    )
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", backend)
    return backend


def test_enumeration_reports_only_loopbacks_and_never_opens_audio(hardware):
    loopback = _load_loopback()
    devices = loopback.list_loopback_devices()

    assert devices == (
        loopback.LoopbackDevice(
            12, "HUAWEI FreeBuds Pro 4 [Loopback]", 44100.0, 2, True
        ),
        loopback.LoopbackDevice(13, "Speakers [Loopback]", 48000.0, 2, False),
    )
    assert isinstance(devices, tuple)
    assert hardware.streams == []
    assert hardware.instances[0].terminated


def test_enumeration_works_when_no_default_endpoint_exists(loopback, hardware):
    hardware.default_error = OSError("no default output")

    assert len(loopback.list_loopback_devices()) == 2
    assert not any(device.is_default for device in loopback.list_loopback_devices())
    assert all(instance.terminated for instance in hardware.instances)


def test_enumeration_cleanup_failure_uses_audio_capture_error(
    loopback, hardware, monkeypatch
):
    def terminate(_self):
        raise OSError("cannot release audio manager")

    monkeypatch.setattr(hardware.PyAudio, "terminate", terminate)
    with pytest.raises(AudioCaptureError, match="cannot release audio manager"):
        loopback.list_loopback_devices()


@pytest.mark.asyncio
@pytest.mark.parametrize("device, rate", [(None, 44100), (13, 48000)])
async def test_native_float_capture_resamples_continuously_into_exact_frames(
    loopback, hardware, device, rate
):
    soxr = pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture(device=device)
    capture.start()
    stream = hardware.streams[-1]
    assert stream.kwargs["rate"] == rate
    assert stream.kwargs["channels"] == 2
    assert stream.kwargs["format"] == hardware.paFloat32
    assert stream.kwargs["input"] is True
    assert stream.kwargs["input_device_index"] == (12 if device is None else device)
    assert stream.kwargs["start"] is False
    assert capture.active_device.sample_rate == rate
    assert stream.active

    time = np.arange(rate // 2, dtype=np.float32) / rate
    left = (0.6 * np.sin(2 * np.pi * 750 * time)).astype(np.float32)
    right = (0.2 * np.sin(2 * np.pi * 1500 * time)).astype(np.float32)
    stereo = np.column_stack((left, right))
    reference = soxr.ResampleStream(rate, 16000, 1, dtype="float32")
    expected_chunks = []
    offset = 0
    for size in [137, 509, 882, 1101, 1440] * 6:
        chunk = stereo[offset : offset + size]
        if not len(chunk):
            break
        offset += len(chunk)
        expected_chunks.append(
            reference.resample_chunk(chunk.mean(axis=1, dtype=np.float32))
        )
        assert stream.emit(chunk) == (None, hardware.paContinue)

    expected = np.concatenate(expected_chunks)
    expected_pcm = np.clip(expected * 32768, -32768, 32767).astype("<i2").tobytes()
    expected_pcm = expected_pcm[: len(expected_pcm) // 640 * 640]
    frames = capture.frames()
    actual = [
        await asyncio.wait_for(frames.__anext__(), 1)
        for _ in range(len(expected_pcm) // 640)
    ]
    assert actual
    assert all(len(frame) == 640 for frame in actual)
    assert b"".join(actual) == expected_pcm
    capture.stop()


@pytest.mark.asyncio
async def test_stereo_is_mixed_to_mono_and_pcm_clips_without_wrapping(
    loopback, hardware
):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    stream = hardware.streams[-1]
    stream.emit(np.full((12000, 2), 2.0, dtype=np.float32))
    frame = await asyncio.wait_for(capture.frames().__anext__(), 1)
    assert np.max(np.frombuffer(frame, "<i2")) == 32767
    capture.stop()

    capture.start()
    stream = hardware.streams[-1]
    stream.emit(np.tile(np.array([[0.7, -0.7]], dtype=np.float32), (12000, 1)))
    frame = await asyncio.wait_for(capture.frames().__anext__(), 1)
    assert frame == bytes(640)
    capture.stop()


@pytest.mark.asyncio
async def test_conversion_runs_on_the_audio_callback_thread(
    loopback, hardware, monkeypatch
):
    soxr = pytest.importorskip("soxr")
    original = soxr.ResampleStream
    processing_threads = []

    class RecordingResampler:
        def __init__(self, *args, **kwargs):
            self.inner = original(*args, **kwargs)

        def resample_chunk(self, data, **kwargs):
            processing_threads.append(threading.get_ident())
            return self.inner.resample_chunk(data, **kwargs)

    monkeypatch.setattr(soxr, "ResampleStream", RecordingResampler)
    capture = loopback.LoopbackCapture()
    capture.start()
    main_thread = threading.get_ident()
    await asyncio.to_thread(
        hardware.streams[-1].emit, np.zeros((12000, 2), dtype=np.float32)
    )
    assert len(await asyncio.wait_for(capture.frames().__anext__(), 1)) == 640
    assert processing_threads and all(t != main_thread for t in processing_threads)
    capture.stop()


@pytest.mark.asyncio
async def test_explicit_microphone_is_rejected_without_fallback(loopback, hardware):
    pytest.importorskip("soxr")
    with pytest.raises(AudioCaptureError, match="loopback"):
        loopback.LoopbackCapture(device=2).start()
    assert hardware.streams == []
    assert hardware.instances[-1].terminated


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field, value", [("defaultSampleRate", 0), ("maxInputChannels", 0)]
)
async def test_invalid_playback_format_is_rejected_before_open(
    loopback, hardware, field, value
):
    pytest.importorskip("soxr")
    hardware.devices[12][field] = value
    with pytest.raises(AudioCaptureError, match="invalid audio settings"):
        loopback.LoopbackCapture().start()
    assert hardware.streams == []
    assert hardware.instances[-1].terminated


@pytest.mark.asyncio
async def test_missing_default_never_falls_back_to_microphone(loopback, hardware):
    pytest.importorskip("soxr")
    hardware.default_error = OSError("no default playback endpoint")
    with pytest.raises(AudioCaptureError, match="no default playback endpoint"):
        loopback.LoopbackCapture().start()
    assert hardware.streams == []
    assert hardware.instances[-1].terminated


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["open", "start"])
async def test_open_failure_is_readable_and_releases_portaudio(
    loopback, hardware, stage
):
    pytest.importorskip("soxr")
    setattr(hardware, f"{stage}_error", OSError("device disconnected"))
    capture = loopback.LoopbackCapture()
    with pytest.raises(AudioCaptureError, match="device disconnected"):
        capture.start()
    assert capture.active_device is None
    assert hardware.instances[-1].terminated
    assert all(stream.closed for stream in hardware.streams)


@pytest.mark.asyncio
async def test_callback_status_is_visible_to_frame_consumer(loopback, hardware):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    result = hardware.streams[-1].emit(np.zeros((882, 2), dtype=np.float32), status=2)
    assert result == (None, hardware.paAbort)
    with pytest.raises(AudioCaptureError, match="status.*2"):
        await asyncio.wait_for(capture.frames().__anext__(), 1)
    capture.stop()


@pytest.mark.asyncio
@pytest.mark.parametrize("throws", [False, True])
async def test_stream_failure_without_callback_wakes_consumer(
    loopback, hardware, monkeypatch, throws
):
    pytest.importorskip("soxr")
    monkeypatch.setattr(loopback, "_HEALTH_INTERVAL", 0.005)
    capture = loopback.LoopbackCapture()
    capture.start()
    if throws:
        hardware.health_error = OSError("output endpoint was removed")
    else:
        hardware.streams[-1].active = False
    with pytest.raises(AudioCaptureError, match="(?i)playback"):
        await asyncio.wait_for(capture.frames().__anext__(), 1)
    capture.stop()


@pytest.mark.asyncio
async def test_overload_is_bounded_and_reported_instead_of_silent_audio_loss(
    loopback, hardware, monkeypatch
):
    pytest.importorskip("soxr")
    scheduled = []
    event_loop = asyncio.get_running_loop()
    original = event_loop.call_soon_threadsafe

    def counted(callback, *args, **kwargs):
        scheduled.append(callback)
        return original(callback, *args, **kwargs)

    monkeypatch.setattr(event_loop, "call_soon_threadsafe", counted)
    capture = loopback.LoopbackCapture()
    capture.start()
    stream = hardware.streams[-1]
    result = None
    for _ in range(80):
        result = stream.emit(np.zeros((882, 2), dtype=np.float32))
        if result[1] == hardware.paAbort:
            break
    assert result == (None, hardware.paAbort)
    assert len(scheduled) <= 1
    with pytest.raises(AudioCaptureError, match="overflow|overload|too slow"):
        await asyncio.wait_for(capture.frames().__anext__(), 1)
    capture.stop()


@pytest.mark.asyncio
async def test_stop_wakes_all_waiters_and_closes_all_handles(loopback, hardware):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    waiters = [asyncio.create_task(capture.frames().__anext__()) for _ in range(2)]
    await asyncio.sleep(0)
    capture.stop()
    for waiter in waiters:
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(waiter, 1)
    assert hardware.streams[-1].stopped
    assert hardware.streams[-1].closed
    assert hardware.instances[-1].terminated
    assert capture.active_device is None
    capture.stop()


@pytest.mark.asyncio
async def test_stop_failure_still_closes_stream_terminates_and_wakes_frames(
    loopback, hardware, monkeypatch
):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    frames = capture.frames()
    stream = hardware.streams[-1]

    def stop_stream():
        raise OSError("stream stop failed")

    monkeypatch.setattr(stream, "stop_stream", stop_stream)
    with pytest.raises(AudioCaptureError, match="stream stop failed"):
        capture.stop()
    assert stream.closed
    assert hardware.instances[-1].terminated
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(frames.__anext__(), 1)


@pytest.mark.asyncio
async def test_start_is_idempotent_for_an_active_session(loopback, hardware):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    capture.start()
    assert len(hardware.instances) == 1
    assert len(hardware.streams) == 1
    capture.stop()


@pytest.mark.asyncio
async def test_restart_isolates_old_callbacks_and_frame_iterators(loopback, hardware):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    old_frames = capture.frames()
    old_stream = hardware.streams[-1]
    capture.stop()
    capture.start()
    assert old_stream.emit(np.zeros((12000, 2), dtype=np.float32)) == (
        None,
        hardware.paComplete,
    )
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(old_frames.__anext__(), 1)
    hardware.streams[-1].emit(np.zeros((12000, 2), dtype=np.float32))
    assert len(await asyncio.wait_for(capture.frames().__anext__(), 1)) == 640
    capture.stop()


@pytest.mark.asyncio
async def test_missing_optional_backend_has_actionable_error(
    loopback, hardware, monkeypatch
):
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", None)
    capture = loopback.LoopbackCapture()
    with pytest.raises(AudioCaptureError, match="PyAudioWPatch"):
        capture.start()
    with pytest.raises(AudioCaptureError, match="PyAudioWPatch"):
        loopback.list_loopback_devices()
    assert hardware.instances == []


@pytest.mark.asyncio
async def test_missing_resampler_does_not_prevent_device_enumeration(
    loopback, hardware, monkeypatch
):
    monkeypatch.setitem(sys.modules, "soxr", None)
    assert len(loopback.list_loopback_devices()) == 2
    with pytest.raises(AudioCaptureError, match="soxr"):
        loopback.LoopbackCapture().start()


@pytest.mark.asyncio
async def test_malformed_callback_data_is_reported(loopback, hardware):
    pytest.importorskip("soxr")
    capture = loopback.LoopbackCapture()
    capture.start()
    callback = hardware.streams[-1].kwargs["stream_callback"]
    assert callback(b"broken", 882, {}, 0) == (None, hardware.paAbort)
    with pytest.raises(AudioCaptureError, match="(?i)playback"):
        await asyncio.wait_for(capture.frames().__anext__(), 1)
    capture.stop()
