"""Windows WASAPI playback capture using PyAudioWPatch loopback endpoints.

The endpoint is opened at its native rate/channel count. PortAudio's audio
callback downmixes and continuously resamples it to the segmenter's 16 kHz,
mono, PCM16 little-endian format. Imports stay lazy for microphone-only users.
"""

from __future__ import annotations

import asyncio
import math
import threading
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from classbot.audio.capture import AudioCaptureError

_QUEUE_SIZE = 50
_FRAME_BYTES = 640
_SAMPLE_RATE = 16_000
_HEALTH_INTERVAL = 0.25


@dataclass(frozen=True, slots=True)
class LoopbackDevice:
    index: int
    name: str
    sample_rate: float
    channels: int
    is_default: bool = False


def _load_portaudio() -> Any:
    try:
        import pyaudiowpatch
    except ImportError as error:
        raise AudioCaptureError(
            "Windows playback capture requires PyAudioWPatch; "
            "install the PyAudioWPatch package."
        ) from error
    return pyaudiowpatch


def _device(info: Any, default_index: int | None = None) -> LoopbackDevice:
    if not info.get("isLoopbackDevice", False):
        raise AudioCaptureError(
            "The selected device is not a playback loopback endpoint"
        )
    index = int(info["index"])
    rate = float(info["defaultSampleRate"])
    channels = int(info["maxInputChannels"])
    if not math.isfinite(rate) or rate <= 0 or channels < 1:
        raise AudioCaptureError(
            "The playback loopback endpoint has invalid audio settings"
        )
    return LoopbackDevice(
        index, str(info["name"]), rate, channels, index == default_index
    )


def list_loopback_devices() -> tuple[LoopbackDevice, ...]:
    """List playback endpoints without opening an input or recording audio."""
    module = _load_portaudio()
    portaudio = None
    try:
        portaudio = module.PyAudio()
        try:
            default_index = int(portaudio.get_default_wasapi_loopback()["index"])
        except (OSError, ValueError, KeyError):
            default_index = None
        return tuple(
            _device(info, default_index)
            for info in portaudio.get_loopback_device_info_generator()
            if info.get("isLoopbackDevice", False)
        )
    except Exception as error:
        if isinstance(error, AudioCaptureError):
            raise
        raise AudioCaptureError(
            f"Could not enumerate playback loopback devices: {error}"
        ) from error
    finally:
        if portaudio is not None:
            try:
                portaudio.terminate()
            except Exception as error:
                raise AudioCaptureError(
                    f"Could not release playback device enumeration: {error}"
                ) from error


@dataclass
class _Session:
    loop: asyncio.AbstractEventLoop
    token: object = field(default_factory=object)
    wake: asyncio.Event = field(default_factory=asyncio.Event)
    lock: Any = field(default_factory=threading.Lock)
    pending: deque[bytes] = field(default_factory=deque)
    error: AudioCaptureError | None = None
    stopped: bool = False
    notification_pending: bool = False

    def _notify_locked(self) -> None:
        # Coalesce wakeups before crossing threads: asyncio's own callback queue
        # must not grow without bound when the UI/event loop is busy.
        if self.notification_pending:
            return
        self.notification_pending = True
        try:
            self.loop.call_soon_threadsafe(self._wake)
        except RuntimeError:  # The application event loop has already closed.
            self.stopped = True
            self.notification_pending = False
            self.pending.clear()

    def _wake(self) -> None:
        with self.lock:
            self.notification_pending = False
            self.wake.set()

    def offer(self, frame: bytes) -> bool:
        with self.lock:
            if self.stopped or self.error is not None:
                return False
            if len(self.pending) >= _QUEUE_SIZE:
                self.error = AudioCaptureError(
                    "Playback capture overflow: the audio consumer is too slow; "
                    "stop and restart playback capture."
                )
                self.pending.clear()
                self._notify_locked()
                return False
            self.pending.append(frame)
            self._notify_locked()
            return True

    def fail(self, error: AudioCaptureError) -> None:
        with self.lock:
            if self.stopped or self.error is not None:
                return
            self.error = error
            self.pending.clear()
            self._notify_locked()

    def stop(self) -> None:
        with self.lock:
            self.stopped = True
            self.pending.clear()
            self._notify_locked()


class LoopbackCapture:
    """Capture all audio played through one Windows playback endpoint.

    A missing/invalid loopback device is an error. Microphone devices are never
    considered as a fallback. ``active_device`` describes the resolved endpoint
    for UI status; a changed Windows default applies on the next ``start``.
    """

    def __init__(self, device: int | None = None):
        self.device = device
        self.sample_rate = _SAMPLE_RATE
        self.blocksize = _SAMPLE_RATE // 50
        self.active_device: LoopbackDevice | None = None
        self._active_device: int | None = None
        self._stream: Any | None = None
        self._portaudio: Any | None = None
        self._session: _Session | None = None
        self._session_token: object | None = None
        self._health_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Open native-rate float audio on the current asyncio event loop."""
        if self._stream is not None:
            return
        session = _Session(asyncio.get_running_loop())
        self._session = session
        self._session_token = session.token
        stream = None
        portaudio = None
        try:
            module = _load_portaudio()
            try:
                import numpy as np
                import soxr
            except ImportError as error:
                raise AudioCaptureError(
                    "Playback capture requires numpy and soxr; install the soxr package."
                ) from error

            portaudio = module.PyAudio()
            if self.device is None:
                info = portaudio.get_default_wasapi_loopback()
                device = _device(info, int(info["index"]))
            else:
                device = _device(portaudio.get_device_info_by_index(self.device))
            resampler = soxr.ResampleStream(
                device.sample_rate, _SAMPLE_RATE, 1, dtype="float32"
            )
            callback = self._make_callback(session, device, module, np, resampler)
            stream = portaudio.open(
                format=module.paFloat32,
                channels=device.channels,
                rate=round(device.sample_rate),
                frames_per_buffer=max(1, round(device.sample_rate / 50)),
                input=True,
                input_device_index=device.index,
                stream_callback=callback,
                start=False,
            )
            stream.start_stream()
            self._stream = stream
            self._portaudio = portaudio
            self.active_device = device
            self._active_device = device.index
            self._health_task = session.loop.create_task(
                self._watch_stream(session, stream)
            )
        except Exception as error:
            self._session_token = None
            session.stop()
            self._release(stream, portaudio)
            if isinstance(error, AudioCaptureError):
                raise
            raise AudioCaptureError(
                f"Could not open playback loopback capture: {error}"
            ) from error

    def _make_callback(
        self,
        session: _Session,
        device: LoopbackDevice,
        module: Any,
        np: Any,
        resampler: Any,
    ) -> Any:
        residual = bytearray()

        def callback(
            in_data: bytes, frame_count: int, _time_info: Any, status: int
        ) -> tuple[None, int]:
            if session.token is not self._session_token or session.stopped:
                return None, module.paComplete
            if session.error is not None:
                return None, module.paAbort
            if status:
                session.fail(
                    AudioCaptureError(f"Playback callback status error: {status}")
                )
                return None, module.paAbort
            try:
                if len(in_data) != frame_count * device.channels * 4:
                    raise ValueError("incomplete native float32 audio block")
                samples = np.frombuffer(in_data, dtype="<f4").reshape(
                    -1, device.channels
                )
                mono = samples.mean(axis=1, dtype=np.float32)
                converted = resampler.resample_chunk(mono)
                converted = np.nan_to_num(converted, nan=0.0, posinf=1.0, neginf=-1.0)
                pcm = np.clip(converted * 32768, -32768, 32767).astype("<i2")
                residual.extend(pcm.tobytes())
                while len(residual) >= _FRAME_BYTES:
                    frame = bytes(residual[:_FRAME_BYTES])
                    del residual[:_FRAME_BYTES]
                    if not session.offer(frame):
                        residual.clear()
                        return None, module.paAbort
                return None, module.paContinue
            except Exception as error:  # noqa: BLE001 - cannot escape PortAudio callback
                session.fail(
                    AudioCaptureError(f"Playback audio processing failed: {error}")
                )
                return None, module.paAbort

        return callback

    async def _watch_stream(self, session: _Session, stream: Any) -> None:
        """Surface a disconnected endpoint even when PortAudio sends no callback."""
        try:
            while session.token is self._session_token:
                await asyncio.sleep(_HEALTH_INTERVAL)
                if session.stopped or session.error is not None:
                    return
                if not stream.is_active():
                    raise AudioCaptureError(
                        "Playback loopback stream stopped unexpectedly"
                    )
        except asyncio.CancelledError:
            return
        except Exception as error:  # noqa: BLE001 - surface background device failures
            session.fail(AudioCaptureError(f"Playback loopback stream failed: {error}"))

    def frames(self) -> AsyncIterator[bytes]:
        # Bind immediately so a delayed consumer cannot cross capture sessions.
        return self._frames_from(self._session)

    async def _frames_from(self, session: _Session | None) -> AsyncIterator[bytes]:
        if session is None:
            raise AudioCaptureError("Playback loopback capture has not been started")
        while True:
            with session.lock:
                if session.error is not None:
                    raise session.error
                if session.stopped:
                    return
                frame = session.pending.popleft() if session.pending else None
                if frame is None:
                    session.wake.clear()
            if frame is not None:
                yield frame
            else:
                await session.wake.wait()

    @staticmethod
    def _release(stream: Any | None, portaudio: Any | None) -> list[Exception]:
        errors = []
        if stream is not None:
            try:
                stream.stop_stream()
            except Exception as error:  # noqa: BLE001 - still close and terminate
                errors.append(error)
            try:
                stream.close()
            except Exception as error:  # noqa: BLE001 - still terminate PortAudio
                errors.append(error)
        if portaudio is not None:
            try:
                portaudio.terminate()
            except Exception as error:  # noqa: BLE001 - report after all cleanup
                errors.append(error)
        return errors

    def stop(self) -> None:
        """Wake consumers and release the WASAPI stream and PortAudio manager."""
        self._session_token = None
        session = self._session
        if session is not None:
            session.stop()
        task, self._health_task = self._health_task, None
        if task is not None:
            task.cancel()
        stream, self._stream = self._stream, None
        portaudio, self._portaudio = self._portaudio, None
        self.active_device = None
        self._active_device = None
        errors = self._release(stream, portaudio)
        if errors:
            raise AudioCaptureError(
                f"Could not close playback loopback capture: {errors[0]}"
            ) from errors[0]
