from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

_QUEUE_SIZE = 50
_STOP = object()
SAMPLE_RATE = 16_000
_LOGGER = logging.getLogger(__name__)


class AudioCaptureError(RuntimeError):
    """Raised when opening or receiving from the microphone fails."""


class AudioCapture:
    """Capture bounded 20 ms PCM frames from a local sounddevice input."""

    def __init__(self, device: int | None = None, sample_rate: int = SAMPLE_RATE):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("sample_rate must be 16000")

        self.device = device
        self.sample_rate = sample_rate
        self.blocksize = sample_rate // 50
        if self.blocksize <= 0:
            raise ValueError("sample_rate must provide a positive 20 ms block")

        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue(_QUEUE_SIZE)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any | None = None
        self._active_device: int | None = None
        self._session_token: object | None = None
        self._stopped = False

    def start(self) -> None:
        if self._stream is not None:
            return

        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[bytes | object] = asyncio.Queue(_QUEUE_SIZE)
        token = object()
        self._loop = loop
        self._queue = queue
        self._session_token = token
        self._stopped = False

        try:
            import sounddevice as sd

            callback = self._make_callback(token, queue, loop)
            errors: list[tuple[int | None, Exception]] = []
            for device in self._candidate_devices(sd):
                stream: Any | None = None
                try:
                    stream = sd.RawInputStream(
                        samplerate=self.sample_rate,
                        blocksize=self.blocksize,
                        channels=1,
                        dtype="int16",
                        device=device,
                        callback=callback,
                    )
                    stream.start()
                except Exception as error:  # noqa: BLE001
                    errors.append((device, error))
                    if stream is not None:
                        try:
                            stream.close()
                        except Exception as close_error:  # noqa: BLE001
                            _LOGGER.debug(
                                "microphone stream close after open failure failed: %s",
                                close_error,
                            )
                    continue

                self._stream = stream
                self._active_device = device
                return

            if errors:
                device, error = errors[-1]
                raise AudioCaptureError(
                    f"microphone open failed for devices "
                    f"{[candidate for candidate, _ in errors]}: {error}"
                ) from error
            raise AudioCaptureError("microphone open failed: no input device")
        except Exception as error:
            self._stream = None
            self._active_device = None
            self._session_token = None
            if isinstance(error, AudioCaptureError):
                raise
            raise AudioCaptureError(f"microphone open failed: {error}") from error

    def _candidate_devices(self, sounddevice: Any) -> list[int | None]:
        if self.device is not None:
            return [self.device]

        candidates: list[int | None] = [None]
        try:
            devices = sounddevice.query_devices()
        except Exception as error:  # noqa: BLE001
            _LOGGER.debug("querying microphone devices failed: %s", error)
            return candidates

        ranked: list[tuple[int, int]] = []
        for position, info in enumerate(devices):
            try:
                if int(info.get("max_input_channels", 0)) < 1:
                    continue
                index = int(info.get("index", position))
                default_rate = float(info.get("default_samplerate", 0) or 0)
            except (AttributeError, TypeError, ValueError):
                continue

            native_rate_priority = 0 if abs(default_rate - self.sample_rate) < 1 else 1
            ranked.append((native_rate_priority, index))

        for _priority, index in sorted(ranked):
            if index not in candidates:
                candidates.append(index)
        return candidates

    def _make_callback(
        self,
        token: object,
        queue: asyncio.Queue[bytes | object],
        loop: asyncio.AbstractEventLoop,
    ) -> Any:
        def callback(indata: Any, _frames: int, _time: Any, status: Any) -> None:
            if self._stopped or token is not self._session_token:
                return

            if status:
                error = AudioCaptureError(f"audio callback error: {status}")
                loop.call_soon_threadsafe(
                    self._enqueue_error,
                    token,
                    queue,
                    error,
                )
                return

            frame = bytes(indata)
            loop.call_soon_threadsafe(self._enqueue_frame, token, queue, frame)

        return callback

    def _enqueue_frame(
        self,
        token: object,
        queue: asyncio.Queue[bytes | object],
        frame: bytes,
    ) -> None:
        if (
            self._stopped
            or token is not self._session_token
            or queue is not self._queue
        ):
            return

        try:
            queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            queue.put_nowait(frame)

    def _enqueue_error(
        self,
        token: object,
        queue: asyncio.Queue[bytes | AudioCaptureError | object],
        error: AudioCaptureError,
    ) -> None:
        if (
            self._stopped
            or token is not self._session_token
            or queue is not self._queue
        ):
            return

        while True:
            try:
                queue.put_nowait(error)
                return
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

    def _signal_stop(self, queue: asyncio.Queue[bytes | object]) -> None:
        while True:
            try:
                queue.put_nowait(_STOP)
                return
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

    def frames(self) -> AsyncIterator[bytes]:
        queue = self._queue
        return self._frames_from(queue)

    async def _frames_from(
        self, queue: asyncio.Queue[bytes | object]
    ) -> AsyncIterator[bytes]:
        while True:
            item = await queue.get()
            if item is _STOP:
                return
            if isinstance(item, AudioCaptureError):
                raise item
            yield item  # type: ignore[misc]

    def stop(self) -> None:
        self._stopped = True
        self._session_token = None
        queue = self._queue
        stream = self._stream
        self._stream = None
        self._active_device = None

        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            self._signal_stop(queue)
