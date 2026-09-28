from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

_QUEUE_SIZE = 50
_STOP = object()


class AudioCapture:
    """Capture bounded 20 ms PCM frames from a local sounddevice input."""

    def __init__(self, device: int | None = None, sample_rate: int = 16000):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")

        self.device = device
        self.sample_rate = sample_rate
        self.blocksize = sample_rate // 50
        if self.blocksize <= 0:
            raise ValueError("sample_rate must provide a positive 20 ms block")

        self._queue: asyncio.Queue[bytes | object] = asyncio.Queue(_QUEUE_SIZE)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: Any | None = None
        self._stopped = False

    def start(self) -> None:
        if self._stream is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(_QUEUE_SIZE)
        self._stopped = False

        import sounddevice as sd

        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=self._callback,
            start=False,
        )
        self._stream = stream
        try:
            stream.start()
        except Exception:  # noqa: BLE001 - device errors must propagate after cleanup
            self._stream = None
            try:
                stream.close()
            finally:
                raise

    def _callback(
        self,
        indata: Any,
        _frames: int,
        _time: Any,
        _status: Any,
    ) -> None:
        if self._stopped or self._loop is None:
            return

        frame = bytes(indata)
        self._loop.call_soon_threadsafe(self._enqueue_frame, frame)

    def _enqueue_frame(self, frame: bytes) -> None:
        if self._stopped:
            return

        try:
            self._queue.put_nowait(frame)
        except asyncio.QueueFull:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            self._queue.put_nowait(frame)

    def _signal_stop(self) -> None:
        while True:
            try:
                self._queue.put_nowait(_STOP)
                return
            except asyncio.QueueFull:
                try:
                    self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            item = await self._queue.get()
            if item is _STOP:
                return
            yield item  # type: ignore[misc]

    def stop(self) -> None:
        self._stopped = True
        stream = self._stream
        self._stream = None

        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            self._signal_stop()
