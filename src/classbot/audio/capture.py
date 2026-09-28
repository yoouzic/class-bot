from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

_QUEUE_SIZE = 50
_STOP = object()
SAMPLE_RATE = 16_000


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

        import sounddevice as sd

        callback = self._make_callback(token, queue, loop)
        stream = sd.RawInputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            channels=1,
            dtype="int16",
            device=self.device,
            callback=callback,
            start=False,
        )
        self._stream = stream
        try:
            stream.start()
        except Exception:  # noqa: BLE001 - device errors must propagate after cleanup
            self._stream = None
            self._session_token = None
            try:
                stream.close()
            finally:
                raise

    def _make_callback(
        self,
        token: object,
        queue: asyncio.Queue[bytes | object],
        loop: asyncio.AbstractEventLoop,
    ) -> Any:
        def callback(indata: Any, _frames: int, _time: Any, _status: Any) -> None:
            if self._stopped or token is not self._session_token:
                return

            frame = bytes(indata)
            loop.call_soon_threadsafe(
                self._enqueue_frame, token, queue, frame
            )

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
            yield item  # type: ignore[misc]

    def stop(self) -> None:
        self._stopped = True
        self._session_token = None
        queue = self._queue
        stream = self._stream
        self._stream = None

        try:
            if stream is not None:
                try:
                    stream.stop()
                finally:
                    stream.close()
        finally:
            self._signal_stop(queue)
