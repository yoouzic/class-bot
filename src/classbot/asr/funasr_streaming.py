from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterable
from typing import Any

import numpy as np

from classbot.domain import TranscriptEvent

_INPUT_QUEUE_SIZE = 32
_EVENTS_END = object()
_FINISH = object()


def build_model(device: str) -> Any:
    from funasr import AutoModel

    return AutoModel(model="paraformer-zh-streaming", device=device)


class FunAsrStreamingRecognizer:
    def __init__(self, device: str = "cpu", chunk_samples: int = 9600):
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")

        self.device = device
        self.chunk_samples = chunk_samples
        self._input_queue: asyncio.Queue[bytes | object] = asyncio.Queue(
            _INPUT_QUEUE_SIZE
        )
        self._events_queue: asyncio.Queue[TranscriptEvent | object] = asyncio.Queue()
        self._model: Any | None = None
        self._worker: asyncio.Task[None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False
        self._events_closed = False
        self._worker_error: Exception | None = None
        self._residual = np.empty(0, dtype=np.float32)
        self._cache: dict[str, Any] = {}
        self._last_text = ""
        self._segment_id = 0
        self._segment_samples = 0

    async def start(self) -> None:
        if self._worker is not None:
            return

        self._loop = asyncio.get_running_loop()
        self._input_queue = asyncio.Queue(_INPUT_QUEUE_SIZE)
        self._events_queue = asyncio.Queue()
        self._events_closed = False
        self._worker_error = None
        self._stopped = False

        try:
            self._model = await asyncio.to_thread(build_model, self.device)
        except BaseException:
            self._stopped = True
            self._close_events()
            raise

        self._worker = asyncio.create_task(self._run_worker())

    def push_audio(self, frame: bytes) -> None:
        self._ensure_running()
        try:
            self._input_queue.put_nowait(bytes(frame))
        except asyncio.QueueFull:
            self._input_queue.get_nowait()
            self._input_queue.put_nowait(bytes(frame))

    def finish_segment(self) -> None:
        self._ensure_running()
        try:
            self._input_queue.put_nowait(_FINISH)
        except asyncio.QueueFull:
            self._input_queue.get_nowait()
            self._input_queue.put_nowait(_FINISH)

    def events(self) -> AsyncIterator[TranscriptEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[TranscriptEvent]:
        while True:
            event = await self._events_queue.get()
            if event is _EVENTS_END:
                if self._worker_error is not None:
                    raise self._worker_error
                return
            yield event  # type: ignore[misc]

    async def stop(self) -> None:
        worker = self._worker
        self._stopped = True

        if worker is not None and not worker.done():
            worker.cancel()
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                if asyncio.current_task() is not None and asyncio.current_task().cancelling():
                    self._close_events()
                    raise
                if not worker.cancelled():
                    raise

        self._worker = None
        self._close_events()

    def _ensure_running(self) -> None:
        if self._worker is None or self._stopped:
            raise RuntimeError("recognizer is not running")

    async def _run_worker(self) -> None:
        try:
            while True:
                item = await self._input_queue.get()
                if item is _FINISH:
                    await self._finish_current_segment()
                    continue
                await self._consume_audio(item)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - worker errors must close events
            self._worker_error = error
            self._close_events()

    async def _consume_audio(self, frame: bytes) -> None:
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return

        if self._residual.size:
            self._residual = np.concatenate((self._residual, samples))
        else:
            self._residual = samples

        while self._residual.size >= self.chunk_samples:
            chunk = self._residual[: self.chunk_samples]
            self._residual = self._residual[self.chunk_samples :]
            self._segment_samples += int(chunk.size)
            await self._infer(chunk, is_final=False)

    async def _finish_current_segment(self) -> None:
        emitted = False
        if self._residual.size:
            residual = self._residual
            self._residual = np.empty(0, dtype=np.float32)
            self._segment_samples += int(residual.size)
            emitted = await self._infer(residual, is_final=True)

        if not emitted and self._last_text:
            self._emit_event(self._last_text, is_final=True)

        self._residual = np.empty(0, dtype=np.float32)
        self._cache = {}
        self._last_text = ""
        self._segment_id += 1
        self._segment_samples = 0

    async def _infer(self, samples: np.ndarray, is_final: bool) -> bool:
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")

        result = await asyncio.to_thread(
            self._model.generate,
            input=samples,
            cache=self._cache,
            is_final=is_final,
            chunk_size=[0, 10, 5],
            encoder_chunk_look_back=4,
            decoder_chunk_look_back=1,
        )

        emitted = False
        for text in _texts_from_result(result):
            if not text.strip():
                continue
            self._emit_event(text, is_final=is_final)
            self._last_text = text
            emitted = True
        return emitted

    def _emit_event(self, text: str, is_final: bool) -> None:
        elapsed_ms = int(self._segment_samples * 1000 / 16000)
        self._events_queue.put_nowait(
            TranscriptEvent(
                text=text,
                is_final=is_final,
                start_ms=0,
                end_ms=elapsed_ms,
                confidence=0.0,
                segment_id=self._segment_id,
            )
        )

    def _close_events(self) -> None:
        if not self._events_closed:
            self._events_closed = True
            self._events_queue.put_nowait(_EVENTS_END)


def _texts_from_result(result: Any) -> Iterable[str]:
    if result is None:
        return ()
    if isinstance(result, str):
        return (result,)
    if isinstance(result, dict):
        text = result.get("text", "")
        return (str(text),) if text is not None else ()
    if isinstance(result, (list, tuple)):
        texts: list[str] = []
        for item in result:
            texts.extend(_texts_from_result(item))
        return texts
    return ()
