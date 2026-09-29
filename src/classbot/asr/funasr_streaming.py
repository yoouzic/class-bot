from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Iterable, Mapping
from typing import Any

import numpy as np

from classbot.domain import TranscriptEvent

_AUDIO_QUEUE_SIZE = 512
_INPUT_QUEUE_SIZE = _AUDIO_QUEUE_SIZE + 1
_FINISH = object()
SAMPLE_RATE = 16_000
_LOGGER = logging.getLogger(__name__)


class _EventsEnd:
    __slots__ = ("error",)

    def __init__(self, error: Exception | None = None):
        self.error = error


def build_model(device: str) -> Any:
    from funasr import AutoModel

    return AutoModel(model="paraformer-zh-streaming", device=device)


class FunAsrStreamingRecognizer:
    def __init__(
        self,
        device: str = "cpu",
        chunk_samples: int = 9600,
        hotwords: Iterable[str] = (),
        corrections: Mapping[str, str] | None = None,
        final_refinement: bool = False,
    ):
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")

        self.device = device
        self.chunk_samples = chunk_samples
        self.hotwords = tuple(
            word.strip() for word in hotwords if str(word).strip()
        )
        self.corrections = {
            str(wrong).strip(): str(right).strip()
            for wrong, right in (corrections or {}).items()
            if str(wrong).strip() and str(right).strip()
        }
        self.final_refinement = final_refinement
        self._input_queue: asyncio.Queue[bytes | object] = asyncio.Queue(
            _INPUT_QUEUE_SIZE
        )
        self._events_queue: asyncio.Queue[TranscriptEvent | _EventsEnd] = (
            asyncio.Queue()
        )
        self._model: Any | None = None
        self._worker: asyncio.Task[None] | None = None
        self._start_task: asyncio.Task[None] | None = None
        self._start_lock = asyncio.Lock()
        self._stop_task: asyncio.Task[None] | None = None
        self._model_load_task: asyncio.Task[Any] | None = None
        self._inference_tasks: set[asyncio.Task[Any]] = set()
        self._generation = 0
        self._stopped = True
        self._events_closed = False
        self._worker_error: Exception | None = None
        self._residual = np.empty(0, dtype=np.float32)
        self._segment_audio: list[np.ndarray] = []
        self._cache: dict[str, Any] = {}
        self._raw_segment_text = ""
        self._segment_text = ""
        self._last_text = ""
        self._segment_id = 0
        self._segment_samples = 0
        self._closing = False

    async def start(self) -> None:
        await self._wait_for_inference_tasks()
        operation: asyncio.Task[None] | None = None
        while operation is None:
            wait_task: asyncio.Task[None] | None = None
            async with self._start_lock:
                if self._stop_task is not None:
                    if self._stop_task.done():
                        self._stop_task = None
                    else:
                        wait_task = self._stop_task
                elif self._worker is not None and not self._stopped:
                    return
                elif self._start_task is not None:
                    if self._start_task.done():
                        self._start_task = None
                    else:
                        wait_task = self._start_task

                if wait_task is None:
                    self._generation += 1
                    generation = self._generation
                    self._begin_session()
                    operation = asyncio.create_task(
                        self._load_and_start(generation)
                    )
                    self._start_task = operation

            if wait_task is not None:
                await asyncio.shield(wait_task)

        try:
            await asyncio.shield(operation)
        except asyncio.CancelledError:
            if self._start_task is operation:
                self._generation += 1
                self._stopped = True
                self._close_events()
            raise
        finally:
            if operation.done() and self._start_task is operation:
                self._start_task = None

    def push_audio(self, frame: bytes) -> None:
        self._ensure_running()

        if self._input_queue.qsize() >= _AUDIO_QUEUE_SIZE:
            self._input_queue.get_nowait()
        self._input_queue.put_nowait(bytes(frame))

    def finish_segment(self) -> None:
        self._ensure_running()
        self._closing = True
        try:
            self._input_queue.put_nowait(_FINISH)
        except asyncio.QueueFull:
            self._input_queue.get_nowait()
            self._input_queue.put_nowait(_FINISH)

    def events(self) -> AsyncIterator[TranscriptEvent]:
        return self._event_stream()

    async def _event_stream(self) -> AsyncIterator[TranscriptEvent]:
        queue = self._events_queue
        while True:
            event = await queue.get()
            if isinstance(event, _EventsEnd):
                if event.error is not None:
                    raise event.error
                return
            yield event

    async def stop(self) -> None:
        if self._stop_task is not None:
            await self._await_cleanup_task(self._stop_task)
            return

        self._generation += 1
        self._stopped = True
        worker = self._worker
        start_task = self._start_task
        cleanup = asyncio.create_task(self._complete_stop(worker, start_task))
        self._stop_task = cleanup
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
            raise
        finally:
            if cleanup.done() and self._stop_task is cleanup:
                self._stop_task = None

    async def _await_cleanup_task(self, task: asyncio.Task[None]) -> None:
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            await asyncio.shield(task)
            raise

    async def _load_and_start(self, generation: int) -> None:
        operation = asyncio.current_task()
        load_task = asyncio.create_task(asyncio.to_thread(build_model, self.device))
        load_task.add_done_callback(self._consume_task_result)
        self._model_load_task = load_task
        try:
            try:
                model = await asyncio.shield(load_task)
            finally:
                if self._model_load_task is load_task:
                    self._model_load_task = None

            if not self._is_current_generation(generation):
                return

            self._model = model
            self._worker = asyncio.create_task(self._run_worker(generation))
        except asyncio.CancelledError:
            if not load_task.done():
                load_task.cancel()
            raise
        except Exception as error:
            if self._is_current_generation(generation):
                self._worker_error = error
                self._stopped = True
                self._close_events(error)
            raise
        finally:
            if self._start_task is operation:
                self._start_task = None

    def _ensure_running(self) -> None:
        if self._worker is None or self._stopped:
            raise RuntimeError("recognizer is not running")

    async def _complete_stop(
        self,
        worker: asyncio.Task[None] | None,
        start_task: asyncio.Task[None] | None,
    ) -> None:
        if worker is not None:
            if not worker.done():
                worker.cancel()
            try:
                await worker
            except asyncio.CancelledError:
                if not worker.cancelled():
                    raise
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                if self._worker_error is None:
                    self._worker_error = error

        if start_task is not None and not start_task.done():
            start_task.cancel()
            try:
                await start_task
            except asyncio.CancelledError:
                if not start_task.cancelled():
                    raise
            except Exception as error:  # noqa: BLE001 - cleanup must continue
                if self._worker_error is None:
                    self._worker_error = error

        await self._wait_for_inference_tasks()

        error = self._worker_error
        self._worker = None
        self._model = None
        self._clear_queue(self._input_queue)
        self._clear_queue(self._events_queue)
        self._reset_transient_state()
        self._events_closed = False
        self._close_events(error)

    async def _run_worker(self, generation: int) -> None:
        current_task = asyncio.current_task()
        try:
            while True:
                item = await self._input_queue.get()
                if item is _FINISH:
                    await self._finish_current_segment()
                    self._closing = False
                    continue
                await self._consume_audio(item)  # type: ignore[arg-type]
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - expose original worker error
            if self._is_current_generation(generation):
                self._worker_error = error
                self._stopped = True
                self._worker = None
                self._close_events(error)
        finally:
            if self._worker is current_task:
                self._worker = None

    async def _consume_audio(self, frame: bytes) -> None:
        samples = np.frombuffer(frame, dtype="<i2").astype(np.float32) / 32768.0
        if samples.size == 0:
            return

        self._segment_audio.append(samples)

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
        if self._residual.size:
            residual = self._residual
            self._residual = np.empty(0, dtype=np.float32)
            self._segment_samples += int(residual.size)
            await self._infer(residual, is_final=True, emit=False)

        if self.final_refinement:
            refined_text = await self._refine_segment()
            if refined_text:
                self._raw_segment_text = refined_text
                self._segment_text = refined_text
                self._last_text = refined_text
        if self._last_text:
            self._emit_event(self._last_text, is_final=True)

        self._residual = np.empty(0, dtype=np.float32)
        self._segment_audio = []
        self._cache = {}
        self._raw_segment_text = ""
        self._segment_text = ""
        self._last_text = ""
        self._segment_id += 1
        self._segment_samples = 0

    async def _infer(
        self,
        samples: np.ndarray,
        is_final: bool,
        *,
        emit: bool = True,
    ) -> bool:
        if self._model is None:
            raise RuntimeError("ASR model is not loaded")

        inference_task = asyncio.create_task(
            asyncio.to_thread(
                self._model.generate,
                input=samples,
                cache=self._cache,
                is_final=is_final,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
                **self._postprocess_kwargs(final_refinement=False),
            )
        )
        self._inference_tasks.add(inference_task)
        inference_task.add_done_callback(self._inference_done)
        try:
            result = await asyncio.shield(inference_task)
        finally:
            if inference_task.done():
                self._inference_tasks.discard(inference_task)

        emitted = False
        for raw_text in _texts_from_result(result):
            if not raw_text.strip():
                continue
            if self._raw_segment_text and raw_text.startswith(
                self._raw_segment_text
            ):
                self._raw_segment_text = raw_text
            else:
                self._raw_segment_text += raw_text
            self._segment_text = self._apply_corrections(
                self._raw_segment_text
            )
            self._last_text = self._segment_text
            if emit:
                self._emit_event(self._segment_text, is_final=is_final)
            emitted = True
        return emitted

    async def _refine_segment(self) -> str:
        if self._model is None or len(self._segment_audio) < 2:
            return ""

        samples = np.concatenate(self._segment_audio)
        if samples.size < self.chunk_samples * 2:
            return ""

        inference_task = asyncio.create_task(
            asyncio.to_thread(
                self._model.generate,
                input=samples,
                cache={},
                is_final=True,
                chunk_size=[0, 10, 5],
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
                **self._postprocess_kwargs(final_refinement=True),
            )
        )
        self._inference_tasks.add(inference_task)
        inference_task.add_done_callback(self._inference_done)
        try:
            result = await asyncio.shield(inference_task)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - refinement is best effort
            _LOGGER.warning("final ASR refinement failed: %s", error)
            return ""
        finally:
            if inference_task.done():
                self._inference_tasks.discard(inference_task)

        refined_text = "".join(
            self._apply_corrections(text)
            for text in _texts_from_result(result)
            if text.strip()
        )
        return refined_text.strip()

    def _apply_corrections(self, text: str) -> str:
        for wrong, right in self.corrections.items():
            text = text.replace(wrong, right)
        return text

    def _postprocess_kwargs(self, *, final_refinement: bool) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if self.hotwords:
            options["hotword"] = " ".join(self.hotwords)

        if final_refinement:
            if self.corrections:
                options["postprocess_hotwords"] = dict(self.corrections)
                options["postprocess_hotword_fuzzy"] = False
        elif self.corrections:
            options["postprocess_hotwords"] = dict(self.corrections)
            options["postprocess_hotword_fuzzy"] = False
        return options

    def _inference_done(self, task: asyncio.Task[Any]) -> None:
        self._inference_tasks.discard(task)
        if not task.cancelled():
            task.exception()

    @staticmethod
    def _consume_task_result(task: asyncio.Task[Any]) -> None:
        if not task.cancelled():
            task.exception()

    def _emit_event(self, text: str, is_final: bool) -> None:
        elapsed_ms = int(self._segment_samples * 1000 / SAMPLE_RATE)
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

    def _begin_session(self) -> None:
        self._input_queue = asyncio.Queue(_INPUT_QUEUE_SIZE)
        self._events_queue = asyncio.Queue()
        self._model = None
        self._worker = None
        self._events_closed = False
        self._stopped = False
        self._reset_transient_state()

    def _reset_transient_state(self) -> None:
        self._residual = np.empty(0, dtype=np.float32)
        self._segment_audio = []
        self._cache = {}
        self._raw_segment_text = ""
        self._segment_text = ""
        self._last_text = ""
        self._segment_id = 0
        self._segment_samples = 0
        self._closing = False
        self._worker_error = None

    def _close_events(self, error: Exception | None = None) -> None:
        if not self._events_closed:
            self._events_closed = True
            self._events_queue.put_nowait(_EventsEnd(error))

    async def _wait_for_inference_tasks(self) -> None:
        while self._inference_tasks:
            tasks = tuple(self._inference_tasks)
            await asyncio.gather(*tasks, return_exceptions=True)
            await asyncio.sleep(0)

    @staticmethod
    def _clear_queue(queue: asyncio.Queue[Any]) -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return

    def _is_current_generation(self, generation: int) -> bool:
        return generation == self._generation and not self._stopped


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
