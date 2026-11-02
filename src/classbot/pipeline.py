from __future__ import annotations

import asyncio
import inspect
import logging
import time
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from classbot.asr.protocol import SpeechRecognizer
from classbot.domain import AnswerCard, CardStatus, QuestionCandidate
from classbot.question_detector import QuestionDetector, normalize_text

Callback = Callable[..., Any]
_STATE_LIMIT = 128
_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _QuestionState:
    card: AnswerCard
    normalized_text: str
    answer_prompt: str
    context: str
    is_final: bool
    revision: int = 0
    task: asyncio.Task[None] | None = None


class LivePipeline:
    """Coordinate transcript events, question detection, and answer streams."""

    def __init__(
        self,
        recognizer: SpeechRecognizer,
        detector: QuestionDetector,
        client: Any,
        on_card: Callback,
        on_delta: Callback,
        on_status: Callback | None = None,
        on_transcript: Callback | None = None,
        *,
        answer_stability_seconds: float = 0.7,
        scenario_name: str = "",
    ) -> None:
        if answer_stability_seconds < 0:
            raise ValueError("answer_stability_seconds must not be negative")
        self._answer_stability_seconds = answer_stability_seconds
        self._scenario_name = scenario_name
        self._recognizer = recognizer
        self._detector = detector
        self._client = client
        self._on_card = on_card
        self._on_delta = on_delta
        self._on_status = on_status
        self._on_transcript = on_transcript

        self._states: OrderedDict[str, _QuestionState] = OrderedDict()
        self._segment_questions: OrderedDict[int | str | None, str] = OrderedDict()
        self._answer_tasks: set[asyncio.Task[None]] = set()
        self._answer_slots = asyncio.Semaphore(2)
        self._lifecycle_lock = asyncio.Lock()
        self._prepare_task: asyncio.Task[None] | None = None
        self._run_task: asyncio.Task[None] | None = None
        self._stop_task: asyncio.Task[None] | None = None
        self._prepared = False
        self._stop_requested = False
        self._recognizer_stopped = False
        self._stop_complete = False
        self._diagnostics: list[str] = []
        self._cleanup_errors: list[BaseException] = []
        self._session_started_at: float | None = None

    @property
    def diagnostics(self) -> tuple[str, ...]:
        """Return recent callback and cleanup diagnostics for UI or logging."""
        return tuple(self._diagnostics)

    async def prepare(self) -> None:
        """Start recognition without consuming its event iterator."""
        async with self._lifecycle_lock:
            if self._prepared or self._stop_requested:
                return
            task = self._prepare_task
            if task is None:
                task = asyncio.create_task(self._prepare_recognizer())
                self._prepare_task = task

        try:
            await asyncio.shield(task)
        finally:
            async with self._lifecycle_lock:
                if self._prepare_task is task and task.done():
                    self._prepare_task = None

    async def run(self) -> None:
        """Consume transcript events until recognition ends or the pipeline stops."""
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("LivePipeline.run() requires an asyncio task")

        async with self._lifecycle_lock:
            if self._run_task is None:
                self._run_task = current_task
                owner = True
                running_task = current_task
            else:
                owner = False
                running_task = self._run_task

        if not owner:
            await asyncio.shield(running_task)
            return

        try:
            await self._run_owner()
        finally:
            async with self._lifecycle_lock:
                if self._run_task is current_task:
                    self._run_task = None

    async def stop(self) -> None:
        """Cancel answer streams and stop recognition; safe to call repeatedly."""
        async with self._lifecycle_lock:
            self._stop_requested = True
            task = self._stop_task
            if task is None:
                task = asyncio.create_task(self._stop_owner())
                self._stop_task = task

        await asyncio.shield(task)

    async def _run_owner(self) -> None:
        try:
            self._session_started_at = time.monotonic()
            if self._stop_requested:
                return
            if not self._prepared:
                await self.prepare()
            if self._stop_requested:
                return

            await self._call(self._on_status, "listening", "on_status")
            async for event in self._recognizer.events():
                if self._stop_requested:
                    break
                await self._call(
                    self._on_transcript,
                    event.text,
                    "on_transcript",
                )
                candidates = self._detector.feed(event)
                if event.is_final:
                    previous_id = self._segment_questions.pop(event.segment_id, None)
                    if previous_id is not None and not candidates:
                        await self._invalidate_question(previous_id)
                for candidate in candidates:
                    if self._stop_requested:
                        break
                    await self._handle_candidate(candidate)
                    if not event.is_final:
                        self._segment_questions[event.segment_id] = candidate.question_id
                        self._segment_questions.move_to_end(event.segment_id)
                        while len(self._segment_questions) > _STATE_LIMIT:
                            self._segment_questions.popitem(last=False)

            await self._wait_for_answer_tasks()
        except BaseException as error:
            await self._cleanup_after_run_error(error)
            raise
        else:
            await self.stop()

    async def _prepare_recognizer(self) -> None:
        try:
            await self._recognizer.start()
        except BaseException as error:
            await self._handle_start_failure(error)
            raise

        async with self._lifecycle_lock:
            should_stop = self._stop_requested or self._stop_complete
            if not should_stop:
                self._prepared = True

        if should_stop:
            cleanup_error = await self._stop_recognizer_once()
            if cleanup_error is not None:
                self._record_cleanup_error("start/stop race", cleanup_error)
            return

        await self._call(self._on_status, "ready", "on_status")

    async def _handle_start_failure(self, error: BaseException) -> None:
        async with self._lifecycle_lock:
            self._stop_requested = True
            self._prepared = False

        cleanup_error = await self._stop_recognizer_once()
        if cleanup_error is not None:
            self._add_exception_note(
                error,
                "recognizer.stop after start failure",
                cleanup_error,
            )
            self._record_cleanup_error(
                "recognizer.stop after start failure",
                cleanup_error,
            )

        async with self._lifecycle_lock:
            self._stop_complete = True
        await self._call(self._on_status, "start_error", "on_status")

    async def _stop_owner(self) -> None:
        async with self._lifecycle_lock:
            if self._stop_complete:
                return
            self._stop_requested = True
            prepare_task = self._prepare_task

        if prepare_task is not None and prepare_task is not asyncio.current_task():
            try:
                await asyncio.shield(prepare_task)
            except BaseException as error:  # noqa: BLE001 - cleanup preserves primary errors
                if not isinstance(error, asyncio.CancelledError):
                    self._record_cleanup_error("prepare during stop", error)

        await self._cancel_answer_tasks()
        cleanup_error = await self._stop_recognizer_once()
        if cleanup_error is not None:
            self._record_cleanup_error("recognizer.stop", cleanup_error)

        async with self._lifecycle_lock:
            self._prepared = False
            self._stop_complete = True
        await self._call(self._on_status, "stopped", "on_status")

    async def _stop_recognizer_once(self) -> BaseException | None:
        async with self._lifecycle_lock:
            if self._recognizer_stopped:
                return None
            self._recognizer_stopped = True

        try:
            await self._recognizer.stop()
        except BaseException as error:  # noqa: BLE001 - stop must be attempted once
            return error
        return None

    async def _cleanup_after_run_error(self, error: BaseException) -> None:
        cleanup_start = len(self._cleanup_errors)
        try:
            await asyncio.shield(self.stop())
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve run error
            if not isinstance(cleanup_error, asyncio.CancelledError):
                self._record_cleanup_error("pipeline stop", cleanup_error)

        for cleanup_error in self._cleanup_errors[cleanup_start:]:
            self._add_exception_note(error, "pipeline cleanup", cleanup_error)

    async def _handle_candidate(self, candidate: QuestionCandidate) -> None:
        normalized_text = normalize_text(candidate.text).rstrip("呢啊呀吧")
        answer_prompt = candidate.text if self._scenario_name else self._answer_prompt(candidate)
        state = self._states.get(candidate.question_id)

        if state is None:
            card = AnswerCard(
                question_id=candidate.question_id,
                question=candidate.text,
                subject=candidate.subject,
                status=CardStatus.DETECTED,
                timestamp_ms=self._elapsed_ms(),
                scenario_name=self._scenario_name,
            )
            state = _QuestionState(
                card=card,
                normalized_text=normalized_text,
                answer_prompt=answer_prompt,
                context=candidate.context,
                is_final=candidate.is_final,
            )
            self._states[candidate.question_id] = state
            await self._evict_old_states()
            await self._call(self._on_card, card, "on_card")
            self._start_answer(state, answer_prompt, candidate.subject)
            return

        self._states.move_to_end(candidate.question_id)
        if state.normalized_text == normalized_text:
            changed = False
            restart_answer = False
            promote_pending = False
            if candidate.is_final and not state.is_final:
                state.is_final = True
                changed = True
                promote_pending = state.card.status is CardStatus.DETECTED
            if candidate.is_final and state.card.question != candidate.text:
                state.card.question = candidate.text
                changed = True
            if state.card.subject != candidate.subject:
                state.card.subject = candidate.subject
                changed = True
            context_changed = state.context != candidate.context
            context_requires_restart = context_changed and (
                state.card.status is CardStatus.DETECTED
                or any(word in candidate.text for word in (
                    "这个", "那个", "这些", "那些", "上述", "前者", "后者", "它", "还有", "分别",
                ))
            )
            if context_requires_restart or promote_pending:
                old_task = state.task
                state.revision += 1
                await self._cancel_task(old_task, "context revision")
                state.task = None
                state.answer_prompt = answer_prompt
                state.context = candidate.context
                state.card.answer = ""
                state.card.sources = ()
                state.card.status = CardStatus.DETECTED
                state.card.error = None
                changed = True
                restart_answer = True
            elif context_changed:
                state.context = candidate.context
            if changed:
                await self._call(self._on_card, state.card, "on_card")
            if restart_answer:
                self._start_answer(state, answer_prompt, candidate.subject)
            return

        old_task = state.task
        state.revision += 1
        await self._cancel_task(old_task, "revision")
        state.task = None
        state.normalized_text = normalized_text
        state.answer_prompt = answer_prompt
        state.context = candidate.context
        state.is_final = candidate.is_final
        state.card.question = candidate.text
        state.card.subject = candidate.subject
        state.card.answer = ""
        state.card.sources = ()
        state.card.status = CardStatus.DETECTED
        state.card.error = None
        await self._call(self._on_card, state.card, "on_card")
        self._start_answer(state, answer_prompt, candidate.subject)

    async def _invalidate_question(self, question_id: str) -> None:
        state = self._states.get(question_id)
        if state is None:
            return
        state.revision += 1
        await self._cancel_task(state.task, "rejected final transcript")
        state.task = None
        state.card.answer = ""
        state.card.sources = ()
        state.card.error = None
        state.card.status = CardStatus.CANCELLED
        await self._call(self._on_card, state.card, "on_card")

    @staticmethod
    def _answer_prompt(candidate: QuestionCandidate) -> str:
        context = candidate.context.strip()
        if not context:
            return candidate.text
        return f"背景信息：{context}\n问题：{candidate.text}"

    async def _evict_old_states(self) -> None:
        while len(self._states) > _STATE_LIMIT:
            _question_id, state = self._states.popitem(last=False)
            state.revision += 1
            await self._cancel_task(state.task, "state eviction")
            state.task = None

    def _start_answer(
        self,
        state: _QuestionState,
        question: str,
        subject: str,
    ) -> None:
        if self._stop_requested:
            return
        revision = state.revision
        task = asyncio.create_task(
            self._answer_question(state, revision, question, subject)
        )
        state.task = task
        self._answer_tasks.add(task)
        task.add_done_callback(self._forget_answer_task)

    async def _answer_question(
        self,
        state: _QuestionState,
        revision: int,
        question: str,
        subject: str,
    ) -> None:
        try:
            async with self._answer_slots:
                if self._is_current(state, revision):
                    await self._generate_answer(state, revision, question, subject)
        except asyncio.CancelledError:
            if self._is_current(state, revision):
                state.card.status = CardStatus.CANCELLED
                await self._call(self._on_card, state.card, "on_card")
            raise

    async def _generate_answer(
        self,
        state: _QuestionState,
        revision: int,
        question: str,
        subject: str,
    ) -> None:
        try:
            if not state.is_final:
                await asyncio.sleep(self._answer_stability_seconds)
            if not self._is_current(state, revision):
                return
            state.card.status = CardStatus.ANSWERING
            state.card.error = None
            await self._call(self._on_card, state.card, "on_card")
            prepare = getattr(self._client, "prepare_answer", None)
            prepare_local = getattr(self._client, "prepare_answer_with_local_context", None)
            if self._scenario_name and callable(prepare_local):
                prepared = await prepare_local(question, subject, local_context=state.context)
            elif prepare is not None:
                prepared = await prepare(question, subject)
            else:
                prepared = None
            if prepared is not None:
                if not self._is_current(state, revision):
                    return
                state.card.sources = prepared.sources
                await self._call(self._on_card, state.card, "on_card")
            grounded_answer = getattr(
                self._client, "grounded_answer_with_web_fallback", None
            ) or getattr(self._client, "grounded_answer", None)
            if self._scenario_name and not callable(grounded_answer):
                raise ValueError("答辩客户端缺少证据核验接口，未发布未经核验的回答")
            if self._scenario_name:
                if not self._is_current(state, revision):
                    return
                if prepared is None:
                    raise ValueError("答辩证据核验缺少本次资料候选")
                # A complete, proof-bound response is validated before any
                # text reaches the card. No free-form draft is sent or shown.
                verified = await grounded_answer(question, subject, prepared=prepared)
                if not self._is_current(state, revision):
                    return
                verified_text = getattr(verified, "text", None)
                if not isinstance(verified_text, str) or not verified_text.strip():
                    raise ValueError("答辩证据核验没有返回可发布的内容")
                state.card.answer = verified_text
                await self._call(
                    self._on_delta, state.card.question_id, state.card.answer, "on_delta",
                )
            else:
                if prepared is not None:
                    stream = self._client.stream_answer(question, subject, prepared=prepared)
                else:
                    stream: AsyncIterator[str] = self._client.stream_answer(question, subject)
                stream_failure: BaseException | None = None
                try:
                    async for delta in stream:
                        if not self._is_current(state, revision):
                            return
                        if not delta:
                            continue
                        state.card.answer += str(delta)
                        await self._call(
                            self._on_delta,
                            state.card.question_id,
                            state.card.answer,
                            "on_delta",
                        )
                except BaseException as error:
                    stream_failure = error
                    raise
                finally:
                    # Callback cancellation can leave the legacy stream at
                    # yield. Its owner closes it in this same answer task.
                    close_stream = getattr(stream, "aclose", None)
                    if callable(close_stream):
                        try:
                            await close_stream()
                        except BaseException as cleanup_error:
                            if stream_failure is None:
                                raise
                            self._record_cleanup_error("answer stream cleanup", cleanup_error)
                            self._add_exception_note(stream_failure, "answer stream cleanup", cleanup_error)

            if self._is_current(state, revision):
                state.card.status = CardStatus.ANSWERED
                state.card.error = None
                await self._call(self._on_card, state.card, "on_card")
        except asyncio.CancelledError:
            if self._is_current(state, revision):
                state.card.status = CardStatus.CANCELLED
                await self._call(self._on_card, state.card, "on_card")
            raise
        except Exception as error:  # noqa: BLE001 - client streams are injected
            if self._is_current(state, revision):
                state.card.status = CardStatus.ERROR
                state.card.error = self._readable_error(error)
                await self._call(self._on_card, state.card, "on_card")
                await self._call(self._on_status, "api_error", "on_status")

    async def _cancel_task(
        self,
        task: asyncio.Task[None] | None,
        context: str,
    ) -> None:
        if task is None:
            return
        task.cancel()
        results = await asyncio.gather(task, return_exceptions=True)
        self._report_task_results(results, context)

    async def _cancel_answer_tasks(self) -> None:
        tasks = tuple(self._answer_tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            self._report_task_results(results, "answer task cleanup")

    async def _wait_for_answer_tasks(self) -> None:
        if self._answer_tasks:
            results = await asyncio.gather(
                *tuple(self._answer_tasks),
                return_exceptions=True,
            )
            self._report_task_results(results, "answer task completion")

    def _report_task_results(
        self,
        results: list[Any],
        context: str,
    ) -> None:
        for result in results:
            if isinstance(result, BaseException) and not isinstance(
                result, asyncio.CancelledError
            ):
                self._record_cleanup_error(context, result)

    def _forget_answer_task(self, task: asyncio.Task[None]) -> None:
        self._answer_tasks.discard(task)
        for state in self._states.values():
            if state.task is task:
                state.task = None
                break
        if task.cancelled():
            return
        error = task.exception()
        if error is not None:
            self._record_cleanup_error("answer task", error)

    def _record_cleanup_error(self, context: str, error: BaseException) -> None:
        self._cleanup_errors.append(error)
        self._record_diagnostic(
            f"{context} failed: {self._readable_error(error)}"
        )

    def _record_callback_error(self, name: str, error: Exception) -> None:
        self._record_diagnostic(
            f"{name} callback failed: {self._readable_error(error)}"
        )

    def _record_diagnostic(self, message: str) -> None:
        self._diagnostics.append(message)
        if len(self._diagnostics) > _STATE_LIMIT:
            del self._diagnostics[:-_STATE_LIMIT]
        _LOGGER.error(message)

    @staticmethod
    def _is_current(state: _QuestionState, revision: int) -> bool:
        return state.revision == revision

    def _elapsed_ms(self) -> int:
        if self._session_started_at is None:
            return 0
        return max(0, int((time.monotonic() - self._session_started_at) * 1000))

    @staticmethod
    def _readable_error(error: BaseException) -> str:
        message = str(error).strip()
        return message or error.__class__.__name__

    @staticmethod
    def _add_exception_note(
        error: BaseException,
        context: str,
        detail: BaseException,
    ) -> None:
        error.add_note(f"{context}: {LivePipeline._readable_error(detail)}")

    async def _call(
        self,
        callback: Callback | None,
        *args: Any,
    ) -> None:
        if callback is None:
            return
        name = args[-1]
        callback_args = args[:-1]
        try:
            result = callback(*callback_args)
            if inspect.isawaitable(result):
                await result
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - UI callbacks are injected
            self._record_callback_error(str(name), error)
