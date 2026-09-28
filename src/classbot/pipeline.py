from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any

from classbot.asr.protocol import SpeechRecognizer
from classbot.domain import AnswerCard, CardStatus, QuestionCandidate
from classbot.question_detector import QuestionDetector, normalize_text

Callback = Callable[..., Any]


@dataclass(slots=True)
class _QuestionState:
    card: AnswerCard
    normalized_text: str
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
    ) -> None:
        self._recognizer = recognizer
        self._detector = detector
        self._client = client
        self._on_card = on_card
        self._on_delta = on_delta
        self._on_status = on_status

        self._states: dict[str, _QuestionState] = {}
        self._answer_tasks: set[asyncio.Task[None]] = set()
        self._prepare_lock = asyncio.Lock()
        self._stop_lock = asyncio.Lock()
        self._prepared = False
        self._stop_requested = False
        self._recognizer_stopped = False
        self._stop_complete = False

    async def prepare(self) -> None:
        """Start recognition without consuming its event iterator."""
        if self._prepared or self._stop_requested:
            return

        async with self._prepare_lock:
            if self._prepared or self._stop_requested:
                return
            await self._recognizer.start()
            self._prepared = True
            await self._call(self._on_status, "ready")

    async def run(self) -> None:
        """Consume transcript events until recognition ends or the pipeline stops."""
        if self._stop_requested:
            return
        if not self._prepared:
            await self.prepare()
        if self._stop_requested:
            return

        try:
            await self._call(self._on_status, "listening")
            async for event in self._recognizer.events():
                if self._stop_requested:
                    break
                candidates = self._detector.feed(event)
                for candidate in candidates:
                    if self._stop_requested:
                        break
                    await self._handle_candidate(candidate)

            await self._wait_for_answer_tasks()
        except asyncio.CancelledError:
            await self.stop()
            raise
        except BaseException:
            await self.stop()
            raise
        else:
            await self.stop()

    async def stop(self) -> None:
        """Cancel answer streams and stop recognition; safe to call repeatedly."""
        async with self._stop_lock:
            if self._stop_complete:
                return

            self._stop_requested = True
            tasks = tuple(self._answer_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

            if not self._recognizer_stopped:
                await self._recognizer.stop()
                self._recognizer_stopped = True

            self._stop_complete = True
            await self._call(self._on_status, "stopped")

    async def _handle_candidate(self, candidate: QuestionCandidate) -> None:
        normalized_text = normalize_text(candidate.text)
        state = self._states.get(candidate.question_id)

        if state is None:
            card = AnswerCard(
                question_id=candidate.question_id,
                question=candidate.text,
                subject=candidate.subject,
                status=CardStatus.DETECTED,
            )
            state = _QuestionState(
                card=card,
                normalized_text=normalized_text,
                is_final=candidate.is_final,
            )
            self._states[candidate.question_id] = state
            await self._call(self._on_card, card)
            self._start_answer(state, candidate.text, candidate.subject)
            return

        if state.normalized_text == normalized_text:
            changed = False
            if candidate.is_final and not state.is_final:
                state.is_final = True
                changed = True
            if candidate.is_final and state.card.question != candidate.text:
                state.card.question = candidate.text
                changed = True
            if state.card.subject != candidate.subject:
                state.card.subject = candidate.subject
                changed = True
            if changed:
                await self._call(self._on_card, state.card)
            return

        state.revision += 1
        if state.task is not None and not state.task.done():
            state.task.cancel()
        state.normalized_text = normalized_text
        state.is_final = candidate.is_final
        state.card.question = candidate.text
        state.card.subject = candidate.subject
        state.card.answer = ""
        state.card.status = CardStatus.DETECTED
        state.card.error = None
        await self._call(self._on_card, state.card)
        self._start_answer(state, candidate.text, candidate.subject)

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
            stream: AsyncIterator[str] = self._client.stream_answer(question, subject)
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
                )

            if self._is_current(state, revision):
                state.card.status = CardStatus.ANSWERED
                state.card.error = None
                await self._call(self._on_card, state.card)
        except asyncio.CancelledError:
            if self._is_current(state, revision):
                state.card.status = CardStatus.CANCELLED
                await self._call(self._on_card, state.card)
            raise
        except Exception as error:  # noqa: BLE001 - client streams are injected
            if self._is_current(state, revision):
                state.card.status = CardStatus.ERROR
                state.card.error = self._readable_error(error)
                await self._call(self._on_card, state.card)

    async def _wait_for_answer_tasks(self) -> None:
        if self._answer_tasks:
            await asyncio.gather(*tuple(self._answer_tasks), return_exceptions=True)

    def _forget_answer_task(self, task: asyncio.Task[None]) -> None:
        self._answer_tasks.discard(task)
        for state in self._states.values():
            if state.task is task:
                state.task = None
                break
        try:
            task.exception()
        except asyncio.CancelledError:
            return

    @staticmethod
    def _is_current(state: _QuestionState, revision: int) -> bool:
        return state.revision == revision

    @staticmethod
    def _readable_error(error: Exception) -> str:
        message = str(error).strip()
        return message or error.__class__.__name__

    @staticmethod
    async def _call(
        callback: Callback | None,
        *args: Any,
    ) -> None:
        if callback is None:
            return
        result = callback(*args)
        if inspect.isawaitable(result):
            await result
