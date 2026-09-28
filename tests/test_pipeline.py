import asyncio
from collections.abc import AsyncIterator, Iterable

import pytest

from classbot.domain import AnswerCard, CardStatus, QuestionCandidate, TranscriptEvent
from classbot.pipeline import LivePipeline


class FakeRecognizer:
    def __init__(self, events: Iterable[TranscriptEvent]) -> None:
        self._events = list(events)
        self.started = 0
        self.stop_calls = 0
        self.events_calls = 0
        self.stopped = asyncio.Event()

    async def start(self) -> None:
        self.started += 1

    def push_audio(self, frame: bytes) -> None:
        del frame

    def finish_segment(self) -> None:
        pass

    def events(self) -> AsyncIterator[TranscriptEvent]:
        self.events_calls += 1

        async def stream() -> AsyncIterator[TranscriptEvent]:
            for event in self._events:
                yield event

        return stream()

    async def stop(self) -> None:
        self.stop_calls += 1
        self.stopped.set()


class FakeDetector:
    def __init__(self, responses: dict[str, list[QuestionCandidate]]) -> None:
        self._responses = responses
        self.events: list[TranscriptEvent] = []

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        self.events.append(event)
        return self._responses.get(event.text, [])


class FakeClient:
    def __init__(self, streams: dict[str, Iterable[str] | BaseException]) -> None:
        self._streams = streams
        self.calls: list[tuple[str, str]] = []
        self.close_calls = 0

    def stream_answer(self, question: str, subject: str) -> AsyncIterator[str]:
        self.calls.append((question, subject))
        stream = self._streams[question]

        async def answer() -> AsyncIterator[str]:
            if isinstance(stream, BaseException):
                raise stream
            for delta in stream:
                await asyncio.sleep(0)
                yield delta

        return answer()

    async def aclose(self) -> None:
        self.close_calls += 1


def candidate(
    question_id: str,
    text: str,
    *,
    is_final: bool = True,
    subject: str = "历史",
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        text=text,
        subject=subject,
        confidence=0.9,
        is_final=is_final,
    )


def event(text: str, *, is_final: bool = True) -> TranscriptEvent:
    return TranscriptEvent(text=text, is_final=is_final)


def card_snapshot(card: AnswerCard) -> tuple[str, str, str, CardStatus, str | None]:
    return (card.question_id, card.question, card.answer, card.status, card.error)


@pytest.mark.asyncio
async def test_prepare_starts_recognizer_without_consuming_events() -> None:
    recognizer = FakeRecognizer([])
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )

    await pipeline.prepare()

    assert recognizer.started == 1
    assert recognizer.events_calls == 0


@pytest.mark.asyncio
async def test_final_question_streams_one_answer_card_to_answered() -> None:
    transcript = event("马克思出生于哪一年？")
    question = candidate("q1", transcript.text)
    recognizer = FakeRecognizer([transcript])
    detector = FakeDetector({transcript.text: [question]})
    client = FakeClient({question.text: ["马克思出生于", "1818年。"]})
    card_updates: list[tuple[str, str, str, CardStatus, str | None]] = []
    deltas: list[tuple[str, str]] = []

    def on_card(card: AnswerCard) -> None:
        card_updates.append(card_snapshot(card))

    async def on_delta(question_id: str, answer: str) -> None:
        deltas.append((question_id, answer))

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        on_delta,
    )

    await pipeline.run()

    assert len(card_updates) == 2
    assert card_updates[0] == (
        "q1",
        question.text,
        "",
        CardStatus.DETECTED,
        None,
    )
    assert card_updates[-1] == (
        "q1",
        question.text,
        "马克思出生于1818年。",
        CardStatus.ANSWERED,
        None,
    )
    assert [answer for _question_id, answer in deltas] == [
        "马克思出生于",
        "马克思出生于1818年。",
    ]
    assert client.calls == [(question.text, "历史")]


@pytest.mark.asyncio
async def test_partial_then_same_final_text_uses_one_card_and_one_request() -> None:
    partial = event("什么是文艺复兴", is_final=False)
    final = event("什么是文艺复兴？")
    partial_question = candidate("q1", partial.text, is_final=False)
    final_question = candidate("q1", final.text)
    recognizer = FakeRecognizer([partial, final])
    detector = FakeDetector(
        {
            partial.text: [partial_question],
            final.text: [final_question],
        }
    )
    client = FakeClient({partial_question.text: ["欧洲历史上的思想文化运动"]})
    card_updates: list[tuple[str, str, str, CardStatus, str | None]] = []

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        lambda card: card_updates.append(card_snapshot(card)),
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert len({update[0] for update in card_updates}) == 1
    assert len(client.calls) == 1
    assert card_updates[-1][1] == final.text
    assert card_updates[-1][2] == "欧洲历史上的思想文化运动"
    assert card_updates[-1][3] is CardStatus.ANSWERED


@pytest.mark.asyncio
async def test_substantive_revision_cancels_old_stream_and_keeps_new_result() -> None:
    old_event = event("旧问题", is_final=False)
    new_event = event("新问题？")
    old_question = candidate("q1", old_event.text, is_final=False)
    new_question = candidate("q1", new_event.text)
    old_delta_seen = asyncio.Event()
    old_cancelled = asyncio.Event()
    never_set = asyncio.Event()

    class RevisionClient:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def stream_answer(self, question: str, subject: str) -> AsyncIterator[str]:
            self.calls.append((question, subject))
            if question == old_question.text:
                return self.old_stream()
            return self.new_stream()

        async def old_stream(self) -> AsyncIterator[str]:
            try:
                yield "旧答案"
                await never_set.wait()
            finally:
                old_cancelled.set()

        async def new_stream(self) -> AsyncIterator[str]:
            yield "新答案"

    client = RevisionClient()

    class RevisionRecognizer(FakeRecognizer):
        def events(self) -> AsyncIterator[TranscriptEvent]:
            self.events_calls += 1

            async def stream() -> AsyncIterator[TranscriptEvent]:
                yield old_event
                await old_delta_seen.wait()
                yield new_event

            return stream()

    recognizer = RevisionRecognizer([])
    detector = FakeDetector(
        {
            old_event.text: [old_question],
            new_event.text: [new_question],
        }
    )
    card_updates: list[tuple[str, str, str, CardStatus, str | None]] = []

    async def on_card(card: AnswerCard) -> None:
        card_updates.append(card_snapshot(card))

    def on_delta(question_id: str, answer: str) -> None:
        assert question_id == "q1"
        if answer == "旧答案":
            old_delta_seen.set()

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        on_delta,
    )

    await pipeline.run()

    assert client.calls == [("旧问题", "历史"), ("新问题？", "历史")]
    assert old_cancelled.is_set()
    assert card_updates[-1] == (
        "q1",
        "新问题？",
        "新答案",
        CardStatus.ANSWERED,
        None,
    )
    assert all("旧答案" not in update[2] for update in card_updates[-2:])
    assert len({update[0] for update in card_updates}) == 1


@pytest.mark.asyncio
async def test_different_question_ids_have_independent_answer_streams() -> None:
    first = event("第一题")
    second = event("第二题")
    first_question = candidate("q1", first.text)
    second_question = candidate("q2", second.text)
    recognizer = FakeRecognizer([first, second])
    detector = FakeDetector(
        {
            first.text: [first_question],
            second.text: [second_question],
        }
    )
    client = FakeClient(
        {
            first_question.text: ["一", "题"],
            second_question.text: ["二", "题"],
        }
    )
    cards: dict[str, AnswerCard] = {}

    def on_card(card: AnswerCard) -> None:
        cards[card.question_id] = AnswerCard(
            question_id=card.question_id,
            question=card.question,
            subject=card.subject,
            answer=card.answer,
            status=card.status,
            error=card.error,
        )

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert cards["q1"].answer == "一题"
    assert cards["q2"].answer == "二题"
    assert cards["q1"].status is CardStatus.ANSWERED
    assert cards["q2"].status is CardStatus.ANSWERED


@pytest.mark.asyncio
async def test_client_error_marks_only_one_card_and_pipeline_continues() -> None:
    failed = event("失败问题")
    successful = event("成功问题")
    failed_question = candidate("q1", failed.text)
    successful_question = candidate("q2", successful.text)
    recognizer = FakeRecognizer([failed, successful])
    detector = FakeDetector(
        {
            failed.text: [failed_question],
            successful.text: [successful_question],
        }
    )
    client = FakeClient(
        {
            failed_question.text: RuntimeError("服务暂时不可用"),
            successful_question.text: ["正常答案"],
        }
    )
    card_updates: dict[str, list[tuple[str, str, str, CardStatus, str | None]]] = {
        "q1": [],
        "q2": [],
    }

    def on_card(card: AnswerCard) -> None:
        card_updates[card.question_id].append(card_snapshot(card))

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert card_updates["q1"][-1][3] is CardStatus.ERROR
    assert card_updates["q1"][-1][4] == "服务暂时不可用"
    assert card_updates["q2"][-1][2:] == ("正常答案", CardStatus.ANSWERED, None)


@pytest.mark.asyncio
async def test_stop_is_idempotent_cancels_answer_tasks_and_does_not_close_client() -> None:
    transcript = event("持续问题")
    question = candidate("q1", transcript.text)
    recognizer = FakeRecognizer([transcript])
    detector = FakeDetector({transcript.text: [question]})
    answer_started = asyncio.Event()
    answer_cancelled = asyncio.Event()
    never_set = asyncio.Event()

    class BlockingClient:
        close_calls = 0

        def stream_answer(self, _question: str, _subject: str) -> AsyncIterator[str]:
            async def answer() -> AsyncIterator[str]:
                answer_started.set()
                try:
                    await never_set.wait()
                    yield "不会到达"
                finally:
                    answer_cancelled.set()

            return answer()

        async def aclose(self) -> None:
            self.close_calls += 1

    client = BlockingClient()
    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        lambda _card: None,
        lambda _question_id, _answer: None,
    )
    run_task = asyncio.create_task(pipeline.run())

    await answer_started.wait()
    await pipeline.stop()
    await pipeline.stop()
    await run_task

    assert answer_cancelled.is_set()
    assert recognizer.stop_calls == 1
    assert client.close_calls == 0


@pytest.mark.asyncio
async def test_non_cancel_recognizer_error_is_propagated_after_cleanup() -> None:
    class FailingRecognizer(FakeRecognizer):
        def events(self) -> AsyncIterator[TranscriptEvent]:
            self.events_calls += 1

            async def stream() -> AsyncIterator[TranscriptEvent]:
                raise RuntimeError("识别器崩溃")
                yield event("不可达")

            return stream()

    recognizer = FailingRecognizer([])
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )

    with pytest.raises(RuntimeError, match="识别器崩溃"):
        await pipeline.run()

    assert recognizer.stop_calls == 1


@pytest.mark.asyncio
async def test_concurrent_run_calls_share_one_event_consumer() -> None:
    class SharedRunRecognizer(FakeRecognizer):
        def __init__(self) -> None:
            super().__init__([])
            self.events_started = asyncio.Event()
            self.release_events = asyncio.Event()

        def events(self) -> AsyncIterator[TranscriptEvent]:
            self.events_calls += 1
            self.events_started.set()

            async def stream() -> AsyncIterator[TranscriptEvent]:
                await self.release_events.wait()
                if False:
                    yield event("不可达")

            return stream()

    recognizer = SharedRunRecognizer()
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )
    first_run = asyncio.create_task(pipeline.run())
    await recognizer.events_started.wait()
    second_run = asyncio.create_task(pipeline.run())
    await asyncio.sleep(0)
    observed_events_calls = recognizer.events_calls
    recognizer.release_events.set()

    await asyncio.gather(first_run, second_run)

    assert observed_events_calls == 1


@pytest.mark.asyncio
async def test_start_failure_stops_once_and_preserves_start_error() -> None:
    class StartFailRecognizer(FakeRecognizer):
        async def start(self) -> None:
            self.started += 1
            raise RuntimeError("recognizer start failed")

    recognizer = StartFailRecognizer([])
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )

    with pytest.raises(RuntimeError, match="recognizer start failed") as caught:
        await pipeline.prepare()

    assert recognizer.stop_calls == 1
    await pipeline.stop()
    assert recognizer.stop_calls == 1
    assert str(caught.value) == "recognizer start failed"


@pytest.mark.asyncio
async def test_stop_waits_for_start_and_does_not_publish_prepared_after_race() -> None:
    class SlowStartRecognizer(FakeRecognizer):
        def __init__(self) -> None:
            super().__init__([])
            self.start_entered = asyncio.Event()
            self.release_start = asyncio.Event()

        async def start(self) -> None:
            self.started += 1
            self.start_entered.set()
            await self.release_start.wait()

    recognizer = SlowStartRecognizer()
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )
    prepare_task = asyncio.create_task(pipeline.prepare())
    await recognizer.start_entered.wait()
    stop_task = asyncio.create_task(pipeline.stop())
    await asyncio.sleep(0)
    stop_was_waiting = not stop_task.done()
    recognizer.release_start.set()

    await asyncio.gather(prepare_task, stop_task)

    assert stop_was_waiting
    assert recognizer.stop_calls == 1
    assert pipeline._prepared is False


@pytest.mark.asyncio
async def test_events_error_wins_over_recognizer_stop_error() -> None:
    class ErroringRecognizer(FakeRecognizer):
        def events(self) -> AsyncIterator[TranscriptEvent]:
            self.events_calls += 1

            async def stream() -> AsyncIterator[TranscriptEvent]:
                raise RuntimeError("events failed")
                yield event("不可达")

            return stream()

        async def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("stop failed")

    recognizer = ErroringRecognizer([])
    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )

    with pytest.raises(RuntimeError, match="events failed") as caught:
        await pipeline.run()

    assert recognizer.stop_calls == 1
    assert any("stop failed" in note for note in caught.value.__notes__)


@pytest.mark.asyncio
async def test_revision_waits_for_async_old_callback_before_new_revision_callback() -> None:
    old_event = event("旧问题", is_final=False)
    new_event = event("新问题？")
    old_question = candidate("q1", old_event.text, is_final=False)
    new_question = candidate("q1", new_event.text)
    old_callback_started = asyncio.Event()
    release_old_callback = asyncio.Event()
    new_event_yielded = asyncio.Event()
    order: list[str] = []

    class RevisionRecognizer(FakeRecognizer):
        def events(self) -> AsyncIterator[TranscriptEvent]:
            self.events_calls += 1

            async def stream() -> AsyncIterator[TranscriptEvent]:
                yield old_event
                await old_callback_started.wait()
                new_event_yielded.set()
                yield new_event

            return stream()

    class RevisionClient:
        def stream_answer(self, question: str, _subject: str) -> AsyncIterator[str]:
            if question == old_question.text:
                return self.old_stream()
            return self.new_stream()

        async def old_stream(self) -> AsyncIterator[str]:
            yield "旧答案"

        async def new_stream(self) -> AsyncIterator[str]:
            yield "新答案"

    async def release_after_new_event() -> None:
        await new_event_yielded.wait()
        await asyncio.sleep(0)
        release_old_callback.set()

    async def on_card(card: AnswerCard) -> None:
        if card.question == new_question.text:
            order.append("new-card")

    async def on_delta(_question_id: str, answer: str) -> None:
        if answer != "旧答案":
            return
        order.append("old-start")
        old_callback_started.set()
        try:
            await release_old_callback.wait()
        except asyncio.CancelledError:
            order.append("old-cancel")
            await release_old_callback.wait()
        order.append("old-end")

    recognizer = RevisionRecognizer([])
    detector = FakeDetector(
        {
            old_event.text: [old_question],
            new_event.text: [new_question],
        }
    )
    pipeline = LivePipeline(
        recognizer,
        detector,
        RevisionClient(),
        on_card,
        on_delta,
    )
    release_task = asyncio.create_task(release_after_new_event())

    await pipeline.run()
    await release_task

    assert order.index("old-end") < order.index("new-card")
    assert "old-cancel" in order


@pytest.mark.asyncio
async def test_recent_states_are_bounded() -> None:
    transcript_events = [event(f"问题-{index}") for index in range(130)]
    candidates = {
        transcript.text: [candidate(f"q-{index}", transcript.text)]
        for index, transcript in enumerate(transcript_events)
    }
    recognizer = FakeRecognizer(transcript_events)
    pipeline = LivePipeline(
        recognizer,
        FakeDetector(candidates),
        FakeClient({transcript.text: [] for transcript in transcript_events}),
        lambda _card: None,
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert len(pipeline._states) <= 128
    assert "q-0" not in pipeline._states
    assert "q-129" in pipeline._states


@pytest.mark.asyncio
async def test_callback_errors_do_not_pollute_client_error_status() -> None:
    transcript = event("回调问题")
    question = candidate("q1", transcript.text)
    recognizer = FakeRecognizer([transcript])
    detector = FakeDetector({transcript.text: [question]})
    client = FakeClient({question.text: ["答案"]})
    card_refs: list[AnswerCard] = []

    def on_status(_status: str) -> None:
        raise RuntimeError("status callback failed")

    def on_card(card: AnswerCard) -> None:
        card_refs.append(card)
        if card.status is CardStatus.ANSWERED:
            raise RuntimeError("card callback failed")

    def on_delta(_question_id: str, _answer: str) -> None:
        raise RuntimeError("delta callback failed")

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        on_delta,
        on_status,
    )

    await pipeline.run()

    assert card_refs[-1].status is CardStatus.ANSWERED
    assert card_refs[-1].error is None
