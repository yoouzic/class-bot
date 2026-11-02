import asyncio
from collections.abc import AsyncIterator, Iterable

import pytest

from classbot.domain import (
    AnswerCard,
    CardStatus,
    QuestionCandidate,
    SourceKind,
    SourceRef,
    TranscriptEvent,
)
from classbot.llm.deepseek_client import PreparedAnswer
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

    def stream_answer(
        self, question: str, subject: str, *, prepared: object | None = None
    ) -> AsyncIterator[str]:
        del prepared
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
    context: str = "",
) -> QuestionCandidate:
    return QuestionCandidate(
        question_id=question_id,
        text=text,
        subject=subject,
        confidence=0.9,
        is_final=is_final,
        context=context,
    )


def event(text: str, *, is_final: bool = True) -> TranscriptEvent:
    return TranscriptEvent(text=text, is_final=is_final)


def card_snapshot(card: AnswerCard) -> tuple[str, str, str, CardStatus, str | None]:
    return (card.question_id, card.question, card.answer, card.status, card.error)


@pytest.mark.asyncio
async def test_live_revisions_wait_for_stability_before_one_request():
    texts = ["认识的过程是什么", "认识的过程是什么呢", "认识的过程有哪些呢"]

    class LiveRecognizer(FakeRecognizer):
        async def events(self):
            for text in texts:
                yield event(text, is_final=False)
                await asyncio.sleep(0.02)
            yield event(texts[-1] + "？")

    detector = FakeDetector({
        text: [candidate("live", text, is_final=False)] for text in texts
    } | {texts[-1] + "？": [candidate("live", texts[-1] + "？")]})
    client = FakeClient({text: ["两次飞跃"] for text in [*texts, texts[-1] + "？"]})
    pipeline = LivePipeline(LiveRecognizer([]), detector, client, lambda _: None, lambda *_: None)
    await pipeline.run()
    assert len(client.calls) == 1
    assert client.calls[0][0] == texts[-1] + "？"


@pytest.mark.asyncio
async def test_rejected_final_transcript_cancels_pending_candidate():
    from classbot.question_detector import QuestionDetector

    recognizer = FakeRecognizer([
        TranscriptEvent(text="认识的过程是什么", is_final=False, segment_id=7),
        TranscriptEvent(text="认识的过程是实践认识再实践再认识", is_final=True, segment_id=7),
    ])
    client = FakeClient({"认识的过程是什么": ["不应该调用"]})
    updates = []
    pipeline = LivePipeline(recognizer, QuestionDetector("马克思主义基本原理"), client,
                            lambda card: updates.append(card_snapshot(card)), lambda *_: None)
    await pipeline.run()
    assert client.calls == []
    assert updates[-1][3] is CardStatus.CANCELLED


@pytest.mark.asyncio
async def test_final_filler_does_not_restart_completed_answer():
    first = event("认识的过程是什么", is_final=False)
    final = event("认识的过程是什么呢")
    client = FakeClient({first.text: ["两次飞跃"], final.text: ["多余回答"]})
    answered = asyncio.Event()

    class StableRecognizer(FakeRecognizer):
        async def events(self):
            yield first
            await answered.wait()
            yield final

    def on_card(card):
        if card.status is CardStatus.ANSWERED:
            answered.set()

    pipeline = LivePipeline(StableRecognizer([]), FakeDetector({
        first.text: [candidate("live", first.text, is_final=False)],
        final.text: [candidate("live", final.text)],
    }), client, on_card, lambda *_: None)
    await pipeline.run()
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_minor_context_revision_on_final_does_not_restart_answer():
    first = event("认识的过程是什么", is_final=False)
    final = event("认识的过程是什么呢")
    answered = asyncio.Event()

    class ContextRecognizer(FakeRecognizer):
        async def events(self):
            yield first
            await answered.wait()
            yield final

    class ContextClient:
        def __init__(self):
            self.calls = []

        async def stream_answer(self, question, subject):
            self.calls.append((question, subject))
            yield "两次飞跃"

    client = ContextClient()
    def on_card(card):
        if card.status is CardStatus.ANSWERED:
            answered.set()

    detector = FakeDetector({
        first.text: [candidate("live", first.text, is_final=False, context="我们来探讨认识的过程")],
        final.text: [candidate("live", final.text, context="我们来探讨这个认识的过程")],
    })
    pipeline = LivePipeline(ContextRecognizer([]), detector, client, on_card, lambda *_: None)
    await pipeline.run()
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_stopping_during_stability_wait_never_calls_api():
    pending_seen = asyncio.Event()

    class WaitingRecognizer(FakeRecognizer):
        async def events(self):
            yield event("认识的过程是什么", is_final=False)
            pending_seen.set()
            await self.stopped.wait()

    client = FakeClient({"认识的过程是什么": ["多余回答"]})
    pipeline = LivePipeline(WaitingRecognizer([]), FakeDetector({
        "认识的过程是什么": [candidate("live", "认识的过程是什么", is_final=False)]
    }), client, lambda _: None, lambda *_: None)
    task = asyncio.create_task(pipeline.run())
    await pending_seen.wait()
    await pipeline.stop()
    await task
    assert client.calls == []


@pytest.mark.asyncio
async def test_rejected_final_cancels_running_answer_and_clears_wrong_text():
    from classbot.question_detector import QuestionDetector
    delta_seen = asyncio.Event()
    cancelled = asyncio.Event()
    cards = []

    class RunningRecognizer(FakeRecognizer):
        async def events(self):
            yield TranscriptEvent(text="认识的过程是什么", is_final=False, segment_id=7)
            await delta_seen.wait()
            yield TranscriptEvent(text="认识是实践的结果", is_final=True, segment_id=7)

    class RunningClient:
        async def stream_answer(self, *_):
            try:
                yield "未确认的回答"
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    pipeline = LivePipeline(RunningRecognizer([]), QuestionDetector("马克思主义基本原理"),
                            RunningClient(), lambda card: cards.append(card_snapshot(card)),
                            lambda *_: delta_seen.set())
    await pipeline.run()
    assert cancelled.is_set()
    assert cards[-1][2] == ""
    assert cards[-1][3] is CardStatus.CANCELLED


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
async def test_transcript_callback_receives_each_event_text() -> None:
    transcript_events = [
        event("请问马克思", is_final=False),
        event("请问马克思的生辰是什么", is_final=True),
    ]
    transcript_texts: list[str] = []
    recognizer = FakeRecognizer(transcript_events)

    pipeline = LivePipeline(
        recognizer,
        FakeDetector({}),
        FakeClient({}),
        lambda _card: None,
        lambda _question_id, _answer: None,
        on_transcript=transcript_texts.append,
    )

    await pipeline.run()

    assert transcript_texts == [
        "请问马克思",
        "请问马克思的生辰是什么",
    ]


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

    assert len(card_updates) == 3
    assert card_updates[0] == (
        "q1",
        question.text,
        "",
        CardStatus.DETECTED,
        None,
    )
    assert card_updates[1] == (
        "q1",
        question.text,
        "",
        CardStatus.ANSWERING,
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
async def test_candidate_context_is_sent_to_answer_model_but_not_card_text() -> None:
    transcript = event("触发回答")
    question = candidate(
        "q1",
        "唯心主义又分为两种分别是什么",
        context="上一节课学习了唯物主义和唯心主义",
    )
    prompt = (
        "背景信息：上一节课学习了唯物主义和唯心主义\n"
        "问题：唯心主义又分为两种分别是什么"
    )
    detector = FakeDetector({transcript.text: [question]})
    client = FakeClient({prompt: ["唯心主义分为主观唯心主义和客观唯心主义。"]})
    cards: list[tuple[str, str, CardStatus]] = []

    pipeline = LivePipeline(
        FakeRecognizer([transcript]),
        detector,
        client,
        lambda card: cards.append((card.question, card.answer, card.status)),
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert client.calls == [(prompt, "历史")]
    assert cards[-1][0] == question.text
    assert cards[-1][1].startswith("唯心主义分为")


@pytest.mark.asyncio
async def test_new_question_card_records_session_elapsed_timestamp() -> None:
    transcript = event("马克思出生于哪一年？")
    question = candidate("q1", transcript.text)
    recognizer = FakeRecognizer([transcript])
    detector = FakeDetector({transcript.text: [question]})
    client = FakeClient({question.text: ["1818年。"]})
    cards: list[AnswerCard] = []

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        cards.append,
        lambda _question_id, _answer: None,
    )

    await pipeline.run()

    assert cards[0].timestamp_ms >= 0
    assert cards[-1].timestamp_ms == cards[0].timestamp_ms


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
    client = FakeClient({final_question.text: ["欧洲历史上的思想文化运动"]})
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
    statuses: list[str] = []

    def on_card(card: AnswerCard) -> None:
        card_updates[card.question_id].append(card_snapshot(card))

    def on_status(status: str) -> None:
        statuses.append(status)
        if status == "api_error":
            raise RuntimeError("status callback must not escape")

    pipeline = LivePipeline(
        recognizer,
        detector,
        client,
        on_card,
        lambda _question_id, _answer: None,
        on_status,
    )

    await pipeline.run()

    assert card_updates["q1"][-1][3] is CardStatus.ERROR
    assert card_updates["q1"][-1][4] == "服务暂时不可用"
    assert card_updates["q2"][-1][2:] == ("正常答案", CardStatus.ANSWERED, None)
    assert statuses.count("api_error") == 1


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


class PreparedFakeClient(FakeClient):
    def __init__(self, prepared, streams):
        super().__init__(streams)
        self.prepared = prepared

    async def prepare_answer(self, question, subject):
        return self.prepared

    def stream_answer(self, question, subject, *, prepared=None):
        return super().stream_answer(question, subject)


@pytest.mark.asyncio
async def test_pipeline_writes_source_refs_and_removes_unknown_tags():
    ref = SourceRef("S1", SourceKind.TEXTBOOK, "教", "PDF第1页")
    prepared = PreparedAnswer([], ("教材：PDF第1页",), (ref,))
    client = PreparedFakeClient(prepared, {"问题": ["答案[S9]"]})
    pipeline = LivePipeline(
        FakeRecognizer([event("问题")]),
        FakeDetector({"问题": [candidate("q1", "问题")]}),
        client,
        lambda _: None,
        lambda *_: None,
    )
    await pipeline.run()
    card = pipeline._states["q1"].card
    assert card.source_refs == (ref,)
    assert card.answer == "答案"
    assert "不存在的来源编号" in card.notice
