"""Defense publishes only locally validated facts from one grounded request."""

import asyncio
from types import SimpleNamespace

import pytest

from classbot.domain import CardStatus, QuestionCandidate
from classbot.pipeline import LivePipeline

QUESTION = "为什么正式计算不能使用参考模型"
SUBJECT = "synthetic project"
CLASSROOM_PARTS = ("课堂第一段", "课堂第二段")
SAFE_ANSWER = "候选仅规定用途，尚不足以证明能力缺失。[S1]"


class GroundedClient:
    def __init__(self, *, result=None, error=None):
        self.prepared = SimpleNamespace(messages=[{"role": "user", "content": QUESTION}],
                                        sources=("[S1] source.md · lines 1–2",))
        self.result = result or SimpleNamespace(text=SAFE_ANSWER, status="supported", claims=())
        self.error = error
        self.prepare_calls, self.stream_calls, self.grounded_calls = [], [], []
        self.grounded_started, self.grounded_release = asyncio.Event(), asyncio.Event()
        self.grounded_cancelled = asyncio.Event()
        self.grounded_owners = []
        self.stream_paused, self.stream_release = asyncio.Event(), asyncio.Event()
        self.after_prepare = None

    async def prepare_answer(self, question, subject):
        self.prepare_calls.append((question, subject, None))
        if self.after_prepare:
            self.after_prepare()
        return self.prepared

    async def prepare_answer_with_local_context(self, question, subject, *, local_context):
        self.prepare_calls.append((question, subject, local_context))
        if self.after_prepare:
            self.after_prepare()
        return self.prepared

    async def stream_answer(self, question, subject, *, prepared=None):
        self.stream_calls.append((question, subject, prepared))
        yield CLASSROOM_PARTS[0]
        self.stream_paused.set()
        await self.stream_release.wait()
        yield CLASSROOM_PARTS[1]

    async def grounded_answer(self, question, subject, *, prepared):
        self.grounded_calls.append((question, subject, prepared))
        self.grounded_owners.append(asyncio.current_task())
        self.grounded_started.set()
        try:
            await self.grounded_release.wait()
            if self.error is not None:
                raise self.error
            return self.result
        except asyncio.CancelledError:
            self.grounded_cancelled.set()
            raise


@pytest.fixture
async def start_pipeline():
    tasks = []

    async def start(client, *, defense=True):
        cards, deltas, statuses = [], [], []
        pipeline = LivePipeline(None, None, client,
                                lambda card: cards.append((card.answer, card.status, card.error)),
                                lambda question_id, answer: deltas.append((question_id, answer)),
                                lambda status: statuses.append(status),
                                scenario_name="synthetic defense" if defense else "")
        await pipeline._handle_candidate(QuestionCandidate(
            "q1", QUESTION, SUBJECT, 1., True,
            context="PRIVATE_MEETING_CONTEXT" if defense else "",
        ))
        task = pipeline._states["q1"].task
        tasks.append(task)
        return SimpleNamespace(pipeline=pipeline, task=task, cards=cards, deltas=deltas,
                               statuses=statuses, card=pipeline._states["q1"].card)

    yield start
    for task in tasks:
        if not task.done():
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def assert_no_answer_published(run):
    assert run.deltas == [] and run.card.answer == ""
    assert all(answer == "" for answer, _, _ in run.cards)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,text", [
    ("supported", SAFE_ANSWER),
    ("insufficient", "当前资料候选不足以支持该前提或结论。"),
])
async def test_defense_awaits_one_grounded_request_without_generating_a_freeform_answer(start_pipeline, status, text):
    client = GroundedClient(result=SimpleNamespace(text=text, status=status, claims=()))
    run = await start_pipeline(client)
    await asyncio.wait_for(client.grounded_started.wait(), 2)

    assert_no_answer_published(run)
    assert client.stream_calls == []
    assert client.prepare_calls == [(QUESTION, SUBJECT, "PRIVATE_MEETING_CONTEXT")]
    assert client.grounded_calls == [(QUESTION, SUBJECT, client.prepared)]
    assert client.grounded_owners == [run.task]
    client.grounded_release.set()
    await asyncio.wait_for(run.task, 2)

    assert run.deltas == [("q1", text)] and run.card.answer == text
    assert run.card.status is CardStatus.ANSWERED and run.card.error is None
    assert all(answer in ("", text) for answer, _, _ in run.cards)
    assert sum(status is CardStatus.ANSWERED for _, status, _ in run.cards) == 1
    assert len(client.grounded_calls) == 1 and client.stream_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [ValueError("invalid JSON"), ValueError("unknown evidence unit")])
async def test_grounding_format_or_unit_failure_sets_error_without_any_publication(start_pipeline, error):
    client = GroundedClient(error=error)
    run = await start_pipeline(client)
    client.grounded_release.set()
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is CardStatus.ERROR and run.card.error == str(error)
    assert run.statuses == ["api_error"]
    assert len(client.grounded_calls) == 1 and client.stream_calls == []


@pytest.mark.asyncio
async def test_cancel_pending_grounded_request_publishes_nothing_and_never_reopens(start_pipeline):
    client = GroundedClient()
    run = await start_pipeline(client)
    await asyncio.wait_for(client.grounded_started.wait(), 2)
    assert_no_answer_published(run)
    run.task.cancel()
    await asyncio.gather(run.task, return_exceptions=True)
    client.grounded_release.set()
    await asyncio.sleep(0)

    assert_no_answer_published(run)
    assert run.task.cancelled() and run.card.status is CardStatus.CANCELLED
    assert client.grounded_cancelled.is_set()
    assert len(client.grounded_calls) == 1 and client.stream_calls == []


@pytest.mark.asyncio
async def test_old_revision_before_grounding_never_starts_request(start_pipeline):
    client = GroundedClient()
    run = await start_pipeline(client)
    client.after_prepare = lambda: setattr(run.pipeline._states["q1"], "revision", 1)
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is not CardStatus.ANSWERED
    assert client.grounded_calls == client.stream_calls == []


@pytest.mark.asyncio
async def test_revision_obsoleted_during_grounding_discards_validated_result(start_pipeline):
    client = GroundedClient()
    run = await start_pipeline(client)
    await asyncio.wait_for(client.grounded_started.wait(), 2)
    run.pipeline._states["q1"].revision += 1
    client.grounded_release.set()
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is not CardStatus.ANSWERED
    assert len(client.grounded_calls) == 1 and client.stream_calls == []


@pytest.mark.asyncio
async def test_classroom_keeps_live_deltas_even_when_client_has_grounded_method(start_pipeline):
    client = GroundedClient()
    run = await start_pipeline(client, defense=False)
    await asyncio.wait_for(client.stream_paused.wait(), 2)
    assert run.deltas == [("q1", CLASSROOM_PARTS[0])]
    assert client.grounded_calls == []
    client.stream_release.set()
    await asyncio.wait_for(run.task, 2)

    assert run.deltas == [("q1", CLASSROOM_PARTS[0]), ("q1", "".join(CLASSROOM_PARTS))]
    assert run.card.status is CardStatus.ANSWERED
    assert client.prepare_calls == [(QUESTION, SUBJECT, None)]
    assert client.grounded_calls == []


@pytest.mark.asyncio
async def test_empty_grounded_text_fails_closed(start_pipeline):
    client = GroundedClient(result=SimpleNamespace(text=" \t\n", status="supported", claims=()))
    run = await start_pipeline(client)
    client.grounded_release.set()
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is CardStatus.ERROR and run.card.error
    assert len(client.grounded_calls) == 1 and client.stream_calls == []


@pytest.mark.asyncio
async def test_missing_prepared_evidence_cannot_bypass_grounding(start_pipeline):
    client = GroundedClient()
    client.prepared = None
    run = await start_pipeline(client)
    client.grounded_release.set()
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is CardStatus.ERROR and run.card.error
    assert client.grounded_calls == client.stream_calls == []


class LegacyClient:
    def __init__(self, adapter):
        self.stream_calls = []
        if adapter != "missing":
            self.grounded_answer = None if adapter == "none" else "not callable"

    async def prepare_answer(self, question, subject):
        return SimpleNamespace(sources=())

    async def stream_answer(self, question, subject, *, prepared=None):
        self.stream_calls.append((question, subject))
        yield "UNVALIDATED_LEGACY_BODY"


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["missing", "none", "noncallable"])
async def test_defense_adapter_without_callable_grounding_fails_closed(start_pipeline, adapter):
    client = LegacyClient(adapter)
    run = await start_pipeline(client)
    await asyncio.wait_for(run.task, 2)

    assert_no_answer_published(run)
    assert run.card.status is CardStatus.ERROR and run.card.error
    assert run.statuses == ["api_error"]
    assert client.stream_calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("adapter", ["missing", "none", "noncallable"])
async def test_classroom_legacy_client_still_streams_without_grounding_adapter(start_pipeline, adapter):
    client = LegacyClient(adapter)
    run = await start_pipeline(client, defense=False)
    await asyncio.wait_for(run.task, 2)

    assert run.card.status is CardStatus.ANSWERED and run.card.error is None
    assert run.card.answer == "UNVALIDATED_LEGACY_BODY"
    assert run.deltas == [("q1", "UNVALIDATED_LEGACY_BODY")]
    assert client.stream_calls == [(QUESTION, SUBJECT)]
