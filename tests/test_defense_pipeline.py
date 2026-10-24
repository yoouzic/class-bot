import asyncio
from types import SimpleNamespace

import pytest

from classbot.domain import CardStatus, QuestionCandidate
from classbot.pipeline import LivePipeline


@pytest.mark.asyncio
async def test_sources_and_scenario_are_bound_before_grounded_publication():
    cards, deltas, grounded_calls = [], [], []

    class Client:
        async def prepare_answer(self, question, subject):
            return SimpleNamespace(sources=("[S1] solver.py:L1-L4",))

        async def stream_answer(self, question, subject, *, prepared=None):
            pytest.fail("defense must never fall back to the unvalidated stream")
            yield "unreachable"

        async def grounded_answer(self, question, subject, *, prepared):
            assert prepared.sources
            grounded_calls.append((question, subject, prepared.sources))
            assert cards[-1] == ("竞赛A", prepared.sources)
            assert deltas == []
            return SimpleNamespace(text="依据 [S1]", status="supported", claims=())

    pipeline = LivePipeline(
        None,
        None,
        Client(),
        lambda card: cards.append((card.scenario_name, card.sources)),
        lambda _, answer: deltas.append(answer),
        scenario_name="竞赛A",
    )
    await pipeline._handle_candidate(
        QuestionCandidate("q1", "为何选择该模型？", "竞赛", 1, True)
    )
    await asyncio.gather(*pipeline._answer_tasks)
    assert cards[0][0] == "竞赛A"
    assert cards[-1] == ("竞赛A", ("[S1] solver.py:L1-L4",))
    assert grounded_calls == [("为何选择该模型？", "竞赛", ("[S1] solver.py:L1-L4",))]
    assert deltas == ["依据 [S1]"]
    card = pipeline._states["q1"].card
    assert card.status is CardStatus.ANSWERED and card.error is None


@pytest.mark.asyncio
async def test_cancelled_retrieval_does_not_attach_sources_to_revised_question():
    ready = asyncio.Event()

    class Client:
        async def prepare_answer(self, question, subject):
            ready.set()
            await asyncio.Event().wait()

        async def stream_answer(self, *args, **kwargs):
            yield "不应出现"

    pipeline = LivePipeline(
        None, None, Client(), lambda *_args: None, lambda *_args: None
    )
    await pipeline._handle_candidate(
        QuestionCandidate("q1", "为何选择该模型？", "竞赛", 1, True)
    )
    await asyncio.wait_for(ready.wait(), 1)
    await pipeline._invalidate_question("q1")
    assert pipeline._states["q1"].card.answer == ""
    assert pipeline._states["q1"].card.sources == ()


@pytest.mark.asyncio
async def test_answer_work_is_bounded_and_cancelled_queue_never_reaches_client():
    release = asyncio.Event()
    started = []

    class Client:
        async def stream_answer(self, question, subject):
            started.append(question)
            await release.wait()
            yield "回答"

    pipeline = LivePipeline(None, None, Client(), lambda *_: None, lambda *_: None)
    for i in range(8):
        await pipeline._handle_candidate(QuestionCandidate(f"q{i}", f"为什么选择模型{i}？", "竞赛", 1, True))
    await asyncio.sleep(0)
    assert len(started) == 2
    await pipeline._invalidate_question("q2")
    release.set()
    await asyncio.gather(*pipeline._answer_tasks)
    assert all("模型2" not in question for question in started)
