"""Single grounded request through the installed HTTP11 stack, entirely offline."""

import asyncio
import gc
import json
from types import AsyncGeneratorType, SimpleNamespace

import httpx
import pytest
from test_sse_ownership import SequenceBackend, _event, _finalizers, _settle, _transport

from classbot.config import Settings
from classbot.domain import CardStatus, QuestionCandidate
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline

QUESTION = "为什么模型甲不能用于正式计算？"
EXCERPT = "方案安排模型甲校验、模型乙计算；这只是用途安排，不证明模型甲能力缺失。"
FACT = "方案安排模型甲校验、模型乙计算，不能据此断言模型甲没有计算能力。"
REVIEW = json.dumps({"status": "supported", "claims": [{
    "text": FACT,
    "evidence": [{"id": "S1", "field": "excerpt", "unit": "E1"}],
}]}, ensure_ascii=False)
SAFE = "【资料事实】" + FACT + "[S1]"


def completed(text):
    return [_event(content=text, finish="stop"), b"data: [DONE]\n\n"]


class TrackedBackend(SequenceBackend):
    def __init__(self, plans):
        super().__init__(plans)
        self.closed_tasks = []
        self.reading_tail = asyncio.Event()
        self.connect_attempts = 0

    async def connect_tcp(self, *args, **kwargs):
        self.connect_attempts += 1
        stage = len(self.streams)
        stream = await super().connect_tcp(*args, **kwargs)
        read, close = stream.read, stream.aclose
        ending = self.plans[stage][1]

        async def tracked_read(*read_args, **read_kwargs):
            if not stream._buffer and ending == "stall":
                self.reading_tail.set()
                await asyncio.Event().wait()
            return await read(*read_args, **read_kwargs)

        async def tracked_close():
            await close()
            self.closed_tasks.append((stage, asyncio.current_task()))

        stream.read, stream.aclose = tracked_read, tracked_close
        return stream


def active_http_generators():
    return {
        id(value) for value in gc.get_objects()
        if isinstance(value, AsyncGeneratorType) and value.ag_frame is not None
        and value.ag_frame.f_globals.get("__name__", "").startswith((
            "httpx.", "httpcore.", "classbot.llm.",
        ))
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("case", [
    "success", "terminal_stall", "terminal_transport",
    "review_cancel", "review_length", "review_missing_done", "review_transport",
    "review_invalid_json", "review_unknown_unit", "review_quote_field", "explicit_high",
    "high_reasoning_length", "review_transport_empty", "long_units", "section_unit",
    "review_wrong_field_unit",
])
async def test_one_grounded_request_releases_http11_in_same_task_without_gc(monkeypatch, case):
    from classbot.llm import sse

    monkeypatch.setattr(sse, "_TAIL_TIMEOUT_SECONDS", .02)
    excerpt = (EXCERPT + "\t α\n") * 15 if case == "long_units" else EXCERPT
    section_context = "## 模型用途\n" + EXCERPT if case == "section_unit" else ""
    raw = json.loads(REVIEW)
    if case == "long_units":
        raw["claims"][0]["evidence"][0]["unit"] = "E2"
    elif case == "section_unit":
        raw["claims"][0]["evidence"][0].update(field="section_context", unit="C1")
    review = completed(json.dumps(raw, ensure_ascii=False))
    review_ending = "eof"
    if case == "review_length":
        review = [_event(content=REVIEW[:60], finish="length"), b"data: [DONE]\n\n"]
    elif case == "high_reasoning_length":
        review = [_event(finish="length"), b"data: [DONE]\n\n"]
    elif case == "review_missing_done":
        review = [_event(content=REVIEW, finish="stop")]
    elif case == "review_invalid_json":
        review = completed('{"status":"supported","claims":[')
    elif case in {"review_unknown_unit", "review_quote_field", "review_wrong_field_unit"}:
        bad = json.loads(REVIEW)
        if case == "review_quote_field":
            bad["claims"][0]["evidence"][0]["quote"] = EXCERPT
        else:
            bad["claims"][0]["evidence"][0]["unit"] = "C1" if case == "review_wrong_field_unit" else "E999"
        review = completed(json.dumps(bad, ensure_ascii=False))
    elif case == "review_transport":
        review, review_ending = [_event(content=REVIEW[:60])], "transport"
    elif case == "review_transport_empty":
        review, review_ending = [], "transport"
    elif case == "review_cancel":
        review, review_ending = [_event(content=REVIEW[:60])], "stall"
    elif case.startswith("terminal_"):
        review_ending = "stall" if case == "terminal_stall" else "transport"
    backend = TrackedBackend([(review, review_ending)])
    high = case in {"explicit_high", "high_reasoning_length"}
    settings = Settings("synthetic-key", "答辩", "deepseek-flash", "http://synthetic.invalid", "cpu",
                        thinking_effort="high", defense_thinking_effort="high" if high else "none")
    queries, deltas, cards, results = [], [], [], []

    async def search(query, limit=5):
        queries.append((query, limit))
        fields = {"section_context": section_context, "section_source": "synthetic.md · lines 1–1"} if section_context else {}
        return (SimpleNamespace(source="synthetic.md · lines 1–2", text=excerpt, **fields),)

    baseline = active_http_generators()
    with _finalizers() as finalized:
        async with httpx.AsyncClient(transport=await _transport(backend), base_url=settings.base_url) as http:
            client = DeepSeekClient(settings, http_client=http,
                                    scenario=SimpleNamespace(name="synthetic", kind="competition"),
                                    knowledge_index=SimpleNamespace(search_async=search))
            grounded_answer = client.grounded_answer

            async def observed_grounded(*args, **kwargs):
                result = await grounded_answer(*args, **kwargs)
                results.append(result)
                return result

            client.grounded_answer = observed_grounded
            pipeline = LivePipeline(None, None, client,
                                    lambda card: cards.append((card.answer, card.status)),
                                    lambda _, answer: deltas.append(answer), scenario_name="synthetic")
            await pipeline._handle_candidate(QuestionCandidate("q1", QUESTION, "答辩", 1., True))
            task = pipeline._states["q1"].task
            if case == "review_cancel":
                await asyncio.wait_for(backend.reading_tail.wait(), 2)
                assert deltas == [] and pipeline._states["q1"].card.answer == ""
                task.cancel()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 2)
            await client.aclose()
        await _settle()
        assert not active_http_generators() - baseline
    assert not finalized
    assert all(stream._closed for stream in backend.streams)
    assert backend.closed_tasks and all(owner is task for _, owner in backend.closed_tasks)
    assert queries == [(QUESTION, 5)]
    card = pipeline._states["q1"].card
    assert backend.connect_attempts == len(backend.requests) == len(backend.streams) == 1
    payload = backend.requests[0]
    review_input = json.loads(payload["messages"][1]["content"])
    assert set(review_input) == {"question", "evidence"}
    assert review_input["question"] == QUESTION
    assert len(review_input["evidence"]) == 1
    item = review_input["evidence"][0]
    assert item["id"] == "S1" and item["source"] == "synthetic.md · lines 1–2"
    for field, original, prefix in [("excerpt", excerpt, "E"), ("section_context", section_context, "C")]:
        if not original:
            assert field not in item
            continue
        units = item[field]
        assert isinstance(units, list)
        assert all(set(unit) == {"unit", "text"} and 0 < len(unit["text"]) <= 500 for unit in units)
        assert [unit["unit"] for unit in units] == [f"{prefix}{i}" for i in range(1, len(units) + 1)]
        assert "".join(unit["text"] for unit in units) == original
    if section_context:
        assert item["section_source"] == "synthetic.md · lines 1–1"
    assert payload["thinking"] == {"type": "enabled" if high else "disabled"}
    assert payload["max_tokens"] == (4096 if high else 2500)
    if high:
        assert payload["reasoning_effort"] == "high"
    else:
        assert "reasoning_effort" not in payload
    if case in {"success", "terminal_stall", "terminal_transport", "explicit_high", "long_units", "section_unit"}:
        assert card.status is CardStatus.ANSWERED and card.error is None
        assert card.answer == SAFE and deltas == [SAFE]
        assert all(answer in ("", SAFE) for answer, _ in cards)
        assert len(results) == 1
        reference = results[0].claims[0]["evidence"][0]
        assert set(reference) == {"id", "field", "unit", "quote"}
        assert reference["id"] == "S1"
        bound = next(unit["text"] for unit in item[reference["field"]]
                     if unit["unit"] == reference["unit"])
        assert reference["quote"] == bound
    else:
        assert results == []
        assert card.answer == "" and deltas == []
        assert all(answer == "" for answer, _ in cards)
        if case == "review_cancel":
            assert task.cancelled() and card.status is CardStatus.CANCELLED
        else:
            assert card.status is CardStatus.ERROR and card.error
