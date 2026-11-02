import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from classbot.config import Settings
from classbot.domain import CardStatus, QuestionCandidate
from classbot.llm.deepseek_client import DeepSeekClient, DeepSeekStreamError
from classbot.llm.grounding import GroundingReviewError
from classbot.pipeline import LivePipeline


@pytest.fixture(autouse=True)
def _web_search_off(monkeypatch):
    """Local-discipline tests run offline; web fallback has its own tests."""
    monkeypatch.setenv("CLASSBOT_WEB_SEARCH", "off")



SOURCE = "计划选择模型 A 做校验，模型 B 生成训练数据；这不能证明 A 缺少训练能力。"
SAFE = "计划选择模型 A 做校验、模型 B 生成训练数据，未证明 A 缺少训练能力。"


def sse(text, *, reason="stop", done=True):
    result = "data: " + json.dumps({"choices": [{"delta": {"content": text}, "finish_reason": reason}]}, ensure_ascii=False) + "\n\n"
    return result + ("data: [DONE]\n" if done else "")


def claims(*, text=SAFE, unit="E1", status="partial"):
    return json.dumps({"status": status, "claims": [
        {"text": text, "evidence": [{"id": "S1", "field": "excerpt", "unit": unit}]},
    ]}, ensure_ascii=False)


async def run_pipeline(handler, *, sources=True):
    queries, deltas = [], []

    def search(query, limit=5):
        queries.append((query, limit))
        return (SimpleNamespace(source="plan.md · lines 1–2", text=SOURCE),) if sources else ()

    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu", thinking_effort="high")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成答辩", kind="defense"),
                                knowledge_index=SimpleNamespace(search=search))
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda _qid, text: deltas.append(text), scenario_name="合成答辩")
        await pipeline._handle_candidate(QuestionCandidate("q1", "为什么模型 A 不能训练？", "答辩", 1, True, "LOCAL_ONLY_8376f"))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    return pipeline._states["q1"].card, queries, deltas


@pytest.mark.asyncio
async def test_only_question_and_same_evidence_are_sent_before_local_claim_rendering():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse(claims()))

    card, queries, deltas = await run_pipeline(handler)
    assert len(requests) == len(queries) == 1
    assert "LOCAL_ONLY_8376f" in queries[0][0]
    wire = json.loads(requests[0]["messages"][1]["content"])
    assert wire == {"question": "为什么模型 A 不能训练？", "evidence": [
        {"id": "S1", "source": "plan.md · lines 1–2", "excerpt": [{"unit": "E1", "text": SOURCE}]},
    ]}
    assert "LOCAL_ONLY_8376f" not in json.dumps(requests[0]["messages"], ensure_ascii=False)
    assert requests[0]["max_tokens"] == 2500 and requests[0]["thinking"] == {"type": "disabled"}
    assert card.status is CardStatus.ANSWERED and SAFE in card.answer and "[S1]" in card.answer
    assert deltas == [card.answer] and "范围提示" in card.answer


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", [
    "plain unverified answer [S1]",
    claims(unit="E999"),
    '{"status":"supported","claims":[],"answer":"unbound opening"}',
    '{"status":"supported","status":"insufficient","claims":[]}',
])
async def test_invalid_structured_answer_never_retries_or_publishes_partial_text(invalid):
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse(invalid))

    card, _, deltas = await run_pipeline(handler)
    assert len(requests) == 1
    assert card.status is CardStatus.ERROR and card.answer == "" and card.error
    assert deltas == []


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "length", "missing_done", "empty"])
async def test_transport_or_incomplete_structured_stream_is_single_attempt_and_fail_closed(failure):
    calls = []

    def handler(request):
        calls.append(json.loads(request.content))
        if failure == "transport":
            raise httpx.ConnectError("offline", request=request)
        return httpx.Response(200, content=sse(
            "" if failure == "empty" else claims(),
            reason="length" if failure == "length" else "stop",
            done=failure != "missing_done",
        ))

    card, _, deltas = await run_pipeline(handler)
    assert len(calls) == 1 and card.status is CardStatus.ERROR
    assert card.answer == "" and deltas == []


@pytest.mark.asyncio
async def test_no_evidence_can_only_publish_explicit_insufficiency():
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse('{"status":"insufficient","claims":[]}'))

    card, queries, deltas = await run_pipeline(handler, sources=False)
    wire = json.loads(requests[0]["messages"][1]["content"])
    assert len(requests) == len(queries) == 1 and wire["evidence"] == []
    assert set(wire) == {"question", "evidence"}
    assert card.status is CardStatus.ANSWERED and card.sources == ()
    assert "依据不足" in card.answer and deltas == [card.answer]


@pytest.mark.asyncio
async def test_request_error_redacts_secret_and_does_not_retry():
    settings = Settings("private-test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content='data: {"error":{"message":"private-test-key rejected"}}\n\n')

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成", kind="defense"),
                                knowledge_index=SimpleNamespace(search=lambda *_args, **_kwargs: ()))
        prepared = await client.prepare_answer("发生了什么？", "答辩")
        with pytest.raises(DeepSeekStreamError, match="REDACTED") as caught:
            await client.grounded_answer("发生了什么？", "答辩", prepared=prepared)
        assert "private-test-key" not in str(caught.value)
        await client.aclose()
    assert len(requests) == 1


@pytest.mark.asyncio
async def test_mismatched_question_is_rejected_before_any_request():
    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: pytest.fail("must not send"))) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成", kind="defense"),
                                knowledge_index=SimpleNamespace(search=lambda *_args, **_kwargs: ()))
        prepared = await client.prepare_answer("旧问题？", "答辩")
        with pytest.raises(GroundingReviewError, match="question does not match"):
            await client.grounded_answer("新问题？", "答辩", prepared=prepared)
        await client.aclose()


@pytest.mark.asyncio
async def test_explicit_defense_thinking_is_preserved_but_incomplete_result_never_retries():
    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu",
                        thinking_effort="max", defense_thinking_effort="low")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=sse("", reason="length"))

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成", kind="defense"),
                                knowledge_index=SimpleNamespace(search=lambda *_args, **_kwargs: ()))
        prepared = await client.prepare_answer("发生了什么？", "答辩")
        with pytest.raises(DeepSeekStreamError, match="length"):
            await client.grounded_answer("发生了什么？", "答辩", prepared=prepared)
        await client.aclose()
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"type": "enabled"}
    assert requests[0]["reasoning_effort"] == "low" and requests[0]["max_tokens"] == 4096
