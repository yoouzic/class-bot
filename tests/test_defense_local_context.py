import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from classbot.config import Settings
from classbot.domain import CardStatus, QuestionCandidate
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline


def review_payload(request):
    content = json.loads(request.content)["messages"][1]["content"]
    if not content.lstrip().startswith("{"):
        return None
    payload = json.loads(content)
    assert set(payload) == {"question", "evidence"}
    return payload


def review_response(*claims, status="supported"):
    return httpx.Response(200, content=sse_content([
        json.dumps({"status": status, "claims": list(claims)}, ensure_ascii=False),
    ]))


def bound_claim(text, source_id, unit="E1"):
    return {"text": text, "evidence": [{"id": source_id, "field": "excerpt", "unit": unit}]}


def original_text(item, field="excerpt"):
    units = item[field]
    assert isinstance(units, list) and all(set(unit) == {"unit", "text"} for unit in units)
    assert [unit["unit"] for unit in units] == [f"{'E' if field == 'excerpt' else 'C'}{i}"
                                               for i in range(1, len(units) + 1)]
    assert all(0 < len(unit["text"]) <= 500 for unit in units)
    return "".join(unit["text"] for unit in units)


@pytest.mark.asyncio
async def test_defense_context_reaches_local_search_but_never_the_remote_request():
    sentinel = "LOCAL_ONLY_CONTEXT_8d534196"
    question = "为什么采用这个协议？"
    queries = []
    requests = []

    def search(query, limit=5):
        queries.append(query)
        return (SimpleNamespace(source="方案.md · lines 1–2", text="方案规定采用 BBM92。"),)

    def handler(request):
        requests.append(json.loads(request.content))
        review = review_payload(request)
        assert review["question"] == question
        assert review["evidence"] == [{"id": "S1", "source": "方案.md · lines 1–2", "excerpt": [{"unit": "E1", "text": "方案规定采用 BBM92。"}]}]
        return review_response(bound_claim("方案规定采用 BBM92。", "S1"))

    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(
            settings, http_client=http, scenario=SimpleNamespace(name="竞赛", kind="competition"),
            knowledge_index=SimpleNamespace(search=search),
        )
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda *_: None, scenario_name="竞赛")
        await pipeline._handle_candidate(QuestionCandidate("q1", question, "答辩", 1, True, sentinel))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    assert len(queries) == 1 and sentinel in queries[0] and question in queries[0]
    assert len(requests) == 1
    serialized_messages = json.dumps([r["messages"] for r in requests], ensure_ascii=False)
    assert sentinel not in serialized_messages
    assert question in serialized_messages and "方案规定采用 BBM92" in serialized_messages
    card = pipeline._states["q1"].card
    assert card.status is CardStatus.ANSWERED
    assert card.answer == "【资料事实】方案规定采用 BBM92。[S1]"
    assert card.sources == ("[S1] 方案.md · lines 1–2",)


@pytest.mark.asyncio
async def test_defense_legacy_client_fails_closed_without_streaming_or_publishing():
    calls, deltas = [], []

    class Client:
        async def prepare_answer(self, question, subject):
            calls.append(("prepare", question))
            return SimpleNamespace(sources=())

        async def stream_answer(self, question, subject, *, prepared=None):
            calls.append(("stream", question))
            yield "当前资料不足。"

    pipeline = LivePipeline(None, None, Client(), lambda *_: None, lambda _, answer: deltas.append(answer), scenario_name="竞赛")
    await pipeline._handle_candidate(QuestionCandidate("q1", "为什么这样选择？", "答辩", 1, True, "本地背景"))
    await asyncio.gather(*pipeline._answer_tasks)
    assert calls == [("prepare", "为什么这样选择？")]
    card = pipeline._states["q1"].card
    assert card.status is CardStatus.ERROR and card.error
    assert card.answer == "" and deltas == []


@pytest.mark.asyncio
async def test_classroom_context_still_reaches_the_existing_answer_interface():
    calls = []

    class Client:
        async def prepare_answer(self, question, subject):
            calls.append(("prepare", question))
            return SimpleNamespace(sources=())

        async def prepare_answer_with_local_context(self, *_args, **_kwargs):
            pytest.fail("classroom must keep its existing answer interface")

        async def stream_answer(self, question, subject, *, prepared=None):
            calls.append(("stream", question))
            yield "课堂答案"

    pipeline = LivePipeline(None, None, Client(), lambda *_: None, lambda *_: None)
    await pipeline._handle_candidate(QuestionCandidate("q1", "它为什么成立？", "课堂", 1, True, "课堂背景"))
    await asyncio.gather(*pipeline._answer_tasks)
    expected = "背景信息：课堂背景\n问题：它为什么成立？"
    assert calls == [("prepare", expected), ("stream", expected)]


@pytest.mark.asyncio
async def test_question_revision_cancels_old_retrieval_and_uses_new_local_context():
    ready = asyncio.Event()
    calls = []

    class Client:
        async def prepare_answer(self, *_args):
            pytest.fail("defense should use the explicit local-context interface")

        async def prepare_answer_with_local_context(self, question, subject, *, local_context):
            calls.append((question, local_context))
            if len(calls) == 1:
                ready.set()
                await asyncio.Event().wait()
            return SimpleNamespace(sources=("[S1] 新资料",))

        async def stream_answer(self, question, subject, *, prepared=None):
            pytest.fail("defense must never bypass grounded_answer")
            yield "unreachable"

        async def grounded_answer(self, question, subject, *, prepared):
            assert question == "为什么选择乙协议？" and prepared.sources == ("[S1] 新资料",)
            return SimpleNamespace(text="新答案", status="supported", claims=())

    pipeline = LivePipeline(None, None, Client(), lambda *_: None, lambda *_: None, scenario_name="竞赛")
    await pipeline._handle_candidate(QuestionCandidate("q1", "为什么选择甲协议？", "答辩", 1, True, "旧背景"))
    await asyncio.wait_for(ready.wait(), 1)
    await pipeline._handle_candidate(QuestionCandidate("q1", "为什么选择乙协议？", "答辩", 1, True, "新背景"))
    await asyncio.gather(*pipeline._answer_tasks, return_exceptions=True)
    assert calls == [("为什么选择甲协议？", "旧背景"), ("为什么选择乙协议？", "新背景")]
    card = pipeline._states["q1"].card
    assert card.question == "为什么选择乙协议？"
    assert card.answer == "新答案" and card.sources == ("[S1] 新资料",)
    assert card.status is CardStatus.ANSWERED


def sse_content(parts):
    return "".join(
        "data: " + json.dumps({"choices": [{"delta": {"content": part}}]}, ensure_ascii=False) + "\n\n"
        for part in parts
    ) + "data: [DONE]\n"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "parts,expected_status,unknown",
    [
        (["【资料事实】已完成实验。[S", "99", "]"], CardStatus.ERROR, "[S99]"),
        (["【资料事实】这是计划要求。[S", "1", "]"], CardStatus.ANSWERED, None),
        (["当前资料不足，无法确认是否完成。"], CardStatus.ANSWERED, None),
    ],
)
async def test_defense_split_grounded_json_ids_are_checked_before_any_publication(parts, expected_status, unknown):
    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    # A citation-looking filename is untrusted data, not an additional allowed ID.
    index = SimpleNamespace(search=lambda *_args, **_kwargs: (
        SimpleNamespace(source="任务书[S99].md · lines 1–2", text="计划要求完成实验。"),
    ))
    requests, deltas = [], []

    def handler(request):
        requests.append(request)
        review = review_payload(request)
        assert original_text(review["evidence"][0]) == "计划要求完成实验。"
        insufficient = "资料不足" in "".join(parts)
        claims = [] if insufficient else [bound_claim("这是计划要求。", "S99" if unknown else "S1")]
        raw = json.dumps({"status": "insufficient" if insufficient else "supported", "claims": claims}, ensure_ascii=False)
        return httpx.Response(200, content=sse_content([raw[:19], raw[19:53], raw[53:]]))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.base_url,
    ) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="竞赛", kind="competition"), knowledge_index=index)
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda _, answer: deltas.append(answer), scenario_name="竞赛")
        await pipeline._handle_candidate(QuestionCandidate("q1", "你们完成了哪些实验？", "答辩", 1, True))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    card = pipeline._states["q1"].card
    assert card.status is expected_status
    if unknown:
        assert "编号不存在" in card.error
        assert card.answer == "" and deltas == [] and len(requests) == 1
    else:
        assert card.error is None
        assert deltas == [card.answer] and len(requests) == 1
        if "资料不足" in "".join(parts):
            assert "【依据不足】" in card.answer and "【补证建议】" in card.answer
            assert "【资料事实】" not in card.answer
        else:
            assert card.answer == "【资料事实】这是计划要求。[S1]"


@pytest.mark.asyncio
async def test_classroom_stream_does_not_apply_defense_source_number_validation():
    settings = Settings("test-key", "课堂", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _request: httpx.Response(200, content=sse_content(["教材符号 [S99]"]))),
        base_url=settings.base_url,
    ) as http:
        client = DeepSeekClient(settings, http_client=http)
        result = [part async for part in client.stream_answer("这个符号是什么意思？", "课堂")]
        await client.aclose()
    assert result == ["教材符号 [S99]"]


@pytest.mark.asyncio
async def test_revision_cancels_partial_unknown_citation_without_poisoning_new_answer():
    first_response_bytes = asyncio.Event()
    old_stream_closed = asyncio.Event()
    requests, deltas = [], []

    class PausedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raw = json.dumps({"status": "supported", "claims": [bound_claim("旧事实。", "S99")]}, ensure_ascii=False)
            split = raw.index("S99") + 1
            yield sse_content([raw[:split]]).removesuffix("data: [DONE]\n").encode()
            first_response_bytes.set()
            await asyncio.Event().wait()
            yield sse_content([raw[split:]]).encode()

        async def aclose(self):
            old_stream_closed.set()

    def handler(request):
        requests.append(request)
        review = review_payload(request)
        assert original_text(review["evidence"][0]) == "采用乙协议。"
        if "甲协议" in review["question"]:
            return httpx.Response(200, stream=PausedStream())
        assert review["question"] == "为什么选择乙协议？"
        return review_response(bound_claim("方案约定采用乙协议。", "S1"))

    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    index = SimpleNamespace(search=lambda *_args, **_kwargs: (SimpleNamespace(source="方案.md · lines 1–2", text="采用乙协议。"),))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="竞赛", kind="competition"), knowledge_index=index)
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda _, answer: deltas.append(answer), scenario_name="竞赛")
        await pipeline._handle_candidate(QuestionCandidate("q1", "为什么选择甲协议？", "答辩", 1, True))
        await asyncio.wait_for(first_response_bytes.wait(), 2)
        assert pipeline._states["q1"].card.answer == "" and deltas == []
        await pipeline._handle_candidate(QuestionCandidate("q1", "为什么选择乙协议？", "答辩", 1, True))
        await asyncio.gather(*pipeline._answer_tasks, return_exceptions=True)
        await client.aclose()
    card = pipeline._states["q1"].card
    assert old_stream_closed.is_set()
    assert card.answer == "【资料事实】方案约定采用乙协议。[S1]" and card.status is CardStatus.ANSWERED
    assert deltas == [card.answer] and len(requests) == 2
    assert card.error is None


@pytest.mark.asyncio
async def test_defense_prompt_does_not_treat_plans_as_completed_work():
    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _request: pytest.fail("no request expected"))) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="竞赛", kind="competition"),
                                knowledge_index=SimpleNamespace(search=lambda *_args, **_kwargs: ()))
        prepared = await client.prepare_answer("已经完成什么？", "答辩")
        await client.aclose()
    prompt = prepared.messages[0]["content"]
    assert "任务书" in prompt and "计划" in prompt and "已完成" in prompt
    assert "不能据此" in prompt


@pytest.mark.parametrize(
    "text,expected",
    [
        ("依据 [S1,S99]", {"S1", "S99"}),
        ("依据 [S1, S99]", {"S1", "S99"}),
        ("依据 [S1、S99]", {"S1", "S99"}),
        ("依据 [S1；S99]", {"S1", "S99"}),
        ("依据 [S1 / S99]", {"S1", "S99"}),
        ("依据 [S1 S99]", {"S1", "S99"}),
        ("依据 [S1,\nS99]", {"S1", "S99"}),
        ("依据 [S99 ]", {"S99"}),
        ("依据 [S1, 99]", {"S1", "S99"}),
        ("依据 [Ｓ１，Ｓ９９]", {"S1", "S99"}),
        ("依据 [S1-S99]", {"S1", "S99"}),
        ("当前资料不足，无法确认。", set()),
    ],
)
def test_shared_source_id_extraction_covers_grouped_and_spaced_citations(text, expected):
    from classbot.llm.citations import allowed_source_ids, cited_source_ids

    assert cited_source_ids(text) == expected
    assert allowed_source_ids(("[S1] 任务书[S99].md · lines 1–2", "普通文件[S88].md")) == {"S1"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "source_id,expected_status",
    [
        ("S99", CardStatus.ERROR),
        ("S1,S99", CardStatus.ERROR),
        ("S1、S2", CardStatus.ERROR),
        ("[S99]", CardStatus.ERROR),
        ("S2", CardStatus.ANSWERED),
    ],
)
async def test_defense_unknown_bound_ids_cannot_bypass_runtime_validation(source_id, expected_status):
    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    index = SimpleNamespace(search=lambda *_args, **_kwargs: (
        SimpleNamespace(source="方案[S99].md · lines 1–2", text="这是计划要求。"),
        SimpleNamespace(source="规范.md · lines 3–4", text="这是接口约定。"),
    ))
    requests, deltas = [], []

    def handler(request):
        requests.append(request)
        review = review_payload(request)
        assert [original_text(item) for item in review["evidence"]] == ["这是计划要求。", "这是接口约定。"]
        return review_response(
            bound_claim("这是计划要求。", "S1"),
            bound_claim("这是接口约定。", source_id),
        )

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url=settings.base_url,
    ) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="竞赛", kind="competition"), knowledge_index=index)
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda _, answer: deltas.append(answer), scenario_name="竞赛")
        await pipeline._handle_candidate(QuestionCandidate("q1", "接口有什么约定？", "答辩", 1, True))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    card = pipeline._states["q1"].card
    assert card.status is expected_status
    if expected_status is CardStatus.ERROR:
        assert "编号不存在" in card.error
        assert card.answer == "" and deltas == [] and len(requests) == 1
    else:
        assert card.error is None
        assert card.answer == "【资料事实】这是计划要求。[S1]\n【资料事实】这是接口约定。[S2]"
        assert deltas == [card.answer] and len(requests) == 1
