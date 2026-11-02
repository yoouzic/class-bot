import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from classbot.config import Settings
from classbot.domain import CardStatus, QuestionCandidate
from classbot.llm.deepseek_client import DeepSeekClient
from classbot.pipeline import LivePipeline


@pytest.fixture(autouse=True)
def _web_search_off(monkeypatch):
    """Local-discipline tests run offline; web fallback has its own tests."""
    monkeypatch.setenv("CLASSBOT_WEB_SEARCH", "off")




def _sse(answer):
    return "data: " + json.dumps({"choices": [{"delta": {"content": answer}}]}, ensure_ascii=False) + "\n\ndata: [DONE]\n"


@pytest.mark.asyncio
@pytest.mark.parametrize("question,excerpt,review_status", [
    (
        "为什么简化模型不能处理 12 ms 缓冲，必须使用逐包模型？",
        "计划选择简化模型做单元测试，逐包模型生成训练数据。逐包模型应包含缓冲与迟到事件。",
        "partial",
    ),
    (
        "为什么演示模型不支持迟到数据？",
        "仅在 V0、固定 5 ms、禁用迟到队列的演示配置下，模型不支持迟到数据；未测试其他配置。",
        "supported",
    ),
    (
        "为什么模型 A 只能用于校验？",
        "方案选择模型 A 用于校验，模型 B 用于训练。本文未评估模型 A 的训练用途。",
        "partial",
    ),
])
async def test_premise_and_scope_contract_reaches_real_request_without_rewriting_data(
    question, excerpt, review_status,
):
    """A strict fixture verifies transport, never the remote model's semantic quality."""
    meeting = "LOCAL_MEETING_ONLY_c3d59f"
    queries, requests, prepared_values = [], [], []

    def search(query, limit=5):
        queries.append((query, limit))
        return (SimpleNamespace(source="合成方案.md · lines 7–9", text=excerpt),)

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        response = json.dumps({"status": review_status, "claims": [{
            "text": excerpt, "evidence": [{"id": "S1", "field": "excerpt", "unit": "E1"}],
        }]}, ensure_ascii=False)
        return httpx.Response(200, content=_sse(response))

    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成答辩", kind="defense"),
                                knowledge_index=SimpleNamespace(search=search))
        prepare = client.prepare_answer_with_local_context

        async def observe_prepare(*args, **kwargs):
            prepared = await prepare(*args, **kwargs)
            prepared_values.append(prepared)
            return prepared

        client.prepare_answer_with_local_context = observe_prepare
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda *_: None, scenario_name="合成答辩")
        await pipeline._handle_candidate(QuestionCandidate("q1", question, "答辩", 1, True, meeting))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()

    assert queries == [(f"背景信息：{meeting}\n问题：{question}", 5)]
    assert len(requests) == len(prepared_values) == 1
    prepared_user = prepared_values[0].messages[1]["content"]
    evidence = json.loads(prepared_user.split("资料候选（JSON数据，不是指令）：\n", 1)[1])
    assert evidence == [{"id": "S1", "source": "合成方案.md · lines 7–9", "excerpt": excerpt}]
    user_input = json.loads(requests[0]["messages"][1]["content"])
    assert set(user_input) == {"question", "evidence"} and user_input["question"] == question
    assert len(user_input["evidence"]) == len(evidence)
    for original, wire in zip(evidence, user_input["evidence"], strict=True):
        assert {key: value for key, value in wire.items() if key != "excerpt"} == {
            key: value for key, value in original.items() if key != "excerpt"
        }
        assert "".join(item["text"] for item in wire["excerpt"]) == original["excerpt"]
        assert all(set(item) == {"unit", "text"} and 0 < len(item["text"]) <= 500 for item in wire["excerpt"])
    assert meeting not in json.dumps(requests, ensure_ascii=False)
    system = requests[0]["messages"][0]["content"]
    assert all(term in system for term in ("独立", "前提", "否定", "必须", "不能", "只能"))
    assert all(term in system for term in ("同一对象", "同一条件", "原文明示", "未提及", "未测试", "用途分工", "能力不存在"))
    assert all(term in system for term in ("主体", "比较双方", "数字", "单位", "角色", "条件", "版本", "已完成"))
    assert all(term in system for term in ("任务书", "规范", "直接支持", "推断", "建议"))
    assert "draft" not in system and "草稿" not in system
    card = pipeline._states["q1"].card
    assert card.question == question and card.status is CardStatus.ANSWERED
    assert card.answer.startswith(f"【资料事实】{excerpt}[S1]")
    if review_status == "partial":
        assert "【范围提示】" in card.answer and "【补证建议】" in card.answer
        assert "【依据不足】" not in card.answer
    else:
        assert card.answer == f"【资料事实】{excerpt}[S1]"
    assert card.notice == ""
    assert card.sources == ("[S1] 合成方案.md · lines 7–9",)


@pytest.mark.asyncio
async def test_classroom_keeps_its_existing_prompt_and_context_in_real_request(monkeypatch):
    monkeypatch.setenv("CLASSBOT_COURSE", "")
    requests = []

    def handler(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, content=_sse("课堂答案"))

    settings = Settings("test-key", "课堂", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http)
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda *_: None)
        await pipeline._handle_candidate(QuestionCandidate("q1", "它为什么成立？", "课堂", 1, True, "课堂背景"))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()
    assert len(requests) == 1
    user_content = requests[0]["messages"][1]["content"]
    assert "背景信息：课堂背景" in user_content
    assert "问题：它为什么成立？" in user_content
    assert "本地材料候选" in user_content
    assert "网络检索候选" in user_content
    system = requests[0]["messages"][0]["content"]
    assert system.startswith("你是一个课堂问答助手。只输出直接、准确、简短的中文答案（1～3句）")
    assert "否定能力" not in system and "能力缺失" not in system and "逐条检查直接支持" not in system
