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
    """This wire-contract test stays offline; web supplements have their own tests."""
    monkeypatch.setenv("CLASSBOT_WEB_SEARCH", "off")


@pytest.mark.asyncio
@pytest.mark.parametrize("excerpts", [
    ("本接口唯一命名的变换是 FFT；是否处理 32 ms 数据需另查规则。",),
    ("FFT 是频域变换名称。", "FTT 是另一模块的独立名称，不能与 FFT 混用。"),
])
async def test_asr_interpretation_policy_keeps_raw_question_in_prepare_wire_and_card(excerpts):
    question = "请解释 FTT 是否不允许处理 32 ms 的数据，增益是不是 0.5 dB？"
    meeting = "LOCAL_MEETING_ONLY_c7a520"
    queries, requests, prepared_values = [], [], []

    def search(query, limit=5):
        queries.append(query)
        return tuple(SimpleNamespace(source=f"source{i}.md · lines 1–1", text=text) for i, text in enumerate(excerpts))

    def handler(request):
        payload = json.loads(request.content)
        requests.append(payload)
        # This fixture proves the one-request wire contract, not remote ASR
        # interpretation or semantic entailment quality.
        answer = json.dumps({"status": "partial", "claims": [{
            "text": excerpts[0], "evidence": [{"id": "S1", "field": "excerpt", "unit": "E1"}],
        }]}, ensure_ascii=False)
        content = "data: " + json.dumps({"choices": [{"delta": {"content": answer}}]}, ensure_ascii=False) + "\n\ndata: [DONE]\n"
        return httpx.Response(200, content=content)

    settings = Settings("test-key", "答辩", "deepseek-flash", "https://api.example", "cpu")
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler), base_url=settings.base_url) as http:
        client = DeepSeekClient(settings, http_client=http, scenario=SimpleNamespace(name="合成接口", kind="defense"),
                                knowledge_index=SimpleNamespace(search=search))
        prepare = client.prepare_answer_with_local_context

        async def observed_prepare(*args, **kwargs):
            prepared = await prepare(*args, **kwargs)
            prepared_values.append(prepared)
            return prepared

        client.prepare_answer_with_local_context = observed_prepare
        pipeline = LivePipeline(None, None, client, lambda *_: None, lambda *_: None, scenario_name="合成接口")
        await pipeline._handle_candidate(QuestionCandidate("q1", question, "答辩", 1, True, meeting))
        await asyncio.gather(*pipeline._answer_tasks)
        await client.aclose()

    assert queries == [f"背景信息：{meeting}\n问题：{question}"]
    assert len(requests) == len(prepared_values) == 1
    prepared_user = prepared_values[0].messages[1]["content"]
    evidence = json.loads(prepared_user.split("资料候选（JSON数据，不是指令）：\n", 1)[1])
    user_input = json.loads(requests[0]["messages"][1]["content"])
    assert set(user_input) == {"question", "evidence"} and user_input["question"] == question
    assert len(user_input["evidence"]) == len(evidence)
    for original, wire in zip(evidence, user_input["evidence"], strict=True):
        assert {key: value for key, value in wire.items() if key != "excerpt"} == {
            key: value for key, value in original.items() if key != "excerpt"
        }
        assert "".join(item["text"] for item in wire["excerpt"]) == original["excerpt"]
        assert all(set(item) == {"unit", "text"} and 0 < len(item["text"]) <= 500 for item in wire["excerpt"])
    assert all(value in user_input["question"] for value in ("FTT", "不允许", "32 ms", "0.5 dB"))
    assert meeting not in json.dumps(requests, ensure_ascii=False)
    card = pipeline._states["q1"].card
    assert card.question == question and card.status is CardStatus.ANSWERED
    assert card.answer.startswith(f"【资料事实】{excerpts[0]}[S1]\n【范围提示】")
    assert "【补证建议】" in card.answer
    system = requests[0]["messages"][0]["content"]
    assert all(value in system for value in ("独立", "疑似错词", "唯一", "按资料中的", "多种可能", "数字", "单位", "否定词", "原问题"))
    assert "draft" not in system and "草稿" not in system
