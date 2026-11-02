import copy
import json
import random

import pytest

from classbot.llm.grounding import (
    GroundedAnswer,
    GroundingReviewError,
    build_grounding_messages,
    parse_grounding_review,
    source_units,
)


@pytest.fixture
def evidence():
    return [
        {"id": "S1", "source": "方案[S99].md · lines 8–12", "excerpt": "方案安排模型甲校验、模型乙训练；这只是要求，不代表已经完成。"},
        {"id": "S2", "source": "审计.md · lines 30–31", "excerpt": "演示配置 V0 下，延迟上限为 12 ms。",
         "section_context": "## 事件实现的验证范围", "section_source": "审计.md · lines 20–29"},
    ]


def review(*, status="supported", text="方案安排模型甲校验、模型乙训练。", source_id="S1",
           field="excerpt", unit="E1", claims=None):
    if claims is None:
        claims = [{"text": text, "evidence": [{"id": source_id, "field": field, "unit": unit}]}]
    return json.dumps({"status": status, "claims": claims}, ensure_ascii=False)


def test_supported_answer_is_assembled_only_from_bound_claims(evidence):
    claims = [
        {"text": "方案安排模型甲校验、模型乙训练。", "evidence": [
            {"id": "S1", "field": "excerpt", "unit": "E1"},
            {"id": "S2", "field": "section_context", "unit": "C1"},
            {"id": "S1", "field": "excerpt", "unit": "E1"},
        ]},
        {"text": "演示配置 V0 下，延迟上限为 12 ms。", "evidence": [
            {"id": "S2", "field": "excerpt", "unit": "E1"},
        ]},
    ]
    result = parse_grounding_review(review(claims=claims), evidence)
    assert isinstance(result, GroundedAnswer)
    assert result.status == "supported"
    assert result.claims[0]["evidence"] == [
        {"id": "S1", "field": "excerpt", "unit": "E1", "quote": evidence[0]["excerpt"]},
        {"id": "S2", "field": "section_context", "unit": "C1", "quote": evidence[1]["section_context"]},
        {"id": "S1", "field": "excerpt", "unit": "E1", "quote": evidence[0]["excerpt"]},
    ]
    assert result.text == (
        "【资料事实】方案安排模型甲校验、模型乙训练。[S1][S2]\n"
        "【资料事实】演示配置 V0 下，延迟上限为 12 ms。[S2]"
    )
    claims[0]["text"] = "事后被修改的文本"
    assert result.claims[0]["text"] == "方案安排模型甲校验、模型乙训练。"


def test_partial_and_insufficient_use_fixed_local_limits_and_advice(evidence):
    partial = parse_grounding_review(review(status="partial"), evidence)
    insufficient = parse_grounding_review(review(status="insufficient", claims=[]), [])
    assert partial.status == "partial" and len(partial.claims) == 1
    assert partial.text.startswith("【资料事实】方案安排模型甲校验、模型乙训练。[S1]\n")
    assert "【范围提示】以上仅列出本次候选直接支持的结论；未列出的部分尚未确认。" in partial.text
    assert "【补证建议】如需确认其他结论，请补充与问题直接相关的原始记录或结果材料。" in partial.text
    assert "【依据不足】" not in partial.text and "不足以完整回答" not in partial.text
    assert insufficient.status == "insufficient" and insufficient.claims == ()
    assert insufficient.text.startswith("【依据不足】") and "【补证建议】" in insufficient.text
    assert "[S" not in insufficient.text
    assert "【资料事实】" not in insufficient.text


@pytest.mark.parametrize("raw", [
    "null", "[]", '"text"', "{}",
    '{"status":"supported","status":"insufficient","claims":[]}',
    '{"status":"insufficient","claims":[],"claims":[]}',
    '{"status":"insufficient","claims":[],"answer":"未经审核首段"}',
    '{"status":"insufficient","claims":[],"number":NaN}',
    '```json\n{"status":"insufficient","claims":[]}\n```',
    '{"status":"insufficient","claims":[]}额外自由总结',
    '{"status":"unsupported","claims":[]}',
    '{"status":true,"claims":[]}',
    '{"status":"insufficient","claims":null}',
    '{"status":"insufficient","claims":{}}',
    '{"status":"supported","claims":[]}',
    '{"status":"partial","claims":[]}',
    '{"status":"supported","claims":[{"text":"甲","text":"乙","evidence":[]}]}',
    '{"status":"supported","claims":[{"text":"甲","evidence":[{"id":"S1","id":"S2","field":"excerpt","quote":"原文引用"}]}]}',
])
def test_malformed_ambiguous_or_untrusted_free_text_responses_are_rejected(raw, evidence):
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(raw, evidence)


@pytest.mark.parametrize("change", [
    {"text": ""}, {"text": "   "}, {"text": "长" * 401}, {"text": 1},
    {"text": "结论[S1]"}, {"text": "结论[S1,S99]"}, {"text": "结论［Ｓ１］"},
    {"text": "结论[S]"}, {"text": "结论【S99】"},
    {"advice": "自由添加的结尾建议"}, {"evidence": []}, {"evidence": "S1"},
])
def test_each_published_claim_has_bounded_text_and_no_unbound_inline_citation(change, evidence):
    payload = json.loads(review())
    payload["claims"][0].update(change)
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(json.dumps(payload), evidence)


@pytest.mark.parametrize("change", [
    {"id": "S99"}, {"id": "方案[S99].md"}, {"id": 1},
    {"field": "source"}, {"field": "section_source"}, {"field": "missing"},
    {"unit": ""}, {"unit": "   "}, {"unit": "E0"}, {"unit": "E99"},
    {"unit": "C1"}, {"unit": 1}, {"extra": "不能忽略额外字段"},
    {"id": "S1", "field": "section_context", "unit": "C1"},
    {"id": "S2", "unit": "E2"}, {"quote": "演示配置 V0 下，延迟上限为 12 ms。"},
])
def test_only_existing_units_in_the_same_bound_candidate_field_are_accepted(change, evidence):
    payload = json.loads(review())
    payload["claims"][0]["evidence"][0].update(change)
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(json.dumps(payload), evidence)


def test_response_and_claim_and_evidence_count_limits(evidence):
    good = json.loads(review())["claims"][0]
    assert len(parse_grounding_review(review(claims=[good] * 5), evidence).claims) == 5
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(claims=[good] * 6), evidence)
    padded = review(status="insufficient", claims=[])
    padded += " " * (24000 - len(padded))
    assert parse_grounding_review(padded, evidence).status == "insufficient"
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(padded + " ", evidence)
    good["evidence"] *= 4
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(claims=[good]), evidence)
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(status="insufficient"), evidence)


def test_maximum_claim_length_and_local_500_character_unit_are_accepted():
    evidence = [{"id": "S1", "source": "长正文.txt", "excerpt": "长" * 500}]
    result = parse_grounding_review(review(text="事实" * 200), evidence)
    assert len(result.claims[0]["text"]) == 400
    assert len(result.claims[0]["evidence"][0]["quote"]) == 500


def test_duplicate_source_ids_do_not_allow_ambiguous_quote_binding(evidence):
    evidence.append({"id": "S1", "source": "其他.md", "excerpt": "不同来源不应共享同一编号"})
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(), evidence)


def test_messages_are_only_raw_question_and_same_evidence(evidence):
    question = "为什么模型甲不能处理 12 ms 缓冲？"
    original = copy.deepcopy(evidence)
    messages = build_grounding_messages(question, evidence)
    assert [message["role"] for message in messages] == ["system", "user"]
    wire_evidence = copy.deepcopy(original)
    for item in wire_evidence:
        item["excerpt"] = [{"unit": "E1", "text": item["excerpt"]}]
        if "section_context" in item:
            item["section_context"] = [{"unit": "C1", "text": item["section_context"]}]
    assert json.loads(messages[1]["content"]) == {"question": question, "evidence": wire_evidence}
    assert evidence == original
    prompt = messages[0]["content"]
    assert all(term in prompt for term in ("独立", "逐句", "主体", "比较双方", "数字", "单位", "角色"))
    assert all(term in prompt for term in ("条件", "版本", "文稿", "规范", "已完成", "前提", "否定"))
    assert all(term in prompt for term in ("supported", "partial", "insufficient", "section_context", "unit", "禁止复制"))
    assert "source" in prompt and "文件名" in prompt
    assert all(term in prompt for term in ("不可信", "直接支持", "推断", "建议"))
    assert "QKD" not in prompt and "解析模型" not in prompt
    assert "draft" not in prompt and "草稿" not in prompt


@pytest.mark.parametrize("field,prefix", [("excerpt", "E"), ("section_context", "C")])
@pytest.mark.parametrize("text", [
    "", "原文", " \r\n\t ", "字" * 500, "字" * 501, "🙂" * 1501,
    "甲" * 300 + "\r\n" + "乙" * 600 + "。尾句\n  ",
    "```python\r\n  value = 1.25\n\n```\n# 标题\n  原文结尾  ",
])
def test_source_units_preserve_every_character_and_have_unique_bounded_ids(text, field, prefix):
    units = source_units(text, field)
    assert "".join(item["text"] for item in units) == text
    assert [item["unit"] for item in units] == [f"{prefix}{i}" for i in range(1, len(units) + 1)]
    assert all(set(item) == {"unit", "text"} and 1 <= len(item["text"]) <= 500 for item in units)


def test_source_units_prefer_line_or_sentence_ends_and_roundtrip_random_inputs():
    assert source_units("a" * 300 + "\n" + "b" * 300, "excerpt")[0]["text"] == "a" * 300 + "\n"
    assert source_units("甲" * 400 + "。" + "乙" * 200, "excerpt")[0]["text"] == "甲" * 400 + "。"
    rng = random.Random(34)
    for _ in range(100):
        text = "".join(rng.choice(("a", "汉", "。", "\r\n", "\t", "🙂", " ")) for _ in range(rng.randrange(2500)))
        units = source_units(text, "excerpt")
        assert "".join(item["text"] for item in units) == text
        assert all(0 < len(item["text"]) <= 500 for item in units)


def test_units_are_resolved_from_the_original_slice_not_remote_quote():
    original = "甲" * 500 + "\n原文条件完整保留。\r\n"
    evidence = [{"id": "S1", "source": "原文.md", "excerpt": original}]
    result = parse_grounding_review(review(unit="E2"), evidence)
    assert result.claims[0]["evidence"][0]["quote"] == "\n原文条件完整保留。\r\n"
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(unit="E3"), evidence)
    with pytest.raises(GroundingReviewError):
        parse_grounding_review(review(), [{"id": "S1", "source": "空白.md", "excerpt": "    "}])


@pytest.mark.parametrize("field", ["source", "section_source", "", None])
def test_source_units_reject_non_evidence_fields(field):
    with pytest.raises(GroundingReviewError):
        source_units("原文", field)
