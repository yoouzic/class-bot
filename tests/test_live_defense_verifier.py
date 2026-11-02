"""The receipt gate must never confuse component mocks with live proof."""
from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import pytest


def harness():
    path = Path(__file__).resolve().parents[1] / "scripts" / "defense" / "verify_live_defense.py"
    assert path.is_file(), "the reusable native verification harness is missing"
    spec = importlib.util.spec_from_file_location("live_defense_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def historical_case(*args, **kwargs):
    """Older fixtures intentionally exercise the pre-review receipt contract."""
    return harness().assess_case(*args, historical_receipt=True, **kwargs)


def complete_events():
    return [
        {"kind": "playback_start", "seconds": 0},
        {"kind": "capture_nonzero", "seconds": 0.1},
        {"kind": "segment", "speech_started": True, "seconds": 0.2},
        {"kind": "transcript", "text": "项目为什么选择量子协议", "is_final": True, "seconds": 1},
        {"kind": "candidate", "question_id": "q1", "text": "项目为什么选择量子协议", "is_final": True, "seconds": 1.1},
        {"kind": "retrieved_hits", "query": "项目为什么选择量子协议", "hits": [{"source": "方案.md · L1-L4", "text": "实际命中内容"}], "seconds": 1.15},
        {"kind": "prepared", "question": "项目为什么选择量子协议", "sources": ["[S1] 方案.md · L1-L4"], "seconds": 1.2},
        {"kind": "first_delta", "question_id": "q1", "seconds": 1.4},
        {"kind": "stream_complete", "question": "项目为什么选择量子协议", "seconds": 2},
        {"kind": "card", "question_id": "q1", "question": "项目为什么选择量子协议", "status": "answered", "answer": "【资料事实】采用该协议。[S1]", "sources": ["[S1] 方案.md · L1-L4"], "seconds": 2.1},
        {"kind": "segment", "speech_ended": True, "seconds": 2.2},
        {"kind": "playback_end", "seconds": 2.3},
    ]


def test_gate_accepts_a_complete_native_case_with_real_source_labels():
    result = historical_case(complete_events(), ["量子", "协议"], [r"方案\.md"])
    assert result["passed"]
    assert result["timings_seconds"]["first_answer_from_playback"] == 1.4


def test_direct_transcript_or_pcm_injection_cannot_pass_without_playback():
    events = [event for event in complete_events() if not event["kind"].startswith("playback")]
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "playback_completed" in result["missing"]


def test_a_later_cancelled_card_invalidates_an_earlier_answered_snapshot():
    events = complete_events()
    events.append({"kind": "card", "question_id": "q1", "question": "项目为什么选择量子协议", "status": "cancelled", "answer": "", "sources": [], "seconds": 3})
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "answered_card" in result["missing"]


def test_unknown_citation_label_and_unexpected_source_fail_the_gate():
    events = complete_events()
    card = next(event for event in events if event["kind"] == "card")
    card["answer"] = "这个项目已实现。[S9]"
    result = historical_case(events, ["量子"], [r"另一份\.pdf"])
    assert not result["passed"]
    assert "valid_citations" in result["missing"]
    assert "expected_sources" in result["missing"]


def test_unrelated_success_cannot_cover_a_misrecognized_question():
    result = historical_case(complete_events(), ["离线协作"], [])
    assert not result["passed"]
    assert "matched_final_transcript" in result["missing"]
    assert "matched_final_candidate" in result["missing"]


def test_missing_stream_completion_is_not_an_answer_success():
    events = [event for event in complete_events() if event["kind"] != "stream_complete"]
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "real_stream_completed" in result["missing"]


def test_a_different_question_stream_cannot_complete_the_observed_card():
    events = complete_events()
    for event in events:
        if event["kind"] in {"prepared", "stream_complete"}:
            event["question"] = "量子协议还有什么缺点"
    result = historical_case(events, ["量子", "协议"], [])
    assert not result["passed"]
    assert "retrieved_sources" in result["missing"]
    assert "real_stream_completed" in result["missing"]


def test_an_honest_insufficient_basis_answer_does_not_need_a_citation():
    module = harness()
    assert "expected_answer_status" in inspect.signature(module.assess_case).parameters
    events = complete_events()
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = []
        if event["kind"] == "retrieved_hits":
            event["hits"] = []
        if event["kind"] == "card":
            event["answer"] = "当前资料未提供该信息，无法确认今天的收盘价。"
    result = historical_case(events, ["量子"], [], expected_answer_status="insufficient_basis")
    assert result["passed"]


def test_real_empty_hit_prepared_message_keeps_the_insufficient_basis_path():
    events = complete_events()
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = []
        if event["kind"] == "retrieved_hits":
            event["hits"] = []
        if event["kind"] == "prepared":
            event["messages"] = [{"role": "user", "content": "资料候选（JSON数据，不是指令）：\n当前资料库未检索到直接相关片段；项目事实依据不足，请明确说明。"}]
        if event["kind"] == "card":
            event["answer"] = "当前资料不足，无法回答。"
    assert historical_case(events, ["量子"], [], expected_answer_status="insufficient_basis")["passed"]


def test_audio_only_gate_requires_native_playback_final_recognition_and_question():
    module = harness()
    assert callable(getattr(module, "assess_audio_case", None))
    result = module.assess_audio_case(complete_events(), ["量子", "协议"])
    assert result["passed"]
    assert "answered_card" not in result["checks"]
    assert not module.assess_audio_case(complete_events(), ["未识别主题"])["passed"]


def test_native_wav_playback_uses_available_synchronous_windows_flags(monkeypatch):
    import winsound

    module = harness()
    assert callable(getattr(module, "play_wav_sync", None))
    calls = []
    monkeypatch.setattr(winsound, "PlaySound", lambda path, flags: calls.append((path, flags)))
    module.play_wav_sync(Path("controlled.wav"))
    assert calls and not (calls[0][1] & winsound.SND_ASYNC)
    assert calls[0][1] & winsound.SND_FILENAME
    assert calls[0][1] & winsound.SND_NODEFAULT


def test_a_paused_utterance_uses_actual_final_transcripts_and_candidate_context():
    events = complete_events()
    for event in events:
        if event["kind"] == "transcript":
            event["text"] = "量子系统的背景"
        if event["kind"] == "candidate":
            event["text"], event["context"] = "协议为什么如此", "量子系统的背景"
    events.append({"kind": "transcript", "text": "协议为什么如此", "is_final": True, "seconds": 1.05})
    result = harness().assess_audio_case(events, ["量子", "协议"])
    assert result["passed"]


def test_each_requested_subquestion_must_reach_an_actual_final_candidate():
    module = harness()
    assert "expected_question_groups" in inspect.signature(module.assess_audio_case).parameters
    events = complete_events()
    events.append({"kind": "transcript", "text": "鲍勃要不要翻转比特", "is_final": True, "seconds": 1.05})
    result = module.assess_audio_case(events, ["量子", "翻转"], expected_question_groups=[["量子"], ["翻转"]])
    assert not result["passed"]
    assert "all_requested_subquestions_detected" in result["missing"]


def two_question_events():
    events = complete_events()
    text = "量子协议有什么适用边界"
    events.extend([
        {"kind": "candidate", "question_id": "q2", "text": text, "is_final": True, "seconds": 2.4},
        {"kind": "retrieved_hits", "query": text, "hits": [{"source": "方案.md · L8-L12", "text": "实际命中边界"}], "seconds": 2.45},
        {"kind": "prepared", "question": text, "sources": ["[S1] 方案.md · L8-L12"], "seconds": 2.5},
        {"kind": "stream_complete", "question": text, "seconds": 3},
        {"kind": "card", "question_id": "q2", "question": text, "status": "answered", "answer": "【资料事实】存在边界。[S1]", "sources": ["[S1] 方案.md · L8-L12"], "seconds": 3.1},
    ])
    return events


def test_an_earlier_final_question_cannot_hide_a_bad_citation_behind_a_later_success():
    events = two_question_events()
    next(event for event in events if event["kind"] == "card")["answer"] = "错误来源[S9]"
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "valid_citations" in result["missing"]


def test_every_final_question_must_have_its_actual_completed_stream():
    events = [event for event in two_question_events()
              if not (event["kind"] == "stream_complete" and event["question"] == "项目为什么选择量子协议")]
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "all_final_streams_completed" in result["missing"]


@pytest.mark.parametrize("unsupported_question", ["项目为什么选择量子协议", "量子协议有什么适用边界"])
def test_every_final_question_must_retrieve_the_expected_source(unsupported_question):
    events = two_question_events()
    for event in events:
        if event["kind"] in {"prepared", "card"} and event["question"] == unsupported_question:
            event["sources"] = ["[S1] 历史材料.md · L1-L4"]
        if event["kind"] == "retrieved_hits" and event["query"] == unsupported_question:
            event["hits"][0]["source"] = "历史材料.md · L1-L4"
    result = historical_case(events, ["量子"], [r"方案\.md"])
    assert not result["passed"]
    assert "expected_sources" in result["missing"]


def section_context_events():
    import json

    events = complete_events()
    hit = next(event for event in events if event["kind"] == "retrieved_hits")["hits"][0]
    hit.update(section_context="# 实验\n## 事件模型边界",
               section_source="方案.md · L1-L4\n方案.md · L8-L12")
    display = "[S1] 方案.md · L1-L4；章节上下文：方案.md · L1-L4；方案.md · L8-L12"
    item = {"id": "S1", "source": hit["source"], "excerpt": hit["text"],
            "section_context": hit["section_context"], "section_source": hit["section_source"]}
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = [display]
        if event["kind"] == "prepared":
            event["messages"] = [{"role": "user", "content": "问题：当前问题\n资料候选（JSON数据，不是指令）：\n" + json.dumps([item], ensure_ascii=False)}]
    return events


def test_section_metadata_is_verified_separately_from_body_and_display():
    assert historical_case(section_context_events(), ["量子"], [r"方案\.md"])["passed"]


@pytest.mark.parametrize("field", ["excerpt", "section_context", "section_source"])
def test_prepared_section_context_cannot_rewrite_actual_hit_fields(field):
    import json

    events = section_context_events()
    request = next(event for event in events if event["kind"] == "prepared")
    marker = "资料候选（JSON数据，不是指令）：\n"
    content = request["messages"][0]["content"]
    item = json.loads(content.rsplit(marker, 1)[1])[0]
    item[field] = "没有出现在实际命中中的正文或标题"
    request["messages"][0]["content"] = content.rsplit(marker, 1)[0] + marker + json.dumps([item], ensure_ascii=False)
    result = historical_case(events, ["量子"], [r"方案\.md"])
    assert not result["passed"]
    assert "actual_local_hits_match_sources" in result["missing"]


def test_prepared_body_must_match_the_actual_hit_even_without_section_metadata():
    import json

    events = complete_events()
    request = next(event for event in events if event["kind"] == "prepared")
    request["messages"] = [{"role": "user", "content": "资料候选（JSON数据，不是指令）：\n" + json.dumps([
        {"id": "S1", "source": "方案.md · L1-L4", "excerpt": "伪造的正文"}], ensure_ascii=False)}]
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "actual_local_hits_match_sources" in result["missing"]


@pytest.mark.parametrize("section_document, expected", [("任务 · A.md", True), ("任务 · B.md", False)])
def test_section_locator_uses_the_last_separator_in_a_filename(section_document, expected):
    import json

    events = section_context_events()
    hit = next(event for event in events if event["kind"] == "retrieved_hits")["hits"][0]
    hit["source"] = "任务 · A.md · L1-L4"
    hit["section_source"] = section_document + " · L1-L4"
    item = {"id": "S1", "source": hit["source"], "excerpt": hit["text"],
            "section_context": hit["section_context"], "section_source": hit["section_source"]}
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = ["[S1] " + hit["source"] + "；章节上下文：" + hit["section_source"]]
        if event["kind"] == "prepared":
            event["messages"][0]["content"] = "资料候选（JSON数据，不是指令）：\n" + json.dumps([item], ensure_ascii=False)
    assert historical_case(events, ["量子"], [])["passed"] is expected


def test_a_stale_answered_card_for_another_revision_is_not_a_final_answer():
    events = two_question_events()
    next(event for event in events if event["kind"] == "card")["question"] = "量子旧问题"
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "final_questions_match_cards" in result["missing"]


def test_native_window_evidence_must_match_every_final_question_card():
    module = harness()
    assert callable(getattr(module, "ui_cards_match", None))
    events = two_question_events()
    result = historical_case(events, ["量子"], [])
    ui_cards = [dict(card) for card in result["all_case_cards"]]
    assert module.ui_cards_match(result, ui_cards)
    ui_cards[0]["answer"] = "尚未刷新旧回答"
    assert not module.ui_cards_match(result, ui_cards)


def test_existing_receipt_files_cannot_be_silently_mixed_with_a_rerun(tmp_path):
    module = harness()
    first = module.Receipt(tmp_path)
    first.record("previous_run")
    original = (tmp_path / "events.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        module.Receipt(tmp_path)
    assert (tmp_path / "events.jsonl").read_bytes() == original


@pytest.mark.asyncio
async def test_defense_local_context_preparation_is_observed_without_changing_remote_messages(tmp_path):
    from types import SimpleNamespace

    from classbot.main import RuntimeDependencies

    class Client:
        async def prepare_answer(self, question, subject):
            raise AssertionError("Defense should call its explicit local-context method")

        async def prepare_answer_with_local_context(self, question, subject, *, local_context):
            assert question == "当前问题"
            assert local_context == "仅供本地检索的背景"
            return SimpleNamespace(sources=("[S1] example.md",), messages=[{"role": "user", "content": "当前问题和实际命中片段"}])

        async def stream_answer(self, *_args, **_kwargs):
            yield "not invoked"

    module = harness()
    receipt = module.Receipt(tmp_path)
    dependencies = module.observe_dependencies(RuntimeDependencies(client_factory=lambda _settings: Client()), receipt)
    client = dependencies.client_factory(SimpleNamespace(model="test"))
    prepared = await client.prepare_answer_with_local_context("当前问题", "答辩", local_context="仅供本地检索的背景")
    observed = [event for event in receipt.events if event["kind"] == "prepared"]
    assert len(observed) == 1
    assert observed[0]["messages"] == prepared.messages
    assert observed[0]["local_context"] == "仅供本地检索的背景"


def test_prepared_labels_without_actual_local_retrieval_events_do_not_pass():
    events = [event for event in complete_events() if event["kind"] != "retrieved_hits"]
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "actual_local_hits_match_sources" in result["missing"]


def test_a_short_subquestion_keeps_its_own_remote_request_and_uses_actual_local_context():
    events = complete_events()
    question = "解析模型有什么适用边界"
    context = "为什么正式训练要使用事件模型"
    for event in events:
        if event["kind"] == "transcript":
            event["text"] = context + question
        elif event["kind"] == "candidate":
            event["text"], event["context"] = question, context
        elif event["kind"] in {"prepared", "stream_complete"}:
            event["question"] = question
        elif event["kind"] == "retrieved_hits":
            event["query"] = context + question
        elif event["kind"] == "card":
            event["question"] = question
    result = historical_case(events, ["解析", "事件"], [])
    assert result["passed"]


def test_a_citation_looking_filename_never_grants_an_extra_source_id():
    events = complete_events()
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = ["[S1] task[S99].md · L1-L4"]
        elif event["kind"] == "retrieved_hits":
            event["hits"][0]["source"] = "task[S99].md · L1-L4"
        if event["kind"] == "card":
            event["answer"] = "【资料事实】已经实现。[S99]"
    result = historical_case(events, ["量子"], [])
    assert not result["passed"]
    assert "valid_citations" in result["missing"]


@pytest.mark.parametrize("group", ["[S1,S99]", "[S1, S99]", "[S1、S99]"])
def test_grouped_unknown_ids_cannot_hide_behind_a_separate_valid_citation(group):
    events = complete_events()
    next(event for event in events if event["kind"] == "card")["answer"] = "资料事实[S1]，额外结论" + group
    assert not historical_case(events, ["量子"], [])["passed"]


def test_grouped_known_ids_are_accepted_when_both_are_actual_retrieved_sources():
    events = complete_events()
    for event in events:
        if event["kind"] in {"prepared", "card"}:
            event["sources"] = ["[S1] 方案.md · L1-L4", "[S2] 方案.md · L8-L12"]
        elif event["kind"] == "retrieved_hits":
            event["hits"].append({"source": "方案.md · L8-L12", "text": "另一实际命中"})
        if event["kind"] == "card":
            event["answer"] = "【资料事实】有两项依据。[S1、S2]"
    assert historical_case(events, ["量子"], [])["passed"]


def test_window_close_after_event_loop_shutdown_does_not_create_a_coroutine():
    class Runtime:
        def stop(self):
            raise AssertionError("No runtime work may be scheduled after loop shutdown")

    harness().request_runtime_stop(Runtime())


@pytest.mark.asyncio
async def test_window_stop_while_running_schedules_runtime_cleanup():
    import asyncio

    stopped = []

    class Runtime:
        async def stop(self):
            stopped.append(True)

    harness().request_runtime_stop(Runtime())
    await asyncio.sleep(0)
    assert stopped == [True]


@pytest.mark.asyncio
async def test_http_observer_records_only_safe_modes_and_preserves_stream_and_request(tmp_path):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    lines = ['data: {"choices":[{"delta":{"reasoning_content":"private reasoning"}}]}',
             'data: {"choices":[{"delta":{},"finish_reason":"length"}]}', 'data: [DONE]']
    received = []

    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

    class HttpClient:
        @asynccontextmanager
        async def stream(self, *args, **kwargs):
            received.append((args, kwargs))
            yield Response()

    client = SimpleNamespace(_http_client=HttpClient())
    module = harness()
    receipt = module.Receipt(tmp_path)
    module.observe_http_stream(client, receipt)
    headers = {"Authorization": "Bearer test-secret"}
    payload = {"thinking": {"type": "enabled"}, "max_tokens": 2048,
               "messages": [{"content": "private source"}]}
    async with client._http_client.stream("POST", "/chat/completions", headers=headers, json=payload) as response:
        actual = [line async for line in response.aiter_lines()]
        assert response.status_code == 200
    assert actual == lines
    assert received[0][1]["json"] is payload
    assert received[0][1]["headers"] is headers
    safe = [event for event in receipt.events if event["kind"].startswith("http_")]
    assert [event["kind"] for event in safe] == ["http_request", "http_finish"]
    assert safe[0]["thinking"] == {"type": "enabled"}
    assert safe[0]["max_tokens"] == 2048
    assert safe[0]["request_number"] == safe[1]["request_number"] == 1
    assert safe[1]["finish_reason"] == "length"
    serialized = str(safe)
    assert "test-secret" not in serialized
    assert "private source" not in serialized
    assert "private reasoning" not in serialized


@pytest.mark.asyncio
async def test_http_observer_closes_early_line_iterator_in_the_request_task(tmp_path):
    import asyncio
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    closed_by = []
    request_task = asyncio.current_task()

    class Response:
        async def aiter_lines(self):
            try:
                yield 'data: {"choices":[{"finish_reason":"length"}]}'
                yield 'data: [DONE]'
            finally:
                closed_by.append(asyncio.current_task())

    class HttpClient:
        @asynccontextmanager
        async def stream(self, *_args, **_kwargs):
            yield Response()

    module = harness()
    client = SimpleNamespace(_http_client=HttpClient())
    module.observe_http_stream(client, module.Receipt(tmp_path))
    async with client._http_client.stream(json={"thinking": {"type": "enabled"}, "max_tokens": 2048}) as response:
        lines = response.aiter_lines()
        async for _line in lines:
            break
    assert closed_by == [request_task]


def test_only_settled_source_failure_can_continue_to_other_cases():
    module = harness()
    result = historical_case(complete_events(), ["量子"], ["not-present-source"])
    assert result["missing"] == ["expected_sources"]
    assert module.can_continue_after_source_failure(result)
    result["ui_widget_text_matches"] = False
    assert not module.can_continue_after_source_failure(result)
    result.pop("ui_widget_text_matches")
    result["checks"]["all_final_streams_completed"] = False
    assert not module.can_continue_after_source_failure(result)
    result["checks"]["all_final_streams_completed"] = True
    result["missing"].append("runtime_healthy")
    assert not module.can_continue_after_source_failure(result)


@pytest.mark.asyncio
async def test_answer_observer_closes_upstream_in_the_same_task(tmp_path):
    import asyncio
    from types import SimpleNamespace

    from classbot.main import RuntimeDependencies

    closed_by = []
    request_task = asyncio.current_task()

    class Client:
        async def prepare_answer(self, *_args):
            raise AssertionError("Not used")

        async def stream_answer(self, *_args, **_kwargs):
            try:
                yield "first"
                yield "second"
            finally:
                closed_by.append(asyncio.current_task())

    module = harness()
    dependencies = module.observe_dependencies(RuntimeDependencies(client_factory=lambda _settings: Client()),
                                               module.Receipt(tmp_path))
    client = dependencies.client_factory(SimpleNamespace(model="test"))
    answer = client.stream_answer("问题", "场景")
    assert await anext(answer) == "first"
    await answer.aclose()
    assert closed_by == [request_task]
