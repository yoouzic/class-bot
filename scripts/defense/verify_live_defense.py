"""Verify native Windows playback through the production defense runtime.

Nothing captures audio or calls DeepSeek until --run-live is supplied. TTS is
offline Windows synthesis, and winsound plays its WAV through Windows' default
render endpoint. Only native LoopbackCapture feeds the production audio loop.
Receipts stay under the requested local output folder; raw capture is not saved.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import wave
from array import array
from contextlib import aclosing, asynccontextmanager
from contextvars import ContextVar
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]


def _normalized(text):
    return re.sub(r"[\W_]+", "", text).lower()


def _prepared_evidence(prepared):
    marker = "资料候选（JSON数据，不是指令）：\n"
    for message in prepared.get("messages", ()):
        content = message.get("content", "")
        if message.get("role") != "user" or marker not in content:
            continue
        raw = content.rsplit(marker, 1)[1]
        if raw == "当前资料库未检索到直接相关片段；项目事实依据不足，请明确说明。":
            return []
        try:
            evidence = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return evidence if isinstance(evidence, list) else None
    return None


def _wire_evidence_units(wire, original):
    """Independently prove lossless, ordered source units from the actual wire."""
    if not isinstance(wire, list) or not isinstance(original, list) or len(wire) != len(original):
        return None
    units = {}
    for actual, expected in zip(wire, original, strict=True):
        if not isinstance(actual, dict) or not isinstance(expected, dict) or set(actual) != set(expected):
            return None
        for field, value in expected.items():
            if field not in {"excerpt", "section_context"}:
                if actual[field] != value:
                    return None
                continue
            parts = actual[field]
            if not isinstance(value, str) or not isinstance(parts, list):
                return None
            texts = []
            prefix = "E" if field == "excerpt" else "C"
            for number, part in enumerate(parts, 1):
                if (not isinstance(part, dict) or set(part) != {"unit", "text"}
                        or part["unit"] != f"{prefix}{number}"
                        or not isinstance(part["text"], str) or not 1 <= len(part["text"]) <= 500):
                    return None
                texts.append(part["text"])
                units[(expected["id"], field, part["unit"])] = part["text"]
            if "".join(texts) != value:
                return None
    return units


def _grounded_answer_checks(events, final_cards):
    """Check the single structured request and local publication independently."""
    from classbot.llm.citations import cited_source_ids

    def stage(kind):
        return [event for event in events if event["kind"] == kind]

    def proof_matches(review, evidence, units):
        if not isinstance(evidence, list) or units is None:
            return False
        claims, status = review.get("claims"), review.get("status")
        if (not isinstance(status, str) or status not in {"supported", "partial", "insufficient"}
                or not isinstance(claims, (list, tuple))):
            return False
        insufficient_lines = ["【依据不足】当前资料不足以完整回答该问题。",
                              "【补证建议】请补充与问题直接相关、包含对象和条件的原始记录或结果材料。"]
        partial_lines = ["【范围提示】以上仅列出本次候选直接支持的结论；未列出的部分尚未确认。",
                         "【补证建议】如需确认其他结论，请补充与问题直接相关的原始记录或结果材料。"]
        if status == "insufficient":
            return not claims and review.get("text") == "\n".join(insufficient_lines)
        if not claims:
            return False
        by_id = {item.get("id"): item for item in evidence if isinstance(item, dict)}
        bound_ids = set()
        rendered = []
        for claim in claims:
            if (not isinstance(claim, dict) or set(claim) != {"text", "evidence"}
                    or not isinstance(claim["text"], str) or not claim["text"].strip()
                    or claim["text"] not in review.get("text", "")
                    or not isinstance(claim["evidence"], list) or not claim["evidence"]):
                return False
            claim_ids = []
            for proof in claim["evidence"]:
                if not isinstance(proof, dict) or set(proof) != {"id", "field", "unit", "quote"}:
                    return False
                identifier, field, quote = proof["id"], proof["field"], proof["quote"]
                if (not isinstance(identifier, str) or identifier not in by_id
                        or not isinstance(field, str) or field not in {"excerpt", "section_context"}
                        or not isinstance(proof["unit"], str)
                        or not isinstance(quote, str) or not quote.strip()
                        or not isinstance(by_id[identifier].get(field), str)
                        or quote not in by_id[identifier][field]
                        or units.get((identifier, field, proof["unit"])) != quote):
                    return False
                bound_ids.add(identifier)
                if identifier not in claim_ids:
                    claim_ids.append(identifier)
            rendered.append("【资料事实】" + claim["text"].strip() + "".join(f"[{key}]" for key in claim_ids))
        if status == "partial":
            rendered.extend(partial_lines)
        return (bound_ids == cited_source_ids(review.get("text", ""))
                and review.get("text") == "\n".join(rendered))

    # A question ID may have older, already validated visible revisions before
    # ASR extends it. Evaluate those real snapshots too, without letting an
    # unrelated or unlinked completion authorize any visible body.
    proof_cards = list(final_cards)
    proof_cards.extend(e for e in stage("card") if e.get("answer", "").strip() and e not in proof_cards)
    per_card = []
    for card in proof_cards:
        question = card.get("question")
        requests = [e for e in stage("prepared") if e.get("question") == question
                    and tuple(e.get("sources", ())) == tuple(card.get("sources", ()))]
        linked = []
        for complete in stage("grounded_answer_complete"):
            if (complete.get("question") != question or not complete.get("answer_id")
                    or not complete.get("question_id") or complete["question_id"] != card.get("question_id")
                    or not isinstance(complete.get("text"), str) or not complete["text"].strip()
                    or complete["seconds"] > card.get("seconds", -1)):
                continue
            for start in stage("grounded_answer_start"):
                if (start.get("answer_id") != complete["answer_id"]
                        or start["seconds"] > complete["seconds"]
                        or any(start.get(key) != complete.get(key)
                               for key in ("question", "question_id", "subject", "prepared_id"))
                        or any(e.get("answer_id") == complete["answer_id"]
                               for e in stage("grounded_answer_error"))):
                    continue
                for request in requests:
                    if (not request.get("prepared_id") or request["prepared_id"] != start.get("prepared_id")
                            or request.get("subject") != start.get("subject")
                            or request.get("messages") != start.get("messages")
                            or tuple(request.get("sources", ())) != tuple(start.get("sources", ()))):
                        continue
                    linked.append((complete, start, request))
        selected = max(linked, key=lambda item: item[0]["seconds"]) if linked else None
        row = {"question_id": card.get("question_id"), "review_completed": bool(selected),
               "payload_matches": False, "quotes_match": False, "text_matches": False,
               "body_hidden": False, "single_complete_request": False, "published_once": False}
        if selected:
            complete, start, request = selected
            evidence = _prepared_evidence(request)
            units = None
            outbound = [e for e in stage("grounded_answer_request")
                        if e.get("answer_id") == complete["answer_id"]
                        and start["seconds"] <= e["seconds"] <= complete["seconds"]]
            if len(outbound) == 1:
                wire = outbound[0]
                finishes = [e for e in stage("http_finish")
                            if e.get("answer_id") == complete["answer_id"]
                            and e.get("request_number") == wire.get("request_number")
                            and e.get("finish_reason") == "stop"
                            and wire["seconds"] <= e["seconds"] <= complete["seconds"]]
                row["single_complete_request"] = bool(finishes)
                messages = wire.get("messages", [])
                if (isinstance(messages, list) and len(messages) == 2
                        and all(isinstance(m, dict) and set(m) == {"role", "content"} for m in messages)
                        and messages[0].get("role") == "system" and messages[1].get("role") == "user"
                        and wire.get("thinking") == {"type": "disabled"} and wire.get("max_tokens") == 2500):
                    try:
                        payload = json.loads(messages[1]["content"])
                    except (ValueError, TypeError):
                        payload = None
                    if isinstance(payload, dict):
                        units = _wire_evidence_units(payload.get("evidence"), evidence)
                    row["payload_matches"] = (isinstance(payload, dict)
                        and set(payload) == {"question", "evidence"}
                        and payload.get("question") == question and units is not None)
            row["quotes_match"] = proof_matches(complete, evidence, units)
            row["text_matches"] = complete["text"] == card.get("answer")
            publications = [e for e in stage("answer_delta")
                            if e.get("question_id") == card.get("question_id")
                            and e["seconds"] >= complete["seconds"]]
            row["published_once"] = (len(publications) == 1
                                     and publications[0].get("answer") == complete["text"])
            row["answer_id"] = complete["answer_id"]
            row["grounding_status"] = complete.get("status")
            row["_complete"] = complete
        per_card.append(row)
    trusted = [row["_complete"] for row in per_card
               if all(row[key] for key in ("review_completed", "payload_matches", "quotes_match",
                                           "text_matches", "single_complete_request"))]
    for row, card in zip(per_card, proof_cards, strict=True):
        visible = [e for e in events if e.get("question_id") == card.get("question_id")
                   and (e["kind"] == "first_delta" or
                        e["kind"] in {"answer_delta", "card"} and e.get("answer", "").strip())]
        row["body_hidden"] = bool(visible) and all(
            any(answer.get("question_id") == event.get("question_id")
                and answer.get("question") == event.get("question")
                and answer.get("text") == event.get("answer", answer.get("text"))
                and answer["seconds"] <= event["seconds"] for answer in trusted)
            for event in visible)
        row.pop("_complete", None)
    per_card = per_card[:len(final_cards)]
    fields = {"all_final_grounded_answers_completed": "review_completed",
              "one_complete_grounded_request_per_answer": "single_complete_request",
              "grounded_payload_matches_prepared": "payload_matches",
              "claim_quotes_match_actual_fields": "quotes_match",
              "verified_body_published_once": "published_once",
              "verified_text_matches_cards": "text_matches", "unvalidated_body_never_displayed": "body_hidden"}
    checks = {name: bool(per_card) and all(row[key] for row in per_card) for name, key in fields.items()}
    checks["no_free_draft_generation"] = not any(e["kind"] in {"stream_start", "draft_request"} for e in events)
    return checks, per_card


def observe_http_stream(client, receipt):
    """Record safe modes and structured outbound data; never headers/reasoning."""
    http_client = getattr(client, "_http_client", None)
    if http_client is None or not callable(getattr(http_client, "stream", None)):
        return
    original_stream = http_client.stream
    request_number = 0

    class ObservedResponse:
        def __init__(self, response, number, context):
            self.response, self.number, self.context = response, number, context
            self.line_iterators = []

        def __getattr__(self, name):
            return getattr(self.response, name)

        def aiter_lines(self):
            iterator = self._observe_lines()
            self.line_iterators.append(iterator)
            return iterator

        async def close_lines(self):
            for iterator in self.line_iterators:
                await iterator.aclose()

        async def _observe_lines(self):
            iterator = self.response.aiter_lines()
            try:
                async for line in iterator:
                    if line.startswith("data:"):
                        try:
                            event = json.loads(line[5:].strip())
                        except (ValueError, TypeError):
                            event = None
                        if isinstance(event, dict) and isinstance(event.get("choices"), list):
                            for choice in event["choices"]:
                                if isinstance(choice, dict) and self.context.get("purpose") == "grounded_answer":
                                    delta = choice.get("delta")
                                    if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                                        content = delta["content"]
                                        key = getattr(getattr(client, "_settings", None), "deepseek_api_key", None)
                                        if key:
                                            content = content.replace(key, "[REDACTED]")
                                        receipt.record("grounded_response", request_number=self.number,
                                                       answer_id=self.context["answer_id"], content=content)
                                if isinstance(choice, dict) and choice.get("finish_reason") is not None:
                                    reason = choice["finish_reason"]
                                    allowed = {"stop", "length", "content_filter", "tool_calls",
                                               "function_call", "insufficient_system_resource"}
                                    receipt.record("http_finish", request_number=self.number,
                                                   **self.context,
                                                   finish_reason=reason if isinstance(reason, str) and reason in allowed else "unsupported")
                    yield line
            finally:
                await iterator.aclose()

    @asynccontextmanager
    async def observed_stream(*args, **kwargs):
        nonlocal request_number
        request_number += 1
        number = request_number
        payload = kwargs.get("json", {})
        context = getattr(receipt, "request_context", None)
        context = (context.get() or {}) if context is not None else {}
        safe_context = {key: context[key] for key in ("purpose", "stream_id", "answer_id", "prepared_id")
                        if key in context}
        mode = payload.get("thinking", {}).get("type")
        receipt.record("http_request", request_number=number,
                       **safe_context,
                       thinking={"type": mode if mode in {"enabled", "disabled"} else "unsupported"},
                       max_tokens=payload.get("max_tokens"))
        if context.get("purpose") == "grounded_answer":
            receipt.record("grounded_answer_request", answer_id=context["answer_id"],
                           request_number=number, messages=payload.get("messages"),
                           thinking=payload.get("thinking"), max_tokens=payload.get("max_tokens"))
        elif context.get("purpose") == "draft":
            receipt.record("draft_request", stream_id=context["stream_id"], prepared_id=context["prepared_id"],
                           request_number=number, messages=payload.get("messages"))
        async with original_stream(*args, **kwargs) as response:
            observed = ObservedResponse(response, number, safe_context)
            try:
                yield observed
            finally:
                await observed.close_lines()
    http_client.stream = observed_stream


def can_continue_after_source_failure(result):
    """Retain a failed verdict while collecting later independent cases."""
    checks = result.get("checks", {})
    return bool(
        set(result.get("missing", [])) == {"expected_sources"}
        and all(checks.get(key) for key in (
            "runtime_healthy", "all_requested_subquestions_detected",
            "all_final_questions_answered", "all_final_streams_completed",
        ))
        and result.get("ui_card_matches", True)
        and result.get("ui_widget_text_matches", True)
    )


def assess_case(events, expected_terms, expected_sources, *, expected_answer_status="supported",
                expected_question_groups=(), historical_receipt=False):
    """Check observed stages; semantic support still requires human review."""
    def matching(text):
        return all(_normalized(term) in _normalized(text) for term in expected_terms)

    def stage(kind):
        return [event for event in events if event["kind"] == kind]

    final_candidates = [event for event in stage("candidate")
                        if event.get("is_final") and matching(event["text"] + " " + event.get("context", ""))]
    all_final_candidates = {event["question_id"]: event for event in stage("candidate") if event.get("is_final")}
    candidate_ids = {event["question_id"] for event in final_candidates}
    latest_cards = {event["question_id"]: event for event in stage("card")}
    answered = [card for key, card in latest_cards.items()
                if key in candidate_ids and card["status"] == "answered"
                and card.get("answer", "").strip()]
    card = answered[-1] if answered else {}
    prepared = [event for event in stage("prepared")
                if bool(card)
                and _normalized(event["question"]).endswith(_normalized(card["question"]))
                and tuple(event["sources"]) == tuple(card.get("sources", ()))]
    insufficient = expected_answer_status == "insufficient_basis"

    def requests_for(final_card):
        if not final_card:
            return []
        return [request for request in stage("prepared")
                if _normalized(request["question"]).endswith(_normalized(final_card["question"]))
                and tuple(request["sources"]) == tuple(final_card.get("sources", ()))]

    def completed(final_card):
        kind = "stream_complete" if historical_receipt else "grounded_answer_complete"
        return any(_normalized(event["question"]) == _normalized(request["question"])
                   for event in stage(kind) for request in requests_for(final_card))

    def valid_citations(final_card):
        from classbot.llm.citations import allowed_source_ids, cited_source_ids

        allowed = allowed_source_ids(final_card.get("sources", ()))
        cited = cited_source_ids(final_card.get("answer", ""))
        return bool(cited <= allowed and (cited or insufficient))

    def actual_hits_match(final_card):
        if not final_card:
            return False
        sources = [re.sub(r"^\[S\d+\]\s*", "", source) for source in final_card.get("sources", ())]
        marker = "资料候选（JSON数据，不是指令）：\n"
        for event in stage("retrieved_hits"):
            if not _normalized(event["query"]).endswith(_normalized(final_card["question"])):
                continue
            expected_items, expected_sources = [], []
            for number, hit in enumerate(event["hits"], 1):
                item = {"id": f"S{number}", "source": hit["source"], "excerpt": hit["text"][:1200]}
                display = hit["source"]
                context, context_source = hit.get("section_context", ""), hit.get("section_source", "")
                document, separator, locator = hit["source"].rpartition(" · ")
                if (isinstance(context, str) and context.strip()
                        and isinstance(context_source, str) and context_source.strip()
                        and separator and locator.strip()
                        and all(line.strip().rpartition(" · ")[0] == document
                                and bool(line.strip().rpartition(" · ")[1])
                                and bool(line.strip().rpartition(" · ")[2].strip())
                                for line in context_source.splitlines() if line.strip())):
                    item.update(section_context=context[:300], section_source=context_source)
                    display += "；章节上下文：" + "；".join(context_source.splitlines())
                expected_items.append(item)
                expected_sources.append(display)
            if expected_sources != sources:
                continue
            for request in requests_for(final_card):
                # Historical synthetic receipt fixtures did not record messages.
                # New native receipts always do, and metadata requires that proof.
                if "messages" not in request:
                    if not any("section_context" in item for item in expected_items):
                        return True
                    continue
                for message in request["messages"]:
                    content = message.get("content", "")
                    if message.get("role") != "user" or marker not in content:
                        continue
                    evidence_text = content.rsplit(marker, 1)[1]
                    if not expected_items and evidence_text == "当前资料库未检索到直接相关片段；项目事实依据不足，请明确说明。":
                        return True
                    try:
                        actual_items = json.loads(evidence_text)
                    except (ValueError, TypeError):
                        continue
                    if actual_items == expected_items:
                        return True
        return False

    final_cards = [latest_cards.get(key, {}) for key in all_final_candidates]
    checks = {
        "playback_completed": bool(stage("playback_start") and stage("playback_end")),
        "native_audio_nonzero": bool(stage("capture_nonzero")),
        "speech_segment_started": any(event.get("speech_started") for event in stage("segment")),
        "speech_segment_ended": any(event.get("speech_ended") for event in stage("segment")),
        "matched_final_transcript": matching(" ".join(event["text"] for event in stage("transcript")
                                                       if event.get("is_final"))),
        "matched_final_candidate": bool(final_candidates),
        "all_requested_subquestions_detected": all(
            any(event.get("is_final") and all(_normalized(term) in _normalized(event["text"])
                                              for term in group)
                for event in stage("candidate"))
            for group in expected_question_groups),
        "retrieved_sources": bool(prepared and (card.get("sources") or insufficient)),
        "real_stream_completed": any(
            _normalized(event["question"]) == _normalized(request["question"])
            for event in stage("stream_complete" if historical_receipt else "grounded_answer_complete")
            for request in prepared),
        "answered_card": bool(answered),
        "all_final_questions_answered": all(
            event["question_id"] in latest_cards
            and latest_cards[event["question_id"]]["status"] == "answered"
            and bool(latest_cards[event["question_id"]].get("answer", "").strip())
            for event in stage("candidate") if event.get("is_final")),
        "final_questions_match_cards": bool(all_final_candidates) and all(
            _normalized(latest_cards.get(key, {}).get("question", "")) == _normalized(candidate["text"])
            for key, candidate in all_final_candidates.items()),
        "all_final_questions_prepared": bool(final_cards) and all(requests_for(final_card) for final_card in final_cards),
        "all_final_streams_completed": bool(final_cards) and all(completed(final_card) for final_card in final_cards),
        "actual_local_hits_match_sources": bool(final_cards) and all(actual_hits_match(final_card) for final_card in final_cards),
        "valid_citations": bool(final_cards) and all(valid_citations(final_card) for final_card in final_cards),
        "expected_sources": not expected_sources or (
            bool(final_cards) and all(
                all(any(re.search(pattern, source) for source in final_card.get("sources", ()))
                    for pattern in expected_sources)
                for final_card in final_cards
            )
        ),
        "runtime_healthy": not any(event.get("status") in {"audio_error", "mic_error", "asr_error", "api_error"}
                                   for event in stage("status")),
    }
    if insufficient:
        checks["insufficient_basis_acknowledged"] = bool(re.search(
            r"资料.{0,10}(?:不足|未提供|没有|不包含)|没有.{0,6}(?:依据|资料)|无法(?:确认|回答|确定|给出)",
            card.get("answer", "")))
    review_details = []
    if not historical_receipt:
        review_checks, review_details = _grounded_answer_checks(events, final_cards)
        checks.update(review_checks)
    starts = stage("playback_start")
    origin = starts[0]["seconds"] if starts else 0
    timings = {}
    for label, kind in (("first_transcript_from_playback", "transcript"),
                        ("first_candidate_from_playback", "candidate"),
                        ("retrieval_from_playback", "prepared"),
                        ("first_answer_from_playback", "first_delta")):
        matches = stage(kind)
        if matches:
            timings[label] = round(matches[0]["seconds"] - origin, 4)
    if card:
        timings["answered_from_playback"] = round(card["seconds"] - origin, 4)
    return {"passed": all(checks.values()), "checks": checks,
            "missing": [key for key, value in checks.items() if not value],
            "timings_seconds": timings, "card": card, "all_case_cards": list(latest_cards.values()),
            "final_question_ids": list(all_final_candidates),
            "grounded_answer_by_final_question": review_details,
            "historical_receipt_mode": historical_receipt,
            "expected_sources_by_final_question": [
                {"question_id": question_id,
                 "missing_patterns": [pattern for pattern in expected_sources
                                      if not any(re.search(pattern, source)
                                                 for source in latest_cards.get(question_id, {}).get("sources", ()))]}
                for question_id in all_final_candidates
            ],
            "semantic_support_review": "required; a matching source label alone is not factual support"}


def ui_cards_match(result, ui_cards):
    actual = {card["question_id"]: card for card in ui_cards}
    expected = {card["question_id"]: card for card in result["all_case_cards"]}
    keys = result["final_question_ids"]
    return bool(keys) and all(
        key in actual and key in expected and actual[key].get("status") == "answered"
        and all(actual[key].get(field) == expected[key].get(field)
                for field in ("question", "answer", "status", "scenario_name"))
        and tuple(actual[key].get("sources", ())) == tuple(expected[key].get("sources", ()))
        for key in keys)


def assess_audio_case(events, expected_terms, *, expected_question_groups=()):
    result = assess_case(events, expected_terms, [], expected_question_groups=expected_question_groups,
                         historical_receipt=True)
    required = {"playback_completed", "native_audio_nonzero", "speech_segment_started",
                "speech_segment_ended", "matched_final_transcript", "matched_final_candidate",
                "runtime_healthy", "all_requested_subquestions_detected"}
    checks = {key: value for key, value in result["checks"].items() if key in required}
    return {"passed": all(checks.values()), "checks": checks,
            "missing": [key for key, value in checks.items() if not value],
            "timings_seconds": {key: value for key, value in result["timings_seconds"].items()
                                if key in {"first_transcript_from_playback", "first_candidate_from_playback"}},
            "scope": "native playback -> WASAPI -> VAD -> actual streaming ASR -> automatic question only; no index or cloud"}


class Receipt:
    def __init__(self, output):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        with (self.output / "events.jsonl").open("x", encoding="utf-8"):
            pass  # A rerun must choose a new output path, including failed runs.
        self.started = time.monotonic()
        self.case_id = None
        self.events = []
        self.cards = {}
        self._first_delta_ids = set()
        self.window = None

    def record(self, kind, **fields):
        event = {"kind": kind, "seconds": round(time.monotonic() - self.started, 4),
                 "case_id": self.case_id, **fields}
        # Capture what crossed the observation boundary, not references to a
        # client's mutable message/claim objects reused later in its lifecycle.
        serialized = json.dumps(event, ensure_ascii=False)
        event = json.loads(serialized)
        self.events.append(event)
        with (self.output / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(serialized + "\n")
        return event

    def upsert_card(self, card):
        self.cards[card.question_id] = asdict(card)
        self.record("card", **asdict(card))
        if self.window is not None:
            self.window.upsert_card(card)

    def update_answer(self, question_id, answer):
        question = self.cards.get(question_id, {}).get("question")
        if question_id not in self._first_delta_ids:
            self._first_delta_ids.add(question_id)
            self.record("first_delta", question_id=question_id, question=question)
        self.record("answer_delta", question_id=question_id, question=question, answer=answer)
        if self.window is not None:
            self.window.update_answer(question_id, answer)

    def set_status(self, status):
        self.record("status", status=status)
        if self.window is not None:
            self.window.set_status(status)

    def set_transcript(self, text):
        self.record("visible_transcript", text=text)
        if self.window is not None:
            self.window.set_transcript(text)

    def set_library_status(self, text, ready=False):
        self.record("library_status", text=text, ready=ready)
        if self.window is not None:
            self.window.set_library_status(text, ready=ready)


def observe_dependencies(dependencies, receipt):
    """Observe real object methods while retaining their original constructors."""
    def capture_factory():
        capture = dependencies.capture_factory()
        frames = capture.frames
        nonzero_cases = set()
        stats = {"frames": 0, "bytes": 0, "peak": 0, "nonzero_frames": 0}

        async def observed_frames():
            async for frame in frames():
                samples = array("h", frame)
                peak = max((abs(value) for value in samples), default=0)
                stats["frames"] += 1
                stats["bytes"] += len(frame)
                stats["peak"] = max(stats["peak"], peak)
                if peak:
                    stats["nonzero_frames"] += 1
                    if receipt.case_id not in nonzero_cases:
                        nonzero_cases.add(receipt.case_id)
                        receipt.record("capture_nonzero", peak=peak, frame_bytes=len(frame))
                yield frame
        capture.frames = observed_frames
        capture.verification_stats = stats
        return capture

    def segmenter_factory():
        segmenter = dependencies.segmenter_factory()
        accept = segmenter.accept

        def observed_accept(frame):
            event = accept(frame)
            if event.speech_started or event.speech_ended:
                receipt.record("segment", **asdict(event))
            return event
        segmenter.accept = observed_accept
        return segmenter

    def recognizer_factory(device):
        recognizer = dependencies.recognizer_factory(device)
        events = recognizer.events

        async def observed_events():
            async for event in events():
                receipt.record("transcript", **asdict(event))
                yield event
        recognizer.events = observed_events
        return recognizer

    def detector_factory(subject):
        detector = dependencies.detector_factory(subject)
        feed = detector.feed

        def observed_feed(event):
            candidates = feed(event)
            for candidate in candidates:
                receipt.record("candidate", **asdict(candidate))
            return candidates
        detector.feed = observed_feed
        return detector

    def client_factory(settings):
        client = dependencies.client_factory(settings)
        receipt.request_context = ContextVar("live_verification_request_context", default=None)
        observe_http_stream(client, receipt)
        prepare_answer, stream_answer = client.prepare_answer, client.stream_answer
        prepare_local = getattr(client, "prepare_answer_with_local_context", None)
        grounded_answer = getattr(client, "grounded_answer", None)
        prepared_objects = {}
        stream_count = grounded_count = 0

        def prepared_id(prepared):
            if prepared is None:
                return None
            key = id(prepared)
            if key not in prepared_objects:
                prepared_objects[key] = (prepared, f"p{len(prepared_objects) + 1}")
            return prepared_objects[key][1]

        async def observed_prepare(question, subject):
            started = time.monotonic()
            prepared = await prepare_answer(question, subject)
            receipt.record("prepared", question=question, subject=subject,
                           prepared_id=prepared_id(prepared), sources=prepared.sources,
                           retrieval_seconds=round(time.monotonic() - started, 4),
                           messages=prepared.messages, method="prepare_answer")
            return prepared

        async def observed_prepare_local(question, subject, *, local_context):
            started = time.monotonic()
            prepared = await prepare_local(question, subject, local_context=local_context)
            receipt.record("prepared", question=question, subject=subject,
                           prepared_id=prepared_id(prepared), sources=prepared.sources,
                           retrieval_seconds=round(time.monotonic() - started, 4),
                           messages=prepared.messages, local_context=local_context,
                           method="prepare_answer_with_local_context")
            return prepared

        async def observed_stream(question, subject, *, prepared=None):
            nonlocal stream_count
            stream_count += 1
            common = {"question": question, "subject": subject,
                      "prepared_id": prepared_id(prepared), "stream_id": f"g{stream_count}"}
            receipt.record("stream_start", **common, model=settings.model)
            draft = []
            context = receipt.request_context.set({"purpose": "draft", "stream_id": common["stream_id"],
                                                   "prepared_id": common["prepared_id"]})
            try:
                async with aclosing(stream_answer(question, subject, prepared=prepared)) as iterator:
                    async for delta in iterator:
                        draft.append(delta)
                        yield delta
            except BaseException as error:
                receipt.record("stream_incomplete", **common, error_type=type(error).__name__, draft="".join(draft))
                raise
            else:
                receipt.record("stream_complete", **common, draft="".join(draft))
            finally:
                receipt.request_context.reset(context)

        async def observed_grounded(question, subject, *, prepared):
            nonlocal grounded_count
            grounded_count += 1
            token = prepared_id(prepared)
            question_ids = [key for key, card in receipt.cards.items() if card.get("question") == question]
            common = {"question": question, "subject": subject, "prepared_id": token,
                      "question_id": question_ids[-1] if question_ids else None,
                      "answer_id": f"a{grounded_count}"}
            receipt.record("grounded_answer_start", **common,
                           sources=prepared.sources, messages=prepared.messages)
            context = receipt.request_context.set({"purpose": "grounded_answer", "answer_id": common["answer_id"]})
            try:
                result = await grounded_answer(question, subject, prepared=prepared)
            except BaseException as error:
                receipt.record("grounded_answer_error", **common, error_type=type(error).__name__)
                raise
            else:
                receipt.record("grounded_answer_complete", **common, text=result.text,
                               status=result.status, claims=result.claims)
                return result
            finally:
                receipt.request_context.reset(context)
        client.prepare_answer = observed_prepare
        if callable(prepare_local):
            client.prepare_answer_with_local_context = observed_prepare_local
        client.stream_answer = observed_stream
        if callable(grounded_answer):
            client.grounded_answer = observed_grounded
        return client

    return replace(dependencies, capture_factory=capture_factory,
                   segmenter_factory=segmenter_factory, recognizer_factory=recognizer_factory,
                   detector_factory=detector_factory, client_factory=client_factory)


def synthesize_cases(cases, output, voice):
    """Synthesize Mandarin locally. The PowerShell script contains no user text."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = [{"id": case["id"], "text": case["text"],
                "wav": str(output / f"{case['id']}.wav")} for case in cases]
    spec = output / "tts-input.json"
    spec.write_text(json.dumps({"voice": voice, "cases": payload}, ensure_ascii=False), encoding="utf-8")
    script = output / "synthesize.ps1"
    script.write_text(
        "param([string]$Spec)\n$ErrorActionPreference='Stop'\n"
        "Add-Type -AssemblyName System.Speech\n"
        "$config=Get-Content -Raw -LiteralPath $Spec -Encoding UTF8 | ConvertFrom-Json\n"
        "$synth=[System.Speech.Synthesis.SpeechSynthesizer]::new()\n"
        "try {\n$selected=$synth.GetInstalledVoices() | Where-Object { $_.Enabled -and "
        "$_.VoiceInfo.Name -eq $config.voice -and $_.VoiceInfo.Culture.Name -eq 'zh-CN' }\n"
        "if (-not $selected) { throw 'Requested enabled Mandarin voice is unavailable' }\n"
        "$synth.SelectVoice($config.voice)\n$synth.Rate=-1\n$synth.Volume=100\n"
        "foreach($item in $config.cases) {\n$synth.SetOutputToWaveFile($item.wav)\n"
        "$synth.Speak($item.text)\n$synth.SetOutputToNull()\n}\n"
        "} finally { $synth.Dispose() }\n", encoding="utf-8-sig")
    completed = subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-File", str(script),
                                "-Spec", str(spec)], capture_output=True, text=True, timeout=90, check=False)
    if completed.returncode:
        raise RuntimeError("Offline Mandarin TTS failed: " + completed.stderr[-1500:])
    return payload


def wav_receipt(path):
    with wave.open(str(path), "rb") as wav:
        return {"wav": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                "duration_seconds": round(wav.getnframes() / wav.getframerate(), 4),
                "sample_rate": wav.getframerate(), "channels": wav.getnchannels(),
                "sample_width": wav.getsampwidth()}


def play_wav_sync(path):
    import winsound
    # Python winsound uses synchronous playback by default; SND_SYNC is not
    # exposed on supported Python versions. Do not set SND_ASYNC.
    winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_NODEFAULT)


def asr_model_receipt(model_path):
    checkpoint = Path(model_path) / "model.pt"
    if not checkpoint.is_file():
        return {"model_path": str(model_path), "checkpoint_present": False}
    with checkpoint.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return {"model_path": str(model_path), "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": digest}


def request_runtime_stop(runtime):
    """Native close can also fire after run_until_complete closed its loop."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(runtime.stop())


async def run_live(args, cases):
    # Delayed imports keep --help and receipt tests free of model/audio imports.
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from classbot.config import Settings
    from classbot.knowledge import KnowledgeIndex, LocalEmbedder
    from classbot.main import RuntimeController
    from classbot.workspace import WorkspaceController

    torch.set_num_threads(args.cpu_threads)

    class CachedOnlyEmbedder(LocalEmbedder):
        def prepare(self, *, progress=None):
            # Use the actual pinned model and its actual encoder, with no download.
            with self._lock:
                self._load(local_files_only=True)

        def encode(self, texts, *, query=False):
            if not query:
                raise RuntimeError("Verification requires an existing complete index; rebuilding is disabled")
            return super().encode(texts, query=True)

    settings = Settings.from_env()
    if not settings.deepseek_api_key:
        raise ValueError("DEEPSEEK_API_KEY is missing")
    if urlsplit(settings.base_url).scheme != "https" or urlsplit(settings.base_url).hostname != "api.deepseek.com":
        raise ValueError("Live verification only sends requests to https://api.deepseek.com")
    receipt = Receipt(args.output)
    receipt.window = getattr(args, "window", None)
    receipt.selected_scenario_id, receipt.audio_device = args.scenario_id, None
    runtime = RuntimeController(settings, receipt)
    if receipt.window is not None:
        # User controls can stop this automated session without leaving capture
        # running after its native window is closed.
        receipt.window.stop_requested.connect(lambda: request_runtime_stop(runtime))
        receipt.window.close_requested.connect(lambda: request_runtime_stop(runtime))
    workspace = WorkspaceController(settings, receipt, runtime)
    workspace.profiles = workspace.store.load()
    profile = workspace.selected_profile
    if profile is None or profile.index_mode != "full":
        raise ValueError("Select an existing full-mode scenario using --scenario-id")
    if receipt.window is not None:
        receipt.window.set_scenarios(workspace.profiles, profile.scenario_id)
        receipt.window.setWindowTitle("Classbot · 实机链路验证 · " + profile.name)
        receipt.window.set_configuration_busy(True)
    receipt.record("session", utc=datetime.now(UTC).isoformat(), scenario=asdict(profile),
                   model=settings.model, api_host=urlsplit(settings.base_url).hostname,
                   classroom_thinking_effort=settings.thinking_effort,
                   defense_thinking_effort=settings.defense_thinking_effort,
                   audio_source="native Windows default WASAPI loopback", raw_audio_saved=False,
                   cpu_threads_requested=args.cpu_threads,
                   torch_cpu_threads_before_asr=torch.get_num_threads(),
                   torch_interop_threads=torch.get_num_interop_threads())

    def progress(message):
        if message.startswith("解析 "):
            raise RuntimeError("No current complete index cache; verification refuses to rebuild it")

    index_started = time.monotonic()
    embedder = CachedOnlyEmbedder(settings.embedding_model, cache_dir=settings.cache_dir / "models")
    index = await asyncio.to_thread(KnowledgeIndex.build, Path(profile.library_path), settings.cache_dir,
                                    model_name=settings.embedding_model, embedder=embedder,
                                    mode=profile.index_mode, progress=progress)
    if index.report.chunks <= 0 or index.report.limited_files:
        raise ValueError("Full cached index is empty or reports limited files")
    workspace.indices[profile.scenario_id] = index
    actual_search = index.search_async

    async def observed_search(query, limit=5):
        hits = await actual_search(query, limit=limit)
        receipt.record("retrieved_hits", query=query, hits=[
            {"source": hit.source, "path": str(hit.path), "locator": hit.locator,
             "score": hit.score, "text": hit.text,
             "section_context": getattr(hit, "section_context", ""),
             "section_source": getattr(hit, "section_source", "")} for hit in hits])
        return hits
    index.search_async = observed_search
    if receipt.window is not None:
        workspace._show_index_status()
    receipt.record("index_loaded", seconds_to_load=round(time.monotonic() - index_started, 4),
                   built_at=index.built_at, revision=index.revision, report=asdict(index.report),
                   vector_shape=list(index._vectors.shape))
    prepare_session = workspace.prepare_session

    async def observed_prepare_session():
        await prepare_session()
        runtime.dependencies = observe_dependencies(runtime.dependencies, receipt)
    runtime.prepare_session = observed_prepare_session
    results = []
    active = None
    try:
        active = await asyncio.wait_for(runtime.start(), args.startup_timeout)
        receipt.record("runtime_started", capture_type=type(active.capture).__name__,
                       segmenter_type=type(active.segmenter).__name__,
                       recognizer_type=type(active.recognizer).__name__,
                       client_type=type(active.client).__name__,
                       segmenter_end_silence_ms=active.segmenter.end_silence_ms,
                       segmenter_max_segment_ms=active.segmenter.max_segment_ms,
                       asr_final_refinement=active.recognizer.final_refinement,
                       asr_hotword_count=len(active.recognizer.hotwords),
                       asr_correction_count=len(active.recognizer.corrections),
                       torch_cpu_threads_after_asr=torch.get_num_threads(),
                       endpoint=asdict(active.capture.active_device),
                       asr_factory="WorkspaceController.prepare_session original FunAsrStreamingRecognizer factory",
                       asr_model_metadata={key: str(value) for key, value in
                                           getattr(active.recognizer._model, "kwargs", {}).items()
                                           if key in {"model", "model_path", "model_revision", "device", "ncpu"}})
        asr_path = getattr(active.recognizer._model, "kwargs", {}).get("model_path")
        if asr_path:
            receipt.record("asr_checkpoint", **await asyncio.to_thread(asr_model_receipt, asr_path))
        for case in cases:
            if active.cleaned:
                break
            await asyncio.sleep(1)
            receipt.case_id = case["id"]
            wav_path = Path(args.wav_dir) / f"{case['id']}.wav"
            receipt.record("playback_start", intended_text=case["text"], **wav_receipt(wav_path))
            await asyncio.to_thread(play_wav_sync, wav_path)
            receipt.record("playback_end")
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                events = [event for event in receipt.events if event["case_id"] == case["id"]]
                result = assess_case(events, case["expected_terms"], case.get("expected_sources", []),
                                     expected_answer_status=case.get("expected_answer_status", "supported"),
                                     expected_question_groups=case.get("expected_question_groups", []))
                pending_answers = any(not task.done() for task in active.pipeline._answer_tasks)
                settled = result["passed"] or can_continue_after_source_failure(result)
                if active.cleaned or (settled and not pending_answers) or any(event["kind"] == "status" and event["status"].endswith("error")
                                           for event in events):
                    break
                await asyncio.sleep(0.25)
            await asyncio.sleep(1)
            events = [event for event in receipt.events if event["case_id"] == case["id"]]
            result = {"case_id": case["id"], "intended_text": case["text"],
                      **assess_case(events, case["expected_terms"], case.get("expected_sources", []),
                                    expected_answer_status=case.get("expected_answer_status", "supported"),
                                    expected_question_groups=case.get("expected_question_groups", []))}
            recognition_terms = case.get("recognition_review_terms", [])
            actual_final_text = " ".join(event["text"] for event in events
                                         if event["kind"] == "transcript" and event.get("is_final"))
            result["recognition_term_review"] = {
                "actual_final_transcript": actual_final_text,
                "missing_terms": [term for term in recognition_terms
                                  if _normalized(term) not in _normalized(actual_final_text)],
                "scope": "literal preservation review; transport pass does not mean exact ASR or factual accuracy"}
            if receipt.window is not None:
                image = Path(args.output) / f"{case['id']}-window.png"
                saved = receipt.window.grab().save(str(image))
                ui_cards = [asdict(card) for card in receipt.window._cards.values()]
                result["ui_screenshot"] = str(image) if saved else None
                result["ui_card_matches"] = ui_cards_match(result, ui_cards)
                widget_cards, widget_images = [], []
                for number, question_id in enumerate(result["final_question_ids"], 1):
                    widget = receipt.window._card_widgets.get(question_id)
                    if widget is None:
                        continue
                    widget_cards.append({
                        "question_id": question_id, "question": widget.question_text_label.text(),
                        "answer": widget.answer_text_label.text(),
                        "status": widget.status_value_label.text().removeprefix("状态："),
                        "sources": widget.sources_text_label.text().splitlines(),
                        "scenario_name": widget.scenario_name_label.text(),
                    })
                    card_image = Path(args.output) / f"{case['id']}-card-{number:02d}.png"
                    widget_images.append({"question_id": question_id, "path": str(card_image),
                                          "saved": widget.grab().save(str(card_image))})
                result["ui_widget_text_matches"] = ui_cards_match(result, widget_cards)
                result["ui_card_screenshots"] = widget_images
                result["passed"] = bool(result["passed"] and result["ui_card_matches"]
                                        and result["ui_widget_text_matches"] and saved
                                        and receipt.window.isVisible()
                                        and all(item["saved"] for item in widget_images))
                receipt.record("ui_snapshot", screenshot=str(image), saved=saved,
                               cards=ui_cards, actual_widget_texts=widget_cards,
                               card_screenshots=widget_images, window_visible=receipt.window.isVisible())
            results.append(result)
            receipt.record("case_verdict", result=result)
            print(json.dumps({"case_id": case["id"], "passed": result["passed"],
                              "missing": result["missing"]}, ensure_ascii=False), flush=True)
            if not result["passed"]:
                if not can_continue_after_source_failure(result):
                    break
                receipt.record("source_gate_failed_continuing", missing=result["missing"])
    finally:
        receipt.case_id = None
        await runtime.stop()
        await workspace.aclose()
        if receipt.window is not None:
            receipt.window.set_configuration_busy(False)
        if active is not None:
            receipt.record("capture_summary", **active.capture.verification_stats)
            receipt.record("cleanup", cleaned=active.cleaned,
                           pipeline_diagnostics=active.pipeline.diagnostics,
                           capture_released=active.capture._stream is None)
        summary = {"passed": len(results) == len(cases) and all(result["passed"] for result in results),
                   "cases": results, "observed_cases": len(results), "requested_cases": len(cases),
                   "scope": "controlled offline Mandarin TTS -> default system playback -> native WASAPI -> production runtime -> local full index -> real DeepSeek",
                   "semantic_support_review": "required; inspect prepared excerpts and final answer in events.jsonl"}
        (Path(args.output) / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary["passed"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions-json", type=Path, required=True,
                        help="Local list of id, text, expected_terms, optional expected_sources regexes")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wav-dir", type=Path)
    parser.add_argument("--voice", default="Microsoft Huihui Desktop")
    parser.add_argument("--prepare-tts", action="store_true")
    parser.add_argument("--run-live", action="store_true")
    parser.add_argument("--show-window", action="store_true", help="Render actual MainWindow cards and save a window screenshot")
    parser.add_argument("--scenario-id")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--startup-timeout", type=float, default=300)
    parser.add_argument("--cpu-threads", type=int, default=4,
                        help="Initial Torch threads in this process; FunASR can apply its own ncpu setting")
    args = parser.parse_args(argv)
    if not args.prepare_tts and not args.run_live:
        parser.error("Supply --prepare-tts or --run-live; no audio/API actions occur by default")
    if args.run_live and (not args.scenario_id or not args.wav_dir):
        parser.error("--run-live needs --scenario-id and --wav-dir")
    cases = json.loads(args.questions_json.read_text(encoding="utf-8-sig"))
    if not isinstance(cases, list) or not cases:
        parser.error("Questions must be a nonempty JSON list")
    ids = set()
    for case in cases:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,60}", case.get("id", "")) or case["id"] in ids:
            parser.error("Case IDs must be unique safe filenames")
        if not isinstance(case.get("text"), str) or not case["text"].strip() or not case.get("expected_terms"):
            parser.error("Each case needs nonempty text and expected_terms")
        ids.add(case["id"])
    if args.prepare_tts:
        payload = synthesize_cases(cases, args.wav_dir or args.output, args.voice)
        for case in payload:
            print(json.dumps(wav_receipt(case["wav"]), ensure_ascii=False), flush=True)
    if args.run_live:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if args.show_window:
            sys.path.insert(0, str(ROOT / "src"))
            import qasync
            from PySide6.QtGui import QFont
            from PySide6.QtWidgets import QApplication

            from classbot.ui.main_window import MainWindow
            application = QApplication.instance() or QApplication([])
            application.setFont(QFont("Microsoft YaHei", 10))
            application.setQuitOnLastWindowClosed(False)
            args.window = MainWindow("实机链路验证")
            args.window.resize(1000, 900)
            args.window.show()
            loop = qasync.QEventLoop(application)
            asyncio.set_event_loop(loop)
            with loop:
                passed = loop.run_until_complete(run_live(args, cases))
            args.window.close()
        else:
            passed = asyncio.run(run_live(args, cases))
        return 0 if passed else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
