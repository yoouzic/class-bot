import pytest

from classbot.domain import QuestionCandidate, TranscriptEvent
from classbot.question_detector import QuestionDetector, normalize_text


def _one(candidates: list[QuestionCandidate]) -> QuestionCandidate:
    assert isinstance(candidates, list)
    assert len(candidates) == 1
    return candidates[0]


def test_normalize_text_removes_whitespace_and_punctuation():
    assert normalize_text(" 马克思的生辰是什么？（请回答） ") == "马克思的生辰是什么请回答"
    assert normalize_text(" Python 3.11：中文？ ") == "Python311中文"
    assert normalize_text(" \t\n") == ""


def test_statement_does_not_emit_question_candidate():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent(text="马克思出生于德国特里尔", is_final=True)
    )

    assert isinstance(candidates, list)
    assert candidates == []


def test_feed_returns_a_list_for_question_candidates():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent(text="马克思的生辰是什么", is_final=False)
    )

    assert isinstance(candidates, list)
    assert len(candidates) == 1
    assert isinstance(candidates[0], QuestionCandidate)


def test_begin_segment_separates_identical_final_without_event_segment_id():
    detector = QuestionDetector(subject="历史")
    event = TranscriptEvent(text="马克思的生辰是什么？", is_final=True)

    first = _one(detector.feed(event))
    detector.begin_segment()
    second = _one(detector.feed(event))

    assert second.question_id != first.question_id


def test_event_segment_id_takes_priority_over_fallback_identity():
    detector = QuestionDetector(subject="历史")
    detector.begin_segment("fallback")

    explicit = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                segment_id="explicit",
            )
        )
    )
    fallback = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么呢",
                is_final=False,
            )
        )
    )

    assert fallback.question_id != explicit.question_id


def test_late_final_from_previous_segment_reuses_previous_question_id():
    detector = QuestionDetector(subject="历史")
    partial = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                segment_id=1,
            )
        )
    )

    detector.begin_segment(segment_id=2)
    new_segment = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                segment_id=2,
            )
        )
    )
    late_final = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                segment_id=1,
            )
        )
    )

    assert new_segment.question_id != partial.question_id
    assert late_final.question_id == partial.question_id


def test_partial_question_triggers_before_final_transcript():
    detector = QuestionDetector(subject="历史")

    candidate = _one(
        detector.feed(
            TranscriptEvent(text="马克思的生辰是什么", is_final=False)
        )
    )

    assert candidate.text == "马克思的生辰是什么"
    assert candidate.is_final is False
    assert 0 <= candidate.confidence <= 1


def test_growing_partial_transcript_with_same_segment_id_reuses_question_id():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=100,
                end_ms=120,
                segment_id="segment-1",
            )
        )
    )
    second = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么呢",
                is_final=False,
                start_ms=200,
                end_ms=220,
                segment_id="segment-1",
            )
        )
    )
    final = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=300,
                end_ms=320,
                segment_id="segment-1",
            )
        )
    )

    assert second.question_id == first.question_id
    assert final.question_id == first.question_id


def test_final_revision_reuses_partial_question_id_and_normalizes_text():
    detector = QuestionDetector(subject="历史")
    partial = _one(
        detector.feed(
            TranscriptEvent(text="马克思的生辰是什么", is_final=False)
        )
    )

    final = _one(
        detector.feed(
            TranscriptEvent(text="马克思的生辰是什么？", is_final=True)
        )
    )

    assert final.question_id == partial.question_id
    assert final.text == normalize_text("马克思的生辰是什么？")
    assert final.is_final is True
    assert 0 <= final.confidence <= 1


def test_any_final_event_closes_active_question_lifecycle():
    detector = QuestionDetector(subject="历史")
    partial = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=100,
                end_ms=120,
                segment_id="segment-1",
            )
        )
    )

    assert (
        detector.feed(
            TranscriptEvent(
                text="这件事的结论是",
                is_final=True,
                start_ms=200,
                end_ms=220,
                segment_id="segment-1",
            )
        )
        == []
    )

    next_partial = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=300,
                end_ms=320,
                segment_id="segment-1",
            )
        )
    )

    assert next_partial.question_id != partial.question_id


def test_identical_final_event_is_emitted_only_once():
    detector = QuestionDetector(subject="历史")
    event = TranscriptEvent(text="马克思的生辰是什么？", is_final=True)

    _one(detector.feed(event))
    second = detector.feed(event)

    assert second == []


def test_late_question_mark_final_revision_is_not_deduplicated():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=True,
                start_ms=100,
                end_ms=120,
                segment_id="segment-1",
            )
        )
    )
    revised = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=100,
                end_ms=120,
                segment_id="segment-1",
            )
        )
    )

    assert revised.question_id == first.question_id
    assert revised.text == normalize_text("马克思的生辰是什么？")
    assert revised.is_final is True

    new_segment = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=100,
                end_ms=120,
                segment_id="segment-2",
            )
        )
    )

    assert new_segment.question_id != first.question_id


def test_embedded_question_mark_inside_quote_does_not_trigger():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent(
            text="老师问‘马克思是谁？’然后继续讲",
            is_final=False,
        )
    )

    assert candidates == []


def test_sentence_final_question_mark_triggers_candidate():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent(text="马克思的生辰是什么？）", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].is_final is True


def test_segment_id_change_creates_new_id_even_when_times_reset():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=0,
                end_ms=0,
                segment_id="segment-1",
            )
        )
    )
    second = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=0,
                end_ms=0,
                segment_id="segment-2",
            )
        )
    )

    assert second.question_id != first.question_id


@pytest.mark.parametrize(
    "text",
    [
        "这件事的结论是",
        "老师解释了为什么失败",
        "吗啡是一种药物",
        "呢绒是一种纺织材料",
    ],
)
def test_nonquestion_substrings_do_not_trigger(text):
    detector = QuestionDetector(subject="语文")

    candidates = detector.feed(
        TranscriptEvent(text=text, is_final=False)
    )

    assert candidates == []


def test_short_or_signal_free_text_does_not_emit_candidate():
    detector = QuestionDetector(subject="历史")

    assert detector.feed(TranscriptEvent(text="为什么", is_final=False)) == []
    assert (
        detector.feed(
            TranscriptEvent(text="马克思出生于德国特里尔", is_final=False)
        )
        == []
    )
