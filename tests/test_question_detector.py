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


def test_growing_partial_transcript_reuses_question_id():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是",
                is_final=False,
                start_ms=100,
                end_ms=100,
            )
        )
    )
    second = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=100,
                end_ms=100,
            )
        )
    )
    final = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=100,
                end_ms=100,
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
                text="马克思的生辰是",
                is_final=False,
                start_ms=100,
                end_ms=100,
            )
        )
    )

    assert (
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是",
                is_final=True,
                start_ms=100,
                end_ms=100,
            )
        )
        == []
    )

    next_partial = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么",
                is_final=False,
                start_ms=100,
                end_ms=100,
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


def test_same_question_text_in_a_new_speech_segment_gets_a_new_id():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=100,
                end_ms=140,
            )
        )
    )
    second = _one(
        detector.feed(
            TranscriptEvent(
                text="马克思的生辰是什么？",
                is_final=True,
                start_ms=200,
                end_ms=240,
            )
        )
    )

    assert second.question_id != first.question_id


def test_nirong_statement_does_not_trigger_on_nonfinal_particle_substring():
    detector = QuestionDetector(subject="语文")

    candidates = detector.feed(
        TranscriptEvent(text="呢绒是一种纺织材料", is_final=True)
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
