from classbot.domain import QuestionCandidate, TranscriptEvent
from classbot.question_detector import QuestionDetector, normalize_text


def test_normalize_text_removes_whitespace_and_punctuation():
    assert normalize_text(" 马克思的生辰是什么？（请回答） ") == "马克思的生辰是什么请回答"
    assert normalize_text(" \t\n") == ""


def test_statement_does_not_emit_question_candidate():
    detector = QuestionDetector(subject="历史")

    candidate = detector.feed(
        TranscriptEvent(text="马克思出生于德国特里尔", is_final=True)
    )

    assert candidate is None


def test_partial_question_triggers_before_final_transcript():
    detector = QuestionDetector(subject="历史")

    candidate = detector.feed(
        TranscriptEvent(text="马克思的生辰是什么", is_final=False)
    )

    assert isinstance(candidate, QuestionCandidate)
    assert candidate.text == "马克思的生辰是什么"
    assert candidate.is_final is False
    assert 0 <= candidate.confidence <= 1


def test_final_revision_reuses_partial_question_id_and_normalizes_text():
    detector = QuestionDetector(subject="历史")
    partial = detector.feed(
        TranscriptEvent(text="马克思的生辰是什么", is_final=False)
    )

    final = detector.feed(
        TranscriptEvent(text="马克思的生辰是什么？", is_final=True)
    )

    assert partial is not None
    assert final is not None
    assert final.question_id == partial.question_id
    assert final.text == normalize_text("马克思的生辰是什么？")
    assert final.is_final is True
    assert 0 <= final.confidence <= 1


def test_identical_final_event_is_emitted_only_once():
    detector = QuestionDetector(subject="历史")
    event = TranscriptEvent(text="马克思的生辰是什么？", is_final=True)

    first = detector.feed(event)
    second = detector.feed(event)

    assert first is not None
    assert second is None


def test_short_or_signal_free_text_does_not_emit_candidate():
    detector = QuestionDetector(subject="历史")

    assert detector.feed(TranscriptEvent(text="为什么", is_final=False)) is None
    assert (
        detector.feed(
            TranscriptEvent(text="马克思出生于德国特里尔", is_final=False)
        )
        is None
    )
