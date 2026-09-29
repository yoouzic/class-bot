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


def test_qingwen_question_triggers_candidate():
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(
        TranscriptEvent(text="请问马克思的生辰", is_final=False)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "请问马克思的生辰"


def test_explicit_vocative_prefix_is_removed_from_question_text():
    detector = QuestionDetector(subject="历史")

    candidate = _one(
        detector.feed(
            TranscriptEvent(
                text="各位同学，马克思的生辰是什么？",
                is_final=True,
            )
        )
    )

    assert candidate.text == "马克思的生辰是什么"


def test_long_lecture_scaffolding_without_question_ending_is_filtered():
    detector = QuestionDetector(subject="哲学")

    assert (
        detector.feed(
            TranscriptEvent(
                text="现在打开教材看一下什么是运动什么是静止",
                is_final=True,
            )
        )
        == []
    )


def test_long_recap_without_question_ending_is_filtered():
    detector = QuestionDetector(subject="哲学")

    assert (
        detector.feed(
            TranscriptEvent(
                text=(
                    "那么接着呢我们又学习了马克思主义唯物论他是如何看世界的"
                    "在这里大家只需要掌握马克思主义的物质观尤其是关于物质的概念"
                    "啊它的定义物质的唯一特性是"
                ),
                is_final=True,
            )
        )
        == []
    )


def test_question_ending_survives_lecture_context_prefix():
    detector = QuestionDetector(subject="哲学")

    candidate = _one(
        detector.feed(
            TranscriptEvent(
                text="我们来看看，哲学的基本问题是什么？",
                is_final=True,
            )
        )
    )

    assert candidate.text == "我们来看看哲学的基本问题是什么"


@pytest.mark.parametrize(
    "text",
    [
        "马克思为什么出生在德国",
        "马克思如何影响历史",
        "马克思是哪一年出生的",
    ],
)
def test_spoken_question_marker_inside_sentence_triggers_candidate(text):
    detector = QuestionDetector(subject="历史")

    candidates = detector.feed(TranscriptEvent(text=text, is_final=False))

    assert len(candidates) == 1
    assert candidates[0].text == normalize_text(text)


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


def test_substantive_revision_reuses_id_within_explicit_segment_only():
    detector = QuestionDetector(subject="历史")
    first = _one(
        detector.feed(
            TranscriptEvent(
                text="请问马克思的生辰是什么",
                is_final=False,
                segment_id="segment-1",
            )
        )
    )
    revised = _one(
        detector.feed(
            TranscriptEvent(
                text="请问马克思出生在哪一年",
                is_final=False,
                segment_id="segment-1",
            )
        )
    )
    other_segment = _one(
        detector.feed(
            TranscriptEvent(
                text="请问马克思出生在哪一年",
                is_final=False,
                segment_id="segment-2",
            )
        )
    )

    assert revised.question_id == first.question_id
    assert other_segment.question_id != first.question_id


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


def test_embedded_explanation_phrase_is_not_a_question():
    detector = QuestionDetector(subject="哲学", hotwords=("世界观",))

    assert detector.feed(
        TranscriptEvent(
            text="其次呢我们又了解了什么是世界观",
            is_final=True,
        )
    ) == []


def test_incomplete_partial_is_waited_for():
    detector = QuestionDetector(subject="哲学", hotwords=("马克思主义",))

    assert detector.feed(
        TranscriptEvent(
            text="第二个重点就是马克思主义是如何",
            is_final=False,
        )
    ) == []


@pytest.mark.parametrize(
    "text",
    [
        "李信同学在吗",
        "声音大一点",
        "答案不就在这里吗",
        "对不对",
    ],
)
def test_management_and_isolated_confirmation_are_filtered(text):
    detector = QuestionDetector(subject="哲学", hotwords=("物质",))

    assert detector.feed(TranscriptEvent(text=text, is_final=True)) == []


def test_long_explanation_returns_only_the_final_academic_question():
    detector = QuestionDetector(subject="哲学", hotwords=("哲学", "世界观"))

    candidates = detector.feed(
        TranscriptEvent(
            text=(
                "哲学是系统化理论化的世界观然后所以诸位"
                "哲学的基本问题是什么呀"
            ),
            is_final=True,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].text == "哲学的基本问题是什么"


def test_called_student_question_removes_only_safe_name_prefix():
    detector = QuestionDetector(subject="哲学", hotwords=("哲学",))

    candidates = detector.feed(
        TranscriptEvent(text="李卓阳哲学的基本问题是什么", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "哲学的基本问题是什么"


def test_choice_question_survives_name_prefix():
    detector = QuestionDetector(subject="哲学")

    candidates = detector.feed(
        TranscriptEvent(text="陈林西第三题选哪个", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "第三题选哪个"


def test_confirmation_with_a_textbook_proposition_is_answerable():
    detector = QuestionDetector(subject="哲学", hotwords=("物质", "意识"))

    candidates = detector.feed(
        TranscriptEvent(
            text="物质和意识的辩证关系对不对",
            is_final=True,
        )
    )

    assert len(candidates) == 1


def test_course_topic_prefix_is_not_removed_as_a_student_name():
    detector = QuestionDetector(
        subject="哲学",
        hotwords=("世界观", "方法论"),
    )

    candidates = detector.feed(
        TranscriptEvent(text="世界观跟方法论的关系是什么", is_final=True)
    )

    assert len(candidates) == 1
    assert candidates[0].text == "世界观跟方法论的关系是什么"


@pytest.mark.parametrize(
    "text",
    [
        "因为我现在的关系呢",
        "强调的就是我们的事件不是凝固的而这个呢",
        "不管我们遇到什么问题啊大家",
        "赵阳没在是吧朱伟汉",
        "就是谁撕维就存在谁",
        "我们需要掌握的哲学的基本问题什么是物质什么",
    ],
)
def test_discourse_and_confirmation_fragments_are_filtered(text):
    detector = QuestionDetector(subject="哲学", hotwords=("关系",))

    assert detector.feed(TranscriptEvent(text=text, is_final=True)) == []


def test_question_before_teacher_answer_is_extracted():
    detector = QuestionDetector(subject="哲学", hotwords=("物质",))

    candidates = detector.feed(
        TranscriptEvent(
            text="这么快就忘了物质的唯一特性是什么呀我客观实在性啊很好",
            is_final=True,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].text == "物质的唯一特性是什么"


def test_management_prefix_before_academic_question_is_removed():
    detector = QuestionDetector(subject="哲学", hotwords=("哲学",))

    candidates = detector.feed(
        TranscriptEvent(
            text="听题目了没有啊哲学是什么样的世界观",
            is_final=True,
        )
    )

    assert len(candidates) == 1
    assert candidates[0].text == "哲学是什么样的世界观"
