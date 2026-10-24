"""Regressions reproduced from native system-playback ASR on 2026-10-16."""
import pytest

from classbot.defense_detector import DefenseQuestionDetector
from classbot.domain import TranscriptEvent


@pytest.mark.parametrize("text", [
    "鲍勃筛选后要不要翻转比特",
    "鲍勃筛选后需不需要翻转比特",
])
def test_spoken_yes_no_followup_without_asr_question_mark_is_detected(text):
    detector = DefenseQuestionDetector("量子密钥分发答辩")
    detector.feed(TranscriptEvent("冻结的贝尔态规则中两个测量基的相关性有什么不同", True, segment_id=1))
    candidates = detector.feed(TranscriptEvent(text, True, segment_id=2))
    assert candidates, "A real ASR final without punctuation is still a question"
    assert candidates[0].text == text
    assert "贝尔态" in candidates[0].context


@pytest.mark.parametrize("text", [
    "下面我们介绍要不要翻转比特的规则",
    "我们来讨论需不需要翻转比特",
])
def test_first_person_narration_with_the_same_words_is_not_a_question(text):
    assert DefenseQuestionDetector("答辩").feed(TranscriptEvent(text, True, segment_id=1)) == []


def test_registration_retry_count_question_without_punctuation_is_detected():
    text = "报名服务请求失败后最多自动重试几次"
    result = DefenseQuestionDetector("报名服务合成验证").feed(TranscriptEvent(text, True, segment_id=1))
    assert result
    assert result[0].text == text


def test_narration_of_retry_count_does_not_become_a_question():
    text = "下面我们介绍报名服务最多重试几次的规则"
    assert DefenseQuestionDetector("报名服务合成验证").feed(TranscriptEvent(text, True, segment_id=1)) == []
