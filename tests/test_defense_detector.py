from classbot.defense_detector import DefenseQuestionDetector
from classbot.domain import TranscriptEvent


def event(text, *, final=True, segment=1):
    return TranscriptEvent(text, is_final=final, segment_id=segment)


def test_defense_requests_and_short_followups():
    detector = DefenseQuestionDetector("组会")
    assert detector.feed(event("介绍一下创新点"))[0].text == "介绍一下创新点"
    assert detector.feed(event("局限呢？", segment=2))[0].context


def test_multiple_questions_stay_together_and_preserve_identifiers():
    detector = DefenseQuestionDetector("竞赛")
    result = detector.feed(
        event("为什么选择 solve_model()？它和 baseline_v2 相比提升多少？")
    )
    assert len(result) == 1
    assert "solve_model()" in result[0].text
    assert "baseline_v2" in result[0].text
    assert "为什么" in result[0].text


def test_long_narrative_prefix_becomes_background():
    detector = DefenseQuestionDetector("组会")
    result = detector.feed(
        event("你们报告中介绍了随机森林的结果。请解释为什么不选择神经网络？")
    )
    assert result[0].text == "请解释为什么不选择神经网络？"
    assert "随机森林" in result[0].context


def test_management_and_narration_do_not_trigger():
    detector = DefenseQuestionDetector("组会")
    for index, text in enumerate(
        (
            "能听到吗？",
            "请打开摄像头",
            "下面我们介绍一下创新点",
            "我们来讨论如何提高准确率",
            "为什么",
        )
    ):
        assert detector.feed(event(text, segment=index)) == []


def test_revisions_use_same_id_and_new_segment_is_independent():
    detector = DefenseQuestionDetector("竞赛")
    first = detector.feed(event("为什么选择随机森林", final=False))[0]
    final = detector.feed(event("为什么选择随机森林？"))[0]
    assert first.question_id == final.question_id
    assert final.is_final
    assert detector.feed(event("为什么选择随机森林？")) == []
    repeated = detector.feed(event("为什么选择随机森林？", segment=2))[0]
    assert repeated.question_id != final.question_id


def test_first_person_project_question_is_not_a_narration():
    detector = DefenseQuestionDetector("竞赛")
    assert (
        detector.feed(event("我们的方法有什么局限？"))[0].text
        == "我们的方法有什么局限？"
    )


def test_common_defense_questions_are_not_removed_as_management_or_fragments():
    detector = DefenseQuestionDetector("竞赛")
    for index, text in enumerate(
        (
            "这个项目的创新点是什么？",
            "你们的核心目标是什么？",
            "你们刚才介绍了随机森林，为什么不用神经网络？",
            "神经网络有效吗？",
        )
    ):
        assert detector.feed(event(text, segment=index)), text


def test_declarative_presentation_verbs_do_not_trigger_answers():
    detector = DefenseQuestionDetector("组会")
    for index, text in enumerate(
        (
            "这个图展示了模型的实验结果。",
            "我们通过对比验证方法有效。",
            "下面分析训练集中的异常数据。",
        )
    ):
        assert detector.feed(event(text, segment=index)) == []
