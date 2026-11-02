import pytest

from classbot.defense_detector import DefenseQuestionDetector
from classbot.domain import TranscriptEvent

STANDALONE_CASES = [
    ("这个项目的创新点是什么", True),
    ("为什么不用神经网络", True),
    ("你们的实验数据可靠吗", True),
    ("这个模块怎么部署", True),
    ("这个方案的局限在哪里", True),
    ("请解释一下这个接口", True),
    ("训练集和测试集怎么划分", True),
    ("这个结果和基线相比如何", True),
    ("系统支持并发吗", True),
    ("介绍一下你们的创新点", True),
    ("这个是不是你们做的", True),
    ("这个是谁负责的", True),
    ("这两个模块怎样连接", True),
    ("部署在哪一层", True),
    ("换个小模型会不会更稳", True),
    ("用了哪几个数据集", True),
    ("能听到吗", False),
    ("请打开摄像头", False),
    ("下面我们介绍一下创新点", False),
    ("这个图展示了模型的实验结果", False),
    ("谢谢大家", False),
    ("那我们先休息一下", False),
    ("我们通过对比验证方法有效", False),
    ("为什么", False),
    ("这是一个陈述句", False),
    ("网络有点卡", False),
    ("这个模型——不对，这个方法的泛化性怎么样", True),
    ("我们用的——呃——随机森林的准确率是多少", True),
    ("谢谢老师，请问这个方案为什么选随机森林", True),
    ("你好，请介绍一下你们的创新点", True),
    ("谢谢，我想问一下你们的实验数据可靠吗", True),
    ("呃", False),
    ("不对", False),
    ("我是说", False),
    ("那不是我们的方案", False),
]


@pytest.mark.parametrize(("text", "expected"), STANDALONE_CASES)
def test_defense_conversation_cases(text, expected):
    detector = DefenseQuestionDetector("答辩")
    event = TranscriptEvent(text=text, is_final=True, segment_id=1)
    assert bool(detector.feed(event)) is expected


def test_defense_followup_uses_context_and_confidence():
    detector = DefenseQuestionDetector("答辩")
    detector.feed(TranscriptEvent(text="你们的方案有什么局限？", is_final=True, segment_id=1))
    candidate = detector.feed(
        TranscriptEvent(text="那如果换成小模型呢？", is_final=True, segment_id=2)
    )[0]
    assert "那如果换成小模型呢" in candidate.text
    assert candidate.context
    assert candidate.confidence >= 0.5
