import pytest

from classbot.domain import TranscriptEvent
from classbot.question_detector import QuestionDetector

STANDALONE_CASES = [
    ("支持中文吗", True),
    ("这个功能呢", True),
    ("可以离线用吗", True),
    ("这个接口怎么调用", True),
    ("部署需要多少内存", True),
    ("为什么选择随机森林", True),
    ("这个参数的作用是什么", True),
    ("它的局限有哪些", True),
    ("这个方案的风险是什么", True),
    ("数据预处理要做哪些步骤", True),
    ("能听到吗", False),
    ("我们先休息一下", False),
    ("谢谢大家", False),
    ("下面我们介绍一下创新点", False),
    ("这个图展示了实验结果", False),
    ("大家注意看这里", False),
    ("答案不就在这里吗", False),
    ("对吧", False),
    ("打开教材翻到第三页", False),
    ("这个和那个的区别", False),
    ("这个这个方案的风险呢", True),
    ("泛化——泛化性能怎么样", True),
    ("我是说，这个接口怎么调用", True),
    ("这个这个", False),
    ("呃", False),
    ("支持中文呀", True),
    ("部署复杂吧", True),
    ("这个方案可行啊", True),
    ("这个接口叫什么来着", True),
    ("能听见吗", False),
    ("声音清楚吗", False),
    ("看到屏幕了吗", False),
    ("共享屏幕了吗", False),
    ("麦克风好了吗", False),
    ("人来齐了吗", False),
    ("点个名好吗", False),
    ("大家安静好吗", False),
    ("先休息好吗", False),
    ("明白了吗", False),
    ("最近好吗", False),
    ("说他在所有之上啊", False),
    ("能听见我说话吗", False),
    ("大家好呀", False),
    ("这是一道很典型的题啊", False),
    ("这个方案就是这样啊", False),
    ("这个方案就是这样吧", False),
    ("我刚才用过来着", False),
    ("我刚才做过这个来着", False),
    ("他刚才要做实验来着", False),
]

CONTEXT_CASES = [
    ("这个系统的部署方式是什么", "那这个呢", True),
    ("支持离线使用吗", "这个呢", True),
    ("模型的评估指标有哪些", "召回率呢", True),
    ("训练用了多久", "那换成数据集B呢", True),
    ("这个接口怎么调用", "它呢", True),
    ("这个系统的部署方式是什么", "那我们先休息一下", False),
    ("模型效果怎么样", "对吧", False),
    ("打开教材", "这个呢", False),
    ("模型效果怎么样", "这个这个", False),
    ("模型效果怎么样", "呃", False),
    ("支持离线使用吗", "召回率呀", True),
    ("支持离线使用吗", "召回率吧", True),
    ("支持离线使用吗", "因为我现在的关系呢", False),
    ("支持离线使用吗", "这个我们等一下再讲呢", False),
    ("支持离线使用吗", "好呢", False),
    ("支持离线使用吗", "讲呢", False),
    ("后面再讲呢", "这个呢", False),
    ("支持离线使用吗", "我们继续吧", False),
    ("支持离线使用吗", "就这样吧", False),
    ("支持离线使用吗", "那好吧", False),
    ("支持离线使用吗", "先这样吧", False),
    ("支持离线使用吗", "好的吧", False),
    ("支持离线使用吗", "可以的呀", False),
    ("支持离线使用吗", "好的呢", False),
    ("支持离线使用吗", "是这样的呢", False),
    ("能离线吗", "这个呢", True),
]


@pytest.mark.parametrize(("text", "expected"), STANDALONE_CASES)
def test_classroom_standalone_cases(text, expected):
    detector = QuestionDetector("历史")
    event = TranscriptEvent(text=text, is_final=True, segment_id=1)
    assert bool(detector.feed(event)) is expected


@pytest.mark.parametrize(("seed", "text", "expected"), CONTEXT_CASES)
def test_classroom_context_cases(seed, text, expected):
    detector = QuestionDetector("历史")
    detector.feed(TranscriptEvent(text=seed, is_final=True, segment_id=1))
    event = TranscriptEvent(text=text, is_final=True, segment_id=2)
    assert bool(detector.feed(event)) is expected


def test_context_is_attached_to_followup():
    detector = QuestionDetector("历史")
    detector.feed(
        TranscriptEvent(text="支持离线使用吗", is_final=True, segment_id=1)
    )
    candidate = detector.feed(
        TranscriptEvent(text="这个呢", is_final=True, segment_id=2)
    )[0]
    assert "支持离线使用吗" in candidate.context
