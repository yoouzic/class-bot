import numpy as np
import pytest

from classbot.knowledge.lexical import continues_topic


def test_literal_evidence_survives_a_different_semantic_score_scale():
    from classbot.knowledge.index import _channel_candidates

    semantic = np.array([.9, .89, .88, .87, .01, .86])
    lexical = np.array([0., 0., 0., 0., 1., 0.])
    original_linear = np.argsort(-(.7 * semantic + .3 * lexical))[:5]
    assert 4 not in original_linear  # The missing literal evidence regression.

    hits = _channel_candidates(semantic, lexical, 5)
    scaled = _channel_candidates(semantic * .01, lexical * .1, 5)

    assert [i for i, _ in hits] == [0, 1, 2, 4, 3]
    assert [i for i, _ in scaled] == [i for i, _ in hits]


@pytest.mark.parametrize("semantic,lexical", [
    ([1., .9, .8, .7, .6, .5], [0.] * 6),
    ([0.] * 6, [1., .9, .8, .7, .6, .5]),
])
def test_unused_channel_slots_return_to_the_other_positive_channel(semantic, lexical):
    from classbot.knowledge.index import _channel_candidates

    assert [i for i, _ in _channel_candidates(np.array(semantic), np.array(lexical), 5)] == list(range(5))


def test_channel_merge_never_adds_zero_rows_or_repeats_excluded_rows():
    from classbot.knowledge.index import _channel_candidates

    assert _channel_candidates(np.zeros(8), np.zeros(8), 5) == []
    hits = _channel_candidates(np.ones(8), np.ones(8), 5, excluded={0, 3, 5})
    assert [i for i, _ in hits] == [1, 2, 4, 6, 7]


def test_equal_semantic_matches_prioritize_the_literal_question_evidence():
    from classbot.knowledge.index import _channel_candidates

    hits = _channel_candidates(np.ones(4), np.array([0., .1, 1., .2]), 5)

    assert [i for i, _ in hits] == [2, 3, 1, 0]


@pytest.mark.parametrize("question,context,expected", [
    ("它的边界是什么", "轨道积分器设计", True),
    ("解析模型有什么适用边界", "为什么采用解析参考模型", True),
    ("如何处理 cache_revision", "cache_revision 的更新方式", True),
    ("what does X return", "How is X defined", True),
    ("在你们冻结的规则中筛选后要不要翻转比特", "请解释时间尺度以及窗口包含什么数据", False),
    ("量子密钥分发接口三种时间尺度以及窗口包含什么数据", "解析模型有什么适用边界", False),
    ("请问今天股票收盘价是多少", "贝尔态筛选后要不要翻转比特", False),
    ("这个有什么限制", "完全不同的背景", True),
])
def test_only_related_or_referential_questions_use_continuity_slots(question, context, expected):
    assert continues_topic(question, context) is expected
