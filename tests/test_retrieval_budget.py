"""Bound lexical query work without a timing-dependent regression threshold."""

import random

import numpy as np
import pytest


class CountedText(str):
    def __new__(cls, value):
        instance = super().__new__(cls, value)
        instance.lookups = 0
        return instance

    def __contains__(self, needle):
        self.lookups += 1
        return super().__contains__(needle)


def test_long_chinese_query_has_a_fixed_per_chunk_membership_budget():
    from classbot.knowledge.lexical import CompactLexicalIndex

    rng = random.Random(20261016)
    text = CountedText("".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(1200)))
    query = "".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(20000))
    index = CompactLexicalIndex((text,))

    scores = index.scores(query)

    assert scores.shape == (1,)
    assert text.lookups <= 64


def test_unrelated_long_query_skips_per_term_checks_after_one_prefilter():
    from classbot.knowledge.lexical import CompactLexicalIndex

    text = CountedText("一" * 1200)
    query = "".join(chr(0x5000 + i) for i in range(800))

    np.testing.assert_array_equal(CompactLexicalIndex((text,)).scores(query), [0])
    assert text.lookups == 0


def test_current_question_has_priority_over_a_long_background():
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(
        ("背景讨论调试缓存模型与表格导出", "轨道偏心率如何计算")
    )
    query = (
        "背景信息："
        + "背景讨论调试缓存模型与表格导出。" * 200
        + "\n问题：轨道偏心率如何计算"
    )

    scores = index.scores(query)

    assert scores[1] > scores[0]
    assert scores[1] >= 0.75


def test_full_code_identifier_is_kept_when_query_terms_exceed_budget():
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(
        ("unrelated", "def ComputeTrajectoryCorrection2(): pass")
    )
    query = (
        "背景信息："
        + "旧资料内容。" * 300
        + "\n问题："
        + "求解步骤如何得到？" * 30
        + "ComputeTrajectoryCorrection2"
    )

    scores = index.scores(query)

    assert scores[1] > 0
    assert scores[0] == 0


def test_identifier_in_middle_of_long_question_is_not_lost_by_text_window():
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(("def ComputeTrajectoryCorrection2(): pass",))
    query = "甲" * 400 + " ComputeTrajectoryCorrection2 " + "乙" * 400

    assert index.scores(query)[0] > 0


def test_chinese_terms_near_question_tail_are_not_lost_to_first_terms():
    from classbot.knowledge.lexical import CompactLexicalIndex

    prefix = "".join(chr(0x5000 + i) for i in range(800))
    index = CompactLexicalIndex(("轨道偏心率",))

    assert index.scores(prefix + "。轨道偏心率")[0] > 0


def test_long_query_projection_avoids_fabricating_cut_ascii_identifiers():
    from classbot.knowledge.lexical import CompactLexicalIndex

    query = "a" * 20000
    index = CompactLexicalIndex(("a" * 256, "a" * 128))

    np.testing.assert_array_equal(index.scores(query), np.zeros(2, dtype=np.float32))


def test_candidates_only_score_requested_chunks_and_keep_global_positions():
    from classbot.knowledge.lexical import CompactLexicalIndex

    texts = tuple(CountedText("轨道偏心率") for _ in range(10))
    index = CompactLexicalIndex(texts)

    scores = index.scores("轨道", candidates=np.asarray([8, 2, 8], dtype=np.int64))

    np.testing.assert_array_equal(scores[[2, 8]], np.ones(2, dtype=np.float32))
    assert np.count_nonzero(scores) == 2
    assert all(text.lookups == 0 for i, text in enumerate(texts) if i not in (2, 8))


@pytest.mark.parametrize("candidates", [[-1], [3], ["1"], [True]])
def test_invalid_candidate_indices_are_rejected(candidates):
    from classbot.knowledge.lexical import CompactLexicalIndex

    with pytest.raises(ValueError, match="candidate"):
        CompactLexicalIndex(("甲", "乙", "丙")).scores("甲", candidates=candidates)
