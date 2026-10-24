"""Conversational evidence stays bounded without losing the preceding subject."""

import numpy as np
import pytest

from classbot.knowledge.index import KnowledgeIndex
from classbot.knowledge.types import BuildReport, KnowledgeChunk


class QueryEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, query=False):
        assert query
        self.calls.append(tuple(texts))
        return np.asarray([
            [0., 1.] if text == "为什么轨道计算采用分段积分器" else [1., 0.]
            for text in texts
        ], dtype=np.float32)


def index_with_two_subjects(tmp_path):
    embedder = QueryEmbedder()
    # The short follow-up favours generic limitations; its preceding question
    # finds the actual integrator's design and assumptions.
    vectors = np.asarray([
        [1., 0.], [.99, .1], [.97, .2], [.95, .3], [0., 1.], [.1, .99],
    ], dtype=np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    chunks = tuple(KnowledgeChunk(f"source{i}.md", "lines 1–1", text) for i, text in enumerate((
        "通用边界讨论一", "通用边界讨论二", "通用边界讨论三", "通用边界讨论四",
        "轨道积分器设计与误差条件", "轨道积分器原始对照记录",
    )))
    index = KnowledgeIndex(root=tmp_path, revision="test", built_at="test",
                           report=BuildReport(6, 0, (), 6), chunks=chunks, vectors=vectors,
                           embedder=embedder, manifest=(), cache_dir=tmp_path)
    # Isolate ranking from tokenisation; neither channel has lexical overlap.
    index._lexical = type("NoLexical", (), {"scores": lambda self, q: np.zeros(6)})()
    return index, embedder


def test_short_followup_keeps_current_matches_and_preceding_subject_evidence(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)
    query = "背景信息：为什么轨道计算采用分段积分器\n问题：它有什么适用边界"

    hits = index.search(query)

    assert [hit.path.name for hit in hits] == [
        "source0.md", "source1.md", "source2.md", "source4.md", "source5.md",
    ]
    assert embedder.calls == [("它有什么适用边界", "为什么轨道计算采用分段积分器")]
    assert len(hits) == 5


@pytest.mark.parametrize("query", [
    "独立的新问题", "背景信息：  \n问题：独立的新问题",
    "普通文本里有\n问题：这不表示给出了结构化背景",
])
def test_noncontextual_queries_keep_one_channel(tmp_path, query):
    index, embedder = index_with_two_subjects(tmp_path)

    assert [hit.path.name for hit in index.search(query)] == [f"source{i}.md" for i in (0, 1, 2, 3, 5)]
    assert embedder.calls == [(query,)]


def test_one_result_always_belongs_to_the_current_query(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)
    query = "背景信息：为什么轨道计算采用分段积分器\n问题：新问题"

    assert index.search(query, limit=1)[0].path.name == "source0.md"
    assert embedder.calls == [("新问题",)]


def test_duplicate_channel_hits_fill_budget_without_repeating_sources(tmp_path):
    index, _ = index_with_two_subjects(tmp_path)
    query = "背景信息：另一个相同向量的背景\n问题：它的限制"

    hits = index.search(query)

    assert [hit.path.name for hit in hits] == [f"source{i}.md" for i in (0, 1, 2, 3, 5)]
    assert len({hit.source for hit in hits}) == 5


def test_zero_score_context_cannot_add_a_spurious_candidate(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)
    index._vectors = np.asarray([[1., 0.]] * 6, dtype=np.float32)
    query = "背景信息：为什么轨道计算采用分段积分器\n问题：它有什么适用边界"

    hits = index.search(query)

    assert len(hits) == 5 and all(hit.score > 0 for hit in hits)
    assert len({hit.source for hit in hits}) == 5
    assert len(embedder.calls[0]) == 2


def test_old_questions_cannot_drown_current_subject_or_recent_context_anchor(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)
    query = (
        "背景信息：过去讨论过模型边界；更早还讨论过文件格式；"
        "为什么轨道计算采用分段积分器\n问题：它有什么适用边界"
    )

    hits = index.search(query)

    assert [hit.path.name for hit in hits] == [
        "source0.md", "source1.md", "source2.md", "source4.md", "source5.md",
    ]
    assert embedder.calls == [("它有什么适用边界", "为什么轨道计算采用分段积分器")]


def test_trailing_context_separators_preserve_last_nonempty_utterance(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)

    index.search("背景信息：旧主题；为什么轨道计算采用分段积分器；  \n问题：它的限制")

    assert embedder.calls == [("它的限制", "为什么轨道计算采用分段积分器")]


def test_separator_only_context_cannot_fail_or_add_an_empty_anchor(tmp_path):
    index, embedder = index_with_two_subjects(tmp_path)

    hits = index.search("背景信息：； ；\n问题：当前问题")

    assert len(hits) == 5
    assert embedder.calls == [("当前问题",)]
