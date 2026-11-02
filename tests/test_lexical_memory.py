"""Preserve lexical ranking while bounding the corpus representation."""

import random
import re
import tracemalloc

import numpy as np
import pytest


def _reference_terms(text):
    """The original lexical scorer, kept here as a compatibility oracle."""
    terms = set()
    for phrase in re.findall(r"[\u3400-\u9fff]+", text):
        for width in (2, 3):
            terms.update(phrase[i : i + width] for i in range(len(phrase) - width + 1))
        if len(phrase) == 1:
            terms.add(phrase)
    for identifier in re.findall(r"[A-Za-z_][A-Za-z_0-9]*|\d+", text):
        terms.add(identifier.lower())
        terms.update(
            part.lower()
            for part in re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+",
                identifier.replace("_", " "),
            )
        )
    return frozenset(terms)


CORPUS = (
    "量子密钥分发系统采用抗干扰滤波，测量误码率。",
    "量子、密钥；中。中文 中文字 中文\n字",
    "中 abc中def 甲𠀀乙",
    "def filterSample(sample10, data_buffer): return XMLParser2(sample10)",
    "sample1 SAMPLE1 snake_case HTTPServer v1.2 123 001 __ _",
    "ABC parseHTTPResponse someXML xYz A1B2 _leading trailing_",
    "",
    "?! ... ---",
)


@pytest.mark.parametrize(
    "query",
    (
        "量子密钥的误码率",
        "抗干扰",
        "中文",
        "中文字",
        "中",
        "字",
        "甲乙",
        "中𠀀乙",
        "量子 sample1 误码率",
        "filterSample data_buffer",
        "sample1",
        "sample10",
        "sample",
        "1",
        "10",
        "001",
        "XMLParser2",
        "XML parser",
        "SNAKE_case",
        "HTTPServer parseHTTPResponse",
        "A1B2",
        "_ __ _leading trailing_",
        "xYz ABC someXML",
        "",
        "   ",
        "?! ---",
    ),
)
def test_compact_scores_match_original_chinese_and_code_lexical_overlap(query):
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(iter(CORPUS))
    query_terms = _reference_terms(query)
    expected = np.asarray(
        [
            len(query_terms & _reference_terms(text)) / max(1, len(query_terms))
            for text in CORPUS
        ],
        dtype=np.float32,
    )

    scores = index.scores(query)

    assert scores.dtype == np.float32
    assert scores.shape == (len(CORPUS),)
    np.testing.assert_array_equal(scores, expected)


def test_identifier_membership_does_not_confuse_sample1_with_sample10():
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(("sample10", "sample1"))

    np.testing.assert_array_equal(
        index.scores("1"), np.asarray([0, 1], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        index.scores("sample1"),
        np.asarray([1 / 3, 1], dtype=np.float32),
    )


def test_single_chinese_character_requires_a_single_character_phrase():
    from classbot.knowledge.lexical import CompactLexicalIndex

    index = CompactLexicalIndex(("中文", "中", "abc中def", "文中。中", "中_文"))

    np.testing.assert_array_equal(
        index.scores("中"),
        np.asarray([0, 1, 1, 1, 1], dtype=np.float32),
    )


def test_empty_corpus_and_queries_have_stable_empty_or_zero_arrays():
    from classbot.knowledge.lexical import CompactLexicalIndex

    assert CompactLexicalIndex(iter(())).scores("问题").shape == (0,)
    index = CompactLexicalIndex(("中文", "sample1"))
    np.testing.assert_array_equal(index.scores("?!"), np.zeros(2, dtype=np.float32))


def test_chinese_corpus_does_not_allocate_a_term_set_for_every_chunk():
    from classbot.knowledge.lexical import CompactLexicalIndex

    rng = random.Random(314159)
    texts = tuple(
        "".join(chr(rng.randrange(0x4E00, 0x9FA5)) for _ in range(1000))
        for _ in range(100)
    )
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        index = CompactLexicalIndex(iter(texts))
        _, peak = tracemalloc.get_traced_memory()
        assert peak - before < 5 * 1024 * 1024
        assert index.scores(texts[0][:3])[0] == 1
    finally:
        tracemalloc.stop()
