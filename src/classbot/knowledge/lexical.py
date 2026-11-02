"""Compact lexical overlap without corpus-wide Chinese n-gram objects."""

from __future__ import annotations

import re
from collections.abc import Iterable

import numpy as np

__all__ = ["MAX_QUERY_CHARS", "MAX_QUERY_TERMS", "CompactLexicalIndex"]

MAX_QUERY_TERMS = 64
MAX_QUERY_CHARS = 512
_MAX_CONTEXT_CHARS = 128
_MAX_CONTEXT_TERMS = 8

_CHINESE_RANGE = r"\u3400-\u9fff"
_CHINESE_PHRASES = re.compile(f"[{_CHINESE_RANGE}]+")
_ASCII_IDENTIFIERS = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|\d+")
_IDENTIFIER_PARTS = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\b)|\d+")
_SEPARATOR = "\0"
_ASCII_CHARACTERS = re.compile(r"[A-Za-z_0-9]")


def _ascii_terms(text: str) -> set[str]:
    terms = set()
    for identifier in _ASCII_IDENTIFIERS.findall(text):
        terms.add(identifier.lower())
        terms.update(
            part.lower()
            for part in _IDENTIFIER_PARTS.findall(identifier.replace("_", " "))
        )
    return terms


def _chinese_terms(text: str) -> set[str]:
    return set(_ordered_chinese_terms(text))


def _ordered_chinese_terms(text: str) -> tuple[str, ...]:
    terms = {}
    for phrase in _CHINESE_PHRASES.findall(text):
        if len(phrase) == 1:
            terms[phrase] = None
        else:
            for position in range(len(phrase) - 1):
                for width in (2, 3):
                    if position + width <= len(phrase):
                        terms[phrase[position : position + width]] = None
    return tuple(terms)


def _bounded_text(text: str, maximum: int) -> str:
    if len(text) <= maximum:
        return text
    head_size = maximum // 2
    tail_size = maximum - head_size - 1
    head, tail = text[:head_size], text[-tail_size:]
    # Cutting a long identifier must not invent a shorter exact code term.
    if _ASCII_CHARACTERS.fullmatch(text[head_size - 1]) and _ASCII_CHARACTERS.fullmatch(
        text[head_size]
    ):
        head = re.sub(r"[A-Za-z_0-9]+$", "", head)
    if _ASCII_CHARACTERS.fullmatch(
        text[-tail_size - 1]
    ) and _ASCII_CHARACTERS.fullmatch(text[-tail_size]):
        tail = re.sub(r"^[A-Za-z_0-9]+", "", tail)
    return head + "\n" + tail


def _sample_terms(terms: tuple[str, ...], budget: int) -> tuple[str, ...]:
    if len(terms) <= budget:
        return terms
    if budget <= 0:
        return ()
    if budget == 1:
        return terms[-1:]
    return tuple(
        terms[position * (len(terms) - 1) // (budget - 1)] for position in range(budget)
    )


def _section_terms(
    text: str, budget: int, *, identifiers_from: str | None = None
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    identifier_text = text if identifiers_from is None else identifiers_from
    full_identifiers = tuple(
        dict.fromkeys(
            value.lower() for value in _ASCII_IDENTIFIERS.findall(identifier_text)
        )
    )
    ascii_parts = _ascii_terms(identifier_text) - set(full_identifiers)
    chinese = _ordered_chinese_terms(text)
    selected_identifiers = _sample_terms(full_identifiers, budget)
    remaining = budget - len(selected_identifiers)
    selected_chinese = _sample_terms(chinese, remaining)
    remaining -= len(selected_chinese)
    selected_parts = _sample_terms(tuple(sorted(ascii_parts)), remaining)
    return selected_identifiers + selected_parts, selected_chinese


def _query_terms(query: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    background, marker, question = query.rpartition("\n问题：")
    if not marker or not question.strip():
        return _section_terms(
            _bounded_text(query, MAX_QUERY_CHARS),
            MAX_QUERY_TERMS,
            identifiers_from=query,
        )
    ascii_terms, chinese_terms = _section_terms(
        _bounded_text(question, MAX_QUERY_CHARS),
        MAX_QUERY_TERMS,
        identifiers_from=question,
    )
    count = len(ascii_terms) + len(chinese_terms)
    context_budget = min(_MAX_CONTEXT_TERMS, count // 4, MAX_QUERY_TERMS - count)
    if context_budget:
        context_ascii, context_chinese = _section_terms(
            _bounded_text(background.removeprefix("背景信息："), _MAX_CONTEXT_CHARS),
            context_budget,
        )
        ascii_terms += tuple(term for term in context_ascii if term not in ascii_terms)
        chinese_terms += tuple(
            term for term in context_chinese if term not in chinese_terms
        )
    return ascii_terms, chinese_terms


_CONTEXT_REFERENCE = re.compile(
    r"这[个种些项套]|那[个些种]|上述|前者|后者|它|\b(?:it|this|that|those|these)\b", re.IGNORECASE,
)
_QUESTION_GRAMS = frozenset(_ordered_chinese_terms(
    "什么，为什么，如何，怎么，为何，是否，要不要，需不需要，有什么，不同，多少，哪种，"
    "请问，请解释，解释一下，你们，我们，这个，那个，应该，可以，能不能",
))
_QUESTION_IDENTIFIERS = frozenset({
    "what", "why", "how", "when", "where", "which", "is", "are", "was", "were",
    "the", "a", "an", "we", "you", "our", "your", "and", "or", "of", "to", "in",
})


def continues_topic(question: str, context: str) -> bool:
    """Reserve continuity evidence for references or shared bounded topic terms.

    This only routes candidate slots. It does not establish factual support or
    rewrite an uncertain ASR word, and question filler is not a shared topic.
    """
    if _CONTEXT_REFERENCE.search(question):
        return True
    question_ascii, question_chinese = _section_terms(_bounded_text(question, MAX_QUERY_CHARS), MAX_QUERY_TERMS)
    context_ascii, context_chinese = _section_terms(_bounded_text(context, MAX_QUERY_CHARS), MAX_QUERY_TERMS)
    common_ascii = set(question_ascii).intersection(context_ascii) - _QUESTION_IDENTIFIERS
    if any(term.strip("_") and not term.isdigit() for term in common_ascii):
        return True
    return any(
        len(term) > 1 and term not in _QUESTION_GRAMS
        for term in set(question_chinese).intersection(context_chinese)
    )


class CompactLexicalIndex:
    """Retain source references and one delimited ASCII term string per chunk.

    Chinese query grams can be checked directly in the immutable source text.
    Only a single-character query term needs Chinese phrase-boundary checks.
    ASCII identifiers and their parts use separators to preserve exact matches.
    """

    __slots__ = ("_ascii", "_texts")

    def __init__(self, texts: Iterable[str]):
        self._texts = tuple(texts)
        self._ascii = tuple(
            _SEPARATOR + _SEPARATOR.join(sorted(terms)) + _SEPARATOR if terms else ""
            for text in self._texts
            for terms in (_ascii_terms(text),)
        )

    def scores(
        self, query: str, *, candidates: Iterable[int] | None = None
    ) -> np.ndarray:
        """Score a bounded lexical query; omitted candidates retain full coverage.

        Long queries keep complete code identifiers and terms spread over the
        question's beginning and end. Structured background contributes at most
        one term per four question terms. Semantic retrieval uses its own query.
        Candidate scoring returns the usual full array, with zero elsewhere.
        """
        positions = range(len(self._texts))
        if candidates is not None:
            selected = {}
            for value in candidates:
                if (
                    isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer))
                    or not 0 <= value < len(self._texts)
                ):
                    raise ValueError("invalid lexical candidate index")
                selected[int(value)] = None
            positions = tuple(selected)
        ascii_terms, chinese_terms = _query_terms(query)
        denominator = len(ascii_terms) + len(chinese_terms)
        scores = np.zeros(len(self._texts), dtype=np.float32)
        if not denominator:
            return scores
        ascii_needles = tuple(_SEPARATOR + term + _SEPARATOR for term in ascii_terms)
        chinese_needles = tuple(term for term in chinese_terms if len(term) > 1)
        # The C regex scanner can reject unrelated chunks in one pass. Counting
        # remains exact for overlapping two/three-character terms after a hit.
        chinese_prefilter = (
            re.compile("|".join(map(re.escape, chinese_needles)))
            if len(chinese_needles) >= 8
            else None
        )
        single_characters = tuple(
            re.compile(f"(?<![{_CHINESE_RANGE}]){term}(?![{_CHINESE_RANGE}])")
            for term in chinese_terms
            if len(term) == 1
        )
        for position in positions:
            text, ascii_text = self._texts[position], self._ascii[position]
            chinese_matches = (
                sum(term in text for term in chinese_needles)
                if chinese_prefilter is None
                or chinese_prefilter.search(text) is not None
                else 0
            )
            matches = (
                sum(term in ascii_text for term in ascii_needles)
                + chinese_matches
                + sum(pattern.search(text) is not None for pattern in single_characters)
            )
            scores[position] = matches / denominator
        return scores
