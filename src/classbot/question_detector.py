from __future__ import annotations

import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from uuid import uuid4

from classbot.domain import QuestionCandidate, TranscriptEvent

STRONG_QUESTION_MARKERS = (
    "什么",
    "哪年",
    "哪一年",
    "哪个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "多少",
    "是否",
    "请问",
    "有没有",
    "是不是",
    "能不能",
    "可不可以",
    "什么是",
    "啥意思",
    "什么意思",
    "对不对",
    "记不记得",
)
QUESTION_PARTICLES = ("吗", "呢")
QUESTION_MARKS = ("?", "？")
QUESTION_ENDINGS = (
    "吗",
    "呢",
    "呀",
    "啊",
    "吧",
    "对吧",
    "记不记得",
    "有没有",
    "是不是",
    "能不能",
    "可不可以",
    "什么",
    "哪个",
    "哪一个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "多少",
    "是否",
)
NARRATIVE_CUES = (
    "现在打开教材",
    "打开教材",
    "看一下",
    "我们来看看",
    "我们接着",
    "接着呢",
    "接着看",
    "接着来看",
    "接下来",
    "大家注意",
    "大家一定",
    "牢牢记住",
    "回顾一下",
    "重新回顾",
    "我们学习了",
    "学习了",
    "我们了解了",
    "了解一下",
    "只需要掌握",
    "这个大家",
    "最后一点",
    "首先",
    "其次",
    "然后呢",
    "也就是说",
    "答案不在",
    "刚刚已经",
    "具体来说",
    "来看一下",
)
_VOCATIVE_PREFIX = re.compile(
    r"^(?:各位(?:同学|同学们)?|大家(?:好)?|同学们?|老师)"
    r"[\s,，:：、—-]*"
)
NARRATIVE_PREFIX_ENDINGS = (
    "解释",
    "说明",
    "讲解",
    "讲述",
    "介绍",
    "讨论",
    "分析",
    "回答",
    "告诉",
    "提到",
    "证明",
    "问",
    "说",
)
NARRATIVE_ASPECT_SUFFIXES = ("", "了", "过", "着")
QUESTION_CLOSERS = frozenset(
    " \t\r\n\"'”’)]}>》」』）】〕〉〗〙〛｝»"
)
RECENT_EVENT_LIMIT = 128
ACTIVE_SEGMENT_LIMIT = 16
QUESTION_SCORE_THRESHOLD = 0.5


@dataclass(slots=True)
class _ActiveQuestion:
    question_id: str
    normalized_text: str
    is_final: bool
    segment_id: int | str | None


def normalize_text(text: str) -> str:
    """Keep letters and numbers while removing whitespace and punctuation."""
    if not text:
        return ""
    return "".join(character for character in text if character.isalnum())


def clean_question_text(text: str) -> str:
    """Remove only explicit vocatives before normalizing question text."""
    return normalize_text(_VOCATIVE_PREFIX.sub("", text.strip()))


class QuestionDetector:
    """Detect local question candidates and track one active speech segment."""

    def __init__(self, subject: str, minimum_length: int = 8) -> None:
        self.subject = subject
        self.minimum_length = minimum_length
        self._fallback_segment_id: int | str = uuid4().hex
        self._active_questions: OrderedDict[int | str, _ActiveQuestion] = OrderedDict()
        self._recent_events: deque[
            tuple[str, bool, int, int, int | str | None, bool]
        ] = deque(maxlen=RECENT_EVENT_LIMIT)

    def begin_segment(self, segment_id: int | str | None = None) -> None:
        """Switch the fallback identity without dropping other segment states."""
        self._fallback_segment_id = (
            segment_id if segment_id is not None else uuid4().hex
        )

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        """Return zero or one candidate for a transcript event."""
        normalized_text = clean_question_text(event.text)
        segment_id = self._segment_id_for(event)
        signature = (
            normalized_text,
            event.is_final,
            event.start_ms,
            event.end_ms,
            segment_id,
            self._has_question_mark(event.text),
        )
        confidence = self._score(event, normalized_text)
        if signature in self._recent_events:
            if event.is_final and (
                len(normalized_text) < self.minimum_length
                or confidence < QUESTION_SCORE_THRESHOLD
            ):
                self._active_questions.pop(segment_id, None)
            return []
        self._recent_events.append(signature)

        if self._is_context_noise(event.text, normalized_text):
            if event.is_final:
                self._active_questions.pop(segment_id, None)
            return []

        if len(normalized_text) < self.minimum_length or (
            confidence < QUESTION_SCORE_THRESHOLD
        ):
            if event.is_final:
                self._active_questions.pop(segment_id, None)
            return []

        question_id = self._question_id_for(segment_id, normalized_text)
        candidate = QuestionCandidate(
            question_id=question_id,
            text=normalized_text,
            subject=self.subject,
            confidence=confidence,
            is_final=event.is_final,
        )

        self._set_active_question(
            segment_id,
            _ActiveQuestion(
                question_id=question_id,
                normalized_text=normalized_text,
                is_final=event.is_final,
                segment_id=segment_id,
            ),
        )
        return [candidate]

    def _segment_id_for(self, event: TranscriptEvent) -> int | str:
        if event.segment_id is not None:
            return event.segment_id
        return self._fallback_segment_id

    def _question_id_for(
        self, segment_id: int | str, normalized_text: str
    ) -> str:
        active = self._active_questions.get(segment_id)
        if active is not None:
            self._active_questions.move_to_end(segment_id)
            return active.question_id
        return uuid4().hex

    def _set_active_question(
        self, segment_id: int | str, active: _ActiveQuestion
    ) -> None:
        self._active_questions[segment_id] = active
        self._active_questions.move_to_end(segment_id)
        while len(self._active_questions) > ACTIVE_SEGMENT_LIMIT:
            self._active_questions.popitem(last=False)

    @classmethod
    def _score(cls, event: TranscriptEvent, normalized_text: str) -> float:
        has_strong_marker = cls._has_strong_marker(normalized_text)
        has_ending_particle = normalized_text.endswith(QUESTION_PARTICLES)
        has_question_mark = cls._has_question_mark(event.text)
        if (
            has_strong_marker
            and not has_question_mark
            and any(mark in event.text for mark in QUESTION_MARKS)
        ):
            has_strong_marker = False

        score = 0.0
        if has_strong_marker:
            score += 0.55
        if has_ending_particle:
            score += 0.55
        if has_question_mark:
            score += 0.65
        if event.pause_ms >= 300:
            score += 0.1
        if event.is_final:
            score += 0.1
        return min(score, 1.0)

    @classmethod
    def _is_context_noise(cls, raw_text: str, normalized_text: str) -> bool:
        if len(normalized_text) < 16:
            return False
        if cls._has_question_ending(raw_text, normalized_text):
            return False
        return any(cue in raw_text or cue in normalized_text for cue in NARRATIVE_CUES)

    @classmethod
    def _has_question_ending(cls, raw_text: str, normalized_text: str) -> bool:
        if cls._has_question_mark(raw_text):
            return True
        return any(normalized_text.endswith(ending) for ending in QUESTION_ENDINGS)

    @staticmethod
    def _has_strong_marker(normalized_text: str) -> bool:
        if any(
            normalized_text.startswith(marker) or normalized_text.endswith(marker)
            for marker in STRONG_QUESTION_MARKERS
        ):
            return True

        for marker in sorted(
            STRONG_QUESTION_MARKERS,
            key=len,
            reverse=True,
        ):
            marker_start = normalized_text.find(marker)
            if marker_start <= 0:
                continue
            marker_end = marker_start + len(marker)
            if marker_end >= len(normalized_text):
                continue
            prefix = normalized_text[:marker_start]
            return not any(
                prefix.endswith(ending + aspect)
                for ending in NARRATIVE_PREFIX_ENDINGS
                for aspect in NARRATIVE_ASPECT_SUFFIXES
            )
        return False

    @staticmethod
    def _has_question_mark(raw_text: str) -> bool:
        text = raw_text.rstrip()
        while text and text[-1] in QUESTION_CLOSERS:
            text = text[:-1].rstrip()
        return text.endswith(QUESTION_MARKS)
