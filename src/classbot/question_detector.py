from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from difflib import SequenceMatcher
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
)
QUESTION_PARTICLES = ("吗", "呢")
QUESTION_MARKS = ("?", "？")
RECENT_EVENT_LIMIT = 128
QUESTION_SCORE_THRESHOLD = 0.5


@dataclass(slots=True)
class _ActiveQuestion:
    question_id: str
    normalized_text: str
    is_final: bool
    start_ms: int
    end_ms: int


def normalize_text(text: str) -> str:
    """Keep letters and numbers while removing whitespace and punctuation."""
    if not text:
        return ""
    return "".join(character for character in text if character.isalnum())


class QuestionDetector:
    """Detect local question candidates and track one active speech segment."""

    def __init__(self, subject: str, minimum_length: int = 8) -> None:
        self.subject = subject
        self.minimum_length = minimum_length
        self._active_question: _ActiveQuestion | None = None
        self._recent_events: deque[tuple[str, bool, int, int]] = deque(
            maxlen=RECENT_EVENT_LIMIT
        )

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        """Return zero or one candidate for a transcript event."""
        normalized_text = normalize_text(event.text)
        signature = (
            normalized_text,
            event.is_final,
            event.start_ms,
            event.end_ms,
        )
        if signature in self._recent_events:
            if event.is_final:
                self._active_question = None
            return []
        self._recent_events.append(signature)

        confidence = self._score(event, normalized_text)
        if (
            not self._has_enough_text(normalized_text, event)
            or confidence < QUESTION_SCORE_THRESHOLD
        ):
            if event.is_final:
                self._active_question = None
            return []

        question_id = self._question_id_for(event, normalized_text)
        candidate = QuestionCandidate(
            question_id=question_id,
            text=normalized_text,
            subject=self.subject,
            confidence=confidence,
            is_final=event.is_final,
        )

        if event.is_final:
            self._active_question = None
        else:
            self._active_question = _ActiveQuestion(
                question_id=question_id,
                normalized_text=normalized_text,
                is_final=False,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            )
        return [candidate]

    def _question_id_for(self, event: TranscriptEvent, normalized_text: str) -> str:
        active = self._active_question
        if (
            active is not None
            and self._same_segment(active, event)
            and self._same_question_text(active.normalized_text, normalized_text)
        ):
            return active.question_id
        return uuid4().hex

    def _has_enough_text(self, normalized_text: str, event: TranscriptEvent) -> bool:
        if len(normalized_text) >= self.minimum_length:
            return True
        return self._has_partial_question_cue(normalized_text, event)

    @staticmethod
    def _has_partial_question_cue(
        normalized_text: str, event: TranscriptEvent
    ) -> bool:
        return (
            not event.is_final
            and len(normalized_text) >= 1
            and normalized_text.endswith("是")
            and "的" in normalized_text
        )

    @staticmethod
    def _same_segment(active: _ActiveQuestion, event: TranscriptEvent) -> bool:
        if (
            active.start_ms == active.end_ms == event.start_ms == event.end_ms == 0
        ):
            return True
        return active.start_ms == event.start_ms

    @staticmethod
    def _same_question_text(previous: str, current: str) -> bool:
        if previous == current:
            return True
        if previous.startswith(current) or current.startswith(previous):
            return True
        return (
            SequenceMatcher(None, previous, current, autojunk=False).ratio() >= 0.6
        )

    @classmethod
    def _score(cls, event: TranscriptEvent, normalized_text: str) -> float:
        has_strong_marker = any(
            marker in normalized_text for marker in STRONG_QUESTION_MARKERS
        )
        has_ending_particle = normalized_text.endswith(QUESTION_PARTICLES)
        has_question_mark = any(mark in event.text for mark in QUESTION_MARKS)
        has_partial_cue = cls._has_partial_question_cue(normalized_text, event)

        score = 0.0
        if has_strong_marker:
            score += 0.55
        if has_ending_particle:
            score += 0.55
        if has_question_mark:
            score += 0.65
        if has_partial_cue:
            score += 0.5
        if event.pause_ms >= 300:
            score += 0.1
        if event.is_final:
            score += 0.1
        return min(score, 1.0)
