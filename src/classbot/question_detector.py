from __future__ import annotations

from uuid import uuid4

from classbot.domain import QuestionCandidate, TranscriptEvent

QUESTION_MARKERS = (
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
    "吗",
    "呢",
)
QUESTION_MARKS = ("?", "？")


def normalize_text(text: str) -> str:
    """Keep letters and numbers while removing whitespace and punctuation."""
    if not text:
        return ""
    return "".join(character for character in text if character.isalnum())


class QuestionDetector:
    """Detect and deduplicate question candidates from transcript events."""

    def __init__(self, subject: str, minimum_length: int = 8) -> None:
        self.subject = subject
        self.minimum_length = minimum_length
        self._states: dict[str, tuple[str, bool]] = {}
        self._question_ids_by_text: dict[str, str] = {}
        self._seen_events: set[tuple[str, bool]] = set()

    def feed(self, event: TranscriptEvent) -> QuestionCandidate | None:
        """Return at most one candidate for a transcript event."""
        normalized_text = normalize_text(event.text)
        event_key = (normalized_text, event.is_final)
        if event_key in self._seen_events:
            return None
        self._seen_events.add(event_key)

        if len(normalized_text) < self.minimum_length:
            return None
        if not self._has_question_signal(event.text, normalized_text):
            return None

        question_id = self._find_question_id(normalized_text, event.is_final)
        if question_id is None:
            question_id = uuid4().hex
        else:
            previous_text, _ = self._states[question_id]
            if previous_text != normalized_text:
                self._question_ids_by_text.pop(previous_text, None)

        self._states[question_id] = (normalized_text, event.is_final)
        self._question_ids_by_text[normalized_text] = question_id

        return QuestionCandidate(
            question_id=question_id,
            text=normalized_text,
            subject=self.subject,
            confidence=self._score(event.text, normalized_text, event.is_final),
            is_final=event.is_final,
        )

    def _find_question_id(self, normalized_text: str, is_final: bool) -> str | None:
        existing_id = self._question_ids_by_text.get(normalized_text)
        if existing_id is not None:
            _, existing_is_final = self._states[existing_id]
            if not existing_is_final or is_final:
                return existing_id

        if not is_final:
            return None

        for question_id, (previous_text, previous_is_final) in reversed(
            tuple(self._states.items())
        ):
            if previous_is_final:
                continue
            if normalized_text.startswith(previous_text):
                return question_id
            if previous_text.startswith(normalized_text):
                return question_id
        return None

    @staticmethod
    def _has_question_signal(raw_text: str, normalized_text: str) -> bool:
        return any(marker in normalized_text for marker in QUESTION_MARKERS) or any(
            mark in raw_text for mark in QUESTION_MARKS
        )

    @staticmethod
    def _score(raw_text: str, normalized_text: str, is_final: bool) -> float:
        matched_markers = sum(
            marker in normalized_text for marker in QUESTION_MARKERS
        )
        score = 0.45 + min(matched_markers * 0.05, 0.25)
        if any(mark in raw_text for mark in QUESTION_MARKS):
            score += 0.1
        if is_final:
            score += 0.1
        return min(score, 1.0)
