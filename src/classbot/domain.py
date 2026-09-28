from dataclasses import dataclass
from enum import StrEnum


class CardStatus(StrEnum):
    DETECTED = "detected"
    ANSWERING = "answering"
    ANSWERED = "answered"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    text: str
    is_final: bool
    start_ms: int = 0
    end_ms: int = 0
    confidence: float = 0.0
    pause_ms: int = 0
    segment_id: int | str | None = None

    def __post_init__(self) -> None:
        if self.start_ms > self.end_ms:
            raise ValueError("start_ms must not exceed end_ms")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    question_id: str
    text: str
    subject: str
    confidence: float
    is_final: bool

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.text.strip():
            raise ValueError("text must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


@dataclass(slots=True)
class AnswerCard:
    question_id: str
    question: str
    subject: str
    answer: str = ""
    status: CardStatus = CardStatus.DETECTED
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.question_id.strip():
            raise ValueError("question_id must not be empty")
        if not self.question.strip():
            raise ValueError("question must not be empty")
