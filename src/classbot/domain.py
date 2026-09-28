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


@dataclass(frozen=True, slots=True)
class QuestionCandidate:
    question_id: str
    text: str
    subject: str
    confidence: float
    is_final: bool


@dataclass(slots=True)
class AnswerCard:
    question_id: str
    question: str
    subject: str
    answer: str = ""
    status: CardStatus = CardStatus.DETECTED
    error: str | None = None
