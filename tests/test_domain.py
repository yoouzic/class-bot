import pytest

from classbot.domain import AnswerCard, CardStatus, QuestionCandidate, TranscriptEvent


def test_domain_types_construct_valid_values():
    event = TranscriptEvent(
        text="什么时候开课？",
        is_final=True,
        start_ms=100,
        end_ms=200,
        confidence=0.9,
        pause_ms=50,
    )
    candidate = QuestionCandidate(
        question_id="q1",
        text=event.text,
        subject="历史",
        confidence=0.9,
        is_final=event.is_final,
    )
    card = AnswerCard(
        question_id=candidate.question_id,
        question=candidate.text,
        subject=candidate.subject,
    )

    assert card.status is CardStatus.DETECTED


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"start_ms": 201, "end_ms": 200},
        {"confidence": -0.1},
        {"confidence": 1.1},
    ],
)
def test_transcript_event_rejects_invalid_values(invalid_values):
    values = {"text": "片段", "is_final": False, "start_ms": 100, "end_ms": 200}
    values.update(invalid_values)

    with pytest.raises(ValueError):
        TranscriptEvent(**values)


@pytest.mark.parametrize(
    "invalid_values",
    [
        {"question_id": ""},
        {"text": ""},
        {"confidence": -0.1},
        {"confidence": 1.1},
    ],
)
def test_question_candidate_rejects_invalid_values(invalid_values):
    values = {
        "question_id": "q1",
        "text": "问题",
        "subject": "历史",
        "confidence": 0.5,
        "is_final": True,
    }
    values.update(invalid_values)

    with pytest.raises(ValueError):
        QuestionCandidate(**values)


@pytest.mark.parametrize("invalid_values", [{"question_id": ""}, {"question": ""}])
def test_answer_card_rejects_empty_required_text(invalid_values):
    values = {"question_id": "q1", "question": "问题", "subject": "历史"}
    values.update(invalid_values)

    with pytest.raises(ValueError):
        AnswerCard(**values)
