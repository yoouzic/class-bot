import pytest

from classbot.audio.segmenter import SegmentEvent, SpeechSegmenter

SPEECH_FRAME = b"\x01" * 640
SILENCE_FRAME = b"\x00" * 640


def test_first_speech_frame_emits_speech_started(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(
        segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME
    )

    started = segmenter.accept(SPEECH_FRAME)
    following = segmenter.accept(SPEECH_FRAME)

    assert started == SegmentEvent(speech_started=True)
    assert following == SegmentEvent()


def test_silence_reaching_threshold_emits_one_speech_ended(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(
        segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME
    )

    segmenter.accept(SPEECH_FRAME)
    before_threshold = segmenter.accept(SILENCE_FRAME)
    ended = segmenter.accept(SILENCE_FRAME)
    after_end = segmenter.accept(SILENCE_FRAME)

    assert before_threshold == SegmentEvent()
    assert ended == SegmentEvent(speech_ended=True)
    assert after_end == SegmentEvent()


def test_speech_after_partial_silence_prevents_early_end(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=60)
    monkeypatch.setattr(
        segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME
    )

    segmenter.accept(SPEECH_FRAME)
    segmenter.accept(SILENCE_FRAME)
    resumed = segmenter.accept(SPEECH_FRAME)
    not_ended = segmenter.accept(SILENCE_FRAME)

    assert resumed == SegmentEvent()
    assert not_ended == SegmentEvent()


def test_leading_silence_never_emits_speech_ended(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: False)

    events = [segmenter.accept(SILENCE_FRAME) for _ in range(4)]

    assert events == [SegmentEvent(), SegmentEvent(), SegmentEvent(), SegmentEvent()]


def test_rejects_non_20ms_pcm_frames():
    segmenter = SpeechSegmenter()

    with pytest.raises(ValueError, match="20 ms"):
        segmenter.accept(b"\x00" * 639)


def test_rejects_non_16khz_sample_rate():
    with pytest.raises(ValueError, match="16000"):
        SpeechSegmenter(sample_rate=44100)


def test_reset_clears_speech_state(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(
        segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME
    )

    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_started=True)
    segmenter.reset()

    assert segmenter.accept(SILENCE_FRAME) == SegmentEvent()
    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_started=True)


def test_continuous_voice_is_split_at_max_duration_and_next_frame_restarts(monkeypatch):
    segmenter = SpeechSegmenter(max_segment_ms=60)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: True)

    events = [segmenter.accept(SPEECH_FRAME) for _ in range(20)]

    assert [number for number, event in enumerate(events) if event.speech_started] == [0, 3, 6, 9, 12, 15, 18]
    assert [number for number, event in enumerate(events) if event.speech_ended] == [2, 5, 8, 11, 14, 17]


def test_default_segment_duration_is_sixty_seconds(monkeypatch):
    segmenter = SpeechSegmenter()
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: True)

    assert segmenter.max_segment_ms == 60_000
    for _ in range(2999):
        assert segmenter.accept(SPEECH_FRAME).speech_ended is False
    assert segmenter.accept(SPEECH_FRAME).speech_ended is True
    assert segmenter.accept(SPEECH_FRAME).speech_started is True


def test_duration_limit_includes_trailing_silence_and_rounds_down_to_frame(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=1000, max_segment_ms=55)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME)

    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_started=True)
    assert segmenter.accept(SILENCE_FRAME) == SegmentEvent(speech_ended=True)
    assert segmenter.accept(SILENCE_FRAME) == SegmentEvent()


def test_reset_clears_segment_duration_counter(monkeypatch):
    segmenter = SpeechSegmenter(max_segment_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: True)
    segmenter.accept(SPEECH_FRAME)

    segmenter.reset()

    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_started=True)
    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_ended=True)


def test_one_frame_limit_emits_start_and_end_without_losing_the_frame(monkeypatch):
    segmenter = SpeechSegmenter(max_segment_ms=20)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: True)

    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(speech_started=True, speech_ended=True)
    assert segmenter.accept(SPEECH_FRAME) == SegmentEvent(
        speech_started=True, speech_ended=True, continued=True,
    )


def test_next_forced_split_start_marks_continuation_for_runtime_preroll(monkeypatch):
    segmenter = SpeechSegmenter(max_segment_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME)

    first = segmenter.accept(SPEECH_FRAME)
    ended = segmenter.accept(SPEECH_FRAME)
    continued = segmenter.accept(SPEECH_FRAME)

    assert first.continued is False
    assert ended.continued is False
    assert continued.speech_started is True
    assert continued.continued is True
    assert segmenter.accept(SPEECH_FRAME).continued is False


def test_silence_or_explicit_reset_clears_pending_continuation(monkeypatch):
    segmenter = SpeechSegmenter(max_segment_ms=20)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == SPEECH_FRAME)
    segmenter.accept(SPEECH_FRAME)
    segmenter.accept(SILENCE_FRAME)

    assert segmenter.accept(SPEECH_FRAME).continued is False
    segmenter.reset()
    assert segmenter.accept(SPEECH_FRAME).continued is False


@pytest.mark.parametrize("limit", [0, -20, 19, True, 40.5, "60000", None])
def test_rejects_invalid_max_segment_duration(limit):
    with pytest.raises(ValueError, match="max_segment_ms"):
        SpeechSegmenter(max_segment_ms=limit)
