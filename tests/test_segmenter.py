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
