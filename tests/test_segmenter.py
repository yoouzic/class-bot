from classbot.audio.segmenter import SegmentEvent, SpeechSegmenter


def test_first_speech_frame_emits_speech_started(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == b"speech")

    started = segmenter.accept(b"speech")
    following = segmenter.accept(b"speech")

    assert started == SegmentEvent(speech_started=True)
    assert following == SegmentEvent()


def test_silence_reaching_threshold_emits_one_speech_ended(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == b"speech")

    segmenter.accept(b"speech")
    before_threshold = segmenter.accept(b"silence")
    ended = segmenter.accept(b"silence")
    after_end = segmenter.accept(b"silence")

    assert before_threshold == SegmentEvent()
    assert ended == SegmentEvent(speech_ended=True)
    assert after_end == SegmentEvent()


def test_speech_after_partial_silence_prevents_early_end(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=60)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: frame == b"speech")

    segmenter.accept(b"speech")
    segmenter.accept(b"silence")
    resumed = segmenter.accept(b"speech")
    not_ended = segmenter.accept(b"silence")

    assert resumed == SegmentEvent()
    assert not_ended == SegmentEvent()


def test_leading_silence_never_emits_speech_ended(monkeypatch):
    segmenter = SpeechSegmenter(end_silence_ms=40)
    monkeypatch.setattr(segmenter, "_is_speech", lambda frame: False)

    events = [segmenter.accept(b"silence") for _ in range(4)]

    assert events == [SegmentEvent(), SegmentEvent(), SegmentEvent(), SegmentEvent()]
