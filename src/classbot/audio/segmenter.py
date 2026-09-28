from __future__ import annotations

import math
from dataclasses import dataclass

import webrtcvad


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    speech_started: bool = False
    speech_ended: bool = False


class SpeechSegmenter:
    def __init__(self, sample_rate: int = 16000, end_silence_ms: int = 320):
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive")

        self.sample_rate = sample_rate
        self.end_silence_ms = end_silence_ms
        self._end_silence_frames = max(1, math.ceil(end_silence_ms / 20))
        self._vad = webrtcvad.Vad(2)
        self._in_speech = False
        self._silence_frames = 0

    def _is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, self.sample_rate)

    def accept(self, frame: bytes) -> SegmentEvent:
        if self._is_speech(frame):
            started = not self._in_speech
            self._in_speech = True
            self._silence_frames = 0
            return SegmentEvent(speech_started=started)

        if not self._in_speech:
            return SegmentEvent()

        self._silence_frames += 1
        if self._silence_frames >= self._end_silence_frames:
            self._in_speech = False
            self._silence_frames = 0
            return SegmentEvent(speech_ended=True)

        return SegmentEvent()
