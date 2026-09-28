from __future__ import annotations

import math
from dataclasses import dataclass

import webrtcvad

SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    speech_started: bool = False
    speech_ended: bool = False


class SpeechSegmenter:
    def __init__(
        self, sample_rate: int = SAMPLE_RATE, end_silence_ms: int = 320
    ):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("sample_rate must be 16000")
        if end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive")

        self.sample_rate = sample_rate
        self.end_silence_ms = end_silence_ms
        self._end_silence_frames = max(1, math.ceil(end_silence_ms / 20))
        self._frame_bytes = sample_rate * 20 // 1000 * 2
        self._vad = webrtcvad.Vad(2)
        self.reset()

    def _is_speech(self, frame: bytes) -> bool:
        return self._vad.is_speech(frame, self.sample_rate)

    def accept(self, frame: bytes) -> SegmentEvent:
        if len(frame) != self._frame_bytes:
            raise ValueError(
                "frame must contain exactly one 20 ms PCM frame "
                f"({self._frame_bytes} bytes)"
            )

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

    def reset(self) -> None:
        self._in_speech = False
        self._silence_frames = 0
