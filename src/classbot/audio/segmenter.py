from __future__ import annotations

import math
from dataclasses import dataclass

import webrtcvad

SAMPLE_RATE = 16_000


@dataclass(frozen=True, slots=True)
class SegmentEvent:
    speech_started: bool = False
    speech_ended: bool = False
    continued: bool = False


class SpeechSegmenter:
    def __init__(
        self, sample_rate: int = SAMPLE_RATE, end_silence_ms: int = 320,
        max_segment_ms: int = 60_000,
    ):
        if sample_rate != SAMPLE_RATE:
            raise ValueError("sample_rate must be 16000")
        if end_silence_ms <= 0:
            raise ValueError("end_silence_ms must be positive")
        if (not isinstance(max_segment_ms, int) or isinstance(max_segment_ms, bool)
                or max_segment_ms < 20):
            raise ValueError("max_segment_ms must be an integer of at least 20")

        self.sample_rate = sample_rate
        self.end_silence_ms = end_silence_ms
        self.max_segment_ms = max_segment_ms
        self._end_silence_frames = max(1, math.ceil(end_silence_ms / 20))
        self._max_segment_frames = max_segment_ms // 20
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

        voiced = self._is_speech(frame)
        started = voiced and not self._in_speech
        continued = started and self._continued_start
        if started:
            self._in_speech = True
            self._continued_start = False
        if not self._in_speech:
            self._continued_start = False
            return SegmentEvent()

        self._segment_frames += 1
        self._silence_frames = 0 if voiced else self._silence_frames + 1
        limit_reached = self._segment_frames >= self._max_segment_frames
        if self._silence_frames >= self._end_silence_frames or limit_reached:
            self.reset()
            self._continued_start = limit_reached and voiced
            return SegmentEvent(speech_started=started, speech_ended=True, continued=continued)

        return SegmentEvent(speech_started=started, continued=continued)

    def reset(self) -> None:
        self._in_speech = False
        self._silence_frames = 0
        self._segment_frames = 0
        self._continued_start = False
