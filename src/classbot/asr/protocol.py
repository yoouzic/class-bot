from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from classbot.domain import TranscriptEvent


class SpeechRecognizer(Protocol):
    async def start(self) -> None:
        ...

    def push_audio(self, frame: bytes) -> None:
        ...

    def finish_segment(self) -> None:
        ...

    def events(self) -> AsyncIterator[TranscriptEvent]:
        ...

    async def stop(self) -> None:
        ...
