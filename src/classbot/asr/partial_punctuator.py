from __future__ import annotations

import logging
import threading
from typing import Any

from classbot.asr import funasr_streaming

_LOGGER = logging.getLogger(__name__)

# Small Chinese punctuation model (~290MB): 20 chars ≈ 9ms, 94 chars ≈ 58ms
# on the RTX 5070, far below the one-partial-per-1-3s cadence.
_PUNC_ALIAS = "ct-punc-c"

_MODEL_CACHE: dict[str, Any] = {}


def clear_model_cache() -> None:
    """Drop warm punctuator models so a later build starts fresh (tests)."""
    _MODEL_CACHE.clear()


def build_punctuator_model(device: str) -> Any:
    warm = _MODEL_CACHE.get(device)
    if warm is not None:
        return warm
    from funasr import AutoModel

    model = AutoModel(model=_PUNC_ALIAS, device=device, disable_update=True)
    _MODEL_CACHE[device] = model
    return model


class PartialPunctuator:
    """Restore punctuation on streaming partial text, best effort.

    The streaming recognizer emits unpunctuated text, so the question
    detector loses its strongest signal (？) and its clause boundaries on
    the live path. This re-punctuates each partial with the small
    ct-punc-c model — the same "streaming ASR + realtime punctuation"
    pattern as FunASR's official 2pass pipeline. One instance may be
    shared across sessions; ``punctuate`` is stateless.
    """

    def __init__(self, device: str):
        self.device = device
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            self._model = build_punctuator_model(self.device)

    def punctuate(self, text: str) -> str:
        model = self._model
        if model is None:
            raise RuntimeError("partial punctuator is not loaded")
        stripped = text.strip()
        if not stripped:
            return ""
        result = model.generate(input=stripped)
        punctuated = "".join(
            part for part in funasr_streaming._texts_from_result(result) if part
        ).strip()
        return punctuated or stripped
