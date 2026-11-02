from __future__ import annotations

import logging
import threading
from collections.abc import Iterable
from typing import Any

import numpy as np

from classbot.asr import funasr_streaming

_LOGGER = logging.getLogger(__name__)

# "paraformer-zh" resolves to the offline SeACo-Paraformer checkpoint: the
# non-streaming, hotword-aware member of the same Paraformer family used for
# streaming. "ct-punc" restores punctuation that the streaming path lacks.
_OFFLINE_ALIAS = "paraformer-zh"
_PUNC_ALIAS = "ct-punc"
_WARM_UP_SAMPLES = 8000


_MODEL_CACHE: dict[str, Any] = {}


def clear_model_cache() -> None:
    """Drop warm refiner models so a later build starts fresh (tests)."""
    _MODEL_CACHE.clear()


def build_refiner_model(device: str) -> Any:
    warm = _MODEL_CACHE.get(device)
    if warm is not None:
        # Same reasoning as the streaming cache: no session state inside the
        # model object, so one build serves every recognition session.
        return warm
    from funasr import AutoModel

    cached = funasr_streaming._cached_model_path(_OFFLINE_ALIAS)
    if cached is not None:
        # Same rule as the streaming model: a corrupt local load must fail
        # visibly, not swap weights.
        model = AutoModel(
            model=str(cached),
            punc_model=_PUNC_ALIAS,
            device=device,
            disable_update=True,
            check_latest=False,
        )
    else:
        model = AutoModel(
            model=_OFFLINE_ALIAS,
            punc_model=_PUNC_ALIAS,
            device=device,
            disable_update=True,
        )
    _MODEL_CACHE[device] = model
    return model


class OfflineRefiner:
    """Full-segment offline re-decode used for final transcript refinement.

    ``load`` is idempotent and safe from any thread, so one instance can be
    shared across recognition sessions; ``refine`` itself is stateless.
    """

    def __init__(self, device: str, hotwords: Iterable[str] = ()):
        self.device = device
        self.hotwords = tuple(
            word.strip() for word in hotwords if str(word).strip()
        )
        self._model: Any | None = None
        self._load_lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            model = build_refiner_model(self.device)
            # The first CUDA forward pass compiles kernels (seconds on some
            # GPUs). Absorb that here, in the background load, so the first
            # real refinement does not block the recognition worker.
            self._warm_up(model)
            self._model = model

    def _warm_up(self, model: Any) -> None:
        silence = np.zeros(_WARM_UP_SAMPLES, dtype=np.float32)
        kwargs: dict[str, Any] = {}
        if self.hotwords:
            kwargs["hotword"] = " ".join(self.hotwords)
        model.generate(input=silence, cache={}, is_final=True, **kwargs)

    def refine(self, samples: np.ndarray) -> str:
        model = self._model
        if model is None:
            raise RuntimeError("offline refiner is not loaded")

        kwargs: dict[str, Any] = {}
        if self.hotwords:
            kwargs["hotword"] = " ".join(self.hotwords)
        result = model.generate(input=samples, cache={}, is_final=True, **kwargs)
        return "".join(
            text
            for text in funasr_streaming._texts_from_result(result)
            if text.strip()
        ).strip()
