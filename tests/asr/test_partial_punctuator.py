"""Realtime partial punctuation follows the official 2pass pattern."""
from __future__ import annotations

import sys
from types import ModuleType

import pytest

from classbot.asr import partial_punctuator
from classbot.asr.partial_punctuator import PartialPunctuator


class FakeGenerateModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"text": self.text}]


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    partial_punctuator.clear_model_cache()
    yield
    partial_punctuator.clear_model_cache()


def test_build_punctuator_model_uses_the_small_alias_and_caches(monkeypatch):
    calls = []

    def construct(**kwargs):
        calls.append(kwargs)
        return object()

    funasr = ModuleType("funasr")
    funasr.AutoModel = construct
    monkeypatch.setitem(sys.modules, "funasr", funasr)

    model = partial_punctuator.build_punctuator_model("cuda:0")
    assert calls == [
        {"model": "ct-punc-c", "device": "cuda:0", "disable_update": True}
    ]
    assert partial_punctuator.build_punctuator_model("cuda:0") is model


def test_punctuate_returns_model_text():
    model = FakeGenerateModel("你好，世界。")
    punctuator = PartialPunctuator(device="cpu")
    punctuator._model = model

    assert punctuator.punctuate("你好世界") == "你好，世界。"
    assert model.calls[0]["input"] == "你好世界"


def test_punctuate_empty_input_skips_the_model():
    model = FakeGenerateModel("文本")
    punctuator = PartialPunctuator(device="cpu")
    punctuator._model = model

    assert punctuator.punctuate("   ") == ""
    assert model.calls == []


def test_punctuate_falls_back_to_input_when_model_output_is_blank():
    class BlankModel:
        def generate(self, **kwargs):
            return [{"text": ""}]

    punctuator = PartialPunctuator(device="cpu")
    punctuator._model = BlankModel()

    assert punctuator.punctuate("原始文本") == "原始文本"


def test_punctuate_before_load_raises():
    punctuator = PartialPunctuator(device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        punctuator.punctuate("文本")


def test_load_is_idempotent(monkeypatch):
    builds = []

    class NoopModel:
        def generate(self, **kwargs):
            return []

    def fake_build(device):
        builds.append(device)
        return NoopModel()

    monkeypatch.setattr(partial_punctuator, "build_punctuator_model", fake_build)
    punctuator = PartialPunctuator(device="cpu")
    punctuator.load()
    punctuator.load()

    assert builds == ["cpu"]
    assert punctuator.loaded
