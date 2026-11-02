"""Offline refinement model selection follows the same cache rules as streaming."""
from __future__ import annotations

import json
import sys
from types import ModuleType

import numpy as np
import pytest

from classbot.asr import offline_refiner
from classbot.asr.offline_refiner import OfflineRefiner

MODEL_ID = "installed-package/offline-model"


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    offline_refiner.clear_model_cache()
    yield
    offline_refiner.clear_model_cache()


def snapshot(tmp_path):
    root = tmp_path / "cached"
    root.mkdir()
    metadata = {"model_name_in_hub": {"ms": MODEL_ID}, "file_path_metas": {
        "init_param": "model.pt", "config": "config.yaml",
        "tokenizer_conf": {"token_list": "tokens.json"},
        "frontend_conf": {"cmvn_file": "am.mvn"}}}
    (root / "configuration.json").write_text(json.dumps(metadata), encoding="utf-8")
    for name in ("model.pt", "config.yaml", "tokens.json", "am.mvn"):
        (root / name).write_bytes(b"test artifact validated by injected AutoModel")
    return root


def dependencies(monkeypatch, cached, *, load_error=None):
    calls = {"snapshot": [], "model": []}
    model = object()

    def resolve(model_id, **kwargs):
        calls["snapshot"].append((model_id, kwargs))
        if isinstance(cached, Exception):
            raise cached
        return str(cached)

    def construct(**kwargs):
        calls["model"].append(kwargs)
        if load_error:
            raise load_error
        return model

    funasr = ModuleType("funasr")
    funasr.AutoModel = construct
    names = ModuleType("funasr.download.name_maps_from_hub")
    names.name_maps_ms = {"paraformer-zh": MODEL_ID}
    hub = ModuleType("modelscope.hub.snapshot_download")
    hub.snapshot_download = resolve
    for name, module in (("funasr", funasr), (names.__name__, names), (hub.__name__, hub)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls, model


class FakeGenerateModel:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"text": self.text}]


def test_complete_cached_artifacts_build_offline_model_with_punctuation(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, model = dependencies(monkeypatch, root)
    assert offline_refiner.build_refiner_model("cuda:0") is model
    assert calls["snapshot"] == [(MODEL_ID, {"revision": "master", "local_files_only": True})]
    assert calls["model"] == [{
        "model": str(root),
        "punc_model": "ct-punc",
        "device": "cuda:0",
        "disable_update": True,
        "check_latest": False,
    }]


def test_missing_cache_preserves_alias_first_download_behavior(monkeypatch):
    calls, model = dependencies(monkeypatch, FileNotFoundError("not cached"))
    assert offline_refiner.build_refiner_model("cpu") is model
    assert calls["model"] == [{
        "model": "paraformer-zh",
        "punc_model": "ct-punc",
        "device": "cpu",
        "disable_update": True,
    }]


def test_corrupt_checkpoint_load_failure_propagates_without_switching_weights(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, _model = dependencies(monkeypatch, root, load_error=RuntimeError("checkpoint corrupt"))
    with pytest.raises(RuntimeError, match="checkpoint corrupt"):
        offline_refiner.build_refiner_model("cpu")
    assert len(calls["model"]) == 1


def test_refine_passes_hotwords_as_one_offline_decode():
    model = FakeGenerateModel("主观唯心主义是。")
    refiner = OfflineRefiner(
        device="cpu", hotwords=(" 主观唯心主义 ", "", "剩余价值"),
    )
    refiner._model = model
    text = refiner.refine(np.zeros(1600, dtype=np.float32))

    assert text == "主观唯心主义是。"
    call = model.calls[0]
    assert call["is_final"] is True
    assert call["cache"] == {}
    assert call["hotword"] == "主观唯心主义 剩余价值"
    assert "chunk_size" not in call
    assert "encoder_chunk_look_back" not in call


def test_refine_without_hotwords_omits_the_parameter():
    model = FakeGenerateModel("文本")
    refiner = OfflineRefiner(device="cpu")
    refiner._model = model
    assert refiner.refine(np.zeros(16, dtype=np.float32)) == "文本"
    assert "hotword" not in model.calls[0]


def test_refine_joins_multiple_result_texts_and_drops_blanks():
    class MultiTextModel:
        def __init__(self):
            self.calls = []

        def generate(self, **kwargs):
            self.calls.append(kwargs)
            return [{"text": "第一句。"}, {"text": ""}, {"text": "第二句。"}]

    model = MultiTextModel()
    refiner = OfflineRefiner(device="cpu")
    refiner._model = model
    assert refiner.refine(np.zeros(16, dtype=np.float32)) == "第一句。第二句。"


def test_refine_before_load_raises():
    refiner = OfflineRefiner(device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        refiner.refine(np.zeros(16, dtype=np.float32))


def test_load_is_idempotent_and_only_builds_once(monkeypatch):
    builds = []

    class NoopModel:
        def generate(self, **kwargs):
            return []

    def fake_build(device):
        builds.append(device)
        return NoopModel()

    monkeypatch.setattr(offline_refiner, "build_refiner_model", fake_build)
    refiner = OfflineRefiner(device="cpu")
    refiner.load()
    refiner.load()

    assert builds == ["cpu"]
    assert refiner.loaded


def test_load_warms_up_the_model_before_marking_loaded(monkeypatch):
    model = FakeGenerateModel("")
    monkeypatch.setattr(offline_refiner, "build_refiner_model", lambda _device: model)
    refiner = OfflineRefiner(device="cpu", hotwords=("剩余价值",))
    refiner.load()

    assert refiner.loaded
    warmup = model.calls[0]
    assert warmup["input"].size == offline_refiner._WARM_UP_SAMPLES
    assert warmup["is_final"] is True
    assert warmup["hotword"] == "剩余价值"


def test_failed_warmup_leaves_the_refiner_unloaded(monkeypatch):
    class WarmupFailModel:
        def generate(self, **kwargs):
            if kwargs["input"].size == offline_refiner._WARM_UP_SAMPLES:
                raise RuntimeError("warmup failure")
            return [{"text": "文本"}]

    model = WarmupFailModel()
    monkeypatch.setattr(offline_refiner, "build_refiner_model", lambda _device: model)
    refiner = OfflineRefiner(device="cpu")

    with pytest.raises(RuntimeError, match="warmup failure"):
        refiner.load()

    assert not refiner.loaded


def test_refiner_model_cache_reuses_the_warm_model(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, _model = dependencies(monkeypatch, root)

    first = offline_refiner.build_refiner_model("cpu")
    assert offline_refiner.build_refiner_model("cpu") is first
    assert len(calls["model"]) == 1

    offline_refiner.clear_model_cache()
    offline_refiner.build_refiner_model("cpu")
    assert len(calls["model"]) == 2
