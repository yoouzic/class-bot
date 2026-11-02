"""Keep warm ASR startup local without changing the selected model or device."""
from __future__ import annotations

import json
import sys
from types import ModuleType

import pytest

from classbot.asr import funasr_streaming

MODEL_ID = "installed-package/streaming-model"


@pytest.fixture(autouse=True)
def _fresh_model_cache():
    funasr_streaming.clear_model_cache()
    yield
    funasr_streaming.clear_model_cache()


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


def dependencies(monkeypatch, cached, *, load_error=None, model=None):
    calls = {"snapshot": [], "model": []}
    model = object() if model is None else model

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
    names.name_maps_ms = {"paraformer-zh-streaming": MODEL_ID}
    hub = ModuleType("modelscope.hub.snapshot_download")
    hub.snapshot_download = resolve
    for name, module in (("funasr", funasr), (names.__name__, names), (hub.__name__, hub)):
        monkeypatch.setitem(sys.modules, name, module)
    return calls, model


def test_complete_cached_artifacts_use_local_only_resolution_and_same_device(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, model = dependencies(monkeypatch, root)
    assert funasr_streaming.build_model("cuda:0") is model
    assert calls["snapshot"] == [(MODEL_ID, {"revision": "master", "local_files_only": True})]
    assert calls["model"] == [{"model": str(root), "device": "cuda:0", "disable_update": True, "check_latest": False}]


def test_missing_cache_preserves_alias_first_download_behavior_without_update_poll(monkeypatch):
    calls, model = dependencies(monkeypatch, FileNotFoundError("not cached"))
    assert funasr_streaming.build_model("cpu") is model
    assert calls["model"] == [{"model": "paraformer-zh-streaming", "device": "cpu", "disable_update": True}]


@pytest.mark.parametrize("damage", ["missing_weight", "empty_weight", "missing_tokenizer", "wrong_repository", "invalid_metadata", "escaping_path"])
def test_incomplete_or_wrong_cache_is_never_selected_as_a_usable_local_model(tmp_path, monkeypatch, damage):
    root = snapshot(tmp_path)
    metadata_path = root / "configuration.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if damage == "missing_weight":
        (root / "model.pt").unlink()
    elif damage == "empty_weight":
        (root / "model.pt").write_bytes(b"")
    elif damage == "missing_tokenizer":
        (root / "tokens.json").unlink()
    elif damage == "wrong_repository":
        metadata["model_name_in_hub"]["ms"] = "another/model"
    elif damage == "invalid_metadata":
        metadata_path.write_text("not json", encoding="utf-8")
    else:
        metadata["file_path_metas"]["init_param"] = "../outside.pt"
        (tmp_path / "outside.pt").write_bytes(b"unrelated checkpoint")
    if damage in {"wrong_repository", "escaping_path"}:
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls, _model = dependencies(monkeypatch, root)
    funasr_streaming.build_model("cpu")
    assert calls["model"] == [{"model": "paraformer-zh-streaming", "device": "cpu", "disable_update": True}]


def test_actual_local_checkpoint_load_failure_propagates_without_switching_weights(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, _model = dependencies(monkeypatch, root, load_error=RuntimeError("checkpoint corrupt"))
    with pytest.raises(RuntimeError, match="checkpoint corrupt"):
        funasr_streaming.build_model("cpu")
    assert len(calls["model"]) == 1
    assert calls["model"][0]["model"] == str(root)


@pytest.mark.parametrize("sequence_type", [list, tuple], ids=["list", "tuple"])
@pytest.mark.parametrize("present", [False, True], ids=["missing_nested_file", "complete_nested_files"])
def test_nested_sequence_artifacts_are_checked_before_selecting_the_local_model(
    tmp_path, monkeypatch, sequence_type, present,
):
    root = snapshot(tmp_path)
    metadata_path = root / "configuration.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["file_path_metas"]["tokenizer_conf"]["extra_files"] = sequence_type([
        "tokens.json", {"vocabulary": sequence_type(["additional-tokens.json"])},
    ])
    if present:
        (root / "additional-tokens.json").write_text('["vocabulary"]', encoding="utf-8")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    calls, _model = dependencies(monkeypatch, root)
    if sequence_type is tuple:
        # JSON normally supplies lists; cover the tuple shape also accepted by
        # FunASR's own recursive metadata helper without changing real files.
        monkeypatch.setattr(funasr_streaming.json, "loads", lambda _value: metadata)
    funasr_streaming.build_model("cpu")
    assert calls["model"][0]["model"] == (str(root) if present else "paraformer-zh-streaming")


class WarmupRecordingModel:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return [{"text": ""}]


def test_built_streaming_model_is_warmed_with_one_silent_chunk(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    recording = WarmupRecordingModel()
    _calls, _model = dependencies(monkeypatch, root, model=recording)
    funasr_streaming.build_model("cpu")

    warmup = recording.calls[0]
    assert warmup["is_final"] is False
    assert warmup["input"].size == funasr_streaming._CHUNK_SAMPLES
    assert warmup["cache"] == {}


def test_model_without_generate_interface_skips_warmup(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    _calls, model = dependencies(monkeypatch, root)
    assert funasr_streaming.build_model("cpu") is model


def test_model_cache_reuses_the_warm_model_across_builds(tmp_path, monkeypatch):
    root = snapshot(tmp_path)
    calls, _model = dependencies(monkeypatch, root)

    first = funasr_streaming.build_model("cpu")
    assert funasr_streaming.build_model("cpu") is first
    assert len(calls["model"]) == 1

    funasr_streaming.clear_model_cache()
    funasr_streaming.build_model("cpu")
    assert len(calls["model"]) == 2
