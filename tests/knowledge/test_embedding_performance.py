"""Device routing at the public encoder boundary, without CUDA test hardware."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from classbot.knowledge.embedding import LocalEmbedder


def install_device_doubles(monkeypatch):
    import torch

    requested = []
    loaded = []
    allocations = []

    class TokenTensor:
        def __init__(self, name, value):
            self.name = name
            self.value = value

        def to(self, device):
            requested.append((self.name, device))
            return self.value

    class Tokenizer:
        model_input_names = ("input_ids", "attention_mask", "token_type_ids")

        @staticmethod
        def from_pretrained(_name, **_kwargs):
            return Tokenizer()

        def __call__(self, texts, **_kwargs):
            assert len(texts) == 2
            return {"input_ids": [[1] * 446 + [2] * 65, [3]]}

        def num_special_tokens_to_add(self, pair=False):
            return 2

        def prepare_for_model(self, ids, **_kwargs):
            values = [101, *ids, 102]
            return {
                "input_ids": values, "attention_mask": [1] * len(values),
                "token_type_ids": [0] * len(values),
            }

        def pad(self, features, **_kwargs):
            width = max(len(item["input_ids"]) for item in features)
            values = {name: torch.tensor([
                item[name] + [0] * (width - len(item[name])) for item in features
            ]) for name in self.model_input_names}
            return {name: TokenTensor(name, value) for name, value in values.items()}

    class Model:
        @staticmethod
        def from_pretrained(_name, **kwargs):
            loaded.append(kwargs)
            return Model()

        def to(self, device):
            requested.append(("model", device))
            return self

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, token_type_ids):
            assert input_ids.device == attention_mask.device == token_type_ids.device
            marker = input_ids[:, 1].to(dtype=torch.float64)
            cls = torch.stack([marker, torch.ones_like(marker), marker * 0], dim=1)
            return SimpleNamespace(last_hidden_state=cls[:, None, :])

    original_zeros = torch.zeros
    original_to = torch.Tensor.to

    def tensor_to(value, *args, **kwargs):
        if args and isinstance(args[0], str):
            requested.append(("overflow_to_sample_mapping", args[0]))
            return value
        return original_to(value, *args, **kwargs)

    def zeros(*args, **kwargs):
        device = kwargs.pop("device", None)
        if device is not None:
            allocations.append(device)
        # Test tensors stay on CPU; the requested target is checked separately.
        return original_zeros(*args, **kwargs)

    monkeypatch.setattr(torch, "zeros", zeros)
    monkeypatch.setattr(torch.Tensor, "to", tensor_to)
    monkeypatch.setitem(
        sys.modules, "transformers",
        SimpleNamespace(AutoModel=Model, AutoTokenizer=Tokenizer),
    )
    return requested, loaded, allocations


@pytest.mark.parametrize("device", [None, "cpu", "cuda", "cuda:1"])
def test_encoder_routes_model_all_tokens_and_pooling_to_explicit_device(
    tmp_path, monkeypatch, device,
):
    requested, loaded, allocations = install_device_doubles(monkeypatch)
    kwargs = {} if device is None else {"device": device}
    embedder = LocalEmbedder(cache_dir=tmp_path, **kwargs)
    target = device or "cpu"
    assert embedder.device == target

    vectors = embedder.encode(["first", "second"])

    assert set(requested) == {
        ("model", target),
        ("input_ids", target),
        ("attention_mask", target),
        ("token_type_ids", target),
        ("overflow_to_sample_mapping", target),
    }
    assert allocations == [target]
    assert loaded[0]["local_files_only"] is True
    assert vectors.dtype == np.float32
    assert vectors.shape == (2, 3)
    windows = np.array([[1, 1, 0], [2, 1, 0], [3, 1, 0]], dtype=np.float64)
    windows /= np.linalg.norm(windows, axis=1, keepdims=True)
    expected = np.stack([windows[0] + windows[1], windows[2]])
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    np.testing.assert_allclose(vectors, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("device", ["auto", "mps", "cpu:0", "cuda:", "cuda:-1", "cuda:1:2"])
def test_encoder_rejects_unsupported_device_without_loading(tmp_path, device):
    with pytest.raises(ValueError, match="device"):
        LocalEmbedder(cache_dir=tmp_path, device=device)


def test_unsupported_tokenizer_fails_explicitly_instead_of_truncating(tmp_path):
    embedder = LocalEmbedder(cache_dir=tmp_path)
    embedder._tokenizer = SimpleNamespace()
    embedder._model = object()

    with pytest.raises(ValueError, match="tokenizer needs"):
        embedder.encode(["text"])


@pytest.mark.parametrize("texts", [
    ["前" * 900 + "尾部结论"],
    ["short", "unaffordable configurations contradict expectations. " * 80 + "tail", "甲" * 1000],
])
def test_real_local_tokenizer_preserves_every_token_in_explicit_windows(tmp_path, texts):
    """Catch real backend overflow loss; cached tokenizer files are sufficient."""
    import torch

    transformers = pytest.importorskip("transformers")
    from classbot.knowledge.embedding import DEFAULT_MODEL, DEFAULT_MODEL_REVISION

    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            DEFAULT_MODEL, revision=DEFAULT_MODEL_REVISION,
            cache_dir=str(Path(__file__).resolve().parents[2] / ".cache/classbot/models"),
            trust_remote_code=False, local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned local tokenizer is not cached")
    raw = tokenizer(texts, add_special_tokens=False, truncation=False, padding=False)["input_ids"]
    expected = []
    for ids in raw:
        for start in range(0, max(1, len(ids)), 446):
            expected.append(ids[start:start + 510])
            if start + 510 >= len(ids):
                break
    observed = []

    class Model:
        def __call__(self, input_ids, attention_mask, token_type_ids):
            for ids, mask, types in zip(input_ids, attention_mask, token_type_ids):
                active = ids[mask.bool()].tolist()
                assert active[0] == tokenizer.cls_token_id
                assert active[-1] == tokenizer.sep_token_id
                assert not types.any()
                observed.append(active[1:-1])
            return SimpleNamespace(last_hidden_state=torch.ones((len(input_ids), 1, 512)))

    embedder = LocalEmbedder(cache_dir=tmp_path)
    embedder._tokenizer = tokenizer
    embedder._model = Model()

    vectors = embedder.encode(texts)

    assert observed == expected
    assert vectors.shape == (len(texts), 512)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
