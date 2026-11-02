import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex, LocalEmbedder


class EqualEmbedder:
    def __init__(self):
        self.calls = 0

    def encode(self, texts, *, query=False):
        self.calls += 1
        return np.ones((len(texts), 3), dtype=np.float32)


class _CharacterTokenizer:
    model_input_names = ("input_ids", "attention_mask", "token_type_ids")

    def __call__(self, texts, **_kwargs):
        return {"input_ids": [[ord(character) for character in text] for text in texts]}

    def num_special_tokens_to_add(self, pair=False):
        assert pair is False
        return 2

    def prepare_for_model(self, ids, **_kwargs):
        values = [101, *ids, 102]
        return {"input_ids": values, "attention_mask": [1] * len(values),
                "token_type_ids": [0] * len(values)}

    def pad(self, features, **_kwargs):
        import torch

        maximum = max(len(feature["input_ids"]) for feature in features)
        return {key: torch.tensor([feature[key] + [0] * (maximum - len(feature[key]))
                                   for feature in features])
                for key in self.model_input_names}


def test_content_hash_detects_edit_even_with_unchanged_size_and_mtime(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    source = root / "notes.md"
    source.write_text("aaaa", encoding="utf-8")
    initial = source.stat()
    cache = tmp_path / "cache"
    first = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    source.write_text("bbbb", encoding="utf-8")
    os.utime(source, ns=(initial.st_atime_ns, initial.st_mtime_ns))
    second = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    assert second.revision != first.revision
    assert second.search("bbbb")[0].text == "bbbb"


def test_additions_deletions_and_nested_cache_are_stale_safe(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")
    cache = root / "local-cache"
    first = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    assert not first.is_stale()
    assert first.report.indexed_files == 1
    (root / "new.txt").write_text("新材料", encoding="utf-8")
    assert first.is_stale()
    second = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    assert second.report.indexed_files == 2
    assert not second.is_stale()
    (root / "notes.txt").unlink()
    assert second.is_stale()


def test_schema_and_local_model_metadata_changes_invalidate_cache(tmp_path, monkeypatch):
    from classbot.knowledge import index as index_module
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")
    model = tmp_path / "local-model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"dummy weights")
    cache = tmp_path / "cache"
    first = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder(), model_name=str(model))
    weights.write_bytes(b"new dummy weights")
    embedder = EqualEmbedder()
    second = KnowledgeIndex.build(root, cache, embedder=embedder, model_name=str(model))
    assert first.revision != second.revision
    assert embedder.calls == 1
    monkeypatch.setattr(index_module, "SCHEMA_VERSION", index_module.SCHEMA_VERSION + 1)
    third = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder(), model_name=str(model))
    assert third.revision != second.revision


def test_atomic_cache_write_failure_keeps_previous_snapshot(tmp_path, monkeypatch):
    from classbot.knowledge import index as index_module
    root = tmp_path / "docs"
    root.mkdir()
    source = root / "notes.txt"
    source.write_text("before", encoding="utf-8")
    cache = tmp_path / "cache"
    KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    stored = next(cache.glob("*.json"))
    previous = stored.read_bytes()
    source.write_text("after", encoding="utf-8")

    def deny_replace(*_args):
        raise OSError("replacement denied")

    monkeypatch.setattr(index_module.os, "replace", deny_replace)
    with pytest.raises(OSError, match="replacement denied"):
        KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    assert stored.read_bytes() == previous
    assert not list(cache.glob("*.tmp"))


def test_invalid_vector_dimensions_do_not_silently_fallback(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")

    class Mismatch:
        def encode(self, texts, *, query=False):
            return np.ones((len(texts), 4 if query else 3), dtype=np.float32)

    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=Mismatch())
    with pytest.raises(ValueError, match="dimension"):
        index.search("材料")


def test_local_embedder_is_lazy_safe_cpu_and_query_instructed(tmp_path, monkeypatch):
    import torch

    from classbot.knowledge.embedding import QUERY_INSTRUCTION
    calls = []

    class Tokenizer(_CharacterTokenizer):
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("tokenizer_load", name, kwargs))
            return Tokenizer()

        def __call__(self, texts, **kwargs):
            calls.append(("tokenize", texts, kwargs))
            return super().__call__(texts, **kwargs)

    class Model:
        @staticmethod
        def from_pretrained(name, **kwargs):
            calls.append(("model_load", name, kwargs))
            return Model()

        def to(self, device):
            calls.append(("device", device))
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            count = kwargs["input_ids"].shape[0]
            return SimpleNamespace(last_hidden_state=torch.ones((count, 4, 3)))

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Model, AutoTokenizer=Tokenizer))
    embedder = LocalEmbedder(cache_dir=tmp_path / "models")
    assert calls == []
    vectors = embedder.encode(["为什么"], query=True)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert next(call for call in calls if call[0] == "model_load")[2]["trust_remote_code"] is False
    assert next(call for call in calls if call[0] == "model_load")[2]["use_safetensors"] is True
    assert next(call for call in calls if call[0] == "model_load")[2]["local_files_only"] is True
    for kind in ("model_load", "tokenizer_load"):
        assert next(call for call in calls if call[0] == kind)[2]["revision"] == "7999e1d3359715c523056ef9478215996d62a620"
    assert ("device", "cpu") in calls
    tokenizing = next(call for call in calls if call[0] == "tokenize")
    assert tokenizing[1] == [QUERY_INSTRUCTION + "为什么"]
    assert tokenizing[2]["truncation"] is False
    assert tokenizing[2]["add_special_tokens"] is False


def test_local_embedder_encodes_tail_overflow_windows_instead_of_truncating(tmp_path, monkeypatch):
    import torch

    class Tokenizer(_CharacterTokenizer):
        @staticmethod
        def from_pretrained(_name, **_kwargs):
            return Tokenizer()

    class Model:
        @staticmethod
        def from_pretrained(_name, **_kwargs):
            return Model()

        def to(self, _device):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            tail = (kwargs["input_ids"] == ord("尾")).any(dim=1).float()
            cls = torch.stack([1 - tail, tail, torch.zeros_like(tail)], dim=1)
            return SimpleNamespace(last_hidden_state=cls[:, None, :])

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoModel=Model, AutoTokenizer=Tokenizer))
    embedder = LocalEmbedder(cache_dir=tmp_path / "models")
    baseline, with_tail = embedder.encode(["前" * 904, "前" * 900 + "尾部结论"])
    assert baseline[1] == 0
    assert with_tail[1] > 0.3
    assert not np.allclose(baseline, with_tail)
    assert np.allclose(np.linalg.norm([baseline, with_tail], axis=1), 1)


def test_model_commit_is_in_cache_identity_and_unpinned_hosted_models_fail(tmp_path, monkeypatch):
    from classbot.knowledge import embedding
    root = tmp_path / "docs"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")
    cache = tmp_path / "cache"
    first = KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    payload = json.loads(next(cache.glob("*.json")).read_text(encoding="utf-8"))
    assert payload["model"]["revision"] == "7999e1d3359715c523056ef9478215996d62a620"
    monkeypatch.setattr(embedding, "DEFAULT_MODEL_REVISION", "b" * 40)
    second_embedder = EqualEmbedder()
    second = KnowledgeIndex.build(root, cache, embedder=second_embedder)
    assert second.revision != first.revision
    assert second_embedder.calls == 1
    with pytest.raises(ValueError, match="immutable commit"):
        KnowledgeIndex.build(root, cache, model_name="example/unpinned", embedder=EqualEmbedder())


def test_credential_name_variants_are_not_indexed(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    for name in ("credentials.local.json", "service-account.json", "client-secret-prod.json"):
        (root / name).write_text(json.dumps({"key": "secret material"}), encoding="utf-8")
    (root / "good.md").write_text("普通材料", encoding="utf-8")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    assert index.report.indexed_files == 1


def test_credential_directories_are_excluded_but_ordinary_private_docs_remain(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    credential_names = (
        "secrets", "credentials", "private_keys", "private-keys", "ssh_keys",
        "credentials.local", "secrets-prod",
    )
    for name in credential_names:
        folder = root / name
        folder.mkdir()
        (folder / "config.json").write_text('{"secret": "do not index"}', encoding="utf-8")
    private_docs = root / "private"
    private_docs.mkdir()
    (private_docs / "notes.md").write_text("项目实验说明", encoding="utf-8")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    assert index.report.indexed_files == 1
    assert index.report.skipped_files == len(credential_names)
    assert index.search("实验")[0].source.startswith("private/notes.md")
    assert all("do not index" not in hit.text for hit in index.search("secret", limit=20))


@pytest.mark.skipif(os.name != "nt", reason="Windows hidden-file attributes")
def test_windows_hidden_attribute_files_and_folders_are_excluded(tmp_path):
    import ctypes
    root = tmp_path / "docs"
    root.mkdir()
    hidden = root / "hidden.txt"
    hidden.write_text("private hidden material", encoding="utf-8")
    folder = root / "hidden-folder"
    folder.mkdir()
    (folder / "notes.md").write_text("private folder material", encoding="utf-8")
    assert ctypes.windll.kernel32.SetFileAttributesW(str(hidden), 0x2)
    assert ctypes.windll.kernel32.SetFileAttributesW(str(folder), 0x2)
    (root / "visible.txt").write_text("普通材料", encoding="utf-8")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    assert index.report.indexed_files == 1
    assert index.report.skipped_files == 2


def test_oversize_cache_fails_without_replacing_previous_valid_cache(tmp_path, monkeypatch):
    from classbot.knowledge import index as index_module
    root = tmp_path / "docs"
    root.mkdir()
    source = root / "notes.txt"
    source.write_text("before", encoding="utf-8")
    cache = tmp_path / "cache"
    KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    stored = next(cache.glob("*.json"))
    previous = stored.read_bytes()
    source.write_text("after", encoding="utf-8")
    monkeypatch.setattr(index_module, "MAX_CACHE_BYTES", 50)
    with pytest.raises(ValueError, match="cache size"):
        KnowledgeIndex.build(root, cache, embedder=EqualEmbedder())
    assert stored.read_bytes() == previous
    assert not list(cache.glob("*.tmp"))
