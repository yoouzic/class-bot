"""Bound model downloads before disk writes, without downloading real weights."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from classbot.knowledge import embedding


def asset(name, data):
    return SimpleNamespace(
        rfilename=name, size=len(data), blob_id=None,
        lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()),
    )


def install_hub(monkeypatch, files, *, bodies=None):
    calls = []
    manifest = SimpleNamespace(
        sha=embedding.DEFAULT_MODEL_REVISION,
        siblings=[asset(name, data) for name, data in files.items()],
    )

    class Api:
        def model_info(self, repo_id, **kwargs):
            calls.append(("manifest", repo_id, kwargs))
            return manifest

    class Response:
        def __init__(self, name):
            self.data = (bodies or files)[name]

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def raise_for_status(self):
            return None

        def iter_bytes(self, chunk_size):
            for start in range(0, len(self.data), 3):
                yield self.data[start:start + 3]

    class Client:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return None

        def stream(self, method, url):
            calls.append(("stream", url))
            return Response(url)

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        HfApi=Api, hf_hub_url=lambda _repo, name, **_kwargs: name, get_token=lambda: None,
    ))
    monkeypatch.setattr("httpx.Client", Client)
    return calls, manifest


def install_quota(monkeypatch, root, limit):
    def available(path):
        assert Path(path).resolve().is_relative_to(root.resolve())
        used = sum(item.stat().st_size for item in root.rglob("*") if item.is_file())
        return max(0, limit - used)

    def require(path, additional):
        if additional > available(path):
            raise RuntimeError("test quota exceeded")

    monkeypatch.setattr(embedding, "cache_available_bytes", available)
    monkeypatch.setattr(embedding, "require_cache_space", require)


def install_transformers(monkeypatch, *, legacy_missing=True):
    loads = []

    class Model:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append(("model", name, kwargs))
            if legacy_missing and name == embedding.DEFAULT_MODEL:
                raise OSError("not cached")
            return Model()

        def to(self, _device):
            return self

        def eval(self):
            return self

    class Tokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            loads.append(("tokenizer", name, kwargs))
            if legacy_missing and name == embedding.DEFAULT_MODEL:
                raise OSError("not cached")
            return Tokenizer()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=Model, AutoTokenizer=Tokenizer,
    ))
    return loads


def test_quota_preparation_preserves_cached_legacy_model_without_network(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 0)
    loads = install_transformers(monkeypatch, legacy_missing=False)
    embedder = embedding.LocalEmbedder(cache_dir=tmp_path)

    embedder.prepare()

    assert loads and all(item[2]["local_files_only"] is True for item in loads)


def test_quota_preflight_blocks_before_weight_download(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 3)
    install_transformers(monkeypatch)
    calls, _ = install_hub(monkeypatch, {"config.json": b"{}", "model.safetensors": b"weights"})

    with pytest.raises(RuntimeError, match="quota"):
        embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert not any(item[0] == "stream" for item in calls)
    assert not any(item.is_file() for item in tmp_path.rglob("*"))


@pytest.mark.parametrize("body", [b"longer than advertised", b"bad", b"wei"])
def test_model_stream_rejects_size_or_digest_and_removes_only_own_temp(tmp_path, monkeypatch, body):
    source = tmp_path / "original.txt"
    source.write_bytes(b"keep original")
    install_quota(monkeypatch, tmp_path, 200_000)
    install_transformers(monkeypatch)
    files = {"config.json": b"{}", "model.safetensors": b"weights"}
    calls, _ = install_hub(monkeypatch, files, bodies={**files, "model.safetensors": body})

    with pytest.raises(ValueError, match="size|digest"):
        embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert calls and source.read_bytes() == b"keep original"
    assert not list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob("classbot-snapshot.json"))


def test_complete_managed_model_reloads_offline_without_duplicate_weights(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 200_000)
    loads = install_transformers(monkeypatch)
    files = {
        "config.json": b"{}", "model.safetensors": b"weights",
        "tokenizer.json": b"{}", "onnx/model.onnx": b"excluded",
    }
    calls, _ = install_hub(monkeypatch, files)
    first = embedding.LocalEmbedder(cache_dir=tmp_path)
    first.prepare()
    snapshots = list(tmp_path.rglob("classbot-snapshot.json"))
    assert len(snapshots) == 1
    snapshot = snapshots[0].parent
    assert (snapshot / "model.safetensors").read_bytes() == b"weights"
    assert not list(tmp_path.rglob("*.onnx"))
    assert all(item[2]["local_files_only"] is True for item in loads)
    assert any(item[1] == str(snapshot) for item in loads)
    stream_count = sum(item[0] == "stream" for item in calls)

    embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert sum(item[0] == "stream" for item in calls) == stream_count
    assert len(list(tmp_path.rglob("model.safetensors"))) == 1


def test_sharded_snapshot_downloads_only_referenced_safetensors(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 200_000)
    install_transformers(monkeypatch)
    index = json.dumps({"weight_map": {"weight": "model-00001-of-00001.safetensors"}}).encode()
    files = {
        "config.json": b"{}", "model.safetensors.index.json": index,
        "model-00001-of-00001.safetensors": b"weights",
        "unused.safetensors": b"unused",
    }
    calls, _ = install_hub(monkeypatch, files)

    embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert not list(tmp_path.rglob("unused.safetensors"))
    assert sum(item == ("stream", "model.safetensors.index.json") for item in calls) == 1


def test_shard_paths_cannot_escape_the_managed_snapshot(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 200_000)
    install_transformers(monkeypatch)
    index = json.dumps({"weight_map": {"weight": "../original.safetensors"}}).encode()
    install_hub(monkeypatch, {
        "config.json": b"{}", "model.safetensors.index.json": index,
        "../original.safetensors": b"weights",
    })

    with pytest.raises(ValueError, match="path"):
        embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert not (tmp_path.parent / "original.safetensors").exists()


def test_hosted_snapshot_sha_must_match_pinned_revision(tmp_path, monkeypatch):
    install_quota(monkeypatch, tmp_path, 200_000)
    install_transformers(monkeypatch)
    calls, manifest = install_hub(monkeypatch, {"config.json": b"{}", "model.safetensors": b"weights"})
    manifest.sha = "0" * 40

    with pytest.raises(ValueError, match="revision"):
        embedding.LocalEmbedder(cache_dir=tmp_path).prepare()

    assert not any(item[0] == "stream" for item in calls)


def test_real_cache_quota_includes_atomic_model_assets_and_marker(tmp_path, monkeypatch):
    from classbot.knowledge.cache_manager import CacheManager

    cache = tmp_path / "cache"
    manager = CacheManager(cache)
    manager.set_limit(2000)
    install_transformers(monkeypatch)
    install_hub(monkeypatch, {
        "config.json": b"{}", "model.safetensors": b"x" * 1200, "tokenizer.json": b"{}",
    })

    with manager.mutation(), manager.quota():
        embedding.LocalEmbedder(cache_dir=cache / "models").prepare()

    assert 1200 < manager.usage_bytes() <= 2000
    assert len(list(cache.rglob("classbot-snapshot.json"))) == 1


def test_real_cache_quota_refuses_model_before_downloading_weights(tmp_path, monkeypatch):
    from classbot.knowledge.cache_manager import CacheLimitError, CacheManager

    cache = tmp_path / "cache"
    manager = CacheManager(cache)
    manager.set_limit(1000)
    install_transformers(monkeypatch)
    calls, _ = install_hub(monkeypatch, {"config.json": b"{}", "model.safetensors": b"x" * 1200})

    with manager.mutation(), manager.quota(), pytest.raises(CacheLimitError):
        embedding.LocalEmbedder(cache_dir=cache / "models").prepare()

    assert manager.usage_bytes() <= 1000
    assert not any(item[0] == "stream" for item in calls)
    assert not list(cache.rglob("*.safetensors"))


def test_existing_asset_hardlink_cannot_modify_original_material(tmp_path, monkeypatch):
    original = tmp_path / "original.txt"
    original.write_bytes(b"weights")
    cache = tmp_path / "models"
    install_quota(monkeypatch, tmp_path, 200_000)
    install_transformers(monkeypatch)
    install_hub(monkeypatch, {"config.json": b"{}", "model.safetensors": b"weights"})
    first = embedding.LocalEmbedder(cache_dir=cache)
    first.prepare()
    weight = next(cache.rglob("model.safetensors"))
    actual_lstat = Path.lstat

    def linked_stat(path):
        info = actual_lstat(path)
        if path == weight:
            return SimpleNamespace(
                st_mode=info.st_mode, st_nlink=2,
                st_file_attributes=getattr(info, "st_file_attributes", 0),
            )
        return info

    # Sandboxed Windows does not allow creating hard links. Exercise the same
    # native lstat result without requiring host permissions or touching sources.
    monkeypatch.setattr(Path, "lstat", linked_stat)

    with pytest.raises(ValueError, match="link"):
        embedding.LocalEmbedder(cache_dir=cache).prepare()

    assert original.read_bytes() == b"weights"


def test_git_blob_identity_is_checked_for_small_model_assets():
    data = b"config contents"
    digest = hashlib.sha1(f"blob {len(data)}\0".encode("ascii") + data).hexdigest()
    record = embedding._model_asset(SimpleNamespace(
        rfilename="config.json", size=len(data), blob_id=digest, lfs=None,
    ))
    hasher = embedding._model_hasher(record)
    hasher.update(data)
    assert record["algorithm"] == "git-sha1" and hasher.hexdigest() == digest
