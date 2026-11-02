"""Metadata publication remains successful when later cleanup is unavailable."""
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex
from classbot.knowledge import index as index_module
from classbot.knowledge.checkpoint import EmbeddingCheckpoint


class Encoder:
    def encode(self, texts, *, query=False):
        return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))


class CacheOnly:
    def encode(self, texts, *, query=False):
        pytest.fail("committed snapshot or complete checkpoint must not be re-encoded")


def _library(tmp_path):
    root = tmp_path / "library"
    root.mkdir()
    (root / "notes.txt").write_text("original source", encoding="utf-8")
    return root, tmp_path / "cache"


def _close(index):
    index._vectors._mmap.close()


def test_checkpoint_unlink_failure_after_commit_is_logged_and_snapshot_reloads(tmp_path, monkeypatch, caplog):
    root, cache = _library(tmp_path)
    first = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")
    old_revision = first.revision
    _close(first)
    (root / "notes.txt").write_text("updated source", encoding="utf-8")
    metadata, = cache.glob("index-*.json")
    state = metadata.with_name(f".{metadata.name}.checkpoint.json")
    original_unlink = Path.unlink
    error = OSError("simulated checkpoint sharing conflict")
    error.winerror = 32

    def unlink(path, *args, **kwargs):
        if path == state:
            assert json.loads(metadata.read_text(encoding="utf-8"))["revision"] != old_revision
            raise error
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as context, caplog.at_level(logging.WARNING):
        context.setattr(Path, "unlink", unlink)
        published = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")

    assert published.revision != old_revision
    assert json.loads(metadata.read_text(encoding="utf-8"))["revision"] == published.revision
    state_payload = json.loads(state.read_text(encoding="utf-8"))
    assert state_payload["completed"] == state_payload["rows"] == 1
    assert state_payload["vectors_file"] == Path(published._vectors.filename).name
    assert "index committed but checkpoint cleanup failed" in caplog.text
    loaded = KnowledgeIndex.build(root, cache, embedder=CacheOnly(), mode="full")
    assert loaded.revision == published.revision
    np.testing.assert_array_equal(loaded._vectors, published._vectors)
    _close(loaded)
    _close(published)


def test_recovery_with_retained_complete_state_never_unlinks_current_vector_file(tmp_path, monkeypatch):
    root, cache = _library(tmp_path)
    with monkeypatch.context() as context:
        # Emulate a process exiting after metadata commit but before state removal.
        context.setattr(EmbeddingCheckpoint, "discard_state", lambda self, **_kwargs: self.close())
        published = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")
    metadata, = cache.glob("index-*.json")
    state = metadata.with_name(f".{metadata.name}.checkpoint.json")
    assert state.exists()
    published_vectors = Path(published._vectors.filename)
    revision = published.revision
    _close(published)
    original_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == published_vectors:
            pytest.fail("a recovered checkpoint may already be the published vector file")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as context:
        # A structurally damaged metadata cache can force recovery with the same
        # source revision; the verified complete checkpoint must remain usable.
        context.setattr(KnowledgeIndex, "_load_cache", staticmethod(lambda *_args, **_kwargs: None))
        context.setattr(Path, "unlink", unlink)
        recovered = KnowledgeIndex.build(root, cache, embedder=CacheOnly(), mode="full")

    assert recovered.revision == revision
    assert Path(recovered._vectors.filename) == published_vectors
    assert published_vectors.exists() and not state.exists()
    loaded = KnowledgeIndex.build(root, cache, embedder=CacheOnly(), mode="full")
    assert loaded.revision == recovered.revision
    _close(loaded)
    _close(recovered)


def test_metadata_failure_before_commit_is_not_swallowed(tmp_path, monkeypatch, caplog):
    root, cache = _library(tmp_path)
    published = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")
    _close(published)
    metadata, = cache.glob("index-*.json")
    previous = metadata.read_bytes()
    (root / "notes.txt").write_text("updated source", encoding="utf-8")
    error = OSError("metadata commit failed")

    def fail(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(index_module, "_atomic_json", fail)

    with pytest.raises(OSError) as raised:
        KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")

    assert raised.value is error
    assert metadata.read_bytes() == previous
    assert "index committed but checkpoint cleanup failed" not in caplog.text
