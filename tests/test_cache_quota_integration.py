"""Production builders retain full coverage and safe snapshots under quotas."""

import gc
import hashlib
import json
import weakref

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex
from classbot.knowledge.cache_manager import (
    CacheBusyError,
    CacheLimitError,
    CacheManager,
)


class Embedder:
    def __init__(self, dimension=4):
        self.dimension = dimension
        self.rows = 0

    def encode(self, texts, *, query=False):
        if not query:
            self.rows += len(texts)
        return np.ones((len(texts), self.dimension), dtype=np.float32)


def material_folders(tmp_path, count=1):
    root, cache = tmp_path / "materials", tmp_path / "cache"
    root.mkdir()
    for number in range(count):
        (root / f"result-{number:03}.txt").write_text(
            f"完整实验结果 {number}。每份资料必须保留。", encoding="utf-8"
        )
    return root, cache


def source_hashes(root):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.iterdir()
        if path.is_file()
    }


def test_low_total_limit_stops_before_allocating_full_matrix(tmp_path):
    root, cache = material_folders(tmp_path, 8)
    before = source_hashes(root)
    manager = CacheManager(cache)
    manager.set_limit(8192)
    with pytest.raises(CacheLimitError, match="上限"):
        KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder(512))
    assert not list(cache.glob("*.npy"))
    assert not list(cache.glob("index-*.json"))
    assert manager.usage_bytes() <= 8192
    assert source_hashes(root) == before


def test_refresh_allocation_failure_preserves_old_complete_snapshot_and_sources(
    tmp_path,
):
    root, cache = material_folders(tmp_path)
    old = KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    metadata = next(cache.glob("index-*.json"))
    previous_metadata = metadata.read_bytes()
    vector_path = cache / json.loads(previous_metadata)["vectors_file"]
    previous_vector = hashlib.sha256(vector_path.read_bytes()).hexdigest()
    for number in range(1, 30):
        (root / f"result-{number:03}.txt").write_text(
            f"新增资料 {number}", encoding="utf-8"
        )
    before = source_hashes(root)
    manager = CacheManager(cache)
    limit = manager.usage_bytes() + 8192
    manager.set_limit(limit)
    with pytest.raises(CacheLimitError):
        KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder(512))
    assert metadata.read_bytes() == previous_metadata
    assert hashlib.sha256(vector_path.read_bytes()).hexdigest() == previous_vector
    assert old.report.indexed_files == 1
    assert old.search("完整实验结果")
    assert source_hashes(root) == before
    assert manager.usage_bytes() <= limit


def test_atomic_metadata_peak_keeps_old_commit_and_resumable_full_vectors(tmp_path):
    root, cache = material_folders(tmp_path)
    old = KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    metadata = next(cache.glob("index-*.json"))
    previous = metadata.read_bytes()
    (root / "result-000.txt").write_text("来源内容" * 4000, encoding="utf-8")
    before = source_hashes(root)
    manager = CacheManager(cache)
    limit = manager.usage_bytes() + 4096
    manager.set_limit(limit)
    progress = []
    with pytest.raises(CacheLimitError):
        KnowledgeIndex.build(
            root, cache, mode="full", embedder=Embedder(), progress=progress.append
        )
    assert "提交完整资料快照。" in progress
    assert metadata.read_bytes() == previous
    assert (cache / json.loads(previous)["vectors_file"]).is_file()
    assert list(cache.glob(".*.checkpoint.json"))
    assert not list(cache.glob("*.tmp"))
    assert manager.usage_bytes() <= limit
    assert source_hashes(root) == before
    assert old.report.indexed_files == 1
    manager.set_limit(1024 * 1024)
    resumed = Embedder()
    completed = KnowledgeIndex.build(root, cache, mode="full", embedder=resumed)
    assert resumed.rows == 0
    assert completed.report.limited_files == 0
    assert completed.report.chunks > old.report.chunks
    assert source_hashes(root) == before


def test_full_index_keeps_every_file_when_optional_shared_sqlite_cannot_fit(
    tmp_path, monkeypatch
):
    from classbot.knowledge import index as module

    class LocalTestEmbedder(Embedder):
        def __init__(self, cache_dir):
            super().__init__()
            self.cache_dir = cache_dir / "models"

        def prepare(self, *, progress=None):
            pass

    monkeypatch.setattr(module, "LocalEmbedder", LocalTestEmbedder)
    root, cache = material_folders(tmp_path, 13)
    before = source_hashes(root)
    manager = CacheManager(cache)
    manager.set_limit(30000)
    embedder = LocalTestEmbedder(cache)
    progress = []
    index = KnowledgeIndex.build(
        root, cache, mode="full", embedder=embedder, progress=progress.append
    )
    assert index.report.indexed_files == 13
    assert index.report.chunks == 13
    assert index.report.limited_files == 0
    assert embedder.rows == 13
    assert {chunk.relative_path for chunk in index._chunks} == set(before)
    assert any("共享向量复用缓存空间不足" in message for message in progress)
    assert not (cache / "embedding-vectors.sqlite3").exists()
    assert manager.usage_bytes() <= 30000
    assert source_hashes(root) == before


def test_loaded_full_index_and_surviving_mmap_view_protect_snapshot_until_released(
    tmp_path,
):
    root, cache = material_folders(tmp_path, 3)
    before = source_hashes(root)
    index = KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    key = index.cache_key
    manager = CacheManager(cache, source_roots=(root,))
    mapping_view = index._vectors[:1]
    index_ref = weakref.ref(index)
    with pytest.raises(CacheBusyError):
        manager.clear_library(key)
    del index
    gc.collect()
    assert index_ref() is None
    with pytest.raises(CacheBusyError):
        manager.clear_library(key)
    del mapping_view
    gc.collect()
    assert manager.clear_library(key) > 0
    assert not list(cache.glob("index-*.json"))
    assert source_hashes(root) == before


def test_cached_index_reload_has_a_fresh_reader_lease_without_new_embeddings(tmp_path):
    root, cache = material_folders(tmp_path)
    first = KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    key = first.cache_key
    del first
    gc.collect()
    reused = Embedder()
    second = KnowledgeIndex.build(root, cache, mode="full", embedder=reused)
    assert reused.rows == 0
    with pytest.raises(CacheBusyError):
        CacheManager(cache).clear_library(key)
    del second
    gc.collect()
    assert CacheManager(cache).clear_library(key) > 0
