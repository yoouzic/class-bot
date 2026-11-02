"""The optional acceleration database must remain disposable even if damaged."""

import numpy as np

from classbot.knowledge import KnowledgeIndex
from classbot.knowledge.cache_manager import CacheManager


def test_explicit_cleanup_can_remove_a_damaged_owned_shared_database(tmp_path):
    database = tmp_path / "embedding-vectors.sqlite3"
    database.write_bytes(b"damaged sqlite content" * 100)
    original = tmp_path / "original.csv"
    original.write_bytes(b"source data")
    size = database.stat().st_size
    assert CacheManager(tmp_path).clear_shared_vectors() == size
    assert not database.exists()
    assert original.read_bytes() == b"source data"


def test_damaged_optional_shared_cache_does_not_block_complete_index(tmp_path, monkeypatch):
    from classbot.knowledge import index as module

    class Encoder:
        def __init__(self):
            self.cache_dir = tmp_path / "cache" / "models"

        def prepare(self, **kwargs):
            pass

        def encode(self, texts, *, query=False):
            return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))

    monkeypatch.setattr(module, "LocalEmbedder", Encoder)
    root, cache = tmp_path / "source", tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    original = root / "notes.txt"
    original.write_text("原始资料", encoding="utf-8")
    database = cache / "embedding-vectors.sqlite3"
    database.write_bytes(b"damaged sqlite content" * 100)
    index = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")
    assert index.report.indexed_files == index.report.chunks == 1
    assert index.report.limited_files == 0
    assert index.search("原始资料")
    assert original.read_text(encoding="utf-8") == "原始资料"


def test_optional_sqlite_failure_mid_build_still_indexes_every_remaining_chunk(tmp_path, monkeypatch):
    from classbot.knowledge import index as module

    class Encoder:
        def __init__(self):
            self.cache_dir = tmp_path / "cache" / "models"

        def prepare(self, **kwargs):
            pass

        def encode(self, texts, *, query=False):
            return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))

    monkeypatch.setattr(module, "LocalEmbedder", Encoder)
    original_encode = module.CachedEmbedder.encode
    calls = 0

    def lose_connection(self, texts, *, query=False):
        nonlocal calls
        calls += 1
        if calls == 2:
            self._connection.close()
        return original_encode(self, texts, query=query)

    monkeypatch.setattr(module.CachedEmbedder, "encode", lose_connection)
    root, cache = tmp_path / "source", tmp_path / "cache"
    root.mkdir()
    for number in range(65):
        (root / f"data-{number}.txt").write_text(f"原始资料 {number}", encoding="utf-8")
    index = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")
    assert calls == 2
    assert index.report.indexed_files == index.report.chunks == 65
    assert index.report.limited_files == 0
    assert len(list(root.iterdir())) == 65
