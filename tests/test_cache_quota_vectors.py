"""Tight disk budgets affect reuse speed, never the returned document rows."""

import hashlib
import os

import numpy as np
import pytest

from classbot.knowledge.cache_manager import (
    CacheLimitError,
    CacheManager,
    CacheSafetyError,
)
from classbot.knowledge.vector_cache import CachedEmbedder


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, query=False):
        self.calls.append(list(texts))
        return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_tight_quota_keeps_readonly_hits_and_computes_every_missing_row(tmp_path):
    database = tmp_path / "embedding-vectors.sqlite3"
    first = CachedEmbedder(Encoder(), database, "fixed-model")
    first.encode(["original"])
    first.close()
    original = digest(database)
    manager = CacheManager(tmp_path)
    manager.set_limit(manager.usage_bytes() + 100_000)
    encoder = Encoder()
    with manager.mutation(), manager.quota():
        cache = CachedEmbedder(encoder, database, "fixed-model")
        assert cache._read_only
        result = cache.encode(["original", "new", "new", "another"])
        assert result.shape == (4, 2)
        assert encoder.calls == [["new", "another"]]
        assert cache.stats["hits"] == 1
        assert cache.stats["cache_rows"] == 1
        cache.close()
    assert digest(database) == original
    assert not database.with_name(database.name + "-journal").exists()
    assert manager.usage_bytes() <= manager.limit_bytes


def test_tight_quota_does_not_create_optional_database(tmp_path):
    manager = CacheManager(tmp_path)
    manager.set_limit(2000)
    with manager.mutation(), manager.quota(), pytest.raises(CacheLimitError):
        CachedEmbedder(Encoder(), tmp_path / "embedding-vectors.sqlite3", "model")
    assert not (tmp_path / "embedding-vectors.sqlite3").exists()


def test_writable_cache_stops_mutating_when_an_index_uses_the_remaining_budget(tmp_path):
    database = tmp_path / "embedding-vectors.sqlite3"
    manager = CacheManager(tmp_path)
    manager.set_limit(3 * 1024**2)
    with manager.mutation(), manager.quota():
        cache = CachedEmbedder(Encoder(), database, "model", max_bytes=128 * 1024)
        assert not cache._read_only
        cache.encode(["first"])
        original = digest(database)
        padding = manager.limit_bytes - manager.usage_bytes() - 1000
        (tmp_path / "synthetic-index-reservation").write_bytes(b"x" * padding)
        assert cache.encode(["first", "still computed"]).shape == (2, 2)
        assert cache.stats["cache_rows"] == 1
        cache.close()
    assert digest(database) == original
    assert manager.usage_bytes() <= manager.limit_bytes


@pytest.mark.parametrize("suffix", ["", "-journal", "-wal", "-shm"])
def test_sqlite_cache_and_sidecars_cannot_write_through_hardlinks(tmp_path, suffix):
    source = tmp_path / "original-data"
    source.write_bytes(b"original data must remain")
    database = tmp_path / "embedding-vectors.sqlite3"
    target = database.with_name(database.name + suffix)
    try:
        os.link(source, target)
    except OSError:
        pytest.skip("hard links not permitted by this runtime")
    with pytest.raises(CacheSafetyError):
        CachedEmbedder(Encoder(), database, "model")
    assert source.read_bytes() == b"original data must remain"
