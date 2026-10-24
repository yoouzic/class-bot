import hashlib
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest


def cache_type():
    from classbot.knowledge.vector_cache import CachedEmbedder

    return CachedEmbedder


class RecordingEmbedder:
    def __init__(self, dimension=3):
        self.calls = []
        self.dimension = dimension

    def vector(self, text, query=False):
        digest = hashlib.sha256(("query:" if query else "document:").encode() + text.encode()).digest()
        vector = np.resize(np.frombuffer(digest, dtype=np.uint8), self.dimension).astype(np.float32) + 1
        return vector / np.linalg.norm(vector)

    def encode(self, texts, *, query=False):
        self.calls.append((list(texts), query))
        return np.stack([self.vector(text, query) for text in texts])


def test_exact_text_duplicates_keep_every_row_and_reuse_committed_vectors(tmp_path):
    delegate = RecordingEmbedder()
    database = tmp_path / "cache.sqlite3"
    cache = cache_type()(delegate, database, "model@immutable:pooling-v1")
    texts = ["same text", "other", "same text", "same text ", "other"]
    result = cache.encode(texts)
    np.testing.assert_array_equal(result, np.stack([delegate.vector(text) for text in texts]))
    assert result.dtype == np.float32
    assert delegate.calls == [(["same text", "other", "same text "], False)]
    assert cache.stats["misses"] == 5
    assert cache.stats["encoded_rows"] == 3
    assert cache.stats["deduplicated_rows"] == 2

    # A second connection sees the committed batch even while the first remains open.
    restarted_delegate = RecordingEmbedder()
    restarted = cache_type()(restarted_delegate, database, "model@immutable:pooling-v1")
    np.testing.assert_array_equal(restarted.encode(texts), result)
    assert restarted_delegate.calls == []
    assert restarted.stats["hits"] == 5
    assert restarted.stats["cache_rows"] == 3
    cache.close()
    restarted.close()


def test_query_mode_bypasses_document_cache_and_preserves_query_semantics(tmp_path):
    delegate = RecordingEmbedder()
    cache = cache_type()(delegate, tmp_path / "cache.sqlite3", "model@immutable:pooling-v1")
    document = cache.encode(["hello"])
    before = cache.stats.copy()
    query = cache.encode(["hello", "hello"], query=True)
    assert delegate.calls[-1] == (["hello", "hello"], True)
    np.testing.assert_array_equal(query[0], delegate.vector("hello", True))
    assert not np.array_equal(query[0], document[0])
    assert cache.stats == before
    cache.close()


def test_model_and_pooling_identities_never_share_vectors(tmp_path):
    database = tmp_path / "cache.sqlite3"
    first = cache_type()(RecordingEmbedder(3), database, "model-a:pooling-v1")
    first.encode(["same"])
    other_delegate = RecordingEmbedder(4)
    other = cache_type()(other_delegate, database, "model-a:pooling-v2")
    assert other.encode(["same"]).shape == (1, 4)
    assert other_delegate.calls == [(["same"], False)]
    first.close()
    other.close()


@pytest.mark.parametrize("mutation", ["blob", "digest", "dtype", "dimension", "zero", "nan", "norm"])
def test_corrupted_cached_vector_is_recomputed(tmp_path, mutation):
    database = tmp_path / "cache.sqlite3"
    first = cache_type()(RecordingEmbedder(), database, "model")
    expected = first.encode(["a"])
    first.close()
    with sqlite3.connect(database) as connection:
        if mutation == "blob":
            connection.execute("UPDATE embedding_cache_vectors SET vector = ?", (b"short",))
        elif mutation == "digest":
            connection.execute("UPDATE embedding_cache_vectors SET vector_digest = ?", ("bad",))
        elif mutation == "dtype":
            connection.execute("UPDATE embedding_cache_vectors SET dtype = ?", ("pickle",))
        elif mutation == "dimension":
            connection.execute("UPDATE embedding_cache_vectors SET dimension = 4")
        else:
            vector = {
                "zero": np.zeros(3, dtype="<f4"),
                "nan": np.array([float("nan"), 0, 1], dtype="<f4"),
                "norm": np.ones(3, dtype="<f4"),
            }[mutation]
            blob = vector.tobytes()
            connection.execute(
                "UPDATE embedding_cache_vectors SET vector = ?, vector_digest = ?",
                (blob, hashlib.sha256(blob).hexdigest()),
            )
    delegate = RecordingEmbedder()
    resumed = cache_type()(delegate, database, "model")
    np.testing.assert_array_equal(resumed.encode(["a"]), expected)
    assert delegate.calls == [(["a"], False)]
    assert resumed.stats["corrupt_rows"] == 1
    resumed.close()


def test_lookup_handles_more_than_sqlite_parameter_batch_limit(tmp_path):
    database = tmp_path / "cache.sqlite3"
    first = cache_type()(RecordingEmbedder(), database, "model")
    texts = [f"text-{index}" for index in range(150)]
    expected = first.encode(texts)
    first.close()
    delegate = RecordingEmbedder()
    resumed = cache_type()(delegate, database, "model")
    np.testing.assert_array_equal(resumed.encode(texts[::-1]), expected[::-1])
    assert delegate.calls == []
    resumed.close()


def test_dimension_change_fails_without_committing_new_rows(tmp_path):
    database = tmp_path / "cache.sqlite3"
    cache = cache_type()(RecordingEmbedder(3), database, "model")
    cache.encode(["old"])
    cache.delegate.dimension = 4
    with pytest.raises(ValueError, match="dimension"):
        cache.encode(["old", "new"])
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM embedding_cache_vectors").fetchone()[0] == 1
    cache.close()


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.array([[float("nan"), 0, 1]]), np.ones((1, 3)), np.ones(3), np.ones((2, 3))])
def test_invalid_delegate_output_does_not_enter_durable_cache(tmp_path, bad):
    class BadEmbedder:
        def encode(self, texts, *, query=False):
            return bad

    database = tmp_path / "cache.sqlite3"
    cache = cache_type()(BadEmbedder(), database, "model")
    with pytest.raises(ValueError):
        cache.encode(["new"])
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT count(*) FROM embedding_cache_vectors").fetchone()[0] == 0
    cache.close()


def test_cache_limits_evict_rows_without_dropping_output_rows(tmp_path):
    database = tmp_path / "cache.sqlite3"
    delegate = RecordingEmbedder()
    cache = cache_type()(delegate, database, "model", max_entries=2)
    texts = ["a", "b", "c", "a"]
    result = cache.encode(texts)
    np.testing.assert_array_equal(result, np.stack([delegate.vector(text) for text in texts]))
    assert cache.stats["cache_rows"] == 2
    with sqlite3.connect(database) as connection:
        columns = {row[1] for row in connection.execute("PRAGMA table_info(embedding_cache_vectors)")}
        assert "text" not in columns
    cache.close()


def test_empty_input_and_invalid_configuration(tmp_path):
    delegate = RecordingEmbedder()
    cache = cache_type()(delegate, tmp_path / "cache.sqlite3", "model")
    assert cache.encode([]).shape == (0, 0)
    assert delegate.calls == []
    with pytest.raises(TypeError, match="text"):
        cache.encode([123])
    cache.close()
    with pytest.raises(ValueError, match="identity"):
        cache_type()(delegate, tmp_path / "other.sqlite3", "")
    with pytest.raises(ValueError, match="max_entries"):
        cache_type()(delegate, tmp_path / "other.sqlite3", "model", max_entries=0)


def test_byte_limit_bounds_database_without_losing_returned_vectors(tmp_path):
    database = tmp_path / "cache.sqlite3"
    delegate = RecordingEmbedder(4096)
    cache = cache_type()(delegate, database, "model", max_bytes=128 * 1024)
    texts = [f"document {index}" for index in range(100)]
    result = cache.encode(texts)
    np.testing.assert_array_equal(result, np.stack([delegate.vector(text) for text in texts]))
    assert 0 < cache.stats["cache_rows"] < len(texts)
    cache.close()
    assert database.stat().st_size <= 128 * 1024


def test_two_connections_writing_the_same_miss_keep_accounting_correct(tmp_path):
    barrier = threading.Barrier(2)

    class SimultaneousEmbedder(RecordingEmbedder):
        def encode(self, texts, *, query=False):
            result = super().encode(texts, query=query)
            barrier.wait(timeout=5)
            return result

    database = tmp_path / "cache.sqlite3"
    first = cache_type()(SimultaneousEmbedder(), database, "model")
    second = cache_type()(SimultaneousEmbedder(), database, "model")
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda cache: cache.encode(["same"]), [first, second]))
    np.testing.assert_array_equal(results[0], results[1])
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT row_count FROM embedding_cache_totals").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM embedding_cache_vectors").fetchone()[0] == 1
    first.close()
    second.close()
