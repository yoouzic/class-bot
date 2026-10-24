"""Durable, exact-content document embeddings without changing source row identity.

The caller supplies an identity including the immutable model and pooling format.
Query instruction semantics remain with the delegate and never enter this cache.
"""
from __future__ import annotations

import hashlib
import sqlite3
import stat
import threading
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np

from .cache_manager import CacheLimitError, CacheSafetyError, cache_available_bytes

_LOOKUP_HASHES = 63  # One identity parameter plus hashes stays at or below 64.
_MAX_VECTOR_BYTES = 1024 * 1024
_ROW_OVERHEAD = 512
_DTYPE = "<f4"


class CachedEmbedder:
    """Reuse exact document text while returning one vector per original input row.

    ``stats`` is cumulative for this instance. Hits/misses count input rows;
    encoded_rows counts unique misses actually sent to the model. Same-batch
    duplicate misses count toward deduplicated_rows. Queries affect no counters.
    Limits apply across all identities in this database, evicting oldest use.
    ``max_bytes`` caps the main SQLite file; payload gets a conservative allowance
    for indexes/pages. SQLite's transient rollback journal is additional.
    """

    def __init__(
        self, delegate, database: Path, identity: str, *,
        max_entries: int = 1_000_000, max_bytes: int = 2 * 1024**3,
    ):
        if not isinstance(identity, str) or not identity.strip():
            raise ValueError("cache identity must include model and pooling format")
        if not isinstance(max_entries, int) or max_entries <= 0:
            raise ValueError("max_entries must be a positive integer")
        if not isinstance(max_bytes, int) or max_bytes < 65536:
            raise ValueError("max_bytes must be at least 65536")
        self.delegate = delegate
        self.database = Path(database)
        self.identity = identity
        self.max_entries = max_entries
        self.max_bytes = max_bytes
        self._payload_budget = max(0, max_bytes * 3 // 4 - 32768)
        self._lock = threading.RLock()
        self._counts = {
            "input_rows": 0, "hits": 0, "misses": 0, "encoded_rows": 0,
            "deduplicated_rows": 0, "corrupt_rows": 0,
        }
        # SQLite writes in place and creates sidecars. Never follow a cache
        # file link to source data, even when no quota context is installed.
        for candidate in (*reversed(self.database.parents), self.database):
            try:
                info = candidate.lstat()
            except FileNotFoundError:
                continue
            if (stat.S_ISLNK(info.st_mode)
                    or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
                    or (stat.S_ISREG(info.st_mode) and info.st_nlink != 1)):
                raise CacheSafetyError("共享向量缓存路径含链接，拒绝写入。")
        for suffix in ("-journal", "-wal", "-shm"):
            sidecar = self.database.with_name(self.database.name + suffix)
            if sidecar.exists() or sidecar.is_symlink():
                info = sidecar.lstat()
                if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                        or getattr(info, "st_file_attributes", 0)
                        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                    raise CacheSafetyError("共享向量缓存日志路径含链接，拒绝写入。")
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._read_only = not self._write_fits_quota()
        if self._read_only and not self.database.is_file():
            raise CacheLimitError("共享向量复用缓存空间不足；全量索引仍可直接计算全部片段。")
        self._connection = sqlite3.connect(
            "file:" + quote(str(self.database.resolve()).replace("\\", "/"), safe="/:") + "?mode=ro"
            if self._read_only else self.database,
            uri=self._read_only, timeout=30, check_same_thread=False,
        )
        try:
            if not self._read_only:
                self._initialize()
            else:
                # Validate the existing schema without creating tables,
                # touching timestamps, pruning rows, or creating a journal.
                self._connection.execute("SELECT row_count FROM embedding_cache_totals WHERE singleton = 1").fetchone()
        except BaseException:
            self._connection.close()
            raise

    def _write_fits_quota(self) -> bool:
        available = cache_available_bytes(self.database)
        if available is None:
            return True
        existing = self.database.stat().st_size if self.database.is_file() else 0
        # SQLite can grow to max_page_count and journal the old pages. Allow
        # record/header overhead too. Under tight quotas use read-only hits or
        # delegate computation instead; row coverage is never reduced.
        peak_growth = 2 * self.max_bytes + self.max_bytes // 64 + 1024**2 - existing
        return available >= peak_growth

    def _initialize(self) -> None:
        connection = self._connection
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        page_size = connection.execute("PRAGMA page_size").fetchone()[0]
        pages = connection.execute("PRAGMA page_count").fetchone()[0]
        if pages * page_size > self.max_bytes:
            raise ValueError("existing cache exceeds max_bytes")
        connection.execute(f"PRAGMA max_page_count={self.max_bytes // page_size}")
        connection.executescript(f"""
            CREATE TABLE IF NOT EXISTS embedding_cache_identities (
                identity TEXT PRIMARY KEY, dimension INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS embedding_cache_vectors (
                identity TEXT NOT NULL, text_hash TEXT NOT NULL,
                text_bytes INTEGER NOT NULL, dimension INTEGER NOT NULL,
                dtype TEXT NOT NULL, vector BLOB NOT NULL,
                vector_digest TEXT NOT NULL, used_at INTEGER NOT NULL,
                PRIMARY KEY (identity, text_hash),
                FOREIGN KEY (identity) REFERENCES embedding_cache_identities(identity)
            );
            CREATE INDEX IF NOT EXISTS embedding_cache_oldest
                ON embedding_cache_vectors(used_at);
            CREATE TABLE IF NOT EXISTS embedding_cache_totals (
                singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
                row_count INTEGER NOT NULL, payload_bytes INTEGER NOT NULL
            );
            INSERT OR IGNORE INTO embedding_cache_totals
                SELECT 1, count(*), coalesce(sum(length(vector) + {_ROW_OVERHEAD}), 0)
                FROM embedding_cache_vectors;
            CREATE TRIGGER IF NOT EXISTS embedding_cache_insert
            AFTER INSERT ON embedding_cache_vectors BEGIN
                UPDATE embedding_cache_totals SET row_count = row_count + 1,
                    payload_bytes = payload_bytes + length(NEW.vector) + {_ROW_OVERHEAD};
            END;
            CREATE TRIGGER IF NOT EXISTS embedding_cache_delete
            AFTER DELETE ON embedding_cache_vectors BEGIN
                UPDATE embedding_cache_totals SET row_count = row_count - 1,
                    payload_bytes = payload_bytes - length(OLD.vector) - {_ROW_OVERHEAD};
            END;
            CREATE TRIGGER IF NOT EXISTS embedding_cache_update
            AFTER UPDATE OF vector ON embedding_cache_vectors BEGIN
                UPDATE embedding_cache_totals SET payload_bytes = payload_bytes
                    + length(NEW.vector) - length(OLD.vector);
            END;
        """)
        with connection:
            self._prune()

    @property
    def stats(self) -> dict[str, int]:
        with self._lock:
            counts = dict(self._counts)
            counts["cache_rows"] = self._connection.execute(
                "SELECT count(*) FROM embedding_cache_vectors WHERE identity = ?",
                (self.identity,),
            ).fetchone()[0]
            return counts

    def encode(self, texts: list[str], *, query: bool = False) -> np.ndarray:
        if query:
            return self.delegate.encode(texts, query=True)
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("document texts must be strings")
        with self._lock:
            return self._encode_documents(texts)

    def _encode_documents(self, texts: list[str]) -> np.ndarray:
        unique = {}
        order = []
        for text in texts:
            content = text.encode("utf-8")
            key = hashlib.sha256(content).hexdigest()
            order.append(key)
            unique.setdefault(key, (text, len(content)))
        connection = self._connection
        metadata = connection.execute(
            "SELECT dimension FROM embedding_cache_identities WHERE identity = ?", (self.identity,),
        ).fetchone()
        dimension = metadata[0] if metadata else None
        if dimension is not None and (not isinstance(dimension, int) or not 0 < dimension <= _MAX_VECTOR_BYTES // 4):
            raise ValueError("cache identity dimension is invalid; use a fresh cache identity")
        found = {}
        corrupt = []
        keys = list(unique)
        for start in range(0, len(keys), _LOOKUP_HASHES):
            batch = keys[start:start + _LOOKUP_HASHES]
            placeholders = ",".join("?" for _ in batch)
            rows = connection.execute(
                "SELECT text_hash, text_bytes, dimension, dtype, vector, vector_digest "
                f"FROM embedding_cache_vectors WHERE identity = ? AND text_hash IN ({placeholders})",
                (self.identity, *batch),
            )
            for key, text_bytes, stored_dimension, dtype, blob, digest in rows:
                vector = self._cached_vector(blob, digest, stored_dimension, dtype, dimension)
                if text_bytes != unique[key][1] or vector is None:
                    corrupt.append(key)
                else:
                    found[key] = vector
        hit_keys = set(found)
        missing = [key for key in keys if key not in found]
        if missing:
            encoded = self._validate_batch(
                self.delegate.encode([unique[key][0] for key in missing]), len(missing), dimension,
            )
            dimension = encoded.shape[1]
            for key, vector in zip(missing, encoded):
                found[key] = vector
        result = np.stack([found[key] for key in order]).astype(np.float32, copy=False)
        now = time.time_ns()
        # Commit before returning, so a subsequent index checkpoint can rely on it.
        if self._read_only or not self._write_fits_quota():
            self._count_batch(order, hit_keys, missing, corrupt)
            return result
        with connection:
            connection.executemany(
                "DELETE FROM embedding_cache_vectors WHERE identity = ? AND text_hash = ?",
                ((self.identity, key) for key in corrupt),
            )
            connection.execute(
                "INSERT OR IGNORE INTO embedding_cache_identities VALUES (?, ?)",
                (self.identity, dimension),
            )
            actual_dimension = connection.execute(
                "SELECT dimension FROM embedding_cache_identities WHERE identity = ?",
                (self.identity,),
            ).fetchone()[0]
            if actual_dimension != dimension:
                raise ValueError("embedding dimension changed within the same cache identity")
            connection.executemany(
                "UPDATE embedding_cache_vectors SET used_at = ? WHERE identity = ? AND text_hash = ?",
                ((now, self.identity, key) for key in hit_keys),
            )
            pending = []
            pending_bytes = 0
            for key in reversed(missing):
                blob = np.asarray(found[key], dtype=_DTYPE).tobytes(order="C")
                cost = len(blob) + _ROW_OVERHEAD
                if len(pending) >= self.max_entries or pending_bytes + cost > self._payload_budget:
                    break
                pending.append((self.identity, key, unique[key][1], dimension, _DTYPE,
                                blob, hashlib.sha256(blob).hexdigest(), now))
                pending_bytes += cost
            self._prune(extra_entries=len(pending), extra_bytes=pending_bytes)
            connection.executemany(
                "INSERT INTO embedding_cache_vectors VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(identity, text_hash) DO UPDATE SET "
                "text_bytes=excluded.text_bytes, dimension=excluded.dimension, dtype=excluded.dtype, "
                "vector=excluded.vector, vector_digest=excluded.vector_digest, used_at=excluded.used_at",
                reversed(pending),
            )
        self._count_batch(order, hit_keys, missing, corrupt)
        return result

    def _count_batch(self, order, hit_keys, missing, corrupt):
        hits = sum(key in hit_keys for key in order)
        misses = len(order) - hits
        self._counts["input_rows"] += len(order)
        self._counts["hits"] += hits
        self._counts["misses"] += misses
        self._counts["encoded_rows"] += len(missing)
        self._counts["deduplicated_rows"] += misses - len(missing)
        self._counts["corrupt_rows"] += len(corrupt)

    @staticmethod
    def _validate_batch(value, rows: int, dimension: int | None) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32, order="C")
        if array.ndim != 2 or array.shape[0] != rows or not 0 < array.shape[1] <= _MAX_VECTOR_BYTES // 4:
            raise ValueError("embedding shape must match input rows and bounded vector dimension")
        if dimension is not None and array.shape[1] != dimension:
            raise ValueError("embedding dimension changed within the same cache identity")
        if not np.isfinite(array).all():
            raise ValueError("embedding vectors must be finite")
        norms = np.linalg.norm(array.astype(np.float64), axis=1)
        if not np.all(np.abs(norms - 1.0) <= 1e-3):
            raise ValueError("embedding vectors must have nonzero unit norm")
        return array

    @classmethod
    def _cached_vector(cls, blob, digest, dimension, dtype, expected_dimension):
        if (not isinstance(blob, bytes) or not isinstance(dimension, int)
                or dimension != expected_dimension or not 0 < dimension <= _MAX_VECTOR_BYTES // 4
                or dtype != _DTYPE or len(blob) != dimension * 4
                or hashlib.sha256(blob).hexdigest() != digest):
            return None
        try:
            return cls._validate_batch(np.frombuffer(blob, dtype=_DTYPE).reshape(1, dimension), 1,
                                       expected_dimension)[0]
        except ValueError:
            return None

    def _prune(self, *, extra_entries: int = 0, extra_bytes: int = 0) -> None:
        connection = self._connection
        count, size = connection.execute(
            "SELECT row_count, payload_bytes FROM embedding_cache_totals WHERE singleton = 1",
        ).fetchone()
        while count + extra_entries > self.max_entries or size + extra_bytes > self._payload_budget:
            oldest = connection.execute(
                "SELECT rowid, length(vector) FROM embedding_cache_vectors ORDER BY used_at, rowid LIMIT 1",
            ).fetchone()
            if oldest is None:
                break
            connection.execute("DELETE FROM embedding_cache_vectors WHERE rowid = ?", (oldest[0],))
            count -= 1
            size -= oldest[1] + _ROW_OVERHEAD

    def close(self) -> None:
        with self._lock:
            self._connection.close()
