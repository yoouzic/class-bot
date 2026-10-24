"""Durable contiguous embedding batches and one OS lock per index builder.

The state file commits a prefix only after its .npy data has been flushed and
fsynced. A crash can leave an uncommitted suffix; the next append overwrites it.
Callers hold BuildLock for the complete scan/build/publish operation.
"""

from __future__ import annotations

import errno
import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np

from .atomic import replace_atomic

_STATE_VERSION = 1
_MAX_STATE_BYTES = 64 * 1024


def _sync_directory(path: Path) -> None:
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _atomic_state(path: Path, payload: dict) -> None:
    from .quota_io import TextQuotaWriter

    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f"{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = Path(output.name)
            json.dump(
                payload,
                TextQuotaWriter(output, path, maximum=_MAX_STATE_BYTES),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            output.flush()
            if output.tell() > _MAX_STATE_BYTES:
                raise ValueError("embedding checkpoint state size exceeds limit")
            os.fsync(output.fileno())
        replace_atomic(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class BuildInProgressError(RuntimeError):
    """Another process already holds the lock for this index."""


class BuildLock:
    """Nonblocking process lock. Its persistent file must never be unlinked."""

    def __init__(self, metadata_path: Path):
        metadata = Path(metadata_path)
        self.path = metadata.with_name(f".{metadata.name}.build.lock")
        self._stream = None

    def __enter__(self) -> Self:
        if self._stream is not None:
            raise RuntimeError("build lock is already acquired")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() or self.path.is_symlink():
            info = self.path.lstat()
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1
                    or getattr(info, "st_file_attributes", 0)
                    & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)):
                raise ValueError("invalid build lock path")
        stream = self.path.open("a+b")
        try:
            if stream.seek(0, os.SEEK_END) == 0:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            stream.close()
            if error.errno in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise BuildInProgressError(
                    "another build is already using this index"
                ) from error
            raise
        except BaseException:
            stream.close()
            raise
        self._stream = stream
        return self

    def __exit__(self, *_exc) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()


class EmbeddingCheckpoint:
    """Append one bounded batch at a time, resuming only a verified prefix.

    ``revision`` identifies source bytes, chunking rules, and model identity.
    Automatic invalidation only removes state: a published snapshot may still
    refer to its vectors after a crash between publication and state cleanup.
    """

    def __init__(self, metadata_path: Path, revision: str, rows: int, max_bytes: int):
        if not isinstance(revision, str) or not revision:
            raise ValueError("checkpoint revision must be nonempty")
        if type(rows) is not int or rows < 0:
            raise ValueError("checkpoint rows must be nonnegative")
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("checkpoint size limit must be nonnegative")
        self.metadata_path = Path(metadata_path)
        self.state_path = self.metadata_path.with_name(
            f".{self.metadata_path.name}.checkpoint.json"
        )
        self.revision = revision
        self.rows = rows
        self.max_bytes = max_bytes
        self.completed = 0
        self.dimension: int | None = None
        self.path: Path | None = None
        self._prefix_hash = hashlib.sha256()
        self._mapping: np.memmap | None = None
        self._closed = False
        self._resume()

    def _payload(self, completed: int, prefix_digest: str) -> dict:
        return {
            "version": _STATE_VERSION,
            "revision": self.revision,
            "rows": self.rows,
            "dimension": self.dimension,
            "completed": completed,
            "vectors_file": self.path.name,
            "prefix_digest": prefix_digest,
        }

    def _resume(self) -> None:
        from .storage import load_mapped_vectors, vector_path

        if not self.state_path.exists():
            return
        mapping = None
        try:
            if (
                self.state_path.is_symlink()
                or self.state_path.stat().st_size > _MAX_STATE_BYTES
            ):
                raise ValueError("invalid embedding checkpoint state")
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            dimension, completed = payload["dimension"], payload["completed"]
            if (
                payload["version"] != _STATE_VERSION
                or payload["revision"] != self.revision
                or payload["rows"] != self.rows
                or type(dimension) is not int
                or dimension < 0
                or ((self.rows == 0) != (dimension == 0))
                or type(completed) is not int
                or not 0 <= completed <= self.rows
                or not isinstance(payload["prefix_digest"], str)
            ):
                raise ValueError("embedding checkpoint identity or shape changed")
            path = vector_path(self.metadata_path, payload["vectors_file"])
            if path.suffix != ".npy":
                raise ValueError("invalid embedding checkpoint vector format")
            mapping = load_mapped_vectors(
                path,
                self.max_bytes,
                writable=True,
                expected_shape=(self.rows, dimension),
            )
            prefix_hash = hashlib.sha256()
            if completed:
                # Hash the mapped prefix directly; no resident copy of prior batches.
                prefix_hash.update(memoryview(mapping[:completed]).cast("B"))
            if prefix_hash.hexdigest() != payload["prefix_digest"]:
                raise ValueError("embedding checkpoint prefix integrity mismatch")
            self.path, self.dimension, self.completed = path, dimension, completed
            self._prefix_hash = prefix_hash
            self._mapping = mapping
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            OverflowError,
            RecursionError,
        ):
            if mapping is not None:
                mapping._mmap.close()
            self.state_path.unlink(missing_ok=True)

    def _allocate(self, dimension: int) -> None:
        from .storage import create_mapped_vectors

        self.path, self._mapping = create_mapped_vectors(
            self.metadata_path, self.rows, dimension, self.max_bytes
        )
        self.dimension = dimension
        # Persist the file identity before the first batch; a crash can resume at 0.
        _atomic_state(self.state_path, self._payload(0, self._prefix_hash.hexdigest()))

    def append(self, start: int, batch: np.ndarray) -> None:
        if self._closed:
            raise ValueError("embedding checkpoint is closed")
        if type(start) is not int or start != self.completed:
            raise ValueError("embedding checkpoint append must be contiguous")
        array = np.asarray(batch)
        if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
            raise ValueError("invalid embedding batch shape")
        end = start + array.shape[0]
        if end > self.rows:
            raise ValueError("embedding batch exceeds checkpoint rows")
        if self.dimension is not None and array.shape[1] != self.dimension:
            raise ValueError("embedding dimension changed during build")
        if self.rows * array.shape[1] * 4 > self.max_bytes:
            raise ValueError(f"index vector size exceeds {self.max_bytes} bytes")
        array = np.asarray(array, dtype=np.float32, order="C")
        if not np.isfinite(array).all():
            raise ValueError("embedding batch values must be finite")
        if self._mapping is None:
            self._allocate(array.shape[1])
        self._mapping[start:end] = array
        self._mapping.flush()
        with self.path.open("r+b") as stream:
            os.fsync(stream.fileno())
        prefix_hash = self._prefix_hash.copy()
        prefix_hash.update(memoryview(array).cast("B"))
        _atomic_state(self.state_path, self._payload(end, prefix_hash.hexdigest()))
        self._prefix_hash, self.completed = prefix_hash, end

    def vectors(self) -> np.memmap:
        from .storage import load_mapped_vectors

        if self.completed != self.rows:
            raise ValueError("embedding checkpoint is incomplete")
        if self.path is None:
            if self._closed:
                raise ValueError("embedding checkpoint is closed")
            self._allocate(0)
        return load_mapped_vectors(
            self.path, self.max_bytes, expected_shape=(self.rows, self.dimension)
        )

    def close(self) -> None:
        if self._mapping is not None:
            self._mapping._mmap.close()
            self._mapping = None
        self._closed = True

    def discard_state(self, *, keep_vectors: bool = True) -> None:
        self.close()
        self.state_path.unlink(missing_ok=True)
        if not keep_vectors and self.path is not None:
            self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
