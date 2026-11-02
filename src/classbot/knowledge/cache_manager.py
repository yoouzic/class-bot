"""Explicit management of generated caches, with reader leases and a total quota.

Source directories are never traversed or modified. Cleanup accepts only direct,
strictly named, single-link cache files. The quota counts logical file bytes,
including shared models and temporary files; it never truncates source coverage.
"""

from __future__ import annotations

import contextvars
import errno
import json
import os
import re
import sqlite3
import stat
import threading
import weakref
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self
from uuid import uuid4

from .atomic import replace_atomic

DEFAULT_LIMIT_BYTES = 10 * 1024**3
_MAX_SMALL_JSON = 64 * 1024
_KEY = re.compile(r"index-[0-9a-f]{24}")
_ARTIFACT = re.compile(
    r"(?P<key>index-[0-9a-f]{24})(?:\.json|\.report\.txt|\.cache-info"
    r"|-[0-9a-f]{32}\.(?:npy|npz))\Z"
)
_CHECKPOINT = re.compile(r"\.(?P<key>index-[0-9a-f]{24})\.json\.checkpoint\.json\Z")
_TEMP = re.compile(
    r"\.(?P<key>index-[0-9a-f]{24})(?:\.json(?:\.checkpoint\.json)?|\.cache-info|\.vectors|\.report)"
    r"\.[A-Za-z0-9_-]{6,}\.tmp\Z"
)
_LEGACY_CHECKPOINT_TEMP = re.compile(
    r"\.\.(?P<key>index-[0-9a-f]{24})\.json\.checkpoint\.json"
    r"\.[A-Za-z0-9_-]{6,}\.tmp\Z"
)
_SHARED_NAMES = frozenset(
    "embedding-vectors.sqlite3" + suffix for suffix in ("", "-journal", "-wal", "-shm")
)
_QUOTA: contextvars.ContextVar[CacheManager | None] = contextvars.ContextVar(
    "classbot_cache_quota", default=None
)


class CacheLimitError(RuntimeError):
    """A complete cache write would exceed the explicitly configured total."""


class CacheBusyError(RuntimeError):
    """A reader or builder currently owns the requested cache."""


class CacheSafetyError(RuntimeError):
    """A cache path cannot be proven safe for a generated-file mutation."""


@dataclass(frozen=True, slots=True)
class LibraryCache:
    key: str
    root: str
    mode: str
    model: str
    size_bytes: int
    file_count: int = 0
    chunk_count: int = 0
    built_at: str = ""
    protected: bool = False
    status: str = "ready"


@dataclass(frozen=True, slots=True)
class CacheInventory:
    libraries: tuple[LibraryCache, ...]
    total_bytes: int
    limit_bytes: int
    shared_bytes: int
    model_bytes: int
    other_bytes: int


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _plain(info: os.stat_result) -> bool:
    return stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not _is_reparse(info)


def _validate_chain(path: Path) -> None:
    for component in reversed((path, *path.parents)):
        try:
            info = component.lstat()
        except FileNotFoundError:
            continue
        if _is_reparse(info):
            raise CacheSafetyError(f"缓存路径包含链接或重解析点，拒绝修改：{component}")


def _key(value: str) -> str:
    if not isinstance(value, str) or _KEY.fullmatch(value) is None:
        raise ValueError("invalid generated index key")
    return value


def _artifact_key(name: str) -> str | None:
    for pattern in (_ARTIFACT, _CHECKPOINT, _TEMP, _LEGACY_CHECKPOINT_TEMP):
        match = pattern.fullmatch(name)
        if match:
            return match["key"]
    return None


class _FileLock:
    """One OS byte-range lock; Windows supports shared reader leases too."""

    def __init__(self, path: Path, *, shared: bool = False, create: bool = True):
        self.path, self.shared, self.create = path, shared, create
        self._stream = None
        self._overlapped = None

    def acquire(self) -> _FileLock:
        _validate_chain(self.path)
        if self.path.exists() and not _plain(self.path.lstat()):
            raise CacheSafetyError(f"缓存锁不是独立普通文件：{self.path.name}")
        if self.create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+b" if self.create else "r+b")
        try:
            if stream.seek(0, os.SEEK_END) == 0 and self.create:
                stream.write(b"\0")
                stream.flush()
            stream.seek(0)
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                class Overlapped(ctypes.Structure):
                    _fields_ = [
                        ("Internal", ctypes.c_size_t),
                        ("InternalHigh", ctypes.c_size_t),
                        ("Offset", wintypes.DWORD),
                        ("OffsetHigh", wintypes.DWORD),
                        ("hEvent", wintypes.HANDLE),
                    ]

                overlapped = Overlapped()
                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                lock = kernel.LockFileEx
                lock.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    ctypes.POINTER(Overlapped),
                ]
                lock.restype = wintypes.BOOL
                flags = 1 | (0 if self.shared else 2)
                if not lock(
                    msvcrt.get_osfhandle(stream.fileno()),
                    flags,
                    0,
                    1,
                    0,
                    ctypes.byref(overlapped),
                ):
                    code = ctypes.get_last_error()
                    if code in {32, 33, 158}:
                        raise CacheBusyError("缓存正在使用或建立索引，请稍后重试。")
                    raise ctypes.WinError(code)
                self._overlapped = overlapped
            else:
                import fcntl

                flag = fcntl.LOCK_SH if self.shared else fcntl.LOCK_EX
                try:
                    fcntl.flock(stream.fileno(), flag | fcntl.LOCK_NB)
                except OSError as error:
                    if error.errno in {errno.EAGAIN, errno.EACCES}:
                        raise CacheBusyError(
                            "缓存正在使用或建立索引，请稍后重试。"
                        ) from error
                    raise
            self._stream = stream
            return self
        except BaseException:
            stream.close()
            raise

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            if os.name == "nt":
                import ctypes
                import msvcrt
                from ctypes import wintypes

                kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                unlock = kernel.UnlockFileEx
                unlock.argtypes = [
                    wintypes.HANDLE,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    wintypes.DWORD,
                    ctypes.c_void_p,
                ]
                unlock.restype = wintypes.BOOL
                if not unlock(
                    msvcrt.get_osfhandle(stream.fileno()),
                    0,
                    1,
                    0,
                    ctypes.byref(self._overlapped),
                ):
                    raise ctypes.WinError(ctypes.get_last_error())
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        finally:
            stream.close()

    def __enter__(self) -> Self:
        return self.acquire()

    def __exit__(self, *_exc) -> None:
        self.close()


class IndexLease:
    """Retain this object as long as a loaded index or its mmap can be used."""

    def __init__(self, lock: _FileLock):
        self._finalizer = weakref.finalize(self, lock.close)

    def close(self) -> None:
        self._finalizer()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


class _MutationState:
    def __init__(self):
        self.thread_lock = threading.RLock()
        self.depth = 0
        self.file_lock: _FileLock | None = None


_STATES: dict[str, _MutationState] = {}
_STATES_LOCK = threading.Lock()


def _state(cache_dir: Path) -> _MutationState:
    with _STATES_LOCK:
        return _STATES.setdefault(os.path.normcase(str(cache_dir)), _MutationState())


def _read_small(path: Path) -> dict:
    info = path.lstat()
    if not _plain(info) or info.st_size > _MAX_SMALL_JSON:
        raise CacheSafetyError(f"缓存管理文件无法安全读取：{path.name}")
    with path.open("rb") as source:
        payload = json.loads(source.read(_MAX_SMALL_JSON + 1))
    if not isinstance(payload, dict):
        raise CacheSafetyError("cache management JSON must be an object")
    return payload


class CacheManager:
    def __init__(self, cache_dir: Path, source_roots=()):
        # Keep lexical spelling until checks finish; resolve() would hide links.
        self.cache_dir = Path(os.path.abspath(cache_dir))
        self.source_roots = tuple(Path(path).resolve() for path in source_roots)
        self.policy_path = self.cache_dir / ".cache-policy"
        self._headers: dict[tuple, dict] = {}

    def _validate_mutation(self) -> None:
        _validate_chain(self.cache_dir)
        cache = self.cache_dir.resolve()
        for source in self.source_roots:
            if source.is_relative_to(cache):
                raise CacheSafetyError("原始资料位于缓存目录中，拒绝清理或修改。请将缓存与原始资料分开存放。")
            # A generated cache below a project library is valid: indexing
            # explicitly excludes that cache subtree. Cleanup still only
            # accepts direct, strictly named, single-link generated files.

    @contextmanager
    def mutation(self) -> Iterator[CacheManager]:
        """Serialize builders, quota updates and cleanup across processes."""
        self._validate_mutation()
        state = _state(self.cache_dir)
        if not state.thread_lock.acquire(blocking=False):
            raise CacheBusyError("资料库正在建立索引或清理，请完成后重试。")
        try:
            if not state.depth:
                state.file_lock = _FileLock(
                    self.cache_dir / ".cache-policy.lock"
                ).acquire()
            state.depth += 1
            try:
                yield self
            finally:
                state.depth -= 1
                if not state.depth and state.file_lock is not None:
                    state.file_lock.close()
                    state.file_lock = None
        finally:
            state.thread_lock.release()

    @contextmanager
    def quota(self) -> Iterator[CacheManager]:
        token = _QUOTA.set(self)
        try:
            self.require_space(0)
            yield self
        finally:
            _QUOTA.reset(token)

    @property
    def limit_bytes(self) -> int:
        try:
            payload = _read_small(self.policy_path)
        except FileNotFoundError:
            return DEFAULT_LIMIT_BYTES
        except (OSError, ValueError, CacheSafetyError) as error:
            raise CacheSafetyError(
                "缓存总上限配置损坏，拒绝继续写入；请检查缓存管理配置。"
            ) from error
        value = payload.get("limit_bytes")
        if payload.get("version") != 1 or type(value) is not int or value <= 0:
            raise CacheSafetyError("缓存总上限配置无效。")
        return value

    def _files(self) -> list[tuple[Path, os.stat_result]]:
        _validate_chain(self.cache_dir)
        result = []
        if not self.cache_dir.exists():
            return result
        pending = [self.cache_dir]
        try:
            while pending:
                directory = pending.pop()
                with os.scandir(directory) as entries:
                    for entry in entries:
                        path = Path(entry.path)
                        try:
                            info = path.lstat()
                        except FileNotFoundError:
                            continue  # A builder may have atomically committed.
                        if stat.S_ISDIR(info.st_mode) and not _is_reparse(info):
                            pending.append(path)
                        else:
                            result.append((path, info))
        except OSError as error:
            raise CacheSafetyError("无法完整统计缓存大小，拒绝低估总占用。") from error
        return result

    def usage_bytes(self) -> int:
        return sum(info.st_size for _path, info in self._files())

    def require_space(self, additional_bytes: int) -> None:
        if type(additional_bytes) is not int or additional_bytes < 0:
            raise ValueError("additional_bytes must be a nonnegative integer")
        current, maximum = self.usage_bytes(), self.limit_bytes
        if current + additional_bytes > maximum:
            raise CacheLimitError(
                f"缓存总占用上限不足：已用 {current:,} 字节，新增需 {additional_bytes:,} 字节，"
                f"上限 {maximum:,} 字节。请清理不用的索引或调高上限；不会删减全量资料。"
            )

    def _write_json(
        self, path: Path, payload: dict, *, maximum: int | None = None
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        if len(data) > _MAX_SMALL_JSON:
            raise ValueError("cache management metadata is too large")
        if path.exists() and not _plain(path.lstat()):
            raise CacheSafetyError("缓存管理文件不是独立普通文件。")
        used = self.usage_bytes()
        cap = self.limit_bytes if maximum is None else maximum
        if used + len(data) > cap:
            raise CacheLimitError(
                "缓存上限不足以原子保存管理配置，请先清理索引或调高上限。"
            )
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as output:
                output.write(data)
                output.flush()
                os.fsync(output.fileno())
            replace_atomic(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def set_limit(self, limit_bytes: int) -> None:
        if type(limit_bytes) is not int or limit_bytes <= 0:
            raise ValueError("limit_bytes must be a positive integer")
        with self.mutation():
            if self.usage_bytes() > limit_bytes:
                raise CacheLimitError(
                    "新上限低于当前缓存占用，请先手动清理不用的索引。"
                )
            self._write_json(
                self.policy_path,
                {"version": 1, "limit_bytes": limit_bytes},
                maximum=limit_bytes,
            )

    def acquire_lease(self, key: str) -> IndexLease:
        self._validate_mutation()
        key = _key(key)
        lock = _FileLock(self.cache_dir / f".{key}.lease.lock", shared=True).acquire()
        return IndexLease(lock)

    lease = acquire_lease

    def _busy(self, key: str) -> bool:
        for name in (f".{key}.lease.lock", f".{key}.json.build.lock"):
            path = self.cache_dir / name
            if not path.exists():
                continue
            try:
                with _FileLock(path, create=False):
                    pass
            except (CacheBusyError, CacheSafetyError, OSError):
                return True
        return False

    def record_library(
        self,
        key: str,
        *,
        root,
        model_name: str = "",
        model=None,
        mode: str = "full",
        built_at: str = "",
        file_count: int = 0,
        chunk_count: int = 0,
        chunks: int | None = None,
    ) -> None:
        key = _key(key)
        with self.mutation():
            metadata = self.cache_dir / f"{key}.json"
            info = metadata.lstat()
            if not _plain(info):
                raise CacheSafetyError("索引元数据不是独立普通文件。")
            if chunks is not None:
                chunk_count = chunks
            if type(file_count) is not int or type(chunk_count) is not int:
                raise ValueError("cache library counts must be integers")
            self._write_json(
                self.cache_dir / f"{key}.cache-info",
                {
                    "version": 1,
                    "metadata_size": info.st_size,
                    "metadata_mtime_ns": info.st_mtime_ns,
                    "root": str(root),
                    "model": model_name
                    or (
                        model.get("name", "")
                        if isinstance(model, dict)
                        else str(model or "")
                    ),
                    "mode": str(mode),
                    "built_at": str(built_at),
                    "file_count": max(0, file_count),
                    "chunk_count": max(0, chunk_count),
                },
            )

    def _header(self, key: str) -> dict:
        path = self.cache_dir / f"{key}.json"
        try:
            info = path.lstat()
            if not _plain(info):
                return {}
        except FileNotFoundError:
            return {}
        signature = (key, info.st_size, info.st_mtime_ns)
        if signature in self._headers:
            return self._headers[signature]
        try:
            saved = _read_small(self.cache_dir / f"{key}.cache-info")
            if (
                saved.get("version") == 1
                and saved.get("metadata_size") == info.st_size
                and saved.get("metadata_mtime_ns") == info.st_mtime_ns
            ):
                self._headers[signature] = saved
                return saved
        except (OSError, ValueError, CacheSafetyError):
            pass
        # The writer places identity fields before the huge chunks array. Read
        # only a bounded prefix; a legacy 339 MB index never becomes a UI load.
        try:
            with path.open("rb") as source:
                raw = source.read(_MAX_SMALL_JSON)
            text = raw.decode("utf-8", errors="ignore")
            decoder = json.JSONDecoder()
            result, position = {}, 1
            if not text.lstrip().startswith("{"):
                return {}
            position = text.index("{") + 1
            while position < len(text):
                position = len(text) - len(text[position:].lstrip(" \t\r\n,"))
                if position >= len(text) or text[position] == "}":
                    break
                name, position = decoder.raw_decode(text, position)
                position = len(text) - len(text[position:].lstrip(" \t\r\n"))
                if text[position : position + 1] != ":":
                    break
                position += 1
                position = len(text) - len(text[position:].lstrip())
                if name == "chunks" and info.st_size > _MAX_SMALL_JSON:
                    break
                value, position = decoder.raw_decode(text, position)
                result[name] = value
            if isinstance(result.get("model"), dict):
                result["model"] = result["model"].get("name", "")
            report = result.get("report", {})
            if isinstance(report, dict):
                result["file_count"] = report.get("indexed_files", 0)
                result["chunk_count"] = report.get("chunks", 0)
            report_path = self.cache_dir / f"{key}.report.txt"
            try:
                if _plain(report_path.lstat()):
                    with report_path.open("rb") as report_file:
                        report_text = report_file.read(8192).decode(
                            "utf-8", errors="ignore"
                        )
                    counts = re.search(r"已索引文件：(\d+)；片段：(\d+)", report_text)
                    snapshot = re.search(r"(?m)^快照：(.+)$", report_text)
                    if (
                        counts
                        and snapshot
                        and snapshot[1].strip() == result.get("built_at")
                    ):
                        result["file_count"], result["chunk_count"] = map(
                            int, counts.groups()
                        )
            except OSError:
                pass
            self._headers[signature] = result
            return result
        except (OSError, ValueError, IndexError, TypeError):
            return {}

    def inventory(self) -> CacheInventory:
        files = self._files()
        grouped: dict[str, list[tuple[Path, os.stat_result]]] = {}
        total = shared = models = 0
        for path, info in files:
            total += info.st_size
            relative = path.relative_to(self.cache_dir)
            if relative.parts[0] == "models":
                models += info.st_size
            elif len(relative.parts) == 1:
                key = _artifact_key(path.name)
                if key:
                    grouped.setdefault(key, []).append((path, info))
                elif path.name in _SHARED_NAMES:
                    shared += info.st_size
        rows = []
        for key, artifacts in sorted(grouped.items()):
            unsafe = any(not _plain(info) for _path, info in artifacts)
            header = self._header(key)
            metadata_exists = any(path.name == f"{key}.json" for path, _ in artifacts)
            status = (
                "unsafe" if unsafe else ("ready" if metadata_exists else "incomplete")
            )

            def count(name, header=header):
                value = header.get(name, 0)
                return value if type(value) is int and value >= 0 else 0

            rows.append(
                LibraryCache(
                    key=key,
                    root=str(header.get("root", "")),
                    mode=str(header.get("mode", "")),
                    model=str(header.get("model", "")),
                    size_bytes=sum(info.st_size for _path, info in artifacts),
                    file_count=count("file_count"),
                    chunk_count=count("chunk_count"),
                    built_at=str(header.get("built_at", "")),
                    protected=unsafe or self._busy(key),
                    status=status,
                )
            )
        return CacheInventory(
            tuple(rows),
            total,
            self.limit_bytes,
            shared,
            models,
            total - shared - models - sum(row.size_bytes for row in rows),
        )

    def _targets(self, key: str) -> list[tuple[Path, os.stat_result]]:
        result = []
        with os.scandir(self.cache_dir) as entries:
            for entry in entries:
                if _artifact_key(entry.name) != key:
                    continue
                path = Path(entry.path)
                info = path.lstat()
                if not _plain(info):
                    raise CacheSafetyError(
                        "索引含链接、目录或硬链接，拒绝清理任何文件。"
                    )
                result.append((path, info))
        return result

    def clear_library(self, key: str, *, protected_keys=()) -> int:
        key = _key(key)
        if key in protected_keys:
            raise CacheBusyError("当前资料库仍在使用，不能清理。")
        with (
            self.mutation(),
            _FileLock(self.cache_dir / f".{key}.json.build.lock"),
            _FileLock(self.cache_dir / f".{key}.lease.lock"),
        ):
            targets = self._targets(key)
            removed = 0
            # A legacy Windows mmap can reject vector deletion; leave
            # its commit metadata intact in that case.
            targets.sort(key=lambda item: item[0].name == f"{key}.json")
            for path, info in targets:
                self._validate_mutation()
                current = path.lstat()
                if (
                    not _plain(current)
                    or current.st_ino != info.st_ino
                    or current.st_dev != info.st_dev
                ):
                    raise CacheSafetyError("索引文件在清理前发生变化，已停止。")
                path.unlink()
                removed += info.st_size
            self._headers.clear()
            return removed

    def clear_shared_vectors(self) -> int:
        with self.mutation():
            targets = []
            for name in sorted(_SHARED_NAMES):
                path = self.cache_dir / name
                if not path.exists() and not path.is_symlink():
                    continue
                info = path.lstat()
                if not _plain(info):
                    raise CacheSafetyError("共享缓存含链接或硬链接，拒绝清理。")
                targets.append((path, info))
            database = self.cache_dir / "embedding-vectors.sqlite3"
            if database.exists():
                connection = None
                try:
                    connection = sqlite3.connect(database, timeout=0)
                    connection.execute("BEGIN EXCLUSIVE")
                    connection.rollback()
                except sqlite3.Error as error:
                    if getattr(error, "sqlite_errorcode", None) not in {
                        sqlite3.SQLITE_NOTADB, sqlite3.SQLITE_CORRUPT,
                    }:
                        raise CacheBusyError(
                            "共享向量缓存正在使用或无法安全打开，未执行清理。"
                        ) from error
                    # Explicit cleanup may remove this disposable owned file
                    # even if it is damaged. Busy/locked still prevents cleanup.
                finally:
                    if connection is not None:
                        connection.close()
            removed = 0
            for path, info in targets:
                self._validate_mutation()
                if not _plain(path.lstat()):
                    raise CacheSafetyError("共享缓存文件在清理前发生变化，已停止。")
                path.unlink()
                removed += info.st_size
            return removed


def _active(path: Path) -> CacheManager | None:
    manager = _QUOTA.get()
    if manager is not None and Path(os.path.abspath(path)).is_relative_to(
        manager.cache_dir
    ):
        return manager
    return None


def require_cache_space(path: Path, additional_bytes: int) -> None:
    manager = _active(path)
    if manager is not None:
        manager.require_space(additional_bytes)


def cache_available_bytes(path: Path) -> int | None:
    manager = _active(path)
    if manager is None:
        return None
    return max(0, manager.limit_bytes - manager.usage_bytes())
