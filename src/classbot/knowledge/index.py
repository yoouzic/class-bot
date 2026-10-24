"""Explicit local index snapshots and hybrid candidate retrieval.

Scores rank candidates; cosine similarity is not a probability, and a source
hit alone does not substantiate a generated claim. No model failure falls back
to lexical-only retrieval.
"""

from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import os
import sqlite3
import tempfile
import weakref
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath

import numpy as np

from . import parsing
from .atomic import replace_atomic
from .cache_manager import CacheLimitError, CacheManager, CacheSafetyError
from .checkpoint import BuildLock, EmbeddingCheckpoint
from .embedding import (
    DEFAULT_MODEL,
    EMBEDDING_ENCODING_VERSION,
    LocalEmbedder,
    model_spec,
)
from .lexical import CompactLexicalIndex, continues_topic
from .policy import validate_index_mode
from .quota_io import TextQuotaWriter
from .sections import MarkdownSectionContexts
from .storage import file_digest, load_vectors, prune_vectors, save_vectors, vector_path
from .types import BuildReport, KnowledgeChunk, KnowledgeHit
from .vector_cache import CachedEmbedder

SCHEMA_VERSION = 5
MAX_CACHE_BYTES = 256 * 1024 * 1024
MAX_VECTOR_BYTES = 256 * 1024 * 1024
FULL_MAX_CACHE_BYTES = 2 * 1024 * 1024 * 1024
FULL_MAX_VECTOR_BYTES = 2 * 1024 * 1024 * 1024
_EMBEDDING_BATCH_SIZE = 64
_LOGGER = logging.getLogger(__name__)


def _digest(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def model_identity(model_name: str) -> dict:
    identifier, revision = model_spec(model_name)
    local_path = Path(model_name)
    if not local_path.is_dir():
        return {"name": identifier, "revision": revision}
    files = []
    for path in sorted(local_path.iterdir()):
        if path.is_file() and path.suffix.lower() in {
            ".json",
            ".txt",
            ".safetensors",
            ".model",
        }:
            info = path.stat()
            files.append((path.name, info.st_size, info.st_mtime_ns, file_digest(path)))
    return {"name": str(local_path.resolve()), "files": files}


def _vectors(value, rows: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] != rows or not array.shape[1]:
        raise ValueError("embedding shape does not match indexed chunks")
    if not np.isfinite(array).all():
        raise ValueError("embedding values must be finite")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("embedding vectors must have finite nonzero norm")
    array = array / norms
    array.flags.writeable = False
    return array


def _atomic_json(
    path: Path, payload: dict, maximum: int | None = None,
    before_commit: Callable[[], None] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary = output.name
            json.dump(
                payload,
                TextQuotaWriter(output, path, maximum=MAX_CACHE_BYTES if maximum is None else maximum),
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
                default=asdict,
            )
            output.flush()
            os.fsync(output.fileno())
            limit = MAX_CACHE_BYTES if maximum is None else maximum
            if os.fstat(output.fileno()).st_size > limit:
                raise ValueError(f"index cache size exceeds {limit} bytes")
        if before_commit is not None:
            before_commit()
        replace_atomic(temporary, path)
    finally:
        if temporary and Path(temporary).exists():
            Path(temporary).unlink()


class IndexBuildCancelled(RuntimeError):
    """Cooperative cancellation leaves committed batches available for resume."""


def _parser_identity() -> dict:
    packages = {}
    for name in ("PyMuPDF", "pypdf", "python-docx", "python-pptx"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {"implementation": parsing.PARSING_VERSION, **packages}


def _chunk_digest(chunks) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(json.dumps(asdict(chunk), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _cached_vectors(array: np.ndarray, rows: int) -> np.ndarray:
    """Validate a published normalized matrix with bounded scratch space."""
    if array.ndim != 2 or array.shape[0] != rows or (rows and not array.shape[1]):
        raise ValueError("embedding shape does not match indexed chunks")
    for start in range(0, rows, 8192):
        batch = array[start:start + 8192]
        if not np.isfinite(batch).all() or not np.allclose(
            np.linalg.norm(batch, axis=1), 1, atol=1e-4
        ):
            raise ValueError("cached vectors must be finite and normalized")
    array.flags.writeable = False
    return array


def _channel_candidates(
    semantic: np.ndarray, lexical: np.ndarray, limit: int, *, excluded: set[int] | None = None,
) -> list[tuple[int, float]]:
    """Keep coverage of both rankings without comparing their score scales.

    Up to two slots (at most half) come from literal matches; the majority is
    semantic. Empty or overlapping channels return their slots to other positive
    candidates. The diagnostic mixed score is not a global rank or confidence.
    """
    if limit <= 0:
        return []
    chosen = set(excluded or ())
    pool = min(limit + len(chosen), len(semantic))
    semantic_rank = heapq.nsmallest(
        pool, range(len(semantic)), key=lambda i: (-float(semantic[i]), -float(lexical[i]), i),
    )
    lexical_rank = heapq.nsmallest(
        pool, range(len(lexical)), key=lambda i: (-float(lexical[i]), -float(semantic[i]), i),
    )
    selected = []

    def take(ranked, values, count):
        added = 0
        for i in ranked:
            if added >= count:
                break
            if i not in chosen and values[i] > 0:
                selected.append((i, float(.7 * semantic[i] + .3 * lexical[i])))
                chosen.add(i)
                added += 1

    lexical_budget = min(2, limit // 2)
    take(semantic_rank, semantic, limit - lexical_budget)
    take(lexical_rank, lexical, lexical_budget)
    take(semantic_rank, semantic, limit - len(selected))
    take(lexical_rank, lexical, limit - len(selected))
    return selected


class KnowledgeIndex:
    def __init__(
        self,
        *,
        root: Path,
        revision: str,
        built_at: str,
        report: BuildReport,
        chunks: tuple[KnowledgeChunk, ...],
        vectors: np.ndarray,
        embedder,
        manifest: tuple[tuple[str, int, int], ...],
        cache_dir: Path,
        mode: str = "balanced",
    ):
        self.root = root
        self.revision = revision
        self.built_at = built_at
        self.report = report
        self._chunks = chunks
        self._vectors = vectors
        self._embedder = embedder
        self._manifest = manifest
        self._cache_dir = cache_dir
        self.mode = mode
        self._retrieval_gate = getattr(embedder, "retrieval_gate", None) or asyncio.Semaphore(1)
        self._lexical = CompactLexicalIndex(chunk.text for chunk in chunks)
        self._sections = MarkdownSectionContexts(chunks)
        self.report_path: Path | None = None
        self.embedding_stats: dict[str, int] = {}

    def report_text(self) -> str:
        report = self.report
        coverage = (
            "部分资料已索引（有文件达到处理边界，详情如下）"
            if report.limited_files
            else "索引已建立；跳过范围详见下方诊断"
        )
        return "\n".join(
            (
                "Classbot 本地资料索引报告",
                f"资料库：{self.root}",
                f"索引模式：{'全量（支持的可读取资料）' if self.mode == 'full' else '均衡（允许有边界的部分覆盖）'}",
                f"快照：{self.built_at}",
                f"状态：{coverage}",
                f"已索引文件：{report.indexed_files}；片段：{report.chunks}",
                f"跳过：{report.skipped_files}；达到处理边界：{report.limited_files}；重复副本：{report.duplicate_files}",
                "去重仅合并精确相同的内容；来源采用一个实际文件位置。",
                "资料只反映索引快照；未索引文件不能作为回答依据。",
                "",
                "详细诊断：",
                *(report.issues or ("无跳过或错误。",)),
            )
        )

    def _save_report(self, metadata: Path) -> None:
        target = metadata.with_suffix(".report.txt")
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                newline="\n",
                dir=target.parent,
                prefix=f".{metadata.stem}.report.",
                suffix=".tmp",
                delete=False,
            ) as output:
                temporary = Path(output.name)
                TextQuotaWriter(output, target).write(self.report_text())
                output.flush()
                os.fsync(output.fileno())
            replace_atomic(temporary, target)
            self.report_path = target
        except (OSError, CacheLimitError):
            # The committed index is still usable and the UI can show report_text().
            self.report_path = None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def build(
        cls,
        root: Path,
        cache_dir: Path,
        *,
        model_name: str = DEFAULT_MODEL,
        embedder=None,
        progress: Callable[[str], None] | None = None,
        mode: str = "balanced",
        cancelled: Callable[[], bool] | None = None,
    ) -> KnowledgeIndex:
        validate_index_mode(mode)
        root = Path(root).resolve(strict=True)
        manager = CacheManager(cache_dir)
        cache_dir = Path(cache_dir).resolve()
        if not root.is_dir():
            raise ValueError("library root must be a folder")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be nonempty")
        if root == cache_dir:
            raise ValueError("index cache must be a separate folder from the library root")
        if root.is_relative_to(cache_dir):
            raise ValueError("library root must not be inside the generated cache folder")
        key = {"root": os.path.normcase(str(root)), "name": model_name, "mode": mode}
        cache_path = cache_dir / f"index-{_digest(key)[:24]}.json"

        def emit(message: str) -> None:
            if cancelled is not None and cancelled():
                raise IndexBuildCancelled("资料索引已取消；已完成的向量批次已保留，再次刷新可续建。")
            if progress is not None:
                progress(message)

        def check_cancelled() -> None:
            if cancelled is not None and cancelled():
                raise IndexBuildCancelled("资料索引已取消；已完成的向量批次已保留，再次刷新可续建。")

        # A generated cache nested inside the selected library is excluded by
        # the scanner. Cleanup applies the stricter source-overlap checks.
        with manager.mutation(), manager.quota():
            lock_paths = (BuildLock(cache_path).path, cache_dir / f".{cache_path.stem}.lease.lock")
            manager.require_space(sum(not path.exists() for path in lock_paths))
            lease = manager.acquire_lease(cache_path.stem)
            transferred = False
            try:
                with BuildLock(cache_path):
                    result = cls._build_locked(
                        root, cache_dir, model_name=model_name, embedder=embedder,
                        progress=emit, mode=mode, cache_path=cache_path,
                        check_cancelled=check_cancelled, notify=progress,
                    )
                # Hold protection until the index AND any surviving mmap views
                # stop using this snapshot (including asynchronous retrieval).
                owner = result._vectors if isinstance(result._vectors, np.memmap) else result
                result._cache_lease_finalizer = weakref.finalize(owner, lease.close)
                transferred = True
                result.cache_key = cache_path.stem
                try:
                    manager.record_library(
                        cache_path.stem, root=root, model_name=model_name, mode=mode,
                        file_count=result.report.indexed_files,
                        chunk_count=result.report.chunks, built_at=result.built_at,
                    )
                except (OSError, CacheLimitError, CacheSafetyError):
                    # Metadata is already published. The optional inventory
                    # acceleration must not turn a committed build into failure.
                    _LOGGER.warning("cache inventory record not saved", exc_info=True)
                return result
            finally:
                if not transferred:
                    lease.close()

    @classmethod
    def _build_locked(
        cls,
        root: Path,
        cache_dir: Path,
        *,
        model_name: str = DEFAULT_MODEL,
        embedder=None,
        progress: Callable[[str], None] | None = None,
        mode: str,
        cache_path: Path,
        check_cancelled: Callable[[], None],
        notify: Callable[[str], None] | None,
    ) -> KnowledgeIndex:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("library root must be a folder")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be nonempty")
        cache_dir = Path(cache_dir).resolve()
        if root == cache_dir:
            raise ValueError(
                "index cache must be a separate folder from the library root"
            )
        emit = progress or (lambda _message: None)
        emit("扫描所选材料文件夹，排除隐藏文件、凭据与依赖目录。")
        scan = parsing.scan(root, exclude=cache_dir, mode=mode, progress=emit)
        model = model_identity(model_name)
        identity = {
            "root": os.path.normcase(str(root)),
            "model": model,
            "encoding": EMBEDDING_ENCODING_VERSION,
            "schema": SCHEMA_VERSION,
            "mode": mode,
            "parsers": _parser_identity(),
        }
        revision = _digest(
            {
                **identity,
                "content": [(file.relative_path, file.digest) for file in scan.files],
                "manifest": scan.manifest,
                "scan_issues": scan.issues,
            }
        )
        active_embedder = (
            embedder
            if embedder is not None
            else LocalEmbedder(
                model_name,
                cache_dir=cache_dir / "models",
                progress=emit,
            )
        )
        if (isinstance(active_embedder, LocalEmbedder)
                and not active_embedder.cache_dir.resolve().is_relative_to(cache_dir)):
            raise ValueError("local embedding cache must be inside the managed cache folder")
        if getattr(active_embedder, "index_model_identity", model) != model:
            raise ValueError("local embedding model changed; refresh again")
        cached = cls._load_cache(cache_path, identity, revision, root, scan)
        embedding_stats = {}
        if cached is not None:
            chunks, vectors, built_at, report = cached
            if chunks and isinstance(active_embedder, LocalEmbedder):
                active_embedder.prepare(progress=emit)
            emit(
                f"已复用本地索引：{report.indexed_files} 个文件，{report.chunks} 个片段。"
            )
        else:
            emit(f"解析 {len(scan.files)} 个材料文件。")
            chunks, report = parsing.parse(scan, progress=emit, mode=mode)
            checkpoint = None
            if mode == "full":
                vectors, checkpoint, embedding_stats = cls._embed_full(
                    cache_path, revision, chunks, active_embedder, emit, model
                )
            elif chunks:
                if isinstance(active_embedder, LocalEmbedder):
                    active_embedder.prepare(progress=emit)
                vectors = None
                for start in range(0, len(chunks), _EMBEDDING_BATCH_SIZE):
                    batch = chunks[start : start + _EMBEDDING_BATCH_SIZE]
                    encoded = _vectors(
                        active_embedder.encode([chunk.text for chunk in batch]),
                        len(batch),
                    )
                    if vectors is None:
                        required = len(chunks) * encoded.shape[1] * 4
                        if required > MAX_VECTOR_BYTES:
                            raise ValueError(
                                f"index vector size exceeds {MAX_VECTOR_BYTES} bytes"
                            )
                        vectors = np.empty(
                            (len(chunks), encoded.shape[1]), dtype=np.float32
                        )
                    elif vectors.shape[1] != encoded.shape[1]:
                        raise ValueError("embedding dimension changed during build")
                    vectors[start : start + len(batch)] = encoded
                    emit(
                        f"本地语义向量 {start + len(batch)}/{len(chunks)}；大资料库首次建立需要时间。"
                    )
                vectors.flags.writeable = False
            else:
                vectors = np.empty((0, 0), dtype=np.float32)
            # Detect additions/removals/edits during a long model load before publishing.
            if (
                parsing.scan(root, include_bytes=False, exclude=cache_dir, mode=mode, progress=emit).manifest
                != scan.manifest
            ):
                raise ValueError("library changed during build; refresh again")
            if model_identity(model_name) != model:
                raise ValueError("local embedding model changed during build; refresh again")
            built_at = datetime.now(UTC).isoformat()
            payload = {
                **identity,
                "revision": revision,
                "built_at": built_at,
                "chunks": chunks,
                "report": asdict(report),
            }
            old_vectors = None
            try:
                old_payload = (
                    json.loads(cache_path.read_text(encoding="utf-8"))
                    if cache_path.is_file()
                    else {}
                )
                if old_payload.get("vectors_file"):
                    old_vectors = vector_path(cache_path, old_payload["vectors_file"])
                del old_payload
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            if checkpoint is not None:
                vector_file = checkpoint.path
                vector_digest = file_digest(vector_file)
            else:
                vector_file, vector_digest = save_vectors(
                    cache_path, vectors, MAX_VECTOR_BYTES
                )
            payload.update(vectors_file=vector_file.name, vectors_digest=vector_digest)
            try:
                emit("提交完整资料快照。")
                _atomic_json(
                    cache_path, payload,
                    FULL_MAX_CACHE_BYTES if mode == "full" else MAX_CACHE_BYTES,
                    before_commit=check_cancelled,
                )
            except BaseException:
                if checkpoint is None:
                    vector_file.unlink(missing_ok=True)
                raise
            if checkpoint is not None:
                try:
                    checkpoint.discard_state(keep_vectors=True)
                except OSError:
                    # Metadata is already committed. A retained complete state
                    # remains valid for later recovery and must not undo success.
                    _LOGGER.warning(
                        "index committed but checkpoint cleanup failed; state retained for recovery",
                        exc_info=True,
                    )
            # Recovery after a previous commit can reuse its complete checkpoint.
            if old_vectors is not None and old_vectors != vector_file:
                try:
                    old_vectors.unlink(missing_ok=True)
                except OSError:
                    pass  # A stale owned cache file can be removed on a later refresh.
            prune_vectors(cache_path, published_file=vector_file.name)
            # Publication is the commit point. A later cancellation must not
            # report the already committed snapshot as an aborted refresh.
            if notify is not None:
                try:
                    notify(f"索引已完成：{report.indexed_files} 个文件，跳过 {report.skipped_files} 项。")
                except Exception:
                    _LOGGER.warning("index committed but final progress delivery failed", exc_info=True)
        result = cls(
            root=root,
            revision=revision,
            built_at=built_at,
            report=report,
            chunks=chunks,
            vectors=vectors,
            embedder=active_embedder,
            manifest=scan.manifest,
            cache_dir=cache_dir,
            mode=mode,
        )
        result._save_report(cache_path)
        result.embedding_stats = embedding_stats
        return result

    @staticmethod
    def _embed_full(cache_path, revision, chunks, embedder, emit, model):
        checkpoint = EmbeddingCheckpoint(
            cache_path, _digest({"revision": revision, "chunks": _chunk_digest(chunks)}), len(chunks), FULL_MAX_VECTOR_BYTES
        )
        content_cache = None
        try:
            emit(f"本地语义向量续建 {checkpoint.completed}/{len(chunks)}；每批完成后保存。")
            if chunks and isinstance(embedder, LocalEmbedder):
                embedder.prepare(progress=emit)
                try:
                    content_cache = CachedEmbedder(
                        embedder, cache_path.parent / "embedding-vectors.sqlite3",
                        _digest({"model": model, "encoding": EMBEDDING_ENCODING_VERSION}),
                    )
                except CacheLimitError:
                    # This is only a reuse accelerator. All chunk rows still
                    # enter the primary full index, computed by the delegate.
                    emit("共享向量复用缓存空间不足；继续为全部片段计算完整索引。")
                except sqlite3.DatabaseError:
                    emit("共享向量缓存损坏或不可用；继续为全部片段计算完整索引，可在缓存管理中清理共享缓存。")
            encoder = content_cache or embedder
            for start in range(checkpoint.completed, len(chunks), _EMBEDDING_BATCH_SIZE):
                emit(f"本地语义向量 {start}/{len(chunks)}；可中断后续建。")
                batch = chunks[start:start + _EMBEDDING_BATCH_SIZE]
                texts = [chunk.text for chunk in batch]
                try:
                    values = encoder.encode(texts)
                except sqlite3.DatabaseError:
                    if content_cache is None:
                        raise
                    content_cache.close()
                    content_cache = None
                    encoder = embedder
                    emit("共享向量缓存不可用；继续为全部片段计算完整索引，可在缓存管理中清理共享缓存。")
                    values = encoder.encode(texts)
                encoded = _vectors(values, len(batch))
                checkpoint.append(start, encoded)
                emit(f"本地语义向量 {checkpoint.completed}/{len(chunks)}；已保存进度。")
            stats = content_cache.stats if content_cache is not None else {}
            if stats:
                emit(f"向量复用 {stats['hits']} 行；实际计算 {stats['encoded_rows']} 行；全部来源位置保留。")
            return checkpoint.vectors(), checkpoint, stats
        finally:
            checkpoint.close()
            if content_cache is not None:
                content_cache.close()

    @staticmethod
    def _load_cache(
        path: Path, identity: dict, revision: str, root: Path, scan: parsing.Scan
    ):
        mode = identity.get("mode", "balanced")
        maximum = FULL_MAX_CACHE_BYTES if mode == "full" else MAX_CACHE_BYTES
        vector_maximum = FULL_MAX_VECTOR_BYTES if mode == "full" else MAX_VECTOR_BYTES
        if not path.is_file() or path.stat().st_size > maximum:
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # JSON turns tuple model manifests into lists; compare canonical fingerprints.
            if any(
                _digest(payload.get(key)) != _digest(value)
                for key, value in identity.items()
            ):
                return None
            if payload.get("revision") != revision:
                return None
            raw_chunks = payload["chunks"]
            chunk_maximum = parsing.FULL_MAX_CHUNKS if mode == "full" else parsing.MAX_CHUNKS
            if not isinstance(raw_chunks, list) or len(raw_chunks) > chunk_maximum:
                return None
            known_files = {file.relative_path for file in scan.files}
            validated_paths: set[str] = set()
            chunks = []
            for item in raw_chunks:
                chunk = KnowledgeChunk(**item)
                if not isinstance(chunk.relative_path, str):
                    return None
                if chunk.relative_path not in validated_paths:
                    relative = PurePosixPath(chunk.relative_path)
                    if (
                        relative.is_absolute()
                        or ".." in relative.parts
                        or "\\" in chunk.relative_path
                        or chunk.relative_path not in known_files
                        or not (root / chunk.relative_path).resolve().is_relative_to(root)
                    ):
                        return None
                    validated_paths.add(chunk.relative_path)
                if (
                    not isinstance(chunk.text, str)
                    or not chunk.text.strip()
                    or len(chunk.text) > parsing.MAX_CHARS
                    or not isinstance(chunk.locator, str)
                    or not chunk.locator
                ):
                    return None
                chunks.append(chunk)
            raw_vectors = load_vectors(
                path,
                payload["vectors_file"],
                payload["vectors_digest"],
                vector_maximum,
            )
            vectors = (
                _cached_vectors(raw_vectors, len(chunks))
                if chunks
                else np.empty((0, 0), dtype=np.float32)
            )
            raw_report = payload["report"]
            report = BuildReport(
                indexed_files=int(raw_report["indexed_files"]),
                skipped_files=int(raw_report["skipped_files"]),
                issues=tuple(raw_report["issues"]),
                chunks=int(raw_report["chunks"]),
                limited_files=int(raw_report.get("limited_files", 0)),
                duplicate_files=int(raw_report.get("duplicate_files", 0)),
            )
            if (
                report.chunks != len(chunks)
                or report.indexed_files < 0
                or report.skipped_files < 0
                or not 0 <= report.limited_files <= report.skipped_files
                or not 0 <= report.duplicate_files <= report.skipped_files
                or any(not isinstance(issue, str) for issue in report.issues)
            ):
                return None
            built_at = payload["built_at"]
            datetime.fromisoformat(built_at)
            return tuple(chunks), vectors, built_at, report
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return None

    def is_stale(self) -> bool:
        """Cheap stat check; explicit refresh performs the stronger content hash check."""
        try:
            return (
                parsing.scan(
                    self.root,
                    include_bytes=False,
                    exclude=self._cache_dir,
                    mode=self.mode,
                ).manifest
                != self._manifest
            )
        except (OSError, ValueError):
            return True

    async def search_async(self, query: str, limit: int = 5) -> tuple[KnowledgeHit, ...]:
        """Keep ownership until CPU work ends, even if its UI waiter is canceled."""
        await self._retrieval_gate.acquire()
        try:
            worker = asyncio.create_task(asyncio.to_thread(self.search, query, limit))
        except BaseException:
            self._retrieval_gate.release()
            raise

        def completed(task):
            self._retrieval_gate.release()
            if not task.cancelled():
                task.exception()  # Consume a late failure after its waiter was canceled.

        worker.add_done_callback(completed)
        return await asyncio.shield(worker)

    def search(self, query: str, limit: int = 5) -> tuple[KnowledgeHit, ...]:
        if not query.strip() or limit <= 0 or not self._chunks:
            return ()
        if len(query) > 20000:
            raise ValueError("retrieval query exceeds 20000 characters")
        background, marker, question = query.rpartition("\n问题：")
        context = background.removeprefix("背景信息：").strip()
        contextual = bool(
            marker and question.strip() and background.startswith("背景信息：") and context
        )
        primary_query = question.strip() if contextual else query
        if contextual:
            # The detector joins recent utterances with this separator. Older
            # topics must not dominate either the current semantic query or
            # the continuity channel when the conversation moves on.
            context = next((part.strip() for part in reversed(context.split("；")) if part.strip()), "")
        anchor_budget = min(2, limit // 2) if (
            contextual and context and continues_topic(primary_query, context)
        ) else 0
        queries = [primary_query, context] if anchor_budget else [primary_query]
        query_vectors = _vectors(self._embedder.encode(queries, query=True), len(queries))
        if query_vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError("query embedding dimension differs from the index")
        semantic = np.clip(self._vectors @ query_vectors[0], 0.0, 1.0)
        lexical = self._lexical.scores(primary_query)
        selected = _channel_candidates(semantic, lexical, limit - anchor_budget)
        if anchor_budget:
            # A short follow-up can rank generic limitations ahead of the
            # preceding subject's concrete specification. Keep the current
            # query's majority, reserving at most two of the same five evidence
            # slots for that local subject. This is candidate continuity, not
            # a claim that prior speech or a positive score is factual evidence.
            primary = selected
            chosen = {i for i, _ in primary}
            anchors = _channel_candidates(
                np.clip(self._vectors @ query_vectors[1], 0.0, 1.0),
                self._lexical.scores(context), anchor_budget, excluded=chosen,
            )
            primary.extend(anchors)
            chosen.update(i for i, _ in anchors)
            for i, score in _channel_candidates(semantic, lexical, limit):
                if len(primary) >= limit:
                    break
                if i not in chosen:
                    primary.append((i, score))
                    chosen.add(i)
            selected = primary
        return tuple(
            KnowledgeHit(
                source=self._chunks[i].source,
                text=self._chunks[i].text,
                path=self.root / self._chunks[i].relative_path,
                locator=self._chunks[i].locator,
                score=score,
                section_context=section[0],
                section_source=section[1],
            )
            for i, score in selected
            for section in (self._sections.for_row(i),)
        )
