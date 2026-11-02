"""Bounded binary vectors with atomic metadata as the commit point."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import stat
import struct
import tempfile
import zipfile
from pathlib import Path
from uuid import uuid4

import numpy as np

from .atomic import replace_atomic
from .cache_manager import require_cache_space
from .quota_io import BinaryQuotaWriter

MAX_NPY_HEADER_BYTES = 4096
_MAX_PRUNE_METADATA_BYTES = 64 * 1024 * 1024
_MAX_PRUNE_CHECKPOINT_BYTES = 64 * 1024


def _is_plain_file(info: os.stat_result) -> bool:
    return (
        stat.S_ISREG(info.st_mode)
        and info.st_nlink == 1
        and not getattr(info, "st_file_attributes", 0)
        & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _reference_vector(path: Path, owned: re.Pattern, maximum: int) -> str | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        return None
    if not _is_plain_file(info) or info.st_size > maximum:
        raise ValueError("vector reference metadata cannot be safely read")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("vector reference metadata size exceeds limit")
    payload = json.loads(raw)
    filename = payload.get("vectors_file") if isinstance(payload, dict) else None
    if not isinstance(filename, str) or owned.fullmatch(filename) is None:
        raise ValueError("invalid vector reference metadata")
    return filename


def prune_vectors(metadata: Path, *, published_file: str | None = None) -> int:
    """Best-effort cleanup of this index's unreferenced generated vector files.

    The caller must hold BuildLock. ``published_file`` may only name the trusted
    current reference immediately after its metadata commit; it avoids rereading
    large metadata. Otherwise metadata reads are capped at 64 MiB and checkpoint
    reads at 64 KiB. Uncertain references disable cleanup. Open Windows mappings
    and links are retained and can be retried on a later successful publication.
    """
    metadata = Path(metadata)
    owned = re.compile(re.escape(metadata.stem) + r"-[0-9a-f]{32}\.(?:npy|npz)")
    protected = set()
    try:
        parent_info = metadata.parent.lstat()
        if not stat.S_ISDIR(parent_info.st_mode) or getattr(
            parent_info, "st_file_attributes", 0
        ) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0):
            return 0
        if published_file is None:
            current = _reference_vector(metadata, owned, _MAX_PRUNE_METADATA_BYTES)
        elif isinstance(published_file, str) and owned.fullmatch(published_file):
            current = published_file
        else:
            return 0
        checkpoint = _reference_vector(
            metadata.with_name(f".{metadata.name}.checkpoint.json"),
            owned,
            _MAX_PRUNE_CHECKPOINT_BYTES,
        )
        protected.update(value for value in (current, checkpoint) if value is not None)
    except (OSError, ValueError, TypeError, RecursionError):
        return 0
    removed = 0
    try:
        with os.scandir(metadata.parent) as entries:
            for entry in entries:
                if entry.name in protected or owned.fullmatch(entry.name) is None:
                    continue
                try:
                    path = Path(entry.path)
                    # DirEntry.stat on Windows can report st_nlink=0 from cached
                    # directory data; lstat obtains the actual hard-link count.
                    if _is_plain_file(path.lstat()):
                        path.unlink()
                        removed += 1
                except OSError:
                    # A live mmap can keep an old snapshot open on Windows.
                    continue
    except OSError:
        pass
    return removed


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_vectors(metadata: Path, vectors: np.ndarray, maximum: int) -> tuple[Path, str]:
    if vectors.nbytes > maximum:
        raise ValueError(f"index vector size exceeds {maximum} bytes")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    target = metadata.with_name(f"{metadata.stem}-{uuid4().hex}.npz")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=metadata.parent, prefix=f".{metadata.stem}.vectors.", suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            np.savez_compressed(BinaryQuotaWriter(output, metadata), vectors=np.asarray(vectors, dtype=np.float32))
            output.flush()
            os.fsync(output.fileno())
        digest = file_digest(temporary)
        replace_atomic(temporary, target)
        return target, digest
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def vector_path(metadata: Path, filename: str) -> Path:
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or "/" in filename
        or "\\" in filename
        or ":" in filename
        or not filename.startswith(metadata.stem + "-")
        or not filename.endswith((".npz", ".npy"))
    ):
        raise ValueError("invalid vector cache path")
    return metadata.parent / filename


def _read_vector_header(
    stream, maximum: int, size: int, *, allow_fortran: bool = False
) -> tuple[tuple[int, int], int]:
    """Bound the header and dimensions before NumPy can allocate or map anything."""
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        length_size, length_format = 2, "<H"
        reader = np.lib.format.read_array_header_1_0
    elif version == (2, 0):
        length_size, length_format = 4, "<I"
        reader = np.lib.format.read_array_header_2_0
    else:
        raise ValueError("invalid vector cache format")
    length_bytes = stream.read(length_size)
    if len(length_bytes) != length_size:
        raise ValueError("invalid vector cache format")
    header_length = struct.unpack(length_format, length_bytes)[0]
    if header_length > MAX_NPY_HEADER_BYTES:
        raise ValueError("vector cache header size exceeds limit")
    header = stream.read(header_length)
    if len(header) != header_length:
        raise ValueError("invalid vector cache format")
    shape, fortran, dtype = reader(
        io.BytesIO(length_bytes + header), max_header_size=MAX_NPY_HEADER_BYTES
    )
    if (
        len(shape) != 2
        or any(type(value) is not int or value < 0 for value in shape)
        or dtype != np.dtype(np.float32)
        or (fortran and not allow_fortran)
        or ((shape[0] == 0) != (shape[1] == 0))
    ):
        raise ValueError("invalid vector cache format")
    expected_bytes = shape[0] * shape[1] * np.dtype(np.float32).itemsize
    if expected_bytes > maximum:
        raise ValueError(f"index vector size exceeds {maximum} bytes")
    offset = stream.tell()
    if size != offset + expected_bytes:
        raise ValueError("vector cache size does not match its shape")
    return shape, offset


def load_mapped_vectors(
    path: Path,
    maximum: int,
    *,
    writable: bool = False,
    expected_shape: tuple[int, int] | None = None,
) -> np.memmap:
    """Map a validated native float32 matrix, without copying the whole matrix."""
    if path.is_symlink() or (writable and not _is_plain_file(path.lstat())):
        raise ValueError("invalid vector cache path")
    size = path.stat().st_size
    if size > maximum + MAX_NPY_HEADER_BYTES + 12:
        raise ValueError("vector cache size exceeds limit")
    with path.open("rb") as stream:
        shape, offset = _read_vector_header(stream, maximum, size)
    if expected_shape is not None and shape != expected_shape:
        raise ValueError("vector cache shape differs from checkpoint")
    return np.memmap(
        path,
        dtype=np.float32,
        mode="r+" if writable else "r",
        offset=offset,
        shape=shape,
        order="C",
    )


def create_mapped_vectors(
    metadata: Path, rows: int, dimension: int, maximum: int
) -> tuple[Path, np.memmap]:
    """Reserve one uniquely owned .npy file; its caller commits metadata later."""
    if (
        type(rows) is not int
        or type(dimension) is not int
        or rows < 0
        or dimension < 0
        or (rows > 0 and dimension == 0)
    ):
        raise ValueError("invalid vector cache shape")
    required = rows * dimension * np.dtype(np.float32).itemsize
    if required > maximum:
        raise ValueError(f"index vector size exceeds {maximum} bytes")
    metadata.parent.mkdir(parents=True, exist_ok=True)
    path = metadata.with_name(f"{metadata.stem}-{uuid4().hex}.npy")
    header = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        header,
        {
            "descr": "<f4" if np.little_endian else ">f4",
            "fortran_order": False,
            "shape": (rows, dimension),
        },
    )
    require_cache_space(metadata, len(header.getvalue()) + required)
    try:
        with path.open("xb") as stream:
            stream.write(header.getvalue())
            stream.truncate(stream.tell() + required)
            stream.flush()
            os.fsync(stream.fileno())
        return path, load_mapped_vectors(path, maximum, writable=True)
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def load_vectors(
    metadata: Path, filename: str, digest: str, maximum: int
) -> np.ndarray:
    path = vector_path(metadata, filename)
    if path.is_symlink() or path.stat().st_size > maximum + 65536:
        raise ValueError("vector cache size exceeds limit")
    if file_digest(path) != digest:
        raise ValueError("vector cache digest mismatch")
    if path.suffix == ".npy":
        return load_mapped_vectors(path, maximum)
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            if (
                len(entries) != 1
                or entries[0].filename != "vectors.npy"
                or entries[0].file_size > maximum + 4096
            ):
                raise ValueError("expanded vector cache exceeds limit")
            with archive.open(entries[0]) as stream:
                _read_vector_header(
                    stream, maximum, entries[0].file_size, allow_fortran=True
                )
        with np.load(path, allow_pickle=False) as archive:
            vectors = archive["vectors"]
    except zipfile.BadZipFile as error:
        raise ValueError("invalid vector cache format") from error
    if vectors.dtype != np.float32 or vectors.nbytes > maximum:
        raise ValueError("invalid vector cache format")
    return vectors
