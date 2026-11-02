"""Bounded, non-executing document readers and source-preserving chunks."""
from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .policy import FullIndexLimitError, validate_index_mode
from .types import BuildReport, KnowledgeChunk

PARSING_VERSION = "text-normalization-v3"

# Keep whitespace (including form feeds and Unicode line separators) intact so
# source line locators retain their meaning. Only sparse, non-text controls are
# eligible for visible replacement; a mostly binary file remains unsupported.
_NON_TEXT_CONTROLS = re.compile(r"[\x00-\x08\x0e-\x1b\x7f-\x84\x86-\x9f]")
_UNDECODABLE_UTF8 = re.compile(r"[\udc80-\udcff]")
_BINARY_SIGNATURES = (
    b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a",
    b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08", b"\x7fELF",
    b"%PDF-", b"SQLite format 3\x00", b"\x1f\x8b",
)
_MAX_TEXT_DAMAGE_FRACTION = 0.01
_MIN_READABLE_CHARS = 100

MAX_FILES = 5000
MAX_ENTRIES = 20000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_CHUNKS = 50000
_HASH_READ_BYTES = 1024 * 1024
MAX_CHARS = 1000
MAX_DOCUMENT_CHARS = 5_000_000
FULL_MAX_FILES = 400_000
FULL_MAX_ENTRIES = 2_000_000
FULL_MAX_FILE_BYTES = 128 * 1024 * 1024
FULL_MAX_TOTAL_BYTES = 16 * 1024 * 1024 * 1024
FULL_MAX_CHUNKS = 2_000_000
FULL_MAX_DOCUMENT_CHARS = 128_000_000
CODE_SUFFIXES = frozenset({
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".c", ".cc", ".cpp", ".h", ".hpp",
    ".java", ".go", ".rs", ".rb", ".php", ".cs", ".swift", ".kt", ".scala", ".sh",
    ".ps1", ".sql", ".r", ".m", ".vue", ".svelte", ".html", ".css", ".scss",
})
TEXT_SUFFIXES = CODE_SUFFIXES | frozenset({
    ".md", ".markdown", ".txt", ".rst", ".json", ".yaml", ".yml", ".toml", ".tex",
    ".csv", ".tsv", ".xml", ".ini", ".cfg", ".ipynb", ".bib", ".log",
})
DOCUMENT_SUFFIXES = frozenset({".docx", ".pdf", ".pptx"})
_DEPENDENCY_DIRS = frozenset({
    "node_modules", "vendor", "venv", "__pycache__", "coverage", "site-packages", "bower_components",
})
_SKIP_DIRS = _DEPENDENCY_DIRS | frozenset({
    "build", "dist", "target", "out", "bin", "obj", "models", "tmp", "temp",
})
_SECRET_NAMES = frozenset({
    "credentials", "credential", "secrets", "secret", "passwords", "password", "id_rsa",
    "id_ed25519", "private_key", "api_key", "apikey", "client_secret", "token", "tokens",
    "service_account",
})
_SECRET_DIR_NAMES = _SECRET_NAMES | frozenset({
    "private_keys", "ssh_keys", "ssh", "gnupg", "gpg", "credential_store", "secret_store",
})


@dataclass(frozen=True, slots=True)
class FileSnapshot:
    relative_path: str
    data: bytes | None
    digest: str
    path: Path | None = None
    size: int = 0
    mtime_ns: int = 0


@dataclass(frozen=True, slots=True)
class Scan:
    files: tuple[FileSnapshot, ...]
    manifest: tuple[tuple[str, int, int], ...]
    skipped: int
    issues: tuple[str, ...]
    limited_files: int = 0
    duplicate_files: int = 0
    mode: str = "balanced"


@dataclass(frozen=True, slots=True)
class _Limits:
    files: int
    entries: int
    file_bytes: int
    total_bytes: int
    chunks: int
    document_chars: int


def _limits(mode: str) -> _Limits:
    validate_index_mode(mode)
    if mode == "full":
        return _Limits(
            FULL_MAX_FILES, FULL_MAX_ENTRIES, FULL_MAX_FILE_BYTES,
            FULL_MAX_TOTAL_BYTES, FULL_MAX_CHUNKS, FULL_MAX_DOCUMENT_CHARS,
        )
    return _Limits(
        MAX_FILES, MAX_ENTRIES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_CHUNKS, MAX_DOCUMENT_CHARS,
    )


def _full_limit(relative_path: str, label: str, limit: int) -> FullIndexLimitError:
    return FullIndexLimitError(
        f"{relative_path}: 全量索引达到{label}限制 {limit}；构建已停止，请调整资料或限制后重试",
    )


class FileChangedError(ValueError):
    """A source changed after its scan; do not publish this build."""


class _ExtractionLimitError(ValueError):
    """Skip a whole file that cannot fit the extraction budget."""


def _file_priority(relative_path: str) -> tuple[int, int, str, str]:
    path = Path(relative_path)
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown", ".txt", ".rst", ".tex", ".docx", ".pptx", ".bib"}:
        priority = 0
    elif suffix in CODE_SUFFIXES:
        priority = 1
    elif suffix == ".pdf":
        priority = 2
    else:
        priority = 3
    return priority, len(path.parts), relative_path.casefold(), relative_path


def _content_identity(relative_path: str, digest: str) -> tuple[str, str]:
    # Identical bytes can mean different documents to different format readers.
    return Path(relative_path).suffix.lower(), digest


def _changed(relative_path: str) -> FileChangedError:
    return FileChangedError(f"{relative_path}: file changed during build; refresh again")


def _metadata_matches(snapshot: FileSnapshot) -> bool:
    if snapshot.path is None or _linked(snapshot.path):
        return False
    info = snapshot.path.stat()
    return (stat.S_ISREG(info.st_mode) and info.st_size == snapshot.size
            and info.st_mtime_ns == snapshot.mtime_ns)


def _hash_file(snapshot: FileSnapshot) -> str:
    """Hash one source with constant memory and reject concurrent changes."""
    try:
        if not _metadata_matches(snapshot):
            raise _changed(snapshot.relative_path)
        digest = hashlib.sha256()
        read_bytes = 0
        assert snapshot.path is not None
        with snapshot.path.open("rb") as source:
            while data := source.read(_HASH_READ_BYTES):
                read_bytes += len(data)
                if read_bytes > snapshot.size:
                    raise _changed(snapshot.relative_path)
                digest.update(data)
        if read_bytes != snapshot.size or not _metadata_matches(snapshot):
            raise _changed(snapshot.relative_path)
        return digest.hexdigest()
    except OSError as error:
        raise _changed(snapshot.relative_path) from error


def _excluded(path: Path, *, directory: bool = False, mode: str = "balanced") -> bool:
    name = path.name.lower()
    attributes = getattr(path.lstat(), "st_file_attributes", 0)
    if name.startswith(".") or attributes & stat.FILE_ATTRIBUTE_HIDDEN:
        return True
    if directory:
        normalized = name.replace("-", "_").replace(".", "_")
        skip_directory = name in (_DEPENDENCY_DIRS if mode == "full" else _SKIP_DIRS)
        return (skip_directory or normalized in _SECRET_DIR_NAMES
                or any(normalized.startswith(token + "_") for token in _SECRET_NAMES))
    stem = path.stem.lower().replace("-", "_").replace(".", "_")
    return (stem in _SECRET_NAMES or any(stem.startswith(token + "_")
            for token in _SECRET_NAMES) or name.endswith((".pem", ".key", ".p12", ".pfx")))


def _linked(path: Path) -> bool:
    attributes = path.lstat()
    return path.is_symlink() or bool(
        getattr(attributes, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    )


def scan(
    root: Path, *, include_bytes: bool = True, exclude: Path | None = None,
    mode: str = "balanced",
    progress: Callable[[str], None] | None = None,
) -> Scan:
    """Collect metadata and streamed digests, never retaining source bytes.

    ``include_bytes=False`` is the cheap metadata-only stale check. The name is
    retained for callers of the original scanner.
    """
    limits = _limits(mode)
    root = root.resolve()
    exclude = exclude.resolve() if exclude is not None else None
    files = []
    candidates = []
    manifest = []
    issues = []
    skipped = 0
    entries = 0
    total_bytes = 0
    limited = 0
    duplicates = 0
    canonical_paths: dict[tuple[str, str], str] = {}

    def walk_error(error: OSError) -> None:
        raise OSError(f"cannot scan library: {error}") from error

    for current, directories, filenames in os.walk(root, followlinks=False, onerror=walk_error):
        if progress is not None:
            progress(f"扫描资料目录：{Path(current).relative_to(root).as_posix()}")
        current_path = Path(current)
        for name in sorted(directories[:]):
            path = current_path / name
            entries += 1
            if entries > limits.entries:
                if mode == "full":
                    raise _full_limit(path.relative_to(root).as_posix(), "目录项数量", limits.entries)
                raise ValueError(f"library entry count exceeds {limits.entries}")
            if (path == exclude or _excluded(path, directory=True, mode=mode) or _linked(path)
                    or not path.resolve().is_relative_to(root)):
                directories.remove(name)
                skipped += 1
                policy = " [全量策略：依赖目录]" if mode == "full" and name.lower() in _DEPENDENCY_DIRS else ""
                issues.append(
                    f"{path.relative_to(root).as_posix()}/: excluded folder/link "
                    f"(contents not scanned){policy}",
                )
        directories.sort()
        for name in sorted(filenames):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            entries += 1
            if entries > limits.entries:
                if mode == "full":
                    raise _full_limit(relative, "目录项数量", limits.entries)
                raise ValueError(f"library entry count exceeds {limits.entries}")
            if _excluded(path) or _linked(path) or not path.resolve().is_relative_to(root):
                skipped += 1
                issues.append(f"{relative}: excluded hidden/credential/link file")
                continue
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                skipped += 1
                issues.append(f"{relative}: unsupported special file")
                continue
            manifest.append((relative, info.st_size, info.st_mtime_ns))
            if len(manifest) > limits.files:
                if mode == "full":
                    raise _full_limit(relative, "文件数量", limits.files)
                raise ValueError(f"library file count exceeds {limits.files}")
            if path.suffix.lower() not in TEXT_SUFFIXES | DOCUMENT_SUFFIXES:
                skipped += 1
                issues.append(f"{relative}: unsupported file type")
                continue
            if info.st_size > limits.file_bytes:
                if mode == "full":
                    raise _full_limit(relative, "单文件字节数", limits.file_bytes)
                skipped += 1
                limited += 1
                issues.append(f"{relative}: file size exceeds {limits.file_bytes} bytes")
                continue
            if include_bytes:
                candidates.append(FileSnapshot(
                    relative, None, "", path, info.st_size, info.st_mtime_ns,
                ))
    for snapshot in sorted(candidates, key=lambda item: _file_priority(item.relative_path)):
        if progress is not None:
            progress(f"校验材料：{snapshot.relative_path}")
        digest = _hash_file(snapshot)
        identity = _content_identity(snapshot.relative_path, digest)
        canonical = canonical_paths.get(identity)
        if canonical is not None:
            skipped += 1
            duplicates += 1
            issues.append(
                f"{snapshot.relative_path}: duplicate content; canonical source: {canonical}",
            )
            continue
        if total_bytes + snapshot.size > limits.total_bytes:
            if mode == "full":
                raise _full_limit(snapshot.relative_path, "总字节数", limits.total_bytes)
            skipped += 1
            limited += 1
            issues.append(
                f"{snapshot.relative_path}: library total byte budget exceeds "
                f"{limits.total_bytes} bytes; whole file skipped",
            )
            continue
        canonical_paths[identity] = snapshot.relative_path
        total_bytes += snapshot.size
        files.append(FileSnapshot(
            snapshot.relative_path, None, digest, snapshot.path, snapshot.size, snapshot.mtime_ns,
        ))
    return Scan(tuple(files), tuple(sorted(manifest)), skipped, tuple(issues), limited, duplicates, mode)


def _parts(text: str, locator: str):
    if len(text) <= MAX_CHARS:
        if text.strip():
            yield locator, text.strip()
        return
    for start in range(0, len(text), MAX_CHARS):
        value = text[start:start + MAX_CHARS].strip()
        if value:
            yield f"{locator}, chars {start + 1}–{min(start + MAX_CHARS, len(text))}", value


def _text_chunks(text: str, *, code: bool):
    lines = text.splitlines()
    start = 1
    group: list[str] = []
    size = 0
    maximum_lines = 50 if code else 80
    for number, line in enumerate(lines, 1):
        if len(line) > MAX_CHARS:
            if group:
                value = "\n".join(group).strip()
                if value:
                    yield f"lines {start}–{number - 1}", value
                group, size = [], 0
            yield from _parts(line, f"line {number}")
            start = number + 1
            continue
        if group and (len(group) >= maximum_lines or size + len(line) + 1 > MAX_CHARS):
            value = "\n".join(group).strip()
            if value:
                yield f"lines {start}–{number - 1}", value
            group, size, start = [], 0, number
        if not group:
            start = number
        group.append(line)
        size += len(line) + (1 if len(group) > 1 else 0)
    if group and "\n".join(group).strip():
        yield f"lines {start}–{len(lines)}", "\n".join(group).strip()


def _check_zip(data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) > 5000 or sum(item.file_size for item in entries) > 64 * 1024 * 1024:
            raise _ExtractionLimitError("expanded document size exceeds safety limit")


def _slide_chunks(shapes, locator: str, *, depth: int = 0):
    if depth > 25:
        raise _ExtractionLimitError("slide group nesting exceeds 25 levels")
    for shape_number, shape in enumerate(shapes, 1):
        if hasattr(shape, "shapes"):
            yield from _slide_chunks(
                shape.shapes, f"{locator}, group {shape_number}", depth=depth + 1,
            )
        if shape.has_text_frame:
            yield from _parts(shape.text_frame.text, f"{locator}, shape {shape_number}")
        if shape.has_table:
            for row_number, row in enumerate(shape.table.rows, 1):
                yield from _parts(
                    " | ".join(cell.text for cell in row.cells),
                    f"{locator}, table shape {shape_number}, row {row_number}",
                )


def _snapshot_data(snapshot: FileSnapshot, *, mode: str = "balanced") -> bytes:
    limits = _limits(mode)
    source_size = len(snapshot.data) if snapshot.data is not None else snapshot.size
    if mode == "full" and source_size > limits.file_bytes:
        raise _full_limit(snapshot.relative_path, "单文件字节数", limits.file_bytes)
    if snapshot.data is not None:
        return snapshot.data
    try:
        if not _metadata_matches(snapshot):
            raise _changed(snapshot.relative_path)
        assert snapshot.path is not None
        with snapshot.path.open("rb") as source:
            data = source.read(limits.file_bytes + 1)
        if (len(data) != snapshot.size or not _metadata_matches(snapshot)
                or hashlib.sha256(data).hexdigest() != snapshot.digest):
            raise _changed(snapshot.relative_path)
        return data
    except OSError as error:
        raise _changed(snapshot.relative_path) from error


def _pdf_pages(data: bytes):
    try:
        import pymupdf
    except ImportError:
        # Older installations can still read PDFs until dependencies update.
        from pypdf import PdfReader
        document = PdfReader(io.BytesIO(data))
        if document.is_encrypted:
            raise ValueError("encrypted PDF is unsupported")
        if len(document.pages) > 2000:
            raise _ExtractionLimitError("PDF page count exceeds 2000")
        for number, page in enumerate(document.pages, 1):
            yield number, page.extract_text() or ""
    else:
        with pymupdf.open(stream=data, filetype="pdf") as document:
            if document.needs_pass:
                raise ValueError("encrypted PDF is unsupported")
            if document.page_count > 2000:
                raise _ExtractionLimitError("PDF page count exceeds 2000")
            for number, page in enumerate(document, 1):
                yield number, page.get_text("text", sort=True) or ""


def _decode_text(data: bytes, relative_path: str, issues: list[str]) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16")
    else:
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = data.decode("gb18030")
            except UnicodeDecodeError:
                # Preserve valid UTF-8 spans; surrogateescape identifies each
                # original undecodable byte without guessing another encoding.
                text = data.decode("utf-8-sig", errors="surrogateescape")

    controls = _NON_TEXT_CONTROLS.findall(text)
    damaged = _UNDECODABLE_UTF8.findall(text)
    if not controls and not damaged:
        return text
    unreadable = sum(not (char.isprintable() or char.isspace()) for char in text)
    if (data.startswith(_BINARY_SIGNATURES)
            or len(text) - unreadable < _MIN_READABLE_CHARS
            or unreadable / max(1, len(text)) > _MAX_TEXT_DAMAGE_FRACTION):
        raise ValueError("binary or insufficiently readable data is not a text document")

    text = _NON_TEXT_CONTROLS.sub(lambda match: f" [U+{ord(match[0]):04X}] ", text)
    text = _UNDECODABLE_UTF8.sub(lambda match: f"\\x{ord(match[0]) - 0xDC00:02x}", text)
    details = []
    if controls:
        examples = ",".join(f"U+{ord(char):04X}" for char in sorted(set(controls))[:8])
        details.append(f"controls={len(controls)} ({examples})")
    if damaged:
        examples = "".join(f"\\x{ord(char) - 0xDC00:02x}" for char in damaged[:8])
        details.append(f"undecodable_utf8_bytes={len(damaged)} ({examples})")
    issues.append(f"{relative_path}: text normalization: {'; '.join(details)}")
    return text


def _read_document(snapshot: FileSnapshot, issues: list[str], *, mode: str = "balanced"):
    suffix = Path(snapshot.relative_path).suffix.lower()
    data = _snapshot_data(snapshot, mode=mode)
    if suffix in TEXT_SUFFIXES:
        text = _decode_text(data, snapshot.relative_path, issues)
        yield from _text_chunks(text, code=suffix in CODE_SUFFIXES)
    elif suffix == ".docx":
        _check_zip(data)
        from docx import Document
        document = Document(io.BytesIO(data))
        for number, paragraph in enumerate(document.paragraphs, 1):
            yield from _parts(paragraph.text, f"paragraph {number}")
        for number, table in enumerate(document.tables, 1):
            for row_number, row in enumerate(table.rows, 1):
                yield from _parts(" | ".join(cell.text for cell in row.cells), f"table {number}, row {row_number}")
    elif suffix == ".pdf":
        for number, text in _pdf_pages(data):
            if not text.strip():
                issues.append(f"{snapshot.relative_path}: page {number} has no extractable text (scanned/blank; OCR not supported)")
            yield from _parts(text, f"page {number}")
    elif suffix == ".pptx":
        _check_zip(data)
        from pptx import Presentation
        document = Presentation(io.BytesIO(data))
        if len(document.slides) > 2000:
            raise _ExtractionLimitError("slide count exceeds 2000")
        for number, slide in enumerate(document.slides, 1):
            yield from _slide_chunks(slide.shapes, f"slide {number}")


def parse(
    scan_result: Scan, *, progress: Callable[[str], None] | None = None,
    mode: str = "balanced",
) -> tuple[tuple[KnowledgeChunk, ...], BuildReport]:
    limits = _limits(mode)
    if mode == "full" and scan_result.limited_files:
        detail = next((issue for issue in scan_result.issues if "exceed" in issue), "资料库")
        raise FullIndexLimitError(
            f"{detail}: 全量索引拒绝已有扫描限制的结果；请使用全量模式重新扫描资料库",
        )
    chunks: list[KnowledgeChunk] = []
    issues = list(scan_result.issues)
    indexed = 0
    skipped = scan_result.skipped
    limited = scan_result.limited_files
    duplicates = scan_result.duplicate_files
    canonical_paths: dict[tuple[str, str], str] = {}
    ordered = sorted(scan_result.files, key=lambda item: _file_priority(item.relative_path))
    for number, snapshot in enumerate(ordered, 1):
        if progress is not None:
            progress(f"解析资料 {number}/{len(ordered)}: {snapshot.relative_path}")
        identity = _content_identity(snapshot.relative_path, snapshot.digest)
        canonical = canonical_paths.get(identity)
        if canonical is not None:
            skipped += 1
            duplicates += 1
            issues.append(f"{snapshot.relative_path}: duplicate content; canonical source: {canonical}")
            continue
        canonical_paths[identity] = snapshot.relative_path
        try:
            file_chunks = []
            characters = 0
            for locator, text in _read_document(snapshot, issues, mode=mode):
                if not text.strip():
                    continue
                characters += len(text)
                if characters > limits.document_chars:
                    if mode == "full":
                        raise _full_limit(snapshot.relative_path, "文档字符数", limits.document_chars)
                    raise _ExtractionLimitError("document text size exceeds extraction limit")
                file_chunks.append(KnowledgeChunk(snapshot.relative_path, locator, text))
                if len(chunks) + len(file_chunks) > limits.chunks:
                    if mode == "full":
                        raise _full_limit(snapshot.relative_path, "分块数量", limits.chunks)
                    raise _ExtractionLimitError(
                        f"library chunk count exceeds {limits.chunks}; whole file skipped",
                    )
            if file_chunks:
                chunks.extend(file_chunks)
                indexed += 1
            else:
                skipped += 1
                issues.append(f"{snapshot.relative_path}: empty/no extractable text")
        except (FileChangedError, FullIndexLimitError):
            raise
        except _ExtractionLimitError as error:
            if mode == "full":
                raise FullIndexLimitError(
                    f"{snapshot.relative_path}: 全量索引达到解析安全限制：{error}；构建已停止",
                ) from error
            skipped += 1
            limited += 1
            issues.append(f"{snapshot.relative_path}: extraction limit: {error}")
        except ValueError as error:
            skipped += 1
            issues.append(f"{snapshot.relative_path}: parse failed: {error}")
        except Exception as error:  # noqa: BLE001 - isolate third-party parser failures per file.
            skipped += 1
            issues.append(f"{snapshot.relative_path}: parse failed ({type(error).__name__}): {error}")
    return tuple(chunks), BuildReport(
        indexed, skipped, tuple(issues), len(chunks), limited, duplicates,
    )
