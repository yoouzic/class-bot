"""Synthetic regressions for bounded library scans and complete-file parsing."""
import hashlib
import os
from pathlib import Path

import pytest

from classbot.knowledge import parsing


def test_scan_retains_metadata_and_hashes_without_retaining_document_bytes(tmp_path):
    source = tmp_path / "notes.md"
    source.write_text("可按需读取的材料", encoding="utf-8")

    result = parsing.scan(tmp_path)

    snapshot = result.files[0]
    assert snapshot.data is None
    assert snapshot.path == source
    assert snapshot.size == source.stat().st_size
    assert snapshot.mtime_ns == source.stat().st_mtime_ns
    assert snapshot.digest == hashlib.sha256(source.read_bytes()).hexdigest()
    chunks, report = parsing.parse(result)
    assert chunks[0].text == "可按需读取的材料"
    assert report.indexed_files == 1


def test_scan_hashes_in_bounded_reads(tmp_path, monkeypatch):
    source = tmp_path / "notes.txt"
    source.write_text("a" * (1024 * 1024 + 12), encoding="utf-8")
    original_open = Path.open
    read_sizes = []

    class BoundedReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def read(self, size=-1):
            read_sizes.append(size)
            assert 0 < size <= 1024 * 1024
            return self.stream.read(size)

    def open_bounded(path, *args, **kwargs):
        return BoundedReader(original_open(path, *args, **kwargs))

    monkeypatch.setattr(Path, "open", open_bounded)
    parsing.scan(tmp_path)
    assert len(read_sizes) >= 2


def test_metadata_scan_skips_reads_and_total_byte_budget(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("123456", encoding="utf-8")
    (tmp_path / "b.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "unsupported.bin").write_bytes(b"unsupported")
    monkeypatch.setattr(parsing, "MAX_TOTAL_BYTES", 8)

    def no_reads(*args, **kwargs):
        pytest.fail("metadata scan must not read file contents")

    monkeypatch.setattr(Path, "open", no_reads)
    result = parsing.scan(tmp_path, include_bytes=False)
    assert {item[0] for item in result.manifest} == {"a.md", "b.md", "unsupported.bin"}


def test_total_budget_skips_whole_file_and_allows_later_smaller_files(tmp_path, monkeypatch):
    for name, text in (("a.md", "123456"), ("b.md", "abcdef"), ("c.md", "xy")):
        (tmp_path / name).write_text(text, encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_TOTAL_BYTES", 8)

    result = parsing.scan(tmp_path)
    chunks, report = parsing.parse(result)

    assert {chunk.relative_path for chunk in chunks} == {"a.md", "c.md"}
    assert len(result.manifest) == 3
    assert report.limited_files == 1
    assert report.skipped_files == 1
    assert any("b.md" in issue and "total" in issue for issue in report.issues)


def test_duplicate_copies_do_not_consume_byte_budget_for_unique_documents(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "b.md").write_text("abcdef", encoding="utf-8")
    (tmp_path / "c.md").write_text("unique", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_TOTAL_BYTES", 12)

    result = parsing.scan(tmp_path)
    chunks, report = parsing.parse(result)

    assert [snapshot.relative_path for snapshot in result.files] == ["a.md", "c.md"]
    assert [chunk.relative_path for chunk in chunks] == ["a.md", "c.md"]
    assert len(result.manifest) == 3
    assert result.duplicate_files == report.duplicate_files == 1
    assert report.limited_files == 0
    assert report.skipped_files == 1
    assert any("b.md" in issue and "a.md" in issue for issue in report.issues)


def test_supplied_snapshots_deduplicate_within_format_only():
    digest = hashlib.sha256(b"malformed").hexdigest()
    snapshots = tuple(
        parsing.FileSnapshot(f"bad.{suffix}", b"malformed", digest)
        for suffix in ("docx", "pdf", "pptx")
    )

    _, report = parsing.parse(parsing.Scan(snapshots, (), 0, ()))

    assert report.duplicate_files == 0
    assert sum("parse failed" in issue for issue in report.issues) == 3


def test_default_budgets_support_realistic_library_and_fourteen_mib_pdf(tmp_path):
    source = tmp_path / "large.pdf"
    with source.open("wb") as stream:
        stream.seek(14 * 1024 * 1024 - 1)
        stream.write(b"x")

    result = parsing.scan(tmp_path)

    assert [snapshot.relative_path for snapshot in result.files] == ["large.pdf"]
    assert parsing.MAX_FILE_BYTES == 32 * 1024 * 1024
    assert parsing.MAX_TOTAL_BYTES == 2 * 1024 * 1024 * 1024
    assert parsing.MAX_CHUNKS == 50000


def test_nested_temporary_folders_pruned_but_generic_outputs_and_sources_kept(tmp_path):
    for name in ("tmp", "temp", "output", "results", "data", "current", "papers", "sources"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "notes.md").write_text(name, encoding="utf-8")

    result = parsing.scan(tmp_path)

    assert {Path(snapshot.relative_path).parts[0] for snapshot in result.files} == {
        "output", "results", "data", "current", "papers", "sources",
    }
    assert result.skipped == 2
    assert sum("excluded folder" in issue for issue in result.issues) == 2
    direct = parsing.scan(tmp_path / "tmp")
    assert [snapshot.relative_path for snapshot in direct.files] == ["notes.md"]


def test_duplicates_use_shortest_canonical_path_and_are_reported(tmp_path):
    (tmp_path / "archive").mkdir()
    (tmp_path / "archive" / "backup.md").write_text("相同的内容", encoding="utf-8")
    (tmp_path / "current.md").write_text("相同的内容", encoding="utf-8")

    chunks, report = parsing.parse(parsing.scan(tmp_path))

    assert [chunk.relative_path for chunk in chunks] == ["current.md"]
    assert report.indexed_files == 1
    assert report.skipped_files == 1
    assert report.duplicate_files == 1
    assert any("archive/backup.md" in issue and "current.md" in issue for issue in report.issues)


def test_scan_globally_prioritizes_documentation_then_code_then_pdf_then_bulk_text(tmp_path):
    for name in ("0.log", "1.csv", "2.json", "3.pdf", "4.py", "5.md"):
        (tmp_path / name).write_text(name, encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "more.tex").write_text("latex", encoding="utf-8")

    result = parsing.scan(tmp_path)

    assert [snapshot.relative_path for snapshot in result.files] == [
        "5.md", "nested/more.tex", "4.py", "3.pdf", "0.log", "1.csv", "2.json",
    ]


def test_chunk_budget_skips_nonfitting_file_without_retaining_partial_file(tmp_path, monkeypatch):
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "b.md").write_text("b" * 2501, encoding="utf-8")
    (tmp_path / "c.md").write_text("c", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_CHUNKS", 3)

    chunks, report = parsing.parse(parsing.scan(tmp_path))

    assert [chunk.relative_path for chunk in chunks] == ["a.md", "c.md"]
    assert report.indexed_files == 2
    assert report.skipped_files == 1
    assert report.limited_files == 1
    assert any("b.md" in issue and "chunk count" in issue for issue in report.issues)


@pytest.mark.parametrize("change", ["content", "metadata", "delete"])
def test_changed_or_missing_file_aborts_parse_instead_of_publishing_mixed_snapshot(tmp_path, change):
    source = tmp_path / "notes.md"
    source.write_text("old", encoding="utf-8")
    result = parsing.scan(tmp_path)
    before = source.stat()
    if change == "content":
        source.write_text("new", encoding="utf-8")
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
    elif change == "metadata":
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
    else:
        source.unlink()

    with pytest.raises(ValueError, match="changed during build") as error:
        parsing.parse(result)
    assert type(error.value).__name__ == "FileChangedError"


def test_progress_reports_parsed_file_path(tmp_path):
    (tmp_path / "notes.md").write_text("notes", encoding="utf-8")
    messages = []

    parsing.parse(parsing.scan(tmp_path), progress=messages.append)

    assert any("1/1" in message and "notes.md" in message for message in messages)
