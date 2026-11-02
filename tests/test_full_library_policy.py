"""Synthetic full-library coverage and safety contracts; no real corpus or model."""
import hashlib
from pathlib import Path

import pytest

from classbot.knowledge import parsing


def _source(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _snapshot(name="notes.txt", data=b"a"):
    return parsing.FileSnapshot(name, data, hashlib.sha256(data).hexdigest())


def test_full_keeps_raw_json_csv_and_tail_after_balanced_chunk_cap(tmp_path, monkeypatch):
    _source(tmp_path, "notes.md", "a" * 2001)
    _source(tmp_path, "output/raw.json", '{"final_result": 97}')
    _source(tmp_path, "results/measurements.csv", "run,tail_accuracy\n42,99")
    _source(tmp_path, "data/historical.csv", "run,historical_result\n1,88")
    monkeypatch.setattr(parsing, "MAX_CHUNKS", 3)

    chunks, report = parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert {chunk.relative_path for chunk in chunks} == {
        "notes.md", "output/raw.json", "results/measurements.csv", "data/historical.csv",
    }
    assert any("tail_accuracy" in chunk.text for chunk in chunks)
    assert report.indexed_files == 4
    assert report.limited_files == 0


def test_full_scans_nested_tmp_temp_including_metadata_for_stale_detection(tmp_path):
    for name in ("tmp/first.json", "results/temp/last.csv", "output/current.json"):
        _source(tmp_path, name, name)
    expected = {"tmp/first.json", "results/temp/last.csv", "output/current.json"}

    result = parsing.scan(tmp_path, mode="full")
    metadata = parsing.scan(tmp_path, mode="full", include_bytes=False)

    assert {file.relative_path for file in result.files} == expected
    assert {entry[0] for entry in metadata.manifest} == expected
    assert metadata.files == ()
    assert result.mode == metadata.mode == "full"


@pytest.mark.parametrize("folder", [
    "out", "build", "dist", "target", "bin", "obj", "models", "tmp", "temp",
])
def test_full_preserves_supported_raw_results_and_config_under_ambiguous_folders(tmp_path, folder):
    prefix = f"project/{folder}"
    marker = tmp_path / "must-not-execute"
    _source(tmp_path, f"{prefix}/results/raw.csv", f"run,result\n1,{folder}_TAIL")
    _source(tmp_path, f"{prefix}/config.json", f'{{"stage": "{folder}"}}')
    _source(tmp_path, f"{prefix}/source.py", f"from pathlib import Path\nPath({str(marker)!r}).touch()")
    (tmp_path / prefix / "weights.bin").write_bytes(b"unsupported binary weights")

    result = parsing.scan(tmp_path, mode="full")
    metadata = parsing.scan(tmp_path, mode="full", include_bytes=False)
    chunks, report = parsing.parse(result, mode="full")

    expected = {f"{prefix}/results/raw.csv", f"{prefix}/config.json", f"{prefix}/source.py"}
    assert {item.relative_path for item in result.files} == expected
    assert {item.relative_path for item in chunks} == expected
    assert any(f"{folder}_TAIL" in chunk.text for chunk in chunks)
    assert {item[0] for item in result.manifest} == expected | {f"{prefix}/weights.bin"}
    assert metadata.manifest == result.manifest
    assert report.indexed_files == 3
    assert report.limited_files == 0
    assert any("weights.bin" in issue and "unsupported file type" in issue for issue in report.issues)
    assert not marker.exists()
    balanced = parsing.scan(tmp_path)
    assert balanced.files == ()
    assert balanced.manifest == ()


@pytest.mark.parametrize("folder", [
    "node_modules", "vendor", "venv", "__pycache__", "coverage", "site-packages", "bower_components",
])
def test_full_still_excludes_dependency_folders_and_names_policy_in_diagnostics(tmp_path, folder):
    _source(tmp_path, f"project/{folder}/dependency.csv", "dependency_result,excluded")
    _source(tmp_path, "results.csv", "experiment_result,retained")

    result = parsing.scan(tmp_path, mode="full")

    assert [item.relative_path for item in result.files] == ["results.csv"]
    assert [item[0] for item in result.manifest] == ["results.csv"]
    assert result.skipped == 1
    assert any(f"project/{folder}/" in issue and "全量策略：依赖目录" in issue for issue in result.issues)


def test_full_keeps_security_dependency_and_explicit_cache_exclusions(tmp_path):
    for name in (".hidden/x.txt", "credentials/x.json", "node_modules/x.js", "cache/x.txt"):
        _source(tmp_path, name, "excluded")
    _source(tmp_path, "credentials.local.json", "excluded credential")
    _source(tmp_path, "private/allowed.json", "ordinary private material")

    result = parsing.scan(tmp_path, mode="full", exclude=tmp_path / "cache")

    assert [file.relative_path for file in result.files] == ["private/allowed.json"]
    assert [entry[0] for entry in result.manifest] == ["private/allowed.json"]
    assert result.skipped == 5


def test_full_deduplicates_same_format_and_retains_all_manifest_entries(tmp_path):
    _source(tmp_path, "current.json", '{"result": 97}')
    _source(tmp_path, "tmp/backup.json", '{"result": 97}')
    _source(tmp_path, "results.csv", '{"result": 97}')

    result = parsing.scan(tmp_path, mode="full")
    chunks, report = parsing.parse(result, mode="full")

    assert {file.relative_path for file in result.files} == {"current.json", "results.csv"}
    assert len(result.manifest) == 3
    assert {chunk.relative_path for chunk in chunks} == {"current.json", "results.csv"}
    assert report.duplicate_files == 1
    assert report.limited_files == 0


def test_full_uses_larger_document_budget_without_truncating_ascii_tail(tmp_path, monkeypatch):
    _source(tmp_path, "results/raw.json", "aaaaa\nASCII_TAIL_RESULT")
    monkeypatch.setattr(parsing, "MAX_DOCUMENT_CHARS", 5)

    chunks, report = parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert any("ASCII_TAIL_RESULT" in chunk.text for chunk in chunks)
    assert report.indexed_files == 1
    assert report.limited_files == 0


def test_full_lazy_read_uses_full_file_budget_and_retains_only_metadata(tmp_path, monkeypatch):
    source = _source(tmp_path, "results.csv", "header\nFULL_TAIL")
    monkeypatch.setattr(parsing, "MAX_FILE_BYTES", 5)

    result = parsing.scan(tmp_path, mode="full")
    chunks, report = parsing.parse(result, mode="full")

    assert result.files[0].data is None
    assert result.files[0].digest == hashlib.sha256(source.read_bytes()).hexdigest()
    assert chunks[0].text == "header\nFULL_TAIL"
    assert report.indexed_files == 1


@pytest.mark.parametrize("metadata_only", [False, True])
def test_full_file_limit_raises_with_path_and_readable_limit(tmp_path, monkeypatch, metadata_only):
    _source(tmp_path, "results.csv", "larger than limit")
    monkeypatch.setattr(parsing, "FULL_MAX_FILE_BYTES", 5, raising=False)

    with pytest.raises(ValueError, match="results.csv.*单文件.*5") as error:
        parsing.scan(tmp_path, mode="full", include_bytes=not metadata_only)

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_total_byte_limit_aborts_instead_of_skipping_file(tmp_path, monkeypatch):
    _source(tmp_path, "a.json", "123456")
    _source(tmp_path, "b.csv", "abcdef")
    monkeypatch.setattr(parsing, "FULL_MAX_TOTAL_BYTES", 8, raising=False)

    with pytest.raises(ValueError, match="b.csv.*总字节.*8") as error:
        parsing.scan(tmp_path, mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


@pytest.mark.parametrize("constant,label", [
    ("FULL_MAX_FILES", "文件数量"), ("FULL_MAX_ENTRIES", "目录项数量"),
])
def test_full_file_and_entry_limits_are_explicit(tmp_path, monkeypatch, constant, label):
    _source(tmp_path, "a.json", "first")
    _source(tmp_path, "b.csv", "second")
    monkeypatch.setattr(parsing, constant, 1, raising=False)

    with pytest.raises(ValueError, match=f"b.csv.*{label}.*1") as error:
        parsing.scan(tmp_path, mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_chunk_limit_aborts_instead_of_returning_prior_files(tmp_path, monkeypatch):
    _source(tmp_path, "a.md", "a")
    _source(tmp_path, "b.csv", "b" * 2001)
    monkeypatch.setattr(parsing, "FULL_MAX_CHUNKS", 2, raising=False)

    with pytest.raises(ValueError, match="b.csv.*分块数量.*2") as error:
        parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_document_limit_aborts_instead_of_skipping_file(tmp_path, monkeypatch):
    _source(tmp_path, "raw.json", "abcdef")
    monkeypatch.setattr(parsing, "FULL_MAX_DOCUMENT_CHARS", 5, raising=False)

    with pytest.raises(ValueError, match="raw.json.*文档字符.*5") as error:
        parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_supplied_snapshot_cannot_bypass_file_size_limit(monkeypatch):
    snapshot = _snapshot(data=b"abcdef")
    monkeypatch.setattr(parsing, "FULL_MAX_FILE_BYTES", 5, raising=False)

    with pytest.raises(ValueError, match="notes.txt.*单文件.*5"):
        parsing.parse(parsing.Scan((snapshot,), (), 0, ()), mode="full")


def test_full_refuses_scan_that_already_omitted_files_due_to_limits():
    snapshot = _snapshot()
    limited = parsing.Scan((snapshot,), (), 1, ("tail.csv: file size exceeds 5 bytes",), 1)

    with pytest.raises(ValueError, match="tail.csv.*扫描.*限制") as error:
        parsing.parse(limited, mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_hashes_sources_in_bounded_reads(tmp_path, monkeypatch):
    _source(tmp_path, "raw.csv", "a" * (1024 * 1024 + 10))
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

    result = parsing.scan(tmp_path, mode="full")

    assert len(read_sizes) >= 3
    assert result.files[0].data is None


def test_full_existing_extraction_safety_limit_is_fatal_with_path(monkeypatch):
    def oversized_document(*_args, **_kwargs):
        raise parsing._ExtractionLimitError("PDF page count exceeds 2000")

    monkeypatch.setattr(parsing, "_read_document", oversized_document)

    with pytest.raises(ValueError, match="synthetic.pdf.*解析安全限制.*2000") as error:
        parsing.parse(parsing.Scan((_snapshot("synthetic.pdf"),), (), 0, ()), mode="full")

    assert type(error.value).__name__ == "FullIndexLimitError"


def test_full_unsupported_blank_invalid_files_remain_reported_diagnostics(tmp_path):
    _source(tmp_path, "good.csv", "observed,97")
    _source(tmp_path, "empty.json", "   ")
    (tmp_path / "image.bin").write_bytes(b"unsupported")
    (tmp_path / "bad.docx").write_bytes(b"invalid")
    _source(tmp_path, "script.py", "raise RuntimeError('must never execute')")

    chunks, report = parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert {chunk.relative_path for chunk in chunks} == {"good.csv", "script.py"}
    assert report.skipped_files == 3
    assert report.limited_files == 0
    assert any("unsupported file type" in issue for issue in report.issues)
    assert any("empty/no extractable" in issue for issue in report.issues)
    assert any("bad.docx" in issue and "parse failed" in issue for issue in report.issues)


def test_full_metadata_scan_does_not_open_sources(tmp_path, monkeypatch):
    _source(tmp_path, "tmp/raw.json", "raw experiment")

    def no_reads(*_args, **_kwargs):
        pytest.fail("metadata scan must not open sources")

    monkeypatch.setattr(Path, "open", no_reads)

    result = parsing.scan(tmp_path, mode="full", include_bytes=False)

    assert result.manifest[0][0] == "tmp/raw.json"


def test_full_bounds_are_explicit_and_fit_large_libraries():
    assert parsing.FULL_MAX_FILE_BYTES == 128 * 1024 * 1024
    assert parsing.FULL_MAX_TOTAL_BYTES == 16 * 1024 * 1024 * 1024
    assert parsing.FULL_MAX_FILES == 400_000
    assert parsing.FULL_MAX_ENTRIES == 2_000_000
    assert parsing.FULL_MAX_DOCUMENT_CHARS == 128_000_000
    assert parsing.FULL_MAX_CHUNKS == 2_000_000


@pytest.mark.parametrize("mode", ["unknown", "FULL", "", None, []])
def test_invalid_scan_and_parse_modes_fail_before_work(tmp_path, mode):
    with pytest.raises(ValueError, match="index mode"):
        parsing.scan(tmp_path, mode=mode)
    with pytest.raises(ValueError, match="index mode"):
        parsing.parse(parsing.Scan((), (), 0, ()), mode=mode)
