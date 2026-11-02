"""Cache management operates only on isolated generated artifacts."""

import json
import os
import sqlite3
from pathlib import Path

import pytest

KEY = "index-" + "a" * 24
OTHER_KEY = "index-" + "b" * 24


def manager_type():
    from classbot.knowledge.cache_manager import CacheManager

    return CacheManager


def library(cache, key=KEY, *, root="materials", padding=0):
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / f"{key}.json"
    metadata.write_text(
        json.dumps(
            {
                "root": root,
                "model": {"name": "test-model"},
                "mode": "full",
                "built_at": "2026-10-16",
                "chunks": ["x" * padding],
                "report": {"indexed_files": 3, "chunks": 4},
            }
        ),
        encoding="utf-8",
    )
    vector = cache / f"{key}-{'c' * 32}.npy"
    vector.write_bytes(b"vector")
    return metadata, vector


def test_inventory_reports_each_index_and_shared_files_without_source_reads(tmp_path):
    cache, source = tmp_path / "cache", tmp_path / "source"
    source.mkdir()
    source_file = source / "original.docx"
    source_file.write_bytes(b"untouched")
    metadata, vector = library(cache, root=str(source))
    (cache / "embedding-vectors.sqlite3").write_bytes(b"12345")
    models = cache / "models"
    models.mkdir()
    (models / "model.bin").write_bytes(b"1234567")
    (cache / "unknown.bin").write_bytes(b"abc")

    result = manager_type()(cache, source_roots=(source,)).inventory()
    (row,) = result.libraries
    assert row.key == KEY
    assert row.root == str(source)
    assert row.model == "test-model"
    assert row.mode == "full"
    assert row.size_bytes == metadata.stat().st_size + vector.stat().st_size
    assert (row.file_count, row.chunk_count) == (3, 4)
    assert not row.protected
    assert result.shared_bytes == 5
    assert result.model_bytes == 7
    assert result.other_bytes == 3
    assert result.total_bytes == row.size_bytes + 15
    assert result.limit_bytes == 10 * 1024**3
    assert source_file.read_bytes() == b"untouched"


def test_cleanup_only_deletes_selected_generated_files(tmp_path):
    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    kept = library(cache, OTHER_KEY)
    checkpoint = cache / f".{KEY}.json.checkpoint.json"
    checkpoint.write_bytes(b"checkpoint")
    unknown = cache / f"{KEY}-my-original.npy"
    unknown.write_bytes(b"not generated")
    source = tmp_path / "original.txt"
    source.write_bytes(b"raw material")
    manager = manager_type()(cache, source_roots=(source.parent / "materials",))
    expected = sum(path.stat().st_size for path in (metadata, vector, checkpoint))
    assert manager.clear_library(KEY) == expected
    assert not metadata.exists() and not vector.exists() and not checkpoint.exists()
    assert all(path.exists() for path in kept)
    assert unknown.read_bytes() == b"not generated"
    assert source.read_bytes() == b"raw material"


@pytest.mark.parametrize(
    "key", ["../original", "index-" + "f" * 23, "index-" + "G" * 24]
)
def test_cleanup_rejects_untrusted_keys(tmp_path, key):
    with pytest.raises(ValueError):
        manager_type()(tmp_path / "cache").clear_library(key)


def test_active_lease_blocks_cleanup_until_final_reference_closes(tmp_path):
    from classbot.knowledge.cache_manager import CacheBusyError

    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    manager = manager_type()(cache)
    first = manager.acquire_lease(KEY)
    second = manager.acquire_lease(KEY)
    assert manager.inventory().libraries[0].protected
    with pytest.raises(CacheBusyError):
        manager.clear_library(KEY)
    first.close()
    with pytest.raises(CacheBusyError):
        manager.clear_library(KEY)
    second.close()
    assert manager.clear_library(KEY) > 0
    assert not metadata.exists() and not vector.exists()


def test_existing_build_lock_blocks_cleanup(tmp_path):
    from classbot.knowledge.cache_manager import CacheBusyError
    from classbot.knowledge.checkpoint import BuildLock

    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    with BuildLock(metadata):
        assert manager_type()(cache).inventory().libraries[0].protected
        with pytest.raises(CacheBusyError):
            manager_type()(cache).clear_library(KEY)
    assert vector.exists()


def test_explicit_protected_key_blocks_cleanup(tmp_path):
    from classbot.knowledge.cache_manager import CacheBusyError

    cache = tmp_path / "cache"
    library(cache)
    with pytest.raises(CacheBusyError):
        manager_type()(cache).clear_library(KEY, protected_keys=(KEY,))


def test_hardlink_in_generated_names_rejects_entire_cleanup(tmp_path):
    from classbot.knowledge.cache_manager import CacheSafetyError

    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    original = tmp_path / "original.npy"
    original.write_bytes(b"source")
    vector.unlink()
    try:
        os.link(original, vector)
    except OSError:
        pytest.skip("hard links not permitted by this runtime")
    manager = manager_type()(cache)
    assert manager.inventory().libraries[0].status == "unsafe"
    with pytest.raises(CacheSafetyError):
        manager.clear_library(KEY)
    assert metadata.exists()
    assert original.read_bytes() == b"source"


def test_hardlink_stat_guard_prevents_any_unlink_even_without_link_privilege(
    tmp_path, monkeypatch
):
    from types import SimpleNamespace

    from classbot.knowledge.cache_manager import CacheSafetyError

    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    original = Path.lstat

    def synthetic_link(path, *args, **kwargs):
        value = original(path, *args, **kwargs)
        if path == vector:
            return SimpleNamespace(
                st_mode=value.st_mode,
                st_nlink=2,
                st_size=value.st_size,
                st_file_attributes=0,
            )
        return value

    monkeypatch.setattr(Path, "lstat", synthetic_link)
    with pytest.raises(CacheSafetyError):
        manager_type()(cache).clear_library(KEY)
    assert metadata.exists() and vector.exists()


def test_symlink_root_and_source_overlap_are_never_cleaned(tmp_path):
    from classbot.knowledge.cache_manager import CacheSafetyError

    cache = tmp_path / "cache"
    metadata, _ = library(cache)
    with pytest.raises(CacheSafetyError):
        manager_type()(cache, source_roots=(cache,)).clear_library(KEY)
    assert metadata.exists()
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(cache, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink not permitted")
    with pytest.raises(CacheSafetyError):
        manager_type()(linked).clear_library(KEY)
    assert metadata.exists()


def test_policy_persists_limit_but_never_automatically_removes_libraries(tmp_path):
    from classbot.knowledge.cache_manager import CacheLimitError

    cache = tmp_path / "cache"
    metadata, _ = library(cache)
    manager = manager_type()(cache)
    with pytest.raises(CacheLimitError):
        manager.set_limit(1)
    manager.set_limit(1024 * 1024)
    assert manager_type()(cache).inventory().limit_bytes == 1024 * 1024
    with pytest.raises(CacheLimitError):
        manager.require_space(1024 * 1024)
    assert metadata.exists()


def test_generated_cache_nested_in_project_library_only_cleans_owned_artifacts(tmp_path):
    source = tmp_path / "project"
    source.mkdir()
    original = source / "results.csv"
    original.write_bytes(b"original experimental data")
    cache = source / ".cache" / "classbot"
    metadata, vector = library(cache, root=str(source))
    unknown = cache / "original-user-file.txt"
    unknown.write_bytes(b"must remain")
    manager = manager_type()(cache, source_roots=(source,))
    manager.set_limit(1024 * 1024)
    assert manager.clear_library(KEY) > 0
    assert not metadata.exists() and not vector.exists()
    assert original.read_bytes() == b"original experimental data"
    assert unknown.read_bytes() == b"must remain"


def test_quota_context_checks_real_shared_model_and_temporary_bytes(tmp_path):
    from classbot.knowledge.cache_manager import (
        cache_available_bytes,
        require_cache_space,
    )

    cache = tmp_path / "cache"
    library(cache)
    manager = manager_type()(cache)
    manager.set_limit(10000)
    before = manager.inventory().total_bytes
    assert cache_available_bytes(cache / "vectors.npy") is None
    with manager.mutation(), manager.quota():
        assert cache_available_bytes(cache / "vectors.npy") == 10000 - before
        require_cache_space(cache / "vectors.npy", 100)
        (cache / "pending.tmp").write_bytes(b"x" * 100)
        assert cache_available_bytes(cache / "vectors.npy") == 9900 - before
        assert cache_available_bytes(tmp_path / "outside") is None
    assert cache_available_bytes(cache / "vectors.npy") is None


def test_registry_avoids_reading_large_metadata_and_stale_registry_falls_back(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    metadata, _ = library(cache, padding=100000)
    manager = manager_type()(cache)
    manager.record_library(
        KEY,
        root="tracked",
        model_name="model",
        mode="full",
        built_at="time",
        file_count=8,
        chunk_count=12,
    )
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path != metadata, "fresh registry must avoid metadata read"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    (row,) = manager.inventory().libraries
    assert (row.root, row.file_count, row.chunk_count) == ("tracked", 8, 12)
    monkeypatch.setattr(Path, "open", original)
    metadata.write_text('{"root":"new","mode":"full"}', encoding="utf-8")
    assert manager.inventory().libraries[0].root == "new"


def test_clear_shared_vectors_preserves_indexes_models_and_unknown_files(tmp_path):
    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    database = cache / "embedding-vectors.sqlite3"
    connection = sqlite3.connect(database)
    with connection:
        connection.execute("CREATE TABLE example (value TEXT)")
    connection.close()
    expected = database.stat().st_size
    assert manager_type()(cache).clear_shared_vectors() == expected
    assert not database.exists()
    assert metadata.exists() and vector.exists()


def test_open_sqlite_transaction_prevents_shared_cleanup(tmp_path):
    from classbot.knowledge.cache_manager import CacheBusyError

    cache = tmp_path / "cache"
    cache.mkdir()
    database = cache / "embedding-vectors.sqlite3"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE example (value TEXT)")
    connection.execute("BEGIN EXCLUSIVE")
    try:
        with pytest.raises(CacheBusyError):
            manager_type()(cache).clear_shared_vectors()
        assert database.exists()
    finally:
        connection.close()


def test_another_manager_in_same_thread_has_reentrant_mutation(tmp_path):
    cache = tmp_path / "cache"
    first, second = manager_type()(cache), manager_type()(cache)
    with first.mutation(), second.mutation():
        assert first.inventory().total_bytes >= 1


def test_another_thread_cannot_mutate_during_build(tmp_path):
    import concurrent.futures

    from classbot.knowledge.cache_manager import CacheBusyError

    manager = manager_type()(tmp_path / "cache")
    with concurrent.futures.ThreadPoolExecutor() as pool, manager.mutation():
        result = pool.submit(manager.set_limit, 1000000)
        with pytest.raises(CacheBusyError):
            result.result(timeout=2)


def test_reader_lease_is_shared_and_blocks_cleanup_in_another_process(tmp_path):
    import subprocess
    import sys

    cache = tmp_path / "cache"
    metadata, _ = library(cache)
    manager = manager_type()(cache)
    script = """
import sys
from pathlib import Path
from classbot.knowledge.cache_manager import CacheManager, CacheBusyError
manager = CacheManager(Path(sys.argv[1]))
lease = manager.acquire_lease(sys.argv[2])
assert manager.inventory().libraries[0].protected
lease.close()
try:
    manager.clear_library(sys.argv[2])
except CacheBusyError:
    print('protected')
else:
    raise AssertionError('active reader was deleted')
"""
    with manager.acquire_lease(KEY):
        result = subprocess.run(
            [sys.executable, "-c", script, str(cache), KEY],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "protected"
    assert metadata.exists()


def test_total_policy_lock_blocks_another_process(tmp_path):
    import subprocess
    import sys

    cache = tmp_path / "cache"
    script = """
import sys
from pathlib import Path
from classbot.knowledge.cache_manager import CacheManager, CacheBusyError
try:
    CacheManager(Path(sys.argv[1])).set_limit(999999)
except CacheBusyError:
    print('busy')
else:
    raise AssertionError('concurrent mutation was allowed')
"""
    with manager_type()(cache).mutation():
        result = subprocess.run(
            [sys.executable, "-c", script, str(cache)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            env={**os.environ, "PYTHONPATH": str(Path("src").resolve())},
        )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "busy"


def test_registry_and_policy_do_not_look_like_index_metadata(tmp_path):
    cache = tmp_path / "cache"
    metadata, _ = library(cache)
    manager = manager_type()(cache)
    manager.set_limit(1000000)
    manager.record_library(KEY, root="materials")
    assert list(cache.glob("*.json")) == [metadata]
    assert list(cache.glob("index-*.json")) == [metadata]


def test_corrupt_policy_fails_closed_without_deleting_anything(tmp_path):
    from classbot.knowledge.cache_manager import CacheSafetyError

    cache = tmp_path / "cache"
    metadata, vector = library(cache)
    manager = manager_type()(cache)
    manager.policy_path.write_text('{"version":1,"limit_bytes":true}', encoding="utf-8")
    with pytest.raises(CacheSafetyError):
        manager.require_space(1)
    assert metadata.exists() and vector.exists()


def test_large_legacy_metadata_uses_prefix_and_matching_report_counts(
    tmp_path, monkeypatch
):
    cache = tmp_path / "cache"
    metadata, _ = library(cache, root="large", padding=100000)
    (cache / f"{KEY}.report.txt").write_text(
        "Classbot 本地资料索引报告\n快照：2026-10-16\n已索引文件：9000；片段：280000\n",
        encoding="utf-8",
    )
    original = Path.read_text

    def no_whole_read(path, *args, **kwargs):
        assert path != metadata
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_whole_read)
    (row,) = manager_type()(cache).inventory().libraries
    assert (row.root, row.file_count, row.chunk_count) == ("large", 9000, 280000)


def test_atomic_policy_failure_keeps_previous_limit_and_removes_temporary(
    tmp_path, monkeypatch
):
    from classbot.knowledge import cache_manager

    manager = manager_type()(tmp_path / "cache")
    manager.set_limit(10000)
    original = manager.policy_path.read_bytes()

    def failed_commit(*_args):
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(cache_manager, "replace_atomic", failed_commit)
    with pytest.raises(OSError, match="simulated"):
        manager.set_limit(20000)
    assert manager.policy_path.read_bytes() == original
    assert not list(manager.cache_dir.glob("*.tmp"))


def test_abandoned_lease_finalizer_unblocks_cleanup(tmp_path):
    import gc

    cache = tmp_path / "cache"
    library(cache)
    manager = manager_type()(cache)
    lease = manager.acquire_lease(KEY)
    assert manager.inventory().libraries[0].protected
    del lease
    gc.collect()
    assert manager.clear_library(KEY) > 0


def test_cleanup_includes_only_strictly_owned_crash_temporary_files(tmp_path):
    cache = tmp_path / "cache"
    library(cache)
    generated = [
        f".{KEY}.json.abcdefgh.tmp",
        f".{KEY}.cache-info.abcdefgh.tmp",
        f".{KEY}.vectors.abcdefgh.tmp",
        f".{KEY}.report.abcdefgh.tmp",
        f".{KEY}.json.checkpoint.json.abcdefgh.tmp",
        f"..{KEY}.json.checkpoint.json.abcdefgh.tmp",
    ]
    unknown = [f".{KEY}.vectors.x.tmp", "tmpabcdefgh.tmp", f".{KEY}.other.abcdefgh.tmp"]
    for name in generated + unknown:
        (cache / name).write_bytes(b"crash temporary")
    manager_type()(cache).clear_library(KEY)
    assert not any((cache / name).exists() for name in generated)
    assert all((cache / name).exists() for name in unknown)
