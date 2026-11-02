"""Pruning is restricted to unreferenced generated files in a synthetic cache."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def _vector(metadata, label, suffix="npy"):
    path = metadata.with_name(f"{metadata.stem}-{label * 32}.{suffix}")
    path.write_bytes(b"synthetic vector")
    return path


def _reference(path, vector):
    path.write_text(json.dumps({"vectors_file": vector.name}), encoding="utf-8")


def _prune(metadata):
    from classbot.knowledge.storage import prune_vectors

    return prune_vectors(metadata)


def test_prunes_orphans_but_keeps_published_and_checkpoint_vectors(tmp_path):
    metadata = tmp_path / "index.json"
    current = _vector(metadata, "a")
    checkpoint = _vector(metadata, "b")
    old = _vector(metadata, "c", "npz")
    _reference(metadata, current)
    _reference(tmp_path / ".index.json.checkpoint.json", checkpoint)

    assert _prune(metadata) == 1
    assert current.exists()
    assert checkpoint.exists()
    assert not old.exists()


def test_checkpoint_vector_is_kept_before_any_metadata_was_published(tmp_path):
    metadata = tmp_path / "index.json"
    pending = _vector(metadata, "a")
    orphan = _vector(metadata, "b")
    _reference(tmp_path / ".index.json.checkpoint.json", pending)

    assert _prune(metadata) == 1
    assert pending.exists()
    assert not orphan.exists()


def test_unreferenced_allocation_before_state_can_be_pruned(tmp_path):
    metadata = tmp_path / "index.json"
    orphan = _vector(metadata, "a")

    assert _prune(metadata) == 1
    assert not orphan.exists()
    assert _prune(metadata) == 0


def test_unknown_names_other_indexes_and_directories_are_untouched(tmp_path):
    metadata = tmp_path / "index.json"
    names = [
        "index-user.npy",
        "index-" + "g" * 32 + ".npy",
        "index-" + "a" * 31 + ".npy",
        "index-" + "A" * 32 + ".npy",
        "other-" + "a" * 32 + ".npy",
        "index-" + "a" * 32 + ".npy.bak",
        "index-" + "a" * 32 + ".csv",
    ]
    for name in names:
        (tmp_path / name).write_bytes(b"user cache")
    directory = tmp_path / ("index-" + "b" * 32 + ".npy")
    directory.mkdir()

    assert _prune(metadata) == 0
    assert all((tmp_path / name).exists() for name in names)
    assert directory.is_dir()


@pytest.mark.parametrize(
    "payload", ["not json", "{}", "[]", '{"vectors_file": "../outside.npy"}']
)
@pytest.mark.parametrize(
    "reference_name", ["index.json", ".index.json.checkpoint.json"]
)
def test_invalid_reference_metadata_disables_cleanup(tmp_path, payload, reference_name):
    metadata = tmp_path / "index.json"
    orphan = _vector(metadata, "a")
    (tmp_path / reference_name).write_text(payload, encoding="utf-8")

    assert _prune(metadata) == 0
    assert orphan.exists()


def test_oversized_checkpoint_disables_cleanup_without_parsing(tmp_path, monkeypatch):
    from classbot.knowledge import storage

    metadata = tmp_path / "index.json"
    orphan = _vector(metadata, "a")
    (tmp_path / ".index.json.checkpoint.json").write_bytes(b" " * (65536 + 1))
    monkeypatch.setattr(
        storage.json,
        "loads",
        lambda *_args, **_kwargs: pytest.fail("oversized checkpoint parsed"),
    )

    assert _prune(metadata) == 0
    assert orphan.exists()


def test_trusted_published_reference_prunes_without_reading_large_metadata(
    tmp_path, monkeypatch
):
    from classbot.knowledge.storage import prune_vectors

    metadata = tmp_path / "index.json"
    metadata.write_bytes(b"large metadata placeholder")
    current = _vector(metadata, "a")
    orphan = _vector(metadata, "b")
    original_open, original_lstat = Path.open, Path.lstat

    def metadata_size(path, *args, **kwargs):
        result = original_lstat(path, *args, **kwargs)
        if path == metadata:
            return SimpleNamespace(
                st_mode=result.st_mode,
                st_nlink=result.st_nlink,
                st_size=2 * 1024 * 1024 * 1024,
                st_file_attributes=0,
            )
        return result

    def no_metadata_read(path, *args, **kwargs):
        if path == metadata:
            pytest.fail("committed reference must avoid reading large metadata")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "lstat", metadata_size)
    monkeypatch.setattr(Path, "open", no_metadata_read)
    assert prune_vectors(metadata) == 0
    assert orphan.exists()
    assert prune_vectors(metadata, published_file=current.name) == 1
    assert current.exists()
    assert not orphan.exists()


def test_explicit_published_reference_still_preserves_checkpoint_and_refuses_bad_state(
    tmp_path,
):
    from classbot.knowledge.storage import prune_vectors

    metadata = tmp_path / "index.json"
    current = _vector(metadata, "a")
    pending = _vector(metadata, "b")
    orphan = _vector(metadata, "c")
    checkpoint = tmp_path / ".index.json.checkpoint.json"
    _reference(checkpoint, pending)
    assert prune_vectors(metadata, published_file=current.name) == 1
    assert current.exists() and pending.exists()
    checkpoint.write_text("malformed state", encoding="utf-8")
    _vector(metadata, "c")
    assert prune_vectors(metadata, published_file=current.name) == 0
    assert orphan.exists()


def test_unreadable_reference_metadata_disables_cleanup(tmp_path, monkeypatch):
    metadata = tmp_path / "index.json"
    orphan = _vector(metadata, "a")
    _reference(metadata, orphan)
    original_open = Path.open

    def denied(path, *args, **kwargs):
        if path == metadata:
            raise PermissionError("synthetic denied metadata")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", denied)
    assert _prune(metadata) == 0
    assert orphan.exists()


def test_unlink_failure_is_skipped_and_retried_on_the_next_prune(tmp_path, monkeypatch):
    metadata = tmp_path / "index.json"
    locked = _vector(metadata, "a")
    orphan = _vector(metadata, "b")
    original_unlink = Path.unlink

    def denied(path, *args, **kwargs):
        if path == locked:
            raise PermissionError("synthetic open Windows mapping")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", denied)
        assert _prune(metadata) == 1
    assert locked.exists()
    assert not orphan.exists()
    assert _prune(metadata) == 1


def test_hard_links_are_not_pruned(tmp_path):
    metadata = tmp_path / "index.json"
    original = tmp_path / "user-data.npy"
    original.write_bytes(b"user data")
    linked = tmp_path / ("index-" + "a" * 32 + ".npy")
    try:
        os.link(original, linked)
    except OSError:
        pytest.skip("hard links unavailable")

    assert _prune(metadata) == 0
    assert linked.exists()
    assert original.read_bytes() == b"user data"


def test_symbolic_links_are_not_pruned(tmp_path):
    metadata = tmp_path / "index.json"
    original = tmp_path / "user-data.npy"
    original.write_bytes(b"user data")
    linked = tmp_path / ("index-" + "a" * 32 + ".npy")
    try:
        linked.symlink_to(original)
    except OSError:
        pytest.skip("symbolic links unavailable")

    assert _prune(metadata) == 0
    assert linked.is_symlink()
    assert original.read_bytes() == b"user data"


@pytest.mark.skipif(os.name != "nt", reason="Windows mappings prevent unlink")
def test_actual_windows_mapping_is_retained_then_pruned_after_close(tmp_path):
    metadata = tmp_path / "index.json"
    path = tmp_path / ("index-" + "a" * 32 + ".npy")
    np.save(path, np.ones((2, 3), dtype=np.float32))
    mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    try:
        assert _prune(metadata) == 0
        assert path.exists()
    finally:
        mapped._mmap.close()
    assert _prune(metadata) == 1


def test_missing_cache_directory_is_a_noop(tmp_path):
    assert _prune(tmp_path / "does-not-exist" / "index.json") == 0
