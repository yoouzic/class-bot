"""Bounded Windows sharing retries preserve durable publication semantics."""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest


def _sharing_error(code=5):
    error = OSError("simulated replace failure")
    error.winerror = code
    return error


def _runtime(monkeypatch, replace, *, platform="nt"):
    from classbot.knowledge import atomic

    delays = []
    monkeypatch.setattr(atomic, "os", SimpleNamespace(name=platform, replace=replace))
    monkeypatch.setattr(atomic, "time", SimpleNamespace(sleep=delays.append))
    return atomic, delays


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_sharing_failure_then_atomic_success(tmp_path, monkeypatch, winerror):
    source, target = tmp_path / "new.tmp", tmp_path / "published.json"
    source.write_bytes(b"new snapshot")
    target.write_bytes(b"old snapshot")
    real_replace = os.replace
    attempts = []

    def replace(src, dst):
        attempts.append((src, dst))
        assert source.read_bytes() == b"new snapshot"
        assert target.read_bytes() == b"old snapshot"
        if len(attempts) < 3:
            raise _sharing_error(winerror)
        real_replace(src, dst)

    atomic, delays = _runtime(monkeypatch, replace)

    atomic.replace_atomic(source, target)

    assert len(attempts) == 3
    assert delays == [0.01, 0.02]
    assert target.read_bytes() == b"new snapshot"
    assert not source.exists()


@pytest.mark.parametrize("platform,winerror", [("posix", 5), ("posix", 32), ("nt", 2), ("nt", 112), ("nt", None)])
def test_other_platform_or_error_is_raised_immediately(monkeypatch, platform, winerror):
    error = _sharing_error(winerror)
    attempts = []

    def replace(src, dst):
        attempts.append((src, dst))
        raise error

    atomic, delays = _runtime(monkeypatch, replace, platform=platform)

    with pytest.raises(OSError) as raised:
        atomic.replace_atomic("source", "target")

    assert raised.value is error
    assert len(attempts) == 1
    assert delays == []


def test_exhaustion_is_bounded_and_preserves_both_files(tmp_path, monkeypatch):
    source, target = tmp_path / "new.tmp", tmp_path / "metadata.json"
    source.write_bytes(b"new snapshot")
    target.write_bytes(b"old snapshot")
    errors = []

    def replace(_src, _dst):
        error = _sharing_error()
        errors.append(error)
        raise error

    atomic, delays = _runtime(monkeypatch, replace)

    with pytest.raises(OSError) as raised:
        atomic.replace_atomic(source, target)

    assert raised.value is errors[-1]
    assert len(errors) == 6
    assert delays == [0.01, 0.02, 0.04, 0.08, 0.16]
    assert sum(delays) <= 0.5
    assert source.read_bytes() == b"new snapshot"
    assert target.read_bytes() == b"old snapshot"


def test_checkpoint_retries_after_closing_temporary(tmp_path, monkeypatch):
    from classbot.knowledge import checkpoint

    target = tmp_path / "checkpoint.json"
    target.write_text('{"completed":1}', encoding="utf-8")
    temporary_streams, attempts = [], []
    real_temporary = checkpoint.tempfile.NamedTemporaryFile
    real_replace = os.replace

    def temporary(*args, **kwargs):
        stream = real_temporary(*args, **kwargs)
        temporary_streams.append(stream)
        return stream

    def replace(src, dst):
        assert temporary_streams[-1].closed
        assert json.loads(src.read_text(encoding="utf-8")) == {"completed": 2}
        attempts.append((src, dst))
        if len(attempts) == 1:
            raise _sharing_error(32)
        real_replace(src, dst)

    monkeypatch.setattr(checkpoint.tempfile, "NamedTemporaryFile", temporary)
    _, delays = _runtime(monkeypatch, replace)

    checkpoint._atomic_state(target, {"completed": 2})

    assert json.loads(target.read_text(encoding="utf-8")) == {"completed": 2}
    assert delays == [0.01]
    assert len(attempts) == 2
    assert list(tmp_path.glob("*.tmp")) == []


def test_exhausted_checkpoint_preserves_committed_prefix_and_cleans_temporary(tmp_path, monkeypatch):
    from classbot.knowledge.checkpoint import EmbeddingCheckpoint

    metadata = tmp_path / "index.json"
    metadata.write_bytes(b"old published metadata")
    checkpoint = EmbeddingCheckpoint(metadata, "revision", 4, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    old_state = checkpoint.state_path.read_bytes()
    error = _sharing_error(5)

    def replace(_src, _dst):
        raise error

    with monkeypatch.context() as context:
        _, delays = _runtime(context, replace)
        with pytest.raises(OSError) as raised:
            checkpoint.append(2, np.full((2, 3), 99, dtype=np.float32))
        assert raised.value is error
        assert len(delays) == 5
    assert checkpoint.completed == 2
    assert checkpoint.state_path.read_bytes() == old_state
    assert metadata.read_bytes() == b"old published metadata"
    assert list(tmp_path.glob("*.tmp")) == []
    checkpoint.close()

    resumed = EmbeddingCheckpoint(metadata, "revision", 4, 1000)
    assert resumed.completed == 2
    resumed.append(2, np.full((2, 3), 7, dtype=np.float32))
    vectors = resumed.vectors()
    np.testing.assert_array_equal(vectors[:2], 1)
    np.testing.assert_array_equal(vectors[2:], 7)
    vectors._mmap.close()
    resumed.close()


def test_exhausted_metadata_keeps_previous_publication_and_cleans_temporary(tmp_path, monkeypatch):
    from classbot.knowledge import index

    target = tmp_path / "index.json"
    target.write_bytes(b"old published metadata")
    error = _sharing_error(33)
    before_commit = []

    def replace(_src, _dst):
        raise error

    _, delays = _runtime(monkeypatch, replace)

    with pytest.raises(OSError) as raised:
        index._atomic_json(target, {"revision": "new"}, 1000,
                           before_commit=lambda: before_commit.append(True))

    assert raised.value is error
    assert before_commit == [True]
    assert len(delays) == 5
    assert target.read_bytes() == b"old published metadata"
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("persistent", [False, True])
def test_scenario_retry_preserves_existing_configuration_on_exhaustion(tmp_path, monkeypatch, persistent):
    from classbot.scenarios import ScenarioProfile, ScenarioStore

    store = ScenarioStore(tmp_path / "scenarios.json")
    old = ScenarioProfile("old", "Existing", "defense", str(tmp_path))
    new = ScenarioProfile("new", "Updated", "defense", str(tmp_path))
    store.save([old])
    old_bytes = store.path.read_bytes()
    real_replace = os.replace
    attempts = []
    error = _sharing_error(5)

    def replace(src, dst):
        assert store.path.read_bytes() == old_bytes
        attempts.append((src, dst))
        if persistent or len(attempts) < 3:
            raise error
        real_replace(src, dst)

    _, delays = _runtime(monkeypatch, replace)
    if persistent:
        with pytest.raises(OSError) as raised:
            store.save([new])
        assert raised.value is error
        assert store.path.read_bytes() == old_bytes
        assert store.load() == (old,)
        assert len(attempts) == 6 and len(delays) == 5
    else:
        store.save([new])
        assert store.load() == (new,)
        assert len(attempts) == 3 and delays == [0.01, 0.02]
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("persistent", [False, True])
def test_vector_retry_preserves_published_vectors_and_metadata(tmp_path, monkeypatch, persistent):
    from classbot.knowledge.storage import load_vectors, save_vectors

    metadata = tmp_path / "index.json"
    original = np.ones((2, 3), dtype=np.float32)
    old_path, old_digest = save_vectors(metadata, original, 1000)
    metadata.write_text(json.dumps({"vectors_file": old_path.name}), encoding="utf-8")
    old_metadata, old_vector = metadata.read_bytes(), old_path.read_bytes()
    old_names = {path.name for path in tmp_path.iterdir()}
    replacement = np.full((2, 3), 7, dtype=np.float32)
    real_replace = os.replace
    attempts = []
    error = _sharing_error(32)

    def replace(src, dst):
        assert metadata.read_bytes() == old_metadata
        assert old_path.read_bytes() == old_vector
        attempts.append((src, dst))
        if persistent or len(attempts) < 3:
            raise error
        real_replace(src, dst)

    _, delays = _runtime(monkeypatch, replace)
    if persistent:
        with pytest.raises(OSError) as raised:
            save_vectors(metadata, replacement, 1000)
        assert raised.value is error
        assert {path.name for path in tmp_path.iterdir()} == old_names
        assert len(attempts) == 6 and len(delays) == 5
    else:
        new_path, digest = save_vectors(metadata, replacement, 1000)
        np.testing.assert_array_equal(load_vectors(metadata, new_path.name, digest, 1000), replacement)
        assert len(attempts) == 3 and delays == [0.01, 0.02]
    assert metadata.read_bytes() == old_metadata
    assert old_path.read_bytes() == old_vector
    np.testing.assert_array_equal(load_vectors(metadata, old_path.name, old_digest, 1000), original)
    assert list(tmp_path.glob("*.tmp")) == []
