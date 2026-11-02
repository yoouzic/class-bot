import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


def checkpoint_type():
    from classbot.knowledge.checkpoint import EmbeddingCheckpoint

    return EmbeddingCheckpoint


def test_batches_resume_without_embedding_the_committed_prefix(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision-a", 5, 1000)
    first = np.arange(6, dtype=np.float32).reshape(2, 3)
    checkpoint.append(0, first)
    path = checkpoint.path
    checkpoint.close()

    resumed = checkpoint_type()(tmp_path / "index.json", "revision-a", 5, 1000)
    assert resumed.completed == 2
    assert resumed.dimension == 3
    assert resumed.path == path
    resumed.append(2, np.ones((3, 3), dtype=np.float32))
    vectors = resumed.vectors()
    assert isinstance(vectors, np.memmap)
    assert not vectors.flags.writeable
    np.testing.assert_array_equal(vectors[:2], first)
    np.testing.assert_array_equal(vectors[2:], 1)
    resumed.close()


def test_uncommitted_suffix_is_overwritten_after_a_crash(tmp_path, monkeypatch):
    from classbot.knowledge import checkpoint as module

    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    original_commit = module._atomic_state

    def crash(*args, **kwargs):
        raise OSError("simulated state write crash")

    monkeypatch.setattr(module, "_atomic_state", crash)
    with pytest.raises(OSError, match="simulated"):
        checkpoint.append(2, np.full((2, 3), 99, dtype=np.float32))
    assert checkpoint.completed == 2
    checkpoint.close()
    monkeypatch.setattr(module, "_atomic_state", original_commit)

    resumed = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    assert resumed.completed == 2
    resumed.append(2, np.full((2, 3), 7, dtype=np.float32))
    np.testing.assert_array_equal(resumed.vectors()[2:], 7)
    resumed.close()


def test_corrupted_committed_prefix_discards_resume_state(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    vector_path = checkpoint.path
    checkpoint.close()
    vectors = np.load(vector_path, mmap_mode="r+", allow_pickle=False)
    vectors[0, 0] = 4
    vectors.flush()
    vectors._mmap.close()

    resumed = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    assert resumed.completed == 0
    assert resumed.dimension is None
    assert resumed.path is None
    assert not resumed.state_path.exists()
    resumed.append(0, np.ones((4, 2), dtype=np.float32))
    assert resumed.path != vector_path
    resumed.close()


@pytest.mark.parametrize("revision,rows", [("other-revision", 4), ("revision", 5)])
def test_changed_identity_never_reuses_previous_vectors(tmp_path, revision, rows):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    previous = checkpoint.path
    checkpoint.close()
    resumed = checkpoint_type()(tmp_path / "index.json", revision, rows, 1000)
    assert resumed.completed == 0
    assert resumed.path is None
    # Automatic invalidation must not delete a vector file a published snapshot may use.
    assert previous.exists()


@pytest.mark.parametrize(
    "start,batch,message",
    [
        (1, np.ones((1, 3)), "contiguous"),
        (0, np.ones((5, 3)), "rows"),
        (0, np.ones((1, 1000)), "size"),
        (0, np.array([[float("nan"), 1]]), "finite"),
        (0, np.ones(3), "shape"),
        (0, np.empty((0, 3)), "shape"),
    ],
)
def test_rejects_invalid_batches_without_creating_vectors(
    tmp_path, start, batch, message
):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    with pytest.raises(ValueError, match=message):
        checkpoint.append(start, batch)
    assert checkpoint.completed == 0
    assert checkpoint.path is None


def test_dimension_is_fixed_and_incomplete_vectors_cannot_be_published(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 4, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension"):
        checkpoint.append(2, np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="incomplete"):
        checkpoint.vectors()
    checkpoint.close()


def test_completed_state_can_be_removed_without_rewriting_or_deleting_vectors(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 2, 1000)
    checkpoint.append(0, np.ones((2, 3), dtype=np.float32))
    path = checkpoint.path
    before = path.stat().st_mtime_ns
    checkpoint.discard_state(keep_vectors=True)
    assert path.exists()
    assert path.stat().st_mtime_ns == before
    assert not checkpoint.state_path.exists()


def test_malicious_checkpoint_does_not_open_or_delete_external_files(tmp_path):
    metadata = tmp_path / "index.json"
    external = tmp_path / "external.npy"
    np.save(external, np.ones((2, 3), dtype=np.float32))
    checkpoint = checkpoint_type()(metadata, "revision", 2, 1000)
    checkpoint.append(0, np.ones((1, 3), dtype=np.float32))
    state_path = checkpoint.state_path
    checkpoint.close()
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    payload["vectors_file"] = "external.npy"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    resumed = checkpoint_type()(metadata, "revision", 2, 1000)
    assert resumed.completed == 0
    assert external.exists()


def test_checkpoint_metadata_stays_bounded_as_batches_accumulate(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 100, 1000)
    for start in range(100):
        checkpoint.append(start, np.ones((1, 1), dtype=np.float32))
    assert checkpoint.state_path.stat().st_size < 1024
    checkpoint.close()


def test_empty_index_has_a_publishable_readonly_vector_file(tmp_path):
    checkpoint = checkpoint_type()(tmp_path / "index.json", "revision", 0, 0)
    vectors = checkpoint.vectors()
    assert vectors.shape == (0, 0)
    assert not vectors.flags.writeable
    checkpoint.close()
    resumed = checkpoint_type()(tmp_path / "index.json", "revision", 0, 0)
    assert resumed.completed == 0
    assert resumed.dimension == 0
    resumed.close()


def test_build_lock_rejects_another_process_and_releases_after_exit(tmp_path):
    from classbot.knowledge.checkpoint import BuildLock

    metadata = tmp_path / "index.json"
    program = (
        "import sys\n"
        "from pathlib import Path\n"
        "from classbot.knowledge.checkpoint import BuildLock, BuildInProgressError\n"
        "try:\n"
        "    with BuildLock(Path(sys.argv[1])): pass\n"
        "except BuildInProgressError:\n"
        "    sys.exit(42)\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    with BuildLock(metadata):
        blocked = subprocess.run(
            [sys.executable, "-c", program, str(metadata)],
            env=environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        assert blocked.returncode == 42, blocked.stderr
    acquired = subprocess.run(
        [sys.executable, "-c", program, str(metadata)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert acquired.returncode == 0, acquired.stderr


def test_process_crash_releases_lock_and_preserves_completed_batches(tmp_path):
    from classbot.knowledge.checkpoint import BuildLock

    metadata = tmp_path / "index.json"
    program = (
        "import os, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "from classbot.knowledge.checkpoint import BuildLock, EmbeddingCheckpoint\n"
        "with BuildLock(Path(sys.argv[1])):\n"
        "    checkpoint = EmbeddingCheckpoint(Path(sys.argv[1]), 'revision', 4, 1000)\n"
        "    checkpoint.append(0, np.full((2, 3), 7, dtype=np.float32))\n"
        "    os._exit(0)\n"
    )
    environment = dict(os.environ, PYTHONPATH=str(Path(__file__).parents[2] / "src"))
    subprocess.run(
        [sys.executable, "-c", program, str(metadata)],
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    with BuildLock(metadata):
        checkpoint = checkpoint_type()(metadata, "revision", 4, 1000)
        assert checkpoint.completed == 2
        checkpoint.append(2, np.ones((2, 3), dtype=np.float32))
        np.testing.assert_array_equal(checkpoint.vectors()[:2], 7)
        checkpoint.close()
