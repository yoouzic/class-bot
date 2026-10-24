"""A persisted checkpoint must never turn a source hardlink into writable data."""

import hashlib
import os

import numpy as np
import pytest

from classbot.knowledge.checkpoint import EmbeddingCheckpoint


def test_resuming_linked_checkpoint_allocates_new_vectors_without_changing_original(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    metadata = cache / ("index-" + "a" * 24 + ".json")
    first = EmbeddingCheckpoint(metadata, "revision", 2, 10000)
    first.append(0, np.array([[0.6, 0.8]], dtype=np.float32))
    vector = first.path
    first.close()
    original = tmp_path / "original.npy"
    try:
        os.link(vector, original)
    except OSError:
        pytest.skip("hard links not permitted by this runtime")
    before = hashlib.sha256(original.read_bytes()).hexdigest()
    resumed = EmbeddingCheckpoint(metadata, "revision", 2, 10000)
    assert resumed.completed == 0
    resumed.append(0, np.array([[0.8, 0.6], [0.8, 0.6]], dtype=np.float32))
    assert resumed.path != vector
    resumed.close()
    assert hashlib.sha256(original.read_bytes()).hexdigest() == before
