import hashlib

import numpy as np
import pytest

from classbot.knowledge.storage import (
    file_digest,
    load_vectors,
    save_vectors,
    vector_path,
)


def test_owned_npy_vectors_load_as_readonly_mmap(tmp_path):
    path = tmp_path / "index-owned.npy"
    np.save(path, np.arange(6, dtype=np.float32).reshape(2, 3))
    vectors = load_vectors(tmp_path / "index.json", path.name, file_digest(path), 1000)
    assert isinstance(vectors, np.memmap)
    assert not vectors.flags.writeable
    np.testing.assert_array_equal(vectors, np.arange(6).reshape(2, 3))


def test_legacy_compressed_vectors_remain_readable(tmp_path):
    source = np.ones((2, 3), dtype=np.float32)
    metadata = tmp_path / "index.json"
    path, digest = save_vectors(metadata, source, 1000)
    assert path.suffix == ".npz"
    np.testing.assert_array_equal(
        load_vectors(metadata, path.name, digest, 1000), source
    )


@pytest.mark.parametrize(
    "shape,dtype,fortran,message",
    [
        ((10**12, 1024), np.dtype("float32"), False, "size"),
        ((2, 3), np.dtype("float64"), False, "format"),
        ((2, 3), np.dtype("O"), False, "format"),
        ((2, 3), np.dtype("float32"), True, "format"),
        ((2, 3, 1), np.dtype("float32"), False, "format"),
        ((0, 10**12), np.dtype("float32"), False, "format"),
    ],
)
def test_npy_header_is_validated_before_mapping(
    tmp_path, monkeypatch, shape, dtype, fortran, message
):
    path = tmp_path / "index-owned.npy"
    with path.open("wb") as stream:
        np.lib.format.write_array_header_1_0(
            stream,
            {
                "descr": np.lib.format.dtype_to_descr(dtype),
                "fortran_order": fortran,
                "shape": shape,
            },
        )
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: pytest.fail("unsafe np.load")
    )
    with pytest.raises(ValueError, match=message):
        load_vectors(tmp_path / "index.json", path.name, file_digest(path), 1000)


def test_npy_digest_and_exact_file_size_are_checked(tmp_path):
    path = tmp_path / "index-owned.npy"
    np.save(path, np.ones((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="digest"):
        load_vectors(
            tmp_path / "index.json",
            path.name,
            hashlib.sha256(b"wrong").hexdigest(),
            1000,
        )
    with path.open("ab") as stream:
        stream.write(b"trailing data")
    with pytest.raises(ValueError, match="size"):
        load_vectors(tmp_path / "index.json", path.name, file_digest(path), 1000)


def test_legacy_save_produces_readable_vectors_from_fortran_arrays(tmp_path):
    source = np.asfortranarray(np.ones((2, 3), dtype=np.float32))
    metadata = tmp_path / "index.json"
    path, digest = save_vectors(metadata, source, 1000)
    np.testing.assert_array_equal(
        load_vectors(metadata, path.name, digest, 1000), source
    )


def test_invalid_legacy_archive_is_reported_as_invalid_cache(tmp_path):
    path = tmp_path / "index-owned.npz"
    path.write_bytes(b"invalid archive")
    with pytest.raises(ValueError, match="format"):
        load_vectors(tmp_path / "index.json", path.name, file_digest(path), 1000)


@pytest.mark.parametrize(
    "filename",
    ["../index-owned.npy", "folder\\index-owned.npy", "index-owned:stream.npy"],
)
def test_owned_vector_paths_reject_traversal_and_windows_streams(tmp_path, filename):
    with pytest.raises(ValueError, match="path"):
        vector_path(tmp_path / "index.json", filename)
