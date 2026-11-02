"""Cache path validation is per source; every chunk retains its own checks."""
import json

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex, parsing
from classbot.knowledge.storage import save_vectors


def _chunk(relative_path="a.txt", **overrides):
    return {"relative_path": relative_path, "locator": "line 1", "text": "source text", **overrides}


def _cache(tmp_path, chunks, *, known=("a.txt", "b.txt")):
    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    for name in ("a.txt", "b.txt"):
        (root / name).write_text("source text", encoding="utf-8")
    metadata = cache / "index-test.json"
    vectors, digest = save_vectors(metadata, np.tile(np.array([1, 0], dtype=np.float32), (len(chunks), 1)), 10000)
    identity, revision = {"mode": "full"}, "unchanged-revision"
    payload = {**identity, "revision": revision, "chunks": chunks,
               "vectors_file": vectors.name, "vectors_digest": digest,
               "built_at": "2026-10-16T00:00:00+00:00",
               "report": {"indexed_files": len(known), "skipped_files": 0,
                          "issues": [], "chunks": len(chunks)}}
    metadata.write_text(json.dumps(payload), encoding="utf-8")
    scan = parsing.Scan(tuple(parsing.FileSnapshot(name, b"source text", "digest") for name in known),
                        (), 0, (), mode="full")
    return metadata, identity, revision, root, scan


def _count_source_resolves(monkeypatch, root):
    original = type(root).resolve
    calls = []

    def resolve(path, *args, **kwargs):
        if path.parent == root:
            calls.append(path.name)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(root), "resolve", resolve)
    return calls


def test_many_chunks_for_one_source_resolve_only_once(tmp_path, monkeypatch):
    arguments = _cache(tmp_path, [_chunk(text=f"source {number}") for number in range(50)])
    calls = _count_source_resolves(monkeypatch, arguments[3])

    loaded = KnowledgeIndex._load_cache(*arguments)

    assert loaded is not None and len(loaded[0]) == 50
    assert calls == ["a.txt"]


def test_each_distinct_source_is_validated_once_even_when_interleaved(tmp_path, monkeypatch):
    arguments = _cache(tmp_path, [_chunk(name) for name in ("a.txt", "b.txt", "a.txt", "b.txt")])
    calls = _count_source_resolves(monkeypatch, arguments[3])

    assert KnowledgeIndex._load_cache(*arguments) is not None
    assert calls == ["a.txt", "b.txt"]


@pytest.mark.parametrize("name", ["../outside.txt", "/outside.txt", "C:/outside.txt", "folder/../a.txt", "folder\\a.txt"])
def test_illegal_paths_are_rejected_even_if_listed_in_scan(tmp_path, name):
    arguments = _cache(tmp_path, [_chunk(name)], known=(name,))

    assert KnowledgeIndex._load_cache(*arguments) is None


@pytest.mark.parametrize("name", ["unknown.txt", None, 1, [], {}])
def test_unknown_or_nonstrings_are_rejected(tmp_path, name):
    arguments = _cache(tmp_path, [_chunk(name)])

    assert KnowledgeIndex._load_cache(*arguments) is None


def test_path_resolving_outside_library_is_rejected(tmp_path, monkeypatch):
    arguments = _cache(tmp_path, [_chunk("linked.txt"), _chunk("linked.txt")], known=("linked.txt",))
    root = arguments[3]
    original = type(root).resolve

    def resolve(path, *args, **kwargs):
        if path == root / "linked.txt":
            return tmp_path / "outside.txt"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(type(root), "resolve", resolve)

    assert KnowledgeIndex._load_cache(*arguments) is None


def test_real_source_symlink_outside_library_is_rejected(tmp_path):
    arguments = _cache(tmp_path, [_chunk("linked.txt")], known=("linked.txt",))
    outside = tmp_path / "outside.txt"
    outside.write_text("outside source", encoding="utf-8")
    try:
        (arguments[3] / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable for this Windows account")

    assert KnowledgeIndex._load_cache(*arguments) is None


@pytest.mark.parametrize("override", [
    {"text": ""}, {"text": " \t\n"}, {"text": None}, {"text": 1}, {"text": "x" * 1001},
    {"locator": ""}, {"locator": None}, {"locator": 1},
])
def test_later_chunk_for_already_validated_path_keeps_all_text_and_locator_checks(tmp_path, override):
    arguments = _cache(tmp_path, [_chunk(), _chunk(**override)])

    assert KnowledgeIndex._load_cache(*arguments) is None
