import json

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex


class WideEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, query=False):
        self.calls.append(len(texts))
        vectors = np.ones((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 2
        return vectors


def test_vectors_are_binary_and_cache_reuse_keeps_provenance(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.md").write_text(
        "研究采用离线协作，连接恢复后同步增量修改。", encoding="utf-8"
    )
    cache = tmp_path / "cache"
    first = KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    metadata = json.loads(next(cache.glob("index-*.json")).read_text(encoding="utf-8"))
    assert "vectors" not in metadata
    assert metadata["vectors_file"].endswith(".npz")
    assert (cache / metadata["vectors_file"]).is_file()
    second_embedder = WideEmbedder()
    second = KnowledgeIndex.build(root, cache, embedder=second_embedder)
    assert second_embedder.calls == []
    assert first.revision == second.revision
    assert second.search("离线协作")[0].source == "notes.md · lines 1–1"


def test_embedding_batches_are_bounded_and_progress_is_visible(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    for i in range(130):
        (root / f"note-{i}.txt").write_text(f"独立材料{i}。", encoding="utf-8")
    embedder = WideEmbedder()
    progress = []
    index = KnowledgeIndex.build(
        root, tmp_path / "cache", embedder=embedder, progress=progress.append
    )
    assert index.report.indexed_files == 130
    assert max(embedder.calls) <= 64
    assert sum(embedder.calls) == 130
    assert any("130/130" in line for line in progress)


def test_corrupt_or_traversing_vector_cache_is_rebuilt(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")
    cache = tmp_path / "cache"
    KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    metadata_path = next(cache.glob("index-*.json"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["vectors_file"] = "../outside.npz"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    embedder = WideEmbedder()
    KnowledgeIndex.build(root, cache, embedder=embedder)
    assert embedder.calls


def test_binary_vector_limit_preserves_previous_index(tmp_path, monkeypatch):
    from classbot.knowledge import index as index_module

    root = tmp_path / "materials"
    root.mkdir()
    source = root / "notes.txt"
    source.write_text("之前资料", encoding="utf-8")
    cache = tmp_path / "cache"
    KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    metadata_path = next(cache.glob("index-*.json"))
    previous = metadata_path.read_bytes()
    source.write_text("修改后的资料", encoding="utf-8")
    monkeypatch.setattr(index_module, "MAX_VECTOR_BYTES", 100)
    with pytest.raises(ValueError, match="vector"):
        KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    assert metadata_path.read_bytes() == previous
    assert not list(cache.glob("*.tmp"))


def test_readable_report_and_partial_coverage_survive_cache_reuse(
    tmp_path, monkeypatch
):
    from classbot.knowledge import parsing

    root = tmp_path / "materials"
    root.mkdir()
    for i in range(3):
        (root / f"note-{i}.txt").write_text(f"材料{i}", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_CHUNKS", 2)
    cache = tmp_path / "cache"
    first = KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    assert first.report.limited_files == 1
    assert first.report_path.is_file()
    assert "部分资料" in first.report_text()
    assert "note-2.txt" in first.report_path.read_text(encoding="utf-8")
    second = KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    assert second.report.limited_files == 1
    assert second.report_text() == first.report_text()


def test_corrupt_binary_vectors_are_rebuilt_without_failure(tmp_path):
    root = tmp_path / "materials"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")
    cache = tmp_path / "cache"
    KnowledgeIndex.build(root, cache, embedder=WideEmbedder())
    metadata = json.loads(next(cache.glob("index-*.json")).read_text(encoding="utf-8"))
    (cache / metadata["vectors_file"]).write_bytes(b"bad archive")
    embedder = WideEmbedder()
    rebuilt = KnowledgeIndex.build(root, cache, embedder=embedder)
    assert embedder.calls
    assert rebuilt.report.indexed_files == 1
