import asyncio
import json
import threading

import numpy as np
import pytest

from classbot.knowledge import KnowledgeIndex


class Embedder:
    def __init__(self, fail_after=None):
        self.rows = 0
        self.fail_after = fail_after

    def encode(self, texts, *, query=False):
        if self.fail_after is not None and self.rows >= self.fail_after:
            raise RuntimeError("simulated process interruption")
        self.rows += len(texts)
        return np.ones((len(texts), 4), dtype=np.float32)


def materials(tmp_path, count=130):
    root = tmp_path / "materials"
    root.mkdir()
    for i in range(count):
        (root / f"file-{i:03}.json").write_text(f'{{"result": {i}}}', encoding="utf-8")
    return root, tmp_path / "cache"


def test_full_build_resumes_committed_batches_and_publishes_only_complete(tmp_path):
    root, cache = materials(tmp_path)
    with pytest.raises(RuntimeError, match="interruption"):
        KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder(64))
    assert not list(cache.glob("index-*.json"))
    resumed = Embedder()
    progress = []
    index = KnowledgeIndex.build(root, cache, mode="full", embedder=resumed, progress=progress.append)
    assert resumed.rows == 66
    assert index.report.chunks == 130
    assert any("64/130" in line for line in progress)
    assert isinstance(index._vectors, np.memmap)
    assert not index._vectors.flags.writeable
    assert not list(cache.glob(".*.checkpoint.json"))
    reused = Embedder()
    again = KnowledgeIndex.build(root, cache, mode="full", embedder=reused)
    assert reused.rows == 0
    assert again.revision == index.revision
    assert isinstance(again._vectors, np.memmap)


def test_full_mode_cannot_reuse_balanced_snapshot_or_omit_tail(tmp_path, monkeypatch):
    from classbot.knowledge import parsing

    root, cache = materials(tmp_path, 4)
    (root / "tmp").mkdir()
    (root / "tmp" / "tail.csv").write_text("a,b\n1,2", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_CHUNKS", 2)
    balanced = KnowledgeIndex.build(root, cache, embedder=Embedder())
    full = KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    assert balanced.report.limited_files == 2
    assert full.report.chunks == 5
    assert full.report.limited_files == 0
    assert full.revision != balanced.revision
    assert len(list(cache.glob("index-*.json"))) == 2
    (root / "tmp" / "tail.csv").write_text("a,b\n9,8,7", encoding="utf-8")
    assert full.is_stale()


def test_cancelled_refresh_preserves_old_published_snapshot_and_resumes(tmp_path):
    root, cache = materials(tmp_path)
    KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    metadata = next(cache.glob("index-*.json"))
    previous = metadata.read_bytes()
    (root / "file-000.json").write_text('{"result": -9}', encoding="utf-8")
    stop = threading.Event()

    def progress(message):
        if message.startswith("本地语义向量 ") and "64/130" in message:
            stop.set()

    with pytest.raises(RuntimeError, match="取消"):
        KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder(), progress=progress, cancelled=stop.is_set)
    assert metadata.read_bytes() == previous
    assert (cache / json.loads(previous)["vectors_file"]).is_file()
    resumed = Embedder()
    KnowledgeIndex.build(root, cache, mode="full", embedder=resumed)
    assert resumed.rows == 66


@pytest.mark.asyncio
async def test_retrieval_gate_survives_cancelled_caller_and_skips_cancelled_waiter(tmp_path):
    root, cache = materials(tmp_path, 1)
    index = KnowledgeIndex.build(root, cache, embedder=Embedder())
    entered, release = threading.Event(), threading.Event()
    calls = []

    def search(query, limit=5):
        calls.append(query)
        entered.set()
        release.wait(3)
        return ()

    index.search = search
    active = asyncio.create_task(index.search_async("old"))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        active.cancel()
        with pytest.raises(asyncio.CancelledError):
            await active
        queued = asyncio.create_task(index.search_async("cancelled queue"))
        await asyncio.sleep(0)
        queued.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued
        latest = asyncio.create_task(index.search_async("latest"))
        await asyncio.sleep(0.02)
        assert calls == ["old"]
        release.set()
        await asyncio.wait_for(latest, 2)
        assert calls == ["old", "latest"]
    finally:
        release.set()


@pytest.mark.parametrize("during_serialization", [False, True])
def test_cancel_before_commit_keeps_previous_snapshot(tmp_path, monkeypatch, during_serialization):
    from classbot.knowledge import index as module

    root, cache = materials(tmp_path, 1)
    KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    metadata = next(cache.glob("index-*.json"))
    previous = metadata.read_bytes()
    (root / "file-000.json").write_text("changed", encoding="utf-8")
    stop = threading.Event()
    original_dump = module.json.dump

    def dump(payload, output, **kwargs):
        result = original_dump(payload, output, **kwargs)
        if during_serialization and "chunks" in payload:
            stop.set()
        return result

    monkeypatch.setattr(module.json, "dump", dump)

    def progress(message):
        if not during_serialization and message == "提交完整资料快照。":
            stop.set()

    with pytest.raises(RuntimeError, match="取消"):
        KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder(), progress=progress, cancelled=stop.is_set)
    assert metadata.read_bytes() == previous
    assert list(cache.glob(".*.checkpoint.json"))


def test_local_model_fingerprint_uses_bytes_even_when_metadata_preserved(tmp_path):
    import os

    from classbot.knowledge.index import model_identity

    folder = tmp_path / "model"
    folder.mkdir()
    config = folder / "config.json"
    config.write_bytes(b"first")
    info = config.stat()
    previous = model_identity(str(folder))
    config.write_bytes(b"other")
    os.utime(config, ns=(info.st_atime_ns, info.st_mtime_ns))
    assert model_identity(str(folder)) != previous


def test_checkpoint_never_reuses_vectors_for_changed_pdf_extraction(tmp_path, monkeypatch):
    pymupdf = pytest.importorskip("pymupdf")
    import sys

    root = tmp_path / "materials"
    root.mkdir()
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((50, 200), "BOTTOM stream-first")
    page.insert_text((50, 100), "TOP visually-first")
    document.save(root / "paper.pdf")
    document.close()
    cache = tmp_path / "cache"
    stop = threading.Event()

    class TextEmbedder(Embedder):
        def encode(self, texts, *, query=False):
            self.rows += len(texts)
            return np.array([[1, 0] if text.startswith("BOTTOM") else [0, 1] for text in texts], dtype=np.float32)

    def progress(message):
        if message == "本地语义向量 1/1；已保存进度。":
            stop.set()

    with monkeypatch.context() as fallback:
        fallback.setitem(sys.modules, "pymupdf", None)
        with pytest.raises(RuntimeError, match="取消"):
            KnowledgeIndex.build(root, cache, mode="full", embedder=TextEmbedder(), progress=progress, cancelled=stop.is_set)
    new_embedder = TextEmbedder()
    index = KnowledgeIndex.build(root, cache, mode="full", embedder=new_embedder)
    assert index._chunks[0].text.startswith("TOP")
    assert new_embedder.rows == 1
    np.testing.assert_array_equal(index._vectors[0], [0, 1])


def test_successful_refresh_prunes_only_unreferenced_generated_vectors(tmp_path):
    root, cache = materials(tmp_path, 1)
    KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    metadata = next(cache.glob("index-*.json"))
    orphan = metadata.with_name(metadata.stem + "-" + "a" * 32 + ".npy")
    np.save(orphan, np.ones((2, 4), dtype=np.float32))
    unrelated = cache / "my-research.npy"
    unrelated.write_bytes(b"user data")
    (root / "file-000.json").write_text("changed", encoding="utf-8")
    KnowledgeIndex.build(root, cache, mode="full", embedder=Embedder())
    assert not orphan.exists()
    assert unrelated.read_bytes() == b"user data"
    published = json.loads(metadata.read_text(encoding="utf-8"))["vectors_file"]
    assert (cache / published).is_file()
