import numpy as np
import pytest


class FakeEmbedder:
    def __init__(self):
        self.calls = []

    def encode(self, texts, *, query=False):
        self.calls.append((tuple(texts), query))
        return np.asarray([
            [1.0, 0.0, 0.1] if any(term in text for term in
                ("鲁棒", "抗干扰", "鲁棒性", "robust", "滤波"))
            else [0.0, 1.0, 0.1]
            for text in texts
        ], dtype=np.float32)


def build(root, cache, embedder=None, **kwargs):
    from classbot.knowledge import KnowledgeIndex
    return KnowledgeIndex.build(root, cache, embedder=embedder or FakeEmbedder(), **kwargs)


def test_semantic_only_retrieval_returns_original_file_and_lines(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "algorithm.py").write_text("# 抗干扰\ndef filter_sample(x):\n    return x\n", encoding="utf-8")
    (root / "README.md").write_text("界面采用蓝色，支持登录。", encoding="utf-8")
    index = build(root, tmp_path / "cache")

    hit = index.search("鲁棒性如何", limit=1)[0]

    assert hit.path == (root / "algorithm.py").resolve()
    assert hit.source == "algorithm.py · lines 1–3"
    assert hit.locator == "lines 1–3"
    assert "return x" in hit.text
    assert np.isfinite(hit.score)
    assert index.report.indexed_files == 2
    assert index.report.chunks == 2
    assert not index.is_stale()


def test_cache_reused_and_invalidated_by_content_model_and_root(tmp_path):
    root = tmp_path / "one"
    root.mkdir()
    file = root / "notes.txt"
    file.write_text("鲁棒", encoding="utf-8")
    cache = tmp_path / "cache"
    first_embedder = FakeEmbedder()
    first = build(root, cache, first_embedder)
    second_embedder = FakeEmbedder()
    second = build(root, cache, second_embedder)
    assert second.revision == first.revision
    assert second.built_at == first.built_at
    assert second_embedder.calls == []

    file.write_text("界面", encoding="utf-8")
    assert first.is_stale()
    changed = build(root, cache, FakeEmbedder())
    assert changed.revision != first.revision
    changed_model_embedder = FakeEmbedder()
    changed_model = build(root, cache, changed_model_embedder, model_name="example/other@" + "a" * 40)
    assert changed_model.revision != changed.revision
    assert changed_model_embedder.calls
    other = tmp_path / "two"
    other.mkdir()
    (other / "notes.txt").write_text("其他项目机密", encoding="utf-8")
    separate = build(other, cache)
    assert separate.root == other.resolve()
    assert "机密" not in first.search("鲁棒")[0].text
    assert "机密" in separate.search("项目")[0].text


def test_failed_refresh_preserves_previous_valid_cache(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    file = root / "notes.md"
    file.write_text("初始材料", encoding="utf-8")
    cache = tmp_path / "cache"
    original = build(root, cache)
    cache_files = {p.name: p.read_bytes() for p in cache.glob("*.json")}
    file.write_text("更新材料", encoding="utf-8")

    class Broken:
        def encode(self, texts, *, query=False):
            raise RuntimeError("local model failed")

    with pytest.raises(RuntimeError, match="local model failed"):
        build(root, cache, Broken())
    assert {p.name: p.read_bytes() for p in cache.glob("*.json")} == cache_files
    assert original.search("初始")[0].text == "初始材料"


def test_excluded_secrets_hidden_dependencies_and_unsupported_reported(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    for name in (".env", ".env.local", "credentials.json", "private_key.pem", ".secret.txt"):
        (root / name).write_text("never included", encoding="utf-8")
    deps = root / "node_modules"
    deps.mkdir()
    (deps / "code.js").write_text("never executed", encoding="utf-8")
    (root / "notes.md").write_text("安全材料", encoding="utf-8")
    (root / "image.png").write_bytes(b"not actually an image")
    (root / "bad.txt").write_bytes(b"\xff\xfe\x00")
    index = build(root, tmp_path / "cache")
    assert index.report.indexed_files == 1
    assert index.report.skipped_files >= 7
    assert any("image.png" in issue for issue in index.report.issues)
    assert any("bad.txt" in issue for issue in index.report.issues)
    assert all("never" not in hit.text for hit in index.search("材料"))


def test_chunk_ranges_bound_code_and_long_chinese_text(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "code.ts").write_text("\n".join(f"sample_{i}={i}" for i in range(130)), encoding="utf-8")
    (root / "long.txt").write_text("中" * 2505, encoding="utf-8")
    index = build(root, tmp_path / "cache")
    hits = index.search("sample 中", limit=100)
    code_hits = [h for h in hits if h.path.suffix == ".ts"]
    text_hits = [h for h in hits if h.path.suffix == ".txt"]
    assert len(code_hits) == 3
    assert {h.locator for h in code_hits} == {"lines 1–50", "lines 51–100", "lines 101–130"}
    assert all(len(h.text) <= 1000 for h in hits)
    assert len(text_hits) == 3
    assert {h.locator for h in text_hits} == {"line 1, chars 1–1000", "line 1, chars 1001–2000", "line 1, chars 2001–2505"}


def test_embedding_values_validated_and_empty_queries_do_not_load_model(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "notes.txt").write_text("材料", encoding="utf-8")

    class Invalid:
        def encode(self, texts, *, query=False):
            return np.full((len(texts), 3), np.nan)

    with pytest.raises(ValueError, match="finite"):
        build(root, tmp_path / "bad-cache", Invalid())
    embedder = FakeEmbedder()
    index = build(root, tmp_path / "cache", embedder)
    calls = len(embedder.calls)
    assert index.search("  ") == ()
    assert index.search("材料", limit=0) == ()
    assert len(embedder.calls) == calls


def test_outside_links_are_not_indexed(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "good.txt").write_text("公开材料", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("不应读取", encoding="utf-8")
    try:
        (root / "linked.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable for this Windows account")
    index = build(root, tmp_path / "cache")
    assert index.report.indexed_files == 1
    assert any("linked.txt" in issue for issue in index.report.issues)


def test_file_limit_fails_and_chunk_limit_reports_skipped_files(tmp_path, monkeypatch):
    from classbot.knowledge import parsing
    root = tmp_path / "root"
    root.mkdir()
    (root / "one.txt").write_text("一", encoding="utf-8")
    (root / "two.txt").write_text("二", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_FILES", 1)
    with pytest.raises(ValueError, match="file count"):
        build(root, tmp_path / "cache")
    monkeypatch.setattr(parsing, "MAX_FILES", 10)
    monkeypatch.setattr(parsing, "MAX_CHUNKS", 1)
    index = build(root, tmp_path / "cache")
    assert index.report.indexed_files == 1
    assert index.report.skipped_files == 1
    assert index.report.limited_files == 1
    assert any("chunk count" in issue for issue in index.report.issues)


def test_oversize_and_empty_files_reported_without_silent_truncation(tmp_path, monkeypatch):
    from classbot.knowledge import parsing
    root = tmp_path / "root"
    root.mkdir()
    (root / "large.txt").write_text("过大" * 100, encoding="utf-8")
    (root / "empty.txt").write_text("", encoding="utf-8")
    monkeypatch.setattr(parsing, "MAX_FILE_BYTES", 100)
    index = build(root, tmp_path / "cache")
    assert index.report.indexed_files == 0
    assert index.report.skipped_files == 2
    assert any("size" in issue for issue in index.report.issues)
    assert any("empty" in issue for issue in index.report.issues)
    assert index.search("任何") == ()


def test_corrupt_cache_is_rebuilt_and_traversal_payload_is_rejected(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "safe.txt").write_text("安全材料", encoding="utf-8")
    cache = tmp_path / "cache"
    build(root, cache)
    cache_file = next(cache.glob("*.json"))
    import json
    payload = json.loads(cache_file.read_text(encoding="utf-8"))
    payload["chunks"][0]["relative_path"] = "../outside.txt"
    cache_file.write_text(json.dumps(payload), encoding="utf-8")
    embedder = FakeEmbedder()
    index = build(root, cache, embedder)
    assert embedder.calls
    assert index.search("材料")[0].path == (root / "safe.txt").resolve()
