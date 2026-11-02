import numpy as np

from classbot.knowledge import KnowledgeIndex, LocalEmbedder


class CountingLocalEmbedder(LocalEmbedder):
    def __init__(self, folder):
        super().__init__(cache_dir=folder)
        self.encoded_rows = 0

    def prepare(self, *, progress=None):
        pass

    def encode(self, texts, *, query=False):
        if not query:
            self.encoded_rows += len(texts)
        return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))


def test_full_index_reuses_exact_text_vectors_without_removing_sources(tmp_path):
    root, cache = tmp_path / "source", tmp_path / "cache"
    root.mkdir()
    common = "甲乙" * 500
    (root / "one.txt").write_text(common + "\n实验记录甲", encoding="utf-8")
    (root / "two.txt").write_text(common + "\n实验记录乙", encoding="utf-8")
    first_encoder = CountingLocalEmbedder(cache / "models")
    first = KnowledgeIndex.build(root, cache, mode="full", embedder=first_encoder)
    assert first.report.chunks == 4
    assert first_encoder.encoded_rows == 3
    assert first.embedding_stats["deduplicated_rows"] == 1
    assert [chunk.text for chunk in first._chunks].count(common) == 2
    assert {chunk.relative_path for chunk in first._chunks} == {"one.txt", "two.txt"}
    (root / "two.txt").write_text(common + "\n实验记录丙", encoding="utf-8")
    second_encoder = CountingLocalEmbedder(cache / "models")
    second = KnowledgeIndex.build(root, cache, mode="full", embedder=second_encoder)
    assert second.report.chunks == 4
    assert second_encoder.encoded_rows == 1
    assert second.embedding_stats["hits"] == 3
    assert {chunk.relative_path for chunk in second._chunks} == {"one.txt", "two.txt"}
    np.testing.assert_array_equal(second._vectors, first._vectors)


def test_encoding_change_invalidates_published_index_and_text_vectors(tmp_path, monkeypatch):
    import classbot.knowledge.index as index_module

    root, cache = tmp_path / "source", tmp_path / "cache"
    root.mkdir()
    (root / "notes.txt").write_text("全部材料必须使用当前编码算法", encoding="utf-8")
    first_encoder = CountingLocalEmbedder(cache / "models")
    first = KnowledgeIndex.build(root, cache, mode="full", embedder=first_encoder)
    assert first_encoder.encoded_rows == 1

    monkeypatch.setattr(index_module, "EMBEDDING_ENCODING_VERSION", "future-window-version")
    second_encoder = CountingLocalEmbedder(cache / "models")
    second = KnowledgeIndex.build(root, cache, mode="full", embedder=second_encoder)
    assert second_encoder.encoded_rows == 1
    assert second.embedding_stats["hits"] == 0
    assert second.revision != first.revision


def test_parser_change_refreshes_sources_but_reuses_unchanged_text_vectors(tmp_path, monkeypatch):
    import classbot.knowledge.index as index_module

    root, cache = tmp_path / "source", tmp_path / "cache"
    root.mkdir()
    (root / "notes.txt").write_text("干净文本在解析器升级后内容不变", encoding="utf-8")
    first_encoder = CountingLocalEmbedder(cache / "models")
    first = KnowledgeIndex.build(root, cache, mode="full", embedder=first_encoder)
    monkeypatch.setattr(index_module.parsing, "PARSING_VERSION", "future-parser-version")
    second_encoder = CountingLocalEmbedder(cache / "models")
    second = KnowledgeIndex.build(root, cache, mode="full", embedder=second_encoder)
    assert first.revision != second.revision
    assert second_encoder.encoded_rows == 0
    assert second.embedding_stats["hits"] == 1
    assert second._chunks == first._chunks
