"""Text decoding preserves sources and diagnoses bounded, readable damage."""
import hashlib

import numpy as np
import pytest

from classbot.knowledge import parsing


def _parse(data: bytes):
    snapshot = parsing.FileSnapshot("paper.txt", data, hashlib.sha256(data).hexdigest())
    return parsing.parse(parsing.Scan((snapshot,), (), 0, (), mode="full"), mode="full")


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "gb18030", "utf-16", "utf-16-be"])
def test_normal_supported_encodings_keep_text_and_line_locator(encoding):
    text = "正文第一行。\r\n第二行含 café 和等待值 100。\r\n末行。"
    data = text.encode(encoding)
    if encoding == "utf-16-be":
        data = b"\xfe\xff" + data

    chunks, report = _parse(data)

    assert [(chunk.locator, chunk.text) for chunk in chunks] == [
        ("lines 1–3", "正文第一行。\n第二行含 café 和等待值 100。\n末行。"),
    ]
    assert report.issues == ()


def test_sparse_controls_are_visible_without_merging_words_or_changing_lines(tmp_path):
    text = "Readable scientific source. " * 18 + "\r\nalpha\x00beta\x01gamma\x9cend\r\nlast line"
    data = text.encode("utf-8")
    source = tmp_path / "paper.txt"
    source.write_bytes(data)

    chunks, report = parsing.parse(parsing.scan(tmp_path, mode="full"), mode="full")

    assert report.indexed_files == 1
    assert report.skipped_files == 0
    assert chunks[0].locator == "lines 1–3"
    assert "alpha [U+0000] beta [U+0001] gamma [U+009C] end" in chunks[0].text
    assert chunks[0].text.count("\n") == 2
    assert len(report.issues) == 1
    assert "normalization" in report.issues[0]
    assert "controls=3" in report.issues[0]
    assert source.read_bytes() == data


def test_sparse_invalid_utf8_keeps_valid_chinese_and_marks_each_lost_byte():
    data = ("可读的中文摘要和 English scientific source。\n" * 20).encode("utf-8")
    data += b"alpha\xed\xa0\xb5\xed\xb0\xb4omega\nlast line"

    chunks, report = _parse(data)

    assert report.indexed_files == 1
    text = "\n".join(chunk.text for chunk in chunks)
    assert "可读的中文摘要" in text
    assert r"alpha\xed\xa0\xb5\xed\xb0\xb4omega" in text
    assert "last line" in text
    assert len(report.issues) == 1
    assert "normalization" in report.issues[0]
    assert "undecodable_utf8_bytes=6" in report.issues[0]
    assert "\\xed" in report.issues[0]


@pytest.mark.parametrize("data", [
    b"\x00" * 2048,
    (b"readable" + b"\x00\x01\x02\x03") * 100,
    b"valid ASCII " * 10 + b"\xff\xfe\xff" * 100,
    b"short\x00text",
    b"\x89PNG\r\n\x1a\n" + b"human readable payload " * 300 + b"\x00",
    b"PK\x03\x04" + b"human readable payload " * 300 + b"\x00",
])
def test_binary_or_insufficiently_readable_data_stays_rejected(data):
    chunks, report = _parse(data)

    assert chunks == ()
    assert report.indexed_files == 0
    assert report.skipped_files == 1
    assert any("parse failed" in issue for issue in report.issues)
    assert not any("normalization" in issue for issue in report.issues)


def test_normal_formfeeds_tabs_and_existing_escape_text_remain_unchanged():
    chunks, report = _parse(b"first\tvalue\nsecond\fthird literal \\xff")

    assert [(chunk.locator, chunk.text) for chunk in chunks] == [
        ("lines 1–3", "first\tvalue\nsecond\nthird literal \\xff"),
    ]
    assert report.issues == ()


def test_encoding_version_is_explicit_for_index_identity():
    assert parsing.PARSING_VERSION == "text-normalization-v3"


@pytest.mark.parametrize("blank", ["", " ", "\t", " \t\u2003 "])
@pytest.mark.parametrize("code", [False, True])
def test_blank_group_before_long_line_never_emits_empty_chunk(blank, code):
    text = (blank + "\n") * 3 + "x" * (parsing.MAX_CHARS + 1) + "\n \nlast line"

    chunks = list(parsing._text_chunks(text, code=code))

    assert all(value.strip() for _, value in chunks)
    assert chunks == [
        ("line 4, chars 1–1000", "x" * parsing.MAX_CHARS),
        ("line 4, chars 1001–1001", "x"),
        ("lines 5–6", "last line"),
    ]


def test_parse_collects_only_nonempty_chunks_from_any_reader(monkeypatch):
    def reader(*_args, **_kwargs):
        yield "paragraph 1", ""
        yield "paragraph 2", " \t\n\u2003 "
        yield "paragraph 3", "readable source"

    monkeypatch.setattr(parsing, "_read_document", reader)

    chunks, report = _parse(b"source")

    assert [(chunk.locator, chunk.text) for chunk in chunks] == [
        ("paragraph 3", "readable source"),
    ]
    assert report.indexed_files == report.chunks == 1
    assert report.issues == ()


def test_full_index_with_blank_group_then_long_line_reloads_without_embedding(tmp_path):
    from classbot.knowledge import KnowledgeIndex

    class Encoder:
        def encode(self, texts, *, query=False):
            return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))

    class CacheOnly:
        def encode(self, texts, *, query=False):
            pytest.fail("valid published index must reload without encoding")

    root, cache = tmp_path / "library", tmp_path / "cache"
    root.mkdir()
    (root / "paper.txt").write_text(" \n\t\n" + "x" * 1001 + "\n\nlast line", encoding="utf-8")
    first = KnowledgeIndex.build(root, cache, embedder=Encoder(), mode="full")

    second = KnowledgeIndex.build(root, cache, embedder=CacheOnly(), mode="full")

    assert first.revision == second.revision
    assert first.built_at == second.built_at
    assert first._chunks == second._chunks
    assert first.report.chunks == 3
    assert all(chunk.text.strip() for chunk in second._chunks)
    np.testing.assert_array_equal(first._vectors, second._vectors)
