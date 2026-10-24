"""Retrieved slices keep their preceding Markdown subject without rereading files."""

import numpy as np
import pytest

from classbot.knowledge import parsing
from classbot.knowledge.index import KnowledgeIndex
from classbot.knowledge.types import BuildReport, KnowledgeChunk


class QueryEncoder:
    def encode(self, texts, *, query=False):
        assert query, "section context must never encode document text"
        return np.tile(np.asarray([1., 0.], dtype=np.float32), (len(texts), 1))


def index_for(tmp_path, chunks, selected):
    vectors = np.tile(np.asarray([0., 1.], dtype=np.float32), (len(chunks), 1))
    vectors[selected] = [1., 0.]
    index = KnowledgeIndex(
        root=tmp_path, revision="same-snapshot", built_at="same-snapshot",
        report=BuildReport(len({c.relative_path for c in chunks}), 0, (), len(chunks)),
        chunks=tuple(chunks), vectors=vectors, embedder=QueryEncoder(), manifest=(),
        cache_dir=tmp_path / "cache",
    )
    index._lexical = type("NoLexical", (), {
        "scores": lambda self, query: np.zeros(len(chunks)),
    })()
    return index


def test_long_line_slice_retains_subject_from_an_actual_preceding_chunk(tmp_path):
    lead = "ExampleModel processes offline batches rather than live input."
    text = "# Research report\n## ExampleModel / Missing capabilities\n\n" + lead + "\n\n" + "isolated block; " * 140
    chunks = [KnowledgeChunk("report.md", locator, value)
              for locator, value in parsing._text_chunks(text, code=False)]
    assert len(chunks) >= 3
    hit = index_for(tmp_path, chunks, 2).search("applicability", limit=1)[0]

    assert hit.text == chunks[2].text
    assert hit.source == chunks[2].source
    assert hit.section_context == (
        "# Research report\n## ExampleModel / Missing capabilities\n前文说明：" + lead
    )
    assert hit.section_source.splitlines() == [chunks[0].source] * 3


def test_section_lead_identifies_subject_when_title_alone_does_not(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 86–91", "previous section text\n\n## 4. ExampleModel / Missing capabilities"),
        KnowledgeChunk("report.md", "lines 92–93", "- ExampleModel uses offline batch processing with deterministic inputs.\n- Another detail."),
        KnowledgeChunk("report.md", "lines 94–95", "independent example batches without persistent session state"),
        KnowledgeChunk("report.md", "lines 96–98", "## Different model\nfuture text must not be used"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("limitations", limit=1)[0]

    assert "offline batch processing" in hit.section_context
    assert "Different model" not in hit.section_context
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[1].source]
    assert hit.text == chunks[2].text


def test_section_context_uses_snapshot_even_if_live_file_disagrees(tmp_path, monkeypatch):
    chunks = [
        KnowledgeChunk("model.markdown", "lines 1–2", "# Snapshot subject\nold heading"),
        KnowledgeChunk("model.markdown", "line 3, chars 1–1000", "applicable conditions"),
    ]
    (tmp_path / "model.markdown").write_text("# Unverified replacement", encoding="utf-8")
    index = index_for(tmp_path, chunks, 1)
    # Retrieval already owns the validated snapshot; no file access is needed.
    monkeypatch.setattr(type(tmp_path), "open", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("must not read live source")))

    hit = index.search("conditions", limit=1)[0]

    assert hit.section_context == "# Snapshot subject\n前文说明：old heading"
    assert hit.section_source.splitlines() == [chunks[0].source] * 2
    assert index.revision == "same-snapshot"


def test_heading_hierarchy_keeps_ancestors_and_replaces_siblings(tmp_path):
    chunks = [KnowledgeChunk("report.md", f"lines {i}–{i}", text) for i, text in enumerate([
        "# Project", "## First model", "### First detail", "old details",
        "## Second model\nits own opening", "second model conditions",
    ], 1)]

    hit = index_for(tmp_path, chunks, 5).search("conditions", limit=1)[0]

    assert hit.section_context == "# Project\n## Second model\n前文说明：its own opening"
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[4].source, chunks[4].source]
    assert "First" not in hit.section_context


def test_chunk_starting_new_heading_cannot_inherit_previous_sibling(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–2", "# Project\n## Previous model"),
        KnowledgeChunk("report.md", "lines 3–4", "## Current model\nits own body"),
    ]

    hit = index_for(tmp_path, chunks, 1).search("body", limit=1)[0]

    assert hit.section_context == "# Project"
    assert hit.section_source == chunks[0].source
    assert "Previous" not in hit.section_context


@pytest.mark.parametrize("fence", ["```", "~~~~"])
def test_fenced_code_headings_are_ignored_across_chunk_boundaries(tmp_path, fence):
    chunks = [KnowledgeChunk("report.md", f"lines {i}–{i}", text) for i, text in enumerate([
        "# Actual subject", fence + "python", "# Code comment is not a subject",
        ("~~~" if fence[0] == "`" else "```"), "## Still code", fence,
        "real conditions",
    ], 1)]

    hit = index_for(tmp_path, chunks, 6).search("conditions", limit=1)[0]

    assert hit.section_context == "# Actual subject"
    assert hit.section_source == chunks[0].source


def test_section_scope_cannot_cross_files_even_when_chunks_are_interleaved(tmp_path):
    chunks = [
        KnowledgeChunk("one.md", "lines 1–1", "# First subject"),
        KnowledgeChunk("two.md", "lines 1–1", "# Second subject"),
        KnowledgeChunk("one.md", "lines 2–2", "first body"),
        KnowledgeChunk("two.md", "lines 2–2", "second body"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("body", limit=1)[0]

    assert hit.section_context == "# First subject"
    assert hit.section_source == chunks[0].source


@pytest.mark.parametrize("name", ["report.md", "report.MARKDOWN", "report.txt", "source.py"])
def test_first_chunk_never_invents_preceding_context(tmp_path, name):
    chunks = [KnowledgeChunk(name, "lines 1–3", "# Subject\n## Details\nbody")]

    hit = index_for(tmp_path, chunks, 0).search("body", limit=1)[0]

    assert (hit.section_context, hit.section_source) == ("", "")


@pytest.mark.parametrize("name", ["report.txt", "source.py", "report.md.backup"])
def test_non_markdown_never_treats_hash_lines_as_headings(tmp_path, name):
    chunks = [
        KnowledgeChunk(name, "lines 1–1", "# ordinary source text"),
        KnowledgeChunk(name, "lines 2–2", "body"),
    ]

    hit = index_for(tmp_path, chunks, 1).search("body", limit=1)[0]

    assert (hit.section_context, hit.section_source) == ("", "")


def test_nearest_three_headings_and_total_context_have_a_fixed_budget(tmp_path):
    chunks = [KnowledgeChunk("report.md", f"lines {i}–{i}", "#" * i + " " + str(i) * 90)
              for i in range(1, 7)]
    chunks.append(KnowledgeChunk("report.md", "lines 7–7", "body"))

    hit = index_for(tmp_path, chunks, 6).search("body", limit=1)[0]

    assert len(hit.section_context) <= 300
    assert len(hit.section_context.splitlines()) == 3
    assert hit.section_source.splitlines() == [chunk.source for chunk in chunks[3:6]]
    assert "6" * 90 in hit.section_context


def test_long_title_is_bounded_but_source_locator_remains_complete(tmp_path):
    chunks = [
        KnowledgeChunk("nested/report.md", "line 9, chars 1–1000", "## " + "标题" * 400),
        KnowledgeChunk("nested/report.md", "line 9, chars 1001–1603", "heading continuation"),
        KnowledgeChunk("nested/report.md", "lines 10–10", "body"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("body", limit=1)[0]

    assert len(hit.section_context) == 300
    assert hit.section_context.endswith("…")
    assert hit.section_source == chunks[0].source
    assert hit.locator == "lines 10–10"


def test_character_fragment_cannot_introduce_fake_heading_or_code_fence(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–1", "# Real heading"),
        KnowledgeChunk("report.md", "line 2, chars 1001–2000", "# Mid-line text"),
        KnowledgeChunk("report.md", "line 3, chars 1001–1003", "```"),
        KnowledgeChunk("report.md", "lines 4–4", "## Real child"),
        KnowledgeChunk("report.md", "lines 5–5", "body"),
    ]

    hit = index_for(tmp_path, chunks, 4).search("body", limit=1)[0]

    assert hit.section_context == "# Real heading\n## Real child"
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[3].source]


def test_only_atx_heading_syntax_changes_context(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–1", "# Real heading ###"),
        KnowledgeChunk("report.md", "lines 2–4", "#not a heading\ninline # symbol\n####### too deep"),
        KnowledgeChunk("report.md", "lines 5–5", "body"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("body", limit=1)[0]

    assert hit.section_context == (
        "# Real heading\n前文说明：#not a heading inline # symbol ####### too deep"
    )
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[1].source]


def test_long_lead_and_title_share_one_budget_with_complete_sources(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–1", "## " + "Long heading " * 60),
        KnowledgeChunk("report.md", "lines 2–3", "event model uses discrete arrivals " * 60),
        KnowledgeChunk("report.md", "lines 4–4", "fixed-parameter limitation"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("limitations", limit=1)[0]

    assert len(hit.section_context) == 300
    assert "event model uses discrete arrivals" in hit.section_context
    assert len(hit.section_context.splitlines()) == 2
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[1].source]


def test_new_section_resets_old_lead_and_never_borrows_future_body(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–2", "## Previous model\nprevious model explanation"),
        KnowledgeChunk("report.md", "lines 3–3", "## Current model"),
        KnowledgeChunk("report.md", "lines 4–4", "current model opening is the hit itself"),
        KnowledgeChunk("report.md", "lines 5–5", "later explanation cannot become preceding context"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("opening", limit=1)[0]

    assert hit.section_context == "## Current model"
    assert hit.section_source == chunks[1].source


def test_opening_explanation_skips_code_and_keeps_wrapped_paragraph(tmp_path):
    chunks = [
        KnowledgeChunk("report.md", "lines 1–4", "## Model\n```python\n# misleading code\n```"),
        KnowledgeChunk("report.md", "lines 5–7", "The model is event based\nand samples discrete arrivals.\n\n"),
        KnowledgeChunk("report.md", "lines 8–8", "limitations"),
    ]

    hit = index_for(tmp_path, chunks, 2).search("limitations", limit=1)[0]

    assert hit.section_context == "## Model\n前文说明：The model is event based and samples discrete arrivals."
    assert hit.section_source.splitlines() == [chunks[0].source, chunks[1].source]
