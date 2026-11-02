from pathlib import Path

from classbot.domain import SourceKind
from classbot.llm.sources import (
    build_general_source_ref,
    build_local_source_ref,
    build_web_source_ref,
    classify_source,
    short_label,
)


def test_code_source_uses_filename_and_absolute_path(tmp_path):
    path = tmp_path / "src" / "solve_model.py"
    path.parent.mkdir()
    path.write_text("print(1)", encoding="utf-8")
    ref = build_local_source_ref("S1", "src/solve_model.py · L10-L24", path, "L10-L24")
    assert ref.kind is SourceKind.CODE
    assert ref.short_label == "solve_model.py"
    assert ref.detail == f"{path.resolve()} · L10-L24"
    assert ref.path == str(path.resolve())
    assert ref.locator == "L10-L24"


def test_slides_pdf_textbook_and_document_labels(tmp_path):
    slides = tmp_path / "答辩材料.pptx"
    pdf = tmp_path / "项目报告.pdf"
    textbook = tmp_path / "马克思主义教材.pdf"
    doc = tmp_path / "方案说明.docx"
    for path in (slides, pdf, textbook, doc):
        path.write_text("x", encoding="utf-8")

    assert classify_source(slides) is SourceKind.SLIDES
    assert short_label(classify_source(slides), slides) == "PPT"
    assert classify_source(pdf) is SourceKind.PDF
    assert short_label(classify_source(pdf), pdf) == "PDF"
    assert classify_source(textbook) is SourceKind.TEXTBOOK
    assert short_label(classify_source(textbook), textbook) == "教"
    assert classify_source(doc) is SourceKind.DOCUMENT
    assert short_label(classify_source(doc), doc) == "文"


def test_classroom_reference_is_textbook():
    path = Path("教材")
    assert classify_source(path, classroom_reference=True) is SourceKind.TEXTBOOK
    ref = build_local_source_ref(
        "S1", "教材", path, "PDF第37页", classroom_reference=True
    )
    assert ref.short_label == "教"
    assert ref.detail == "PDF第37页"


def test_web_and_general_refs():
    web = build_web_source_ref("W1", "剩余价值理论", "https://example.com/a")
    assert web.kind is SourceKind.WEB
    assert web.short_label == "网"
    assert web.url == "https://example.com/a"
    general = build_general_source_ref()
    assert general.kind is SourceKind.GENERAL
    assert general.short_label == "通识"
