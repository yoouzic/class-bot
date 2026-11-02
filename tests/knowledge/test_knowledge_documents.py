import io

import numpy as np

from classbot.knowledge import KnowledgeIndex


class EqualEmbedder:
    def encode(self, texts, *, query=False):
        return np.ones((len(texts), 3), dtype=np.float32)


def test_docx_paragraphs_and_table_row_locations(tmp_path):
    from docx import Document
    root = tmp_path / "docs"
    root.mkdir()
    document = Document()
    document.add_paragraph("第一段介绍")
    document.add_paragraph("第二段创新点")
    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "实验指标"
    table.cell(0, 1).text = "准确率"
    table.cell(1, 0).text = "实验结果"
    table.cell(1, 1).text = "97%"
    document.save(root / "report.docx")

    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    hits = index.search("实验结果", limit=10)

    assert hits[0].locator == "table 1, row 2"
    assert hits[0].text == "实验结果 | 97%"
    assert {h.locator for h in hits} == {"paragraph 1", "paragraph 2", "table 1, row 1", "table 1, row 2"}


def _text_pdf():
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 250 Td (Experiment accuracy: 97 percent) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    writer.add_blank_page(width=300, height=300)
    data = io.BytesIO()
    writer.write(data)
    return data.getvalue()


def test_pdf_pages_and_scanned_blank_report(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "experiment.pdf").write_bytes(_text_pdf())
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    hit = index.search("accuracy")[0]
    assert hit.locator == "page 1"
    assert "97 percent" in hit.text
    assert any("page 2" in issue and "scanned/blank" in issue for issue in index.report.issues)


def test_pptx_slide_text_and_table_locations(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches
    root = tmp_path / "docs"
    root.mkdir()
    document = Presentation()
    slide = document.slides.add_slide(document.slide_layouts[6])
    box = slide.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
    box.text = "研究目标"
    table = slide.shapes.add_table(1, 2, Inches(1), Inches(3), Inches(5), Inches(1)).table
    table.cell(0, 0).text = "实验结果"
    table.cell(0, 1).text = "97%"
    document.save(root / "slides.pptx")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    hits = index.search("实验结果", limit=10)
    assert hits[0].locator == "slide 1, table shape 2, row 1"
    assert hits[0].text == "实验结果 | 97%"
    assert any(hit.locator == "slide 1, shape 1" for hit in hits)


def test_pptx_nested_group_text_and_tables_preserve_real_group_locations(tmp_path):
    from pptx import Presentation
    from pptx.util import Inches
    root = tmp_path / "docs"
    root.mkdir()
    document = Presentation()
    slide = document.slides.add_slide(document.slide_layouts[6])
    group = slide.shapes.add_group_shape()
    box = group.shapes.add_textbox(Inches(1), Inches(1), Inches(5), Inches(1))
    box.text = "分组实验结果97%"
    table_shape = slide.shapes.add_table(1, 2, Inches(1), Inches(3), Inches(5), Inches(1))
    group.shapes.add_group_shape([table_shape])
    table = table_shape.table
    table.cell(0, 0).text = "准确率"
    table.cell(0, 1).text = "99%"
    document.save(root / "grouped.pptx")

    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    hits = index.search("实验准确率", limit=10)

    assert index.report.indexed_files == 1
    assert {hit.locator for hit in hits} == {
        "slide 1, group 1, shape 1", "slide 1, group 1, group 2, table shape 1, row 1",
    }
    assert {hit.text for hit in hits} == {"分组实验结果97%", "准确率 | 99%"}


def test_malformed_documents_report_failures_and_code_never_executes(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    marker = tmp_path / "unsafe-executed"
    (root / "script.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()", encoding="utf-8")
    for suffix in ("docx", "pdf", "pptx"):
        (root / f"bad.{suffix}").write_bytes(b"malformed")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    assert index.report.indexed_files == 1
    assert index.report.skipped_files == 3
    assert len([issue for issue in index.report.issues if "parse failed" in issue]) == 3
    assert not marker.exists()


def test_identifier_lexical_match_breaks_equal_semantic_scores(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    (root / "first.py").write_text("def login_user(): pass", encoding="utf-8")
    (root / "second.py").write_text("def smoothKalmanFilter(): pass", encoding="utf-8")
    index = KnowledgeIndex.build(root, tmp_path / "cache", embedder=EqualEmbedder())
    assert index.search("smoothKalmanFilter")[0].path.name == "second.py"
    assert index.search("Kalman filter")[0].path.name == "second.py"
