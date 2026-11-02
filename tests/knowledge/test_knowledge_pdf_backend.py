"""PDF backend contracts use small synthetic sources only."""
import hashlib
import io
import sys
from types import SimpleNamespace

from classbot.knowledge import parsing


def _pdf_snapshot(*, encrypted=False):
    from pypdf import PdfWriter
    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    if encrypted:
        writer.encrypt("test-password")
    stream = io.BytesIO()
    writer.write(stream)
    data = stream.getvalue()
    return parsing.FileSnapshot("synthetic.pdf", data, hashlib.sha256(data).hexdigest())


def test_pdf_prefers_pymupdf_sorted_page_text_and_closes_document(monkeypatch):
    calls = []
    state = {"closed": False}

    class Page:
        def __init__(self, text):
            self.text = text

        def get_text(self, mode, *, sort):
            calls.append((mode, sort))
            return self.text

    class Document:
        needs_pass = False
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            state["closed"] = True

        def __iter__(self):
            return iter((Page("Reading order text"), Page("")))

    snapshot = _pdf_snapshot()

    def open_document(*, stream, filetype):
        assert stream == snapshot.data
        assert filetype == "pdf"
        return Document()

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=open_document))
    issues = []

    extracted = list(parsing._read_document(snapshot, issues))

    assert extracted == [("page 1", "Reading order text")]
    assert calls == [("text", True), ("text", True)]
    assert state["closed"]
    assert any("page 2" in issue and "scanned/blank" in issue for issue in issues)


def test_pymupdf_page_limit_skips_document_before_extracting(monkeypatch):
    class Document:
        needs_pass = False
        page_count = 2001

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def __iter__(self):
            raise AssertionError("oversize PDF must not extract pages")

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=lambda **kwargs: Document()))

    chunks, report = parsing.parse(parsing.Scan((_pdf_snapshot(),), (), 0, ()))

    assert chunks == ()
    assert report.limited_files == 1
    assert any("PDF page count exceeds 2000" in issue for issue in report.issues)


def test_encrypted_pdf_reports_unsupported_without_password_attempt():
    chunks, report = parsing.parse(parsing.Scan((_pdf_snapshot(encrypted=True),), (), 0, ()))

    assert chunks == ()
    assert any("encrypted PDF is unsupported" in issue for issue in report.issues)


def test_pymupdf_failure_is_reported_without_switching_to_pypdf(monkeypatch):
    def cannot_open(**kwargs):
        raise ValueError("synthetic PyMuPDF failure")

    monkeypatch.setitem(sys.modules, "pymupdf", SimpleNamespace(open=cannot_open))

    chunks, report = parsing.parse(parsing.Scan((_pdf_snapshot(),), (), 0, ()))

    assert chunks == ()
    assert any("synthetic PyMuPDF failure" in issue for issue in report.issues)


def test_pypdf_fallback_when_pymupdf_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pymupdf", None)

    chunks, report = parsing.parse(parsing.Scan((_pdf_snapshot(),), (), 0, ()))

    assert chunks == ()
    assert any("page 1" in issue and "scanned/blank" in issue for issue in report.issues)
