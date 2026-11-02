from classbot.domain import SourceKind, SourceRef
from classbot.ui.source_rendering import (
    render_answer_html,
    render_answer_plain,
    render_sources_html,
    render_sources_plain,
)


def ref(source_id, kind, label, detail):
    return SourceRef(source_id, kind, label, detail)


def test_render_answer_html_expands_and_colors_local_and_web():
    refs = (
        ref("S1", SourceKind.TEXTBOOK, "教", "PDF第37页"),
        ref("W1", SourceKind.WEB, "网", "标题 · https://example.com"),
    )
    html = render_answer_html("答案[S1][W1]", refs)
    assert "#8be3ae" in html and "#f0b45c" in html
    assert ">教<" in html and ">网<" in html
    assert "（" in html and "、" in html


def test_render_answer_plain_uses_chinese_labels_and_merges():
    refs = (
        ref("S1", SourceKind.TEXTBOOK, "教", "PDF第37页"),
        ref("W1", SourceKind.WEB, "网", "标题"),
    )
    assert render_answer_plain("答案[S1][W1]", refs) == "答案（教、网）"
    assert render_answer_plain("答案[s1]", refs) == "答案（教）"


def test_render_answer_html_escapes_model_markup():
    refs = (ref("S1", SourceKind.TEXTBOOK, "教", "PDF第37页"),)
    html = render_answer_html("<script>alert(1)</script>[S1]", refs)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert ">教<" in html


def test_render_sources_html_includes_code_path_and_uncited_marker():
    refs = (
        ref("S1", SourceKind.CODE, "solve_model.py", "D:\\项目\\src\\solve_model.py · L10-L24"),
        ref("W1", SourceKind.WEB, "网", "标题 · https://example.com"),
    )
    html = render_sources_html(refs, {"S1"})
    assert "[代码]" in html
    assert "D:\\项目\\src\\solve_model.py" in html
    assert "（未引用）" in html
    assert render_sources_plain(refs, {"S1"}).startswith("[代码]")


def test_render_preserves_line_breaks():
    refs = (ref("S1", SourceKind.TEXTBOOK, "教", "PDF第1页"),)
    html = render_answer_html("第一行[S1]\n第二行", refs)
    assert "<br>" in html
    assert "第一行" in html and "第二行" in html
    assert render_answer_plain("第一行[S1]\n第二行", refs) == (
        "第一行（教）\n第二行"
    )
