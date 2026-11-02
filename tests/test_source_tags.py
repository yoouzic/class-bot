from classbot.domain import SourceKind, SourceRef
from classbot.llm.source_tags import audit_source_tags, cited_source_ids


def ref(source_id, kind=SourceKind.TEXTBOOK, label="教"):
    return SourceRef(source_id, kind, label, "detail")


def test_audit_removes_unknown_ids_and_keeps_known_ids():
    text, notice = audit_source_tags("答案[S1]，补充[S9]。", (ref("S1"),))
    assert text == "答案[S1]，补充。"
    assert "已移除 1 个不存在的来源编号" in notice


def test_audit_normalizes_fullwidth_tags():
    text, notice = audit_source_tags("答案【S1】。", (ref("S1"),))
    assert text == "答案[S1]。"
    assert notice == ""


def test_audit_reports_missing_tags_but_skips_headers():
    text, notice = audit_source_tags(
        "【资料事实】有依据。[S1]\n【说明】未核验。\n另一句没有来源。",
        (ref("S1"),),
    )
    assert "有 1 句未标注来源" in notice


def test_audit_skips_clarification_without_tags():
    text, notice = audit_source_tags("问题尚不完整，请等待补全。", (ref("S1"),))
    assert text == "问题尚不完整，请等待补全。"
    assert notice == ""


def test_cited_source_ids_reads_all_tag_forms():
    assert cited_source_ids("甲[S1]，乙【W2】。") == {"S1", "W2"}
