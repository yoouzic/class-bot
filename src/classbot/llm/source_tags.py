from __future__ import annotations

import re

from classbot.domain import SourceRef

_TAG = re.compile(r"[\[【〔]\s*([SW]\d+|G)\s*[\]】〕]")
_SKIP_LINE_PREFIXES = (
    "【说明】", "【补证建议】", "【范围提示】", "【依据不足】", "【本地资料】",
)
_CLAIM_HEADERS = ("【资料事实】", "【网络补充】")
_CLARIFICATION = re.compile(
    r"问题(?:尚)?不完整|请(?:等待)?补全|无法确定|依据不足"
)
_SENTENCE = re.compile(
    r"[^。！？!?；;\n]+"
    r"(?:[。！？!?；;]\s*"
    r"(?:[\[【〔]\s*(?:[SW]\d+|G)\s*[\]】〕]\s*)*|$)"
)


def cited_source_ids(text: str) -> set[str]:
    return {match.group(1) for match in _TAG.finditer(text)}


def _sentence_has_tag(sentence: str) -> bool:
    return bool(_TAG.search(sentence))


def audit_source_tags(
    text: str,
    refs: tuple[SourceRef, ...],
) -> tuple[str, str]:
    allowed = {ref.id for ref in refs}
    unknown = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal unknown
        source_id = match.group(1)
        if source_id in allowed:
            return f"[{source_id}]"
        unknown += 1
        return ""

    cleaned = _TAG.sub(replace, text)
    if not cleaned.strip():
        return cleaned, (
            f"已移除 {unknown} 个不存在的来源编号。" if unknown else ""
        )

    whole_is_clarification = not cited_source_ids(cleaned) and bool(
        _CLARIFICATION.search(cleaned)
    )
    missing = 0
    if not whole_is_clarification:
        for sentence in _SENTENCE.findall(cleaned):
            stripped = sentence.strip()
            if not stripped:
                continue
            if stripped.startswith(_SKIP_LINE_PREFIXES):
                continue
            probe = stripped
            for header in _CLAIM_HEADERS:
                probe = probe.replace(header, "")
            probe = probe.strip(" \t\r\n：:")
            if not probe or _sentence_has_tag(probe):
                continue
            if _CLARIFICATION.search(probe):
                continue
            missing += 1

    notices = []
    if unknown:
        notices.append(f"已移除 {unknown} 个不存在的来源编号。")
    if missing:
        notices.append(f"有 {missing} 句未标注来源，请核对下方候选。")
    return cleaned, "".join(notices)
