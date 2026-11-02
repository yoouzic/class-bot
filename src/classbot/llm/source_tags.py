from __future__ import annotations

import re

from classbot.domain import SourceRef

_TAG = re.compile(r"[\[【〔]\s*([SW]\d+|G)\s*[\]】〕]", re.IGNORECASE)
_SKIP_LINE_PREFIXES = (
    "【说明】", "【补证建议】", "【范围提示】", "【依据不足】", "【本地资料】",
)
_CLAIM_HEADERS = ("【资料事实】", "【网络补充】")
_NETWORK_FAILURE_NOTE = "网络检索也未找到可用的一般依据。"
_CLARIFICATION = re.compile(
    r"问题(?:尚)?不完整|请(?:等待)?补全|无法确定|依据不足"
)
_SENTENCE = re.compile(
    r"[^。！？!?；;\n]+"
    r"(?:[。！？!?；;]\s*"
    r"(?:[\[【〔]\s*(?:[SW]\d+|G)\s*[\]】〕]\s*)*|$)",
    re.IGNORECASE,
)
_CLAIM_SENTENCE = re.compile(
    r"[^。！？!?]+"
    r"(?:[。！？!?]\s*(?:[\[【〔]\s*(?:[SW]\d+|G)\s*[\]】〕]\s*)*|$)",
    re.IGNORECASE,
)


def cited_source_ids(text: str) -> set[str]:
    return {match.group(1).upper() for match in _TAG.finditer(text)}


def _sentence_missing(sentence: str) -> bool:
    probe = sentence.strip(" \t\r\n：:")
    if not probe or _TAG.search(probe):
        return False
    return not _CLARIFICATION.search(probe)


def audit_source_tags(
    text: str,
    refs: tuple[SourceRef, ...],
) -> tuple[str, str]:
    allowed = {ref.id for ref in refs}
    unknown = 0

    def replace(match: re.Match[str]) -> str:
        nonlocal unknown
        source_id = match.group(1).upper()
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
        for line in cleaned.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(_SKIP_LINE_PREFIXES):
                continue
            if stripped == _NETWORK_FAILURE_NOTE:
                continue
            header = next(
                (item for item in _CLAIM_HEADERS if stripped.startswith(item)),
                None,
            )
            if header is not None:
                probe = stripped[len(header):].strip(" \t\r\n：:")
                if not probe or _TAG.search(probe):
                    # Locally assembled claims carry one trailing tag run for
                    # the whole claim, so a tag anywhere covers every sentence.
                    continue
                missing += sum(
                    1
                    for sentence in _CLAIM_SENTENCE.findall(probe)
                    if _sentence_missing(sentence)
                )
                continue
            for sentence in _SENTENCE.findall(stripped):
                if _sentence_missing(sentence):
                    missing += 1

    notices = []
    if unknown:
        notices.append(f"已移除 {unknown} 个不存在的来源编号。")
    if missing:
        notices.append(f"有 {missing} 句未标注来源，请核对下方候选。")
    return cleaned, "".join(notices)
