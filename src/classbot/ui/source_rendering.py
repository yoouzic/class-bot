from __future__ import annotations

import html
import re

from classbot.domain import SourceKind, SourceRef
from classbot.llm.sources import source_type_label

_TAG_RUN = re.compile(
    r"(?:[\[【〔]\s*(?:[SW]\d+|G)\s*[\]】〕](?:\s*(?=[\[【〔]))?)+",
    re.IGNORECASE,
)
_TAG = re.compile(r"[\[【〔]\s*([SW]\d+|G)\s*[\]】〕]", re.IGNORECASE)
_LOCAL_COLOR = "#8be3ae"
_WEB_COLOR = "#f0b45c"
_GENERAL_COLOR = "#aab2bf"


def _color(kind: SourceKind) -> str:
    if kind is SourceKind.WEB:
        return _WEB_COLOR
    if kind is SourceKind.GENERAL:
        return _GENERAL_COLOR
    return _LOCAL_COLOR


def _labels(run: str, refs: dict[str, SourceRef]) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = []
    seen: set[str] = set()
    for source_id in _TAG.findall(run):
        ref = refs.get(source_id.upper())
        label = ref.short_label if ref is not None else "来源待核"
        color = _color(ref.kind) if ref is not None else _GENERAL_COLOR
        if label in seen:
            continue
        seen.add(label)
        labels.append((label, color))
    return labels


def _replace_run(
    run: str,
    refs: dict[str, SourceRef],
    *,
    rich: bool,
) -> str:
    labels = _labels(run, refs)
    if not labels:
        return ""
    if not rich:
        return "（" + "、".join(label for label, _ in labels) + "）"
    inner = "、".join(
        f'<span style="color:{color}; font-weight:600;">{html.escape(label)}</span>'
        for label, color in labels
    )
    return f"（{inner}）"


def render_answer_html(answer: str, source_refs: tuple[SourceRef, ...]) -> str:
    refs = {ref.id: ref for ref in source_refs}
    escaped = html.escape(answer).replace("\n", "<br>")
    return _TAG_RUN.sub(
        lambda match: _replace_run(match.group(0), refs, rich=True),
        escaped,
    )


def render_answer_plain(answer: str, source_refs: tuple[SourceRef, ...]) -> str:
    refs = {ref.id: ref for ref in source_refs}
    return _TAG_RUN.sub(
        lambda match: _replace_run(match.group(0), refs, rich=False),
        answer,
    )


def _ordered_refs(
    source_refs: tuple[SourceRef, ...],
    cited_ids: set[str],
) -> list[SourceRef]:
    cited = [ref for ref in source_refs if ref.id in cited_ids]
    uncited = [ref for ref in source_refs if ref.id not in cited_ids]
    return cited + uncited


def render_sources_html(
    source_refs: tuple[SourceRef, ...],
    cited_ids: set[str],
) -> str:
    lines = []
    for ref in _ordered_refs(source_refs, cited_ids):
        cited = ref.id in cited_ids
        color = _color(ref.kind) if cited else _GENERAL_COLOR
        suffix = "" if cited else "（未引用）"
        line = f"[{source_type_label(ref.kind)}] {ref.detail}{suffix}"
        lines.append(f'<div style="color:{color};">{html.escape(line)}</div>')
    return "".join(lines)


def render_sources_plain(
    source_refs: tuple[SourceRef, ...],
    cited_ids: set[str],
) -> str:
    lines = []
    for ref in _ordered_refs(source_refs, cited_ids):
        suffix = "" if ref.id in cited_ids else "（未引用）"
        lines.append(f"[{source_type_label(ref.kind)}] {ref.detail}{suffix}")
    return "\n".join(lines)
