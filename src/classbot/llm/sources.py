from __future__ import annotations

from pathlib import Path

from classbot.domain import SourceKind, SourceRef

_CODE_SUFFIXES = frozenset(
    {
        ".py", ".pyw", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx",
        ".vue", ".svelte", ".java", ".kt", ".kts", ".scala", ".groovy",
        ".c", ".h", ".cc", ".cpp", ".cxx", ".hpp", ".hh", ".cs", ".fs",
        ".fsx", ".vb", ".go", ".rs", ".rb", ".php", ".swift", ".m", ".mm",
        ".dart", ".lua", ".pl", ".pm", ".r", ".jl", ".ex", ".exs", ".erl",
        ".hrl", ".hs", ".lhs", ".clj", ".cljs", ".cljc", ".edn", ".sh",
        ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat", ".cmd", ".sql",
        ".asm", ".s", ".v", ".sv", ".vhd", ".vhdl", ".sol", ".tf", ".hcl",
        ".proto", ".graphql", ".gql", ".css", ".scss", ".sass", ".less",
        ".html", ".htm", ".xml", ".ipynb",
    }
)
_SLIDE_SUFFIXES = frozenset({".ppt", ".pptx", ".pps", ".ppsx", ".odp"})
_TEXTBOOK_MARKERS = ("教材", "课本", "教科书")

_TYPE_LABELS = {
    SourceKind.TEXTBOOK: "教材",
    SourceKind.SLIDES: "PPT",
    SourceKind.CODE: "代码",
    SourceKind.DOCUMENT: "文档",
    SourceKind.PDF: "PDF",
    SourceKind.WEB: "网络",
    SourceKind.GENERAL: "通识",
}


def classify_source(
    path: Path,
    *,
    classroom_reference: bool = False,
) -> SourceKind:
    if classroom_reference:
        return SourceKind.TEXTBOOK
    suffix = path.suffix.lower()
    if suffix in _CODE_SUFFIXES:
        return SourceKind.CODE
    if suffix in _SLIDE_SUFFIXES:
        return SourceKind.SLIDES
    if suffix == ".pdf":
        if any(marker in path.stem for marker in _TEXTBOOK_MARKERS):
            return SourceKind.TEXTBOOK
        return SourceKind.PDF
    return SourceKind.DOCUMENT


def short_label(kind: SourceKind, path: Path | None = None) -> str:
    if kind is SourceKind.TEXTBOOK:
        return "教"
    if kind is SourceKind.SLIDES:
        return "PPT"
    if kind is SourceKind.CODE:
        return path.name if path is not None else "代码"
    if kind is SourceKind.DOCUMENT:
        return "文"
    if kind is SourceKind.PDF:
        return "PDF"
    if kind is SourceKind.WEB:
        return "网"
    return "通识"


def source_type_label(kind: SourceKind) -> str:
    return _TYPE_LABELS[kind]


def build_local_source_ref(
    source_id: str,
    source: str,
    path: Path,
    locator: str,
    *,
    classroom_reference: bool = False,
) -> SourceRef:
    kind = classify_source(path, classroom_reference=classroom_reference)
    label = short_label(kind, path)
    resolved = str(path.resolve())
    if kind is SourceKind.CODE:
        detail = f"{resolved} · {locator}".rstrip(" ·")
    elif kind is SourceKind.TEXTBOOK:
        detail = locator or "教材片段"
    else:
        name = path.name or source
        detail = f"{name} · {locator}".rstrip(" ·")
    return SourceRef(
        id=source_id,
        kind=kind,
        short_label=label,
        detail=detail,
        path=resolved if kind is SourceKind.CODE else "",
        locator=locator,
    )


def build_web_source_ref(source_id: str, title: str, url: str) -> SourceRef:
    return SourceRef(
        id=source_id,
        kind=SourceKind.WEB,
        short_label="网",
        detail=f"{title} · {url}",
        url=url,
    )


def build_general_source_ref() -> SourceRef:
    return SourceRef(
        id="G",
        kind=SourceKind.GENERAL,
        short_label="通识",
        detail="模型内部知识（未检索）",
    )
