"""Small Markdown heading context derived only from an owned index snapshot."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .types import KnowledgeChunk

MAX_SECTION_CONTEXT_CHARS = 300
MAX_SECTION_HEADINGS = 3
_MAX_LEAD_CHARS = 180
_HEADING = re.compile(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|[ \t]*)$")
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})(.*)$")
_FRAGMENT = re.compile(r"^line \d+, chars (\d+)[–-]\d+$")


@dataclass(frozen=True, slots=True)
class _Heading:
    level: int
    text: str
    source: str
    row: int


@dataclass(frozen=True, slots=True)
class _Lead:
    text: str
    source: str
    row: int


@dataclass(frozen=True, slots=True)
class _Context:
    headings: tuple[_Heading, ...]
    lead: _Lead | None


def _shorten(text: str, maximum: int) -> str:
    return text if len(text) <= maximum else text[:maximum - 1] + "…"


def _render(context: _Context) -> tuple[str, str]:
    # The nearest title and its opening explanation establish the subject.
    # Older ancestors use only the remaining budget. Every retained line has
    # one corresponding, complete, original chunk locator.
    if not context.headings:
        return "", ""
    lead = context.lead
    lead_text = _shorten("前文说明：" + lead.text, _MAX_LEAD_CHARS) if lead else ""
    remaining = MAX_SECTION_CONTEXT_CHARS - (len(lead_text) + 1 if lead else 0)
    nearest = context.headings[-1]
    selected = [(_shorten(nearest.text, remaining), nearest.source)]
    remaining -= len(selected[0][0]) + 1
    for heading in reversed(context.headings[-MAX_SECTION_HEADINGS:-1]):
        if remaining < 8:
            break
        text = _shorten(heading.text, remaining)
        selected.append((text, heading.source))
        remaining -= len(text) + 1
    selected.reverse()
    if lead:
        selected.append((lead_text, lead.source))
    return (
        "\n".join(text for text, _ in selected),
        "\n".join(source for _, source in selected),
    )


class MarkdownSectionContexts:
    """Lazily scan a hit's Markdown file in its original indexed chunk order.

    No current filesystem contents are consulted. Each file's finished mapping
    is published at once, so concurrent readers cannot observe a partial scan.
    A hit carries headings and a short opening explanation preceding its body,
    outside that same chunk. Text already in the hit remains ordinary evidence.
    """

    def __init__(self, chunks: tuple[KnowledgeChunk, ...]):
        self._chunks = chunks
        self._files: dict[str, list[int]] = {}
        self._contexts: dict[str, dict[int, _Context]] = {}
        for row, chunk in enumerate(chunks):
            if chunk.relative_path.lower().endswith((".md", ".markdown")):
                self._files.setdefault(chunk.relative_path, []).append(row)

    def for_row(self, row: int) -> tuple[str, str]:
        path = self._chunks[row].relative_path
        if path not in self._files:
            return "", ""
        if path not in self._contexts:
            self._contexts[path] = self._scan(self._files[path])
        return _render(self._contexts[path][row])

    def _scan(self, rows: list[int]) -> dict[int, _Context]:
        stack: list[_Heading] = []
        fence: tuple[str, int] | None = None
        lead: _Lead | None = None
        lead_open = True
        contexts = {}
        for row in rows:
            chunk = self._chunks[row]
            fragment = _FRAGMENT.fullmatch(chunk.locator)
            continuation = bool(fragment and int(fragment[1]) > 1)
            applicable = None
            for number, line in enumerate(chunk.text.splitlines()):
                # A character slice in the middle of a line cannot introduce
                # Markdown syntax, even if it happens to start with '#'/```.
                at_line_start = not (number == 0 and continuation)
                delimiter = _FENCE.match(line) if at_line_start else None
                if fence is not None:
                    if (delimiter and delimiter[1][0] == fence[0]
                            and len(delimiter[1]) >= fence[1] and not delimiter[2].strip()):
                        fence = None
                        continue
                    if applicable is None and line.strip():
                        applicable = _Context(tuple(h for h in stack if h.row != row),
                                              lead if lead and lead.row != row else None)
                    continue
                if delimiter and (delimiter[1][0] != "`" or "`" not in delimiter[2]):
                    fence = delimiter[1][0], len(delimiter[1])
                    if lead is not None:
                        lead_open = False
                    continue
                heading = _HEADING.match(line) if at_line_start else None
                if heading:
                    level = len(heading[1])
                    title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2] or "").strip()
                    stack = [h for h in stack if h.level < level]
                    if title:
                        stack.append(_Heading(level, f"{heading[1]} {title}", chunk.source, row))
                    lead, lead_open = None, True
                elif line.strip():
                    if applicable is None:
                        applicable = _Context(tuple(h for h in stack if h.row != row),
                                              lead if lead and lead.row != row else None)
                    if stack and lead is None and at_line_start:
                        lead = _Lead(_shorten(line.strip(), _MAX_LEAD_CHARS), chunk.source, row)
                    elif lead is not None and lead.row == row and lead_open:
                        lead = _Lead(_shorten(lead.text + " " + line.strip(), _MAX_LEAD_CHARS),
                                     lead.source, row)
                elif lead is not None:
                    lead_open = False
            contexts[row] = (
                applicable if applicable is not None else _Context(
                    tuple(h for h in stack if h.row != row),
                    lead if lead and lead.row != row else None,
                )
            )
        return contexts
