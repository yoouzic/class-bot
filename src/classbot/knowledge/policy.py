"""Shared validation and explicit failures for library indexing policies."""
from __future__ import annotations

INDEX_MODES = frozenset({"balanced", "full"})


def validate_index_mode(mode: str) -> str:
    if not isinstance(mode, str) or mode not in INDEX_MODES:
        raise ValueError("invalid index mode; expected balanced or full")
    return mode


class FullIndexLimitError(ValueError):
    """Full coverage hit a safety bound; the previous index must remain usable."""
