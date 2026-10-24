"""Shared source-label parsing; label membership never establishes factual support."""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_SOURCE_LABEL = re.compile(r"^\[(S[0-9]+)\](?=\s|$)")
_BRACKET_GROUP = re.compile(r"\[([^\[\]]+)\]")
_EXPLICIT_ID = re.compile(r"S([0-9]+)", re.IGNORECASE)
_SEPARATORS = re.compile(r"[\s,，、;；/–—-]+")


def allowed_source_ids(sources: Iterable[str]) -> set[str]:
    """Read only each generated leading label, never IDs inside a filename."""
    return {
        match.group(1) for source in sources
        if (match := _SOURCE_LABEL.match(source))
    }


def cited_source_ids(text: str) -> set[str]:
    """Read singleton/grouped labels, including common Chinese separators.

    Fullwidth brackets/letters/digits are normalized. Explicit S-prefixed IDs
    remain visible even in malformed groups; numeric shorthand such as [S1,2]
    is also recognized. Ordinary prose brackets without a leading source ID
    are ignored. This checks which IDs were cited, not whether claims follow.
    """
    identifiers: set[str] = set()
    for match in _BRACKET_GROUP.finditer(unicodedata.normalize("NFKC", text)):
        body = match.group(1).strip()
        if not re.match(r"S[0-9]", body, flags=re.IGNORECASE):
            continue
        identifiers.update(f"S{number}" for number in _EXPLICIT_ID.findall(body))
        identifiers.update(
            f"S{part}" for part in _SEPARATORS.split(body)
            if re.fullmatch(r"[0-9]+", part)
        )
    return identifiers
