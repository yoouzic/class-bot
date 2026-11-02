from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass

_TAG = re.compile(
    r"[\[【〔]\s*(?:S\d+|W\d+|G)\s*[\]】〕]", re.IGNORECASE
)


@dataclass(frozen=True, slots=True)
class WebClaim:
    text: str
    web_ids: tuple[str, ...]


def parse_web_supplement(
    raw: str,
    allowed_ids: set[str],
) -> tuple[WebClaim, ...]:
    if not isinstance(raw, str) or len(raw) > 8000:
        return ()
    try:
        payload = json.loads(raw)
    except (ValueError, TypeError):
        return ()
    if not isinstance(payload, dict) or set(payload) != {"claims"}:
        return ()
    claims = payload["claims"]
    if not isinstance(claims, list) or not 1 <= len(claims) <= 3:
        return ()
    parsed = []
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "web_ids"}:
            return ()
        text = claim["text"]
        ids = claim["web_ids"]
        if not isinstance(text, str) or not text.strip() or len(text) > 400:
            return ()
        if _TAG.search(unicodedata.normalize("NFKC", text)):
            return ()
        if not isinstance(ids, list) or not 1 <= len(ids) <= 3:
            return ()
        if any(
            not isinstance(item, str) or item not in allowed_ids
            for item in ids
        ):
            return ()
        parsed.append(
            WebClaim(text.strip(), tuple(dict.fromkeys(ids)))
        )
    return tuple(parsed)
