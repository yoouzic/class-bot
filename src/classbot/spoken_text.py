from __future__ import annotations

import re

_REDUPLICATIONS = frozenset(
    {
        "看看", "想想", "试试", "说说", "听听", "问问", "找找", "聊聊",
        "常常", "刚刚", "明明", "天天", "慢慢", "渐渐", "悄悄", "轻轻",
        "好好", "多多", "少少", "人人", "家家", "处处", "时时", "纷纷",
        "研究", "商量", "讨论", "休息", "考虑", "了解", "学习",
    }
)
_FILLER = re.compile(
    r"(^|[。！？?!；;\n])"
    r"(?:(?:嗯|呃|啊|额|唔|哦)[，,、。；;：:\s]*|"
    r"(?:就是说|然后就是)[，,、。；;：:\s]*|"
    r"那个[，,、。；;：:\s]+)+"
)
_SINGLE_CHAR_REPEAT = re.compile(r"([\u4e00-\u9fff])\1{2,}")
_PHRASE_REPEAT = re.compile(
    r"([\u4e00-\u9fff]{2,6})(?:[\s，,、。；;：:—-]*\1)+"
)
_SAFE_REPAIR = re.compile(
    r"(?:[，,、。；;：:\s—]|^)(?:我是说|我想说的是|我是想说|或者说|"
    r"准确地说|更准确地说|重新说|换个说法|而是)"
)
_AMBIGUOUS_REPAIR = re.compile(
    r"(?:[，,、。；;：:\s—]|^)(?:不是|不对)(?=[，,、。；;：:\s—])"
)


def _strip_fillers(text: str) -> str:
    return _FILLER.sub(r"\1", text)


def _collapse_phrase_repeats(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        phrase = match.group(1)
        return match.group(0) if phrase in _REDUPLICATIONS else phrase

    return _PHRASE_REPEAT.sub(replace, text)


def _collapse_repeats(text: str) -> str:
    for _ in range(3):
        collapsed = _collapse_phrase_repeats(
            _SINGLE_CHAR_REPEAT.sub(r"\1", text)
        )
        if collapsed == text:
            return collapsed
        text = collapsed
    return text


def _apply_self_repair(text: str) -> str:
    markers = [
        match
        for pattern in (_SAFE_REPAIR, _AMBIGUOUS_REPAIR)
        for match in pattern.finditer(text)
    ]
    if not markers:
        return text
    last = max(markers, key=lambda match: match.end())
    tail = text[last.end() :].lstrip("，,、。；;：: \t")
    return tail or text


def normalize_spoken_text(text: str) -> str:
    if not text:
        return ""
    normalized = _strip_fillers(text.strip())
    normalized = _collapse_repeats(normalized)
    normalized = _apply_self_repair(normalized)
    return _strip_fillers(normalized.strip())
