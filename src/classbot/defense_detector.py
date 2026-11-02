"""Question extraction for project presentations, without classroom topic gates."""

from __future__ import annotations

import re
from collections import OrderedDict, deque
from uuid import uuid4

from classbot.domain import QuestionCandidate, TranscriptEvent
from classbot.spoken_text import normalize_spoken_text

_QUESTION = re.compile(
    r"为什么|为何|如何|怎么|什么|哪些|哪种|哪个|哪里|多少|几次|多大|多久|是否|"
    r"有没有|能否|能不能|可不可以|要不要|需不需要|凭什么|有何|何处|[吗呢?？]$"
)
_REQUEST = re.compile(
    r"(?:^|[，,：:]|请|能否|能不能|麻烦|可以)(?:你们|你)?(?:再|具体|详细|简单)?"
    r"(?:介绍|解释|说明|谈谈|讲讲|对比|分析|证明|展示|阐述|描述|比较)"
)
_NARRATION = re.compile(
    r"^(?:接下来|下面|现在)?(?:我们|我)(?:来|将|会|先|再|已经|接着|主要|介绍|解释|说明|讨论|分析)"
)
_MANAGEMENT = re.compile(
    r"^(?:(?:大家|老师|你们|你|现在|请问)[，,\s]*)?"
    r"(?:能听到吗|听得到吗|听得见吗|听清楚了吗|声音(?:正常吗|有吗|可以吗)|"
    r"网络(?:正常吗|卡了吗)|(?:请|麻烦)?(?:打开|关闭|调整)(?:一下)?(?:摄像头|麦克风|声音)|"
    r"(?:屏幕共享|共享屏幕|摄像头|麦克风)(?:正常吗|打开了吗|关闭了吗|有问题吗))"
    r"[？?。]*$"
)
_INCOMPLETE = re.compile(
    r"^(?:为什么|为何|如何|怎么|是否|能否)[？?。\s]*$|"
    r"(?:因为|以及|为什么要|如何去)[？?。\s]*$"
)
_SMALL_TALK = re.compile(
    r"^(?:你好|大家好|早上好|晚上好|谢谢|多谢|辛苦了|再见|拜拜|吃了吗|最近怎么样)"
)


class DefenseQuestionDetector:
    def __init__(self, subject: str) -> None:
        self.subject = subject
        self._segment = 0
        self._active: OrderedDict[int | str, tuple[str, str, bool]] = OrderedDict()
        self._context: deque[str] = deque(maxlen=4)

    def begin_segment(self) -> None:
        self._segment += 1

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        raw = normalize_spoken_text(event.text.strip())
        segment = event.segment_id if event.segment_id is not None else self._segment
        clauses = [
            part.strip()
            for part in re.findall(r"[^。！？?!；;]+[。！？?!；;]*", raw)
            if part.strip()
        ]
        selected = []
        prefix = []
        for clause in clauses:
            stripped = clause.rstrip("。！？?!；; ")
            question = bool(_QUESTION.search(clause) or _QUESTION.search(stripped))
            request = _REQUEST.search(stripped)
            eligible = (
                len(stripped) >= 3
                and not _NARRATION.search(stripped)
                and not _MANAGEMENT.search(clause)
                and not _SMALL_TALK.search(stripped)
                and not _INCOMPLETE.search(clause)
                and (question or request is not None)
            )
            if eligible:
                # Preserve code identifiers and all subquestions, stripping only
                # a preceding statement within the same long spoken clause.
                start = request.start() if request and not question else 0
                selected.append(clause[start:])
                if start:
                    prefix.append(clause[:start])
            elif not selected:
                prefix.append(clause)

        text = "".join(selected)[-1600:]
        context = "；".join((*self._context, *prefix))[-1200:]
        if (
            event.is_final
            and raw
            and not _MANAGEMENT.search(raw)
            and (not self._context or self._context[-1] != raw)
        ):
            self._context.append(raw[-800:])
        if not text:
            return []
        normalized = re.sub(r"[\s。！？?!；;]+", "", text)
        active = self._active.get(segment)
        if active is not None and active[1:] == (normalized, event.is_final):
            return []
        question_id = active[0] if active else uuid4().hex
        self._active[segment] = (question_id, normalized, event.is_final)
        self._active.move_to_end(segment)
        while len(self._active) > 16:
            self._active.popitem(last=False)
        score = self._score(raw, normalized, bool(request), event.is_final)
        return [
            QuestionCandidate(
                question_id, text, self.subject, score, event.is_final, context
            )
        ]

    @staticmethod
    def _score(raw: str, normalized: str, request: bool, is_final: bool) -> float:
        score = 0.0
        if "?" in raw or "？" in raw:
            score += 0.35
        if _QUESTION.search(normalized):
            score += 0.45
        if request:
            score += 0.2
        if is_final:
            score += 0.1
        return min(score, 1.0)
