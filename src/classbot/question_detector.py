from __future__ import annotations

import re
from collections import OrderedDict, deque
from dataclasses import dataclass
from uuid import uuid4

from classbot.domain import QuestionCandidate, TranscriptEvent
from classbot.spoken_text import normalize_spoken_text

STRONG_QUESTION_MARKERS = (
    "哪些",
    "怎样",
    "哪几",
    "哪两",
    "分几",
    "什么",
    "哪年",
    "哪一年",
    "哪个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "多少",
    "多久",
    "是否",
    "请问",
    "有没有",
    "是不是",
    "能不能",
    "可不可以",
    "会不会",
    "什么时候",
    "哪一",
    "哪一个",
    "哪几个",
    "究竟",
    "什么是",
    "啥意思",
    "什么意思",
    "对不对",
    "对吧",
    "是吧",
    "记不记得",
    "在哪",
    "哪里",
)
QUESTION_PARTICLES = ("吗", "呢")
QUESTION_MARKS = ("?", "？")
QUESTION_ENDINGS = (
    "哪些",
    "吗",
    "呢",
    "呀",
    "啊",
    "吧",
    "对吧",
    "记不记得",
    "有没有",
    "是不是",
    "能不能",
    "可不可以",
    "会不会",
    "什么",
    "哪个",
    "哪一个",
    "哪几个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "怎样",
    "怎么样",
    "在哪",
    "哪里",
    "多少",
    "多久",
    "是否",
    "对不对",
    "来着",
)
NARRATIVE_CUES = (
    "现在打开教材",
    "打开教材",
    "看一下",
    "我们来看看",
    "我们接着",
    "接着呢",
    "接着看",
    "接着来看",
    "接下来",
    "大家注意",
    "大家一定",
    "牢牢记住",
    "回顾一下",
    "重新回顾",
    "我们学习了",
    "学习了",
    "我们了解了",
    "了解一下",
    "只需要掌握",
    "这个大家",
    "最后一点",
    "首先",
    "其次",
    "然后呢",
    "也就是说",
    "答案不在",
    "刚刚已经",
    "具体来说",
    "来看一下",
)
_VOCATIVE_PREFIX = re.compile(
    r"^(?:各位(?:同学|同学们)?|诸位|大家(?:好)?|同学们?|老师)"
    r"[\s,，:：、—-]*"
)
NARRATIVE_PREFIX_ENDINGS = (
    "解释",
    "说明",
    "讲解",
    "讲述",
    "介绍",
    "讨论",
    "分析",
    "回答",
    "告诉",
    "提到",
    "证明",
    "问",
    "说",
)
NARRATIVE_ASPECT_SUFFIXES = ("", "了", "过", "着")
QUESTION_CLOSERS = frozenset(
    " \t\r\n\"'”’)]}>》」』）】〕〉〗〙〛｝»"
)
RECENT_EVENT_LIMIT = 128
ACTIVE_SEGMENT_LIMIT = 16
CONTEXT_EVENT_LIMIT = 4
CONTEXT_MAX_CHARS = 320
QUESTION_SCORE_THRESHOLD = 0.5
_QUESTION_BOUNDARY_RE = re.compile(
    r"[。！？?!；;]+|所以|那么|接着(?:呢)?|然后(?:呢)?|"
    r"也就是说|具体来说|总的来说|回顾一下|刚(?:刚)?已经|"
    r"用一句话来说|完整的来说|听题目了没有(?:啊)?|诸位|"
    r"(?:现在各位(?:有没有记住)?)|(?:现在大家(?:再回忆一下)?)|"
    r"(?:我们(?:上节课)?(?:大致)?学习了)|(?:我们(?:又)?了解了)"
)
_SAFE_TOPIC_PREFIXES = (
    "哲学",
    "世界观",
    "物质",
    "意识",
    "唯物",
    "唯心",
    "马克思",
    "辩证",
    "第一性",
    "运动",
    "时间",
    "空间",
    "方法论",
)
_SAFE_NAME_PREFIX_RE = re.compile(
    r"^(?!(?:请问|为什么|为何|如何|怎么|是否|有没有|是不是|"
    r"能不能|可不可以|什么|哪个|哪些|哪种|谁))"
    r"(?!(?:属于|就是|称为|叫做|分为|包括|具有|认为|以为|所谓))"
    r"(?!(?:哲学|世界观|物质|意识|唯物|唯心|马克思|辩证|"
    r"第一性|运动|时间|空间|方法论))"
    r"[\u4e00-\u9fff]{2,4}(?=(?:第?\d+题|"
    r"第[一二三四五六七八九十百]+题|"
    + "|".join(map(re.escape, _SAFE_TOPIC_PREFIXES))
    + r"))"
)
_MANAGEMENT_RE = re.compile(
    r"(?:同学|大家|各位)?(?:在吗|到了吗|声音(?:大|小)(?:一点)?|"
    r"听(?:清|题目)?了?吗|能听到吗|请安静|打开教材|翻到)"
)
_CONFIRMATION_ONLY = ("对不对", "对吧", "是吧", "答案不就在这里吗")
_CONFIRMATION_RE = re.compile(
    r"(?:对不对|对吧|是吧|答案不就在这里)(?:[\u4e00-\u9fff]{0,6})?$"
)
_PARTICLE_TAILS = ("呢", "吗")
_FOLLOWUP_STARTS = (
    "那", "那么", "还有", "另外", "换成", "如果",
    "这个", "它", "他们", "她们",
)
_CONTEXT_QUESTION_SIGNALS = (
    "什么", "哪", "谁", "为什么", "为何", "如何", "怎么",
    "多少", "多久", "是否", "有没有", "是不是", "能不能",
    "可不可以", "会不会",
)
# 短"呢"句前接这些连词时是在续接前文的理由/叙述，不是在追问。
_DISCOURSE_CONTINUATION_PREFIXES = (
    "因为", "所以", "但是", "可是", "然后", "而且", "不过",
)


def _is_particle_question(
    normalized_text: str,
    *,
    has_context: bool,
) -> bool:
    if not normalized_text or not normalized_text.endswith(_PARTICLE_TAILS):
        return False
    if normalized_text.endswith("着呢"):
        return False
    if _MANAGEMENT_RE.search(normalized_text) or _CONFIRMATION_RE.search(
        normalized_text
    ):
        return False
    if normalized_text.endswith("吗"):
        return len(normalized_text) >= 4
    if has_context:
        return len(normalized_text) >= 2
    if normalized_text.startswith(_DISCOURSE_CONTINUATION_PREFIXES):
        return False
    return len(normalized_text) >= 5 and (
        "的" in normalized_text
        or normalized_text.startswith(
            ("这个", "那个", "它", "他们", "她们")
        )
    )


def _is_short_question(
    normalized_text: str,
    *,
    has_context: bool,
) -> bool:
    if len(normalized_text) >= 7:
        return False
    return _is_particle_question(normalized_text, has_context=has_context)


_INCOMPLETE_PARTIAL_TAILS = (
    "是如何",
    "是怎么",
    "是怎样",
    "是为什么",
    "是哪些",
    "是哪个",
    "是哪一个",
)
_TRAILING_FILLERS = (
    "同学们",
    "各位",
    "同学",
    "嗯",
    "呃",
    "么儿",
    "呀",
    "啊",
    "吧",
)
_ACADEMIC_HINTS = (
    "认识",
    "实践",
    "真理",
    "感性",
    "理性",
    "飞跃",
    "哲学",
    "世界观",
    "物质",
    "意识",
    "唯物",
    "唯心",
    "马克思",
    "辩证",
    "第一性",
    "运动",
    "时间",
    "空间",
    "方法论",
    "原理",
    "概念",
    "关系",
    "特性",
    "选项",
    "联系",
    "矛盾",
    "规律",
    "本质",
    "现象",
    "实体",
    "理念",
    "定义",
    "数据",
    "接口",
    "配置",
    "版本",
    "函数",
    "参数",
)
_COMPREHENSION_TAILS = (
    "能理解吧",
    "理解吧",
    "能理解吗",
    "能明白吧",
    "明白了吧",
    "听懂了吗",
    "懂了吗",
    "清楚了吗",
)
_PARTICLE_QUESTION_SIGNALS = (
    "哪些",
    "怎样",
    "怎么样",
    "哪几",
    "哪两",
    "分几",
    "还有",
    "有没有",
    "是不是",
    "能不能",
    "可不可以",
    "会不会",
    "什么",
    "哪个",
    "谁",
    "为什么",
    "为何",
    "如何",
    "怎么",
    "在哪",
    "哪里",
    "印象",
    "记得",
    "请问",
    "对不对",
)
_NONQUESTION_MARKER_PREFIXES = (
    "就是",
    "遇到",
    "讲到",
    "提到",
    "包括",
    "包含",
    "解决",
    "说明",
)
_NONQUESTION_START_PREFIXES = ("就是", "不管", "强调的就是")
_TRAILING_QUESTION_PARTICLES = ("吗", "呢", "呀", "啊", "吧", "来着")
_TEACHER_ANSWER_CUES = (
    "很好",
    "非常好",
    "正确",
    "答案",
    "没错",
    "对的",
)
_CLASSROOM_RESPONSE_CUES = ("谁来回答", "请回答", "来回答")
_TERMINAL_QUESTION_MARKERS = frozenset(
    {
        "什么",
        "哪个",
        "哪一个",
        "谁",
        "哪年",
        "哪一年",
        "多少",
    }
)
_ACADEMIC_SPAN_DISTANCE = 28
_LONG_NOISY_QUESTION_LENGTH = 20
_INCOMPLETE_QUESTION_RE = re.compile(
    r"什么样的(?:情(?:况(?:下+)?)?)?(?:呢|啊|呀|吧)?$|"
    r"(?:是不是|能不能|可不可以)(?:啊|呀|呢|吧|不是你)*$"
)
_ALTERNATIVE_RIGHT_WRONG_RE = re.compile(r"(?:对的|错的|正确|错误)[，,]?还是")


def _is_alternative_question(normalized_text: str) -> bool:
    """究竟…还是 and 对的…还是 pose a choice; either side is the answer."""
    if "究竟" in normalized_text and "还是" in normalized_text:
        return True
    return bool(_ALTERNATIVE_RIGHT_WRONG_RE.search(normalized_text))


@dataclass(slots=True)
class _ActiveQuestion:
    question_id: str
    normalized_text: str
    is_final: bool
    segment_id: int | str | None


def normalize_text(text: str) -> str:
    """Keep letters and numbers while removing whitespace and punctuation."""
    if not text:
        return ""
    return "".join(character for character in text if character.isalnum())


def clean_question_text(text: str) -> str:
    """Remove only explicit vocatives before normalizing question text."""
    return normalize_text(_VOCATIVE_PREFIX.sub("", text.strip()))


def _strip_trailing_fillers(text: str) -> str:
    for filler in _TRAILING_FILLERS:
        while text.endswith(filler):
            text = text[: -len(filler)]
    return text


def _strip_classroom_response_tail(text: str) -> str:
    positions = [
        position
        for cue in _CLASSROOM_RESPONSE_CUES
        if (position := text.find(cue)) >= 8
    ]
    if not positions:
        return text
    return text[: min(positions)].rstrip(" \t\r\n,，:：")


def _strip_safe_name_prefix(text: str) -> str:
    match = _SAFE_NAME_PREFIX_RE.match(text)
    if match is None:
        return text
    return text[match.end() :]


def _normalized_clause(text: str) -> str:
    text = _VOCATIVE_PREFIX.sub("", text.strip())
    text = _strip_safe_name_prefix(text)
    text = _strip_trailing_fillers(text)
    return normalize_text(text)


def _has_teaching_lead(text: str) -> bool:
    return bool(
        re.search(r"(?:学习|了解|讲解|介绍|掌握|回顾)(?:了|过)?", text)
    )


def _valid_internal_question_suffix(marker: str, suffix: str) -> bool:
    suffix = suffix.lstrip("吗呢呀啊吧呃嗯")
    if not suffix:
        return True
    if any(suffix.startswith(prefix) for prefix in _CLASSROOM_RESPONSE_CUES):
        return False
    if any(cue in suffix for cue in _TEACHER_ANSWER_CUES):
        return True
    if suffix.startswith(("你", "我", "他", "大家", "同学")):
        return False
    if marker in {"什么", "哪个", "哪一个", "谁", "哪年", "哪一年", "多少", "多久"}:
        if len(suffix) <= 2:
            # 多少度 / 几个人 / 什么料: a short unit or noun completes the
            # question without further words.
            return True
        return suffix.startswith(
            (
                "是",
                "样",
                "的",
                "出生",
                "名字",
                "时候",
                "概念",
                "定义",
                "关系",
                "内容",
                "作用",
                "特点",
                "原因",
                "意义",
                "问题",
                "东西",
                "哪",
            )
        )
    return len(suffix) >= 2


def _looks_like_question_clause(
    raw_clause: str,
    normalized_clause: str,
    original_text: str,
    *,
    is_final: bool = False,
    has_context: bool = False,
) -> bool:
    if not normalized_clause:
        return False
    if normalized_clause.startswith(_NONQUESTION_START_PREFIXES) and not (
        QuestionDetector._has_question_mark(raw_clause)
    ):
        return False
    if QuestionDetector._has_question_mark(raw_clause):
        return True
    if QuestionDetector._has_question_ending(
        raw_clause, normalized_clause
    ):
        if not is_final and normalized_clause.endswith(
            _INCOMPLETE_PARTIAL_TAILS
        ):
            return False
        if normalized_clause.endswith("着呢"):
            # "放着呢/坐着呢" is a continuative state, not a question 呢.
            return False
        if (
            normalized_clause.endswith(_TRAILING_QUESTION_PARTICLES)
            and not any(
                signal in normalized_clause
                for signal in _PARTICLE_QUESTION_SIGNALS
            )
            and len(normalized_clause) < 10
            and not _is_particle_question(
                normalized_clause, has_context=has_context
            )
        ):
            # A bare "…吗/呢" tail on a short clause is usually a filler or
            # confirmation; a longer one ("…支持中文吗") is a real question.
            return False
        if normalized_clause.endswith("什么") and normalized_clause.count(
            "什么"
        ) > 1:
            return False
        return not (
            _has_teaching_lead(original_text)
            and normalized_clause.startswith(("什么是", "什么样的"))
        )

    if normalized_clause.startswith(
        (
            "为什么",
            "为何",
            "如何",
            "怎么",
            "是否",
            "有没有",
            "是不是",
            "能不能",
            "可不可以",
            "请问",
            "什么是",
            "是什么",
            "究竟",
            "什么样的",
            "哪个",
            "哪些",
            "哪种",
            "谁",
        )
    ):
        return is_final or not normalized_clause.endswith(
            _INCOMPLETE_PARTIAL_TAILS
        )

    if _is_alternative_question(normalized_clause):
        return is_final or not normalized_clause.endswith(
            _INCOMPLETE_PARTIAL_TAILS
        )

    for marker in sorted(STRONG_QUESTION_MARKERS, key=len, reverse=True):
        marker_start = normalized_clause.find(marker)
        if marker_start <= 0:
            continue
        if marker in ("对不对", "对吧", "是吧"):
            continue
        if marker.startswith("能") and normalized_clause[marker_start - 1] == "不":
            # "不能不能" contains "能不能" across the boundary; that negation
            # is not a yes/no question.
            continue
        marker_end = marker_start + len(marker)
        if marker == "是否" and normalized_clause[marker_end : marker_end + 1] in {
            "认",
            "定",
        }:
            # "就是否认联系" splits 否认 into 是否+认; that 否 is not the
            # interrogative 是否.
            continue
        if (
            _has_teaching_lead(original_text)
            and not QuestionDetector._has_question_ending(
                raw_clause, normalized_clause
            )
        ):
            continue
        prefix = normalized_clause[:marker_start]
        if prefix.endswith(_NONQUESTION_MARKER_PREFIXES):
            continue
        if any(
            prefix.endswith(ending + aspect)
            for ending in NARRATIVE_PREFIX_ENDINGS
            for aspect in NARRATIVE_ASPECT_SUFFIXES
        ):
            continue
        marker_end = marker_start + len(marker)
        suffix = normalized_clause[marker_end:]
        if _valid_internal_question_suffix(marker, suffix):
            return True
    return False


def _trim_teacher_answer(normalized_text: str) -> str:
    """Drop an answer/feedback tail that follows a detected question."""
    for marker in sorted(STRONG_QUESTION_MARKERS, key=len, reverse=True):
        marker_start = normalized_text.find(marker)
        while marker_start >= 0:
            marker_end = marker_start + len(marker)
            suffix = normalized_text[marker_end:]
            # The cue must open the tail; a later mention such as "带着你
            # 自己的一个答案" is an instruction, not the teacher answering.
            if suffix and suffix.lstrip("所以因为那这就啊呃嗯").startswith(
                _TEACHER_ANSWER_CUES
            ):
                topic_positions = [
                    normalized_text.rfind(topic, 0, marker_start + 1)
                    for topic in _SAFE_TOPIC_PREFIXES
                ]
                topic_start = max(topic_positions, default=-1)
                if topic_start >= 0:
                    return normalized_text[topic_start:marker_end]
                return normalized_text[:marker_end]
            marker_start = normalized_text.find(marker, marker_start + 1)
    return normalized_text


def _has_question_form_evidence(normalized_text: str) -> bool:
    """Clear interrogative form, independent of course vocabulary."""
    if normalized_text.endswith((*QUESTION_ENDINGS, *QUESTION_PARTICLES)):
        return True
    if _is_alternative_question(normalized_text):
        return True
    if "到底" in normalized_text and any(
        marker in normalized_text
        for marker in ("怎样", "怎么", "如何", "什么", "谁", "哪个", "哪些")
    ):
        return True
    if any(
        word in normalized_text for word in ("什么时候", "啥时候", "何时")
    ):
        return True
    return normalized_text.startswith(
        (
            "为什么",
            "为何",
            "请问",
            "究竟",
            "如何",
            "怎么",
            "什么是",
            "是什么",
            "什么样的",
            "哪些",
            "哪个",
            "是不是",
            "有没有",
            "能不能",
            "可不可以",
            "会不会",
            "什么时候",
        )
    )


def _extract_recent_academic_span(
    normalized_text: str,
    hotwords: tuple[str, ...],
) -> str:
    if len(normalized_text) <= _LONG_NOISY_QUESTION_LENGTH:
        return normalized_text

    hints = tuple(
        dict.fromkeys((*_ACADEMIC_HINTS, *_SAFE_TOPIC_PREFIXES, *hotwords))
    )
    for marker in sorted(STRONG_QUESTION_MARKERS, key=len, reverse=True):
        marker_start = normalized_text.rfind(marker)
        if marker_start <= 0:
            continue
        if marker == "什么" and any(
            word in normalized_text[max(0, marker_start - 16) : marker_start]
            for word in ("不管", "无论", "任何", "随便")
        ):
            # "不管…发生什么" uses 什么 as an indefinite pronoun, not a
            # question; the span rescue must not manufacture a candidate.
            continue
        suffix = normalized_text[marker_start + len(marker) :]
        suffix_is_valid = _valid_internal_question_suffix(marker, suffix)
        if not suffix_is_valid and marker not in _TERMINAL_QUESTION_MARKERS:
            continue

        marker_end = marker_start + len(marker)
        end = (
            marker_end
            if marker in _TERMINAL_QUESTION_MARKERS
            else len(normalized_text)
        )
        anchor_positions = [
            position
            for hint in hints
            if (position := normalized_text.rfind(hint, 0, marker_start + 1)) >= 0
            and marker_start - position <= _ACADEMIC_SPAN_DISTANCE
        ]
        if not anchor_positions:
            # A teacher may enumerate after a long lead-in whose topic was
            # separated by a discourse boundary. Keep the short question;
            # _context_prefix supplies the preceding explanation separately.
            tail = normalized_text[marker_start:]
            if (
                marker in {"哪两", "哪几", "哪些"}
                and suffix_is_valid
                and 8 <= len(tail) <= _LONG_NOISY_QUESTION_LENGTH
            ):
                return tail
            continue

        start = min(anchor_positions)
        span = normalized_text[start:end]
        if len(span) >= 8:
            return span
    return normalized_text


def _extract_question_text(
    raw_text: str,
    hotwords: tuple[str, ...] = (),
    *,
    is_final: bool = False,
    has_context: bool = False,
) -> str:
    """Extract the last complete answerable-looking clause from speech."""
    text = normalize_spoken_text(
        _strip_classroom_response_tail(
            _VOCATIVE_PREFIX.sub("", raw_text.strip())
        )
    )
    if not text:
        return ""

    clauses = [
        clause.strip()
        for clause in _QUESTION_BOUNDARY_RE.split(text)
        if clause.strip()
    ]
    if not clauses:
        clauses = [text]

    for clause in reversed(clauses):
        normalized = _normalized_clause(clause)
        if _looks_like_question_clause(
            clause, normalized, text,
            is_final=is_final, has_context=has_context,
        ):
            normalized = _trim_teacher_answer(normalized)
            return _extract_recent_academic_span(normalized, hotwords)
        span = _extract_recent_academic_span(normalized, hotwords)
        if span != normalized and _looks_like_question_clause(
            span, span, span,
            is_final=is_final, has_context=has_context,
        ):
            return _trim_teacher_answer(span)
    return ""


class QuestionDetector:
    """Detect local question candidates and track one active speech segment."""

    def __init__(
        self,
        subject: str,
        minimum_length: int = 7,
        hotwords: tuple[str, ...] = (),
    ) -> None:
        self.subject = subject
        self.minimum_length = minimum_length
        self.hotwords = tuple(word for word in hotwords if word.strip())
        self._fallback_segment_id: int | str = uuid4().hex
        self._active_questions: OrderedDict[int | str, _ActiveQuestion] = OrderedDict()
        self._recent_events: deque[
            tuple[str, bool, int, int, int | str | None, bool]
        ] = deque(maxlen=RECENT_EVENT_LIMIT)
        self._recent_context: deque[str] = deque(maxlen=CONTEXT_EVENT_LIMIT)

    def begin_segment(self, segment_id: int | str | None = None) -> None:
        """Switch the fallback identity without dropping other segment states."""
        self._fallback_segment_id = (
            segment_id if segment_id is not None else uuid4().hex
        )

    def feed(self, event: TranscriptEvent) -> list[QuestionCandidate]:
        """Return zero or one candidate for a transcript event."""
        has_context = any(
            self._is_context_eligible(text) for text in self._recent_context
        )
        normalized_text = _extract_question_text(
            event.text,
            hotwords=self.hotwords,
            is_final=event.is_final,
            has_context=has_context,
        )
        context = self._context_for(event.text, normalized_text)
        short_question = _is_short_question(
            normalized_text, has_context=bool(context)
        )
        segment_id = self._segment_id_for(event)
        signature = (
            normalized_text,
            event.is_final,
            event.start_ms,
            event.end_ms,
            segment_id,
            self._has_question_mark(event.text),
        )
        confidence = self._score(event, normalized_text)
        if signature in self._recent_events:
            if event.is_final and (
                len(normalized_text) < self.minimum_length
                or confidence < QUESTION_SCORE_THRESHOLD
            ):
                self._active_questions.pop(segment_id, None)
            return []
        self._recent_events.append(signature)
        if event.is_final:
            self._remember_context(event.text)

        if not normalized_text or not self._is_answerable(
            event, normalized_text, short_question=short_question
        ):
            if event.is_final:
                self._active_questions.pop(segment_id, None)
            return []

        too_short = len(normalized_text) < self.minimum_length
        if (
            too_short
            and not short_question
            and not self._is_choice_question(normalized_text)
        ) or confidence < QUESTION_SCORE_THRESHOLD:
            if event.is_final:
                self._active_questions.pop(segment_id, None)
            return []

        question_id = self._question_id_for(segment_id, normalized_text)
        candidate = QuestionCandidate(
            question_id=question_id,
            text=normalized_text,
            subject=self.subject,
            confidence=confidence,
            is_final=event.is_final,
            context=context,
        )

        self._set_active_question(
            segment_id,
            _ActiveQuestion(
                question_id=question_id,
                normalized_text=normalized_text,
                is_final=event.is_final,
                segment_id=segment_id,
            ),
        )
        return [candidate]

    def _is_answerable(
        self,
        event: TranscriptEvent,
        normalized_text: str,
        *,
        short_question: bool = False,
    ) -> bool:
        if (
            len(normalized_text) < self.minimum_length
            and not short_question
            and not self._is_choice_question(normalized_text)
        ):
            return False

        if _INCOMPLETE_QUESTION_RE.search(normalized_text):
            return False
        if (
            not event.is_final
            and "是不是" in normalized_text
            and not normalized_text.endswith(QUESTION_PARTICLES)
            and not self._has_question_mark(event.text)
            and not any(
                marker in normalized_text
                for marker in STRONG_QUESTION_MARKERS
                if marker not in {"是不是", "对不对", "对吧", "是吧"}
            )
        ):
            # In running speech this is often a confirmation followed by more
            # lecture, not a completed yes/no proposition. Recheck the final.
            return False
        if _MANAGEMENT_RE.search(normalized_text):
            return False

        if normalized_text.endswith(_COMPREHENSION_TAILS):
            # "这个大家能理解吧" checks the audience, it does not ask a
            # course question, even with academic words in the clause.
            return False

        has_academic_hint = self._has_academic_hint(normalized_text)
        if (
            len(normalized_text) > _LONG_NOISY_QUESTION_LENGTH
            and not has_academic_hint
            and not self._is_choice_question(normalized_text)
            and not self._has_question_mark(event.text)
            and not _has_question_form_evidence(normalized_text)
        ):
            return False

        if (
            _CONFIRMATION_RE.search(normalized_text)
            and not self._has_academic_hint(normalized_text)
        ):
            return False

        return event.is_final or not normalized_text.endswith(
            _INCOMPLETE_PARTIAL_TAILS
        )

    def _has_academic_hint(self, normalized_text: str) -> bool:
        return any(
            hint in normalized_text
            for hint in (*_ACADEMIC_HINTS, *self.hotwords)
        )

    def _context_for(self, raw_text: str, question_text: str) -> str:
        recent = [
            text
            for text in self._recent_context
            if self._is_context_eligible(text)
        ]
        question_hints = tuple(
            hint
            for hint in (*_ACADEMIC_HINTS, *self.hotwords)
            if hint in question_text
        )
        if question_hints:
            related = [
                text
                for text in recent
                if any(hint in text for hint in question_hints)
            ]
            recent = related or recent[-1:]

        parts = recent[-2:]
        prefix = self._context_prefix(raw_text, question_text)
        if prefix:
            parts.append(prefix)
        if not parts:
            return ""

        compact: list[str] = []
        remaining = CONTEXT_MAX_CHARS
        for part in reversed(parts[-2:]):
            if remaining <= 0:
                break
            clipped = part[-remaining:]
            if clipped:
                compact.append(clipped)
                remaining -= len(clipped) + 1
        return "；".join(reversed(compact))

    def _remember_context(self, raw_text: str) -> None:
        normalized = normalize_text(raw_text)
        if not self._is_context_eligible(normalized):
            return
        if self._recent_context and self._recent_context[-1] == normalized:
            return
        self._recent_context.append(normalized)

    @staticmethod
    def _context_prefix(raw_text: str, question_text: str) -> str:
        normalized_raw = normalize_text(raw_text)
        normalized_question = normalize_text(question_text)
        if not normalized_raw or not normalized_question:
            return ""
        question_start = normalized_raw.rfind(normalized_question)
        if question_start <= 0:
            return ""
        prefix = normalized_raw[:question_start]
        return prefix if QuestionDetector._is_context_eligible(prefix) else ""

    @staticmethod
    def _is_context_eligible(normalized_text: str) -> bool:
        if len(normalized_text) < 8:
            return (
                len(normalized_text) >= 4
                and (
                    normalized_text.endswith(("吗", "呢", "呀", "啊"))
                    or any(
                        signal in normalized_text
                        for signal in _CONTEXT_QUESTION_SIGNALS
                    )
                )
                and not normalized_text.endswith("着呢")
                and not _MANAGEMENT_RE.search(normalized_text)
                and not _CONFIRMATION_RE.search(normalized_text)
            )
        if _MANAGEMENT_RE.search(normalized_text):
            return False
        return not _CONFIRMATION_RE.search(normalized_text)

    @staticmethod
    def _is_choice_question(normalized_text: str) -> bool:
        return bool(
            re.search(
                r"第(?:\d+|[一二三四五六七八九十百]+)题|"
                r"选(?:哪个|哪项|哪一项)|属于哪|正确的是",
                normalized_text,
            )
        )

    def _segment_id_for(self, event: TranscriptEvent) -> int | str:
        if event.segment_id is not None:
            return event.segment_id
        return self._fallback_segment_id

    def _question_id_for(
        self, segment_id: int | str, normalized_text: str
    ) -> str:
        active = self._active_questions.get(segment_id)
        if active is not None:
            self._active_questions.move_to_end(segment_id)
            return active.question_id
        return uuid4().hex

    def _set_active_question(
        self, segment_id: int | str, active: _ActiveQuestion
    ) -> None:
        self._active_questions[segment_id] = active
        self._active_questions.move_to_end(segment_id)
        while len(self._active_questions) > ACTIVE_SEGMENT_LIMIT:
            self._active_questions.popitem(last=False)

    @classmethod
    def _score(cls, event: TranscriptEvent, normalized_text: str) -> float:
        has_strong_marker = cls._has_strong_marker(normalized_text)
        has_ending_particle = normalized_text.endswith(QUESTION_PARTICLES)
        has_question_mark = cls._has_question_mark(event.text)
        if (
            has_strong_marker
            and not has_question_mark
            and any(mark in event.text for mark in QUESTION_MARKS)
        ):
            has_strong_marker = False

        score = 0.0
        if has_strong_marker:
            score += 0.55
        if has_ending_particle:
            score += 0.55
        if has_question_mark:
            score += 0.65
        if event.pause_ms >= 300:
            score += 0.1
        if event.is_final:
            score += 0.1
        return min(score, 1.0)

    @classmethod
    def _is_context_noise(cls, raw_text: str, normalized_text: str) -> bool:
        if len(normalized_text) < 16:
            return False
        if cls._has_question_ending(raw_text, normalized_text):
            return False
        return any(cue in raw_text or cue in normalized_text for cue in NARRATIVE_CUES)

    @classmethod
    def _has_question_ending(cls, raw_text: str, normalized_text: str) -> bool:
        if cls._has_question_mark(raw_text):
            return True
        return any(normalized_text.endswith(ending) for ending in QUESTION_ENDINGS)

    @staticmethod
    def _has_strong_marker(normalized_text: str) -> bool:
        if any(
            normalized_text.startswith(marker) or normalized_text.endswith(marker)
            for marker in STRONG_QUESTION_MARKERS
        ):
            return True
        if _is_alternative_question(normalized_text):
            return True

        for marker in sorted(
            STRONG_QUESTION_MARKERS,
            key=len,
            reverse=True,
        ):
            marker_start = normalized_text.find(marker)
            if marker_start <= 0:
                continue
            if marker.startswith("能") and normalized_text[marker_start - 1] == "不":
                continue
            marker_end = marker_start + len(marker)
            if marker == "是否" and normalized_text[marker_end : marker_end + 1] in {
                "认",
                "定",
            }:
                continue
            if marker_end >= len(normalized_text):
                continue
            prefix = normalized_text[:marker_start]
            return not any(
                prefix.endswith(ending + aspect)
                for ending in NARRATIVE_PREFIX_ENDINGS
                for aspect in NARRATIVE_ASPECT_SUFFIXES
            )
        return False

    @staticmethod
    def _has_question_mark(raw_text: str) -> bool:
        text = raw_text.rstrip()
        while text and text[-1] in QUESTION_CLOSERS:
            text = text[:-1].rstrip()
        return text.endswith(QUESTION_MARKS)
