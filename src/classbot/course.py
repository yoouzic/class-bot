from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

COURSES_ROOT = Path(__file__).resolve().parents[2] / "courses"
_COURSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_GENERIC_QUERY_RE = re.compile(
    r"什么样的(?:情况下|情况)?|在什么情况下|有什么|有哪些|"
    r"为什么|是什么|什么|怎样|如何|哪些|哪个|哪几|分几|"
    r"是不是|是否|能不能|可不可以|请问|我们的|我们|这个|那个|一个|问题"
)
_PAGE_MARKER = re.compile(r"^\[([^\]]+)\]\s*")


def _reference_query_terms(text: str) -> set[str]:
    # Strip question scaffolding before scoring: a shared "情况下" is not evidence.
    text = _GENERIC_QUERY_RE.sub(" ", text)
    fragments = re.split(r"[的呢吗啊呃吧呀在，。；：！？\s]+", text)
    return set().union(*(_terms(fragment) for fragment in fragments))


@dataclass(frozen=True, slots=True)
class ReferenceExcerpt:
    text: str
    page_label: str = ""
    source: str = "教材"


@dataclass(frozen=True, slots=True)
class CourseProfile:
    course_id: str
    hotwords: tuple[str, ...] = ()
    corrections: dict[str, str] | None = None
    reference_chunks: tuple[str, ...] = ()
    _reference_terms: frozenset[str] = field(
        init=False, repr=False, compare=False, default=frozenset()
    )

    def __post_init__(self) -> None:
        if self.corrections is None:
            object.__setattr__(self, "corrections", {})
        object.__setattr__(
            self,
            "_reference_terms",
            frozenset(
                term
                for chunk in self.reference_chunks
                for term in _terms(chunk)
            ),
        )

    def find_reference(self, question: str, limit: int = 3) -> tuple[str, ...]:
        return self._ranked_references(question, limit=limit)

    def find_reference_records(
        self,
        question: str,
        *,
        limit: int = 4,
        max_chars: int = 700,
    ) -> tuple[ReferenceExcerpt, ...]:
        references = self._ranked_references(question, limit=limit)
        query_terms = tuple(
            sorted(_terms(question), key=lambda term: (-len(term), term))
        )
        records = []
        for chunk in references:
            text = _excerpt_reference(chunk, query_terms, max_chars=max_chars)
            match = _PAGE_MARKER.match(text)
            records.append(
                ReferenceExcerpt(
                    text=text,
                    page_label=match.group(1) if match else "",
                )
            )
        return tuple(records)

    def reference_coverage(self, question: str) -> float:
        if not self.reference_chunks:
            return 0.0
        query_terms = _reference_query_terms(question)
        if not query_terms:
            return 1.0
        overlap = query_terms & self._reference_terms
        if not overlap:
            return 0.0
        # 触发阈值只用于决定是否补网络；三处实词重合已足以让本地候选
        # 先回答，避免长问句因分母过大而每题都联网。
        denominator = min(3, len(query_terms))
        return min(1.0, len(overlap) / denominator)

    def find_reference_excerpts(
        self,
        question: str,
        *,
        limit: int = 4,
        max_chars: int = 700,
    ) -> tuple[str, ...]:
        return tuple(
            record.text
            for record in self.find_reference_records(
                question, limit=limit, max_chars=max_chars
            )
        )

    def _ranked_references(
        self, question: str, *, limit: int
    ) -> tuple[str, ...]:
        if not question.strip() or not self.reference_chunks or limit <= 0:
            return ()

        query_terms = _reference_query_terms(question)
        scored: list[tuple[int, int, str]] = []
        for index, chunk in enumerate(self.reference_chunks):
            score = len(query_terms & _terms(chunk))
            if score:
                scored.append((score, -index, chunk))

        scored.sort(reverse=True)
        return tuple(chunk for _score, _index, chunk in scored[:limit])


def load_course_profile(course_id: str | None) -> CourseProfile:
    normalized = (course_id or "").strip()
    if not normalized:
        return CourseProfile("")
    if _COURSE_ID.fullmatch(normalized) is None:
        raise ValueError(f"invalid course id: {normalized}")

    profile_root = COURSES_ROOT / normalized
    if not profile_root.is_dir():
        raise FileNotFoundError(f"course profile not found: {normalized}")

    hotwords = _read_hotwords(profile_root / "hotwords.txt")
    corrections = _read_corrections(profile_root / "corrections.txt")
    reference_chunks = _read_reference(profile_root / "reference.txt")
    return CourseProfile(
        course_id=normalized,
        hotwords=hotwords,
        corrections=corrections,
        reference_chunks=reference_chunks,
    )


def _read_hotwords(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    words: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        word = line.strip()
        if not word or word.startswith("#") or word in seen:
            continue
        seen.add(word)
        words.append(word)
    return tuple(words)


def _read_corrections(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    corrections: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=>" not in line:
            continue
        wrong, right = (part.strip() for part in line.split("=>", 1))
        if wrong and right:
            corrections[wrong] = right
    return corrections


def _read_reference(path: Path) -> tuple[str, ...]:
    if not path.is_file():
        return ()
    return tuple(
        _clean_pdf_spacing(line.strip())
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _clean_pdf_spacing(text: str) -> str:
    """Remove extraction spaces inserted between adjacent Chinese characters."""
    text = re.sub(
        r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text
    )
    text = re.sub(
        r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、）》』】）])",
        "",
        text,
    )
    return re.sub(r"([（「『【《“‘])\s+", r"\1", text)


def _excerpt_reference(
    chunk: str,
    query_terms: tuple[str, ...],
    *,
    max_chars: int,
) -> str:
    page_marker, separator, body = chunk.partition("]")
    if separator:
        page_marker = f"{page_marker}]"
    else:
        page_marker = ""
        body = chunk

    available = max_chars - len(page_marker) - 1
    if available <= 0:
        return page_marker[:max_chars]

    positions = [
        body.find(term) for term in query_terms if term and body.find(term) >= 0
    ]
    hit_position = min(positions) if positions else 0
    before = min(120, hit_position)
    start = max(0, hit_position - before)
    end = min(len(body), start + available)
    if end - start < available:
        start = max(0, end - available)

    excerpt = body[start:end].strip()
    if start > 0:
        excerpt = "…" + excerpt
    if end < len(body):
        excerpt += "…"
    result = f"{page_marker} {excerpt}".strip()
    if len(result) > max_chars:
        result = f"{result[: max_chars - 1]}…"
    return result


def _terms(text: str) -> set[str]:
    normalized = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))
    terms = {normalized[index : index + 2] for index in range(len(normalized) - 1)}
    terms.update(
        normalized[index : index + 3] for index in range(len(normalized) - 2)
    )
    return terms
