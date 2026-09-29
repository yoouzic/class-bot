from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

COURSES_ROOT = Path(__file__).resolve().parents[2] / "courses"
_COURSE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True, slots=True)
class CourseProfile:
    course_id: str
    hotwords: tuple[str, ...] = ()
    corrections: dict[str, str] | None = None
    reference_chunks: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.corrections is None:
            object.__setattr__(self, "corrections", {})

    def find_reference(self, question: str, limit: int = 3) -> tuple[str, ...]:
        if not question.strip() or not self.reference_chunks or limit <= 0:
            return ()

        query_terms = _terms(question)
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
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _terms(text: str) -> set[str]:
    normalized = "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text))
    terms = {normalized[index : index + 2] for index in range(len(normalized) - 1)}
    terms.update(
        normalized[index : index + 3] for index in range(len(normalized) - 2)
    )
    return terms
