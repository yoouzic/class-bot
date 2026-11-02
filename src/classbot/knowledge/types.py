"""Source locations and build diagnostics, without inference about evidence."""
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class KnowledgeChunk:
    relative_path: str
    locator: str
    text: str

    @property
    def source(self) -> str:
        return f"{self.relative_path} · {self.locator}"


@dataclass(frozen=True, slots=True)
class KnowledgeHit:
    source: str
    text: str
    path: Path
    locator: str
    score: float
    section_context: str = ""
    section_source: str = ""


@dataclass(frozen=True, slots=True)
class BuildReport:
    indexed_files: int
    skipped_files: int
    issues: tuple[str, ...]
    chunks: int
    limited_files: int = 0
    duplicate_files: int = 0
