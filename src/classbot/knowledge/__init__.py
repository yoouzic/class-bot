"""Local folder libraries with explicit provenance and semantic candidates."""
from .embedding import DEFAULT_MODEL, DEFAULT_MODEL_REVISION, LocalEmbedder
from .index import KnowledgeIndex
from .types import BuildReport, KnowledgeChunk, KnowledgeHit

__all__ = [
    "DEFAULT_MODEL", "DEFAULT_MODEL_REVISION", "BuildReport", "KnowledgeChunk", "KnowledgeHit", "KnowledgeIndex",
    "LocalEmbedder",
]
