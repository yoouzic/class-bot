"""Validated, immutable scenario selection independent of process environment."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .knowledge.atomic import replace_atomic
from .knowledge.cache_manager import CacheManager
from .knowledge.policy import validate_index_mode
from .knowledge.quota_io import BinaryQuotaWriter, TextQuotaWriter

SCENARIO_KINDS = frozenset({"group_meeting", "competition", "defense"})
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}\Z")


@dataclass(frozen=True, slots=True)
class ScenarioProfile:
    scenario_id: str
    name: str
    kind: str
    library_path: str
    index_mode: str = "balanced"

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_id, str) or not _ID.fullmatch(self.scenario_id):
            raise ValueError("invalid scenario id")
        for field, maximum in (("name", 120), ("library_path", 32768)):
            value = getattr(self, field)
            if (not isinstance(value, str) or not value.strip() or len(value) > maximum
                    or any(ord(char) < 32 for char in value)):
                raise ValueError(f"invalid scenario {field}")
        if not isinstance(self.kind, str) or self.kind not in SCENARIO_KINDS:
            raise ValueError("invalid scenario kind")
        validate_index_mode(self.index_mode)


class ScenarioStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.last_backup_path: Path | None = None

    def load(self) -> tuple[ScenarioProfile, ...]:
        if not self.path.exists():
            return ()
        try:
            if self.path.stat().st_size > 1024 * 1024:
                raise ValueError("scenario configuration is too large")
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("schema") != 1 or not isinstance(payload.get("profiles"), list):
                raise ValueError("unsupported scenario configuration schema")
            profiles = tuple(ScenarioProfile(**item) for item in payload["profiles"])
            return self._validate(profiles)
        except (OSError, TypeError, KeyError, AttributeError, ValueError) as error:
            raise ValueError(f"cannot load scenario configuration: {error}") from error

    def save(self, profiles: Iterable[ScenarioProfile]) -> None:
        profiles = self._validate(tuple(profiles))
        manager = CacheManager(self.path.parent)
        with manager.mutation(), manager.quota():
            self._save_locked(profiles)

    def _save_locked(self, profiles: tuple[ScenarioProfile, ...]) -> None:
        payload = {"schema": 1, "profiles": [asdict(profile) for profile in profiles]}
        self.last_backup_path = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent,
                newline="\n",
                prefix=f".{self.path.name}.", suffix=".tmp", delete=False,
            ) as output:
                temporary = output.name
                json.dump(payload, TextQuotaWriter(output, self.path, maximum=1024**2), ensure_ascii=False, indent=2)
                output.flush()
                os.fsync(output.fileno())
            if self.path.exists():
                try:
                    self.load()
                except ValueError:
                    self.last_backup_path = self._preserve_invalid_configuration()
            replace_atomic(temporary, self.path)
        finally:
            if temporary and Path(temporary).exists():
                Path(temporary).unlink()

    def _preserve_invalid_configuration(self) -> Path:
        """Keep original bytes in a unique, durable recovery copy before replacing."""
        backup: Path | None = None
        try:
            with (
                self.path.open("rb") as original,
                tempfile.NamedTemporaryFile(
                    mode="wb", dir=self.path.parent,
                    prefix=f"{self.path.name}.invalid-", suffix=".bak", delete=False,
                ) as output,
            ):
                backup = Path(output.name)
                shutil.copyfileobj(original, BinaryQuotaWriter(output, self.path))
                output.flush()
                os.fsync(output.fileno())
            return backup
        except BaseException:
            if backup is not None:
                backup.unlink(missing_ok=True)
            raise

    @staticmethod
    def _validate(profiles: tuple[ScenarioProfile, ...]) -> tuple[ScenarioProfile, ...]:
        if len(profiles) > 1000:
            raise ValueError("too many scenarios")
        seen: set[str] = set()
        for profile in profiles:
            if not isinstance(profile, ScenarioProfile):
                raise TypeError("invalid scenario profile")
            if profile.scenario_id in seen:
                raise ValueError(f"duplicate scenario id: {profile.scenario_id}")
            seen.add(profile.scenario_id)
        return profiles
