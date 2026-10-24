"""Persistent scene configuration obeys the same atomic disk budget as indexes."""

import pytest

from classbot.knowledge.cache_manager import CacheLimitError, CacheManager
from classbot.scenarios import ScenarioProfile, ScenarioStore


def test_scene_save_at_capacity_preserves_old_configuration(tmp_path):
    store = ScenarioStore(tmp_path / "scenarios.json")
    profile = ScenarioProfile("one", "原场景", "competition", str(tmp_path / "source"), "full")
    store.save((profile,))
    previous = store.path.read_bytes()
    manager = CacheManager(tmp_path)
    manager.set_limit(manager.usage_bytes() + 300)
    next_profile = ScenarioProfile("two", "新场景" * 20, "competition", str(tmp_path / "another-source"), "full")
    with pytest.raises(CacheLimitError):
        store.save((profile, next_profile))
    assert store.path.read_bytes() == previous
    assert store.load() == (profile,)
    assert not list(tmp_path.glob("*.tmp"))
    assert manager.usage_bytes() <= manager.limit_bytes


def test_invalid_scene_backup_cannot_grow_past_total_limit(tmp_path):
    path = tmp_path / "scenarios.json"
    path.write_bytes(b"invalid" * 500)
    original = path.read_bytes()
    manager = CacheManager(tmp_path)
    manager.set_limit(manager.usage_bytes() + 1000)
    profile = ScenarioProfile("one", "场景", "competition", str(tmp_path / "source"), "full")
    with pytest.raises(CacheLimitError):
        ScenarioStore(path).save((profile,))
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.bak"))
    assert manager.usage_bytes() <= manager.limit_bytes
