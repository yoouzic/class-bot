import json

import pytest

from classbot.scenarios import ScenarioProfile, ScenarioStore


def test_legacy_four_field_profile_defaults_to_balanced(tmp_path):
    store = ScenarioStore(tmp_path / "scenarios.json")
    store.path.write_text(json.dumps({"schema": 1, "profiles": [{
        "scenario_id": "legacy", "name": "旧场景", "kind": "defense", "library_path": "D:/资料",
    }]}), encoding="utf-8")

    assert store.load()[0].index_mode == "balanced"
    assert ScenarioProfile("legacy", "旧场景", "defense", "D:/资料").index_mode == "balanced"


def test_full_profile_mode_is_persisted_and_round_trips(tmp_path):
    store = ScenarioStore(tmp_path / "scenarios.json")
    profile = ScenarioProfile("full", "全量资料", "competition", "D:/实验", "full")

    store.save([profile])

    assert store.load() == (profile,)
    assert json.loads(store.path.read_text(encoding="utf-8"))["profiles"][0]["index_mode"] == "full"


@pytest.mark.parametrize("mode", ["unknown", "FULL", "", None, []])
def test_profile_rejects_unknown_index_mode(mode):
    with pytest.raises(ValueError, match="index mode"):
        ScenarioProfile("demo", "答辩", "defense", "D:/资料", mode)


def test_saved_unknown_mode_cannot_load(tmp_path):
    store = ScenarioStore(tmp_path / "scenarios.json")
    store.path.write_text(json.dumps({"schema": 1, "profiles": [{
        "scenario_id": "demo", "name": "答辩", "kind": "defense", "library_path": "D:/资料",
        "index_mode": "unknown",
    }]}), encoding="utf-8")

    with pytest.raises(ValueError, match="index mode"):
        store.load()
