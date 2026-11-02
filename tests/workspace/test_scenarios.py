import json

import pytest


def test_scenario_store_round_trip_and_missing_store(tmp_path):
    from classbot.scenarios import ScenarioProfile, ScenarioStore
    store = ScenarioStore(tmp_path / "config" / "scenarios.json")
    assert store.load() == ()
    profiles = (
        ScenarioProfile("research", "课题组会", "group_meeting", str(tmp_path / "资料")),
        ScenarioProfile("contest", "竞赛答辩", "competition", str(tmp_path / "竞赛")),
    )
    store.save(profiles)
    assert store.load() == profiles
    assert json.loads(store.path.read_text(encoding="utf-8"))["schema"] == 1


@pytest.mark.parametrize("field,value", [
    ("scenario_id", "../unsafe"), ("scenario_id", ""), ("name", "  "),
    ("kind", "unknown"), ("library_path", ""), ("library_path", "bad\x00folder"),
])
def test_scenario_profile_validates_untrusted_config(field, value):
    from classbot.scenarios import ScenarioProfile
    values = {"scenario_id": "demo", "name": "答辩", "kind": "defense", "library_path": "D:/材料"}
    values[field] = value
    with pytest.raises(ValueError):
        ScenarioProfile(**values)


def test_duplicate_ids_and_bad_json_do_not_overwrite_valid_config(tmp_path):
    from classbot.scenarios import ScenarioProfile, ScenarioStore
    store = ScenarioStore(tmp_path / "scenarios.json")
    profile = ScenarioProfile("demo", "答辩", "defense", "D:/材料")
    store.save([profile])
    before = store.path.read_bytes()
    with pytest.raises(ValueError, match="duplicate"):
        store.save([profile, profile])
    assert store.path.read_bytes() == before
    store.path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="scenario"):
        store.load()


@pytest.mark.parametrize("original", [b"{broken", b"\xffinvalid-utf8", b'{"schema":99}'])
def test_replacing_invalid_configuration_preserves_exact_original(tmp_path, original):
    from classbot.scenarios import ScenarioProfile, ScenarioStore

    store = ScenarioStore(tmp_path / "scenarios.json")
    store.path.write_bytes(original)
    profile = ScenarioProfile("new", "新场景", "defense", str(tmp_path))

    store.save([profile])

    backups = list(tmp_path.glob("scenarios.json.invalid-*.bak"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == original
    assert store.load() == (profile,)
    assert store.last_backup_path == backups[0]


def test_failed_invalid_configuration_backup_aborts_replacement(tmp_path, monkeypatch):
    from classbot import scenarios

    store = scenarios.ScenarioStore(tmp_path / "scenarios.json")
    original = b"{recoverable original"
    store.path.write_bytes(original)
    real_temporary = scenarios.tempfile.NamedTemporaryFile

    def temporary(*args, **kwargs):
        if kwargs.get("suffix") == ".bak":
            raise OSError("backup disk unavailable")
        return real_temporary(*args, **kwargs)

    monkeypatch.setattr(scenarios.tempfile, "NamedTemporaryFile", temporary)
    profile = scenarios.ScenarioProfile("new", "新场景", "defense", str(tmp_path))

    with pytest.raises(OSError, match="backup"):
        store.save([profile])

    assert store.path.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []


def test_invalid_configuration_backups_never_replace_previous_recovery_copy(tmp_path):
    from classbot.scenarios import ScenarioProfile, ScenarioStore

    store = ScenarioStore(tmp_path / "scenarios.json")
    profile = ScenarioProfile("new", "新场景", "defense", str(tmp_path))
    for original in (b"{first", b"{second"):
        store.path.write_bytes(original)
        store.save([profile])

    assert {path.read_bytes() for path in tmp_path.glob("scenarios.json.invalid-*.bak")} == {
        b"{first", b"{second",
    }


def test_valid_configuration_update_does_not_create_recovery_backup(tmp_path):
    from classbot.scenarios import ScenarioProfile, ScenarioStore

    store = ScenarioStore(tmp_path / "scenarios.json")
    profile = ScenarioProfile("new", "新场景", "defense", str(tmp_path))
    store.save([profile])
    store.save([])

    assert store.load() == ()
    assert list(tmp_path.glob("*.bak")) == []
