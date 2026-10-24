from dataclasses import replace
from types import SimpleNamespace

import pytest

from classbot.config import Settings
from classbot.scenarios import ScenarioProfile
from classbot.workspace import WorkspaceController


class Window:
    selected_scenario_id = "demo"
    audio_device = 7
    audio_source = "loopback"

    def set_library_status(self, text, ready=False):
        self.library = (text, ready)

    def set_configuration_busy(self, busy):
        self.busy = busy


def workspace(tmp_path):
    settings = replace(
        Settings(None, "历史", "deepseek-flash", "https://example", "cpu"),
        cache_dir=tmp_path / "cache",
    )
    runtime = SimpleNamespace(
        current_runtime=None, settings=settings, dependencies=None
    )
    workspace = WorkspaceController(settings, Window(), runtime)
    workspace.profiles = (
        ScenarioProfile("demo", "国赛", "competition", str(tmp_path)),
    )
    return workspace


@pytest.mark.asyncio
async def test_defense_start_needs_ready_index_never_uses_microphone(tmp_path):
    controller = workspace(tmp_path)
    with pytest.raises(ValueError, match="索引"):
        await controller.prepare_session()
    assert controller.runtime.dependencies is None


@pytest.mark.asyncio
async def test_runtime_factories_bind_profile_index_and_playback(tmp_path):
    controller = workspace(tmp_path)
    index = SimpleNamespace(root=tmp_path, is_stale=lambda: False)
    controller.indices["demo"] = index
    await controller.prepare_session()
    assert controller.runtime.settings.subject == "国赛"
    assert controller.runtime.settings.audio_source == "loopback"
    capture = controller.runtime.dependencies.capture_factory()
    assert capture.device == 7
    detector = controller.runtime.dependencies.detector_factory("ignored")
    assert detector.subject == "国赛"
    client = controller.runtime.dependencies.client_factory(controller.runtime.settings)
    try:
        assert client._knowledge_index is index
        assert client._scenario.name == "国赛"
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_defense_keeps_a_question_across_a_brief_pause_but_classroom_keeps_its_timing(tmp_path):
    controller = workspace(tmp_path)
    controller.indices["demo"] = SimpleNamespace(root=tmp_path, is_stale=lambda: False)

    def speech_ends(segmenter):
        voiced, silent = b"\x01\x00" * 320, b"\x00\x00" * 320
        segmenter._is_speech = lambda frame: frame == voiced
        # A 400ms pause between two parts of one question, then a real end.
        frames = [voiced] * 4 + [silent] * 20 + [voiced] * 4 + [silent] * 30
        return [i for i, frame in enumerate(frames) if segmenter.accept(frame).speech_ended]

    await controller.prepare_session()
    defense = controller.runtime.dependencies.segmenter_factory()
    assert speech_ends(defense) == [57]
    assert defense.max_segment_ms == 60_000

    controller.window.selected_scenario_id = "__classroom__"
    await controller.prepare_session()
    classroom = controller.runtime.dependencies.segmenter_factory()
    assert speech_ends(classroom) == [19, 43]


@pytest.mark.asyncio
async def test_changed_folder_snapshot_requires_refresh(tmp_path):
    controller = workspace(tmp_path)
    controller.indices["demo"] = SimpleNamespace(root=tmp_path, is_stale=lambda: True)
    with pytest.raises(ValueError, match="更新"):
        await controller.prepare_session()


@pytest.mark.asyncio
async def test_placeholder_is_blocked_and_classroom_is_explicit(tmp_path):
    controller = workspace(tmp_path)
    controller.window.selected_scenario_id = "__defense__"
    with pytest.raises(ValueError):
        await controller.prepare_session()
    controller.window.selected_scenario_id = "__classroom__"
    await controller.prepare_session()
    assert controller.runtime.settings.subject == "历史"
    assert controller.runtime.settings.audio_source == "microphone"


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_delayed_audio_enumeration_cannot_overwrite_new_scenario(
    tmp_path, monkeypatch, fail
):
    import asyncio
    import threading

    controller = workspace(tmp_path)
    started = threading.Event()
    release = threading.Event()
    updates = []
    controller.window.set_library_status("场景B索引就绪", ready=True)

    def enumerate_devices():
        started.set()
        release.wait(2)
        if fail:
            raise OSError("old endpoint failed")
        return ()

    monkeypatch.setattr("classbot.workspace.list_loopback_devices", enumerate_devices)
    controller.window.set_audio_devices = lambda *args: updates.append(args)
    task = asyncio.create_task(controller.refresh_devices())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        controller._generation += 1
        controller.window.audio_source = "microphone"
        release.set()
        await task
        assert updates == []
        assert controller.window.library == ("场景B索引就绪", True)
    finally:
        release.set()
