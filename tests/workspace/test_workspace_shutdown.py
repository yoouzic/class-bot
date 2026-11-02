import asyncio
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from classbot.config import Settings
from classbot.knowledge.index import IndexBuildCancelled
from classbot.scenarios import ScenarioProfile
from classbot.workspace import WorkspaceController


@pytest.mark.asyncio
async def test_async_close_drains_cancelled_index_worker_before_loop_exit(tmp_path, monkeypatch):
    entered, completed = threading.Event(), threading.Event()
    settings = replace(Settings(None, "测试", "fake", "https://example", "cpu"), cache_dir=tmp_path / "cache")
    window = SimpleNamespace(
        selected_scenario_id="demo", set_library_status=lambda *a, **kw: None,
        set_configuration_busy=lambda busy: None,
    )
    controller = WorkspaceController(settings, window, SimpleNamespace(current_runtime=None))
    controller.profiles = (ScenarioProfile("demo", "测试", "defense", str(tmp_path), "full"),)
    monkeypatch.setattr(controller, "_shared_embedder", lambda *_: None)

    def build(*args, cancelled, **kwargs):
        entered.set()
        while not cancelled():
            completed.wait(0.01)
        completed.set()
        raise IndexBuildCancelled("取消")

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    controller._schedule(controller.build_index())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        await asyncio.wait_for(controller.aclose(), 2)
        assert completed.is_set()
        assert not controller._busy
        assert not controller._tasks
    finally:
        controller.close()
