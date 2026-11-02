"""No-model resource, cancellation and UI lifecycle contracts."""
import asyncio
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from classbot.config import Settings
from classbot.knowledge.embedding import LocalEmbedder
from classbot.knowledge.index import IndexBuildCancelled
from classbot.scenarios import ScenarioProfile
from classbot.workspace import WorkspaceController


class Window:
    selected_scenario_id = "a"
    audio_device = 7
    audio_source = "loopback"

    def __init__(self):
        self.messages = []

    def set_library_status(self, text, ready=False):
        self.messages.append((text, ready))

    def set_configuration_busy(self, busy):
        self.busy = busy

    def set_scenarios(self, profiles, selected):
        self.selected_scenario_id = selected


def controller(tmp_path):
    settings = replace(
        Settings(None, "历史", "deepseek-flash", "https://example", "cpu"),
        cache_dir=tmp_path / "cache",
    )
    runtime = SimpleNamespace(current_runtime=None, settings=settings)
    workspace = WorkspaceController(settings, Window(), runtime)
    workspace.profiles = tuple(
        ScenarioProfile(name, name, "competition", str(tmp_path / name), "full")
        for name in "abcd"
    )
    return workspace


def snapshot(root):
    return SimpleNamespace(
        root=Path(root), built_at="now", report_path=None,
        report=SimpleNamespace(indexed_files=1, chunks=1, skipped_files=0, issues=()),
    )


@pytest.mark.asyncio
async def test_build_passes_profile_mode_and_a_cooperative_cancel_signal(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    received = {}

    def build(root, cache, **kwargs):
        received.update(kwargs)
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)

    await workspace.build_index()

    assert received["mode"] == "full"
    assert callable(received["cancelled"])
    assert received["cancelled"]() is False
    assert isinstance(received["embedder"], LocalEmbedder)


@pytest.mark.asyncio
async def test_libraries_share_one_model_and_lru_evicts_indices_and_reports_together(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    embedders = []

    def build(root, cache, **kwargs):
        embedders.append(kwargs.get("embedder"))
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    for name in "abc":
        workspace.window.selected_scenario_id = name
        await workspace.build_index()
    workspace.window.selected_scenario_id = "a"
    workspace._select_scenario("a")
    workspace.window.selected_scenario_id = "d"

    await workspace.build_index()

    assert embedders[0] is not None
    assert all(embedder is embedders[0] for embedder in embedders)
    assert set(workspace.indices) == set(workspace.reports) == {"a", "c", "d"}
    assert len(set(workspace.indices) | set(workspace.reports)) == 3


@pytest.mark.asyncio
async def test_close_signals_worker_and_ignores_late_progress(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    started, exited = threading.Event(), threading.Event()
    received = {}

    def build(root, cache, **kwargs):
        received.update(kwargs)
        started.set()
        if not callable(kwargs.get("cancelled")):
            return snapshot(root)
        while not kwargs["cancelled"]():
            if exited.wait(0.005):
                return snapshot(root)
        kwargs["progress"]("关闭后的进度")
        exited.set()
        raise IndexBuildCancelled("stopped at checkpoint")

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    operation = asyncio.create_task(workspace.build_index())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        workspace.close()
        await asyncio.wait_for(operation, 1)

        assert callable(received.get("cancelled"))
        assert received["cancelled"]() is True
        assert exited.is_set()
        assert workspace.indices == {}
        assert all(text != "关闭后的进度" for text, _ready in workspace.window.messages)
    finally:
        exited.set()
        await operation


@pytest.mark.asyncio
async def test_canceled_build_waits_for_worker_to_stop_at_checkpoint(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    started, canceled, release, exited = (threading.Event() for _ in range(4))

    def build(root, cache, **kwargs):
        started.set()
        if not callable(kwargs.get("cancelled")):
            release.wait(1)
            return snapshot(root)
        while not kwargs["cancelled"]():
            if release.wait(0.005):
                return snapshot(root)
        canceled.set()
        release.wait(1)
        exited.set()
        raise IndexBuildCancelled("checkpoint retained")

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    operation = asyncio.create_task(workspace.build_index())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        operation.cancel()
        assert await asyncio.to_thread(canceled.wait, 0.5)
        assert not operation.done()
        assert workspace._busy is True
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert exited.is_set()
        assert workspace._busy is False
        assert workspace.indices == {}
    finally:
        release.set()
        try:
            await operation
        except asyncio.CancelledError:
            pass


@pytest.mark.asyncio
async def test_late_progress_after_close_is_a_noop(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    received = {}

    def build(root, cache, **kwargs):
        received.update(kwargs)
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    await workspace.build_index()
    workspace.close()
    before = list(workspace.window.messages)

    received["progress"]("stale message")

    assert workspace.window.messages == before


def test_loop_closed_during_progress_does_not_escape_from_worker(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    received = {}

    def closed_loop(*_args):
        raise RuntimeError("Event loop is closed")

    async def immediate_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    def build(root, cache, **kwargs):
        received.update(kwargs)
        kwargs["progress"]("late")
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    monkeypatch.setattr("classbot.workspace.asyncio.to_thread", immediate_thread)
    monkeypatch.setattr("classbot.workspace.asyncio.get_running_loop", lambda: SimpleNamespace(
        call_soon_threadsafe=closed_loop,
    ))
    operation = workspace.build_index()

    with pytest.raises(StopIteration):
        operation.send(None)

    assert received["cancelled"]() is True
    assert workspace.indices == {}


@pytest.mark.parametrize("fail", [False, True])
def test_shared_embedder_prepare_temporarily_binds_build_progress(tmp_path, monkeypatch, fail):
    base, build = [], []
    embedder = LocalEmbedder(cache_dir=tmp_path, progress=base.append)

    def load(**_kwargs):
        embedder._progress("loading")
        if fail:
            raise ValueError("synthetic load failure")

    monkeypatch.setattr(embedder, "_load", load)
    if fail:
        with pytest.raises(ValueError, match="synthetic"):
            embedder.prepare(progress=build.append)
    else:
        embedder.prepare(progress=build.append)
    embedder._progress("restored")

    assert build == ["loading"]
    assert base == ["restored"]


def test_saving_recovered_configuration_surfaces_backup_path(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    workspace = controller(tmp_path)
    backup = tmp_path / "scenarios.invalid-copy.bak"
    dialog = SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted,
        values=lambda: {"name": "edited", "kind": "competition", "library_path": str(tmp_path), "index_mode": "full"},
    )
    monkeypatch.setattr("classbot.workspace.ScenarioDialog", lambda *_args, **_kwargs: dialog)
    monkeypatch.setattr(workspace, "_schedule", lambda awaitable: awaitable.close())

    def save(profiles):
        workspace.store.last_backup_path = backup

    monkeypatch.setattr(workspace.store, "save", save)

    workspace._edit_scenario(True)

    assert any(str(backup) in text for text, _ready in workspace.window.messages)


def test_profile_validation_race_preserves_configuration_and_reports_error(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDialog

    workspace = controller(tmp_path)
    before = workspace.profiles
    warnings = []
    dialog = SimpleNamespace(
        exec=lambda: QDialog.DialogCode.Accepted,
        values=lambda: {"name": "", "kind": "competition", "library_path": str(tmp_path)},
    )
    monkeypatch.setattr("classbot.workspace.ScenarioDialog", lambda *_args, **_kwargs: dialog)
    monkeypatch.setattr("classbot.workspace.QMessageBox.warning", lambda *_args: warnings.append(_args))

    workspace._edit_scenario(True)

    assert workspace.profiles == before
    assert len(warnings) == 1
    assert not workspace.store.path.exists()


@pytest.mark.asyncio
async def test_local_model_file_metadata_change_uses_a_new_embedder(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    model = tmp_path / "local-model"
    model.mkdir()
    weights = model / "model.safetensors"
    weights.write_bytes(b"old weights")
    workspace.settings = replace(workspace.settings, embedding_model=str(model))
    embedders = []

    def build(root, cache, **kwargs):
        embedders.append(kwargs.get("embedder"))
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    await workspace.build_index()
    weights.write_bytes(b"changed weights and size")
    workspace.window.selected_scenario_id = "b"
    await workspace.build_index()

    assert embedders[0] is not None
    assert embedders[1] is not embedders[0]


def test_local_embedder_has_one_shared_retrieval_gate(tmp_path):
    embedder = LocalEmbedder(cache_dir=tmp_path)

    assert isinstance(embedder.retrieval_gate, asyncio.Semaphore)
    assert embedder.retrieval_gate._value == 1


def test_progress_burst_keeps_only_one_pending_callback_and_latest_message(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    callbacks = []

    async def immediate_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    def build(root, cache, **kwargs):
        for number in range(10_000):
            kwargs["progress"](f"扫描 {number}")
        assert len(callbacks) == 1
        callbacks.pop()()
        assert workspace.window.messages[-1][0] == "扫描 9999"
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    monkeypatch.setattr("classbot.workspace.asyncio.to_thread", immediate_thread)
    monkeypatch.setattr("classbot.workspace.asyncio.get_running_loop", lambda: SimpleNamespace(
        call_soon_threadsafe=lambda callback: callbacks.append(callback),
    ))
    operation = workspace.build_index()

    with pytest.raises(StopIteration):
        operation.send(None)

    assert "a" in workspace.indices
    assert callbacks == []


@pytest.mark.asyncio
async def test_selection_change_during_build_preserves_current_index(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    started, release = threading.Event(), threading.Event()

    def build(root, cache, **kwargs):
        if Path(root).name == "d":
            started.set()
            release.wait(1)
        return snapshot(root)

    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", build)
    for name in "abc":
        workspace.window.selected_scenario_id = name
        await workspace.build_index()
    current = workspace.indices["a"]
    workspace.window.selected_scenario_id = "d"
    operation = asyncio.create_task(workspace.build_index())
    try:
        assert await asyncio.to_thread(started.wait, 1)
        workspace.window.selected_scenario_id = "a"
        workspace._select_scenario("a")
        assert workspace.indices["a"] is current
        release.set()
        await operation

        assert workspace.indices["a"] is current
        assert set(workspace.indices) == {"a", "b", "c"}
        assert "d" not in workspace.reports
    finally:
        release.set()
        await operation


@pytest.mark.asyncio
async def test_model_identity_is_inspected_off_the_ui_thread(tmp_path, monkeypatch):
    workspace = controller(tmp_path)
    ui_thread = threading.get_ident()
    checked_on = []

    def identity(model_name):
        checked_on.append(threading.get_ident())
        return {"name": model_name, "revision": "pinned"}

    monkeypatch.setattr("classbot.workspace.model_identity", identity)
    monkeypatch.setattr("classbot.workspace.KnowledgeIndex.build", lambda root, cache, **kwargs: snapshot(root))

    await workspace.build_index()

    assert checked_on
    assert all(worker_thread != ui_thread for worker_thread in checked_on)
