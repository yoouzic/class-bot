"""The real Qt entry point releases idle indexes and retains the current one."""

import os
import time
import weakref
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from classbot.config import Settings
from classbot.knowledge import KnowledgeIndex
from classbot.scenarios import ScenarioProfile, ScenarioStore
from classbot.ui.main_window import MainWindow
from classbot.workspace import WorkspaceController


class Encoder:
    def encode(self, texts, *, query=False):
        return np.tile(np.array([0.6, 0.8], dtype=np.float32), (len(texts), 1))


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_ready(qapp, dialog):
    deadline = time.monotonic() + 5
    while dialog.busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert not dialog.busy
    qapp.processEvents()


def test_cache_button_releases_idle_index_and_clears_only_its_generated_files(qapp, tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    profiles, snapshots = [], []
    for name in ("current", "unused"):
        source = tmp_path / name
        source.mkdir()
        (source / "original.txt").write_text(f"original {name}", encoding="utf-8")
        profiles.append(ScenarioProfile(name, name, "competition", str(source), "full"))
        snapshots.append(KnowledgeIndex.build(source, cache, embedder=Encoder(), mode="full"))
    ScenarioStore(cache / "scenarios.json").save(profiles)
    settings = replace(Settings(None, "test", "deepseek-flash", "https://example", "cpu"), cache_dir=cache)
    # A stopped runtime's dependency factories can also retain an idle index.
    runtime = SimpleNamespace(current_runtime=None, settings=settings,
                              dependencies=SimpleNamespace(factory=lambda index=snapshots[1]: index))
    window = MainWindow("test")
    controller = WorkspaceController(settings, window, runtime)
    controller.initialize()
    controller.indices.update(zip((profile.scenario_id for profile in profiles), snapshots))
    controller.reports.update(controller.indices)
    current, unused = snapshots
    unused_key = unused.cache_key
    idle_ref = weakref.ref(unused)
    del snapshots, unused
    try:
        window.cache_management_button.click()
        dialog = controller._cache_dialog
        assert dialog is not None
        wait_ready(qapp, dialog)
        assert idle_ref() is None
        rows = {row.key: row for row in dialog._rows}
        assert rows[current.cache_key].protected
        assert not rows[unused_key].protected
        row_number = next(i for i, row in enumerate(dialog._rows) if row.key == unused_key)
        dialog.table.selectRow(row_number)
        monkeypatch.setattr(QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes)
        dialog.clear_button.click()
        wait_ready(qapp, dialog)
        assert not list(cache.glob(unused_key + ".*"))
        assert current.search("original current")
        assert controller.indices == {"current": current}
        assert (tmp_path / "current" / "original.txt").read_text(encoding="utf-8") == "original current"
        assert (tmp_path / "unused" / "original.txt").read_text(encoding="utf-8") == "original unused"
        dialog.close()
        qapp.processEvents()
        assert controller._cache_dialog is None
    finally:
        controller.close()
        window.close()


def test_cache_manager_does_not_open_during_a_live_session(qapp, tmp_path):
    settings = replace(Settings(None, "test", "deepseek-flash", "https://example", "cpu"), cache_dir=tmp_path)
    window = MainWindow("test")
    runtime = SimpleNamespace(current_runtime=object(), settings=settings)
    controller = WorkspaceController(settings, window, runtime)
    controller.show_cache_manager()
    assert controller._cache_dialog is None
    window.close()
