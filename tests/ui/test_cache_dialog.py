"""Cache management is explicit, protected, and does not block Qt work."""

import os
import threading
import time
from dataclasses import dataclass, replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from classbot.ui.cache_dialog import CacheDialog, format_cache_bytes
from classbot.ui.main_window import MainWindow


@dataclass(frozen=True)
class Library:
    key: str
    root: str
    mode: str = "full"
    model: str = "local-model"
    size_bytes: int = 1024**3
    file_count: int = 4_934
    chunk_count: int = 280_788
    built_at: str = "2026-10-16T10:00:00"
    protected: bool = False
    status: str = "ready"


@dataclass(frozen=True)
class Inventory:
    libraries: tuple
    total_bytes: int = 3 * 1024**3
    limit_bytes: int = 10 * 1024**3
    shared_bytes: int = 1024**3
    model_bytes: int = 92 * 1024**2
    other_bytes: int = 24


class Manager:
    def __init__(self, libraries=()) -> None:
        self.snapshot = Inventory(tuple(libraries))
        self.calls = []
        self.inventory_error = ""
        self.inventory_thread = None

    def inventory(self):
        self.inventory_thread = threading.get_ident()
        if self.inventory_error:
            raise RuntimeError(self.inventory_error)
        return self.snapshot

    def clear_library(self, key, *, protected_keys=()):
        self.calls.append(("clear_library", key, protected_keys))
        library = next(item for item in self.snapshot.libraries if item.key == key)
        self.snapshot = replace(
            self.snapshot,
            libraries=tuple(item for item in self.snapshot.libraries if item.key != key),
            total_bytes=self.snapshot.total_bytes - library.size_bytes,
        )
        return library.size_bytes

    def clear_shared_vectors(self):
        self.calls.append(("clear_shared",))
        size = self.snapshot.shared_bytes
        self.snapshot = replace(
            self.snapshot,
            shared_bytes=0,
            total_bytes=self.snapshot.total_bytes - size,
        )
        return size

    def set_limit(self, limit):
        self.calls.append(("set_limit", limit))
        if limit < self.snapshot.total_bytes:
            raise RuntimeError("上限低于当前占用，请先清理缓存或提高上限")
        self.snapshot = replace(self.snapshot, limit_bytes=limit)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    created = app is None
    if created:
        app = QApplication([])
    yield app
    if created:
        app.quit()


def wait_ready(qapp, dialog):
    deadline = time.monotonic() + 5
    while dialog.busy and time.monotonic() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    assert not dialog.busy, "cache worker did not finish"
    qapp.processEvents()


@pytest.fixture
def make_dialog(qapp, tmp_path):
    dialogs = []

    def make(*, libraries=None, **kwargs):
        manager = Manager(
            libraries if libraries is not None else [Library("first", str(tmp_path))]
        )
        dialog = CacheDialog(manager, **kwargs)
        dialogs.append(dialog)
        wait_ready(qapp, dialog)
        return dialog, manager

    yield make
    for dialog in dialogs:
        wait_ready(qapp, dialog)
        dialog.close()


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, "0 B"), (1024, "1.0 KiB"), (1024**2, "1.0 MiB"), (1024**3, "1.00 GiB")],
)
def test_format_cache_bytes(value, expected):
    assert format_cache_bytes(value) == expected


def test_inventory_shows_each_library_and_all_shared_categories(make_dialog, tmp_path):
    profile = SimpleNamespace(name="组会 <b>资料</b>", library_path=tmp_path)
    dialog, manager = make_dialog(profiles=[profile])
    assert dialog.table.rowCount() == 1
    assert "组会 <b>资料</b>" in dialog.table.item(0, 0).text()
    assert dialog.table.item(0, 1).text() == "完整"
    assert dialog.table.item(0, 2).text() == "1.00 GiB"
    assert dialog.table.item(0, 3).text() == "4,934 / 280,788"
    assert dialog.table.item(0, 4).text() == "已保存"
    assert "3.00 GiB" in dialog.usage_label.text()
    assert "共享向量 1.00 GiB" in dialog.categories_label.text()
    assert "共享模型 92.0 MiB" in dialog.categories_label.text()
    assert "其他缓存 24 B" in dialog.categories_label.text()
    assert dialog.usage_label.textFormat() == Qt.TextFormat.PlainText
    assert manager.calls == []
    assert manager.inventory_thread != threading.get_ident()


def test_delete_requires_confirmation_and_clears_only_selected_key(
    make_dialog, qapp, monkeypatch, tmp_path
):
    libraries = [Library("first", str(tmp_path)), Library("second", str(tmp_path / "other"))]
    dialog, manager = make_dialog(libraries=libraries)
    changes = []
    dialog.cache_changed.connect(changes.append)
    dialog.table.selectRow(1)
    assert dialog.clear_button.isEnabled()
    prompts = []

    def answer_no(*args):
        prompts.append(args)
        return QMessageBox.StandardButton.No

    monkeypatch.setattr(QMessageBox, "question", answer_no)
    dialog.clear_button.click()
    assert manager.calls == []
    assert "原始资料不会修改" in prompts[0][2]
    assert "重新建立完整索引" in prompts[0][2]
    assert prompts[0][-1] == QMessageBox.StandardButton.No
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    dialog.clear_button.click()
    wait_ready(qapp, dialog)
    assert manager.calls == [("clear_library", "second", frozenset())]
    assert [library.key for library in manager.snapshot.libraries] == ["first"]
    assert dialog.table.rowCount() == 1
    assert changes == [
        {"operation": "clear_library", "key": "second", "freed_bytes": 1024**3}
    ]
    assert "原始资料未修改" in dialog.status_label.text()


@pytest.mark.parametrize(
    ("protected", "status", "protected_keys"),
    [(True, "ready", ()), (False, "unsafe", ()), (False, "ready", ("first",))],
)
def test_in_use_or_unsafe_rows_cannot_be_deleted(
    make_dialog, tmp_path, protected, status, protected_keys
):
    dialog, manager = make_dialog(
        libraries=[Library("first", str(tmp_path), protected=protected, status=status)],
        protected_keys=protected_keys,
    )
    dialog.table.selectRow(0)
    assert not dialog.clear_button.isEnabled()
    dialog._clear_selected()
    assert manager.calls == []


def test_shared_clear_explains_existing_indexes_remain_and_uses_separate_action(
    make_dialog, qapp, monkeypatch
):
    dialog, manager = make_dialog()
    prompts = []

    def confirm(*args):
        prompts.append(args)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "question", confirm)
    dialog.clear_shared_button.click()
    wait_ready(qapp, dialog)
    assert "现有完整索引仍可使用" in prompts[0][2]
    assert manager.calls == [("clear_shared",)]
    assert len(manager.snapshot.libraries) == 1
    assert not dialog.clear_shared_button.isEnabled()


def test_limit_saved_or_rejected_without_deleting_indices(make_dialog, qapp):
    dialog, manager = make_dialog()
    dialog.limit_spin.setValue(12.5)
    dialog.save_limit_button.click()
    wait_ready(qapp, dialog)
    assert manager.calls == [("set_limit", round(12.5 * 1024**3))]
    assert "上限已保存" in dialog.status_label.text()
    dialog.limit_spin.setValue(1)
    dialog.save_limit_button.click()
    wait_ready(qapp, dialog)
    assert "操作未完成" in dialog.status_label.text()
    assert "请先清理缓存或提高上限" in dialog.status_label.text()
    assert manager.snapshot.limit_bytes == round(12.5 * 1024**3)
    assert len(manager.snapshot.libraries) == 1
    assert all(call[0] == "set_limit" for call in manager.calls)


def test_refresh_preserves_key_selection_and_reports_over_limit(make_dialog, qapp):
    dialog, manager = make_dialog()
    dialog.table.selectRow(0)
    manager.snapshot = replace(manager.snapshot, total_bytes=12 * 1024**3)
    dialog.refresh_button.click()
    wait_ready(qapp, dialog)
    assert dialog.table.currentRow() == 0
    assert dialog.clear_button.isEnabled()
    assert "已超出上限" in dialog.usage_label.text()


def test_worker_keeps_qt_responsive_and_blocks_close_during_mutation(
    make_dialog, qapp, monkeypatch
):
    dialog, manager = make_dialog()
    entered = threading.Event()
    release = threading.Event()

    def set_limit(limit):
        entered.set()
        assert release.wait(5)
        manager.snapshot = replace(manager.snapshot, limit_bytes=limit)

    monkeypatch.setattr(manager, "set_limit", set_limit)
    dialog.show()
    dialog.limit_spin.setValue(12)
    dialog.save_limit_button.click()
    try:
        assert entered.wait(2)
        qapp.processEvents()
        assert dialog.busy
        assert not dialog.clear_button.isEnabled()
        assert not dialog.refresh_button.isEnabled()
        dialog.reject()
        dialog.close()
        assert dialog.isVisible()
    finally:
        release.set()
        wait_ready(qapp, dialog)


def test_mutation_success_still_notifies_if_refresh_fails(make_dialog, qapp, monkeypatch):
    dialog, manager = make_dialog()
    changes = []
    dialog.cache_changed.connect(changes.append)

    def set_limit(limit):
        manager.inventory_error = "统计文件暂不可读"

    monkeypatch.setattr(manager, "set_limit", set_limit)
    dialog.save_limit_button.click()
    wait_ready(qapp, dialog)
    assert changes == [{"operation": "set_limit", "key": None, "freed_bytes": 0}]
    assert "上限已保存" in dialog.status_label.text()
    assert "最新统计读取失败" in dialog.status_label.text()
    assert not dialog.clear_button.isEnabled()
    assert not dialog.save_limit_button.isEnabled()


def test_failed_cleanup_refreshes_actual_partial_progress_without_success_event(
    make_dialog, qapp, monkeypatch
):
    dialog, manager = make_dialog()
    changes = []
    dialog.cache_changed.connect(changes.append)
    dialog.table.selectRow(0)

    def partially_clear(_key, *, protected_keys=()):
        manager.snapshot = replace(
            manager.snapshot,
            total_bytes=manager.snapshot.total_bytes - 512 * 1024**2,
            libraries=(replace(manager.snapshot.libraries[0], size_bytes=512 * 1024**2),),
        )
        raise PermissionError("后续缓存文件正被其他程序占用")

    monkeypatch.setattr(manager, "clear_library", partially_clear)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    dialog.clear_button.click()
    wait_ready(qapp, dialog)
    assert "2.50 GiB" in dialog.usage_label.text()
    assert dialog.table.item(0, 2).text() == "512.0 MiB"
    assert dialog.table.item(0, 4).text() == "已保存"
    assert "后续缓存文件正被其他程序占用" in dialog.status_label.text()
    assert "清理可能已移除部分缓存；最新占用已刷新" in dialog.status_label.text()
    assert changes == []


def test_failed_mutation_attempt_refreshes_rows_even_when_error_has_no_message(
    make_dialog, qapp, monkeypatch
):
    dialog, manager = make_dialog()
    changes = []
    dialog.cache_changed.connect(changes.append)

    def reject_limit(_limit):
        manager.snapshot = replace(manager.snapshot, libraries=(), total_bytes=1024**3)
        raise RuntimeError()

    monkeypatch.setattr(manager, "set_limit", reject_limit)
    dialog.save_limit_button.click()
    wait_ready(qapp, dialog)
    assert dialog.table.rowCount() == 0
    assert "1.00 GiB" in dialog.usage_label.text()
    assert "操作未完成：RuntimeError" in dialog.status_label.text()
    assert "最新占用已刷新" in dialog.status_label.text()
    assert changes == []


def test_failed_cleanup_and_failed_refresh_keep_error_truthful_and_disable_mutations(
    make_dialog, qapp, monkeypatch
):
    dialog, manager = make_dialog()
    changes = []
    dialog.cache_changed.connect(changes.append)
    dialog.table.selectRow(0)

    def partially_clear(_key, *, protected_keys=()):
        manager.inventory_error = "最新统计暂不可读"
        raise PermissionError("无法移除后续文件")

    monkeypatch.setattr(manager, "clear_library", partially_clear)
    monkeypatch.setattr(
        QMessageBox, "question", lambda *args: QMessageBox.StandardButton.Yes
    )
    dialog.clear_button.click()
    wait_ready(qapp, dialog)
    message = dialog.status_label.text()
    assert "清理可能已移除部分缓存；最新占用未能刷新" in message
    assert "最新统计暂不可读" in message
    assert "已清理" not in message
    assert not dialog.clear_button.isEnabled()
    assert not dialog.clear_shared_button.isEnabled()
    assert not dialog.save_limit_button.isEnabled()
    assert dialog.refresh_button.isEnabled()
    assert changes == []


def test_main_window_cache_management_signal_and_busy_guards(qapp):
    window = MainWindow("测试")
    try:
        requests = []
        window.cache_management_requested.connect(lambda: requests.append(True))
        assert window.cache_management_button.isEnabled()
        window.cache_management_button.click()
        assert requests == [True]
        window.set_configuration_busy(True)
        assert not window.cache_management_button.isEnabled()
        window.set_configuration_busy(False)
        assert window.cache_management_button.isEnabled()
        window.set_status("listening")
        assert not window.cache_management_button.isEnabled()
        window.set_status("stopped")
        assert window.cache_management_button.isEnabled()
    finally:
        window.close()
