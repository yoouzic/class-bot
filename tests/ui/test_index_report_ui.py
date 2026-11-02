import os
from dataclasses import replace
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit

from classbot.config import Settings
from classbot.scenarios import ScenarioProfile
from classbot.ui.main_window import MainWindow
from classbot.workspace import WorkspaceController


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def controller(qapp, tmp_path):
    settings = replace(
        Settings(None, "历史", "deepseek-flash", "https://example", "cpu"),
        cache_dir=tmp_path / "cache",
        startup_mode="defense",
    )
    window = MainWindow("历史")
    runtime = SimpleNamespace(current_runtime=None, settings=settings)
    instance = WorkspaceController(settings, window, runtime)
    instance.profiles = (
        ScenarioProfile("demo", "国赛", "competition", str(tmp_path)),
    )
    window.set_scenarios(instance.profiles, "demo")
    yield instance
    instance.close()
    window.close()


def index_snapshot(tmp_path, *, chunks=12, limited=2):
    report_path = tmp_path / "index-report.txt"
    report_path.write_text("persistent report", encoding="utf-8")
    details = "\n".join(
        [
            f"资料夹：{tmp_path}",
            "快照：2026-10-15T08:00:00+00:00",
            "已索引文件：4；片段：12；跳过：3；受限：2；重复：1",
            "<b>这段报告应保持纯文本</b>",
        ]
        + [f"诊断 {number}" for number in range(10)]
    )
    return SimpleNamespace(
        root=tmp_path,
        built_at="2026-10-15T08:00:00+00:00",
        report=SimpleNamespace(
            indexed_files=4,
            chunks=chunks,
            skipped_files=3,
            limited_files=limited,
            duplicate_files=1,
            issues=tuple(f"诊断 {number}" for number in range(10)),
        ),
        report_path=report_path,
        report_text=lambda: details,
    )


def test_report_button_requires_profile_and_keeps_readonly_access_while_listening(
    controller,
):
    window = controller.window
    assert not window.index_report_button.isEnabled()
    window.set_library_status("可用", ready=True)
    assert window.index_report_button.isEnabled()
    requested = []
    window.index_report_requested.connect(lambda: requested.append(True))
    window.set_status("listening")
    window.index_report_button.click()
    assert requested == [True]

    window.set_status("stopped")
    window.set_scenarios([], "__defense__")
    window.set_library_status("旧场景回调", ready=True)
    assert not window.index_report_button.isEnabled()


def test_partial_status_distinguishes_limits_duplicates_and_bounds_diagnostics(
    controller, tmp_path,
):
    index = index_snapshot(tmp_path)
    controller.indices["demo"] = index
    controller._show_index_status()

    text = controller.window.library_status_label.text()
    assert "部分资料已索引" in text
    assert "受限 2" in text
    assert "跳过 3" in text
    assert "重复 1" in text
    assert "诊断 2" in text
    assert "诊断 3" not in text
    assert "另有 7 条诊断" in text
    assert str(index.report_path) in text
    assert controller.window.toggle_button.isEnabled()
    assert controller.window.index_report_button.isEnabled()


@pytest.mark.parametrize("latest_report", [False, True])
def test_report_dialog_shows_complete_plain_selectable_readonly_text(
    controller, tmp_path, monkeypatch, latest_report,
):
    index = index_snapshot(tmp_path)
    controller.indices["demo"] = index
    if latest_report:
        controller.reports["demo"] = index
        controller.indices["demo"] = SimpleNamespace(
            report_text=lambda: "older usable index report"
        )
    observed = []

    def inspect_dialog(dialog):
        editor = dialog.findChild(QPlainTextEdit, "indexReportText")
        assert editor is not None
        observed.append((
            dialog.windowTitle(),
            editor.toPlainText(),
            editor.isReadOnly(),
            editor.textInteractionFlags(),
        ))
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(QDialog, "exec", inspect_dialog)
    controller.show_index_report()

    assert len(observed) == 1
    title, text, readonly, flags = observed[0]
    assert "索引详情" in title
    assert text == index.report_text()
    assert "诊断 9" in text
    assert "<b>" in text
    assert readonly
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse


def test_initialize_connects_report_signal(controller, monkeypatch):
    observed = []
    monkeypatch.setattr(controller.store, "load", lambda: controller.profiles)
    monkeypatch.setattr(controller, "show_index_report", lambda: observed.append(True))
    controller.initialize()
    controller.window.index_report_requested.emit()
    assert observed == [True]


def test_long_report_preview_keeps_window_and_answer_area_bounded(
    controller, tmp_path, qapp,
):
    index = index_snapshot(tmp_path)
    index.report.issues = ("very-long-unbroken-diagnostic-" * 100,) * 10
    controller.indices["demo"] = index
    controller._show_index_status()
    controller.window.show()
    qapp.processEvents()

    assert controller.window.width() == 560
    assert controller.window.height() == 750
    assert controller.window.cards_list.height() >= 180


def test_failed_report_stays_accessible_after_reselect_without_ready_index(
    controller, tmp_path,
):
    latest = index_snapshot(tmp_path, chunks=0)
    controller.reports["demo"] = latest
    controller.window.set_scenarios(controller.profiles, "demo")
    controller._show_index_status()

    assert controller.window.index_report_button.isEnabled()
    assert not controller.window.toggle_button.isEnabled()
    assert str(latest.report_path) in controller.window.library_status_label.text()


def build_inline(controller, monkeypatch):
    async def immediate_thread(function, *args, **kwargs):
        return function(*args, **kwargs)

    monkeypatch.setattr("classbot.workspace.asyncio.to_thread", immediate_thread)
    monkeypatch.setattr(
        "classbot.workspace.asyncio.get_running_loop", lambda: SimpleNamespace(),
    )
    operation = controller.build_index()
    with pytest.raises(StopIteration):
        operation.send(None)


def test_zero_chunk_rebuild_keeps_usable_index_and_exposes_latest_report(
    controller, tmp_path, monkeypatch,
):
    previous = index_snapshot(tmp_path, limited=0)
    latest = index_snapshot(tmp_path, chunks=0, limited=4)
    controller.indices["demo"] = previous
    monkeypatch.setattr(
        "classbot.workspace.KnowledgeIndex.build", lambda *args, **kwargs: latest,
    )

    build_inline(controller, monkeypatch)

    assert controller.indices["demo"] is previous
    assert controller.reports["demo"] is latest
    assert "索引失败" in controller.window.library_status_label.text()
    assert "已保留上次索引快照" in controller.window.library_status_label.text()
    assert controller.window.index_report_button.isEnabled()
    assert controller.window.toggle_button.isEnabled()


def test_zero_chunk_first_build_report_is_available_without_allowing_start(
    controller, tmp_path, monkeypatch,
):
    latest = index_snapshot(tmp_path, chunks=0, limited=4)
    monkeypatch.setattr(
        "classbot.workspace.KnowledgeIndex.build", lambda *args, **kwargs: latest,
    )
    build_inline(controller, monkeypatch)

    assert "demo" not in controller.indices
    assert controller.reports["demo"] is latest
    assert controller.window.index_report_button.isEnabled()
    assert not controller.window.toggle_button.isEnabled()
