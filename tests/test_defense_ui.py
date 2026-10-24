import os
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel

from classbot.domain import AnswerCard, CardStatus
from classbot.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(qapp):
    instance = MainWindow("历史")
    yield instance
    instance.close()


def profiles():
    return [
        SimpleNamespace(
            scenario_id="project",
            name="项目答辩",
            kind="competition",
            library_path=Path("D:/材料/项目"),
        ),
        SimpleNamespace(
            scenario_id="interview",
            name="保研面试",
            kind="group_meeting",
            library_path=Path("D:/材料/面试"),
        ),
    ]


def test_scenario_population_preserves_classroom_and_selects_defense_silently(window):
    changed = []
    window.scenario_changed.connect(changed.append)

    window.set_scenarios(profiles(), "project")

    assert window.scenario_combo.findData("__classroom__") >= 0
    assert window.selected_scenario_id == "project"
    assert window.audio_source == "loopback"
    assert not window.audio_source_combo.isEnabled()
    assert str(profiles()[0].library_path) in window.library_path_label.text()
    assert not window.toggle_button.isEnabled()
    assert changed == []


def test_index_readiness_gates_defense_but_classroom_remains_available(window):
    window.set_scenarios(profiles(), "project")
    window.set_library_status("已索引 12 份材料", ready=True)
    assert window.toggle_button.isEnabled()

    changed = []
    window.scenario_changed.connect(changed.append)
    window.scenario_combo.setCurrentIndex(window.scenario_combo.findData("interview"))
    assert changed == ["interview"]
    assert not window.toggle_button.isEnabled()
    assert str(profiles()[1].library_path) in window.library_path_label.text()

    window.scenario_combo.setCurrentIndex(
        window.scenario_combo.findData("__classroom__")
    )
    assert window.audio_source == "microphone"
    assert window.toggle_button.isEnabled()
    assert not window.index_button.isEnabled()
    assert not window.edit_scenario_button.isEnabled()


def test_no_profile_defense_placeholder_never_starts_microphone(window):
    window.set_scenarios([], "__defense__")

    assert window.selected_scenario_id == "__defense__"
    assert window.audio_source == "loopback"
    assert window.add_scenario_button.isEnabled()
    assert not window.toggle_button.isEnabled()
    assert not window.edit_scenario_button.isEnabled()
    assert not window.index_button.isEnabled()
    assert "创建场景" in window.library_status_label.text()
    window.set_library_status("旧场景的就绪回调", ready=True)
    assert not window.toggle_button.isEnabled()


def test_missing_saved_profile_falls_back_to_defense_without_microphone(window):
    window.set_scenarios([], "deleted-profile")
    assert window.selected_scenario_id == "__defense__"
    assert window.audio_source == "loopback"


def test_scenario_actions_emit_without_starting_background_work(window):
    window.set_scenarios(profiles(), "project")
    actions = []
    window.scenario_add_requested.connect(lambda: actions.append("add"))
    window.scenario_edit_requested.connect(lambda: actions.append("edit"))
    window.index_requested.connect(lambda: actions.append("index"))

    window.add_scenario_button.click()
    window.edit_scenario_button.click()
    window.index_button.click()

    assert actions == ["add", "edit", "index"]


def test_audio_device_selection_has_default_and_preserves_explicit_choice(window):
    events = []
    window.audio_device_changed.connect(events.append)
    devices = [
        SimpleNamespace(index=2, name="扬声器", is_default=True),
        SimpleNamespace(index=8, name="耳机", is_default=False),
    ]
    window.set_scenarios(profiles(), "project")
    window.set_audio_devices(devices, selected_device=8)

    assert window.audio_device == 8
    assert window.audio_device_combo.itemData(0) is None
    assert "系统默认播放设备" in window.audio_device_combo.itemText(0)
    assert events == []

    window.audio_device_combo.setCurrentIndex(0)
    assert events == [None]
    assert window.audio_device is None


def test_configuration_lock_covers_busy_pending_and_active_listener(window):
    window.set_scenarios(profiles(), "project")
    window.set_library_status("就绪", ready=True)
    controls = [
        window.scenario_combo,
        window.add_scenario_button,
        window.edit_scenario_button,
        window.index_button,
        window.audio_device_combo,
    ]

    window.set_configuration_busy(True)
    assert all(not control.isEnabled() for control in controls)
    assert not window.toggle_button.isEnabled()
    window.set_configuration_busy(False)
    assert all(control.isEnabled() for control in controls)

    window.toggle_button.click()
    assert all(not control.isEnabled() for control in controls)
    window.set_status("listening")
    assert all(not control.isEnabled() for control in controls)
    assert window.toggle_button.isEnabled()

    window.set_configuration_busy(True)
    assert window.toggle_button.isEnabled(), (
        "pause must remain available while listening"
    )
    window.toggle_button.click()
    assert not window.toggle_button.isEnabled()
    window.set_status("stopped")
    assert not window.toggle_button.isEnabled()
    window.set_configuration_busy(False)
    assert window.toggle_button.isEnabled()


@pytest.mark.parametrize(
    "status", ["audio_error", "indexing", "ready", "loading", "index_error"]
)
def test_defense_statuses_have_readable_labels(window, status):
    window.set_status(status)
    assert window.status_label.text() != status


@dataclass
class DefenseCard(AnswerCard):
    scenario_name: str = ""
    sources: tuple[str, ...] = ()


def test_answer_sources_and_model_text_are_plain_and_selectable(window):
    card = DefenseCard(
        "q1",
        "<b>问题</b>",
        "项目",
        answer="<img src=x>材料事实：暂无支持",
        status=CardStatus.ANSWERED,
        scenario_name="<b>项目</b>",
        sources=("[S1] 项目说明.pdf · PDF第3页", "<b>路径.txt</b> · 第2段"),
    )
    window.upsert_card(card)
    widget = window.cards_list.itemWidget(window.cards_list.item(0))

    for name in [
        "questionTextLabel",
        "answerTextLabel",
        "sourcesTextLabel",
        "scenarioNameLabel",
    ]:
        label = widget.findChild(QLabel, name)
        assert label.textFormat() == Qt.TextFormat.PlainText
        assert (
            label.textInteractionFlags() & Qt.TextInteractionFlag.TextSelectableByMouse
        )
    assert widget.findChild(QLabel, "scenarioNameLabel").text() == "<b>项目</b>"
    assert "[S1] 项目说明.pdf" in widget.findChild(QLabel, "sourcesTextLabel").text()
    assert "检索候选（不代表直接依据）" in MainWindow.render_card_text(card)


def test_empty_sources_do_not_show_an_empty_references_section(window):
    window.upsert_card(AnswerCard("q1", "问题", "历史"))
    widget = window.cards_list.itemWidget(window.cards_list.item(0))
    assert widget.sources_section.isHidden()


def test_scenario_dialog_requires_name_and_existing_folder(qapp, tmp_path):
    from classbot.ui.scenario_dialog import ScenarioDialog

    dialog = ScenarioDialog()
    assert [
        dialog.kind_combo.itemData(i) for i in range(dialog.kind_combo.count())
    ] == [
        "group_meeting",
        "competition",
        "defense",
    ]
    assert not dialog.save_button.isEnabled()
    dialog.name_edit.setText("数学建模答辩")
    dialog.library_path_edit.setText(str(tmp_path / "missing"))
    assert not dialog.save_button.isEnabled()
    dialog.library_path_edit.setText(str(tmp_path))
    assert dialog.save_button.isEnabled()
    assert dialog.values() == {
        "name": "数学建模答辩",
        "kind": "group_meeting",
        "index_mode": "full",
        "library_path": str(tmp_path),
    }
    dialog.close()


def test_scenario_dialog_prefills_existing_profile(qapp):
    from classbot.ui.scenario_dialog import ScenarioDialog

    dialog = ScenarioDialog(profile=profiles()[1])
    assert dialog.name_edit.text() == "保研面试"
    assert dialog.kind_combo.currentData() == "group_meeting"
    assert dialog.library_path_edit.text() == str(profiles()[1].library_path)
    dialog.close()
