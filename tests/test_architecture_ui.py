import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox, QDialog

from classbot.domain import AnswerCard, CardStatus
from classbot.scenarios import ScenarioProfile
from classbot.ui.main_window import MainWindow
from classbot.ui.scenario_dialog import ScenarioDialog


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


def test_new_scenario_offers_generic_full_and_balanced_modes(qapp, tmp_path):
    dialog = ScenarioDialog()
    try:
        mode = dialog.findChild(QComboBox, "indexModeCombo")
        assert mode is not None
        assert {mode.itemData(i) for i in range(mode.count())} == {"full", "balanced"}
        assert mode.currentData() == "full"
        dialog.name_edit.setText("任何项目的答辩")
        dialog.library_path_edit.setText(str(tmp_path))
        assert dialog.values().get("index_mode") == "full"
        mode.setCurrentIndex(mode.findData("balanced"))
        assert dialog.values().get("index_mode") == "balanced"
    finally:
        dialog.close()


@pytest.mark.parametrize("mode", ["full", "balanced"])
def test_editing_scenario_preserves_existing_index_mode(qapp, tmp_path, mode):
    profile = ScenarioProfile("demo", "场景", "defense", str(tmp_path), mode)
    dialog = ScenarioDialog(profile=profile)
    try:
        assert dialog.values().get("index_mode") == mode
    finally:
        dialog.close()


@pytest.mark.parametrize("name", ["名" * 121, "invalid\tname"])
def test_dialog_rejects_schema_invalid_name_without_losing_edit(qapp, tmp_path, name):
    dialog = ScenarioDialog()
    try:
        dialog.show()
        dialog.library_path_edit.setText(str(tmp_path))
        dialog.name_edit.setText(name)

        assert not dialog.save_button.isEnabled()
        assert dialog.validation_label.text()
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted
        assert dialog.isVisible()
        assert dialog.name_edit.text() == name
    finally:
        dialog.close()


def test_dialog_validates_kind_against_profile_schema(qapp, tmp_path):
    dialog = ScenarioDialog()
    try:
        dialog.name_edit.setText("场景")
        dialog.library_path_edit.setText(str(tmp_path))
        dialog.kind_combo.setCurrentIndex(-1)

        assert not dialog.save_button.isEnabled()
        dialog.accept()
        assert dialog.result() != QDialog.DialogCode.Accepted
    finally:
        dialog.close()


def test_scene_switch_clears_live_header_and_retains_tagged_history(qapp, tmp_path):
    window = MainWindow("课堂")
    try:
        profiles = [
            ScenarioProfile("a", "场景甲", "defense", str(tmp_path)),
            ScenarioProfile("b", "场景乙", "defense", str(tmp_path)),
        ]
        window.set_scenarios(profiles, "a")
        window.set_transcript("甲的转写")
        window.upsert_card(AnswerCard(
            "q-a", "甲的问题", "场景甲", answer="甲的回答",
            status=CardStatus.ANSWERED, scenario_name="场景甲",
            sources=("[S1] 甲.txt · lines 1–2",),
        ))

        window.scenario_combo.setCurrentIndex(window.scenario_combo.findData("b"))

        assert window.transcript_label.text() == "等待转写…"
        assert window.current_question_label.text() == "当前问题：等待识别…"
        assert window._current_question_id is None
        assert window.cards_list.count() == 1
        history = window.cards_list.item(0).text()
        assert "场景甲" in history
        assert "甲的回答" in history
    finally:
        window.close()
