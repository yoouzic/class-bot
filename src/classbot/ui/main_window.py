"""Classroom floating window for displaying live classbot answers."""

from collections.abc import Iterable
from dataclasses import replace
from typing import ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QCloseEvent, QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from classbot.domain import AnswerCard, CardStatus
from classbot.llm.source_tags import cited_source_ids
from classbot.ui import capture_affinity
from classbot.ui.source_rendering import (
    render_answer_html,
    render_answer_plain,
    render_sources_html,
    render_sources_plain,
)


def _format_elapsed_ms(timestamp_ms: int) -> str:
    total_seconds = max(0, timestamp_ms) // 1000
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _answer_text(card: AnswerCard) -> str:
    if card.answer:
        return card.answer
    if card.status == CardStatus.ERROR:
        return "回答失败"
    if card.status == CardStatus.ANSWERED:
        return "未得到有效简答"
    if card.status == CardStatus.CANCELLED:
        return "已取消回答"
    return "等待回答…"


def _status_text(card: AnswerCard) -> str:
    return getattr(card.status, "value", str(card.status))


def _plain_selectable(label: QLabel) -> None:
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)


class _AnswerCardWidget(QFrame):
    """Visual card with separate semantic regions for question and answer."""

    _STYLE = """
    QFrame#answerCard {
        background-color: #2b2d31;
        border: 1px solid #4a4e58;
        border-radius: 10px;
    }
    QLabel#timeLabel {
        color: #aab2bf;
        font-size: 11px;
    }
    QLabel#statusValueLabel {
        color: #b7c0cc;
        font-size: 11px;
    }
    QLabel#scenarioNameLabel {
        color: #cbd5e1;
        font-size: 11px;
    }
    QFrame#questionSection {
        background-color: #25384b;
        border: 1px solid #3a5b78;
        border-left: 4px solid #8ecbff;
        border-radius: 6px;
    }
    QLabel#questionTitleLabel {
        color: #8ecbff;
        font-weight: 600;
    }
    QLabel#questionTextLabel {
        color: #f3f7fb;
    }
    QFrame#answerSection {
        background-color: #253b31;
        border: 1px solid #3b6b50;
        border-left: 4px solid #8be3ae;
        border-radius: 6px;
    }
    QLabel#answerTitleLabel {
        color: #8be3ae;
        font-weight: 600;
    }
    QLabel#answerTextLabel {
        color: #f1fff6;
    }
    QFrame#sourcesSection {
        background-color: #303238;
        border: 1px solid #4a4e58;
        border-radius: 6px;
    }
    QLabel#sourcesTitleLabel {
        color: #cbd5e1;
    }
    QLabel#sourcesTextLabel {
        color: #b7c0cc;
    }
    QLabel#errorLabel {
        color: #ff9d9d;
    }
    QLabel#noticeLabel {
        color: #aab2bf;
        font-size: 11px;
    }
    """

    def __init__(self, card: AnswerCard) -> None:
        super().__init__()
        self.setObjectName("answerCard")
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        self.setStyleSheet(self._STYLE)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 9, 10, 10)
        layout.setSpacing(7)

        header = QHBoxLayout()
        header.setSpacing(8)
        self.time_label = QLabel()
        self.time_label.setObjectName("timeLabel")
        header.addWidget(self.time_label, stretch=1)
        self.scenario_name_label = QLabel()
        self.scenario_name_label.setObjectName("scenarioNameLabel")
        self.scenario_name_label.setWordWrap(True)
        self.scenario_name_label.setMaximumWidth(200)
        _plain_selectable(self.scenario_name_label)
        header.addWidget(self.scenario_name_label)
        self.status_value_label = QLabel()
        self.status_value_label.setObjectName("statusValueLabel")
        header.addWidget(self.status_value_label)
        layout.addLayout(header)

        self.question_section, self.question_text_label = self._make_section(
            "问题",
            "questionSection",
            "questionTitleLabel",
            "questionTextLabel",
        )
        layout.addWidget(self.question_section)

        self.answer_section, self.answer_text_label = self._make_section(
            "回答",
            "answerSection",
            "answerTitleLabel",
            "answerTextLabel",
        )
        layout.addWidget(self.answer_section)

        self.sources_section, self.sources_text_label = self._make_section(
            "检索候选（不代表直接依据）",
            "sourcesSection",
            "sourcesTitleLabel",
            "sourcesTextLabel",
        )
        layout.addWidget(self.sources_section)
        self.sources_title_label = self.sources_section.findChild(
            QLabel, "sourcesTitleLabel"
        )

        self.error_label = QLabel()
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        _plain_selectable(self.error_label)
        layout.addWidget(self.error_label)

        self.notice_label = QLabel()
        self.notice_label.setObjectName("noticeLabel")
        self.notice_label.setWordWrap(True)
        _plain_selectable(self.notice_label)
        layout.addWidget(self.notice_label)

        self.update_card(card)

    def _make_section(
        self,
        title: str,
        section_name: str,
        title_name: str,
        text_name: str,
    ) -> tuple[QFrame, QLabel]:
        section = QFrame(self)
        section.setObjectName(section_name)
        section_layout = QVBoxLayout(section)
        section_layout.setContentsMargins(9, 7, 9, 8)
        section_layout.setSpacing(3)

        title_label = QLabel(title)
        title_label.setObjectName(title_name)
        section_layout.addWidget(title_label)

        text_label = QLabel()
        text_label.setObjectName(text_name)
        text_label.setWordWrap(True)
        _plain_selectable(text_label)
        section_layout.addWidget(text_label)
        return section, text_label

    def update_card(self, card: AnswerCard) -> None:
        self.time_label.setText(f"[{_format_elapsed_ms(card.timestamp_ms)}]")
        self.status_value_label.setText(f"状态：{_status_text(card)}")
        self.question_text_label.setText(card.question)
        scenario_name = getattr(card, "scenario_name", "")
        self.scenario_name_label.setText(scenario_name)
        self.scenario_name_label.setVisible(bool(scenario_name))

        answer_text = _answer_text(card)
        source_refs = tuple(getattr(card, "source_refs", ()))
        if source_refs:
            self.answer_text_label.setTextFormat(Qt.TextFormat.RichText)
            self.answer_text_label.setText(
                render_answer_html(answer_text, source_refs)
            )
        else:
            self.answer_text_label.setTextFormat(Qt.TextFormat.PlainText)
            self.answer_text_label.setText(answer_text)

        sources = tuple(getattr(card, "sources", ()))
        if source_refs:
            cited = cited_source_ids(card.answer)
            self.sources_text_label.setTextFormat(Qt.TextFormat.RichText)
            self.sources_text_label.setText(
                render_sources_html(source_refs, cited)
            )
            self.sources_title_label.setText(
                "来源与候选（绿=本地，橙=网络，灰=通识）"
            )
            self.sources_section.setVisible(True)
        else:
            self.sources_text_label.setTextFormat(Qt.TextFormat.PlainText)
            self.sources_text_label.setText("\n".join(sources))
            self.sources_title_label.setText("检索候选（不代表直接依据）")
            self.sources_section.setVisible(bool(sources))

        notice = getattr(card, "notice", "")
        self.notice_label.setText(notice)
        self.notice_label.setVisible(bool(notice))
        self.error_label.setText(f"错误：{card.error}" if card.error else "")
        self.error_label.setVisible(bool(card.error))
        self.updateGeometry()


class MainWindow(QMainWindow):
    """A small always-on-top window for classroom listening feedback."""

    start_requested = Signal()
    stop_requested = Signal()
    close_requested = Signal()
    scenario_changed = Signal(str)
    scenario_add_requested = Signal()
    scenario_edit_requested = Signal()
    index_requested = Signal()
    index_report_requested = Signal()
    cache_management_requested = Signal()
    audio_source_changed = Signal(str)
    audio_device_changed = Signal(object)

    _STATUS_LABELS: ClassVar[dict[str, str]] = {
        "listening": "● 正在监听",
        "stopped": "已停止",
        "asr_error": "ASR 不可用",
        "mic_error": "麦克风错误",
        "api_error": "API 错误",
        "audio_error": "音频输入不可用",
        "indexing": "正在建立资料索引…",
        "ready": "已就绪",
        "loading": "正在加载…",
        "index_error": "资料索引失败",
    }
    _MAX_CARDS: ClassVar[int] = 128

    def __init__(self, subject: str) -> None:
        super().__init__()
        self._is_listening = False
        self._pending_action: str | None = None
        self._configuration_busy = False
        self._library_ready = False
        self._report_available = False
        self._applied_capture_hidden: bool | None = None
        self._scenario_profiles: dict[str, object] = {}
        self._cards: dict[str, AnswerCard] = {}
        self._items: dict[str, QListWidgetItem] = {}
        self._card_widgets: dict[str, _AnswerCardWidget] = {}
        self._current_question_id: str | None = None

        self.setWindowTitle(f"Classbot · {subject}")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.resize(560, 750)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        scenario_row = QHBoxLayout()
        scenario_row.addWidget(QLabel("场景"))
        self.scenario_combo = QComboBox()
        self.scenario_combo.setObjectName("scenarioCombo")
        self.scenario_combo.addItem("课堂模式（麦克风）", "__classroom__")
        self.scenario_combo.addItem("线上答辩（请创建场景）", "__defense__")
        self.scenario_combo.currentIndexChanged.connect(self._scenario_selected)
        scenario_row.addWidget(self.scenario_combo, stretch=1)
        self.add_scenario_button = QPushButton("新建")
        self.add_scenario_button.setObjectName("addScenarioButton")
        self.add_scenario_button.clicked.connect(self.scenario_add_requested.emit)
        scenario_row.addWidget(self.add_scenario_button)
        self.edit_scenario_button = QPushButton("编辑")
        self.edit_scenario_button.setObjectName("editScenarioButton")
        self.edit_scenario_button.clicked.connect(self.scenario_edit_requested.emit)
        scenario_row.addWidget(self.edit_scenario_button)
        layout.addLayout(scenario_row)

        audio_row = QHBoxLayout()
        audio_row.addWidget(QLabel("音频"))
        self.audio_source_combo = QComboBox()
        self.audio_source_combo.setObjectName("audioSourceCombo")
        self.audio_source_combo.setToolTip(
            "答辩场景使用电脑播放声音；课堂模式使用麦克风。"
        )
        audio_row.addWidget(self.audio_source_combo)
        self.audio_device_combo = QComboBox()
        self.audio_device_combo.setObjectName("audioDeviceCombo")
        self.audio_device_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.audio_device_combo.currentIndexChanged.connect(
            lambda: self.audio_device_changed.emit(self.audio_device)
        )
        audio_row.addWidget(self.audio_device_combo, stretch=1)
        layout.addLayout(audio_row)

        self.library_path_label = QLabel()
        self.library_path_label.setObjectName("libraryPathLabel")
        self.library_path_label.setWordWrap(True)
        self.library_path_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.library_path_label.setMaximumHeight(60)
        _plain_selectable(self.library_path_label)
        layout.addWidget(self.library_path_label)
        self.library_status_label = QLabel()
        self.library_status_label.setObjectName("libraryStatusLabel")
        self.library_status_label.setWordWrap(True)
        self.library_status_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self.library_status_label.setMaximumHeight(180)
        _plain_selectable(self.library_status_label)
        layout.addWidget(self.library_status_label)
        library_row = QHBoxLayout()
        library_row.addStretch()
        self.cache_management_button = QPushButton("缓存管理")
        self.cache_management_button.setObjectName("cacheManagementButton")
        self.cache_management_button.clicked.connect(self.cache_management_requested.emit)
        library_row.addWidget(self.cache_management_button)
        self.index_button = QPushButton("刷新资料索引")
        self.index_button.setObjectName("indexButton")
        self.index_button.clicked.connect(self.index_requested.emit)
        library_row.addWidget(self.index_button)
        self.index_report_button = QPushButton("索引详情")
        self.index_report_button.setObjectName("indexReportButton")
        self.index_report_button.clicked.connect(self.index_report_requested.emit)
        library_row.addWidget(self.index_report_button)
        layout.addLayout(library_row)

        self.cloud_note_label = QLabel(
            "本地检索；问题和相关材料片段会发送给 DeepSeek。"
        )
        self.cloud_note_label.setObjectName("cloudNoteLabel")
        self.cloud_note_label.setWordWrap(True)
        layout.addWidget(self.cloud_note_label)

        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.status_label = QLabel("已停止")
        self.status_label.setObjectName("statusLabel")
        _plain_selectable(self.status_label)
        status_row.addWidget(self.status_label, stretch=1)
        self.capture_hidden_checkbox = QCheckBox("对屏幕共享隐藏")
        self.capture_hidden_checkbox.setObjectName("captureHiddenCheckbox")
        self.capture_hidden_checkbox.setToolTip(
            "勾选后，屏幕共享、录制和截图中不会出现本窗口；"
            "仅 Windows 10 2004（build 19041）及以上系统支持。"
        )
        supported = capture_affinity.is_supported()
        self.capture_hidden_checkbox.setEnabled(supported)
        self.capture_hidden_checkbox.setChecked(supported)
        self.capture_hidden_checkbox.toggled.connect(self._apply_capture_hiding)
        status_row.addWidget(self.capture_hidden_checkbox)
        layout.addLayout(status_row)

        self.transcript_label = QLabel("等待转写…")
        self.transcript_label.setObjectName("transcriptLabel")
        self.transcript_label.setWordWrap(True)
        _plain_selectable(self.transcript_label)
        layout.addWidget(self.transcript_label)

        self.current_question_label = QLabel("当前问题：等待识别…")
        self.current_question_label.setObjectName("currentQuestionLabel")
        self.current_question_label.setWordWrap(True)
        _plain_selectable(self.current_question_label)
        layout.addWidget(self.current_question_label)

        self.cards_list = QListWidget()
        self.cards_list.setObjectName("cardsList")
        self.cards_list.setWordWrap(True)
        self.cards_list.setUniformItemSizes(False)
        self.cards_list.setSpacing(8)
        self.cards_list.setStyleSheet(
            """
            QListWidget#cardsList {
                background-color: #1f2023;
                border: none;
                padding: 4px 2px;
            }
            QListWidget#cardsList::item {
                background: transparent;
                border: none;
                padding: 0px;
            }
            """
        )
        layout.addWidget(self.cards_list, stretch=1)

        self.toggle_button = QPushButton("开始监听")
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.clicked.connect(self._toggle_listening)
        layout.addWidget(self.toggle_button)
        self._sync_scenario()
        self.set_audio_devices([])

    @property
    def selected_scenario_id(self) -> str:
        return self.scenario_combo.currentData() or "__classroom__"

    @property
    def audio_source(self) -> str:
        return (
            "microphone" if self.selected_scenario_id == "__classroom__" else "loopback"
        )

    @property
    def audio_device(self) -> int | None:
        return self.audio_device_combo.currentData()

    def set_scenarios(self, profiles: Iterable[object], selected_id: str) -> None:
        """Populate local profiles without triggering configuration changes."""
        self._scenario_profiles = {
            profile.scenario_id: profile
            for profile in profiles
            if profile.scenario_id not in {"__classroom__", "__defense__"}
        }
        self.scenario_combo.blockSignals(True)
        try:
            self.scenario_combo.clear()
            self.scenario_combo.addItem("课堂模式（麦克风）", "__classroom__")
            self.scenario_combo.addItem("线上答辩（请创建场景）", "__defense__")
            for scenario_id, profile in self._scenario_profiles.items():
                self.scenario_combo.addItem(profile.name, scenario_id)
            index = self.scenario_combo.findData(selected_id)
            if index < 0:
                index = self.scenario_combo.findData("__defense__")
            self.scenario_combo.setCurrentIndex(index)
        finally:
            self.scenario_combo.blockSignals(False)
        self._sync_scenario()

    def _scenario_selected(self) -> None:
        self._sync_scenario()
        self.scenario_changed.emit(self.selected_scenario_id)
        self.audio_source_changed.emit(self.audio_source)

    def _sync_scenario(self) -> None:
        classroom = self.selected_scenario_id == "__classroom__"
        self.set_transcript("")
        self._current_question_id = None
        self.current_question_label.setText("当前问题：等待识别…")
        self._library_ready = False
        self._report_available = False
        self.audio_source_combo.clear()
        self.audio_source_combo.addItem(
            "麦克风（课堂）" if classroom else "电脑全部播放声音",
            self.audio_source,
        )
        profile = self._scenario_profiles.get(self.selected_scenario_id)
        path = str(profile.library_path) if profile is not None else ""
        self.library_path_label.setText(
            f"资料夹：{path}"
            if path
            else ("课堂模式：使用课程资料" if classroom else "资料夹：尚未选择")
        )
        self.library_path_label.setToolTip(path)
        self.library_status_label.setText(
            "课堂模式无需答辩资料索引"
            if classroom
            else ("请先刷新资料索引" if profile else "请先创建场景并选择资料夹")
        )
        self.cloud_note_label.setVisible(not classroom)
        if self.audio_device_combo.count():
            self.audio_device_combo.setItemText(
                0, "系统默认麦克风" if classroom else "系统默认播放设备"
            )
        self._refresh_controls()

    def set_audio_devices(
        self,
        devices: Iterable[object],
        selected_device: int | None = None,
    ) -> None:
        """Populate available inputs or playback outputs without opening audio."""
        self.audio_device_combo.blockSignals(True)
        try:
            self.audio_device_combo.clear()
            default_label = (
                "系统默认麦克风"
                if self.audio_source == "microphone"
                else "系统默认播放设备"
            )
            self.audio_device_combo.addItem(default_label, None)
            for device in devices:
                suffix = "（当前默认）" if getattr(device, "is_default", False) else ""
                self.audio_device_combo.addItem(f"{device.name}{suffix}", device.index)
            index = self.audio_device_combo.findData(selected_device)
            self.audio_device_combo.setCurrentIndex(max(0, index))
        finally:
            self.audio_device_combo.blockSignals(False)

    def set_library_status(self, text: str, ready: bool = False) -> None:
        self._library_ready = ready
        self.library_status_label.setText(text)
        self.library_status_label.setToolTip(text)
        self._refresh_controls()

    def set_report_available(self, available: bool) -> None:
        self._report_available = available
        self._refresh_controls()

    def set_configuration_busy(self, busy: bool) -> None:
        self._configuration_busy = busy
        self._refresh_controls()

    def _refresh_controls(self) -> None:
        locked = (
            self._configuration_busy
            or self._is_listening
            or self._pending_action is not None
        )
        defense = self.selected_scenario_id != "__classroom__"
        has_profile = self.selected_scenario_id in self._scenario_profiles
        self.scenario_combo.setEnabled(not locked)
        self.add_scenario_button.setEnabled(not locked)
        self.edit_scenario_button.setEnabled(not locked and has_profile)
        self.index_button.setEnabled(not locked and has_profile)
        self.cache_management_button.setEnabled(not locked)
        self.index_report_button.setEnabled(
            has_profile and (self._report_available or self._library_ready)
        )
        self.audio_source_combo.setEnabled(False)
        self.audio_device_combo.setEnabled(not locked)
        ready = not defense or (has_profile and self._library_ready)
        self.toggle_button.setEnabled(
            self._pending_action is None
            and (self._is_listening or (not self._configuration_busy and ready))
        )

    def _toggle_listening(self) -> None:
        if not self.toggle_button.isEnabled() or self._pending_action is not None:
            return

        if self._is_listening:
            self._pending_action = "stop"
            self._refresh_controls()
            self.stop_requested.emit()
        else:
            self._pending_action = "start"
            self._refresh_controls()
            self.start_requested.emit()

    def _set_listening(self, listening: bool) -> None:
        self._is_listening = listening
        self._pending_action = None
        self.toggle_button.setText("暂停监听" if listening else "开始监听")
        self._refresh_controls()

    def closeEvent(self, event: QCloseEvent) -> None:
        self.close_requested.emit()
        super().closeEvent(event)

    def showEvent(self, event: QShowEvent) -> None:
        """Reapply capture exclusion; the native handle may have been recreated."""
        super().showEvent(event)
        self._apply_capture_hiding(self.capture_hidden_checkbox.isChecked())

    def _apply_capture_hiding(self, hidden: bool) -> bool:
        """Apply the checked capture exclusion to the native window handle."""
        if not self.capture_hidden_checkbox.isEnabled():
            return False
        if capture_affinity.set_capture_visible(
            int(self.winId()), visible=not hidden
        ):
            self._applied_capture_hidden = hidden
            return True
        # The system refused the change, so the window keeps its previous
        # affinity; restore the checkbox to describe that real state.
        self.capture_hidden_checkbox.blockSignals(True)
        try:
            still_hidden = self._applied_capture_hidden is True
            self.capture_hidden_checkbox.setChecked(still_hidden)
            self.capture_hidden_checkbox.setEnabled(False)
            self.capture_hidden_checkbox.setToolTip(
                "当前环境无法限制屏幕共享，本窗口会正常出现在共享画面中。"
            )
        finally:
            self.capture_hidden_checkbox.blockSignals(False)
        return False

    def set_status(self, status: str) -> None:
        """Update the listening status without doing any background work."""
        self.status_label.setText(self._STATUS_LABELS.get(status, status))
        if status == "listening":
            self._set_listening(True)
        elif status in {"stopped", "asr_error", "mic_error", "audio_error"}:
            self._set_listening(False)
        elif status == "api_error":
            self._pending_action = None
            if self._is_listening:
                self.toggle_button.setText("暂停监听")
                self._refresh_controls()
            else:
                self._set_listening(False)

    def set_transcript(self, text: str) -> None:
        """Display the latest transcript or its empty-state placeholder."""
        self.transcript_label.setText(text or "等待转写…")

    @staticmethod
    def format_elapsed_ms(timestamp_ms: int) -> str:
        """Format a session-relative timestamp as ``HH:MM:SS``."""
        return _format_elapsed_ms(timestamp_ms)

    @staticmethod
    def render_card_text(card: AnswerCard) -> str:
        """Render one complete card from its current snapshot."""
        source_refs = tuple(getattr(card, "source_refs", ()))
        answer = render_answer_plain(_answer_text(card), source_refs)
        status = _status_text(card)
        lines = [
            f"[{MainWindow.format_elapsed_ms(card.timestamp_ms)}] 问题：{card.question}",
            f"回答：{answer}",
            f"状态：{status}",
        ]
        if card.error:
            lines.append(f"错误：{card.error}")
        if getattr(card, "scenario_name", ""):
            lines.append(f"场景：{card.scenario_name}")
        if source_refs:
            cited = cited_source_ids(card.answer)
            lines.append("来源与候选（不代表直接依据）：")
            sources_text = render_sources_plain(source_refs, cited)
            if sources_text:
                lines.extend(sources_text.splitlines())
        elif getattr(card, "sources", ()):
            lines.append("检索候选（不代表直接依据）：")
            lines.extend(card.sources)
        notice = getattr(card, "notice", "")
        if notice:
            lines.append(f"提示：{notice}")
        return "\n".join(lines)

    def upsert_card(self, card: AnswerCard) -> None:
        """Insert or redraw the card identified by ``question_id``."""
        snapshot = replace(card)
        item = self._items.get(snapshot.question_id)
        is_new = item is None
        if item is None:
            if len(self._cards) >= self._MAX_CARDS:
                self._remove_oldest_card()
            self._cards[snapshot.question_id] = snapshot
            item = QListWidgetItem()
            self.cards_list.addItem(item)
            self._items[snapshot.question_id] = item
            card_widget = _AnswerCardWidget(snapshot)
            self._card_widgets[snapshot.question_id] = card_widget
            self.cards_list.setItemWidget(item, card_widget)
        else:
            self._cards[snapshot.question_id] = snapshot
            card_widget = self._card_widgets[snapshot.question_id]

        item.setText(self.render_card_text(snapshot))
        card_widget.update_card(snapshot)
        item.setSizeHint(card_widget.sizeHint())
        if is_new:
            self.cards_list.scrollToBottom()
        if is_new or snapshot.question_id == self._current_question_id:
            self._current_question_id = snapshot.question_id
            self.current_question_label.setText(
                "当前问题 ["
                f"{self.format_elapsed_ms(snapshot.timestamp_ms)}]\n"
                f"{snapshot.question}"
            )

    def _remove_oldest_card(self) -> None:
        oldest_question_id = next(iter(self._cards))
        self._cards.pop(oldest_question_id)
        oldest_item = self._items.pop(oldest_question_id)
        oldest_widget = self._card_widgets.pop(oldest_question_id)
        self.cards_list.removeItemWidget(oldest_item)
        oldest_widget.deleteLater()
        row = self.cards_list.row(oldest_item)
        if row >= 0:
            self.cards_list.takeItem(row)

    def update_answer(self, question_id: str, answer: str) -> None:
        """Replace a known card's answer with its complete cumulative value."""
        card = self._cards.get(question_id)
        if card is None:
            return
        self.upsert_card(replace(card, answer=answer, error=None))
