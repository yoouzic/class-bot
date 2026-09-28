"""Classroom floating window for displaying live classbot answers."""

from dataclasses import replace
from typing import ClassVar

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from classbot.domain import AnswerCard


class MainWindow(QMainWindow):
    """A small always-on-top window for classroom listening feedback."""

    start_requested = Signal()
    stop_requested = Signal()

    _STATUS_LABELS: ClassVar[dict[str, str]] = {
        "listening": "● 正在监听",
        "stopped": "已停止",
        "asr_error": "ASR 不可用",
        "api_error": "API 错误",
    }

    def __init__(self, subject: str) -> None:
        super().__init__()
        self._is_listening = False
        self._cards: dict[str, AnswerCard] = {}
        self._items: dict[str, QListWidgetItem] = {}

        self.setWindowTitle(f"Classbot · {subject}")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.Tool)
        self.resize(420, 560)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        self.status_label = QLabel("已停止")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)

        self.transcript_label = QLabel("等待转写…")
        self.transcript_label.setObjectName("transcriptLabel")
        self.transcript_label.setWordWrap(True)
        layout.addWidget(self.transcript_label)

        self.cards_list = QListWidget()
        self.cards_list.setObjectName("cardsList")
        self.cards_list.setWordWrap(True)
        self.cards_list.setUniformItemSizes(False)
        layout.addWidget(self.cards_list, stretch=1)

        self.toggle_button = QPushButton("开始监听")
        self.toggle_button.setObjectName("toggleButton")
        self.toggle_button.clicked.connect(self._toggle_listening)
        layout.addWidget(self.toggle_button)

    def _toggle_listening(self) -> None:
        if self._is_listening:
            self._set_listening(False)
            self.stop_requested.emit()
        else:
            self._set_listening(True)
            self.start_requested.emit()

    def _set_listening(self, listening: bool) -> None:
        self._is_listening = listening
        self.toggle_button.setText("暂停监听" if listening else "开始监听")

    def set_status(self, status: str) -> None:
        """Update the listening status without doing any background work."""
        self.status_label.setText(self._STATUS_LABELS.get(status, status))
        if status == "listening":
            self._set_listening(True)
        elif status in {"stopped", "asr_error", "api_error"}:
            self._set_listening(False)

    def set_transcript(self, text: str) -> None:
        """Display the latest transcript or its empty-state placeholder."""
        self.transcript_label.setText(text or "等待转写…")

    @staticmethod
    def render_card_text(card: AnswerCard) -> str:
        """Render one complete card from its current snapshot."""
        answer = card.answer or "等待回答…"
        status = getattr(card.status, "value", str(card.status))
        lines = [
            f"问题：{card.question}",
            f"回答：{answer}",
            f"状态：{status}",
        ]
        if card.error:
            lines.append(f"错误：{card.error}")
        return "\n".join(lines)

    def upsert_card(self, card: AnswerCard) -> None:
        """Insert or redraw the card identified by ``question_id``."""
        snapshot = replace(card)
        self._cards[snapshot.question_id] = snapshot

        item = self._items.get(snapshot.question_id)
        if item is None:
            item = QListWidgetItem()
            self.cards_list.addItem(item)
            self._items[snapshot.question_id] = item

        item.setText(self.render_card_text(snapshot))

    def update_answer(self, question_id: str, answer: str) -> None:
        """Replace a known card's answer with its complete cumulative value."""
        card = self._cards.get(question_id)
        if card is None:
            return
        self.upsert_card(replace(card, answer=answer))
