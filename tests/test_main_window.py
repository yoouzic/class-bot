import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from classbot.domain import AnswerCard, CardStatus
from classbot.ui.main_window import MainWindow


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    created_app = app is None
    if created_app:
        app = QApplication([])
    yield app
    if created_app:
        app.quit()


@pytest.fixture
def window(qapp):
    main_window = MainWindow("历史")
    yield main_window
    main_window.close()


def make_card(
    question_id: str = "q1",
    question: str = "马克思出生于哪一年？",
    *,
    answer: str = "",
    status: CardStatus = CardStatus.DETECTED,
    error: str | None = None,
    timestamp_ms: int = 0,
) -> AnswerCard:
    return AnswerCard(
        question_id=question_id,
        question=question,
        subject="历史",
        answer=answer,
        status=status,
        error=error,
        timestamp_ms=timestamp_ms,
    )


def test_window_starts_with_classroom_floating_state(window):
    assert "历史" in window.windowTitle()
    assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert window.windowFlags() & Qt.WindowType.Tool
    assert window.size().width() == 420
    assert window.size().height() == 560
    assert window.status_label.text() == "已停止"
    assert window.transcript_label.text() == "等待转写…"
    assert window.current_question_label.text() == "当前问题：等待识别…"
    assert window.cards_list.count() == 0
    assert window.toggle_button.text() == "开始监听"


def test_toggle_button_emits_and_switches_text(window):
    started = []
    stopped = []
    window.start_requested.connect(lambda: started.append(True))
    window.stop_requested.connect(lambda: stopped.append(True))

    window.toggle_button.click()

    assert started == [True]
    assert stopped == []
    assert not window.toggle_button.isEnabled()

    window.set_status("listening")

    assert window.toggle_button.text() == "暂停监听"

    window.toggle_button.click()

    assert started == [True]
    assert stopped == [True]
    assert not window.toggle_button.isEnabled()

    window.set_status("stopped")

    assert window.toggle_button.text() == "开始监听"


def test_pending_toggle_ignores_repeated_click_until_status_ack(window):
    started = []
    stopped = []
    window.start_requested.connect(lambda: started.append(True))
    window.stop_requested.connect(lambda: stopped.append(True))

    window.toggle_button.click()
    window.toggle_button.click()

    assert started == [True]
    assert stopped == []
    assert not window.toggle_button.isEnabled()

    window.set_status("listening")

    assert window.toggle_button.isEnabled()
    assert window.toggle_button.text() == "暂停监听"

    window.toggle_button.click()
    window.toggle_button.click()

    assert stopped == [True]
    assert not window.toggle_button.isEnabled()

    window.set_status("stopped")

    assert window.toggle_button.isEnabled()
    assert window.toggle_button.text() == "开始监听"


@pytest.mark.parametrize(
    ("status", "label", "button"),
    [
        ("listening", "● 正在监听", "暂停监听"),
        ("stopped", "已停止", "开始监听"),
        ("asr_error", "ASR 不可用", "开始监听"),
        ("mic_error", "麦克风错误", "开始监听"),
        ("api_error", "API 错误", "暂停监听"),
        ("custom_status", "custom_status", None),
    ],
)
def test_set_status_maps_labels_and_recovers_button(window, status, label, button):
    window.set_status("listening")
    window.set_status(status)

    assert window.status_label.text() == label
    if button is not None:
        assert window.toggle_button.text() == button


def test_api_error_keeps_active_listener_pause_control(window):
    window.set_status("listening")
    window.toggle_button.setEnabled(False)
    window._pending_action = "stop"

    window.set_status("api_error")

    assert window._is_listening is True
    assert window._pending_action is None
    assert window.toggle_button.isEnabled()
    assert window.toggle_button.text() == "暂停监听"


def test_set_transcript_uses_placeholder_for_empty_text(window):
    window.set_transcript("第一行\n第二行")
    assert window.transcript_label.text() == "第一行\n第二行"
    assert window.transcript_label.wordWrap()

    window.set_transcript("")

    assert window.transcript_label.text() == "等待转写…"


def test_upsert_updates_existing_question_id_without_duplicate_item(window):
    window.upsert_card(make_card())
    original_item = window.cards_list.item(0)

    window.upsert_card(
        make_card(
            question="马克思是哪国人？",
            answer="德国人。",
            status=CardStatus.ANSWERED,
        )
    )

    assert window.cards_list.count() == 1
    assert window.cards_list.item(0) is original_item
    assert "问题：马克思是哪国人？" in window.cards_list.item(0).text()
    assert "回答：德国人。" in window.cards_list.item(0).text()
    assert "状态：answered" in window.cards_list.item(0).text()


def test_update_answer_replaces_placeholder_with_complete_answer(window):
    window.upsert_card(make_card(status=CardStatus.ANSWERING))
    assert "回答：等待回答…" in window.cards_list.item(0).text()

    window.update_answer("q1", "马克思出生于1818年。")

    rendered = window.cards_list.item(0).text()
    assert "问题：马克思出生于哪一年？" in rendered
    assert "回答：马克思出生于1818年。" in rendered
    assert "等待回答…" not in rendered
    assert "状态：answering" in rendered


def test_update_answer_ignores_unknown_question_id(window):
    window.update_answer("missing", "不会显示")

    assert window.cards_list.count() == 0


def test_answered_card_without_answer_shows_no_valid_short_answer(window):
    window.upsert_card(make_card(status=CardStatus.ANSWERED))

    rendered = window.cards_list.item(0).text()

    assert "回答：未得到有效简答" in rendered
    assert "等待回答…" not in rendered


def test_error_card_renders_error_and_full_card_state(window):
    card = make_card(
        status=CardStatus.ERROR,
        error="API 请求失败",
    )

    assert "问题：马克思出生于哪一年？" in MainWindow.render_card_text(card)

    window.upsert_card(card)
    rendered = window.cards_list.item(0).text()

    assert "问题：马克思出生于哪一年？" in rendered
    assert "回答：回答失败" in rendered
    assert "状态：error" in rendered
    assert "错误：API 请求失败" in rendered


def test_error_card_shows_failure_and_new_answer_clears_old_error(window):
    window.upsert_card(make_card(status=CardStatus.ERROR, error="旧 API 错误"))

    failed_render = window.cards_list.item(0).text()
    assert "等待回答…" not in failed_render
    assert "失败" in failed_render
    assert "旧 API 错误" in failed_render

    window.update_answer("q1", "新的累计答案")

    answered_render = window.cards_list.item(0).text()
    assert "回答：新的累计答案" in answered_render
    assert "旧 API 错误" not in answered_render
    assert "错误：" not in answered_render


def test_current_question_label_tracks_newest_card_and_card_shows_timestamp(window):
    window.upsert_card(
        make_card(
            question="哲学的基本问题是什么",
            timestamp_ms=65_000,
        )
    )

    assert window.current_question_label.text() == (
        "当前问题 [00:01:05]\n哲学的基本问题是什么"
    )
    assert "[00:01:05]" in window.cards_list.item(0).text()

    window.upsert_card(
        make_card(
            question_id="q2",
            question="物质的概念是什么",
            timestamp_ms=66_000,
        )
    )

    assert window.current_question_label.text() == (
        "当前问题 [00:01:06]\n物质的概念是什么"
    )


def test_close_emits_close_requested(window):
    close_events = []
    window.close_requested.connect(lambda: close_events.append(True))

    window.close()

    assert close_events == [True]


def test_card_limit_evicts_oldest_new_card_but_not_existing_upsert(window):
    for index in range(128):
        window.upsert_card(
            make_card(
                question_id=f"q{index}",
                question=f"问题 {index}",
            )
        )

    window.upsert_card(make_card(question_id="q0", question="问题 0 修订"))

    assert len(window._cards) == 128
    assert len(window._items) == 128
    assert "q0" in window._cards
    assert "q1" in window._cards
    assert window.cards_list.count() == 128

    window.upsert_card(make_card(question_id="q128", question="问题 128"))

    assert len(window._cards) == 128
    assert len(window._items) == 128
    assert window.cards_list.count() == 128
    assert "q0" not in window._cards
    assert "q0" not in window._items
    assert "q1" in window._cards
    assert "q128" in window._cards
    assert "问题 1" in window.cards_list.item(0).text()
    assert "问题 128" in window.cards_list.item(127).text()
