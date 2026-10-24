"""Inspect and remove generated library caches without touching source material."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from os.path import abspath, normcase
from typing import TYPE_CHECKING

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from classbot.knowledge.cache_manager import CacheInventory, CacheManager


def format_cache_bytes(size: int) -> str:
    """Use binary units consistently with the actual file sizes."""
    if size >= 1024**3:
        return f"{size / 1024**3:.2f} GiB"
    if size >= 1024**2:
        return f"{size / 1024**2:.1f} MiB"
    if size >= 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size} B"


@dataclass(frozen=True)
class _Outcome:
    value: object = None
    inventory: object = None
    refresh_error: str = ""
    mutation_error: str = ""


class _JobSignals(QObject):
    finished = Signal(str, object, object)


class _CacheJob(QRunnable):
    def __init__(self, operation: str, action: Callable[[], _Outcome]) -> None:
        super().__init__()
        self.operation = operation
        self.action = action
        self.signals = _JobSignals()

    def run(self) -> None:
        try:
            result = self.action()
        except Exception as error:  # noqa: BLE001 - finish the Qt worker even for an unexpected filesystem failure.
            self.signals.finished.emit(self.operation, None, str(error))
        else:
            self.signals.finished.emit(self.operation, result, None)


class CacheDialog(QDialog):
    """Manage only cache artifacts; filesystem work runs outside the Qt thread."""

    cache_changed = Signal(object)

    def __init__(
        self,
        manager: CacheManager,
        *,
        profiles: Iterable[object] = (),
        protected_keys: Iterable[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.manager = manager
        self._protected_keys = frozenset(protected_keys)
        self._profiles = tuple(profiles)
        self._inventory: CacheInventory | None = None
        self._rows: list[object] = []
        self._busy = False
        self._job: _CacheJob | None = None
        self._operation_key: str | None = None
        self.setWindowTitle("资料库缓存管理")
        self.resize(1040, 650)

        layout = QVBoxLayout(self)
        self.usage_label = QLabel("正在统计缓存…")
        self.usage_label.setObjectName("cacheUsageLabel")
        self._plain_label(self.usage_label)
        layout.addWidget(self.usage_label)
        self.categories_label = QLabel()
        self.categories_label.setObjectName("cacheCategoriesLabel")
        self._plain_label(self.categories_label)
        layout.addWidget(self.categories_label)

        self.table = QTableWidget(0, 5)
        self.table.setObjectName("libraryCacheTable")
        self.table.setHorizontalHeaderLabels(
            ["资料库 / 场景", "索引模式", "索引大小", "文件 / 片段", "状态"]
        )
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        for column in range(1, 5):
            self.table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.table.itemSelectionChanged.connect(self._refresh_controls)
        layout.addWidget(self.table, stretch=1)

        actions = QHBoxLayout()
        self.refresh_button = QPushButton("刷新统计")
        self.refresh_button.setObjectName("refreshCacheButton")
        self.refresh_button.clicked.connect(self.refresh)
        actions.addWidget(self.refresh_button)
        self.clear_button = QPushButton("清理选中索引…")
        self.clear_button.setObjectName("clearLibraryCacheButton")
        self.clear_button.clicked.connect(self._clear_selected)
        actions.addWidget(self.clear_button)
        self.clear_shared_button = QPushButton("清理共享向量缓存…")
        self.clear_shared_button.setObjectName("clearSharedVectorCacheButton")
        self.clear_shared_button.clicked.connect(self._clear_shared)
        actions.addWidget(self.clear_shared_button)
        actions.addStretch()
        layout.addLayout(actions)

        limit_row = QHBoxLayout()
        limit_row.addWidget(QLabel("缓存总占用上限"))
        self.limit_spin = QDoubleSpinBox()
        self.limit_spin.setObjectName("cacheLimitSpin")
        self.limit_spin.setRange(0.01, 1024 * 1024)
        self.limit_spin.setDecimals(2)
        self.limit_spin.setSingleStep(1)
        self.limit_spin.setSuffix(" GiB")
        self.limit_spin.valueChanged.connect(self._refresh_controls)
        limit_row.addWidget(self.limit_spin)
        self.save_limit_button = QPushButton("保存上限")
        self.save_limit_button.setObjectName("saveCacheLimitButton")
        self.save_limit_button.clicked.connect(self._save_limit)
        limit_row.addWidget(self.save_limit_button)
        limit_row.addStretch()
        layout.addLayout(limit_row)

        note = QLabel(
            "仅清理程序生成的索引或共享向量缓存，原始资料不会修改。"
            "清理索引后，使用该资料库需重新建立完整索引；共享向量缓存仅用于加速重建。\n"
            "总占用包含索引、共享向量、模型和其他缓存。空间不足时停止建立索引，"
            "保留已有完整快照，不会为节省空间跳过资料。模型与未知文件不在此处删除。"
            "重建期间的新旧快照与临时文件也计入上限。\n"
            "若要清理当前已加载的资料库，请先停止监听并切换到课堂模式。"
        )
        note.setObjectName("cacheSafetyNote")
        self._plain_label(note)
        layout.addWidget(note)
        self.status_label = QLabel()
        self.status_label.setObjectName("cacheActionStatusLabel")
        self._plain_label(self.status_label)
        layout.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.buttons.button(QDialogButtonBox.StandardButton.Close).setText("关闭")
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.refresh()

    @staticmethod
    def _plain_label(label: QLabel) -> None:
        label.setTextFormat(Qt.TextFormat.PlainText)
        label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        label.setWordWrap(True)

    @property
    def busy(self) -> bool:
        return self._busy

    def refresh(self) -> None:
        if not self._busy:
            self._submit(
                "refresh", lambda: _Outcome(inventory=self.manager.inventory())
            )

    def _submit(self, operation: str, action: Callable[[], _Outcome]) -> None:
        if self._busy:
            return
        self._busy = True
        self.status_label.setText("正在统计缓存…" if operation == "refresh" else "正在处理…")
        self._refresh_controls()
        self._job = _CacheJob(operation, action)
        self._job.signals.finished.connect(self._finished)
        QThreadPool.globalInstance().start(self._job)

    def _mutate(self, operation: str, action: Callable[[], object]) -> None:
        def apply_and_refresh() -> _Outcome:
            value = None
            mutation_error = ""
            try:
                value = action()
            except Exception as error:  # noqa: BLE001 - a partially completed cleanup still needs fresh inventory.
                mutation_error = str(error) or type(error).__name__
            try:
                inventory = self.manager.inventory()
            except Exception as error:  # noqa: BLE001 - a failed refresh must not hide a completed cache mutation.
                return _Outcome(
                    value=value,
                    mutation_error=mutation_error,
                    refresh_error=str(error) or type(error).__name__,
                )
            return _Outcome(
                value=value, inventory=inventory, mutation_error=mutation_error
            )

        self._submit(operation, apply_and_refresh)

    @Slot(str, object, object)
    def _finished(self, operation: str, result: _Outcome | None, error: str | None) -> None:
        self._busy = False
        self._job = None
        if error is not None:
            self.status_label.setText(f"操作未完成：{error}")
        elif result is not None:
            if result.inventory is not None:
                self._render_inventory(result.inventory)
            if result.mutation_error:
                self.status_label.setText(f"操作未完成：{result.mutation_error}。")
                if operation.startswith("clear"):
                    self.status_label.setText(
                        self.status_label.text() + "清理可能已移除部分缓存；"
                        + ("最新占用已刷新。" if result.inventory is not None else "最新占用未能刷新，请重试统计。")
                    )
                elif result.inventory is not None:
                    self.status_label.setText(self.status_label.text() + "最新占用已刷新。")
            elif operation == "refresh":
                self.status_label.setText("统计完成。可选择不再需要的索引进行清理。")
            else:
                freed = int(result.value or 0) if operation.startswith("clear") else 0
                self.status_label.setText(
                    f"已清理 {format_cache_bytes(freed)}，原始资料未修改。"
                    if operation.startswith("clear")
                    else "总占用上限已保存。"
                )
                self.cache_changed.emit(
                    {
                        "operation": operation,
                        "key": self._operation_key,
                        "freed_bytes": freed,
                    }
                )
            if result.refresh_error:
                self._inventory = None
                self.status_label.setText(
                    self.status_label.text() + f" 最新统计读取失败：{result.refresh_error}"
                )
        self._operation_key = None
        self._refresh_controls()

    def _render_inventory(self, inventory: CacheInventory) -> None:
        previous = self._selected_library()
        previous_key = previous.key if previous is not None else None
        self._inventory = inventory
        self.usage_label.setText(
            f"缓存总占用：{format_cache_bytes(inventory.total_bytes)} / "
            f"上限 {format_cache_bytes(inventory.limit_bytes)}"
            + ("（已超出上限，请清理缓存或提高上限）" if inventory.total_bytes > inventory.limit_bytes else "")
        )
        index_bytes = sum(library.size_bytes for library in inventory.libraries)
        self.categories_label.setText(
            f"资料库索引 {format_cache_bytes(index_bytes)}　"
            f"共享向量 {format_cache_bytes(inventory.shared_bytes)}　"
            f"共享模型 {format_cache_bytes(inventory.model_bytes)}　"
            f"其他缓存 {format_cache_bytes(inventory.other_bytes)}"
        )
        self.limit_spin.blockSignals(True)
        try:
            self.limit_spin.setValue(inventory.limit_bytes / 1024**3)
        finally:
            self.limit_spin.blockSignals(False)
        self.table.blockSignals(True)
        try:
            self._rows = list(inventory.libraries)
            self.table.setRowCount(len(self._rows))
            for row, library in enumerate(self._rows):
                names = [
                    str(profile.name)
                    for profile in self._profiles
                    if normcase(abspath(str(profile.library_path)))
                    == normcase(abspath(str(library.root)))
                ]
                root = str(library.root) if library.root else "未记录资料路径"
                name = f"{'、'.join(names)} · {root}" if names else root
                status = {
                    "ready": "已保存",
                    "incomplete": "未完成 / 历史文件",
                    "unsafe": "安全检查未通过",
                }.get(library.status, str(library.status))
                if library.status != "unsafe" and (
                    library.protected or library.key in self._protected_keys
                ):
                    status = "使用中，不能清理"
                values = (
                    name,
                    {"full": "完整", "balanced": "均衡"}.get(library.mode, library.mode or "旧版"),
                    format_cache_bytes(library.size_bytes),
                    f"{library.file_count:,} / {library.chunk_count:,}",
                    status,
                )
                for column, value in enumerate(values):
                    item = QTableWidgetItem(str(value))
                    item.setToolTip(
                        f"{root}\n模型：{library.model}\n建立时间：{library.built_at or '未记录'}\n索引：{library.key}"
                    )
                    self.table.setItem(row, column, item)
                if library.key == previous_key:
                    self.table.selectRow(row)
        finally:
            self.table.blockSignals(False)

    def _selected_library(self):
        row = self.table.currentRow()
        return self._rows[row] if 0 <= row < len(self._rows) else None

    def _refresh_controls(self) -> None:
        ready = self._inventory is not None and not self._busy
        selected = self._selected_library()
        can_clear = (
            ready
            and selected is not None
            and not selected.protected
            and selected.key not in self._protected_keys
            and selected.status != "unsafe"
        )
        self.table.setEnabled(not self._busy)
        self.refresh_button.setEnabled(not self._busy)
        self.clear_button.setEnabled(bool(can_clear))
        self.clear_shared_button.setEnabled(
            bool(ready and self._inventory.shared_bytes > 0)
        )
        self.limit_spin.setEnabled(ready)
        self.save_limit_button.setEnabled(ready)
        self.buttons.setEnabled(not self._busy)

    def _clear_selected(self) -> None:
        library = self._selected_library()
        if not self.clear_button.isEnabled() or library is None:
            return
        answer = QMessageBox.question(
            self,
            "清理资料库索引",
            f"清理这个资料库的生成索引（{format_cache_bytes(library.size_bytes)}）？\n"
            f"{library.root}\n\n原始资料不会修改。下次使用前需要重新建立完整索引。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._operation_key = library.key
        self._mutate(
            "clear_library",
            lambda: self.manager.clear_library(
                library.key, protected_keys=self._protected_keys  # gitleaks:allow -- cache identifiers, not credentials
            ),
        )

    def _clear_shared(self) -> None:
        if not self.clear_shared_button.isEnabled():
            return
        answer = QMessageBox.question(
            self,
            "清理共享向量缓存",
            "清理用于加速重建的共享向量缓存？\n\n"
            "现有完整索引仍可使用，原始资料不会修改；以后刷新索引可能需要更长时间。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self._mutate("clear_shared", self.manager.clear_shared_vectors)

    def _save_limit(self) -> None:
        if not self.save_limit_button.isEnabled():
            return
        limit = round(self.limit_spin.value() * 1024**3)
        self._mutate("set_limit", lambda: self.manager.set_limit(limit))

    def reject(self) -> None:
        if not self._busy:
            super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._busy:
            event.ignore()
        else:
            super().closeEvent(event)
