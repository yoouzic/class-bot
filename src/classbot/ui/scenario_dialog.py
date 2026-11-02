"""Local scenario setup; no indexing or network access occurs in this dialog."""

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from classbot.scenarios import ScenarioProfile


class ScenarioDialog(QDialog):
    """Create or edit one named scenario with an existing local material folder."""

    def __init__(
        self, parent: QWidget | None = None, profile: object | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("编辑场景" if profile is not None else "新建答辩场景")
        self.resize(520, 250)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit()
        self.name_edit.setObjectName("scenarioNameEdit")
        self.name_edit.setPlaceholderText("例如：课题组周会、科创项目答辩")
        form.addRow("场景名称", self.name_edit)
        self.kind_combo = QComboBox()
        self.kind_combo.setObjectName("scenarioKindCombo")
        self.kind_combo.addItem("组会", "group_meeting")
        self.kind_combo.addItem("竞赛答辩", "competition")
        self.kind_combo.addItem("其他答辩", "defense")
        form.addRow("场景类型", self.kind_combo)

        self.index_mode_combo = QComboBox()
        self.index_mode_combo.setObjectName("indexModeCombo")
        self.index_mode_combo.addItem("完整索引（推荐）", "full")
        self.index_mode_combo.addItem("平衡模式（限制大型资料）", "balanced")
        self.index_mode_combo.setToolTip(
            "完整索引覆盖安全筛选后可解析的资料，达到容量边界会报错；"
            "平衡模式限制大型文件和资料总量。隐藏、依赖、敏感路径及不支持格式仍会跳过，"
            "扫描PDF需先OCR。"
        )
        form.addRow("资料索引", self.index_mode_combo)

        folder_row = QHBoxLayout()
        self.library_path_edit = QLineEdit()
        self.library_path_edit.setObjectName("libraryPathEdit")
        self.library_path_edit.setPlaceholderText("选择这个场景专用的本地材料夹")
        folder_row.addWidget(self.library_path_edit, stretch=1)
        self.browse_button = QPushButton("选择文件夹…")
        self.browse_button.clicked.connect(self._choose_folder)
        folder_row.addWidget(self.browse_button)
        form.addRow("资料文件夹", folder_row)
        layout.addLayout(form)
        self.validation_label = QLabel()
        self.validation_label.setWordWrap(True)
        self.validation_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.validation_label)
        note = QLabel("资料保存在本地；回答时会将问题和相关材料片段发送给 DeepSeek。")
        note.setWordWrap(True)
        layout.addWidget(note)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.save_button = buttons.button(QDialogButtonBox.StandardButton.Save)
        self.save_button.setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.name_edit.textChanged.connect(self._validate)
        self.library_path_edit.textChanged.connect(self._validate)
        self.kind_combo.currentIndexChanged.connect(self._validate)
        self.index_mode_combo.currentIndexChanged.connect(self._validate)

        if profile is not None:
            self.name_edit.setText(profile.name)
            kind_index = self.kind_combo.findData(str(profile.kind))
            self.kind_combo.setCurrentIndex(max(0, kind_index))
            self.library_path_edit.setText(str(profile.library_path))
            self.index_mode_combo.setCurrentIndex(self.index_mode_combo.findData(
                getattr(profile, "index_mode", "balanced")
            ))
        self._validate()

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "选择场景材料文件夹", self.library_path_edit.text().strip()
        )
        if folder:
            self.library_path_edit.setText(folder)

    def _validate(self) -> bool:
        name = self.name_edit.text().strip()
        folder = self.library_path_edit.text().strip()
        message = ""
        if not name:
            message = "请输入场景名称。"
        elif not folder:
            message = "请选择这个场景的材料文件夹。"
        else:
            try:
                exists = Path(folder).expanduser().is_dir()
            except (OSError, RuntimeError, ValueError):
                exists = False
            if not exists:
                message = "资料文件夹不存在，请重新选择。"
        if not message:
            try:
                ScenarioProfile("dialog-preview", **self.values())
            except (OSError, RuntimeError, TypeError, ValueError) as error:
                message = {
                    "invalid scenario name": "场景名称最多120个字符，且不能包含控制字符。",
                    "invalid scenario kind": "请选择场景类型。",
                }.get(str(error), f"场景信息无效：{error}")
        self.validation_label.setText(message)
        self.save_button.setEnabled(not message)
        return not message

    def accept(self) -> None:
        if self._validate():
            super().accept()

    def values(self) -> dict[str, str]:
        return {
            "name": self.name_edit.text().strip(),
            "kind": self.kind_combo.currentData(),
            "index_mode": self.index_mode_combo.currentData(),
            "library_path": str(
                Path(self.library_path_edit.text().strip()).expanduser()
            ),
        }
