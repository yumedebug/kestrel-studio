"""Left column (new task, sessions, task history, workspace) and plan/log panels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.agent.sessions import task_status_label
from app.ui.i18n import ja
from app.ui.theme import current_palette
from app.ui.widgets import muted_label

_STATUS_MARK = {
    "completed": "✓",
    "running": "▶",
    "paused": "Ⅱ",
    "awaiting_approval": "!",
    "failed": "×",
    "stopped": "■",
    "pending": "•",
}


class TaskListPanel(QWidget):
    new_task_requested = Signal()
    task_selected = Signal(str)
    memory_requested = Signal()
    workspace_changed = Signal(str)
    full_access_changed = Signal(bool)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 10, 4, 10)
        root.setSpacing(10)

        self.new_button = QPushButton(ja("＋  New Task"))
        self.new_button.setProperty("variant", "primary")
        self.new_button.setCursor(Qt.PointingHandCursor)
        self.new_button.clicked.connect(self.new_task_requested.emit)
        root.addWidget(self.new_button)

        root.addWidget(_caption("Agent Sessions"))
        self.sessions = QListWidget()
        self.sessions.setSpacing(3)
        self.sessions.setViewportMargins(3, 3, 3, 3)
        self.sessions.currentItemChanged.connect(self._selected)
        root.addWidget(self.sessions, 1)

        root.addWidget(_caption("Task History"))
        self.history = QListWidget()
        self.history.setMaximumHeight(130)
        self.history.setSpacing(3)
        self.history.setViewportMargins(3, 3, 3, 3)
        self.history.currentItemChanged.connect(self._selected)
        root.addWidget(self.history)

        self.memory_button = QPushButton(ja("Memory"))
        self.memory_button.setCursor(Qt.PointingHandCursor)
        self.memory_button.clicked.connect(self.memory_requested.emit)
        root.addWidget(self.memory_button)

        root.addWidget(_caption("Workspace"))
        row = QHBoxLayout()
        row.setSpacing(4)
        self.workspace_edit = QLineEdit()
        self.workspace_edit.setReadOnly(True)
        self.workspace_edit.setPlaceholderText(ja("作業ディレクトリを選択"))
        row.addWidget(self.workspace_edit, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(34)
        browse.clicked.connect(self._browse)
        row.addWidget(browse)
        root.addLayout(row)

        self.full_access = QCheckBox(ja("Full Access Mode"))
        self.full_access.setToolTip(
            ja(
                "作業ディレクトリ外のファイル変更を自動で許可します。"
                "インストール・管理者権限・システム設定の変更は常に個別承認が必要です。"
            )
        )
        self.full_access.toggled.connect(self.full_access_changed.emit)
        root.addWidget(self.full_access)

        hint = muted_label(
            "ファイルの作成・編集・移動・削除はどのモードでも利用できます。"
        )
        hint.setStyleSheet("font-size: 8pt;")
        root.addWidget(hint)

    # -- data --------------------------------------------------------------
    def refresh(self, tasks: list[Any], current_id: str = "") -> None:
        self.sessions.blockSignals(True)
        self.history.blockSignals(True)
        try:
            self.sessions.clear()
            self.history.clear()
            for task in tasks:
                mark = _STATUS_MARK.get(task.status, "•")
                label = f"{mark}  {task.goal[:46] or '(no goal)'}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, task.id)
                item.setToolTip(
                    f"{task.goal}\n{ja(task_status_label(task.status))} · {task.updated_at}"
                )
                self.sessions.addItem(item)
                if task.status in {"completed", "failed", "stopped"}:
                    history_item = QListWidgetItem(
                        f"{mark}  {task.goal[:40]}  ·  {ja(task_status_label(task.status))}"
                    )
                    history_item.setData(Qt.UserRole, task.id)
                    self.history.addItem(history_item)
                if task.id == current_id:
                    self.sessions.setCurrentItem(item)
        finally:
            self.sessions.blockSignals(False)
            self.history.blockSignals(False)

    def select(self, task_id: str) -> None:
        for row in range(self.sessions.count()):
            item = self.sessions.item(row)
            if item.data(Qt.UserRole) == task_id:
                self.sessions.setCurrentItem(item)
                return

    def _selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        task_id = str(current.data(Qt.UserRole) or "")
        if task_id:
            self.task_selected.emit(task_id)

    def _browse(self) -> None:
        start = self.workspace_edit.text() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, ja("作業ディレクトリを選択"), start)
        if directory:
            self.workspace_edit.setText(directory)
            self.workspace_changed.emit(directory)

    def set_workspace(self, path: str) -> None:
        self.workspace_edit.setText(path)
        self.workspace_edit.setToolTip(path)

    def set_full_access(self, enabled: bool) -> None:
        self.full_access.blockSignals(True)
        self.full_access.setChecked(bool(enabled))
        self.full_access.blockSignals(False)


class PlanPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(9)
        self.notes = muted_label("")
        self.notes.setStyleSheet("font-size: 8.5pt;")
        layout.addWidget(self.notes)
        self.list = QListWidget()
        self.list.setSpacing(4)
        self.list.setViewportMargins(3, 3, 3, 3)
        layout.addWidget(self.list, 1)

    def set_plan(self, plan: dict[str, Any], current: int = 0) -> None:
        self.notes.setText(str(plan.get("notes") or ""))
        self.list.clear()
        steps = plan.get("steps") or []
        for index, step in enumerate(steps):
            status = str(step.get("status") or "pending")
            mark = {"completed": "✓", "active": "→", "skipped": "·"}.get(status, "•")
            if index == current and status != "completed":
                mark = "→"
            item = QListWidgetItem(f"{mark}  {index + 1}. {step.get('title', '')}")
            if status == "completed":
                item.setForeground(Qt.gray)
            if index == current:
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            if step.get("tool"):
                item.setToolTip(f"tool: {step.get('tool')}")
            self.list.addItem(item)
        if steps:
            self.list.setCurrentRow(min(current, len(steps) - 1))

    def clear(self) -> None:
        self.notes.setText("")
        self.list.clear()


class EventLogPanel(QWidget):
    """Plain, append-only log used for thoughts, tool results and file changes."""

    def __init__(self, empty_text: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(9)
        self.empty = QLabel(ja(empty_text))
        self.empty.setProperty("variant", "faint")
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setObjectName("codeView")
        self.view.setMaximumBlockCount(4000)
        layout.addWidget(self.view, 1)

    def append(self, text: str) -> None:
        if not text:
            return
        self.empty.hide()
        self.view.appendPlainText(text)
        bar = self.view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def clear(self) -> None:
        self.view.clear()
        self.empty.show()


def _caption(text: str) -> QLabel:
    label = QLabel(ja(text))
    label.setProperty("variant", "faint")
    label.setStyleSheet("font-size: 8pt; font-weight: 600; letter-spacing: 0.4px;")
    return label


__all__ = ["EventLogPanel", "PlanPanel", "TaskListPanel"]
