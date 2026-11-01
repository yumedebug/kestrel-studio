"""Right-hand progress column: task, progress, tools, approvals, model."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QProgressBar,
    QVBoxLayout,
    QWidget,
)

from app.agent.sessions import task_status_label
from app.ui.i18n import ja
from app.ui.theme import current_palette
from app.ui.widgets import Card, muted_label


def _caption(text: str) -> QLabel:
    label = QLabel(ja(text))
    label.setProperty("variant", "faint")
    label.setStyleSheet("font-size: 8pt; font-weight: 600; letter-spacing: 0.4px;")
    return label


class ProgressPanel(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._build()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        current = Card()
        current_layout = QVBoxLayout(current)
        current_layout.setContentsMargins(16, 14, 16, 14)
        current_layout.setSpacing(6)
        current_layout.addWidget(_caption("現在のタスク"))
        self.task_goal = QLabel(ja("タスクはまだありません"))
        self.task_goal.setWordWrap(True)
        self.task_goal.setStyleSheet("font-size: 9.5pt; font-weight: 600;")
        current_layout.addWidget(self.task_goal)
        self.task_state = QLabel("—")
        self.task_state.setStyleSheet("font-size: 8.5pt;")
        current_layout.addWidget(self.task_state)
        root.addWidget(current)

        progress = Card()
        progress_layout = QVBoxLayout(progress)
        progress_layout.setContentsMargins(16, 14, 16, 14)
        progress_layout.setSpacing(6)
        progress_layout.addWidget(_caption("進捗"))
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self.progress_bar)
        self.progress_text = muted_label("0 / 0 ステップ")
        self.progress_text.setStyleSheet("font-size: 8.5pt;")
        progress_layout.addWidget(self.progress_text)
        self.steps_text = muted_label("")
        self.steps_text.setStyleSheet("font-size: 8pt;")
        progress_layout.addWidget(self.steps_text)
        root.addWidget(progress)

        root.addWidget(_caption("実行中のツール"))
        self.active_tools = QListWidget()
        self.active_tools.setFixedHeight(64)
        root.addWidget(self.active_tools)

        root.addWidget(_caption("承認待ち"))
        self.pending = QLabel(ja("なし"))
        self.pending.setWordWrap(True)
        self.pending.setStyleSheet(
            f"font-size: 8.5pt; color: {current_palette()['warning']};"
        )
        root.addWidget(self.pending)

        info = Card(soft=True)
        info_layout = QVBoxLayout(info)
        info_layout.setContentsMargins(14, 12, 14, 12)
        info_layout.setSpacing(4)
        info_layout.addWidget(_caption("作業ディレクトリ"))
        self.workspace = QLabel("—")
        self.workspace.setWordWrap(True)
        self.workspace.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.workspace.setStyleSheet("font-size: 8.5pt;")
        info_layout.addWidget(self.workspace)
        info_layout.addSpacing(6)
        info_layout.addWidget(_caption("モデル"))
        self.model = QLabel(ja("モデル未選択"))
        self.model.setWordWrap(True)
        self.model.setStyleSheet("font-size: 8.5pt;")
        info_layout.addWidget(self.model)
        info_layout.addSpacing(6)
        info_layout.addWidget(_caption("上限"))
        self.limits = QLabel("—")
        self.limits.setWordWrap(True)
        self.limits.setStyleSheet("font-size: 8pt;")
        info_layout.addWidget(self.limits)
        root.addWidget(info)
        root.addStretch(1)

    # -- updates -----------------------------------------------------------
    def set_task(self, task: dict[str, Any] | None) -> None:
        if not task:
            self.task_goal.setText(ja("タスクはまだありません"))
            self.task_state.setText("—")
            self.set_state("idle")
            return
        self.task_goal.setText(str(task.get("goal") or ""))
        status = str(task.get("status") or "pending")
        self.set_state(status)
        if task.get("workspace"):
            self.set_workspace(str(task["workspace"]))
        if task.get("model"):
            self.set_model_status(str(task["model"]))

    def set_state(self, state: str) -> None:
        colors = current_palette()
        color = {
            "running": colors["accent"],
            "paused": colors["warning"],
            "stopping": colors["warning"],
            "completed": colors["success"],
            "failed": colors["danger"],
            "stopped": colors["danger"],
            "awaiting_approval": colors["warning"],
            "idle": colors["text_faint"],
        }.get(state, colors["text_muted"])
        self.task_state.setText(ja(task_status_label(state)) if state else "—")
        self.task_state.setStyleSheet(f"font-size: 8.5pt; color: {color}; font-weight: 600;")

    def set_plan_progress(self, steps: list[dict[str, Any]], current: int = 0) -> None:
        total = len(steps)
        done = sum(1 for step in steps if str(step.get("status")) == "completed")
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.progress_text.setText(f"{done} / {total} ステップ完了")
        lines = []
        for index, step in enumerate(steps):
            mark = {
                "completed": "✓",
                "active": "→",
                "skipped": "·",
            }.get(str(step.get("status")), "•")
            if index == current and str(step.get("status")) not in {"completed"}:
                mark = "→"
            lines.append(f"{mark} {step.get('title', '')}")
        self.steps_text.setText("\n".join(lines))

    def set_active_tools(self, tools: list[str]) -> None:
        self.active_tools.clear()
        for tool in tools:
            self.active_tools.addItem(tool)

    def set_pending(self, description: str | None) -> None:
        if description:
            self.pending.setText(description)
        else:
            self.pending.setText(ja("なし"))
        self.pending.setStyleSheet(
            f"font-size: 8.5pt; color: "
            f"{current_palette()['warning'] if description else current_palette()['text_faint']};"
        )

    def set_workspace(self, path: str) -> None:
        self.workspace.setText(path or "—")
        self.workspace.setToolTip(path)

    def set_model_status(self, text: str) -> None:
        self.model.setText(text or ja("モデル未選択"))

    def set_limits(self, limits: Any) -> None:
        self.limits.setText(
            f"最大 {limits.max_steps} ステップ / {int(limits.max_seconds)} 秒 / "
            f"再試行 {limits.max_retries} 回"
        )


__all__ = ["ProgressPanel"]
