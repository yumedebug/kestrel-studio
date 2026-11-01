"""Approval dialog: Approve / Always Allow / Reject / Stop Task.

The dialog always renders the exact command, shell, working directory, purpose,
target, risk and whether administrator rights are needed. "Always Allow" is only
enabled when the permission manager marked the action as grantable; installation,
elevation and system changes never are.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.agent.permissions import ApprovalOutcome, ApprovalRequest
from app.agent.process_manager import shell_label
from app.ui.i18n import ja
from app.ui.theme import current_palette
from app.ui.widgets import Card, muted_label, section_title


class ApprovalDialog(QDialog):
    """Modal confirmation for one agent action."""

    def __init__(self, request: ApprovalRequest, parent: QWidget | None = None):
        super().__init__(parent)
        self.request = request
        self.outcome: ApprovalOutcome | None = None
        self.setWindowTitle(ja("操作の承認"))
        self.setModal(True)
        self.setMinimumWidth(660)
        self.setMaximumWidth(860)

        colors = current_palette()
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 20, 22, 18)
        root.setSpacing(14)

        header = QHBoxLayout()
        header.addWidget(section_title("操作の承認が必要です"))
        header.addStretch(1)
        risk = QLabel(ja(request.risk))
        risk.setStyleSheet(
            f"color: {colors['warning']}; font-size: 9pt; font-weight: 600;"
            f" border: 1px solid {colors['warning']}; border-radius: 9px; padding: 4px 10px;"
        )
        header.addWidget(risk)
        root.addLayout(header)
        root.addWidget(
            muted_label(
                "この操作は自動実行の対象外です。内容を確認してから許可してください。"
            )
        )

        card = Card(soft=True)
        grid = QGridLayout(card)
        grid.setContentsMargins(16, 14, 16, 14)
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(8)
        row = 0
        self._add_row(grid, row, "シェル", shell_label(request.shell) if request.command else "—")
        row += 1
        self._add_row(grid, row, "作業ディレクトリ", request.cwd or "—")
        row += 1
        self._add_row(grid, row, "目的", request.purpose or "—")
        row += 1
        self._add_row(grid, row, "変更対象", request.target or "—")
        row += 1
        self._add_row(grid, row, "想定リスク", request.risk)
        row += 1
        self._add_row(
            grid,
            row,
            "管理者権限",
            "必要（Windows の UAC 画面が表示されます）" if request.requires_admin else "不要",
        )
        row += 1
        if request.reason:
            self._add_row(grid, row, "判定理由", request.reason)
            row += 1
        root.addWidget(card)

        if request.command:
            command_title = QLabel(ja("実行するコマンド"))
            command_title.setProperty("variant", "muted")
            root.addWidget(command_title)
            command_view = QPlainTextEdit(request.command)
            command_view.setReadOnly(True)
            command_view.setFixedHeight(92)
            command_view.setObjectName("codeView")
            command_view.setFrameShape(QFrame.NoFrame)
            wrapper = Card(soft=True)
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(10, 8, 10, 8)
            wrapper_layout.addWidget(command_view)
            root.addWidget(wrapper)

        if not request.grantable:
            note = QLabel(
                ja(
                    "この操作は「常に許可」できません。インストール・管理者権限・"
                    "システム設定の変更は毎回個別に承認が必要です。"
                )
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {colors['text_muted']}; font-size: 8.5pt;")
            root.addWidget(note)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.stop_button = QPushButton(ja("Stop Task"))
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(lambda: self._finish("stop"))
        self.reject_button = QPushButton(ja("Reject"))
        self.reject_button.setProperty("variant", "danger")
        self.reject_button.setCursor(Qt.PointingHandCursor)
        self.reject_button.clicked.connect(lambda: self._finish("reject"))
        self.always_button = QPushButton(ja("Always Allow"))
        self.always_button.setCursor(Qt.PointingHandCursor)
        self.always_button.setEnabled(request.grantable)
        if not request.grantable:
            self.always_button.setToolTip(
                ja("この種類の操作は保存できません。毎回の承認が必要です。")
            )
        self.always_button.clicked.connect(lambda: self._finish("always_allow"))
        self.allow_button = QPushButton(ja("Allow Once"))
        self.allow_button.setProperty("variant", "primary")
        self.allow_button.setCursor(Qt.PointingHandCursor)
        self.allow_button.setDefault(True)
        self.allow_button.clicked.connect(lambda: self._finish("allow_once"))
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.reject_button)
        buttons.addStretch(1)
        buttons.addWidget(self.always_button)
        buttons.addWidget(self.allow_button)
        root.addLayout(buttons)

    def _add_row(self, grid: QGridLayout, row: int, label: str, value: str) -> None:
        key = QLabel(ja(label))
        key.setProperty("variant", "muted")
        key.setStyleSheet("font-size: 9pt;")
        value_label = QLabel(value)
        value_label.setWordWrap(True)
        value_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        value_label.setStyleSheet("font-size: 9pt;")
        grid.addWidget(key, row, 0, Qt.AlignTop)
        grid.addWidget(value_label, row, 1)

    def _finish(self, choice: str) -> None:
        self.outcome = ApprovalOutcome(choice, self.request.fingerprint)
        if choice in {"allow_once", "always_allow"}:
            self.accept()
        else:
            self.reject()

    def reject(self) -> None:  # type: ignore[override]
        if self.outcome is None:
            self.outcome = ApprovalOutcome("reject", self.request.fingerprint)
        super().reject()


__all__ = ["ApprovalDialog"]
