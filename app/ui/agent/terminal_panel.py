"""Terminal panel: agent and manually typed commands in one timeline.

Agent commands and user commands are visually tagged and stored separately
(``source`` = agent / user), but both go through the same permission checks.
"""

from __future__ import annotations

import time
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.agent.process_manager import SHELL_LABELS
from app.ui.i18n import ja
from app.ui.theme import current_palette

MAX_OUTPUT_CHARS = 400_000


class TerminalPanel(QWidget):
    command_submitted = Signal(str, str)  # command, shell
    stop_requested = Signal()
    show_status_message = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._colors = current_palette()
        self._auto_scroll = True
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(8)
        self.source_label = QLabel(ja("ターミナル"))
        self.source_label.setObjectName("sectionTitle")
        toolbar.addWidget(self.source_label)

        self.shell_combo = QComboBox()
        for key, label in SHELL_LABELS.items():
            self.shell_combo.addItem(ja(label), key)
        self.shell_combo.setFixedWidth(170)
        toolbar.addWidget(self.shell_combo)
        toolbar.addStretch(1)

        self.stop_button = QPushButton(ja("実行中のコマンドを停止"))
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_requested.emit)
        toolbar.addWidget(self.stop_button)

        self.copy_button = QPushButton(ja("出力をコピー"))
        self.copy_button.setCursor(Qt.PointingHandCursor)
        self.copy_button.clicked.connect(self._copy_output)
        toolbar.addWidget(self.copy_button)

        self.clear_button = QPushButton(ja("出力をクリア"))
        self.clear_button.setCursor(Qt.PointingHandCursor)
        self.clear_button.clicked.connect(self.clear_output)
        toolbar.addWidget(self.clear_button)
        root.addLayout(toolbar)

        frame = QFrame()
        frame.setProperty("card", "soft")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(10, 8, 10, 8)
        frame_layout.setSpacing(6)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setObjectName("codeView")
        self.output.setFrameShape(QFrame.NoFrame)
        self.output.setMaximumBlockCount(20_000)
        font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        font.setPointSizeF(9.0)
        self.output.setFont(font)
        frame_layout.addWidget(self.output, 1)

        input_row = QHBoxLayout()
        input_row.setSpacing(6)
        prompt = QLabel("›")
        prompt.setStyleSheet(f"color: {self._colors['accent']}; font-weight: 600;")
        input_row.addWidget(prompt)
        self.input = QLineEdit()
        self.input.setPlaceholderText(
            ja("手動でコマンドを入力（Agent と同じ承認ルールが適用されます）")
        )
        self.input.returnPressed.connect(self._submit)
        input_row.addWidget(self.input, 1)
        self.run_button = QPushButton(ja("実行"))
        self.run_button.setProperty("variant", "primary")
        self.run_button.setCursor(Qt.PointingHandCursor)
        self.run_button.clicked.connect(self._submit)
        input_row.addWidget(self.run_button)
        frame_layout.addLayout(input_row)
        root.addWidget(frame, 1)

        self.status_label = QLabel("")
        self.status_label.setProperty("variant", "faint")
        self.status_label.setStyleSheet("font-size: 8.5pt;")
        root.addWidget(self.status_label)

    # -- behaviour ---------------------------------------------------------
    def current_shell(self) -> str:
        return str(self.shell_combo.currentData() or "cmd")

    def _submit(self) -> None:
        command = self.input.text().strip()
        if not command:
            return
        self.input.clear()
        self.command_submitted.emit(command, self.current_shell())

    def _copy_output(self) -> None:
        from PySide6.QtWidgets import QApplication

        QApplication.clipboard().setText(self.output.toPlainText())
        self.show_status_message.emit("出力をクリップボードにコピーしました")

    def clear_output(self) -> None:
        self.output.clear()
        self.status_label.setText("")

    # -- rendering ---------------------------------------------------------
    def append_command(
        self,
        command: str,
        shell: str = "",
        cwd: str = "",
        source: str = "agent",
        purpose: str = "",
    ) -> None:
        tag = "AGENT" if source == "agent" else "USER"
        color = self._colors["accent"] if source == "agent" else self._colors["text_muted"]
        stamp = time.strftime("%H:%M:%S")
        header = f"[{stamp}] [{tag}] ({shell or 'cmd'}) {command}"
        if purpose:
            header += f"    # {purpose}"
        if cwd:
            header += f"\n           cwd: {cwd}"
        self._append(f'<span style="color:{color}; font-weight:600;">{_escape(header)}</span>')

    def append_output(self, text: str, stream: str = "stdout") -> None:
        if not text:
            return
        color = self._colors["danger"] if stream == "stderr" else self._colors["text"]
        self._append(f'<span style="color:{color};">{_escape(text.rstrip(chr(10)))}</span>')

    def append_result(
        self,
        exit_code: Any = None,
        duration: float = 0.0,
        status: str = "done",
        cancelled: bool = False,
        timed_out: bool = False,
    ) -> None:
        if cancelled:
            color, text = self._colors["warning"], "コマンドを停止しました"
        elif timed_out:
            color, text = self._colors["warning"], "タイムアウトしました"
        elif exit_code == 0:
            color, text = self._colors["success"], "終了コード 0"
        else:
            color, text = self._colors["danger"], f"終了コード {exit_code}"
        detail = f"{text} · {duration:.2f}s" if duration else text
        self._append(f'<span style="color:{color};">── {_escape(detail)}</span>')
        self.status_label.setText(detail)

    def append_info(self, text: str) -> None:
        self._append(
            f'<span style="color:{self._colors["text_muted"]};">{_escape(text)}</span>'
        )

    def _append(self, html: str) -> None:
        bar = self.output.verticalScrollBar()
        at_bottom = bar.value() >= bar.maximum() - 4
        self.output.appendHtml(html)
        if self.output.blockCount() > 18_000:
            self.output.clear()
        if at_bottom or self._auto_scroll:
            bar.setValue(bar.maximum())


def _escape(text: str) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace(" ", "&nbsp;")
    )


__all__ = ["TerminalPanel"]
