"""User-facing Computer Use controls with an explicit approval boundary."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QHideEvent, QPixmap, QShowEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.agent.computer_use import ComputerUse, ComputerUseError
from app.agent.permissions import ActionRequest, ApprovalOutcome
from app.ui.services import AppServices
from app.ui.widgets import Card, muted_label


class ComputerUsePage(QWidget):
    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self._build()
        self._refresh_screenshot()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh_screenshot)
        self._timer.setInterval(1500)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self._refresh_screenshot()
        self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(16)
        title = QLabel("Computer Use")
        title.setProperty("variant", "title")
        root.addWidget(title)
        root.addWidget(
            muted_label(
                "画面を確認し、必要な操作だけを明示的に承認して実行します。"
            )
        )
        safety = QLabel("● すべての入力操作は毎回確認されます")
        safety.setProperty("variant", "success")
        root.addWidget(safety)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        left = QVBoxLayout()
        left_widget = QWidget()
        left_widget.setLayout(left)
        preview_card = Card()
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(16, 16, 16, 16)
        preview_header = QHBoxLayout()
        preview_header.addWidget(QLabel("画面プレビュー"))
        preview_header.addStretch(1)
        self.capture_status = QLabel("待機中")
        self.capture_status.setProperty("variant", "muted")
        preview_header.addWidget(self.capture_status)
        preview_layout.addLayout(preview_header)
        self.preview = QLabel("スクリーンショットを取得中…")
        self.preview.setMinimumHeight(300)
        self.preview.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.preview, 1)
        left.addWidget(preview_card, 1)
        splitter.addWidget(left_widget)

        right = QVBoxLayout()
        right_widget = QWidget()
        right_widget.setLayout(right)
        action_card = Card()
        action_layout = QVBoxLayout(action_card)
        action_layout.setContentsMargins(18, 18, 18, 18)
        action_layout.addWidget(QLabel("操作"))
        action_layout.addWidget(muted_label("座標クリックまたはテキスト入力を選択してください。"))
        self.x = QSpinBox()
        self.x.setRange(0, 10000)
        self.y = QSpinBox()
        self.y.setRange(0, 10000)
        coordinates = QGridLayout()
        coordinates.addWidget(QLabel("X"), 0, 0)
        coordinates.addWidget(self.x, 0, 1)
        coordinates.addWidget(QLabel("Y"), 1, 0)
        coordinates.addWidget(self.y, 1, 1)
        self.click_button = QPushButton("承認してクリック")
        self.click_button.setProperty("variant", "primary")
        self.click_button.clicked.connect(self._click)
        coordinates.addWidget(self.click_button, 0, 2, 2, 1)
        action_layout.addLayout(coordinates)
        self.text = QLineEdit()
        self.text.setMaxLength(4000)
        self.text.setPlaceholderText("入力するテキスト")
        self.type_button = QPushButton("承認して入力")
        self.type_button.clicked.connect(self._type)
        type_row = QHBoxLayout()
        type_row.addWidget(self.text, 1)
        type_row.addWidget(self.type_button)
        action_layout.addLayout(type_row)
        right.addWidget(action_card)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("最近の操作"))
        clear = QPushButton("履歴を削除")
        clear.setProperty("variant", "ghost")
        clear.clicked.connect(self._clear_history)
        history_row.addWidget(clear)
        right.addLayout(history_row)
        self.history = QListWidget()
        self.history.setMinimumHeight(150)
        right.addWidget(self.history, 1)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)

    def _approval(self, request: ActionRequest) -> ApprovalOutcome:
        result = QMessageBox.question(
            self,
            "Computer Use の確認",
            f"{request.description}\n\nこの操作を実行しますか？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return ApprovalOutcome("allow_once" if result == QMessageBox.Yes else "reject")

    def _computer(self) -> ComputerUse:
        return ComputerUse(self.services.agent_permissions, self._approval)

    def _click(self) -> None:
        try:
            self._computer().click(self.x.value(), self.y.value())
            self.services.agent_store.add_computer_action(
                action="click", target=f"{self.x.value()},{self.y.value()}",
                approved=True, verified=True,
            )
        except ComputerUseError as exc:
            QMessageBox.warning(self, "Computer Use", str(exc))

    def _type(self) -> None:
        try:
            self._computer().type_text(self.text.text())
            self.services.agent_store.add_computer_action(
                action="type", target=self.text.text()[:120],
                approved=True, verified=True,
            )
        except ComputerUseError as exc:
            QMessageBox.warning(self, "Computer Use", str(exc))

    def _refresh_screenshot(self) -> None:
        screen = self.window().screen()
        if screen is None:
            return
        pixmap: QPixmap = screen.grabWindow(0)
        if pixmap.isNull():
            return
        self.capture_status.setText("ライブ")
        self.preview.setPixmap(pixmap.scaled(
            self.preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        self._refresh_history()

    def _refresh_history(self) -> None:
        self.history.clear()
        try:
            for item in self.services.agent_store.list_computer_actions(limit=20):
                self.history.addItem(
                    f"{item.get('created_at', '')[:19]}  {item.get('action', '')}  "
                    f"{'verified' if item.get('verified') else 'not verified'}"
                )
        except Exception:
            pass

    def _clear_history(self) -> None:
        try:
            self.services.agent_store.clear_computer_actions()
            self.history.clear()
        except Exception:
            pass
