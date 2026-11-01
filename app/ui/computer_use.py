"""User-facing Computer Use controls with an explicit approval boundary."""

from __future__ import annotations

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
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
        self._timer.start(1000)

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 14, 20, 14)
        root.setSpacing(12)
        title = QLabel("Computer Use")
        title.setObjectName("sectionTitle")
        root.addWidget(title)
        root.addWidget(
            muted_label(
                "画面確認と限定的な入力を提供します。すべての入力操作は毎回確認が必要です。"
            )
        )

        card = Card()
        form = QFormLayout(card)
        self.x = QSpinBox()
        self.x.setRange(0, 10000)
        self.y = QSpinBox()
        self.y.setRange(0, 10000)
        coordinates = QHBoxLayout()
        coordinates.addWidget(QLabel("X"))
        coordinates.addWidget(self.x)
        coordinates.addWidget(QLabel("Y"))
        coordinates.addWidget(self.y)
        self.click_button = QPushButton("承認してクリック")
        self.click_button.clicked.connect(self._click)
        coordinates.addWidget(self.click_button)
        form.addRow("ポインター", coordinates)
        self.text = QLineEdit()
        self.text.setMaxLength(4000)
        self.type_button = QPushButton("承認して入力")
        self.type_button.clicked.connect(self._type)
        type_row = QHBoxLayout()
        type_row.addWidget(self.text, 1)
        type_row.addWidget(self.type_button)
        form.addRow("テキスト", type_row)
        root.addWidget(card)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("Computer Use 履歴"))
        clear = QPushButton("履歴を削除")
        clear.clicked.connect(self._clear_history)
        history_row.addWidget(clear)
        root.addLayout(history_row)
        self.history = QListWidget()
        self.history.setMaximumHeight(130)
        root.addWidget(self.history)

        self.preview = QLabel("スクリーンショットを取得中…")
        self.preview.setMinimumHeight(300)
        self.preview.setAlignment(Qt.AlignCenter)
        root.addWidget(self.preview, 1)

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
