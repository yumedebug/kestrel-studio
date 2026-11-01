"""Long-term memory browser: view, edit and delete memories from the GUI."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.agent.memory import MEMORY_CATEGORIES, MemoryRejected, MemoryStore
from app.ui.i18n import ja
from app.ui.theme import current_palette
from app.ui.widgets import muted_label, section_title


class MemoryEditDialog(QDialog):
    def __init__(
        self,
        memory: MemoryStore,
        record: Any = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.memory = memory
        self.record = record
        self.saved = None
        self.setWindowTitle(ja("記憶を編集" if record else "記憶を追加"))
        self.setModal(True)
        self.setMinimumWidth(560)

        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 16)
        root.setSpacing(10)
        root.addWidget(section_title("記憶を編集" if record else "記憶を追加"))
        root.addWidget(
            muted_label(
                "API キー、パスワード、トークンなどは保存できません。"
                "記憶はエージェントへの参考情報であり、権限設定より優先されません。"
            )
        )

        row = QHBoxLayout()
        row.addWidget(QLabel(ja("分類")))
        self.category = QComboBox()
        for key, label in MEMORY_CATEGORIES.items():
            self.category.addItem(ja(label), key)
        row.addWidget(self.category)
        row.addStretch(1)
        root.addLayout(row)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel(ja("キー")))
        self.key_edit = QLineEdit()
        self.key_edit.setPlaceholderText("例: このプロジェクトのテスト方法")
        key_row.addWidget(self.key_edit, 1)
        root.addLayout(key_row)

        root.addWidget(QLabel(ja("内容")))
        self.value_edit = QPlainTextEdit()
        self.value_edit.setPlaceholderText("覚えておく内容")
        self.value_edit.setFixedHeight(150)
        root.addWidget(self.value_edit)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        root.addWidget(self.message)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        cancel = QPushButton(ja("キャンセル"))
        cancel.clicked.connect(self.reject)
        save = QPushButton(ja("保存"))
        save.setProperty("variant", "primary")
        save.clicked.connect(self._save)
        buttons.addWidget(cancel)
        buttons.addWidget(save)
        root.addLayout(buttons)

        if record is not None:
            index = self.category.findData(record.category)
            if index >= 0:
                self.category.setCurrentIndex(index)
            self.key_edit.setText(record.key)
            self.value_edit.setPlainText(record.value)

    def _save(self) -> None:
        key = self.key_edit.text().strip()
        value = self.value_edit.toPlainText().strip()
        category = str(self.category.currentData())
        try:
            if self.record is None:
                self.saved = self.memory.save(category, key, value, source="user")
            else:
                self.saved = self.memory.update(
                    self.record.id, key=key, value=value, category=category
                )
        except MemoryRejected as exc:
            self.message.setText(ja(str(exc)))
            self.message.setStyleSheet(f"color: {current_palette()['danger']};")
            return
        except Exception as exc:
            self.message.setText(str(exc))
            return
        self.accept()


class MemoryPage(QWidget):
    show_status_message = Signal(str)

    def __init__(self, memory: MemoryStore, parent: QWidget | None = None):
        super().__init__(parent)
        self.memory = memory
        self._records: list[Any] = []
        self._build()
        self.refresh()

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        header = QHBoxLayout()
        header.addWidget(section_title("Memory"))
        header.addStretch(1)
        self.category_filter = QComboBox()
        self.category_filter.addItem(ja("すべて"), "")
        for key, label in MEMORY_CATEGORIES.items():
            self.category_filter.addItem(ja(label), key)
        self.category_filter.currentIndexChanged.connect(self.refresh)
        header.addWidget(self.category_filter)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText(ja("記憶を検索…"))
        self.search_edit.setFixedWidth(220)
        self.search_edit.textChanged.connect(self.refresh)
        header.addWidget(self.search_edit)
        root.addLayout(header)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(
            [ja("分類"), ja("キー"), ja("内容"), ja("更新日時")]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 110)
        self.table.setColumnWidth(1, 200)
        self.table.setColumnWidth(2, 420)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.doubleClicked.connect(lambda _index: self._edit())
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.add_button = QPushButton(ja("追加"))
        self.add_button.clicked.connect(self._add)
        self.edit_button = QPushButton(ja("編集"))
        self.edit_button.clicked.connect(self._edit)
        self.delete_button = QPushButton(ja("削除"))
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self._delete)
        refresh = QPushButton(ja("再読み込み"))
        refresh.clicked.connect(self.refresh)
        actions.addWidget(self.add_button)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.delete_button)
        actions.addWidget(refresh)
        actions.addStretch(1)
        self.count_label = muted_label("")
        self.count_label.setStyleSheet("font-size: 8.5pt;")
        actions.addWidget(self.count_label)
        root.addLayout(actions)

    # -- data --------------------------------------------------------------
    def refresh(self) -> None:
        category = str(self.category_filter.currentData() or "")
        query = self.search_edit.text().strip()
        try:
            if query:
                records = self.memory.search(query, category=category or None, limit=300)
            else:
                records = self.memory.list(category=category or None, limit=300)
        except Exception:
            records = []
        self._records = records
        self.table.setRowCount(0)
        for record in records:
            row = self.table.rowCount()
            self.table.insertRow(row)
            values = [
                MEMORY_CATEGORIES.get(record.category, record.category),
                record.key,
                record.value.replace("\n", " ")[:300],
                record.updated_at[:19].replace("T", " "),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(ja(value) if column == 0 else value)
                if column == 0:
                    item.setData(Qt.UserRole, record.id)
                self.table.setItem(row, column, item)
        self.count_label.setText(f"{len(records)} 件")

    def _selected_record(self) -> Any:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._records):
            return None
        return self._records[row]

    def _add(self) -> None:
        dialog = MemoryEditDialog(self.memory, None, self)
        if dialog.exec() == QDialog.Accepted:
            self.refresh()
            self.show_status_message.emit("記憶を保存しました")

    def _edit(self) -> None:
        record = self._selected_record()
        if record is None:
            self.show_status_message.emit("編集する記憶を選択してください")
            return
        dialog = MemoryEditDialog(self.memory, record, self)
        if dialog.exec() == QDialog.Accepted:
            self.refresh()
            self.show_status_message.emit("記憶を更新しました")

    def _delete(self) -> None:
        record = self._selected_record()
        if record is None:
            self.show_status_message.emit("削除する記憶を選択してください")
            return
        confirm = QMessageBox.question(
            self,
            ja("記憶を削除"),
            ja(f"「{record.key}」を削除しますか？"),
            QMessageBox.Yes | QMessageBox.Cancel,
            QMessageBox.Cancel,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            self.memory.delete(record.id)
        except Exception as exc:
            QMessageBox.critical(self, ja("削除できませんでした"), str(exc))
            return
        self.refresh()
        self.show_status_message.emit("記憶を削除しました")


__all__ = ["MemoryEditDialog", "MemoryPage"]
