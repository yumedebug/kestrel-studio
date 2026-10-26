"""Chat page: conversations, streaming responses, Markdown and SQLite history."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, Signal
from PySide6.QtGui import QAction, QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.core.database import Chat, Database, DatabaseError, Message
from app.core.types import GenerationParams, ModelTarget
from app.ui.dialogs import confirm, show_error
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import (
    EmptyState,
    MessageBubble,
    make_separator,
    muted_label,
    section_title,
    set_variant,
)
from app.workers.workers import ChatWorker

log = logging.getLogger(__name__)


class GenerationDialog(QDialog):
    def __init__(
        self,
        params: GenerationParams,
        system_prompt: str,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Generation settings")
        self.setMinimumWidth(540)
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(section_title("Generation parameters"))
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(12)

        self.temperature = QDoubleSpinBox()
        self.temperature.setRange(0.0, 2.0)
        self.temperature.setSingleStep(0.05)
        self.temperature.setDecimals(2)
        self.temperature.setValue(params.temperature)
        form.addRow("Temperature", self.temperature)

        self.top_p = QDoubleSpinBox()
        self.top_p.setRange(0.0, 1.0)
        self.top_p.setSingleStep(0.01)
        self.top_p.setDecimals(2)
        self.top_p.setValue(params.top_p)
        form.addRow("Top P", self.top_p)

        self.top_k = QSpinBox()
        self.top_k.setRange(0, 1000)
        self.top_k.setValue(params.top_k)
        form.addRow("Top K", self.top_k)

        self.repeat_penalty = QDoubleSpinBox()
        self.repeat_penalty.setRange(0.0, 4.0)
        self.repeat_penalty.setSingleStep(0.05)
        self.repeat_penalty.setDecimals(2)
        self.repeat_penalty.setValue(params.repeat_penalty)
        form.addRow("Repeat Penalty", self.repeat_penalty)

        self.max_tokens = QSpinBox()
        self.max_tokens.setRange(1, 131072)
        self.max_tokens.setSingleStep(64)
        self.max_tokens.setValue(params.max_tokens)
        form.addRow("Max Tokens", self.max_tokens)

        self.seed = QSpinBox()
        self.seed.setRange(-1, 2_147_483_647)
        self.seed.setSpecialValueText("Random")
        self.seed.setValue(params.seed)
        form.addRow("Seed", self.seed)
        layout.addLayout(form)

        layout.addWidget(section_title("System prompt"))
        self.system_prompt = QPlainTextEdit()
        self.system_prompt.setPlainText(system_prompt)
        self.system_prompt.setPlaceholderText("Instructions sent before the conversation…")
        self.system_prompt.setFixedHeight(100)
        layout.addWidget(self.system_prompt)

        hint = muted_label(
            "Top K and Repeat Penalty are always honored by llama.cpp. Cloud providers "
            "receive only the parameters their API supports. Parameters are saved with "
            "the current chat."
        )
        hint.setStyleSheet("font-size: 9pt;")
        layout.addWidget(hint)

        buttons = QDialogButtonBox()
        reset = QPushButton("Reset")
        reset.clicked.connect(self._reset)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Apply")
        save.setProperty("variant", "primary")
        save.clicked.connect(self.accept)
        buttons.addButton(reset, QDialogButtonBox.ResetRole)
        buttons.addButton(cancel, QDialogButtonBox.RejectRole)
        buttons.addButton(save, QDialogButtonBox.AcceptRole)
        layout.addWidget(buttons)
        self._initial = (params, system_prompt)

    def _reset(self) -> None:
        params, system_prompt = self._initial
        self.temperature.setValue(params.temperature)
        self.top_p.setValue(params.top_p)
        self.top_k.setValue(params.top_k)
        self.repeat_penalty.setValue(params.repeat_penalty)
        self.max_tokens.setValue(params.max_tokens)
        self.seed.setValue(params.seed)
        self.system_prompt.setPlainText(system_prompt)

    def values(self) -> tuple[GenerationParams, str]:
        return (
            GenerationParams(
                temperature=self.temperature.value(),
                top_p=self.top_p.value(),
                top_k=self.top_k.value(),
                repeat_penalty=self.repeat_penalty.value(),
                max_tokens=self.max_tokens.value(),
                seed=self.seed.value(),
            ),
            self.system_prompt.toPlainText().strip(),
        )


class PromptEdit(QPlainTextEdit):
    send_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setPlaceholderText(
            "Message the model…  Enter to send, Shift+Enter for a new line"
        )
        self.setFixedHeight(76)
        self.setTabChangesFocus(True)
        self._max_height = 170
        self.document().contentsChanged.connect(self._adjust_height)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                return
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def _adjust_height(self) -> None:
        doc_height = self.document().size().height()
        margins = self.contentsMargins().top() + self.contentsMargins().bottom() + 22
        desired = int(doc_height + margins)
        self.setFixedHeight(max(76, min(desired, self._max_height)))


class ChatPage(QWidget):
    cloud_requested = Signal()
    models_page_requested = Signal()
    generation_active = Signal(bool)
    show_status_message = Signal(str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.db: Database = services.database
        self._current_chat_id: str | None = None
        self._current_chat: Chat | None = None
        self._messages: list[Message] = []
        self._chat_active = False
        self._worker: ChatWorker | None = None
        self._assistant_bubble: MessageBubble | None = None
        self._stream_text = ""
        self._stream_model_label = ""
        self._stream_chat_id: str | None = None
        self._stream_started_at = 0.0
        self._status: dict[str, Any] = {}
        self._pending_params = GenerationParams.from_settings(
            services.settings.data.get("chat", {})
        )
        self._pending_system = str(services.settings.get("chat.system_prompt", ""))
        self._generating = False
        self._pending_update = False
        self._draft_hint: QLabel | None = None
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(70)
        self._update_timer.timeout.connect(self._flush_stream_text)
        self._build()
        self.refresh_conversations()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        self.setObjectName("chatPage")
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Conversation sidebar (inside the page).
        conversations = QFrame()
        conversations.setFixedWidth(238)
        conversations.setStyleSheet("QFrame { background: transparent; }")
        conv_layout = QVBoxLayout(conversations)
        conv_layout.setContentsMargins(14, 16, 10, 14)
        conv_layout.setSpacing(10)
        header = QHBoxLayout()
        header.addWidget(section_title("Chats"))
        header.addStretch(1)
        self.new_chat_button = QPushButton("New Chat")
        self.new_chat_button.setProperty("variant", "chip")
        self.new_chat_button.setCursor(Qt.PointingHandCursor)
        self.new_chat_button.clicked.connect(self.new_chat)
        header.addWidget(self.new_chat_button)
        conv_layout.addLayout(header)

        self.chat_list = QListWidget()
        self.chat_list.setSpacing(2)
        self.chat_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self._chat_context_menu)
        self.chat_list.currentItemChanged.connect(self._conversation_selected)
        conv_layout.addWidget(self.chat_list, 1)

        self.history_info = muted_label("History stays on this PC.")
        self.history_info.setStyleSheet("font-size: 8.5pt;")
        conv_layout.addWidget(self.history_info)
        splitter.addWidget(conversations)

        # Main chat panel.
        main = QWidget()
        main_layout = QVBoxLayout(main)
        main_layout.setContentsMargins(18, 12, 18, 12)
        main_layout.setSpacing(14)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(10)
        self.title_label = QLabel("Chat")
        self.title_label.setObjectName("sectionTitle")
        toolbar.addWidget(self.title_label)
        self.model_badge = QLabel("")
        self.model_badge.setProperty("variant", "faint")
        self.model_badge.setStyleSheet("font-size: 8.5pt;")
        toolbar.addWidget(self.model_badge)
        toolbar.addStretch(1)
        self.params_button = QPushButton("Parameters")
        self.params_button.setProperty("variant", "ghost")
        self.params_button.setCursor(Qt.PointingHandCursor)
        self.params_button.clicked.connect(self._edit_params)
        toolbar.addWidget(self.params_button)
        self.delete_chat_button = QPushButton("Delete")
        self.delete_chat_button.setProperty("variant", "ghost")
        self.delete_chat_button.setCursor(Qt.PointingHandCursor)
        self.delete_chat_button.clicked.connect(self._delete_current_chat)
        toolbar.addWidget(self.delete_chat_button)
        main_layout.addLayout(toolbar)
        main_layout.addWidget(make_separator())

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_welcome())
        self.stack.addWidget(self._build_chat_view())
        main_layout.addWidget(self.stack, 1)

        main_layout.addWidget(self._build_composer())
        splitter.addWidget(main)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([238, 900])
        outer.addWidget(splitter)

    def _build_welcome(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)
        self.welcome = EmptyState(
            "chat",
            "Start a conversation",
            "Load a local GGUF model or connect a cloud provider, then send your first "
            "message. Conversations are stored in a local SQLite database.",
            "New Chat",
        )
        self.welcome.action.clicked.connect(self.new_chat)
        layout.addWidget(self.welcome)
        return page

    def _build_chat_view(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(6, 10, 12, 14)
        self.messages_layout.setSpacing(10)
        self.messages_layout.addStretch(1)
        self.scroll.setWidget(self.messages_container)
        layout.addWidget(self.scroll, 1)

        self.error_bar = QFrame()
        self.error_bar.setStyleSheet(
            f"QFrame {{ background: rgba(239, 91, 107, 0.10); "
            f"border: 1px solid {current_palette()['danger']}; border-radius: 8px; }}"
        )
        error_layout = QHBoxLayout(self.error_bar)
        error_layout.setContentsMargins(10, 7, 8, 7)
        self.error_label = QLabel("")
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet(
            f"color: {current_palette()['danger']}; background: transparent;"
        )
        error_layout.addWidget(self.error_label, 1)
        close = QPushButton("×")
        close.setProperty("variant", "ghost")
        close.setFixedSize(24, 24)
        close.clicked.connect(self.error_bar.hide)
        error_layout.addWidget(close)
        self.error_bar.hide()
        layout.addWidget(self.error_bar)
        return page

    def _build_composer(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        self.model_hint = QHBoxLayout()
        self.model_hint.setContentsMargins(2, 0, 2, 0)
        self.model_hint_label = muted_label("No model loaded")
        self.model_hint_label.setStyleSheet("font-size: 8.5pt;")
        self.model_hint.addWidget(self.model_hint_label)
        self.choose_cloud_button = QPushButton("Cloud model")
        self.choose_cloud_button.setProperty("variant", "ghost")
        self.choose_cloud_button.setStyleSheet("font-size: 8.5pt;")
        self.choose_cloud_button.setCursor(Qt.PointingHandCursor)
        self.choose_cloud_button.clicked.connect(self.cloud_requested.emit)
        self.model_hint.addWidget(self.choose_cloud_button)
        self.choose_local_button = QPushButton("My Models")
        self.choose_local_button.setProperty("variant", "ghost")
        self.choose_local_button.setStyleSheet("font-size: 8.5pt;")
        self.choose_local_button.setCursor(Qt.PointingHandCursor)
        self.choose_local_button.clicked.connect(self.models_page_requested.emit)
        self.model_hint.addWidget(self.choose_local_button)
        self.model_hint.addStretch(1)
        self.generation_info = QLabel("")
        self.generation_info.setProperty("variant", "faint")
        self.generation_info.setStyleSheet("font-size: 8.5pt;")
        self.model_hint.addWidget(self.generation_info)
        layout.addLayout(self.model_hint)

        composer = QFrame()
        composer.setProperty("card", "true")
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(12, 9, 10, 8)
        composer_layout.setSpacing(10)
        self.prompt = PromptEdit()
        self.prompt.send_requested.connect(self.send_message)
        composer_layout.addWidget(self.prompt)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.system_button = QPushButton("System prompt")
        self.system_button.setProperty("variant", "ghost")
        self.system_button.setCursor(Qt.PointingHandCursor)
        self.system_button.clicked.connect(self._edit_params)
        actions.addWidget(self.system_button)
        self.new_chat_shortcut = QShortcut(QKeySequence("Ctrl+N"), self)
        self.new_chat_shortcut.activated.connect(self.new_chat)
        actions.addStretch(1)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.hide()
        actions.addWidget(self.stop_button)
        self.send_button = QPushButton("Send")
        self.send_button.setProperty("variant", "primary")
        self.send_button.setCursor(Qt.PointingHandCursor)
        self.send_button.clicked.connect(self.send_message)
        actions.addWidget(self.send_button)
        composer_layout.addLayout(actions)
        layout.addWidget(composer)
        return wrapper

    # -- model / status ----------------------------------------------------
    def set_inference_status(self, status: dict[str, Any]) -> None:
        self._status = status
        state = status.get("state", "idle")
        ready = state == "ready"
        self.send_button.setEnabled(ready and not self._generating)
        self.choose_cloud_button.setVisible(not ready)
        self.choose_local_button.setVisible(not ready)
        if ready:
            name = status.get("display_name") or status.get("model_id") or "Model"
            kind = "Local" if status.get("kind") == "local" else "Cloud"
            self.model_hint_label.setText(f"{kind} · {name}")
            self.model_hint_label.setStyleSheet(
                f"color: {current_palette()['text_muted']}; font-size: 8.5pt;"
            )
            self.model_badge.setText(name)
            self.prompt.setPlaceholderText(
                "Message the model…  Enter to send, Shift+Enter for a new line"
            )
        elif state == "loading":
            self.model_hint_label.setText("Loading model…")
            self.model_hint_label.setStyleSheet(
                f"color: {current_palette()['warning']}; font-size: 8.5pt;"
            )
            self.prompt.setPlaceholderText("The model is loading…")
        elif state == "error":
            self.model_hint_label.setText("Model load failed")
            self.model_hint_label.setStyleSheet(
                f"color: {current_palette()['danger']}; font-size: 8.5pt;"
            )
            self.prompt.setPlaceholderText("Load a model to start chatting…")
        else:
            self.model_hint_label.setText("No model loaded")
            self.model_hint_label.setStyleSheet(
                f"color: {current_palette()['text_muted']}; font-size: 8.5pt;"
            )
            self.prompt.setPlaceholderText("Load a model to start chatting…")
        self._apply_generation_controls()

    def _apply_generation_controls(self) -> None:
        busy = self._generating
        self.stop_button.setVisible(busy)
        self.send_button.setVisible(not busy)
        self.prompt.setEnabled(not busy)
        self.new_chat_button.setEnabled(not busy)
        self.chat_list.setEnabled(not busy)
        self.delete_chat_button.setEnabled(not busy and bool(self._current_chat_id))
        self.params_button.setEnabled(not busy)
        if not busy:
            self.stop_button.setEnabled(True)
            self.stop_button.setText("Stop")
            self.send_button.setEnabled(self._status.get("state") == "ready")

    def _show_error(self, message: str, detail: str = "") -> None:
        self.error_label.setText(message)
        self.error_label.setToolTip(detail)
        self.error_bar.show()
        self.show_status_message.emit(message)

    def _hide_error(self) -> None:
        self.error_bar.hide()

    # -- conversations -----------------------------------------------------
    def refresh_conversations(self) -> None:
        try:
            chats = self.db.list_chats()
        except DatabaseError as exc:
            self._show_error(f"Could not read chat history: {exc}")
            return
        blocked = self.chat_list.blockSignals(True)
        self.chat_list.clear()
        selected_row = -1
        for row, chat in enumerate(chats):
            item = QListWidgetItem(chat.title or "Untitled chat")
            item.setData(Qt.UserRole, chat.id)
            item.setToolTip(f"{chat.model or 'No model'}\n{_friendly_date(chat.updated_at)}")
            self.chat_list.addItem(item)
            if chat.id == self._current_chat_id:
                selected_row = row
        self.chat_list.blockSignals(blocked)
        if selected_row >= 0:
            self.chat_list.setCurrentRow(selected_row)
        if self._current_chat_id:
            self.delete_chat_button.setEnabled(not self._generating)
            self.params_button.setEnabled(not self._generating)
        else:
            self.delete_chat_button.setEnabled(False)

    def _conversation_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        if self._generating:
            return
        chat_id = current.data(Qt.UserRole)
        if chat_id == self._current_chat_id and self._chat_active:
            return
        self.load_chat(str(chat_id))

    def load_chat(self, chat_id: str) -> None:
        chat = self.db.get_chat(chat_id)
        if chat is None:
            self.refresh_conversations()
            return
        self._current_chat_id = chat.id
        self._current_chat = chat
        self._chat_active = True
        self._messages = self.db.list_messages(chat.id)
        self._pending_params = GenerationParams.from_settings(chat.params or {})
        self._pending_system = chat.system_prompt or ""
        self.title_label.setText(chat.title)
        self.model_badge.setText(chat.model)
        self._render_messages(chat)
        self.stack.setCurrentIndex(1)
        self._hide_error()
        self.refresh_conversations()
        self._scroll_to_bottom()
        self.prompt.setFocus()

    def _render_messages(self, chat: Chat) -> None:
        self._clear_messages()
        for message in self._messages:
            self._add_bubble(
                MessageBubble(
                    message.role,
                    message.content,
                    model_label=str(message.meta.get("model") or chat.model),
                    timestamp=message.created_at,
                ),
                scroll=False,
                record=False,
            )
        if not self._messages:
            hint = QLabel(
                "New conversation. Pick a model in the header, then say hello."
            )
            hint.setProperty("variant", "faint")
            hint.setAlignment(Qt.AlignCenter)
            hint.setContentsMargins(0, 24, 0, 0)
            self.messages_layout.insertWidget(self.messages_layout.count() - 1, hint)
            self._draft_hint = hint

    def _clear_messages(self) -> None:
        self._draft_hint = None
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._assistant_bubble = None

    def _add_bubble(
        self,
        bubble: MessageBubble,
        *,
        scroll: bool = True,
        record: bool = True,
        user: bool | None = None,
    ) -> MessageBubble:
        if self._draft_hint is not None:
            self._draft_hint.deleteLater()
            self._draft_hint = None
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        is_user = bubble.role == "user" if user is None else user
        if is_user:
            row_layout.addStretch(1)
            row_layout.addWidget(bubble)
        else:
            row_layout.addWidget(bubble)
            row_layout.addStretch(1)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        bubble._row_widget = row  # type: ignore[attr-defined]
        if record:
            self._messages.append(
                Message(
                    id="transient",
                    chat_id=self._current_chat_id or "",
                    role=bubble.role,
                    content="",
                )
            )
        if scroll:
            QTimer.singleShot(20, self._scroll_to_bottom)
        return bubble

    def _remove_bubble(self, bubble: MessageBubble) -> None:
        row = getattr(bubble, "_row_widget", None)
        if row is not None:
            self.messages_layout.removeWidget(row)
            row.deleteLater()
        else:
            bubble.deleteLater()
        if self._assistant_bubble is bubble:
            self._assistant_bubble = None

    def _scroll_to_bottom(self) -> None:
        bar = self.scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _near_bottom(self) -> bool:
        bar = self.scroll.verticalScrollBar()
        return bar.value() >= bar.maximum() - 90

    def new_chat(self) -> None:
        if self._generating:
            return
        self._current_chat_id = None
        self._current_chat = None
        self._chat_active = True
        self._messages = []
        self._pending_params = GenerationParams.from_settings(
            self.services.settings.data.get("chat", {})
        )
        self._pending_system = str(self.services.settings.get("chat.system_prompt", ""))
        self._clear_messages()
        self.title_label.setText("New chat")
        self.model_badge.setText(
            self._status.get("display_name", "") if self._status.get("state") == "ready" else ""
        )
        self.stack.setCurrentIndex(1)
        self.chat_list.clearSelection()
        self.delete_chat_button.setEnabled(False)
        self._hide_error()
        hint = QLabel("New conversation. Your messages will be saved locally.")
        hint.setProperty("variant", "faint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setContentsMargins(0, 24, 0, 0)
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, hint)
        self._draft_hint = hint
        self.prompt.setFocus()

    def _chat_context_menu(self, pos) -> None:
        if self._generating:
            return
        item = self.chat_list.itemAt(pos)
        if item is None:
            return
        chat_id = str(item.data(Qt.UserRole))
        menu = QMenu(self)
        open_action = QAction("Open", self)
        open_action.triggered.connect(lambda: self.load_chat(chat_id))
        rename_action = QAction("Rename…", self)
        rename_action.triggered.connect(lambda: self._rename_chat(chat_id))
        delete_action = QAction("Delete chat…", self)
        delete_action.triggered.connect(lambda: self._delete_chat(chat_id))
        menu.addAction(open_action)
        menu.addAction(rename_action)
        menu.addSeparator()
        menu.addAction(delete_action)
        menu.exec(self.chat_list.mapToGlobal(pos))

    def _rename_chat(self, chat_id: str) -> None:
        chat = self.db.get_chat(chat_id)
        if chat is None:
            return
        title, ok = QInputDialog.getText(
            self, "Rename chat", "Title:", text=chat.title
        )
        if not ok or not title.strip():
            return
        self.db.update_chat(chat_id, title=title.strip()[:120])
        if chat_id == self._current_chat_id:
            self.title_label.setText(title.strip()[:120])
        self.refresh_conversations()

    def _delete_current_chat(self) -> None:
        if self._current_chat_id:
            self._delete_chat(self._current_chat_id)

    def _delete_chat(self, chat_id: str) -> None:
        if self._generating:
            return
        chat = self.db.get_chat(chat_id)
        if chat is None:
            return
        if not confirm(
            self,
            "Delete chat",
            f"Delete “{chat.title}”? This removes the conversation from the local "
            "database. It cannot be undone.",
        ):
            return
        self.db.delete_chat(chat_id)
        if chat_id == self._current_chat_id:
            self._current_chat_id = None
            self._current_chat = None
            self._chat_active = False
            self._clear_messages()
            self.stack.setCurrentIndex(0)
            self.title_label.setText("Chat")
            self.model_badge.setText("")
        self.refresh_conversations()
        self.show_status_message.emit("Chat deleted")

    # -- parameters --------------------------------------------------------
    def _edit_params(self) -> None:
        dialog = GenerationDialog(self._pending_params, self._pending_system, self)
        if dialog.exec() != QDialog.Accepted:
            return
        params, system_prompt = dialog.values()
        self._pending_params = params
        self._pending_system = system_prompt
        if self._current_chat_id:
            self.db.update_chat(
                self._current_chat_id, params=params.to_dict(), system_prompt=system_prompt
            )
            self._current_chat = self.db.get_chat(self._current_chat_id)
        self.show_status_message.emit("Generation settings saved for this chat")

    # -- generation --------------------------------------------------------
    def send_message(self) -> None:
        if self._generating:
            return
        text = self.prompt.toPlainText().strip()
        if not text:
            return
        status = self._status
        if status.get("state") != "ready":
            self._show_error(
                "No model is loaded. Load a GGUF model in My Models or choose a cloud "
                "model before sending a message."
            )
            return
        self._hide_error()
        params = self._pending_params
        system_prompt = self._pending_system
        target_name = status.get("display_name") or status.get("model_id") or "Model"

        if not self._chat_active:
            self.new_chat()
        try:
            if self._current_chat_id is None:
                title = _title_from(text)
                chat = self.db.create_chat(
                    title=title,
                    model=target_name,
                    provider=status.get("provider") or status.get("kind") or "",
                    params=params.to_dict(),
                    system_prompt=system_prompt,
                )
                self._current_chat_id = chat.id
                self._current_chat = chat
                self._messages = []
                self.title_label.setText(chat.title)
            else:
                self.db.update_chat(
                    self._current_chat_id,
                    model=target_name,
                    provider=status.get("provider") or status.get("kind") or "",
                    params=params.to_dict(),
                    system_prompt=system_prompt,
                )
            user_message = self.db.add_message(self._current_chat_id, "user", text)
        except DatabaseError as exc:
            self._show_error(f"Could not save the message: {exc}")
            return

        self._messages.append(user_message)
        self.prompt.clear()
        self._add_bubble(
            MessageBubble("user", text, timestamp=user_message.created_at),
            record=False,
        )
        self._hide_error()
        self._start_generation(
            chat_id=self._current_chat_id,
            system_prompt=system_prompt,
            params=params,
            target_name=target_name,
        )

    def _start_generation(
        self,
        *,
        chat_id: str,
        system_prompt: str,
        params: GenerationParams,
        target_name: str,
    ) -> None:
        history: list[dict[str, str]] = []
        try:
            for message in self.db.list_messages(chat_id):
                if message.role in {"user", "assistant", "system"} and message.content:
                    history.append({"role": message.role, "content": message.content})
        except DatabaseError as exc:
            self._show_error(f"Could not read the conversation: {exc}")
            return
        self._stream_text = ""
        self._stream_chat_id = chat_id
        self._stream_model_label = target_name
        self._stream_started_at = time.monotonic()
        self._assistant_bubble = self._add_bubble(
            MessageBubble(
                "assistant",
                "",
                model_label=target_name,
                streaming=True,
            ),
            record=False,
        )
        self._generating = True
        self._apply_generation_controls()
        self.generation_active.emit(True)
        self.generation_info.setText("Generating…")
        self._update_timer.start()

        worker = ChatWorker(
            self.services.inference,
            history,
            params,
            system_prompt=system_prompt,
            model_label=target_name,
            parent=self,
        )
        worker.delta.connect(self._on_delta)
        worker.completed.connect(self._on_completed)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()
        self._scroll_to_bottom()

    def _on_delta(self, chunk: str) -> None:
        self._stream_text += chunk
        self._pending_update = True

    def _flush_stream_text(self) -> None:
        if not self._pending_update or self._assistant_bubble is None:
            return
        self._pending_update = False
        was_near_bottom = self._near_bottom()
        self._assistant_bubble.set_streaming_text(self._stream_text)
        self.generation_info.setText(
            f"Generating… {len(self._stream_text):,} characters"
        )
        if was_near_bottom:
            QTimer.singleShot(10, self._scroll_to_bottom)

    def _finish_generation(self) -> None:
        self._update_timer.stop()
        self._flush_stream_text()
        self._generating = False
        self._apply_generation_controls()
        self.generation_active.emit(False)
        self._worker = None

    def _on_completed(self, payload: dict[str, Any]) -> None:
        content = str(payload.get("content") or "")
        usage = payload.get("usage")
        cancelled = bool(payload.get("cancelled"))
        finish_reason = payload.get("finish_reason") or ("cancelled" if cancelled else "stop")
        chat_id = self._stream_chat_id
        model_label = payload.get("model_label") or self._stream_model_label
        elapsed = max(0.001, time.monotonic() - self._stream_started_at)

        if self._assistant_bubble is not None:
            if content.strip():
                self._assistant_bubble.finalize(content)
            else:
                self._remove_bubble(self._assistant_bubble)
            self._assistant_bubble = None

        if chat_id and content.strip():
            try:
                meta = {
                    "model": model_label,
                    "finish_reason": finish_reason,
                    "cancelled": cancelled,
                    "usage": usage,
                }
                self.db.add_message(chat_id, "assistant", content, meta)
            except DatabaseError as exc:
                self._show_error(f"Could not save the response: {exc}")

        if usage:
            tokens = usage.get("completion_tokens")
            info = f"{tokens:,} tokens" if isinstance(tokens, int) else ""
        else:
            info = ""
        speed = len(content) / elapsed
        suffix = " · stopped" if cancelled else ""
        self.generation_info.setText(
            f"{info + ' · ' if info else ''}{speed:,.0f} chars/s{suffix}"
        )
        if not content.strip():
            self.generation_info.setText("Stopped" if cancelled else "The model returned an empty response.")
        self._finish_generation()
        if chat_id == self._current_chat_id:
            self.refresh_conversations()
            self._scroll_to_bottom()
        else:
            self.refresh_conversations()

    def _on_failed(self, message: str, detail: str) -> None:
        if self._assistant_bubble is not None:
            self._remove_bubble(self._assistant_bubble)
            self._assistant_bubble = None
        self._show_error(message, detail)
        self.generation_info.setText("Generation failed")
        self._finish_generation()

    def stop_generation(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping…")
            self.generation_info.setText("Stopping…")

    # -- lifecycle ---------------------------------------------------------
    def shutdown(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(8000)
        self._update_timer.stop()


def _title_from(text: str) -> str:
    clean = " ".join(text.split())
    if len(clean) > 60:
        clean = clean[:57].rstrip() + "…"
    return clean or "New chat"


def _friendly_date(value: str) -> str:
    try:
        dt = datetime.fromisoformat(value).astimezone()
    except (ValueError, TypeError):
        return ""
    today = datetime.now(timezone.utc).astimezone().date()
    if dt.date() == today:
        return f"Today {dt:%H:%M}"
    from datetime import timedelta

    if dt.date() == today - timedelta(days=1):
        return "Yesterday"
    return dt.strftime("%Y-%m-%d")
