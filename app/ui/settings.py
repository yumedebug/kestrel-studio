"""Settings page: general, inference, server, storage and cloud credentials."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.constants import APP_NAME, APP_VERSION
from app.core.cloud_models import fetch_models
from app.core.providers import (
    KIND_LABELS,
    PROVIDERS,
    PROVIDER_ORDER,
    credential_key,
    get_provider,
    provider_models,
)
from app.core.settings import DEFAULT_SYSTEM_PROMPT
from app.ui.dialogs import confirm, open_url, show_error
from app.ui.i18n import localize_widget_tree
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import Card, muted_label, section_title
from app.workers.workers import TaskWorker

log = logging.getLogger(__name__)


class SettingsPage(QWidget):
    settings_saved = Signal()
    cloud_changed = Signal()
    history_cleared = Signal()
    show_status_message = Signal(str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.settings = services.settings
        self._workers: list[TaskWorker] = []
        self._cloud_worker: TaskWorker | None = None
        self._current_provider = "openai"
        self._build()
        self._load_values()
        self._load_provider(self._current_provider)

    # -- construction helpers ----------------------------------------------
    def _section(self, title: str, description: str) -> tuple[Card, QFormLayout]:
        card = Card()
        outer = QVBoxLayout(card)
        outer.setContentsMargins(22, 20, 22, 20)
        outer.setSpacing(14)
        outer.addWidget(section_title(title))
        if description:
            desc = muted_label(description)
            desc.setStyleSheet("font-size: 8.5pt;")
            outer.addWidget(desc)
        form = QFormLayout()
        form.setHorizontalSpacing(24)
        form.setVerticalSpacing(13)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        outer.addLayout(form)
        return card, form

    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 16, 20, 10)
        header_layout.setSpacing(10)
        header_layout.addWidget(section_title("Settings"))
        header_layout.addStretch(1)
        self.saved_label = muted_label("")
        self.saved_label.setStyleSheet(f"font-size: 8.5pt; color: {current_palette()['success']};")
        header_layout.addWidget(self.saved_label)
        self.save_button = QPushButton("Save settings")
        self.save_button.setProperty("variant", "primary")
        self.save_button.setCursor(Qt.PointingHandCursor)
        self.save_button.clicked.connect(self.save)
        header_layout.addWidget(self.save_button)
        root.addWidget(header)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        self.scroll = scroll
        self._cards: dict[str, QWidget] = {}
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(24, 8, 24, 24)
        self.content_layout.setSpacing(18)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        # General ----------------------------------------------------------
        card, form = self._section(
            "General",
            "Appearance, where GGUF files are stored, and what happens at launch.",
        )
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("Dark", "dark")
        self.theme_combo.addItem("Light", "light")
        form.addRow("Theme", self.theme_combo)
        self.font_size = QSpinBox()
        self.font_size.setRange(8, 24)
        self.font_size.setSuffix(" pt")
        form.addRow("UI font size", self.font_size)
        dir_row = QWidget()
        dir_layout = QHBoxLayout(dir_row)
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_layout.setSpacing(6)
        self.model_dir = QLineEdit()
        self.model_dir.setPlaceholderText("Folder for downloaded and scanned GGUF models")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._browse_model_dir)
        dir_layout.addWidget(self.model_dir, 1)
        dir_layout.addWidget(browse)
        form.addRow("Model Directory", dir_row)
        self.start_behavior = QComboBox()
        self.start_behavior.addItem("Show chat", "chat")
        self.start_behavior.addItem("Resume last conversation", "resume_last_chat")
        self.start_behavior.addItem("Load last model", "load_last_model")
        form.addRow("Start Behavior", self.start_behavior)
        self.content_layout.addWidget(card)
        self._cards["general"] = card

        # Inference --------------------------------------------------------
        card, form = self._section(
            "Inference",
            "Defaults for llama.cpp. Each model can override these values when loaded.",
        )
        self.default_ctx = QSpinBox()
        self.default_ctx.setRange(128, 1_048_576)
        self.default_ctx.setSingleStep(512)
        self.default_ctx.setSuffix(" tokens")
        form.addRow("Default Context Length", self.default_ctx)
        self.cpu_threads = QSpinBox()
        self.cpu_threads.setRange(0, 256)
        self.cpu_threads.setSpecialValueText("Auto")
        form.addRow("CPU Threads", self.cpu_threads)
        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 999)
        form.addRow("GPU Layers", self.gpu_layers)
        self.batch_size = QSpinBox()
        self.batch_size.setRange(32, 8192)
        self.batch_size.setSingleStep(64)
        form.addRow("Batch Size", self.batch_size)
        self.flash_attention = QCheckBox("Enable flash attention when supported by llama.cpp")
        form.addRow("", self.flash_attention)
        self.backend_combo = QComboBox()
        self.backend_combo.addItem("Auto (CUDA → Vulkan → CPU)", "auto")
        self.backend_combo.addItem("CPU", "cpu")
        self.backend_combo.addItem("CUDA (NVIDIA)", "cuda")
        self.backend_combo.addItem("Vulkan", "vulkan")
        form.addRow("Backend Selection", self.backend_combo)
        self.auto_fallback = QCheckBox("Fall back to CPU if CUDA cannot initialize")
        form.addRow("", self.auto_fallback)
        exe_row = QWidget()
        exe_layout = QHBoxLayout(exe_row)
        exe_layout.setContentsMargins(0, 0, 0, 0)
        exe_layout.setSpacing(6)
        self.llama_path = QLineEdit()
        self.llama_path.setPlaceholderText("Auto-detect runtime\\llama-server.exe")
        browse_exe = QPushButton("Browse…")
        browse_exe.clicked.connect(self._browse_llama)
        exe_layout.addWidget(self.llama_path, 1)
        exe_layout.addWidget(browse_exe)
        form.addRow("llama-server Path", exe_row)
        hint = muted_label(
            "Install the engine with scripts\\fetch_llama_cpp.py. GPU layers are passed to "
            "llama-server; 0 = CPU inference."
        )
        hint.setStyleSheet("font-size: 8.5pt;")
        form.addRow("", hint)
        self.content_layout.addWidget(card)
        self._cards["inference"] = card

        # Server -----------------------------------------------------------
        card, form = self._section(
            "Server",
            "The OpenAI-compatible API binds to loopback by default.",
        )
        self.server_host = QComboBox()
        self.server_host.setEditable(True)
        self.server_host.addItems(["127.0.0.1", "localhost", "0.0.0.0"])
        form.addRow("Host", self.server_host)
        self.server_port = QSpinBox()
        self.server_port.setRange(1024, 65535)
        form.addRow("Port", self.server_port)
        self.auto_start = QCheckBox("Start the server when the app launches")
        form.addRow("", self.auto_start)
        self.allow_external = QCheckBox("Allow external network access (no authentication)")
        self.allow_external.toggled.connect(self._external_toggled)
        form.addRow("", self.allow_external)
        external_hint = muted_label(
            "When external access is enabled, any machine that can reach this port can use "
            "your model. There is no API key on the server. A firewall may still block it."
        )
        external_hint.setStyleSheet(f"font-size: 8.5pt; color: {current_palette()['warning']};")
        form.addRow("", external_hint)
        self.content_layout.addWidget(card)
        self._cards["server"] = card

        # Agent ------------------------------------------------------------
        self.content_layout.addWidget(self._build_agent_card())

        # Storage ----------------------------------------------------------
        card, form = self._section(
            "Storage",
            "Chat history is a local SQLite database. Nothing is uploaded.",
        )
        self.storage_info = muted_label("")
        self.storage_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Chat database", self.storage_info)
        storage_buttons = QHBoxLayout()
        open_data = QPushButton("Open data folder")
        open_data.clicked.connect(self._open_data_dir)
        clear_chats = QPushButton("Clear chat history…")
        clear_chats.setProperty("variant", "danger")
        clear_chats.clicked.connect(self._clear_history)
        clear_cache = QPushButton("Clear download cache…")
        clear_cache.clicked.connect(self._clear_cache)
        storage_buttons.addWidget(open_data)
        storage_buttons.addWidget(clear_cache)
        storage_buttons.addWidget(clear_chats)
        storage_buttons.addStretch(1)
        form.addRow("", storage_buttons)
        self.content_layout.addWidget(card)
        self._cards["storage"] = card

        # Cloud ------------------------------------------------------------
        cloud_card = Card()
        cloud_outer = QVBoxLayout(cloud_card)
        cloud_outer.setContentsMargins(22, 20, 22, 20)
        cloud_outer.setSpacing(14)
        cloud_outer.addWidget(section_title("Cloud Providers"))
        cloud_outer.addWidget(
            muted_label(
                "Use your own API keys with OpenAI, Anthropic, Gemini, OpenRouter, Groq, "
                "Mistral, DeepSeek, xAI, Together, Ollama, LM Studio or a custom "
                "OpenAI-compatible endpoint. Keys go to Windows Credential Manager."
            )
        )
        cloud_split = QHBoxLayout()
        cloud_split.setSpacing(14)
        self.provider_list = QListWidget()
        self.provider_list.setFixedWidth(210)
        self.provider_list.setSpacing(2)
        for provider_id in PROVIDER_ORDER:
            spec = PROVIDERS[provider_id]
            item = QListWidgetItem(spec.name)
            item.setData(Qt.UserRole, provider_id)
            self.provider_list.addItem(item)
        self.provider_list.currentItemChanged.connect(self._provider_selected)
        cloud_split.addWidget(self.provider_list)
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(0, 0, 0, 0)
        detail_layout.setSpacing(8)
        self.provider_name = QLabel("")
        self.provider_name.setStyleSheet("font-weight: 600;")
        self.provider_kind = muted_label("")
        self.provider_kind.setStyleSheet("font-size: 8.5pt;")
        self.provider_notes = muted_label("")
        self.provider_notes.setStyleSheet("font-size: 8.5pt;")
        detail_layout.addWidget(self.provider_name)
        detail_layout.addWidget(self.provider_kind)
        detail_layout.addWidget(self.provider_notes)
        pform = QFormLayout()
        pform.setHorizontalSpacing(20)
        pform.setVerticalSpacing(11)
        self.provider_base = QLineEdit()
        self.provider_base.setPlaceholderText("https://api.example.com/v1")
        pform.addRow("Base URL", self.provider_base)
        self.provider_model = QComboBox()
        self.provider_model.setEditable(True)
        pform.addRow("Default model", self.provider_model)
        self.provider_key = QLineEdit()
        self.provider_key.setEchoMode(QLineEdit.Password)
        self.provider_key.setPlaceholderText("API キーを入力")
        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        key_row.addWidget(self.provider_key, 1)
        self.provider_key_visible = QCheckBox("表示")
        self.provider_key_visible.toggled.connect(
            lambda visible: self.provider_key.setEchoMode(
                QLineEdit.Normal if visible else QLineEdit.Password
            )
        )
        key_row.addWidget(self.provider_key_visible)
        pform.addRow("API キー", key_row)
        detail_layout.addLayout(pform)
        self.provider_key_status = muted_label("")
        self.provider_key_status.setStyleSheet("font-size: 8.5pt;")
        detail_layout.addWidget(self.provider_key_status)
        prouds = QHBoxLayout()
        self.save_provider_button = QPushButton("Save provider")
        self.save_provider_button.setProperty("variant", "primary")
        self.save_provider_button.clicked.connect(self._save_provider)
        self.remove_key_button = QPushButton("Remove key")
        self.remove_key_button.setProperty("variant", "danger")
        self.remove_key_button.clicked.connect(self._remove_provider_key)
        self.test_provider_button = QPushButton("Test / fetch models")
        self.test_provider_button.clicked.connect(self._test_provider)
        self.provider_docs_button = QPushButton("Get API key")
        self.provider_docs_button.clicked.connect(self._open_provider_keys)
        prouds.addWidget(self.save_provider_button)
        prouds.addWidget(self.test_provider_button)
        prouds.addWidget(self.remove_key_button)
        prouds.addWidget(self.provider_docs_button)
        prouds.addStretch(1)
        detail_layout.addLayout(prouds)
        self.provider_message = QLabel("")
        self.provider_message.setWordWrap(True)
        self.provider_message.setStyleSheet("font-size: 8.5pt;")
        detail_layout.addWidget(self.provider_message)
        detail_layout.addStretch(1)
        cloud_split.addWidget(detail, 1)
        cloud_outer.addLayout(cloud_split)
        self.content_layout.addWidget(cloud_card)
        self._cards["cloud"] = cloud_card

        # About ------------------------------------------------------------
        about = muted_label(
            f"{APP_NAME} {APP_VERSION} · Windows desktop app · "
            "Local GGUF inference via llama.cpp, cloud inference via provider APIs. "
            "Chats never leave this PC except as prompts sent to a cloud provider you choose."
        )
        about.setStyleSheet("font-size: 8pt;")
        about.setAlignment(Qt.AlignCenter)
        self.content_layout.addWidget(about)
        self.content_layout.addStretch(1)

    def _build_agent_card(self) -> Card:
        card = Card()
        outer = QVBoxLayout(card)
        outer.setContentsMargins(22, 20, 22, 20)
        outer.setSpacing(14)
        outer.addWidget(section_title("Agent"))
        outer.addWidget(
            muted_label(
                "Hybrid Agent の既定値、ターミナル、権限を設定します。"
                "インストールと危険な操作の承認は常に有効で、無効化できません。"
            )
        )

        outer.addWidget(muted_label("General"))
        general = QFormLayout()
        general.setHorizontalSpacing(24)
        general.setVerticalSpacing(11)
        self.agent_mode = QComboBox()
        self.agent_mode.addItem("Autonomous Mode", "autonomous")
        self.agent_mode.addItem("Build Mode", "build")
        self.agent_mode.addItem("Plan Mode", "plan")
        self.agent_mode.addItem("Chat Mode", "chat")
        general.addRow("Default Agent Mode", self.agent_mode)
        self.agent_model = QComboBox()
        general.addRow("Default Model", self.agent_model)
        self.agent_max_steps = QSpinBox()
        self.agent_max_steps.setRange(1, 200)
        general.addRow("Maximum Steps", self.agent_max_steps)
        self.agent_max_seconds = QSpinBox()
        self.agent_max_seconds.setRange(30, 86400)
        self.agent_max_seconds.setSuffix(" 秒")
        general.addRow("Maximum Execution Time", self.agent_max_seconds)
        self.agent_max_retries = QSpinBox()
        self.agent_max_retries.setRange(0, 10)
        general.addRow("Maximum Retries", self.agent_max_retries)
        outer.addLayout(general)

        outer.addWidget(muted_label("Terminal"))
        terminal = QFormLayout()
        terminal.setHorizontalSpacing(24)
        terminal.setVerticalSpacing(11)
        self.agent_shell = QComboBox()
        self.agent_shell.addItem("CMD (cmd.exe)", "cmd")
        self.agent_shell.addItem("PowerShell", "powershell")
        self.agent_shell.addItem("Python", "python")
        self.agent_shell.addItem("Git", "git")
        terminal.addRow("Default Shell", self.agent_shell)
        self.agent_timeout = QSpinBox()
        self.agent_timeout.setRange(1, 3600)
        self.agent_timeout.setSuffix(" 秒")
        terminal.addRow("Command Timeout", self.agent_timeout)
        self.agent_show_output = QCheckBox("Show Command Output")
        terminal.addRow("", self.agent_show_output)
        self.agent_save_history = QCheckBox("Save Command History")
        terminal.addRow("", self.agent_save_history)
        outer.addLayout(terminal)

        outer.addWidget(muted_label("Permissions"))
        permissions = QFormLayout()
        permissions.setHorizontalSpacing(24)
        permissions.setVerticalSpacing(8)
        self.agent_full_access = QCheckBox(
            "Full Access Mode（作業ディレクトリ外のファイル変更を自動で許可）"
        )
        permissions.addRow("", self.agent_full_access)
        self.agent_file_access = QCheckBox("File Access（ファイル操作を許可）")
        permissions.addRow("", self.agent_file_access)
        self.agent_command_execution = QCheckBox("Command Execution（コマンド実行を許可）")
        permissions.addRow("", self.agent_command_execution)
        self.agent_install_approval = QCheckBox("Installation Approval（常に有効）")
        self.agent_install_approval.setChecked(True)
        self.agent_install_approval.setEnabled(False)
        permissions.addRow("", self.agent_install_approval)
        self.agent_danger_approval = QCheckBox("Dangerous Command Approval（常に有効）")
        self.agent_danger_approval.setChecked(True)
        self.agent_danger_approval.setEnabled(False)
        permissions.addRow("", self.agent_danger_approval)
        outer.addLayout(permissions)

        grants_header = QHBoxLayout()
        grants_header.addWidget(muted_label("保存済みの許可（Always Allow）"))
        grants_header.addStretch(1)
        self.agent_grant_count = muted_label("0 件")
        self.agent_grant_count.setStyleSheet("font-size: 8.5pt;")
        grants_header.addWidget(self.agent_grant_count)
        outer.addLayout(grants_header)
        grant_hint = muted_label(
            "許可は保存した種類と範囲にのみ適用されます。危険な操作・インストール・"
            "管理者権限の操作は保存済みでも毎回承認が必要です。"
            "エージェント自身はこの一覧を追加・変更・削除できません。"
        )
        grant_hint.setStyleSheet("font-size: 8.5pt;")
        outer.addWidget(grant_hint)
        self.agent_grants = QTableWidget(0, 4)
        self.agent_grants.setHorizontalHeaderLabels(
            ["種類", "対象範囲", "対象", "最終使用"]
        )
        self.agent_grants.setColumnWidth(0, 90)
        self.agent_grants.setColumnWidth(1, 300)
        self.agent_grants.setColumnWidth(2, 110)
        self.agent_grants.setSelectionBehavior(QTableWidget.SelectRows)
        self.agent_grants.setEditTriggers(QTableWidget.NoEditTriggers)
        self.agent_grants.setMinimumHeight(120)
        outer.addWidget(self.agent_grants)
        grant_actions = QHBoxLayout()
        self.agent_revoke_button = QPushButton("選択した許可を削除")
        self.agent_revoke_button.setProperty("variant", "danger")
        self.agent_revoke_button.clicked.connect(self._delete_agent_grant)
        grant_actions.addWidget(self.agent_revoke_button)
        grant_actions.addStretch(1)
        outer.addLayout(grant_actions)
        self._cards["agent"] = card
        return card

    # -- load / save -------------------------------------------------------
    def _load_values(self) -> None:
        s = self.settings
        self._set_combo(self.theme_combo, s.get("general.theme", "dark"))
        self.font_size.setValue(int(s.get("general.font_size", 10)))
        self.model_dir.setText(str(s.get("general.model_directory")))
        self._set_combo(self.start_behavior, s.get("general.start_behavior", "chat"))
        self.default_ctx.setValue(int(s.get("inference.default_context_length", 4096)))
        self.cpu_threads.setValue(int(s.get("inference.cpu_threads", 0)))
        self.gpu_layers.setValue(int(s.get("inference.gpu_layers", 0)))
        self.batch_size.setValue(int(s.get("inference.batch_size", 512)))
        self.flash_attention.setChecked(bool(s.get("inference.flash_attention", False)))
        self._set_combo(self.backend_combo, s.get("inference.backend", "auto"))
        self.auto_fallback.setChecked(bool(s.get("inference.auto_cpu_fallback", True)))
        self.llama_path.setText(str(s.get("inference.llama_server_path", "")))
        self.server_host.setCurrentText(str(s.get("server.host", "127.0.0.1")))
        self.server_port.setValue(int(s.get("server.port", 1234)))
        self.auto_start.setChecked(bool(s.get("server.auto_start", False)))
        self.allow_external.setChecked(bool(s.get("server.allow_external", False)))
        self._set_combo(self.agent_mode, s.get("agent.default_mode", "autonomous"))
        self._refresh_agent_models()
        preferred = str(s.get("agent.default_model", "") or "")
        if preferred:
            index = self.agent_model.findData(preferred)
            if index < 0:
                self.agent_model.addItem(preferred, preferred)
                index = self.agent_model.findData(preferred)
            self.agent_model.setCurrentIndex(max(0, index))
        self.agent_max_steps.setValue(int(s.get("agent.max_steps", 20)))
        self.agent_max_seconds.setValue(int(s.get("agent.max_seconds", 900)))
        self.agent_max_retries.setValue(int(s.get("agent.max_retries", 3)))
        self._set_combo(self.agent_shell, s.get("agent.default_shell", "cmd"))
        self.agent_timeout.setValue(int(s.get("agent.command_timeout", 120)))
        self.agent_show_output.setChecked(bool(s.get("agent.show_command_output", True)))
        self.agent_save_history.setChecked(bool(s.get("agent.save_command_history", True)))
        self.agent_full_access.setChecked(bool(s.get("agent.full_access", False)))
        self.agent_file_access.setChecked(bool(s.get("agent.file_access", True)))
        self.agent_command_execution.setChecked(
            bool(s.get("agent.command_execution", True))
        )
        self.agent_install_approval.setChecked(True)
        self.agent_danger_approval.setChecked(True)
        self._refresh_grants()
        self._refresh_storage()
        self.saved_label.setText("")

    @staticmethod
    def _set_combo(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def save(self) -> None:
        host = self.server_host.currentText().strip() or "127.0.0.1"
        model_dir = self.model_dir.text().strip()
        if not model_dir:
            show_error(self, "Invalid model directory", "Choose a folder for models.")
            return
        try:
            Path(model_dir).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            show_error(self, "Invalid model directory", f"Could not create the folder: {exc}")
            return
        if self.port_in_use_by_server() and self.server_port.value() != int(
            self.settings.get("server.port", 1234)
        ):
            self.show_status_message.emit(
                "Port change saved. Restart the API server to use the new port."
            )
        previous_theme = self.settings.get("general.theme", "dark")
        previous_font = int(self.settings.get("general.font_size", 10))
        previous_dir = self.settings.get("general.model_directory", "")
        self.settings.update(
            {
                "general": {
                    "theme": self.theme_combo.currentData(),
                    "font_size": self.font_size.value(),
                    "model_directory": str(Path(model_dir)),
                    "start_behavior": self.start_behavior.currentData(),
                },
                "inference": {
                    "default_context_length": self.default_ctx.value(),
                    "cpu_threads": self.cpu_threads.value(),
                    "gpu_layers": self.gpu_layers.value(),
                    "batch_size": self.batch_size.value(),
                    "flash_attention": self.flash_attention.isChecked(),
                    "backend": self.backend_combo.currentData(),
                    "auto_cpu_fallback": self.auto_fallback.isChecked(),
                    "llama_server_path": self.llama_path.text().strip(),
                },
                "server": {
                    "host": host,
                    "port": self.server_port.value(),
                    "auto_start": self.auto_start.isChecked(),
                    "allow_external": self.allow_external.isChecked(),
                },
                "agent": {
                    "default_mode": self.agent_mode.currentData(),
                    "default_model": str(self.agent_model.currentData() or ""),
                    "max_steps": self.agent_max_steps.value(),
                    "max_seconds": self.agent_max_seconds.value(),
                    "max_retries": self.agent_max_retries.value(),
                    "default_shell": self.agent_shell.currentData(),
                    "command_timeout": self.agent_timeout.value(),
                    "show_command_output": self.agent_show_output.isChecked(),
                    "save_command_history": self.agent_save_history.isChecked(),
                    "full_access": self.agent_full_access.isChecked(),
                    "file_access": self.agent_file_access.isChecked(),
                    "command_execution": self.agent_command_execution.isChecked(),
                },
            }
        )
        self.saved_label.setText("Saved")
        self.show_status_message.emit("Settings saved")
        self._refresh_storage()
        self.settings_saved.emit()
        if (
            previous_theme != self.settings.get("general.theme")
            or previous_font != int(self.settings.get("general.font_size", 10))
        ):
            self.show_status_message.emit("Theme updated")
        if previous_dir != str(self.settings.get("general.model_directory")):
            self.show_status_message.emit("Model directory changed; rescanning…")

    def port_in_use_by_server(self) -> bool:
        return self.services.api_server.running

    def _browse_model_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose model directory", self.model_dir.text()
        )
        if directory:
            self.model_dir.setText(directory)

    def _browse_llama(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select llama-server executable", "", "Executables (*.exe);;All files (*.*)"
        )
        if path:
            self.llama_path.setText(path)

    # -- agent -------------------------------------------------------------
    def _refresh_agent_models(self) -> None:
        current = str(self.agent_model.currentData() or "")
        self.agent_model.blockSignals(True)
        self.agent_model.clear()
        self.agent_model.addItem("（読み込み中のモデル）", "")
        try:
            entries = self.services.library.entries()
        except Exception:
            entries = []
        for entry in entries:
            name = str(getattr(entry, "name", entry))
            self.agent_model.addItem(name, name)
        index = self.agent_model.findData(current)
        self.agent_model.setCurrentIndex(index if index >= 0 else 0)
        self.agent_model.blockSignals(False)

    def _refresh_grants(self) -> None:
        table = self.agent_grants
        table.setRowCount(0)
        try:
            grants = self.services.agent_permissions.list_grants()
        except Exception:
            grants = []
        for grant in grants:
            row = table.rowCount()
            table.insertRow(row)
            values = [
                grant.kind,
                grant.scope or "—",
                grant.target or "—",
                (grant.last_used or "—")[:19].replace("T", " "),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 0:
                    item.setData(Qt.UserRole, grant.id)
                table.setItem(row, column, item)
        self.agent_grant_count.setText(f"{len(grants)} 件")

    def _delete_agent_grant(self) -> None:
        row = self.agent_grants.currentRow()
        if row < 0:
            self.show_status_message.emit("削除する許可を選択してください")
            return
        item = self.agent_grants.item(row, 0)
        grant_id = str(item.data(Qt.UserRole) or "") if item is not None else ""
        if not grant_id:
            return
        if not confirm(
            self,
            "保存済みの許可を削除",
            "この許可を削除しますか？同じ操作は再び承認が必要になります。",
        ):
            return
        try:
            self.services.agent_permissions.revoke_grant(grant_id, origin="user")
        except Exception as exc:
            show_error(self, "許可を削除できませんでした", str(exc))
            return
        self._refresh_grants()
        self.show_status_message.emit("保存済みの許可を削除しました")

    def _external_toggled(self, checked: bool) -> None:
        if not checked:
            return
        if not confirm(
            self,
            "Enable external access?",
            "The API server has no authentication. Enabling external access lets other "
            "machines on your network query your model and use your computer's resources.\n\n"
            "Enable this only on a trusted network.",
        ):
            self.allow_external.setChecked(False)

    # -- storage -----------------------------------------------------------
    def _refresh_storage(self) -> None:
        try:
            stats = self.services.database.stats()
            partials = sum(
                1
                for task in self.services.downloads.snapshot()
                if task.status in {"cancelled", "error", "paused"}
            )
            size_mb = stats["bytes"] / (1024 * 1024)
            self.storage_info.setText(
                f"{stats['path']}\n"
                f"{stats['chats']} conversations · {stats['messages']} messages · "
                f"{size_mb:.1f} MB · {partials} resumable/partial downloads"
            )
        except Exception as exc:
            self.storage_info.setText(f"Could not read storage stats: {exc}")

    def _open_data_dir(self) -> None:
        from app.core import paths

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(paths.data_dir())))

    def _clear_history(self) -> None:
        if not confirm(
            self,
            "Clear chat history",
            "Delete every conversation and message from the local database? "
            "This cannot be undone. Model files and settings are not affected.",
        ):
            return
        try:
            self.services.database.clear_history()
        except Exception as exc:
            show_error(self, "Could not clear history", str(exc))
            return
        self._refresh_storage()
        self.history_cleared.emit()
        self.show_status_message.emit("Chat history cleared")

    def _clear_cache(self) -> None:
        if not confirm(
            self,
            "Clear download cache",
            "Delete partial .part downloads? Completed model files will not be touched.",
        ):
            return
        files, freed = self.services.downloads.clear_partial_downloads()
        self._refresh_storage()
        self.show_status_message.emit(
            f"Removed {files} partial file(s), freed {freed / (1024 * 1024):.1f} MB"
        )

    # -- cloud providers ---------------------------------------------------
    def _provider_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        provider_id = str(current.data(Qt.UserRole))
        self._load_provider(provider_id)

    def _load_provider(self, provider_id: str) -> None:
        spec = get_provider(provider_id)
        if spec is None:
            return
        self._current_provider = provider_id
        self.provider_name.setText(spec.name)
        self.provider_kind.setText(
            f"{KIND_LABELS.get(spec.kind, spec.kind)}"
            + (" · local server" if spec.local else "")
            + (" · API key required" if spec.key_required else " · no key required")
        )
        self.provider_notes.setText(spec.notes or "")
        cfg = self.settings.get(f"cloud.providers.{provider_id}", {}) or {}
        if not isinstance(cfg, dict):
            cfg = {}
        self.provider_base.setText(str(cfg.get("base_url") or spec.base_url))
        self.provider_base.setEnabled(True)
        self.provider_model.clear()
        default_model = str(cfg.get("default_model") or spec.default_model)
        for model in provider_models(provider_id, cfg):
            self.provider_model.addItem(model)
        self.provider_model.setCurrentText(default_model)
        self.provider_key.clear()
        self.provider_key.setPlaceholderText(
            "Saved — leave blank to keep"
            if self.services.cloud_configured(provider_id)
            else ("Optional" if not spec.key_required else "Paste API key")
        )
        self.provider_key.setEnabled(spec.key_required or provider_id in {"custom", "ollama", "lmstudio"})
        self._update_provider_status()
        self.provider_message.setText("")
        self.provider_docs_button.setEnabled(bool(spec.key_url or spec.docs_url))
        self.test_provider_button.setEnabled(bool(self.provider_base.text().strip()))
        self.provider_list.setCurrentRow(
            PROVIDER_ORDER.index(provider_id) if provider_id in PROVIDER_ORDER else -1
        )

    def _update_provider_status(self) -> None:
        spec = get_provider(self._current_provider)
        if spec is None:
            return
        if not spec.key_required:
            text = "No API key required for this provider."
            color = current_palette()["text_muted"]
        elif self.services.cloud_configured(spec.id):
            text = "API key saved in Windows Credential Manager."
            color = current_palette()["success"]
        else:
            text = "No API key saved yet."
            color = current_palette()["warning"]
        self.provider_key_status.setText(text)
        self.provider_key_status.setStyleSheet(f"font-size: 8.5pt; color: {color};")
        self.remove_key_button.setEnabled(self.services.cloud_configured(spec.id))
        # Update the list marker.
        for row in range(self.provider_list.count()):
            item = self.provider_list.item(row)
            pid = str(item.data(Qt.UserRole))
            marker = "  ✓" if self.services.cloud_configured(pid) else ""
            item.setText(PROVIDERS[pid].name + marker)
        localize_widget_tree(self)

    def _save_provider(self) -> None:
        spec = get_provider(self._current_provider)
        if spec is None:
            return
        base_url = self.provider_base.text().strip()
        if spec.id == "custom" and not base_url:
            self.provider_message.setText("A custom provider needs a base URL.")
            self.provider_message.setStyleSheet(
                f"font-size: 8.5pt; color: {current_palette()['danger']};"
            )
            return
        model = self.provider_model.currentText().strip()
        try:
            self.settings.set(
                f"cloud.providers.{spec.id}",
                {
                    "base_url": base_url or spec.base_url,
                    "default_model": model,
                },
            )
            key = self.provider_key.text().strip()
            if key:
                self.services.credentials.set(credential_key(spec.id), key)
                self.provider_key.clear()
        except Exception as exc:
            show_error(self, "Could not save provider", str(exc))
            return
        self.provider_key.setPlaceholderText(
            "Saved — leave blank to keep" if self.services.cloud_configured(spec.id) else "Paste API key"
        )
        self._update_provider_status()
        self.provider_message.setText("Provider saved. Pick it from the model menu at the top.")
        self.provider_message.setStyleSheet(
            f"font-size: 8.5pt; color: {current_palette()['success']};"
        )
        self.show_status_message.emit(f"{spec.name} settings saved")
        self.cloud_changed.emit()
        localize_widget_tree(self)

    def _remove_provider_key(self) -> None:
        spec = get_provider(self._current_provider)
        if spec is None:
            return
        if not confirm(self, "Remove API key", f"Remove the stored API key for {spec.name}?"):
            return
        try:
            self.services.credentials.delete(credential_key(spec.id))
        except Exception as exc:
            show_error(self, "Could not remove key", str(exc))
            return
        self._update_provider_status()
        self.provider_key.clear()
        self.show_status_message.emit(f"{spec.name} API key removed")
        self.cloud_changed.emit()

    def _test_provider(self) -> None:
        provider_id = self._current_provider
        if self._cloud_worker is not None and self._cloud_worker.isRunning():
            return
        # Persist the base URL/key first so the test uses what is on screen.
        try:
            if self.provider_key.text().strip():
                self.services.credentials.set(
                    credential_key(provider_id), self.provider_key.text().strip()
                )
                self.provider_key.clear()
            self.settings.set(
                f"cloud.providers.{provider_id}",
                {
                    "base_url": self.provider_base.text().strip(),
                    "default_model": self.provider_model.currentText().strip(),
                },
            )
        except Exception as exc:
            self.provider_message.setText(str(exc))
            return
        self._update_provider_status()
        self.test_provider_button.setEnabled(False)
        self.test_provider_button.setText("Testing…")
        self.provider_message.setText("Contacting the provider…")
        self.provider_message.setStyleSheet(
            f"font-size: 8.5pt; color: {current_palette()['text_muted']};"
        )

        def work():
            return fetch_models(provider_id, self.settings, self.services.credentials)

        def done(models):
            self.test_provider_button.setEnabled(True)
            self.test_provider_button.setText("Test / fetch models")
            current = self.provider_model.currentText()
            if models:
                self.provider_model.clear()
                for model in models:
                    self.provider_model.addItem(model)
                if current:
                    self.provider_model.setCurrentText(current)
                else:
                    self.provider_model.setCurrentText(models[0])
                self.provider_message.setText(
                    f"Connected. Found {len(models)} models — choose one and save."
                )
                self.provider_message.setStyleSheet(
                    f"font-size: 8.5pt; color: {current_palette()['success']};"
                )
            else:
                self.provider_message.setText(
                    "Connected, but the provider returned no models. Enter a model ID manually."
                )
                self.provider_message.setStyleSheet(
                    f"font-size: 8.5pt; color: {current_palette()['warning']};"
                )
            localize_widget_tree(self)

        def failed(message, detail):
            self.test_provider_button.setEnabled(True)
            self.test_provider_button.setText("Test / fetch models")
            self.provider_message.setText(message)
            self.provider_message.setToolTip(detail)
            self.provider_message.setStyleSheet(
                f"font-size: 8.5pt; color: {current_palette()['danger']};"
            )
            localize_widget_tree(self)

        worker = TaskWorker(work)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._cleanup_cloud_worker)
        self._cloud_worker = worker
        worker.start()

    def _cleanup_cloud_worker(self) -> None:
        self._cloud_worker = None

    def _open_provider_keys(self) -> None:
        spec = get_provider(self._current_provider)
        if spec:
            open_url(spec.key_url or spec.docs_url)

    # -- external hooks ----------------------------------------------------
    def open_section(self, name: str) -> None:
        """Scroll to a settings card (general/inference/server/agent/storage/cloud)."""
        card = self._cards.get(name)
        if card is not None:
            self.scroll.ensureWidgetVisible(card, 0, 12)

    def refresh_from_settings(self) -> None:
        self._load_values()
        self._load_provider(self._current_provider)

    def update_inference_status(self, _status: dict[str, Any]) -> None:
        pass

    def shutdown(self) -> None:
        workers = list(self._workers)
        if self._cloud_worker is not None:
            workers.append(self._cloud_worker)
        for worker in workers:
            if worker.isRunning():
                worker.wait(4000)
        self._workers.clear()
        self._cloud_worker = None
