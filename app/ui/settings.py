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
        self._set_combo(self.backend_combo, s.get("inference.backend", "auto"))
        self.auto_fallback.setChecked(bool(s.get("inference.auto_cpu_fallback", True)))
        self.llama_path.setText(str(s.get("inference.llama_server_path", "")))
        self.server_host.setCurrentText(str(s.get("server.host", "127.0.0.1")))
        self.server_port.setValue(int(s.get("server.port", 1234)))
        self.auto_start.setChecked(bool(s.get("server.auto_start", False)))
        self.allow_external.setChecked(bool(s.get("server.allow_external", False)))
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
        """Scroll to a settings card (general/inference/server/storage/cloud)."""
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
