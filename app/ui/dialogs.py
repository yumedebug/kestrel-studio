"""Dialogs: cloud model picker and per-model load settings."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.core.cloud_models import fetch_models, provider_config
from app.core.model_manager import ModelEntry, ModelManager
from app.core.providers import PROVIDERS, PROVIDER_ORDER, get_provider, provider_models
from app.core.settings import SettingsStore
from app.core.types import LoadOptions, ModelTarget
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import Card, muted_label, section_title
from app.workers.workers import TaskWorker


def show_error(parent: QWidget, title: str, message: str, detail: str = "") -> None:
    box = QMessageBox(parent)
    box.setIcon(QMessageBox.Critical)
    box.setWindowTitle(title)
    box.setText(message)
    if detail:
        box.setDetailedText(detail[:8000])
    box.exec()


def show_info(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.information(parent, title, message)


def confirm(parent: QWidget, title: str, message: str) -> bool:
    result = QMessageBox.question(
        parent,
        title,
        message,
        QMessageBox.Yes | QMessageBox.Cancel,
        QMessageBox.Cancel,
    )
    return result == QMessageBox.Yes


def open_url(url: str) -> None:
    from PySide6.QtCore import QUrl

    if url:
        QDesktopServices.openUrl(QUrl(url))


class CloudModelDialog(QDialog):
    """Pick a provider + model ID, optionally fetching the provider's list."""

    request_settings = Signal()

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.selected_target: ModelTarget | None = None
        self._worker: TaskWorker | None = None
        self.setWindowTitle("Choose a cloud model")
        self.setMinimumWidth(560)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(section_title("Cloud model"))
        layout.addWidget(
            muted_label(
                "Use your own API key with a provider. Keys are stored in Windows "
                "Credential Manager and never written to settings.json."
            )
        )

        form = QFormLayout()
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.provider_combo = QComboBox()
        for provider_id in PROVIDER_ORDER:
            spec = PROVIDERS[provider_id]
            if spec.id == "custom":
                label = spec.name
            elif spec.local:
                label = spec.name
            elif services.cloud_configured(provider_id):
                label = f"{spec.name}  ✓ key saved"
            else:
                label = f"{spec.name}  (no key)"
            self.provider_combo.addItem(label, provider_id)
        form.addRow("Provider", self.provider_combo)

        self.base_url = QLineEdit()
        self.base_url.setPlaceholderText("https://api.example.com/v1")
        form.addRow("Base URL", self.base_url)

        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.setMinimumWidth(330)
        form.addRow("Model ID", self.model_combo)

        self.key_status = QLabel()
        self.key_status.setWordWrap(True)
        self.key_status.setProperty("variant", "muted")
        form.addRow("Credential", self.key_status)
        layout.addLayout(form)

        self.notes = muted_label("")
        self.notes.setStyleSheet("font-size: 9pt;")
        layout.addWidget(self.notes)

        actions = QHBoxLayout()
        self.fetch_button = QPushButton("Fetch model list")
        self.fetch_button.setCursor(Qt.PointingHandCursor)
        self.fetch_button.clicked.connect(self._fetch_models)
        self.key_button = QPushButton("Manage API keys")
        self.key_button.setCursor(Qt.PointingHandCursor)
        self.key_button.clicked.connect(self._manage_keys)
        self.docs_button = QPushButton("Provider docs")
        self.docs_button.clicked.connect(self._open_docs)
        actions.addWidget(self.fetch_button)
        actions.addWidget(self.key_button)
        actions.addWidget(self.docs_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.message = QLabel("")
        self.message.setWordWrap(True)
        layout.addWidget(self.message)

        buttons = QDialogButtonBox()
        self.load_button = QPushButton("Load model")
        self.load_button.setProperty("variant", "primary")
        self.load_button.setCursor(Qt.PointingHandCursor)
        self.load_button.clicked.connect(self._accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addButton(self.load_button, QDialogButtonBox.AcceptRole)
        buttons.addButton(cancel, QDialogButtonBox.RejectRole)
        layout.addWidget(buttons)

        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        self._provider_changed()

    # -- provider handling -------------------------------------------------
    def provider_id(self) -> str:
        return str(self.provider_combo.currentData())

    def _provider_changed(self) -> None:
        provider_id = self.provider_id()
        spec = get_provider(provider_id)
        if spec is None:
            return
        cfg = provider_config(self.services.settings, provider_id)
        base_url = cfg["base_url"] or spec.base_url
        self.base_url.setText(base_url)
        self.base_url.setEnabled(bool(spec.id == "custom") or not base_url)
        default_model = cfg["default_model"] or spec.default_model
        self.model_combo.clear()
        for model in provider_models(provider_id, cfg):
            self.model_combo.addItem(model)
        self.model_combo.setCurrentText(default_model)
        self.notes.setText(
            f"{spec.name}: {spec.notes}" if spec.notes else spec.name
        )
        self.docs_button.setEnabled(bool(spec.docs_url))
        self.fetch_button.setEnabled(bool(base_url))
        self._update_key_status()
        self.message.clear()

    def _update_key_status(self) -> None:
        spec = get_provider(self.provider_id())
        if spec is None:
            return
        if not spec.key_required:
            self.key_status.setText("No API key required (local / self-hosted).")
            self.key_status.setProperty("variant", "muted")
            return
        if self.services.cloud_configured(spec.id):
            self.key_status.setText("API key saved in Windows Credential Manager.")
            self.key_status.setProperty("variant", "success")
        else:
            self.key_status.setText(
                "No API key yet. Add one in Settings → Cloud Providers."
            )
            self.key_status.setProperty("variant", "warning")
        self.key_status.style().unpolish(self.key_status)
        self.key_status.style().polish(self.key_status)

    def _manage_keys(self) -> None:
        self.request_settings.emit()

    def _open_docs(self) -> None:
        spec = get_provider(self.provider_id())
        if spec:
            open_url(spec.key_url or spec.docs_url)

    # -- model list --------------------------------------------------------
    def _fetch_models(self) -> None:
        provider_id = self.provider_id()
        if self._worker is not None and self._worker.isRunning():
            return
        self.fetch_button.setEnabled(False)
        self.fetch_button.setText("Fetching…")
        self.message.setText("Contacting the provider…")
        self.message.setStyleSheet("")

        def work():
            return fetch_models(provider_id, self.services.settings, self.services.credentials)

        worker = TaskWorker(work)
        worker.succeeded.connect(self._models_loaded)
        worker.failed.connect(self._models_failed)
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _models_loaded(self, models) -> None:
        self.fetch_button.setEnabled(True)
        self.fetch_button.setText("Fetch model list")
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for model in models or []:
            self.model_combo.addItem(str(model))
        if current:
            self.model_combo.setCurrentText(current)
        if models:
            self.message.setText(f"Loaded {len(models)} model IDs.")
        else:
            self.message.setText("The provider returned an empty model list.")
        self.message.setStyleSheet(
            f"color: {current_palette()['text_muted']}; background: transparent;"
        )

    def _models_failed(self, message: str, detail: str) -> None:
        self.fetch_button.setEnabled(True)
        self.fetch_button.setText("Fetch model list")
        self.message.setText(message)
        self.message.setStyleSheet(
            f"color: {current_palette()['danger']}; background: transparent;"
        )
        self.message.setToolTip(detail)

    # -- accept ------------------------------------------------------------
    def _accept(self) -> None:
        provider_id = self.provider_id()
        spec = get_provider(provider_id)
        if spec is None:
            return
        model = self.model_combo.currentText().strip()
        if not model:
            self.message.setText("Enter or choose a model ID.")
            self.message.setStyleSheet(
                f"color: {current_palette()['danger']}; background: transparent;"
            )
            return
        base_url = self.base_url.text().strip()
        if not base_url:
            self.message.setText("Set the provider base URL first.")
            return
        if spec.key_required and not self.services.cloud_configured(provider_id):
            self.message.setText(
                "This provider needs an API key. Open Settings → Cloud Providers, save "
                "a key, then try again."
            )
            self.message.setStyleSheet(
                f"color: {current_palette()['warning']}; background: transparent;"
            )
            return
        self.services.settings.set(
            f"cloud.providers.{provider_id}",
            {"base_url": base_url, "default_model": model},
        )
        self.selected_target = ModelTarget(
            kind="cloud",
            display_name=model,
            provider=provider_id,
            model=model,
        )
        self.accept()


class ModelLoadDialog(QDialog):
    def __init__(
        self,
        entry: ModelEntry,
        settings: SettingsStore,
        library: ModelManager,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.entry = entry
        self.settings = settings
        self.library = library
        self.options: LoadOptions | None = None
        self.setWindowTitle(f"Load {entry.name}")
        self.setMinimumWidth(520)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(12)
        layout.addWidget(section_title("Load settings"))
        info = muted_label(
            f"{entry.name}\n"
            f"{entry.quant or 'Unknown quant'} · {entry.architecture or 'Unknown architecture'} · "
            f"{entry.size_label} · {entry.path}"
        )
        info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(info)

        card = Card(soft=True)
        form = QFormLayout(card)
        form.setContentsMargins(14, 14, 14, 14)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(10)

        defaults = entry.resolved_options()
        default_ctx = int(
            defaults.context_length
            or self.settings.get("inference.default_context_length", 4096)
        )
        if entry.context_length:
            default_ctx = min(default_ctx, entry.context_length)
        self.context = QSpinBox()
        self.context.setRange(128, 1_048_576)
        self.context.setSingleStep(512)
        self.context.setValue(max(128, default_ctx))
        self.context.setSuffix(" tokens")
        form.addRow("Context Length", self.context)

        self.gpu_layers = QSpinBox()
        self.gpu_layers.setRange(0, 999)
        gpu_default = defaults.gpu_layers
        if gpu_default is None:
            gpu_default = self.settings.get("inference.gpu_layers", 0)
        self.gpu_layers.setValue(int(gpu_default or 0))
        form.addRow("GPU Layers", self.gpu_layers)

        self.threads = QSpinBox()
        self.threads.setRange(0, 256)
        thread_default = defaults.cpu_threads
        if thread_default is None:
            thread_default = self.settings.get("inference.cpu_threads", 0)
        self.threads.setValue(int(thread_default or 0))
        self.threads.setSpecialValueText("Auto")
        form.addRow("CPU Threads", self.threads)

        self.batch = QSpinBox()
        self.batch.setRange(32, 8192)
        self.batch.setSingleStep(64)
        batch_default = defaults.batch_size
        if batch_default is None:
            batch_default = self.settings.get("inference.batch_size", 512)
        self.batch.setValue(int(batch_default or 512))
        form.addRow("Batch Size", self.batch)
        layout.addWidget(card)

        hint = muted_label(
            "GPU Layers = 0 runs on CPU. 999 offloads every layer and requires a CUDA or "
            "Vulkan llama.cpp build. Settings are saved per model."
        )
        hint.setStyleSheet("font-size: 9pt;")
        layout.addWidget(hint)
        if entry.metadata_error:
            warning = QLabel(f"Metadata warning: {entry.metadata_error}")
            warning.setWordWrap(True)
            warning.setStyleSheet(f"color: {current_palette()['warning']};")
            layout.addWidget(warning)

        buttons = QHBoxLayout()
        reset = QPushButton("Reset to defaults")
        reset.clicked.connect(self._reset)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        load = QPushButton("Load Model")
        load.setProperty("variant", "primary")
        load.clicked.connect(self._accept)
        buttons.addWidget(cancel)
        buttons.addWidget(load)
        layout.addLayout(buttons)

    def _reset(self) -> None:
        self.context.setValue(int(self.settings.get("inference.default_context_length", 4096)))
        self.gpu_layers.setValue(int(self.settings.get("inference.gpu_layers", 0)))
        self.threads.setValue(int(self.settings.get("inference.cpu_threads", 0)))
        self.batch.setValue(int(self.settings.get("inference.batch_size", 512)))

    def _accept(self) -> None:
        self.options = LoadOptions(
            context_length=self.context.value(),
            gpu_layers=self.gpu_layers.value(),
            cpu_threads=self.threads.value(),
            batch_size=self.batch.value(),
        )
        try:
            self.library.save_load_options(self.entry.id, self.options)
        except Exception as exc:
            show_error(self, "Could not save settings", str(exc))
            return
        self.accept()
