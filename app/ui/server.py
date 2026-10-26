"""Local Server page: OpenAI-compatible API status and controls."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.api.server import ApiServerError
from app.ui.dialogs import show_error
from app.ui.i18n import localize_widget_tree
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import Card, StatusChip, muted_label, section_title
from app.workers.workers import TaskWorker


class ServerPage(QWidget):
    settings_requested = Signal()
    show_status_message = Signal(str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.api = services.api_server
        self._workers: list[TaskWorker] = []
        self._build()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self.refresh)
        self._timer.start()
        self.refresh()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        layout.setSpacing(18)

        header = QHBoxLayout()
        header.addWidget(section_title("OpenAI-compatible API"))
        header.addStretch(1)
        self.settings_button = QPushButton("Server settings")
        self.settings_button.setProperty("variant", "ghost")
        self.settings_button.clicked.connect(self.settings_requested.emit)
        header.addWidget(self.settings_button)
        layout.addLayout(header)
        layout.addWidget(
            muted_label(
                "Share the loaded model with other applications. The API is served by "
                "this app and uses the same single inference backend as the chat window."
            )
        )

        status_card = Card()
        status_layout = QHBoxLayout(status_card)
        status_layout.setContentsMargins(22, 20, 22, 20)
        status_layout.setSpacing(22)
        self.chip = StatusChip()
        self.chip.set_status("idle", "Server stopped")
        status_layout.addWidget(self.chip)
        self.status_detail = muted_label("")
        status_layout.addWidget(self.status_detail, 1)
        self.start_button = QPushButton("Start Server")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setCursor(Qt.PointingHandCursor)
        self.start_button.clicked.connect(self._start)
        self.stop_button = QPushButton("Stop Server")
        self.stop_button.setProperty("variant", "danger")
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.clicked.connect(self._stop)
        status_layout.addWidget(self.start_button)
        status_layout.addWidget(self.stop_button)
        layout.addWidget(status_card)

        grid_card = Card(soft=True)
        grid = QGridLayout(grid_card)
        grid.setContentsMargins(22, 18, 22, 18)
        grid.setHorizontalSpacing(28)
        grid.setVerticalSpacing(11)
        self.values: dict[str, QLabel] = {}
        rows = [
            ("status", "Server Status"),
            ("host", "Host"),
            ("port", "Port"),
            ("base", "Base URL"),
            ("api", "API Base (OpenAI)"),
            ("model", "Loaded Model"),
            ("active", "Active Requests"),
            ("requests", "Requests Served"),
            ("external", "Network Exposure"),
        ]
        for row, (key, label) in enumerate(rows):
            name = QLabel(label)
            name.setProperty("variant", "muted")
            name.setStyleSheet("font-size: 9pt;")
            value = QLabel("")
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.values[key] = value
            grid.addWidget(name, row, 0)
            grid.addWidget(value, row, 1)
            if key in {"base", "api"}:
                copy = QPushButton("Copy")
                copy.setProperty("variant", "ghost")
                copy.setCursor(Qt.PointingHandCursor)
                copy.clicked.connect(
                    lambda _=False, k=key: self._copy_url(k)
                )
                grid.addWidget(copy, row, 2)
        self.values["base"].setProperty("variant", "mono")
        self.values["api"].setProperty("variant", "mono")
        grid.setColumnStretch(1, 1)
        layout.addWidget(grid_card)

        actions = QHBoxLayout()
        self.docs_button = QPushButton("Open API docs")
        self.docs_button.clicked.connect(self._open_docs)
        self.models_button = QPushButton("Open /v1/models")
        self.models_button.clicked.connect(self._open_models)
        for button in (self.docs_button, self.models_button):
            button.setProperty("variant", "ghost")
            button.setCursor(Qt.PointingHandCursor)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        layout.addWidget(section_title("Quick test"))
        sample = Card(soft=True)
        sample_layout = QVBoxLayout(sample)
        sample_layout.setContentsMargins(14, 12, 14, 12)
        sample_layout.setSpacing(6)
        sample_layout.addWidget(muted_label("PowerShell (after starting the server):"))
        self.sample = QLabel("")
        self.sample.setProperty("variant", "mono")
        self.sample.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sample.setWordWrap(True)
        sample_layout.addWidget(self.sample)
        copy_sample = QPushButton("Copy command")
        copy_sample.setProperty("variant", "ghost")
        copy_sample.clicked.connect(self._copy_sample)
        sample_layout.addWidget(copy_sample, 0, Qt.AlignLeft)
        layout.addWidget(sample)
        layout.addStretch(1)

        self.external_warning = muted_label(
            "External access is disabled. Only 127.0.0.1 can connect."
        )
        self.external_warning.setStyleSheet(f"color: {current_palette()['warning']};")
        layout.addWidget(self.external_warning)
        self.refresh()

    def _base_url(self) -> str:
        info = self.api.info()
        return info["base_url"]

    def _settings_signature(self) -> tuple:
        return (
            str(self.services.settings.get("server.host")),
            int(self.services.settings.get("server.port")),
            bool(self.services.settings.get("server.allow_external")),
        )

    def _start(self) -> None:
        self.start_button.setEnabled(False)
        self.start_button.setText("Starting…")
        host, port, allow = self._settings_signature()

        def work():
            return self.api.start(host=host, port=port, allow_external=allow)

        def done(_info):
            self.start_button.setText("Start Server")
            self.start_button.setEnabled(True)
            self.show_status_message.emit("API server started")
            self.refresh()

        def failed(message: str, detail: str):
            self.start_button.setText("Start Server")
            self.start_button.setEnabled(True)
            show_error(self, "Could not start server", message, detail)
            self.refresh()

        self._run(work, done, failed)

    def _stop(self) -> None:
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Stopping…")

        def done(_result=None):
            self.stop_button.setText("Stop Server")
            self.stop_button.setEnabled(True)
            self.show_status_message.emit("API server stopped")
            self.refresh()

        def failed(message: str, detail: str):
            self.stop_button.setText("Stop Server")
            self.stop_button.setEnabled(True)
            show_error(self, "Could not stop server", message, detail)

        self._run(self.api.stop, done, failed)

    def _run(self, fn, done, failed) -> None:
        worker = TaskWorker(fn)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)
        worker.finished.connect(lambda: self._workers.remove(worker) if worker in self._workers else None)
        worker.start()

    def refresh(self) -> None:
        info = self.api.info()
        running = bool(info["running"])
        status = self.services.inference.status()
        if running:
            self.chip.set_status("ready", "Running")
            self.start_button.setEnabled(False)
            self.start_button.setText("Start Server")
            self.stop_button.setVisible(True)
            self.status_detail.setText(
                "Serving the loaded model. Keep this window open while clients connect."
            )
        else:
            self.chip.set_status("idle", "Server stopped")
            self.start_button.setEnabled(True)
            self.stop_button.setVisible(False)
            self.status_detail.setText(
                "Start the server, then point any OpenAI-compatible client at the API base URL."
            )
        self.values["status"].setText("Running" if running else "Stopped")
        self.values["status"].setStyleSheet(
            f"color: {current_palette()['success' if running else 'text_muted']};"
        )
        self.values["host"].setText(str(info["host"]))
        self.values["port"].setText(str(info["port"]))
        self.values["base"].setText(info["base_url"])
        self.values["api"].setText(info["api_base_url"])
        self.values["model"].setText(
            status.get("model_id") if status.get("state") == "ready" else "No model loaded"
        )
        self.values["active"].setText(str(status.get("active_requests", 0)))
        self.values["requests"].setText(str(info.get("request_count", 0)))
        self.values["external"].setText(
            "External interfaces enabled (see Settings)" if info.get("external") else "Loopback only (127.0.0.1)"
        )
        self.values["external"].setStyleSheet(
            f"color: {current_palette()['warning' if info.get('external') else 'text']};"
        )
        self.external_warning.setVisible(bool(info.get("external")))
        if info.get("external"):
            self.external_warning.setText(
                "Warning: the server is reachable from other machines on your network. "
                "There is no authentication layer."
            )
        self.docs_button.setEnabled(running)
        self.models_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        base = info["base_url"]
        self.sample.setText(
            "Invoke-RestMethod -Uri \"{base}/v1/chat/completions\" -Method Post "
            "-ContentType \"application/json\" -Body '{{\"model\":\"{model}\","
            "\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}]}}'".format(
                base=base,
                model=status.get("model_id") or "local/model",
            )
        )
        localize_widget_tree(self)
        if not running:
            self.start_button.setEnabled(True)

    def update_inference_status(self, _status: dict[str, Any]) -> None:
        self.refresh()

    def _copy_url(self, key: str) -> None:
        QApplication.clipboard().setText(self.values[key].text())
        self.show_status_message.emit("URL copied to clipboard")

    def _copy_sample(self) -> None:
        QApplication.clipboard().setText(self.sample.text())
        self.show_status_message.emit("Sample command copied")

    def _open_docs(self) -> None:
        from PySide6.QtCore import QUrl

        if self.api.running:
            QDesktopServices.openUrl(QUrl(f"{self._base_url()}/docs"))

    def _open_models(self) -> None:
        from PySide6.QtCore import QUrl

        if self.api.running:
            QDesktopServices.openUrl(QUrl(f"{self._base_url()}/v1/models"))

    def shutdown(self) -> None:
        self._timer.stop()
