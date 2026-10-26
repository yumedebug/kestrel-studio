"""Discover page: Hugging Face GGUF search, file listing and downloads."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.downloader import DownloadManager, DownloadTask
from app.core.hf_client import HFError, HFModel, HFRepoDetails, HuggingFaceClient, human_size
from app.core.types import BackendError
from app.ui.dialogs import open_url, show_error
from app.ui.i18n import localize_widget_tree
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import (
    Card,
    EmptyState,
    StatusChip,
    icon_button,
    muted_label,
    section_title,
)
from app.workers.workers import TaskWorker

log = logging.getLogger(__name__)


class TokenDialog(QDialog):
    def __init__(self, hf: HuggingFaceClient, parent: QWidget | None = None):
        super().__init__(parent)
        self.hf = hf
        self._worker: TaskWorker | None = None
        self.setWindowTitle("Hugging Face token")
        self.setMinimumWidth(520)
        self.setModal(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 16)
        layout.setSpacing(14)
        layout.addWidget(section_title("Hugging Face access token"))
        layout.addWidget(
            muted_label(
                "A token is only needed for gated or private repositories. It is stored in "
                "Windows Credential Manager, not in settings.json. Public GGUF search and "
                "downloads work without one."
            )
        )
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.field = QLineEdit()
        self.field.setEchoMode(QLineEdit.Password)
        self.field.setPlaceholderText("hf_…")
        layout.addWidget(self.field)
        actions = QHBoxLayout()
        self.validate_button = QPushButton("Validate")
        self.validate_button.clicked.connect(self._validate)
        self.open_button = QPushButton("Get a token")
        self.open_button.clicked.connect(lambda: open_url("https://huggingface.co/settings/tokens"))
        clear = QPushButton("Remove token")
        clear.setProperty("variant", "danger")
        clear.clicked.connect(self._clear)
        actions.addWidget(self.validate_button)
        actions.addWidget(self.open_button)
        actions.addWidget(clear)
        actions.addStretch(1)
        layout.addLayout(actions)
        self.message = muted_label("")
        layout.addWidget(self.message)
        buttons = QDialogButtonBox()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        save = QPushButton("Save token")
        save.setProperty("variant", "primary")
        save.clicked.connect(self._save)
        buttons.addButton(cancel, QDialogButtonBox.RejectRole)
        buttons.addButton(save, QDialogButtonBox.AcceptRole)
        layout.addWidget(buttons)
        saved, text = hf.token_status()
        self.status.setText(text)
        self.status.setStyleSheet(
            f"color: {current_palette()['success' if saved else 'text_muted']};"
        )

    def _save(self) -> None:
        token = self.field.text().strip()
        if not token:
            self.message.setText("Enter a token, or use Remove token.")
            return
        try:
            self.hf.set_token(token)
        except Exception as exc:
            self.message.setText(f"Could not save the token: {exc}")
            return
        self.accept()

    def _clear(self) -> None:
        try:
            self.hf.set_token("")
        except Exception as exc:
            self.message.setText(f"Could not remove the token: {exc}")
            return
        self.field.clear()
        self.status.setText("No token saved (public models only)")
        self.message.setText("Token removed.")

    def _validate(self) -> None:
        token = self.field.text().strip()
        if token:
            try:
                self.hf.set_token(token)
            except Exception as exc:
                self.message.setText(str(exc))
                return
        if self._worker is not None and self._worker.isRunning():
            return
        self.validate_button.setEnabled(False)
        self.message.setText("Checking with Hugging Face…")

        worker = TaskWorker(self.hf.validate_token)
        worker.succeeded.connect(self._validated)
        worker.failed.connect(lambda m, d: self._validated((False, m)))
        worker.finished.connect(worker.deleteLater)
        self._worker = worker
        worker.start()

    def _validated(self, result: Any) -> None:
        self.validate_button.setEnabled(True)
        ok, message = result if isinstance(result, tuple) else (False, str(result))
        self.message.setText(message)
        self.message.setStyleSheet(
            f"color: {current_palette()['success' if ok else 'danger']};"
        )
        self.status.setText(
            "Token stored in Windows Credential Manager" if ok else self.hf.token_status()[1]
        )


class DownloadRow(QWidget):
    cancel_requested = Signal(str)
    resume_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, task: DownloadTask, parent: QWidget | None = None):
        super().__init__(parent)
        self.task_id = task.id
        self.setObjectName("downloadRow")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)
        top = QHBoxLayout()
        top.setSpacing(8)
        self.title = QLabel(Path(task.dest_path).name)
        self.title.setStyleSheet("font-weight: 600;")
        self.title.setToolTip(task.dest_path)
        top.addWidget(self.title, 1)
        self.status = QLabel("")
        self.status.setProperty("variant", "faint")
        self.status.setStyleSheet("font-size: 8.5pt;")
        top.addWidget(self.status)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setProperty("variant", "ghost")
        self.cancel_button.clicked.connect(lambda: self.cancel_requested.emit(self.task_id))
        self.resume_button = QPushButton("Resume")
        self.resume_button.setProperty("variant", "ghost")
        self.resume_button.clicked.connect(lambda: self.resume_requested.emit(self.task_id))
        self.remove_button = QPushButton("Remove")
        self.remove_button.setProperty("variant", "ghost")
        self.remove_button.clicked.connect(lambda: self.remove_requested.emit(self.task_id))
        for button in (self.cancel_button, self.resume_button, self.remove_button):
            button.setCursor(Qt.PointingHandCursor)
            top.addWidget(button)
        layout.addLayout(top)
        self.progress = QProgressBar()
        self.progress.setFixedHeight(6)
        self.progress.setTextVisible(False)
        layout.addWidget(self.progress)
        self.detail = muted_label("")
        self.detail.setStyleSheet("font-size: 8pt;")
        self.detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.detail)
        self.update_task(task)

    def update_task(self, task: DownloadTask) -> None:
        self.title.setText(Path(task.dest_path).name)
        self.title.setToolTip(f"{task.repo_id}\n{task.dest_path}")
        if task.total_bytes:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(task.percent * 10))
        else:
            self.progress.setRange(0, 0)
        status = task.status
        labels = {
            "queued": "Queued",
            "downloading": f"{task.percent:.1f}%",
            "verifying": "Verifying checksum…",
            "completed": "Completed",
            "cancelled": "Cancelled",
            "paused": "Paused",
            "cancelling": "Stopping…",
            "error": "Error",
        }
        label = labels.get(status, status.title())
        detail = task.size_label
        if task.status == "downloading" and task.eta_label:
            detail += f" · {task.eta_label}"
        if task.error:
            detail = task.error
            self.detail.setStyleSheet(
                f"color: {current_palette()['danger']}; font-size: 8pt;"
            )
            self.detail.setToolTip(task.error)
        else:
            self.detail.setStyleSheet(
                f"color: {current_palette()['text_faint']}; font-size: 8pt;"
            )
            self.detail.setToolTip("")
        self.detail.setText(detail)
        self.status.setText(label)
        self.status.setStyleSheet(
            "font-size: 8.5pt; color: "
            + {
                "completed": current_palette()["success"],
                "error": current_palette()["danger"],
                "downloading": current_palette()["accent"],
                "verifying": current_palette()["warning"],
            }.get(task.status, current_palette()["text_faint"])
            + ";"
        )
        self.cancel_button.setVisible(task.can_cancel)
        self.resume_button.setVisible(task.can_resume)
        self.remove_button.setVisible(task.can_remove)
        self.progress.setVisible(status not in {"completed", "cancelled", "paused", "error"})
        self.remove_button.setText("Remove")


class DiscoverPage(QWidget):
    library_changed = Signal()
    show_status_message = Signal(str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.hf: HuggingFaceClient = services.hf
        self.downloads: DownloadManager = services.downloads
        self._results: list[HFModel] = []
        self._details: HFRepoDetails | None = None
        self._search_worker: TaskWorker | None = None
        self._detail_worker: TaskWorker | None = None
        self._workers: list[TaskWorker] = []
        self._download_rows: dict[str, DownloadRow] = {}
        self._completed_seen: set[str] = set()
        self._initial_search_done = False
        self._searching = False
        self._search_pending = False
        self._loading_more = False
        self._limit = 25
        self._destination = Path(str(services.settings.get("general.model_directory")))
        self._build()
        self._poll_timer = QTimer(self)
        self._poll_timer.setInterval(500)
        self._poll_timer.timeout.connect(self._poll_downloads)
        self._poll_timer.start()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(14)

        top = QHBoxLayout()
        top.setSpacing(12)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Hugging Face for GGUF models…")
        self.search_input.setClearButtonEnabled(True)
        self.search_input.returnPressed.connect(self.search)
        top.addWidget(self.search_input, 1)
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("Most downloaded", "downloads")
        self.sort_combo.addItem("Most liked", "likes")
        self.sort_combo.addItem("Recently updated", "lastModified")
        self.sort_combo.currentIndexChanged.connect(self.search)
        top.addWidget(self.sort_combo)
        self.gguf_check = QCheckBox("GGUF only")
        self.gguf_check.setChecked(True)
        self.gguf_check.stateChanged.connect(self.search)
        top.addWidget(self.gguf_check)
        self.search_button = QPushButton("Search")
        self.search_button.setProperty("variant", "primary")
        self.search_button.clicked.connect(self.search)
        top.addWidget(self.search_button)
        self.token_button = QPushButton("HF Token")
        self.token_button.setProperty("variant", "ghost")
        self.token_button.setToolTip(self.hf.token_status()[1])
        self.token_button.clicked.connect(self._open_token_dialog)
        top.addWidget(self.token_button)
        layout.addLayout(top)

        self.search_status = muted_label("")
        self.search_status.setStyleSheet("font-size: 8.5pt;")
        layout.addWidget(self.search_status)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Results.
        results_card = Card()
        results_layout = QVBoxLayout(results_card)
        results_layout.setContentsMargins(10, 10, 10, 10)
        results_layout.setSpacing(6)
        results_header = QHBoxLayout()
        results_header.addWidget(section_title("Search results"))
        results_header.addStretch(1)
        self.load_more_button = QPushButton("Load more")
        self.load_more_button.setProperty("variant", "ghost")
        self.load_more_button.clicked.connect(self.load_more)
        self.load_more_button.hide()
        results_header.addWidget(self.load_more_button)
        results_layout.addLayout(results_header)
        self.results_list = QListWidget()
        self.results_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.results_list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.results_list.setSpacing(3)
        self.results_list.currentItemChanged.connect(self._result_selected)
        results_layout.addWidget(self.results_list, 1)
        self.results_empty = QLabel("Search for a model, or press Search to explore GGUF models.")
        self.results_empty.setProperty("variant", "faint")
        self.results_empty.setWordWrap(True)
        self.results_empty.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.results_empty)
        splitter.addWidget(results_card)

        # Details.
        details_card = Card()
        details_layout = QVBoxLayout(details_card)
        details_layout.setContentsMargins(16, 14, 16, 14)
        details_layout.setSpacing(12)
        self.details_stack = QVBoxLayout()
        self.details_title = QLabel("Select a model")
        self.details_title.setObjectName("sectionTitle")
        self.details_title.setWordWrap(True)
        self.details_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.details_meta = muted_label("")
        self.details_license = muted_label("")
        self.details_arch = muted_label("")
        self.card_link = QPushButton("View model card & license")
        self.card_link.setProperty("variant", "ghost")
        self.card_link.clicked.connect(self._open_card)
        self.card_link.hide()
        self.gated_label = QLabel("Gated repository — you may need an accepted license and token.")
        self.gated_label.setWordWrap(True)
        self.gated_label.setStyleSheet(f"color: {current_palette()['warning']};")
        self.gated_label.hide()
        self.details_stack.addWidget(self.details_title)
        self.details_stack.addWidget(self.details_meta)
        self.details_stack.addWidget(self.details_license)
        self.details_stack.addWidget(self.details_arch)
        self.details_stack.addWidget(self.card_link)
        self.details_stack.addWidget(self.gated_label)
        self.file_table = QTableWidget(0, 4)
        self.file_table.setHorizontalHeaderLabels(["File", "Quant", "Size", ""])
        self.file_table.verticalHeader().setVisible(False)
        self.file_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.file_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.file_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.file_table.setAlternatingRowColors(True)
        self.file_table.setShowGrid(False)
        self.file_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.file_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.file_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.file_table.verticalHeader().setDefaultSectionSize(38)
        self.file_table.itemSelectionChanged.connect(self._file_selected)
        self.file_table.itemDoubleClicked.connect(lambda _item: self._download_selected())
        self.file_table.setMinimumHeight(170)
        self.file_table.setToolTip("Double-click a file to download it")
        save_row = QHBoxLayout()
        save_label = muted_label("Save to")
        save_label.setStyleSheet("font-size: 9pt;")
        self.destination_label = QLineEdit(str(self._destination))
        self.destination_label.setReadOnly(True)
        self.destination_label.setToolTip("Change this in Settings or with Browse")
        browse = QPushButton("Browse…")
        browse.clicked.connect(self._choose_destination)
        open_folder = QPushButton("Open folder")
        open_folder.clicked.connect(self._open_destination)
        save_row.addWidget(save_label)
        save_row.addWidget(self.destination_label, 1)
        save_row.addWidget(browse)
        save_row.addWidget(open_folder)
        details_layout.addLayout(self.details_stack)
        self.download_button = QPushButton("Download selected file")
        self.download_button.setProperty("variant", "primary")
        self.download_button.setEnabled(False)
        self.download_button.clicked.connect(self._download_selected)
        details_layout.addWidget(self.file_table, 1)
        details_layout.addLayout(save_row)
        details_layout.addWidget(self.download_button)
        splitter.addWidget(details_card)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        splitter.setSizes([360, 640])
        layout.addWidget(splitter, 1)

        # Downloads panel.
        downloads_card = Card(soft=True)
        downloads_layout = QVBoxLayout(downloads_card)
        downloads_layout.setContentsMargins(12, 8, 12, 8)
        downloads_layout.setSpacing(5)
        downloads_header = QHBoxLayout()
        downloads_header.addWidget(section_title("Downloads"))
        self.downloads_summary = muted_label("No active downloads")
        self.downloads_summary.setWordWrap(False)
        self.downloads_summary.setStyleSheet("font-size: 8.5pt;")
        downloads_header.addWidget(self.downloads_summary)
        downloads_header.addStretch(1)
        open_downloads = QPushButton("Open downloads folder")
        open_downloads.setProperty("variant", "ghost")
        open_downloads.clicked.connect(self._open_downloads_folder)
        downloads_header.addWidget(open_downloads)
        downloads_layout.addLayout(downloads_header)
        self.downloads_scroll = QScrollArea()
        self.downloads_scroll.setWidgetResizable(True)
        self.downloads_scroll.setMinimumHeight(70)
        self.downloads_scroll.setMaximumHeight(190)
        self.downloads_container = QWidget()
        self.downloads_container_layout = QVBoxLayout(self.downloads_container)
        self.downloads_container_layout.setContentsMargins(0, 0, 0, 0)
        self.downloads_container_layout.setSpacing(3)
        self.downloads_container_layout.addStretch(1)
        self.downloads_scroll.setWidget(self.downloads_container)
        downloads_layout.addWidget(self.downloads_scroll)
        self.downloads_empty = muted_label(
            "Downloads appear here with progress, cancel and resume controls."
        )
        self.downloads_empty.setStyleSheet("font-size: 8.5pt;")
        downloads_layout.addWidget(self.downloads_empty)
        layout.addWidget(downloads_card)

    # -- search ------------------------------------------------------------
    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if not self._initial_search_done:
            self._initial_search_done = True
            QTimer.singleShot(250, self.search)

    def disable_automatic_search(self) -> None:
        """Keep headless/package smoke checks independent of network access."""
        self._initial_search_done = True

    def search(self, *_args) -> None:
        if self._searching:
            # Keep the latest user intent. The current Hub request cannot be
            # interrupted, but the new query/sort/filter is run as soon as it
            # finishes instead of silently dropping the user's Search click.
            self._search_pending = True
            return
        query = self.search_input.text().strip()
        sort = str(self.sort_combo.currentData())
        gguf_only = self.gguf_check.isChecked()
        self._limit = 25
        self._searching = True
        self.search_button.setEnabled(False)
        self.search_button.setText("Searching…")
        self.search_status.setText("Contacting Hugging Face…")
        self.load_more_button.hide()

        def work():
            return self.hf.search(
                query, limit=self._limit, sort=sort, gguf_only=gguf_only
            )

        def done(models):
            self._searching = False
            if self._search_pending:
                self._search_pending = False
                QTimer.singleShot(0, self.search)
                return
            self.search_button.setEnabled(True)
            self.search_button.setText("Search")
            self._results = list(models)
            self._populate_results(models)
            self.search_status.setText(
                f"{len(models)} GGUF repositories" if models else "No models found."
            )
            self.load_more_button.setVisible(len(models) >= self._limit)
            localize_widget_tree(self)

        def failed(message, detail):
            self._searching = False
            if self._search_pending:
                self._search_pending = False
                QTimer.singleShot(0, self.search)
                return
            self.search_button.setEnabled(True)
            self.search_button.setText("Search")
            self.search_status.setText(message)
            self.search_status.setStyleSheet(
                f"color: {current_palette()['danger']}; font-size: 8.5pt;"
            )
            self.search_status.setToolTip(detail)
            self.show_status_message.emit(message)
            localize_widget_tree(self)

        self._run_task(work, done, failed)

    def load_more(self) -> None:
        if self._searching:
            return
        self._searching = True
        self._limit += 25
        self.search_button.setEnabled(False)
        query = self.search_input.text().strip()
        sort = str(self.sort_combo.currentData())
        gguf_only = self.gguf_check.isChecked()

        def work():
            return self.hf.search(query, limit=self._limit, sort=sort, gguf_only=gguf_only)

        def done(models):
            self._searching = False
            if self._search_pending:
                self._search_pending = False
                QTimer.singleShot(0, self.search)
                return
            self.search_button.setEnabled(True)
            self._results = list(models)
            self._populate_results(models)
            self.search_status.setText(f"{len(models)} GGUF repositories")
            self.load_more_button.setVisible(len(models) >= self._limit)

        def failed(message, detail):
            self._searching = False
            if self._search_pending:
                self._search_pending = False
                QTimer.singleShot(0, self.search)
                return
            self.search_button.setEnabled(True)
            self.search_status.setText(message)
            self.search_status.setStyleSheet(
                f"color: {current_palette()['danger']}; font-size: 8.5pt;"
            )

        self._run_task(work, done, failed)

    def _populate_results(self, models: list[HFModel]) -> None:
        previous = self._selected_repo_id()
        self.results_list.blockSignals(True)
        self.results_list.clear()
        for model in models:
            item = QListWidgetItem()
            item.setData(Qt.UserRole, model.repo_id)
            item.setSizeHint(QSize(100, 64))
            widget = self._result_widget(model)
            self.results_list.addItem(item)
            self.results_list.setItemWidget(item, widget)
        self.results_list.blockSignals(False)
        self.results_empty.setVisible(not models)
        if previous:
            for row in range(self.results_list.count()):
                if self.results_list.item(row).data(Qt.UserRole) == previous:
                    self.results_list.setCurrentRow(row)
                    break
        if models and self.results_list.currentRow() < 0:
            self.results_list.setCurrentRow(0)

    def _result_widget(self, model: HFModel) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        title = QLabel(model.repo_id)
        title.setStyleSheet("font-weight: 600;")
        title.setToolTip(model.repo_id)
        meta_bits = [
            f"{model.downloads:,} downloads",
            f"{model.likes:,} likes",
            model.license_label,
        ]
        if model.parameters:
            meta_bits.insert(0, model.parameters)
        meta = QLabel(" · ".join(meta_bits))
        meta.setProperty("variant", "faint")
        meta.setStyleSheet("font-size: 8pt;")
        meta.setWordWrap(False)
        tags = QLabel(
            "  ".join(
                tag for tag in model.tags[:5] if not tag.startswith("license:")
            )
        )
        tags.setProperty("variant", "faint")
        tags.setStyleSheet("font-size: 7.5pt;")
        tags.setWordWrap(False)
        layout.addWidget(title)
        layout.addWidget(meta)
        if model.tags:
            layout.addWidget(tags)
        return widget

    def _result_selected(self, current: QListWidgetItem | None, _previous=None) -> None:
        if current is None:
            return
        repo_id = str(current.data(Qt.UserRole))
        self._load_details(repo_id)

    def _load_details(self, repo_id: str) -> None:
        self.details_title.setText(repo_id)
        self.details_meta.setText("Loading repository…")
        self.details_license.setText("")
        self.details_arch.setText("")
        self.file_table.setRowCount(0)
        self.download_button.setEnabled(False)
        self.card_link.hide()
        self.gated_label.hide()
        self._details = None

        def work():
            return self.hf.repo_details(repo_id)

        def done(details: HFRepoDetails):
            if self._selected_repo_id() != repo_id:
                return
            self._details = details
            self._render_details(details)

        def failed(message, detail):
            if self._selected_repo_id() != repo_id:
                return
            self.details_meta.setText(message)
            self.details_meta.setStyleSheet(
                f"color: {current_palette()['danger']};"
            )
            self.show_status_message.emit(message)

        self._run_task(work, done, failed)

    def _render_details(self, details: HFRepoDetails) -> None:
        self.details_title.setText(details.repo_id)
        self.details_meta.setText(
            f"{details.author or 'unknown author'} · {details.downloads:,} downloads · "
            f"{details.likes:,} likes · updated {details.last_modified or 'unknown'}"
        )
        self.details_meta.setStyleSheet("")
        self.details_license.setText(f"License: {details.license or 'not specified'}")
        bits = []
        if details.pipeline_tag:
            bits.append(details.pipeline_tag)
        gguf_tags = [tag for tag in details.tags if tag.startswith("gguf")]
        if gguf_tags:
            bits.append(", ".join(gguf_tags[:3]))
        self.details_arch.setText(" · ".join(bits))
        self.card_link.setVisible(True)
        self.gated_label.setVisible(
            bool(details.gated) and not self.hf.token_status()[0]
        )
        self._populate_files(details)
        if not any(f.filename.lower().endswith(".gguf") for f in details.files):
            self.details_meta.setText(
                "This repository has no GGUF files visible. It may be gated or use a "
                "different format."
            )
        localize_widget_tree(self)

    def _populate_files(self, details: HFRepoDetails) -> None:
        self.file_table.setRowCount(0)
        for file in details.files:
            row = self.file_table.rowCount()
            self.file_table.insertRow(row)
            name_item = QTableWidgetItem(file.filename)
            name_item.setToolTip(
                f"{file.filename}\n{human_size(file.size)}"
                + (f"\nSHA256: {file.sha256[:16]}…" if file.sha256 else "")
            )
            quant_item = QTableWidgetItem(file.quant or "—")
            size_item = QTableWidgetItem(file.size_label)
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.file_table.setItem(row, 0, name_item)
            self.file_table.setItem(row, 1, quant_item)
            self.file_table.setItem(row, 2, size_item)
            button = QPushButton("Download")
            button.setProperty("variant", "ghost")
            button.setCursor(Qt.PointingHandCursor)
            button.clicked.connect(lambda _=False, r=row: self._download_row(r))
            self.file_table.setCellWidget(row, 3, button)
        if details.files:
            self.file_table.selectRow(0)
        self._file_selected()

    def _file_selected(self) -> None:
        file = self._selected_file()
        self.download_button.setEnabled(file is not None)
        if file is not None:
            self.download_button.setText(
                f"Download {Path(file.filename).name} ({file.size_label})"
            )
        else:
            self.download_button.setText("Download selected file")
        localize_widget_tree(self)

    def _selected_file(self):
        if self._details is None:
            return None
        row = self.file_table.currentRow()
        if row < 0 or row >= len(self._details.files):
            return None
        return self._details.files[row]

    def _selected_repo_id(self) -> str:
        item = self.results_list.currentItem()
        return str(item.data(Qt.UserRole)) if item else ""

    def _download_row(self, row: int) -> None:
        if self._details is None or row < 0 or row >= len(self._details.files):
            return
        self.file_table.selectRow(row)
        self._download_selected()

    def _download_selected(self) -> None:
        file = self._selected_file()
        details = self._details
        if file is None or details is None:
            return
        if details.gated and not self.hf.token_status()[0]:
            show_error(
                self,
                "Gated model",
                "This repository is gated. Accept its license on the Hugging Face model "
                "card, then add a Hugging Face token in Discover.",
            )
            return
        try:
            task = self.downloads.start(
                details.repo_id,
                file.filename,
                file.download_url,
                self._destination,
                expected_sha256=file.sha256,
                total_bytes=file.size or 0,
            )
        except Exception as exc:
            show_error(self, "Could not start download", str(exc))
            return
        if task.status == "error" and not task.part_path.exists():
            show_error(self, "Download", task.error)
            return
        self.show_status_message.emit(f"Downloading {Path(task.dest_path).name}")
        self._poll_downloads(force=True)

    def _choose_destination(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self, "Choose download folder", str(self._destination)
        )
        if not directory:
            return
        self._destination = Path(directory)
        self.destination_label.setText(directory)
        self.services.settings.set("general.model_directory", directory)
        self.services.library.model_directory = self._destination

        def done(_summary) -> None:
            self.library_changed.emit()
            self.show_status_message.emit(f"Model folder changed to {directory}")

        def failed(message: str, _detail: str) -> None:
            self.show_status_message.emit(f"Could not scan the new folder: {message}")

        self._run_task(lambda: self.services.library.refresh(self._destination), done, failed)

    def _open_destination(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._destination)))

    def _open_downloads_folder(self) -> None:
        path = Path(str(self.services.settings.get("general.model_directory")))
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def _open_card(self) -> None:
        if self._details:
            open_url(self._details.card_url)

    def _open_token_dialog(self) -> None:
        dialog = TokenDialog(self.hf, self)
        if dialog.exec() == QDialog.Accepted:
            self.token_button.setToolTip(self.hf.token_status()[1])
            self.show_status_message.emit("Hugging Face token saved")
            self.search()

    # -- downloads ---------------------------------------------------------
    def _poll_downloads(self, force: bool = False) -> None:
        tasks = self.downloads.snapshot()
        active = [t for t in tasks if t.status in {"queued", "downloading", "verifying", "cancelling"}]
        self.downloads_summary.setText(
            f"{len(active)} active · {len(tasks)} total" if tasks else "No active downloads"
        )
        self.downloads_empty.setVisible(not tasks)
        self.downloads_scroll.setVisible(bool(tasks))
        seen: set[str] = set()
        for task in tasks:
            seen.add(task.id)
            if task.status == "completed" and task.id not in self._completed_seen:
                self._completed_seen.add(task.id)
                self.library_changed.emit()
                self.show_status_message.emit(
                    f"Download complete: {Path(task.dest_path).name}"
                )
            row = self._download_rows.get(task.id)
            if row is None:
                row = DownloadRow(task)
                row.cancel_requested.connect(self._cancel_download)
                row.resume_requested.connect(self._resume_download)
                row.remove_requested.connect(self._remove_download)
                self.downloads_container_layout.insertWidget(
                    self.downloads_container_layout.count() - 1, row
                )
                self._download_rows[task.id] = row
            row.update_task(task)
        localize_widget_tree(self)
        for task_id in list(self._download_rows):
            if task_id not in seen:
                row = self._download_rows.pop(task_id)
                self.downloads_container_layout.removeWidget(row)
                row.deleteLater()

    def _cancel_download(self, task_id: str) -> None:
        self.downloads.cancel(task_id)
        self._poll_downloads(force=True)

    def _resume_download(self, task_id: str) -> None:
        if not self.downloads.resume(task_id):
            self.show_status_message.emit("Cannot resume: the partial file is missing.")
        self._poll_downloads(force=True)

    def _remove_download(self, task_id: str) -> None:
        self.downloads.remove(task_id, delete_partial=False)
        self._poll_downloads(force=True)

    # -- worker helpers ----------------------------------------------------
    def _run_task(self, fn, done, failed) -> None:
        worker = TaskWorker(fn)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)

        def cleanup():
            try:
                self._workers.remove(worker)
            except ValueError:
                pass

        worker.finished.connect(cleanup)
        worker.start()

    def refresh_destination(self) -> None:
        self._destination = Path(str(self.services.settings.get("general.model_directory")))
        self.destination_label.setText(str(self._destination))

    def shutdown(self) -> None:
        self._poll_timer.stop()
        self.downloads.shutdown()
