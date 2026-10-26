"""My Models: GGUF library, load/unload, per-model settings and file actions."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.model_manager import ModelEntry, ModelManagerError
from app.core.types import LoadResult
from app.ui.dialogs import ModelLoadDialog, confirm, show_error
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import Card, muted_label, section_title
from app.workers.workers import ModelLoadWorker, TaskWorker, UnloadWorker

log = logging.getLogger(__name__)

_COLUMNS = ["Model Name", "Size", "Quant", "Architecture", "Context", "Path", "Status"]


class ModelsPage(QWidget):
    models_changed = Signal()
    model_changed = Signal()
    show_status_message = Signal(str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.library: ModelManager = services.library
        self._load_worker: Any = None
        self._unload_worker: Any = None
        self._workers: list[Any] = []
        self._build()
        self.refresh()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 14)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.addWidget(section_title("Model library"))
        self.summary = muted_label("")
        self.summary.setStyleSheet("font-size: 9pt;")
        header.addWidget(self.summary)
        header.addStretch(1)
        self.add_button = QPushButton("Add GGUF file")
        self.add_button.setProperty("variant", "primary")
        self.add_button.setCursor(Qt.PointingHandCursor)
        self.add_button.clicked.connect(self._add_file)
        header.addWidget(self.add_button)
        self.rescan_button = QPushButton("Rescan folder")
        self.rescan_button.setCursor(Qt.PointingHandCursor)
        self.rescan_button.clicked.connect(self._rescan)
        header.addWidget(self.rescan_button)
        self.open_dir_button = QPushButton("Open folder")
        self.open_dir_button.clicked.connect(self._open_model_dir)
        header.addWidget(self.open_dir_button)
        layout.addLayout(header)

        self.directory_label = muted_label("")
        self.directory_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.directory_label.setStyleSheet("font-size: 8.5pt;")
        layout.addWidget(self.directory_label)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.setSortingEnabled(True)
        header_view = self.table.horizontalHeader()
        header_view.setSectionResizeMode(0, QHeaderView.Stretch)
        for column in range(1, len(_COLUMNS)):
            header_view.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header_view.setSectionResizeMode(5, QHeaderView.Stretch)
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.itemDoubleClicked.connect(lambda _item: self._load_with_settings())
        layout.addWidget(self.table, 1)

        self.empty_label = muted_label(
            "No GGUF models yet. Download one in Discover, or add an existing .gguf file."
        )
        self.empty_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.empty_label)

        actions = QHBoxLayout()
        actions.setSpacing(9)
        self.load_button = QPushButton("Load")
        self.load_button.setProperty("variant", "primary")
        self.load_button.clicked.connect(self._load_selected)
        self.load_settings_button = QPushButton("Load with settings…")
        self.load_settings_button.clicked.connect(self._load_with_settings)
        self.unload_button = QPushButton("Unload")
        self.unload_button.clicked.connect(self._unload)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self._reveal_selected)
        self.remove_button = QPushButton("Remove from Library")
        self.remove_button.clicked.connect(self._remove_selected)
        self.delete_button = QPushButton("Delete File…")
        self.delete_button.setProperty("variant", "danger")
        self.delete_button.clicked.connect(self._delete_selected)
        for button in (
            self.load_button,
            self.load_settings_button,
            self.unload_button,
            self.open_folder_button,
            self.remove_button,
            self.delete_button,
        ):
            button.setCursor(Qt.PointingHandCursor)
            actions.addWidget(button)
        actions.addStretch(1)
        layout.addLayout(actions)

        hint = muted_label(
            "Remove from Library only forgets the entry; the GGUF file stays on disk. "
            "Delete File removes the actual file after confirmation. Loading uses the "
            "saved per-model context, GPU layer, thread and batch settings."
        )
        hint.setStyleSheet("font-size: 8.5pt;")
        layout.addWidget(hint)

        self.loading_label = muted_label("")
        self.loading_label.setStyleSheet(f"color: {current_palette()['warning']};")
        layout.addWidget(self.loading_label)
        self._selection_changed()

    # -- data --------------------------------------------------------------
    def refresh(self) -> None:
        selected_id = self._selected_id()
        entries = self.library.entries()
        status = self.services.inference.status()
        loaded_path = ""
        if status.get("target"):
            loaded_path = status["target"].get("path") or ""
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        for entry in entries:
            row = self.table.rowCount()
            self.table.insertRow(row)
            name_item = QTableWidgetItem(entry.name)
            name_item.setData(Qt.UserRole, entry.id)
            name_item.setToolTip(
                f"{entry.path}\n"
                + (f"{entry.parameters_label} parameters\n" if entry.parameter_count else "")
                + (entry.metadata_error or "")
            )
            self.table.setItem(row, 0, name_item)
            size_item = QTableWidgetItem(entry.size_label)
            size_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.table.setItem(row, 1, size_item)
            self.table.setItem(row, 2, QTableWidgetItem(entry.quant or "—"))
            self.table.setItem(row, 3, QTableWidgetItem(entry.architecture or "—"))
            self.table.setItem(row, 4, QTableWidgetItem(entry.context_label))
            path_item = QTableWidgetItem(_short_path(entry.path, 3))
            path_item.setToolTip(entry.path)
            self.table.setItem(row, 5, path_item)
            is_loaded = (
                entry.status == "ready"
                and loaded_path
                and os.path.normcase(str(Path(entry.path).resolve()))
                == os.path.normcase(str(Path(loaded_path).resolve()))
            )
            status_text = "Loaded" if is_loaded else entry.status_label
            status_item = QTableWidgetItem(status_text)
            if is_loaded:
                status_item.setForeground(Qt.GlobalColor.green)
            elif entry.status == "invalid":
                status_item.setForeground(Qt.GlobalColor.red)
            elif entry.status == "missing":
                status_item.setForeground(Qt.GlobalColor.gray)
            if entry.metadata_error:
                status_item.setToolTip(entry.metadata_error)
            self.table.setItem(row, 6, status_item)
            if entry.id == selected_id:
                self.table.selectRow(row)
        self.table.setSortingEnabled(True)
        self.empty_label.setVisible(not entries)
        model_dir = str(self.services.settings.get("general.model_directory"))
        self.directory_label.setText(
            f"Library folder: {model_dir}  ·  {len(entries)} entries · "
            f"{sum(1 for e in entries if e.status == 'ready')} ready"
        )
        self._selection_changed()

    def _selected_entry(self) -> ModelEntry | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        return self.library.get(str(item.data(Qt.UserRole)))

    def _selected_id(self) -> str:
        entry = self._selected_entry()
        return entry.id if entry else ""

    def _selection_changed(self) -> None:
        entry = self._selected_entry()
        has_selection = entry is not None
        valid = bool(entry and entry.status == "ready")
        status = self.services.inference.status()
        loaded_path = ""
        if status.get("target"):
            loaded_path = status["target"].get("path") or ""
        is_loaded = bool(
            entry
            and loaded_path
            and os.path.normcase(str(Path(entry.path).resolve()))
            == os.path.normcase(str(Path(loaded_path).resolve()))
        )
        busy = status.get("state") == "loading"
        self.load_button.setEnabled(valid and not busy and not is_loaded)
        self.load_settings_button.setEnabled(valid and not busy and not is_loaded)
        self.unload_button.setEnabled(is_loaded and not busy)
        self.open_folder_button.setEnabled(has_selection)
        self.remove_button.setEnabled(has_selection)
        self.delete_button.setEnabled(has_selection and not is_loaded)
        self.loading_label.setText(
            f"Loading {status.get('display_name')}…" if busy else ""
        )

    def update_inference_status(self, status: dict[str, Any]) -> None:
        self.refresh()

    # -- actions -----------------------------------------------------------
    def _add_file(self) -> None:
        start_dir = str(self.services.settings.get("general.model_directory"))
        path, _ = QFileDialog.getOpenFileName(
            self, "Add GGUF model", start_dir, "GGUF models (*.gguf);;All files (*.*)"
        )
        if not path:
            return
        try:
            entry = self.library.add_file(path)
        except ModelManagerError as exc:
            show_error(self, "Could not add model", str(exc))
            return
        except Exception as exc:
            show_error(self, "Could not add model", str(exc))
            return
        self.refresh()
        self.models_changed.emit()
        self.show_status_message.emit(f"Added {entry.name} to the library")

    def _rescan(self) -> None:
        self.rescan_button.setEnabled(False)
        self.rescan_button.setText("Scanning…")
        directory = self.services.settings.get("general.model_directory")

        def work():
            return self.library.refresh(directory)

        def done(summary):
            self.rescan_button.setEnabled(True)
            self.rescan_button.setText("Rescan folder")
            self.refresh()
            self.models_changed.emit()
            self.show_status_message.emit(
                f"Scan complete: {summary.added} added, {summary.updated} refreshed, "
                f"{summary.missing} missing, {summary.invalid} invalid"
            )

        def failed(message, detail):
            self.rescan_button.setEnabled(True)
            self.rescan_button.setText("Rescan folder")
            show_error(self, "Scan failed", message, detail)

        worker = TaskWorker(work)
        worker.succeeded.connect(done)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        self._workers.append(worker)
        worker.start()

    def _load_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        self._start_load(entry, entry.resolved_options())

    def _load_with_settings(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        dialog = ModelLoadDialog(entry, self.services.settings, self.library, self)
        if dialog.exec() != QDialog.Accepted or dialog.options is None:
            return
        self._start_load(entry, dialog.options)

    def _start_load(self, entry: ModelEntry, options) -> None:
        if self.services.inference.status().get("state") == "loading":
            self.show_status_message.emit("A model is already loading.")
            return
        self.load_button.setEnabled(False)
        self.load_settings_button.setEnabled(False)
        self.loading_label.setText(f"Loading {entry.name}… this can take a while.")
        self.loading_label.setStyleSheet(f"color: {current_palette()['warning']};")
        worker = ModelLoadWorker(
            self.services.inference, entry.target(), options, parent=self
        )

        def finished(result: LoadResult):
            self._load_worker = None
            self.loading_label.setText("")
            if result.ok:
                if result.warning:
                    self.loading_label.setText(result.warning)
                    self.loading_label.setStyleSheet(
                        f"color: {current_palette()['warning']};"
                    )
                self.show_status_message.emit(f"Loaded {entry.name}")
            else:
                self.loading_label.setText("Model load failed")
                self.loading_label.setStyleSheet(
                    f"color: {current_palette()['danger']};"
                )
                show_error(self, "Model load failed", result.message)
            self.refresh()
            self.model_changed.emit()

        def failed(message: str, detail: str):
            self._load_worker = None
            self.loading_label.setText("Model load failed")
            self.loading_label.setStyleSheet(
                f"color: {current_palette()['danger']};"
            )
            show_error(self, "Model load failed", message, detail)
            self.refresh()
            self.model_changed.emit()

        worker.finished_load.connect(finished)
        worker.failed.connect(failed)
        worker.finished.connect(worker.deleteLater)
        self._load_worker = worker
        worker.start()

    def _unload(self) -> None:
        status = self.services.inference.status()
        if status.get("state") != "ready":
            return
        self.unload_button.setEnabled(False)
        self.loading_label.setText("Unloading…")
        worker = UnloadWorker(self.services.inference, parent=self)

        def done():
            self._unload_worker = None
            self.loading_label.setText("")
            self.refresh()
            self.model_changed.emit()
            self.show_status_message.emit("Model unloaded")

        worker.finished_unload.connect(done)
        worker.finished.connect(worker.deleteLater)
        self._unload_worker = worker
        worker.start()

    def _reveal_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        _reveal_in_explorer(Path(entry.path))

    def _open_model_dir(self) -> None:
        directory = Path(str(self.services.settings.get("general.model_directory")))
        directory.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(_url(directory))

    def _remove_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        if not confirm(
            self,
            "Remove from library",
            f"Remove “{entry.name}” from the library?\n\n"
            f"The GGUF file will NOT be deleted:\n{entry.path}",
        ):
            return
        try:
            self.library.remove_from_library(entry.id)
        except Exception as exc:
            show_error(self, "Could not remove entry", str(exc))
            return
        self.refresh()
        self.models_changed.emit()
        self.show_status_message.emit("Removed from library (file kept)")

    def _delete_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        status = self.services.inference.status()
        if status.get("target") and os.path.normcase(
            str(Path(status["target"].get("path") or ""))
        ) == os.path.normcase(str(Path(entry.path))):
            show_error(
                self,
                "Model is loaded",
                "Unload this model before deleting the file.",
            )
            return
        if not confirm(
            self,
            "Delete model file",
            f"Permanently delete this file?\n\n{entry.path}\n\nThis cannot be undone.",
        ):
            return
        try:
            self.library.delete_file(entry.id)
        except ModelManagerError as exc:
            show_error(self, "Could not delete file", str(exc))
            return
        self.refresh()
        self.models_changed.emit()
        self.show_status_message.emit(f"Deleted {entry.name}")

    def shutdown(self) -> None:
        for worker in (self._load_worker, self._unload_worker, *self._workers):
            if worker is not None and worker.isRunning():
                worker.wait(8000)


def _short_path(path: str, keep_parts: int = 3) -> str:
    parts = Path(path).parts
    if len(parts) <= keep_parts:
        return path
    return str(Path(*parts[-keep_parts:]))


def _url(path: Path):
    from PySide6.QtCore import QUrl

    return QUrl.fromLocalFile(str(path))


def _reveal_in_explorer(path: Path) -> None:
    if os.name == "nt":
        try:
            subprocess.Popen(["explorer", "/select,", str(path)])
            return
        except OSError:
            pass
    folder = path.parent if path.suffix else path
    QDesktopServices.openUrl(_url(folder))
