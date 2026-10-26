"""Main window: fixed sidebar navigation, top bar and page wiring."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from PySide6.QtCore import QByteArray, QSize, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.constants import APP_NAME, APP_VERSION
from app.core.settings import SettingsStore
from app.core.types import LoadOptions, LoadResult, ModelTarget
from app.ui.chat import ChatPage
from app.ui.hf_discover import DiscoverPage
from app.ui.dialogs import CloudModelDialog, show_error
from app.ui.icons import app_icon, make_icon
from app.ui.i18n import localize_widget_tree
from app.ui.models import ModelsPage
from app.ui.server import ServerPage
from app.ui.settings import SettingsPage
from app.ui.services import AppServices
from app.ui.theme import current_palette, set_window_theme
from app.ui.widgets import StatusChip
from app.workers.workers import TaskWorker, UnloadWorker

log = logging.getLogger(__name__)

# key, label, icon name, page class attribute
NAV_PAGES: tuple[tuple[str, str, str], ...] = (
    ("chat", "チャット", "chat"),
    ("discover", "モデルを探す", "compass"),
    ("models", "モデル管理", "cube"),
    ("server", "API サーバー", "server"),
    ("settings", "設定", "settings"),
)

_STATE_TEXT = {
    "idle": "モデル未選択",
    "loading": "読み込み中…",
    "unloading": "モデルを解放中…",
    "ready": "準備完了",
    "error": "読み込み失敗",
}


class MainWindow(QMainWindow):
    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.settings: SettingsStore = services.settings
        self._status: dict[str, Any] = services.inference.status()
        self._workers: list[TaskWorker] = []
        self._scan_worker: TaskWorker | None = None
        self._unload_worker: UnloadWorker | None = None
        self._model_dir = str(self.settings.get("general.model_directory", ""))
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.timeout.connect(self._clear_transient_status)

        self._build()
        self._wire()
        self._restore_geometry()
        self._apply_status(self._status)

        services.bridge.event.connect(self._on_inference_event)
        services.bridge.attach(services.inference)

        self.statusBar().setSizeGripEnabled(False)
        self.statusBar().setFixedHeight(24)
        self.setWindowIcon(app_icon())
        localize_widget_tree(self)

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        self.setObjectName("mainWindow")
        root = QWidget()
        root.setObjectName("root")
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._build_sidebar())

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        right_layout.addWidget(self._build_topbar())

        self.stack = QStackedWidget()
        self.pages: dict[str, QWidget] = {}
        page_types = {
            "chat": ChatPage,
            "discover": DiscoverPage,
            "models": ModelsPage,
            "server": ServerPage,
            "settings": SettingsPage,
        }
        for key, _label, _icon in NAV_PAGES:
            page = page_types[key](self.services)
            self.pages[key] = page
            self.stack.addWidget(page)
        right_layout.addWidget(self.stack, 1)
        outer.addWidget(right, 1)
        self.setCentralWidget(root)

        self.resize(1280, 820)
        self.setMinimumSize(1040, 660)

    def _build_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(216)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(12, 16, 12, 12)
        layout.setSpacing(8)

        brand = QHBoxLayout()
        brand.setContentsMargins(6, 0, 4, 0)
        brand.setSpacing(9)
        logo = QLabel()
        logo.setPixmap(app_icon().pixmap(30, 30))
        logo.setFixedSize(30, 30)
        brand.addWidget(logo)
        brand_text = QVBoxLayout()
        brand_text.setContentsMargins(0, 0, 0, 0)
        brand_text.setSpacing(0)
        name = QLabel(APP_NAME)
        name.setObjectName("appName")
        tagline = QLabel(f"{APP_VERSION} · LOCAL + CLOUD")
        tagline.setObjectName("appTagline")
        brand_text.addWidget(name)
        brand_text.addWidget(tagline)
        brand.addLayout(brand_text, 1)
        layout.addLayout(brand)
        layout.addSpacing(14)

        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)
        self.nav_buttons: dict[str, QPushButton] = {}
        self._nav_icons = {key: icon for key, _label, icon in NAV_PAGES}
        colors = current_palette()
        for index, (key, label, icon_name) in enumerate(NAV_PAGES):
            button = QPushButton(f"  {label}")
            button.setProperty("variant", "nav")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            button.setIcon(make_icon(icon_name, colors["text_muted"], colors["accent_hover"]))
            button.setIconSize(QSize(17, 17))
            button.setMinimumHeight(38)
            button.clicked.connect(lambda _checked=False, k=key: self.switch_page(k))
            self.nav_group.addButton(button, index)
            self.nav_buttons[key] = button
            layout.addWidget(button)
        self.nav_buttons["chat"].setChecked(True)

        layout.addStretch(1)

        self.library_summary = QLabel("")
        self.library_summary.setProperty("variant", "faint")
        self.library_summary.setStyleSheet("font-size: 8.5pt; padding: 4px 8px;")
        self.library_summary.setWordWrap(True)
        layout.addWidget(self.library_summary)

        hint = QLabel("クラウドモデルを使う場合を除き、会話はこの PC 内に保存されます。")
        hint.setProperty("variant", "faint")
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 8pt; padding: 2px 8px 0 8px;")
        layout.addWidget(hint)
        return sidebar

    def _build_topbar(self) -> QFrame:
        topbar = QFrame()
        topbar.setObjectName("topbar")
        topbar.setMinimumHeight(64)
        layout = QHBoxLayout(topbar)
        layout.setContentsMargins(22, 10, 22, 10)
        layout.setSpacing(12)

        self.page_title = QLabel("チャット")
        self.page_title.setObjectName("pageTitle")
        layout.addWidget(self.page_title)
        layout.addStretch(1)

        self.model_name = QLabel("モデルが選択されていません")
        self.model_name.setProperty("variant", "muted")
        self.model_name.setStyleSheet("font-size: 9pt;")
        self.model_name.setMaximumWidth(320)
        layout.addWidget(self.model_name)

        self.change_model_button = QPushButton("モデルを選択")
        self.change_model_button.setProperty("variant", "chip")
        self.change_model_button.setCursor(Qt.PointingHandCursor)
        self.change_model_button.clicked.connect(self._show_model_menu)
        layout.addWidget(self.change_model_button)

        self.status_chip = StatusChip()
        self.status_chip.set_status("idle", "モデル未選択")
        layout.addWidget(self.status_chip)
        return topbar

    # -- wiring ------------------------------------------------------------
    def _wire(self) -> None:
        chat = self.pages["chat"]
        discover = self.pages["discover"]
        models = self.pages["models"]
        server = self.pages["server"]
        settings = self.pages["settings"]

        chat.cloud_requested.connect(self.choose_cloud_model)
        chat.models_page_requested.connect(lambda: self.switch_page("models"))
        chat.generation_active.connect(self._generation_active)
        chat.show_status_message.connect(self.show_status_message)

        discover.library_changed.connect(self._library_changed)
        discover.show_status_message.connect(self.show_status_message)

        models.models_changed.connect(self._library_changed)
        models.model_changed.connect(self._refresh_status)
        models.show_status_message.connect(self.show_status_message)

        server.settings_requested.connect(self._open_server_settings)
        server.show_status_message.connect(self.show_status_message)

        settings.settings_saved.connect(self._settings_saved)
        settings.cloud_changed.connect(self._refresh_status)
        settings.history_cleared.connect(self._history_cleared)
        settings.show_status_message.connect(self.show_status_message)

        self.switch_page("chat")

    # -- navigation --------------------------------------------------------
    def switch_page(self, key: str) -> None:
        if key not in self.pages:
            return
        page = self.pages[key]
        self.stack.setCurrentWidget(page)
        button = self.nav_buttons.get(key)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        label = next(item[1] for item in NAV_PAGES if item[0] == key)
        self.page_title.setText(label)
        self.setWindowTitle(f"{APP_NAME} — {label}")

        try:
            if key == "chat":
                page.refresh_conversations()
            elif key == "models":
                page.refresh()
            elif key == "discover":
                page.refresh_destination()
            elif key == "server":
                page.refresh()
        except Exception:
            log.exception("failed to refresh page %s", key)
        localize_widget_tree(self)

    # -- model selection ---------------------------------------------------
    def _show_model_menu(self) -> None:
        from PySide6.QtWidgets import QMenu

        menu = QMenu(self)
        local_action = menu.addAction("Choose local model…")
        local_action.triggered.connect(lambda: self.switch_page("models"))
        cloud_action = menu.addAction("Choose cloud model…")
        cloud_action.triggered.connect(self.choose_cloud_model)
        menu.addSeparator()
        unload_action = menu.addAction("Unload model")
        state = self._status.get("state")
        unload_action.setEnabled(state in {"ready", "loading"})
        unload_action.triggered.connect(self.unload_model)
        menu.exec(self.change_model_button.mapToGlobal(self.change_model_button.rect().bottomLeft()))

    def choose_cloud_model(self) -> None:
        if self._generation_running():
            self.show_status_message("Stop generating before changing models.")
            return
        dialog = CloudModelDialog(self.services, self)
        dialog.request_settings.connect(lambda: self._open_cloud_settings(dialog))
        if dialog.exec() == QDialog.Accepted and dialog.selected_target is not None:
            self.load_target(dialog.selected_target)

    def load_target(self, target: ModelTarget, *, silent: bool = False) -> None:
        options = LoadOptions()
        if target.kind == "local":
            entry = self.services.library.find_by_path(target.path)
            if entry is not None:
                options = entry.resolved_options()
        self.show_status_message(f"Loading {target.display_name}…")

        def done(result: LoadResult) -> None:
            if result.ok:
                self.show_status_message(result.warning or f"Loaded {target.display_name}")
                self.switch_page("chat")
            elif not silent:
                show_error(self, "Could not load model", result.message)

        def failed(message: str, detail: str) -> None:
            if not silent:
                show_error(self, "Could not load model", message, detail)

        self._run_task(lambda: self.services.inference.load(target, options), done, failed)

    def unload_model(self) -> None:
        if self._generation_running():
            self.show_status_message("Stop generating before unloading the model.")
            return
        try:
            if self._unload_worker is not None and self._unload_worker.isRunning():
                return
        except RuntimeError:
            self._unload_worker = None
        worker = UnloadWorker(self.services.inference, self)
        worker.finished_unload.connect(lambda: self.show_status_message("Model unloaded"))
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._clear_unload_worker)
        self._unload_worker = worker
        worker.start()

    def _clear_unload_worker(self) -> None:
        self._unload_worker = None

    def _open_cloud_settings(self, dialog: QDialog | None = None) -> None:
        if dialog is not None:
            dialog.reject()
        self.switch_page("settings")

        def scroll() -> None:
            self.pages["settings"].open_section("cloud")

        QTimer.singleShot(0, scroll)

    def _open_server_settings(self) -> None:
        self.switch_page("settings")
        QTimer.singleShot(0, lambda: self.pages["settings"].open_section("server"))

    def _generation_running(self) -> bool:
        try:
            return bool(getattr(self.pages["chat"], "_generating", False))
        except Exception:
            return False

    def _generation_active(self, active: bool) -> None:
        self.change_model_button.setEnabled(not active)
        if active:
            self.change_model_button.setToolTip("Stop generating to change models.")
        else:
            self.change_model_button.setToolTip("")

    # -- inference status --------------------------------------------------
    def _on_inference_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "status":
            self._apply_status(event.get("status") or {})
        elif kind == "warning":
            message = str(event.get("message") or "")
            if message:
                self.show_status_message(message, timeout=9000)

    def _refresh_status(self) -> None:
        self._apply_status(self.services.inference.status())

    def _apply_status(self, status: dict[str, Any]) -> None:
        self._status = status
        state = str(status.get("state") or "idle")
        name = status.get("display_name") or status.get("model_id") or ""
        target = status.get("target") or {}
        if state == "ready" and name:
            kind = "Local" if status.get("kind") == "local" else "Cloud"
            provider = status.get("provider") or ""
            kind_label = "ローカル" if kind == "Local" else "クラウド"
            self.model_name.setText(f"{name}  ·  {kind_label}")
            detail = target.get("path") or provider
            self.model_name.setToolTip(f"{kind}: {name}\n{detail}" if detail else f"{kind}: {name}")
        elif state == "loading":
            loading_name = name or "model"
            self.model_name.setText(f"読み込み中: {loading_name}…")
            self.model_name.setToolTip("")
        elif state == "unloading":
            self.model_name.setText("モデルを解放中…")
            self.model_name.setToolTip("")
        else:
            self.model_name.setText("モデルが選択されていません")
            self.model_name.setToolTip("")

        text = _STATE_TEXT.get(state, state.title())
        detail = str(status.get("error") or status.get("warning") or "")
        self.status_chip.set_status("busy" if state == "unloading" else state, text, detail)
        if state == "error" and status.get("error"):
            self.show_status_message(str(status["error"]), timeout=12000)

        for key in ("chat", "models", "server", "settings"):
            page = self.pages.get(key)
            if page is None:
                continue
            try:
                if key == "chat":
                    page.set_inference_status(status)
                else:
                    page.update_inference_status(status)
            except Exception:
                log.exception("failed to update %s page status", key)

        self._update_sidebar_summary()
        localize_widget_tree(self)

    def _update_sidebar_summary(self) -> None:
        try:
            entries = self.services.library.entries()
            downloads = self.services.downloads.active_count()
        except Exception:
            return
        parts = [f"{len(entries)} model{'s' if len(entries) != 1 else ''}"]
        if downloads:
            parts.append(f"{downloads} download{'s' if downloads != 1 else ''}")
        self.library_summary.setText(" · ".join(parts))

    # -- settings / library events ----------------------------------------
    def _settings_saved(self) -> None:
        app = QApplication.instance()
        if app is not None:
            set_window_theme(
                app,
                str(self.settings.get("general.theme", "dark")),
                int(self.settings.get("general.font_size", 10)),
            )
            self._refresh_icons()
        self._apply_status(self._status)
        self._update_sidebar_summary()
        models = self.pages["models"]
        models.refresh()
        self.pages["discover"].refresh_destination()
        self.pages["server"].refresh()

        model_dir = str(self.settings.get("general.model_directory", ""))
        if model_dir != self._model_dir:
            self._model_dir = model_dir
            self.scan_library()

    def _refresh_icons(self) -> None:
        colors = current_palette()
        for key, button in self.nav_buttons.items():
            button.setIcon(
                make_icon(self._nav_icons[key], colors["text_muted"], colors["accent_hover"])
            )
        self.status_chip._apply()

    def _library_changed(self) -> None:
        try:
            self.pages["models"].refresh()
        except Exception:
            log.exception("could not refresh the model library")
        self._update_sidebar_summary()

    def _history_cleared(self) -> None:
        chat = self.pages["chat"]
        chat.new_chat()
        chat.refresh_conversations()

    # -- startup / background work ----------------------------------------
    def run_startup_tasks(self) -> None:
        warnings = list(getattr(self.settings, "load_warnings", []) or [])
        if warnings:
            self.show_status_message(
                "Settings were repaired: " + "; ".join(warnings[:2]), timeout=12000
            )
        self.scan_library()

        behavior = str(self.settings.get("general.start_behavior", "chat"))
        if behavior == "resume_last_chat":
            self._resume_last_chat()
        elif behavior == "load_last_model":
            QTimer.singleShot(400, self._load_last_model)

        if bool(self.settings.get("server.auto_start", False)):
            QTimer.singleShot(700, self._auto_start_server)

    def _resume_last_chat(self) -> None:
        chat_id = self.settings.get("last.chat_id", None)
        if not chat_id:
            return
        chat = self.pages["chat"]
        if chat.db.get_chat(str(chat_id)) is None:
            return
        chat.load_chat(str(chat_id))

    def _load_last_model(self) -> None:
        raw = self.settings.get("cloud.last_target", None)
        if not isinstance(raw, dict):
            return
        target = ModelTarget.from_dict(raw)
        if target is None:
            return
        if target.kind == "local" and not Path(target.path).is_file():
            self.show_status_message("The last local model is no longer on disk.")
            return
        if target.kind == "cloud" and not self.services.cloud_configured(target.provider):
            self.show_status_message(
                "The last cloud provider has no API key; choose a model again."
            )
            return
        self.load_target(target, silent=True)

    def _auto_start_server(self) -> None:
        if self.services.api_server.running:
            return
        api = self.services.api_server

        def done(info: dict[str, Any]) -> None:
            self.pages["server"].refresh()
            self.show_status_message(f"API server listening at {info.get('base_url', '')}")

        def failed(message: str, _detail: str) -> None:
            self.pages["server"].refresh()
            self.show_status_message(f"API server did not start: {message}", timeout=12000)

        self._run_task(api.start, done, failed)

    def scan_library(self) -> None:
        if self._scan_worker is not None and self._scan_worker.isRunning():
            return
        settings_dir = Path(str(self.settings.get("general.model_directory", "")))
        self._model_dir = str(settings_dir)
        self.services.library.model_directory = settings_dir
        self.show_status_message("Scanning the model folder…", timeout=2500)

        def done(summary: Any) -> None:
            self._scan_worker = None
            changed = getattr(summary, "added", 0) + getattr(summary, "updated", 0)
            missing = getattr(summary, "missing", 0)
            invalid = getattr(summary, "invalid", 0)
            if changed:
                self.show_status_message(f"Model library updated ({changed} change(s))")
            elif missing or invalid:
                self.show_status_message(
                    f"Model scan finished: {missing} missing, {invalid} invalid"
                )
            else:
                self._clear_transient_status()
            self._library_changed()
            self.pages["discover"].refresh_destination()

        def failed(message: str, _detail: str) -> None:
            self._scan_worker = None
            self.show_status_message(f"Model scan failed: {message}", timeout=9000)

        self._scan_worker = self._run_task(self.services.library.refresh, done, failed)

    # -- worker helper / status bar ---------------------------------------
    def _run_task(
        self,
        fn: Callable[[], Any],
        done: Callable[[Any], None] | None = None,
        failed: Callable[[str, str], None] | None = None,
    ) -> TaskWorker:
        worker = TaskWorker(fn)
        if done is not None:
            worker.succeeded.connect(done)
        if failed is not None:
            worker.failed.connect(failed)
        self._workers.append(worker)

        def cleanup() -> None:
            try:
                self._workers.remove(worker)
            except ValueError:
                pass
            worker.deleteLater()

        worker.finished.connect(cleanup)
        worker.start()
        return worker

    def show_status_message(self, message: str, timeout: int = 5000) -> None:
        self.statusBar().showMessage(message, timeout)

    def _clear_transient_status(self) -> None:
        self.statusBar().clearMessage()

    # -- geometry / shutdown ----------------------------------------------
    def _restore_geometry(self) -> None:
        raw = self.settings.get("window.geometry", "") or ""
        restored = False
        if isinstance(raw, str) and raw:
            try:
                restored = bool(self.restoreGeometry(QByteArray.fromBase64(raw.encode("ascii"))))
            except Exception:
                restored = False
        if not restored:
            screen = QApplication.primaryScreen()
            if screen is not None:
                available = screen.availableGeometry()
                width = min(1280, max(1040, available.width() - 120))
                height = min(820, max(660, available.height() - 100))
                self.resize(width, height)
                self.move(
                    available.x() + (available.width() - width) // 2,
                    available.y() + (available.height() - height) // 2,
                )

    def _save_geometry(self) -> None:
        try:
            data = bytes(self.saveGeometry().toBase64()).decode("ascii")
            self.settings.set("window.geometry", data, save=False)
            self.settings.save()
        except Exception:
            log.exception("could not save window geometry")

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        self._save_geometry()
        try:
            self.services.bridge.event.disconnect(self._on_inference_event)
        except (TypeError, RuntimeError):
            pass
        try:
            self.services.bridge.detach(self.services.inference)
        except Exception:
            log.exception("could not detach inference bridge")
        self._status_timer.stop()
        event.accept()

    def shutdown(self) -> None:
        """Stop Qt-side workers. Engine/API shutdown happens in main.py."""
        self._save_geometry()
        for key in ("chat", "discover", "models", "settings", "server"):
            page = self.pages.get(key)
            if page is None:
                continue
            try:
                page.shutdown()
            except Exception:
                log.exception("error shutting down %s page", key)
        for worker in list(self._workers):
            if worker.isRunning():
                worker.wait(6000)
        if self._unload_worker is not None and self._unload_worker.isRunning():
            self._unload_worker.wait(6000)
