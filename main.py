"""Kestrel Studio entry point.

Boots the Qt application, wires the shared services and shuts every child
process down cleanly on exit. Import-safe: all side effects live in main().
"""

from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
import sys
import threading
import traceback
from pathlib import Path

from app.constants import APP_DESCRIPTION, APP_ID, APP_NAME, APP_ORG, APP_VERSION

log = logging.getLogger("kestrel")


def configure_logging() -> Path:
    from app.core import paths

    paths.ensure_dirs()
    log_path = paths.logs_dir() / "kestrel.log"
    handlers: list[logging.Handler] = [
        logging.handlers.RotatingFileHandler(
            log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
        )
    ]
    if sys.stderr is not None:
        handlers.append(logging.StreamHandler(sys.stderr))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.captureWarnings(True)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.error").setLevel(logging.WARNING)
    return log_path


def _fatal_dialog(title: str, message: str, detail: str = "") -> None:
    """Best-effort GUI error report; never raises."""
    if os.environ.get("KESTREL_NO_DIALOGS") or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
        log.error("%s: %s", title, message)
        return
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        if QApplication.instance() is not None:
            if QApplication.platformName() == "offscreen":
                log.error("%s: %s", title, message)
                return
            box = QMessageBox()
            box.setIcon(QMessageBox.Critical)
            box.setWindowTitle(title)
            box.setText(message)
            if detail:
                box.setDetailedText(detail[:8000])
            box.exec()
            return
    except Exception:
        pass
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.user32.MessageBoxW(None, f"{message}\n\n{detail[:2000]}", title, 0x10)
        else:
            print(f"{title}: {message}\n{detail}", file=sys.stderr or sys.stdout)
    except Exception:
        pass


def _install_excepthooks(log_path: Path) -> None:
    def handle(exc_type, exc_value, exc_tb) -> None:
        if issubclass(exc_type, KeyboardInterrupt):
            return
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        log.critical("Unhandled exception:\n%s", text)
        _fatal_dialog(
            f"{APP_NAME} encountered an error",
            "An unexpected error occurred. The app will keep running if possible.",
            f"{text}\n\nLog file: {log_path}",
        )

    sys.excepthook = handle

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_type is SystemExit:
            return
        text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        log.error("Unhandled exception in thread %s:\n%s", args.thread.name if args.thread else "?", text)

    threading.excepthook = thread_hook


def _create_services(settings):
    from app.api.server import ApiServer
    from app.core import paths
    from app.core.credentials import create_credential_store
    from app.core.database import Database
    from app.core.downloader import DownloadManager
    from app.core.hf_client import HuggingFaceClient
    from app.core.inference import InferenceManager
    from app.core.model_manager import ModelManager
    from app.ui.services import AppServices

    credentials = create_credential_store(paths.config_dir() / "secrets.dat")
    database = Database(paths.database_file())
    hf = HuggingFaceClient(credentials, settings)
    library = ModelManager(
        paths.library_file(), Path(str(settings.get("general.model_directory")))
    )

    def register_download(path: Path) -> None:
        try:
            entry = library.add_file(path)
            log.info("downloaded model registered: %s", entry.name)
        except Exception:
            log.exception("could not register downloaded model %s", path)

    downloads = DownloadManager(credentials, on_complete=register_download)
    inference = InferenceManager(settings, credentials)
    api_server = ApiServer(inference, settings)
    return AppServices(
        settings=settings,
        credentials=credentials,
        database=database,
        hf=hf,
        library=library,
        downloads=downloads,
        inference=inference,
        api_server=api_server,
    )


def _shutdown(services, window) -> None:
    if window is not None:
        try:
            window.close()
        except Exception:
            log.exception("error closing the window")
        try:
            window.shutdown()
        except Exception:
            log.exception("error shutting down the window")
    if services is None:
        return
    for name, fn in (
        ("API server", services.api_server.shutdown),
        ("downloads", services.downloads.shutdown),
        ("inference", services.inference.shutdown),
        ("database", services.database.close),
    ):
        try:
            fn()
        except Exception:
            log.exception("error shutting down %s", name)
    log.info("shutdown complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=APP_ID,
        description=APP_DESCRIPTION,
        add_help=True,
    )
    parser.add_argument("--version", action="store_true", help="print the version and exit")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    log_path = configure_logging()
    _install_excepthooks(log_path)
    args, qt_args = build_parser().parse_known_args(argv[1:])
    if args.version:
        print(f"{APP_NAME} {APP_VERSION}")
        return 0

    log.info("starting %s %s (Python %s)", APP_NAME, APP_VERSION, sys.version.split()[0])

    # High-DPI: fractional scaling is passed through on Windows instead of
    # rounding to 100/200 %, which keeps 125 % and 150 % displays sharp.
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    if hasattr(QGuiApplication, "setHighDpiScaleFactorRoundingPolicy"):
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
        )

    from app.core import paths
    from app.core.settings import SettingsStore

    paths.ensure_dirs()
    try:
        settings = SettingsStore()
    except Exception as exc:
        log.exception("could not load settings")
        _fatal_dialog(f"{APP_NAME} could not start", f"Settings could not be loaded: {exc}")
        return 1

    app = QApplication([argv[0], *qt_args])
    app.setApplicationName(APP_NAME)
    app.setApplicationDisplayName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName(APP_ORG)
    app.setDesktopFileName(APP_ID)

    from app.ui.icons import app_icon
    from app.ui.theme import set_window_theme

    app.setWindowIcon(app_icon())
    set_window_theme(
        app,
        str(settings.get("general.theme", "dark")),
        int(settings.get("general.font_size", 10)),
    )

    services = None
    window = None
    exit_code = 0
    try:
        services = _create_services(settings)
        from app.ui.main_window import MainWindow

        window = MainWindow(services)
        window.show()

        if args.smoke_test:
            def smoke() -> None:
                for key in ("discover", "models", "server", "settings", "chat"):
                    window.switch_page(key)
                    app.processEvents()
                log.info("smoke test: all pages created")
                print("KESTREL_SMOKE_OK", flush=True)
                window.close()
                QTimer.singleShot(150, app.quit)

            QTimer.singleShot(400, smoke)
        else:
            QTimer.singleShot(150, window.run_startup_tasks)

        exit_code = app.exec()
    except Exception as exc:
        log.exception("fatal startup error")
        _fatal_dialog(
            f"{APP_NAME} could not start",
            f"Startup failed: {exc}",
            f"{traceback.format_exc()}\n\nLog file: {log_path}",
        )
        exit_code = 1
    finally:
        _shutdown(services, window)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
