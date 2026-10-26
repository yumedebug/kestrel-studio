"""Filesystem locations used by the application.

Settings live in %APPDATA%, while large/volatile data (models, chats, logs)
lives in %LOCALAPPDATA%. Environment overrides make tests hermetic.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from app.constants import APP_ID, MODEL_DIRECTORY_ENV


def app_root() -> Path:
    """Root of the source checkout, or the folder containing the frozen exe."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    override = os.environ.get("KESTREL_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / APP_ID


def data_dir() -> Path:
    override = os.environ.get(MODEL_DIRECTORY_ENV)
    if override:
        return Path(override).expanduser().resolve()
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_ID


def default_model_dir() -> Path:
    return data_dir() / "models"


def logs_dir() -> Path:
    return data_dir() / "logs"


def downloads_dir() -> Path:
    return data_dir() / "downloads"


def ensure_dirs() -> None:
    for path in (config_dir(), data_dir(), default_model_dir(), logs_dir(), downloads_dir()):
        path.mkdir(parents=True, exist_ok=True)


def settings_file() -> Path:
    return config_dir() / "settings.json"


def database_file() -> Path:
    return data_dir() / "chats.db"


def library_file() -> Path:
    return config_dir() / "library.json"


def runtime_dir() -> Path:
    return app_root() / "runtime"


def find_runtime_executable(name: str = "llama-server.exe") -> Path | None:
    """Look for a llama.cpp runtime shipped next to the app or installed locally."""
    roots = [runtime_dir(), data_dir() / "runtime", app_root()]
    preferred = ["cuda", "vulkan", "cpu", "bin", ""]
    for root in roots:
        for sub in preferred:
            candidate = root / sub / name if sub else root / name
            if candidate.is_file():
                return candidate
    # Bare "llama-server" on PATH (Linux/macOS developer machines).
    import shutil

    found = shutil.which("llama-server")
    return Path(found) if found else None
