"""Application identity, defaults and theme tokens.

Rename the application here (and regenerate assets/icon.ico) to rebrand it.
"""

from __future__ import annotations

from pathlib import Path

APP_NAME = "Kestrel Studio"
APP_SHORT_NAME = "Kestrel"
APP_ID = "KestrelStudio"
APP_VERSION = "0.1.2"
APP_ORG = "Kestrel"
APP_DESCRIPTION = "Local GGUF models and cloud providers in one desktop studio"
APP_WEBSITE = "https://github.com/"

DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 1234

# Theme colours are also used by custom-painted widgets.
DARK = {
    "bg": "#17181b",
    "bg_elevated": "#202126",
    "bg_input": "#292a30",
    "bg_hover": "#33343b",
    "border": "#34353b",
    "border_strong": "#484950",
    "text": "#f5f5f7",
    "text_muted": "#b2b3bb",
    "text_faint": "#888a94",
    "accent": "#0a84ff",
    "accent_hover": "#409cff",
    "accent_soft": "#243a52",
    "success": "#30d158",
    "warning": "#ffd60a",
    "danger": "#ff6961",
    "code_bg": "#141519",
    "user_bg": "#252f3b",
    "selection": "#345b83",
}

LIGHT = {
    "bg": "#f5f5f7",
    "bg_elevated": "#ffffff",
    "bg_input": "#ffffff",
    "bg_hover": "#e9e9ed",
    "border": "#dedee3",
    "border_strong": "#c9c9d0",
    "text": "#1d1d1f",
    "text_muted": "#626269",
    "text_faint": "#86868b",
    "accent": "#007aff",
    "accent_hover": "#006be6",
    "accent_soft": "#e5f1ff",
    "success": "#248a3d",
    "warning": "#9a6700",
    "danger": "#d70015",
    "code_bg": "#f0f0f2",
    "user_bg": "#e7f1fc",
    "selection": "#b9d8fa",
}

THEMES = {"dark": DARK, "light": LIGHT}

MODEL_DIRECTORY_ENV = "KESTREL_DATA_DIR"
ASSETS_DIRNAME = "assets"


def assets_dir() -> Path:
    """Directory containing bundled assets (works from source and PyInstaller)."""
    import sys

    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        candidate = Path(sys._MEIPASS) / ASSETS_DIRNAME
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parent.parent / ASSETS_DIRNAME

