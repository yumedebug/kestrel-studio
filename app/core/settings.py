"""Persistent JSON settings with validation.

The store is Qt-free and atomically writes to disk after every mutation. Invalid
values detected on load are clamped to safe defaults and reported through
``load_warnings`` so the UI can surface them instead of crashing.
"""

from __future__ import annotations

import copy
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable

from app.core import paths
from app.constants import (
    DEFAULT_SERVER_HOST,
    DEFAULT_SERVER_PORT,
)

DEFAULT_SYSTEM_PROMPT = "You are a helpful, concise assistant. Answer in the user's language."

DEFAULTS: dict[str, Any] = {
    "version": 1,
    "general": {
        "theme": "light",
        "model_directory": "",
        "start_behavior": "chat",  # chat | resume_last_chat | load_last_model
        "font_size": 10,
    },
    "inference": {
        "default_context_length": 4096,
        "cpu_threads": 0,  # 0 = automatic
        "gpu_layers": 0,  # 0 = CPU, 999 = all layers
        "batch_size": 512,
        "backend": "auto",  # auto | cpu | cuda | vulkan
        "auto_cpu_fallback": True,
        "llama_server_path": "",
        "load_timeout_seconds": 180,
    },
    "server": {
        "host": DEFAULT_SERVER_HOST,
        "port": DEFAULT_SERVER_PORT,
        "auto_start": False,
        "allow_external": False,
    },
    "chat": {
        "system_prompt": DEFAULT_SYSTEM_PROMPT,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repeat_penalty": 1.1,
        "max_tokens": 1024,
        "seed": -1,
    },
    "cloud": {
        "providers": {},  # id -> {base_url, default_model}
        "last_target": None,  # {"kind":"cloud", "provider":..., "model":...}
    },
    "last": {
        "chat_id": None,
        "local_model_path": None,
    },
}

# (path, min, max) integer constraints. 0 threads means auto.
_INT_RANGES = {
    "general.font_size": (8, 24),
    "inference.default_context_length": (128, 1_048_576),
    "inference.cpu_threads": (0, 256),
    "inference.gpu_layers": (0, 999),
    "inference.batch_size": (32, 8192),
    "inference.load_timeout_seconds": (15, 1800),
    "server.port": (1024, 65535),
    "chat.max_tokens": (1, 131072),
    "chat.top_k": (0, 1000),
    "chat.seed": (-1, 2_147_483_647),
}
_FLOAT_RANGES = {
    "chat.temperature": (0.0, 2.0),
    "chat.top_p": (0.0, 1.0),
    "chat.repeat_penalty": (0.0, 4.0),
}
_ENUMS = {
    "general.theme": {"dark", "light"},
    "general.start_behavior": {"chat", "resume_last_chat", "load_last_model"},
    "inference.backend": {"auto", "cpu", "cuda", "vulkan"},
}
_BOOLS = {
    "inference.auto_cpu_fallback",
    "server.auto_start",
    "server.allow_external",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _split(path: str) -> list[str]:
    return [part for part in path.split(".") if part]


def _traverse(data: dict[str, Any], path: str, create: bool = False) -> tuple[dict[str, Any] | None, str]:
    parts = _split(path)
    if not parts:
        raise KeyError(path)
    node: Any = data
    for part in parts[:-1]:
        if not isinstance(node, dict):
            return None, parts[-1]
        if part not in node or not isinstance(node[part], dict):
            if not create:
                return None, parts[-1]
            node[part] = {}
        node = node[part]
    if not isinstance(node, dict):
        return None, parts[-1]
    return node, parts[-1]


def normalize(data: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    """Merge with defaults and coerce/validate values. Returns (data, warnings)."""
    warnings: list[str] = []
    merged = _deep_merge(DEFAULTS, data or {})

    def get_value(path: str, default: Any) -> Any:
        node, key = _traverse(merged, path)
        if node is None or key not in node:
            return default
        return node[key]

    def default_value(path: str) -> Any:
        node, key = _traverse(DEFAULTS, path)
        if node is None or key not in node:
            return None
        return node[key]

    def set_value(path: str, value: Any) -> None:
        node, key = _traverse(merged, path, create=True)
        if node is not None:
            node[key] = value

    for path, (lo, hi) in _INT_RANGES.items():
        default = default_value(path)
        raw = get_value(path, default)
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = int(raw)
        except (TypeError, ValueError):
            warnings.append(f"{path}: invalid value {raw!r}; reset to {default}")
            value = int(default)
        if value < lo or value > hi:
            warnings.append(f"{path}: {value} outside {lo}..{hi}; clamped")
            value = max(lo, min(hi, value))
        set_value(path, value)

    for path, (lo, hi) in _FLOAT_RANGES.items():
        default = default_value(path)
        raw = get_value(path, default)
        try:
            if isinstance(raw, bool):
                raise ValueError
            value = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"{path}: invalid value {raw!r}; reset to {default}")
            value = float(default)
        if value < lo or value > hi:
            warnings.append(f"{path}: {value} outside {lo}..{hi}; clamped")
            value = max(lo, min(hi, value))
        set_value(path, value)

    for path, allowed in _ENUMS.items():
        raw = get_value(path, default_value(path))
        if raw not in allowed:
            warnings.append(f"{path}: unsupported value {raw!r}; using {default_value(path)}")
            set_value(path, default_value(path))

    for path in _BOOLS:
        raw = get_value(path, default_value(path))
        if not isinstance(raw, bool):
            warnings.append(f"{path}: invalid boolean; using {bool(default_value(path))}")
            set_value(path, bool(default_value(path)))

    model_dir = get_value("general.model_directory", "")
    if not isinstance(model_dir, str) or not model_dir.strip():
        model_dir = str(paths.default_model_dir())
    set_value("general.model_directory", str(model_dir))

    host = get_value("server.host", DEFAULT_SERVER_HOST)
    if not isinstance(host, str) or not host.strip():
        set_value("server.host", DEFAULT_SERVER_HOST)

    system_prompt = get_value("chat.system_prompt", DEFAULT_SYSTEM_PROMPT)
    if not isinstance(system_prompt, str):
        set_value("chat.system_prompt", DEFAULT_SYSTEM_PROMPT)

    llama_path = get_value("inference.llama_server_path", "")
    if not isinstance(llama_path, str):
        set_value("inference.llama_server_path", "")

    providers = get_value("cloud.providers", {})
    if not isinstance(providers, dict):
        warnings.append("cloud.providers: invalid object; reset")
        set_value("cloud.providers", {})
    else:
        clean: dict[str, Any] = {}
        for provider_id, cfg in providers.items():
            if not isinstance(cfg, dict):
                continue
            clean[str(provider_id)] = {
                "base_url": str(cfg.get("base_url") or ""),
                "default_model": str(cfg.get("default_model") or ""),
            }
        set_value("cloud.providers", clean)

    last = get_value("cloud.last_target", None)
    if last is not None and not (
        isinstance(last, dict)
        and last.get("kind") in {"local", "cloud"}
    ):
        set_value("cloud.last_target", None)

    return merged, warnings


class SettingsStore:
    """Thread-safe JSON settings store."""

    def __init__(self, path: os.PathLike[str] | str | None = None):
        self.path = Path(path) if path else paths.settings_file()
        self._lock = threading.RLock()
        self._data: dict[str, Any] = copy.deepcopy(DEFAULTS)
        self.load_warnings: list[str] = []
        self.load()

    def load(self) -> None:
        raw: dict[str, Any] = {}
        warnings: list[str] = []
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text(encoding="utf-8-sig"))
                if not isinstance(raw, dict):
                    raise ValueError("settings root must be a JSON object")
            except (OSError, ValueError, UnicodeError) as exc:
                backup = self.path.with_suffix(".broken.json")
                try:
                    self.path.replace(backup)
                except OSError:
                    pass
                warnings.append(f"settings.json could not be read ({exc}); defaults restored")
                raw = {}
        normalized, more = normalize(raw)
        warnings.extend(more)
        with self._lock:
            self._data = normalized
            self.load_warnings = warnings
        if not self.path.exists() or warnings:
            self.save()

    def save(self) -> None:
        with self._lock:
            payload = json.dumps(self._data, ensure_ascii=False, indent=2)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, self.path)

    @property
    def data(self) -> dict[str, Any]:
        with self._lock:
            return self._data

    def get(self, dotted: str, default: Any = None) -> Any:
        with self._lock:
            node, key = _traverse(self._data, dotted)
            if node is None or key not in node:
                if default is not None:
                    return default
                fallback, _ = _traverse(DEFAULTS, dotted)
                if fallback is not None and key in fallback:
                    return fallback[key]
                return default
            return node[key]

    def set(self, dotted: str, value: Any, save: bool = True) -> None:
        # Validate by merging a patch through normalize().
        with self._lock:
            patch: dict[str, Any] = {}
            node = patch
            parts = _split(dotted)
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
            candidate = _deep_merge(self._data, patch)
        normalized, warnings = normalize(candidate)
        if warnings:
            # Re-read the effective value; invalid input is safely clamped.
            pass
        with self._lock:
            self._data = normalized
            self.load_warnings = warnings
        if save:
            self.save()

    def update(self, patch: dict[str, Any], save: bool = True) -> None:
        with self._lock:
            candidate = _deep_merge(self._data, patch)
        normalized, warnings = normalize(candidate)
        with self._lock:
            self._data = normalized
            self.load_warnings = warnings
        if save:
            self.save()

    def reset(self, save: bool = True) -> None:
        normalized, _ = normalize({})
        normalized["general"]["model_directory"] = str(paths.default_model_dir())
        with self._lock:
            self._data = normalized
        if save:
            self.save()

    def require_keys(self, keys: Iterable[str]) -> list[str]:
        missing = []
        for key in keys:
            node, leaf = _traverse(self._data, key)
            if node is None or leaf not in node:
                missing.append(key)
        return missing
