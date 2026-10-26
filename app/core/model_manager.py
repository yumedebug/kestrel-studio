"""GGUF model library: registration, metadata cache, per-model load options."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from app.core.gguf import GGUFInfo, read_gguf
from app.core.paths import library_file
from app.core.types import LoadOptions, ModelTarget

log = logging.getLogger(__name__)


class ModelManagerError(RuntimeError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _entry_id(path: Path) -> str:
    key = os.path.normcase(str(path.resolve()))
    return "model_" + hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


@dataclass
class ModelEntry:
    id: str
    path: str
    name: str
    size: int = 0
    quant: str = ""
    architecture: str = ""
    context_length: int | None = None
    parameter_count: int | None = None
    has_chat_template: bool = False
    added_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    source_repo: str = ""
    source_file: str = ""
    load_options: dict[str, Any] = field(default_factory=dict)
    metadata_error: str = ""
    mtime: float = 0.0

    @property
    def file_path(self) -> Path:
        return Path(self.path)

    @property
    def exists(self) -> bool:
        try:
            return self.file_path.is_file()
        except OSError:
            return False

    @property
    def status(self) -> str:
        if not self.exists:
            return "missing"
        if self.metadata_error:
            return "invalid"
        return "ready"

    @property
    def status_label(self) -> str:
        return {"ready": "Ready", "missing": "Missing", "invalid": "Invalid"}.get(
            self.status, self.status.title()
        )

    @property
    def size_label(self) -> str:
        size = self.size
        if not size:
            return "—"
        value = float(size)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024 or unit == "TB":
                return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
            value /= 1024
        return f"{size} B"

    @property
    def parameters_label(self) -> str:
        count = self.parameter_count
        if not count:
            return "—"
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}B"
        return f"{count / 1_000_000:.0f}M"

    @property
    def context_label(self) -> str:
        return f"{self.context_length:,}" if self.context_length else "—"

    def target(self) -> ModelTarget:
        return ModelTarget(
            kind="local",
            display_name=self.name,
            path=self.path,
            entry_id=self.id,
        )

    def resolved_options(self) -> LoadOptions:
        return LoadOptions(
            context_length=self.load_options.get("context_length"),
            gpu_layers=self.load_options.get("gpu_layers"),
            cpu_threads=self.load_options.get("cpu_threads"),
            batch_size=self.load_options.get("batch_size"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelEntry | None":
        try:
            path = str(data["path"])
            return cls(
                id=str(data.get("id") or _entry_id(Path(path))),
                path=path,
                name=str(data.get("name") or Path(path).stem),
                size=int(data.get("size") or 0),
                quant=str(data.get("quant") or ""),
                architecture=str(data.get("architecture") or ""),
                context_length=data.get("context_length"),
                parameter_count=data.get("parameter_count"),
                has_chat_template=bool(data.get("has_chat_template")),
                added_at=str(data.get("added_at") or _now()),
                updated_at=str(data.get("updated_at") or _now()),
                source_repo=str(data.get("source_repo") or ""),
                source_file=str(data.get("source_file") or ""),
                load_options=dict(data.get("load_options") or {}),
                metadata_error=str(data.get("metadata_error") or ""),
                mtime=float(data.get("mtime") or 0.0),
            )
        except (KeyError, TypeError, ValueError):
            return None


@dataclass
class ScanSummary:
    added: int = 0
    updated: int = 0
    missing: int = 0
    invalid: int = 0


class ModelManager:
    def __init__(self, path: Path | None = None, model_directory: Path | None = None):
        self._path = Path(path) if path else library_file()
        self._lock = threading.RLock()
        self._entries: dict[str, ModelEntry] = {}
        self.model_directory = Path(model_directory) if model_directory else None
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            entries = raw.get("models", []) if isinstance(raw, dict) else []
            for item in entries:
                entry = ModelEntry.from_dict(item)
                if entry is not None:
                    self._entries[entry.id] = entry
        except (OSError, ValueError, UnicodeError) as exc:
            log.warning("library.json could not be read: %s", exc)
            try:
                self._path.replace(self._path.with_suffix(".broken.json"))
            except OSError:
                pass

    def _save(self) -> None:
        with self._lock:
            entries = [entry.to_dict() for entry in self._entries.values()]
        payload = {"version": 1, "models": entries}
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)

    # -- queries -----------------------------------------------------------
    def entries(self) -> list[ModelEntry]:
        with self._lock:
            items = list(self._entries.values())
        items.sort(key=lambda e: (e.status != "ready", e.name.lower()))
        return items

    def get(self, entry_id: str) -> ModelEntry | None:
        with self._lock:
            return self._entries.get(entry_id)

    def find_by_path(self, path: Path | str) -> ModelEntry | None:
        wanted = os.path.normcase(str(Path(path).resolve()))
        with self._lock:
            for entry in self._entries.values():
                try:
                    if os.path.normcase(str(entry.file_path.resolve())) == wanted:
                        return entry
                except OSError:
                    continue
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._entries)

    # -- mutations ---------------------------------------------------------
    def add_file(
        self,
        path: Path | str,
        *,
        source_repo: str = "",
        source_file: str = "",
        allow_invalid: bool = False,
        save: bool = True,
    ) -> ModelEntry:
        path = Path(path).expanduser()
        try:
            path = path.resolve()
        except OSError:
            pass
        if not path.is_file():
            raise ModelManagerError(f"Model file not found: {path}")
        if path.suffix.lower() != ".gguf":
            raise ModelManagerError("Only .gguf files can be added to the library.")

        info = read_gguf(path)
        if not info.ok and not allow_invalid:
            raise ModelManagerError(
                info.error or "The file is not a valid GGUF model."
            )

        with self._lock:
            existing = None
            for entry in self._entries.values():
                try:
                    if os.path.normcase(str(entry.file_path.resolve())) == os.path.normcase(
                        str(path)
                    ):
                        existing = entry
                        break
                except OSError:
                    continue
            entry = existing or ModelEntry(id=_entry_id(path), path=str(path), name=path.stem)
            self._apply_info(entry, info)
            entry.source_repo = source_repo or entry.source_repo
            entry.source_file = source_file or entry.source_file
            entry.updated_at = _now()
            if existing is None:
                self._entries[entry.id] = entry
        if save:
            self._save()
        return entry

    def _apply_info(self, entry: ModelEntry, info: GGUFInfo) -> None:
        try:
            stat = Path(entry.path).stat()
            entry.size = stat.st_size
            entry.mtime = stat.st_mtime
        except OSError:
            entry.size = info.file_size
        entry.quant = info.quant or entry.quant
        entry.architecture = info.architecture or entry.architecture
        entry.context_length = info.context_length or entry.context_length
        entry.parameter_count = info.parameter_count or entry.parameter_count
        entry.has_chat_template = info.has_chat_template
        entry.metadata_error = "" if info.ok else (info.error or "Unknown GGUF error")
        if info.name and (not entry.name or entry.name == Path(entry.path).stem):
            entry.name = info.name

    def refresh(self, model_directory: Path | str | None = None) -> ScanSummary:
        """Scan the model directory, add new GGUF files and refresh metadata."""
        directory = Path(model_directory) if model_directory else self.model_directory
        summary = ScanSummary()
        if directory is None:
            return summary
        self.model_directory = directory
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            log.warning("cannot create model directory %s: %s", directory, exc)
            return summary

        paths: list[Path] = []
        try:
            for root, dirs, files in os.walk(directory):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for name in files:
                    if name.lower().endswith(".gguf") and not name.endswith(".part"):
                        paths.append(Path(root) / name)
        except OSError as exc:
            log.warning("model scan failed: %s", exc)
            return summary

        seen: set[str] = set()
        with self._lock:
            known = {os.path.normcase(entry.path): entry for entry in self._entries.values()}
        for path in paths:
            key = os.path.normcase(str(path.resolve()))
            seen.add(key)
            entry = known.get(key)
            try:
                stat = path.stat()
            except OSError:
                continue
            if entry is not None and entry.mtime == stat.st_mtime and entry.size == stat.st_size:
                continue
            info = read_gguf(path)
            with self._lock:
                current = None
                for candidate in self._entries.values():
                    try:
                        if os.path.normcase(str(candidate.file_path.resolve())) == key:
                            current = candidate
                            break
                    except OSError:
                        continue
                is_new = current is None
                if is_new:
                    current = ModelEntry(id=_entry_id(path), path=str(path), name=path.stem)
                    self._entries[current.id] = current
                self._apply_info(current, info)
                current.updated_at = _now()
                if info.ok:
                    summary.added += 1 if is_new else 0
                    summary.updated += 0 if is_new else 1
                else:
                    summary.invalid += 1

        # Track files in the managed directory that disappeared.
        with self._lock:
            for entry in self._entries.values():
                if not entry.exists:
                    if entry.status == "missing":
                        summary.missing += 1
            self._save()
        return summary

    def save_load_options(
        self,
        entry_id: str,
        options: LoadOptions,
        *,
        base: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                raise ModelManagerError("Model entry not found.")
            merged = dict(base or {})
            merged.update(
                {
                    "context_length": options.context_length,
                    "gpu_layers": options.gpu_layers,
                    "cpu_threads": options.cpu_threads,
                    "batch_size": options.batch_size,
                }
            )
            entry.load_options = merged
            entry.updated_at = _now()
        self._save()

    def remove_from_library(self, entry_id: str) -> None:
        """Unregister only; the GGUF file stays on disk."""
        with self._lock:
            self._entries.pop(entry_id, None)
        self._save()

    def delete_file(self, entry_id: str) -> None:
        """Delete the file and unregister it. Callers must confirm first."""
        with self._lock:
            entry = self._entries.get(entry_id)
        if entry is None:
            raise ModelManagerError("Model entry not found.")
        path = entry.file_path
        try:
            if path.exists():
                path.unlink()
        except OSError as exc:
            raise ModelManagerError(f"Could not delete {path}: {exc}") from exc
        with self._lock:
            self._entries.pop(entry_id, None)
        self._save()

    def update_source(self, entry_id: str, repo_id: str, filename: str) -> None:
        with self._lock:
            entry = self._entries.get(entry_id)
            if entry is None:
                return
            entry.source_repo = repo_id
            entry.source_file = filename
        self._save()

    def remove_missing(self) -> int:
        removed = 0
        with self._lock:
            for entry_id, entry in list(self._entries.items()):
                if not entry.exists:
                    del self._entries[entry_id]
                    removed += 1
        if removed:
            self._save()
        return removed
