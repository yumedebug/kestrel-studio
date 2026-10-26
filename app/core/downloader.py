"""Background GGUF downloads with progress, cancellation and resume.

huggingface_hub is used for discovery and (via the Hub's resolve URLs) for
retrieval.  The transfer itself is an HTTP Range request handled by httpx so we
can report byte progress, cancel immediately, and resume from ``*.part`` files
after a disconnect or an app restart.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import httpx

from app.core.credentials import CredentialStore
from app.core.hf_client import HF_TOKEN_KEY, human_size
from app.core.paths import data_dir

log = logging.getLogger(__name__)

ACTIVE_STATUSES = {"queued", "downloading", "verifying", "cancelling"}
RESUMABLE_STATUSES = {"paused", "cancelled", "error"}


@dataclass
class DownloadTask:
    id: str
    repo_id: str
    filename: str
    url: str
    dest_path: str
    expected_sha256: str = ""
    total_bytes: int = 0
    downloaded_bytes: int = 0
    status: str = "queued"
    error: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    speed_bps: float = 0.0
    verified: bool = False

    @property
    def part_path(self) -> Path:
        return Path(self.dest_path + ".part")

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return max(0.0, min(100.0, self.downloaded_bytes * 100.0 / self.total_bytes))

    @property
    def size_label(self) -> str:
        if self.total_bytes:
            return f"{human_size(self.downloaded_bytes)} / {human_size(self.total_bytes)}"
        return human_size(self.downloaded_bytes)

    @property
    def eta_label(self) -> str:
        if self.speed_bps <= 0 or self.total_bytes <= 0:
            return ""
        remaining = max(0, self.total_bytes - self.downloaded_bytes)
        seconds = int(remaining / self.speed_bps)
        if seconds <= 0:
            return ""
        if seconds < 60:
            return f"{seconds}s left"
        if seconds < 3600:
            return f"{seconds // 60}m {seconds % 60}s left"
        return f"{seconds // 3600}h {(seconds % 3600) // 60}m left"

    def to_public_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update(
            {
                "percent": round(self.percent, 1),
                "size_label": self.size_label,
                "eta_label": self.eta_label,
                "can_cancel": self.status in {"queued", "downloading"},
                "can_resume": self.status in RESUMABLE_STATUSES and self.part_path.exists(),
                "can_remove": self.status not in ACTIVE_STATUSES,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DownloadTask | None":
        try:
            task = cls(
                id=str(data["id"]),
                repo_id=str(data["repo_id"]),
                filename=str(data["filename"]),
                url=str(data["url"]),
                dest_path=str(data["dest_path"]),
                expected_sha256=str(data.get("expected_sha256") or ""),
                total_bytes=int(data.get("total_bytes") or 0),
                downloaded_bytes=int(data.get("downloaded_bytes") or 0),
                status=str(data.get("status") or "paused"),
                error=str(data.get("error") or ""),
                created_at=str(data.get("created_at") or ""),
                updated_at=str(data.get("updated_at") or ""),
                verified=bool(data.get("verified")),
            )
        except (KeyError, TypeError, ValueError):
            return None
        if task.status in ACTIVE_STATUSES:
            task.status = "paused"
        return task


def safe_component(value: str) -> str:
    value = value.replace("\\", "/").replace("/", "__")
    value = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", value)
    value = value.replace("..", "_")
    return value.strip(" .")[:160] or "model"


class DownloadManager:
    def __init__(
        self,
        credentials: CredentialStore,
        *,
        state_path: Path | None = None,
        on_complete: Callable[[Path], None] | None = None,
        transport: httpx.BaseTransport | None = None,
    ):
        self.credentials = credentials
        self._state_path = state_path or (data_dir() / "downloads.json")
        self._on_complete = on_complete
        self._transport = transport
        self._lock = threading.RLock()
        self._tasks: dict[str, DownloadTask] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._cancel_events: dict[str, threading.Event] = {}
        self._last_save = 0.0
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if not self._state_path.exists():
            return
        try:
            raw = json.loads(self._state_path.read_text(encoding="utf-8"))
            entries = raw.get("tasks", []) if isinstance(raw, dict) else []
            for entry in entries:
                task = DownloadTask.from_dict(entry)
                if task is None:
                    continue
                if task.status == "completed" and not Path(task.dest_path).exists():
                    task.status = "error"
                    task.error = "Downloaded file is no longer on disk."
                self._tasks[task.id] = task
        except (OSError, ValueError) as exc:
            log.warning("could not load download history: %s", exc)

    def _save(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_save < 1.0:
            return
        self._last_save = now
        with self._lock:
            tasks = [asdict(task) for task in self._tasks.values()]
        payload = {"version": 1, "tasks": tasks}
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_path)
        except OSError as exc:
            log.warning("could not save download state: %s", exc)

    # -- API ---------------------------------------------------------------
    def start(
        self,
        repo_id: str,
        filename: str,
        url: str,
        dest_dir: Path | str,
        *,
        expected_sha256: str = "",
        total_bytes: int = 0,
    ) -> DownloadTask:
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        repo_folder = safe_component(repo_id)
        base_name = Path(unquote(urlparse(url).path or filename)).name
        base_name = safe_component(base_name)
        if not base_name.lower().endswith(".gguf"):
            base_name += ".gguf"
        dest = dest_dir / repo_folder / base_name

        with self._lock:
            for existing in self._tasks.values():
                if Path(existing.dest_path) != dest:
                    continue
                if existing.status in ACTIVE_STATUSES:
                    return existing
                if existing.status in RESUMABLE_STATUSES and existing.part_path.exists():
                    existing.status = "queued"
                    existing.error = ""
                    self._save(force=True)
                    self._spawn(existing.id)
                    return existing
            if dest.exists():
                task = DownloadTask(
                    id=f"dl_{uuid.uuid4().hex[:12]}",
                    repo_id=repo_id,
                    filename=filename,
                    url=url,
                    dest_path=str(dest),
                    total_bytes=total_bytes,
                    downloaded_bytes=dest.stat().st_size,
                    status="error",
                    error=f"A file already exists at {dest}. Remove it or choose another folder.",
                )
                self._tasks[task.id] = task
                self._save(force=True)
                return task

            task = DownloadTask(
                id=f"dl_{uuid.uuid4().hex[:12]}",
                repo_id=repo_id,
                filename=filename,
                url=url,
                dest_path=str(dest),
                expected_sha256=expected_sha256 or "",
                total_bytes=max(0, total_bytes),
            )
            partial = task.part_path
            if partial.exists():
                task.downloaded_bytes = partial.stat().st_size
                task.status = "paused"
            self._tasks[task.id] = task
        self._save(force=True)
        if task.status != "paused":
            self._spawn(task.id)
        return task

    def _spawn(self, task_id: str) -> None:
        with self._lock:
            thread = self._threads.get(task_id)
            if thread is not None and thread.is_alive():
                return
            event = threading.Event()
            self._cancel_events[task_id] = event
            thread = threading.Thread(
                target=self._run,
                args=(task_id, event),
                name=f"download-{task_id}",
                daemon=True,
            )
            self._threads[task_id] = thread
        thread.start()

    def resume(self, task_id: str) -> bool:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status in ACTIVE_STATUSES:
                return False
            if not task.part_path.exists():
                task.status = "error"
                task.error = "The partial download file is missing; start the download again."
                self._save(force=True)
                return False
            task.status = "queued"
            task.error = ""
            task.speed_bps = 0.0
        self._save(force=True)
        self._spawn(task_id)
        return True

    def cancel(self, task_id: str) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if task.status not in ACTIVE_STATUSES:
                return
            event = self._cancel_events.get(task_id)
            if event is not None:
                event.set()
            task.status = "cancelling"
        self._save(force=True)

    def remove(self, task_id: str, *, delete_partial: bool = True) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            if task.status in ACTIVE_STATUSES:
                event = self._cancel_events.get(task_id)
                if event:
                    event.set()
                return  # The worker removes it after noticing cancellation.
            self._tasks.pop(task_id, None)
            self._threads.pop(task_id, None)
            self._cancel_events.pop(task_id, None)
        if delete_partial:
            try:
                task.part_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._save(force=True)

    def snapshot(self) -> list[DownloadTask]:
        with self._lock:
            tasks = list(self._tasks.values())
        # Return copies so the UI thread never sees torn state.
        copies = []
        for task in tasks:
            copies.append(replace(task))
        copies.sort(key=lambda t: t.created_at, reverse=True)
        return copies

    def active_count(self) -> int:
        with self._lock:
            return sum(1 for t in self._tasks.values() if t.status in ACTIVE_STATUSES)

    def clear_partial_downloads(self) -> tuple[int, int]:
        """Delete *.part files for non-active tasks. Returns (files, bytes)."""
        files = 0
        freed = 0
        with self._lock:
            tasks = list(self._tasks.values())
        for task in tasks:
            if task.status in ACTIVE_STATUSES:
                continue
            if task.part_path.exists():
                try:
                    freed += task.part_path.stat().st_size
                    task.part_path.unlink()
                    files += 1
                except OSError:
                    pass
            with self._lock:
                live = self._tasks.get(task.id)
                if live is not None and live.status in RESUMABLE_STATUSES:
                    live.downloaded_bytes = 0
                    live.error = "Partial download removed."
        self._save(force=True)
        return files, freed

    def shutdown(self) -> None:
        with self._lock:
            for event in self._cancel_events.values():
                event.set()
            threads = list(self._threads.values())
        for thread in threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        self._save(force=True)

    # -- worker ------------------------------------------------------------
    def _set(self, task_id: str, **fields: Any) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            for key, value in fields.items():
                setattr(task, key, value)
            task.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    def _run(self, task_id: str, cancel: threading.Event) -> None:
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            url = task.url
            dest = Path(task.dest_path)
            partial = task.part_path
            expected_sha = task.expected_sha256
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            existing = partial.stat().st_size if partial.exists() else 0
            headers = {"User-Agent": "KestrelStudio/0.1 (+local desktop app)"}
            token = None
            try:
                token = self.credentials.get(HF_TOKEN_KEY)
            except Exception:
                token = None
            if token:
                headers["Authorization"] = f"Bearer {token}"
            if existing:
                headers["Range"] = f"bytes={existing}-"

            self._set(task_id, status="downloading", error="", speed_bps=0.0)
            self._save(force=True)
            with httpx.Client(
                timeout=httpx.Timeout(connect=20.0, read=90.0, write=90.0, pool=10.0),
                follow_redirects=True,
                transport=self._transport,
            ) as client:
                with client.stream("GET", url, headers=headers) as response:
                    if response.status_code in (401, 403):
                        raise _DownloadError(
                            "Hugging Face refused the download. This model may be gated: "
                            "accept its license on the model page and add a Hugging Face token "
                            "in Discover.",
                            kind="auth",
                        )
                    if response.status_code == 404:
                        raise _DownloadError(
                            "The file no longer exists on Hugging Face (HTTP 404).", kind="not_found"
                        )
                    if response.status_code >= 400:
                        raise _DownloadError(
                            f"Download failed with HTTP {response.status_code}.", kind="http"
                        )
                    total = 0
                    if response.status_code == 206:
                        content_range = response.headers.get("Content-Range", "")
                        match = re.match(r"bytes\s+\d+-\d+/(\d+)", content_range)
                        if match:
                            total = int(match.group(1))
                        mode = "ab"
                    else:
                        # Server ignored Range (or this is a fresh download).
                        existing = 0
                        mode = "wb"
                        length = response.headers.get("Content-Length")
                        if length and length.isdigit():
                            total = int(length)
                    if not total:
                        with self._lock:
                            current = self._tasks.get(task_id)
                            total = current.total_bytes if current else 0
                    self._set(task_id, total_bytes=total, downloaded_bytes=existing)
                    self._save(force=True)

                    written = 0
                    last_report = time.monotonic()
                    last_bytes = existing
                    with partial.open(mode) as fh:
                        for chunk in response.iter_bytes(1024 * 256):
                            if cancel.is_set():
                                fh.flush()
                                self._set(
                                    task_id,
                                    status="cancelled",
                                    downloaded_bytes=existing + written,
                                    speed_bps=0.0,
                                )
                                self._save(force=True)
                                with self._lock:
                                    self._cancel_events.pop(task_id, None)
                                    self._threads.pop(task_id, None)
                                return
                            if not chunk:
                                continue
                            fh.write(chunk)
                            written += len(chunk)
                            now = time.monotonic()
                            if now - last_report >= 0.35:
                                fh.flush()
                                speed = (existing + written - last_bytes) / max(
                                    now - last_report, 0.001
                                )
                                self._set(
                                    task_id,
                                    downloaded_bytes=existing + written,
                                    speed_bps=speed,
                                )
                                self._save()
                                last_report = now
                                last_bytes = existing + written
                        fh.flush()
                        os.fsync(fh.fileno())

            final_size = partial.stat().st_size
            self._set(
                task_id,
                downloaded_bytes=final_size,
                status="verifying",
                speed_bps=0.0,
            )
            self._save(force=True)
            self._verify_and_finalize(task_id, partial, dest, expected_sha)
        except _DownloadError as exc:
            self._fail(task_id, str(exc))
        except httpx.HTTPError as exc:
            self._fail(
                task_id,
                f"Network error while downloading: {exc}. The partial file was kept; "
                "press Resume when the connection is back.",
            )
        except OSError as exc:
            self._fail(task_id, f"Could not write the file: {exc}")
        except Exception as exc:
            log.exception("download failed")
            self._fail(task_id, f"Unexpected download error: {exc}")

    def _verify_and_finalize(
        self, task_id: str, partial: Path, dest: Path, expected_sha: str
    ) -> None:
        try:
            with partial.open("rb") as fh:
                magic = fh.read(4)
            if magic != b"GGUF":
                try:
                    partial.unlink()
                except OSError:
                    pass
                self._fail(
                    task_id,
                    "The downloaded file is not a GGUF model (bad header). "
                    "The file was deleted; check that you picked a .gguf file.",
                )
                return
            if expected_sha:
                digest = hashlib.sha256()
                with partial.open("rb") as fh:
                    while True:
                        block = fh.read(4 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                actual = digest.hexdigest()
                if actual.lower() != expected_sha.lower():
                    try:
                        partial.unlink()
                    except OSError:
                        pass
                    self._fail(
                        task_id,
                        "Checksum verification failed. The file was deleted; please retry "
                        "the download.",
                    )
                    return
                self._set(task_id, verified=True)
            if dest.exists():
                self._fail(
                    task_id,
                    f"A file already exists at {dest}. Remove it and retry.",
                )
                return
            os.replace(partial, dest)
            self._set(
                task_id,
                status="completed",
                downloaded_bytes=dest.stat().st_size,
                speed_bps=0.0,
            )
            self._save(force=True)
            if self._on_complete is not None:
                try:
                    self._on_complete(dest)
                except Exception:
                    log.exception("on_complete callback failed")
        except OSError as exc:
            self._fail(task_id, f"Could not finalise the download: {exc}")

    def _fail(self, task_id: str, message: str) -> None:
        self._set(task_id, status="error", error=message, speed_bps=0.0)
        self._save(force=True)
        with self._lock:
            self._cancel_events.pop(task_id, None)
            self._threads.pop(task_id, None)


class _DownloadError(RuntimeError):
    def __init__(self, message: str, kind: str = "download"):
        super().__init__(message)
        self.kind = kind
