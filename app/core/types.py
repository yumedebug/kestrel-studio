"""Shared, Qt-free value types used by backends, the manager, GUI and API."""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any


class BackendError(RuntimeError):
    """An error that is safe and useful to show to a human.

    ``kind`` is a stable machine-readable category used by the API layer.
    """

    def __init__(
        self,
        message: str,
        *,
        kind: str = "inference_error",
        detail: str = "",
        status_code: int = 500,
    ):
        super().__init__(message)
        self.message = message
        self.kind = kind
        self.detail = detail
        self.status_code = status_code

    def __str__(self) -> str:
        return self.message

    def to_dict(self) -> dict[str, Any]:
        return {"message": self.message, "kind": self.kind, "detail": self.detail}


@dataclass
class ModelTarget:
    kind: str  # "local" | "cloud"
    display_name: str
    model: str = ""
    provider: str = ""
    path: str = ""
    entry_id: str = ""

    @property
    def id(self) -> str:
        if self.kind == "local":
            return f"local:{self.path}"
        return f"cloud:{self.provider}:{self.model}"

    @property
    def api_model_id(self) -> str:
        """ID exposed through the OpenAI-compatible /v1/models endpoint."""
        if self.kind == "local":
            from pathlib import Path

            return f"local/{Path(self.path).stem}"
        return self.model

    @property
    def short_label(self) -> str:
        if self.kind == "local":
            return self.display_name
        return f"{self.model}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelTarget | None":
        kind = data.get("kind")
        if kind == "local" and data.get("path"):
            return cls(
                kind="local",
                display_name=str(data.get("display_name") or "Local model"),
                path=str(data["path"]),
                entry_id=str(data.get("entry_id") or ""),
            )
        if kind == "cloud" and data.get("provider") and data.get("model"):
            return cls(
                kind="cloud",
                display_name=str(data.get("display_name") or data["model"]),
                provider=str(data["provider"]),
                model=str(data["model"]),
            )
        return None


@dataclass
class GenerationParams:
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    repeat_penalty: float = 1.1
    max_tokens: int = 1024
    seed: int = -1
    stop: list[str] = field(default_factory=list)

    @classmethod
    def from_settings(cls, values: dict[str, Any]) -> "GenerationParams":
        return cls(
            temperature=float(values.get("temperature", 0.7)),
            top_p=float(values.get("top_p", 0.95)),
            top_k=int(values.get("top_k", 40)),
            repeat_penalty=float(values.get("repeat_penalty", 1.1)),
            max_tokens=int(values.get("max_tokens", 1024)),
            seed=int(values.get("seed", -1)),
            stop=list(values.get("stop") or []),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LoadOptions:
    context_length: int | None = None
    gpu_layers: int | None = None
    cpu_threads: int | None = None
    batch_size: int | None = None

    def resolved(self, settings: Any) -> "LoadOptions":
        """Fill unset values from the global inference settings."""
        from os import cpu_count

        ctx = self.context_length or settings.get("inference.default_context_length", 4096)
        gpu = (
            self.gpu_layers
            if self.gpu_layers is not None
            else settings.get("inference.gpu_layers", 0)
        )
        threads = (
            self.cpu_threads
            if self.cpu_threads is not None
            else settings.get("inference.cpu_threads", 0)
        )
        if not threads or threads <= 0:
            threads = max(1, (cpu_count() or 4) // 2)
        batch = self.batch_size or settings.get("inference.batch_size", 512)
        return LoadOptions(
            context_length=max(128, int(ctx)),
            gpu_layers=max(0, int(gpu)),
            cpu_threads=max(1, int(threads)),
            batch_size=max(32, int(batch)),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StreamEvent:
    type: str  # delta | usage | done
    text: str = ""
    usage: dict[str, int] | None = None
    finish_reason: str | None = None


@dataclass
class LoadResult:
    ok: bool
    message: str = ""
    warning: str = ""
    target: ModelTarget | None = None

    @property
    def user_message(self) -> str:
        if self.ok:
            return self.warning or self.message
        return self.message


class CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: list[Any] = []

    def add_callback(self, callback: Any) -> None:
        """Register a function (e.g. response.close) to run on cancellation."""
        with self._lock:
            if not self._event.is_set():
                self._callbacks.append(callback)
                return
        try:
            callback()
        except Exception:
            pass

    def remove_callback(self, callback: Any) -> None:
        with self._lock:
            try:
                self._callbacks.remove(callback)
            except ValueError:
                pass

    def cancel(self) -> None:
        with self._lock:
            if self._event.is_set():
                return
            self._event.set()
            callbacks = list(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            try:
                callback()
            except Exception:
                pass

    def is_set(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise GenerationCancelled()

    def wait(self, timeout: float) -> bool:
        return self._event.wait(timeout)


class GenerationCancelled(Exception):
    """Internal control-flow signal for a cancelled generation."""
