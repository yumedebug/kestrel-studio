"""QThread workers. All blocking I/O runs here, never in the GUI thread."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from PySide6.QtCore import QThread, Signal

from app.core.inference import InferenceManager
from app.core.types import (
    BackendError,
    CancelToken,
    GenerationParams,
    LoadOptions,
    LoadResult,
    ModelTarget,
)

log = logging.getLogger(__name__)


class TaskWorker(QThread):
    succeeded = Signal(object)
    failed = Signal(str, str)

    def __init__(self, fn: Callable[[], Any], parent=None):
        super().__init__(parent)
        self._fn = fn

    def run(self) -> None:
        try:
            result = self._fn()
        except BackendError as exc:
            self.failed.emit(exc.message, exc.detail or exc.kind)
        except Exception as exc:
            log.exception("background task failed")
            self.failed.emit(str(exc) or type(exc).__name__, repr(exc))
        else:
            self.succeeded.emit(result)


class ChatWorker(QThread):
    delta = Signal(str)
    completed = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        manager: InferenceManager,
        messages: list[dict[str, str]],
        params: GenerationParams,
        *,
        system_prompt: str = "",
        model_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.manager = manager
        self.messages = [dict(m) for m in messages]
        self.params = params
        self.system_prompt = system_prompt
        self.model_label = model_label
        self.cancel_token = CancelToken()
        self._last_emit = 0.0
        self._pending = ""

    def stop(self) -> None:
        self.cancel_token.cancel()

    def run(self) -> None:
        parts: list[str] = []
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        self._last_emit = time.monotonic()
        try:
            for event in self.manager.stream_chat(
                self.messages,
                self.params,
                system_prompt=self.system_prompt,
                cancel=self.cancel_token,
            ):
                if event.type == "delta":
                    parts.append(event.text)
                    self._pending += event.text
                    now = time.monotonic()
                    if now - self._last_emit >= 0.03:
                        self.delta.emit(self._pending)
                        self._pending = ""
                        self._last_emit = now
                elif event.type == "usage":
                    usage = event.usage
                elif event.type == "done":
                    finish_reason = event.finish_reason
            if self._pending:
                self.delta.emit(self._pending)
                self._pending = ""
            self.completed.emit(
                {
                    "content": "".join(parts),
                    "usage": usage,
                    "finish_reason": finish_reason or "stop",
                    "cancelled": self.cancel_token.is_set()
                    or finish_reason == "cancelled",
                    "model_label": self.model_label,
                }
            )
        except BackendError as exc:
            if self.cancel_token.is_set():
                self.completed.emit(
                    {
                        "content": "".join(parts),
                        "usage": usage,
                        "finish_reason": "cancelled",
                        "cancelled": True,
                        "error": exc.message,
                        "model_label": self.model_label,
                    }
                )
            else:
                self.failed.emit(exc.message, exc.detail or exc.kind)
        except Exception as exc:
            log.exception("chat worker failed")
            self.failed.emit(str(exc) or type(exc).__name__, repr(exc))


class ModelLoadWorker(QThread):
    finished_load = Signal(object)
    failed = Signal(str, str)

    def __init__(
        self,
        manager: InferenceManager,
        target: ModelTarget,
        options: LoadOptions,
        parent=None,
    ):
        super().__init__(parent)
        self.manager = manager
        self.target = target
        self.options = options

    def run(self) -> None:
        try:
            result: LoadResult = self.manager.load(self.target, self.options)
        except BackendError as exc:
            self.failed.emit(exc.message, exc.detail)
            return
        except Exception as exc:
            log.exception("model load worker failed")
            self.failed.emit(str(exc), repr(exc))
            return
        self.finished_load.emit(result)


class UnloadWorker(QThread):
    finished_unload = Signal()

    def __init__(self, manager: InferenceManager, parent=None):
        super().__init__(parent)
        self.manager = manager

    def run(self) -> None:
        try:
            self.manager.unload()
        except Exception:
            log.exception("unload failed")
        finally:
            self.finished_unload.emit()
