"""Single shared inference backend for the GUI and the API server.

Only one backend can be loaded at a time.  Loading a different model first
uploads/cancels in-flight requests, tears down the old backend and starts the
new one, so no orphan llama-server processes are left behind.  API requests get
a clean 503 while the swap is happening.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Iterable, Iterator

from app.core.backends.base import InferenceBackend
from app.core.backends.cloud import CloudBackend
from app.core.backends.llama_cpp import LlamaCppBackend
from app.core.credentials import CredentialStore
from app.core.settings import SettingsStore
from app.core.types import (
    BackendError,
    CancelToken,
    GenerationCancelled,
    GenerationParams,
    LoadOptions,
    LoadResult,
    ModelTarget,
    StreamEvent,
)

log = logging.getLogger(__name__)


class InferenceManager:
    def __init__(
        self,
        settings: SettingsStore,
        credentials: CredentialStore,
        *,
        backend_factory: Callable[[ModelTarget], InferenceBackend] | None = None,
    ):
        self.settings = settings
        self.credentials = credentials
        self._backend_factory = backend_factory
        self._lock = threading.RLock()
        self._backend: InferenceBackend | None = None
        self._target: ModelTarget | None = None
        self._options: LoadOptions | None = None
        self._state = "idle"
        self._error = ""
        self._warning = ""
        self._active = 0
        self._generation = 0
        self._tokens: set[CancelToken] = set()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []
        self._loaded_at: float | None = None

    # -- listeners ---------------------------------------------------------
    def add_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[dict[str, Any]], None]) -> None:
        with self._lock:
            try:
                self._listeners.remove(listener)
            except ValueError:
                pass

    def _emit(self, event: dict[str, Any]) -> None:
        with self._lock:
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                log.exception("inference listener failed")

    def _notify_status(self) -> None:
        self._emit({"type": "status", "status": self.status()})

    # -- status ------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        with self._lock:
            backend = self._backend
            target = self._target
            data = {
                "state": self._state,
                "error": self._error,
                "warning": self._warning,
                "active_requests": self._active,
                "generation": self._generation,
                "loaded_at": self._loaded_at,
                "options": self._options.to_dict() if self._options else None,
                "target": target.to_dict() if target else None,
                "model_id": target.api_model_id if target else "",
                "display_name": target.display_name if target else "",
                "kind": target.kind if target else "",
                "provider": target.provider if target else "",
                "backend": getattr(backend, "name", "") if backend else "",
            }
            if backend is not None:
                try:
                    data["backend_info"] = backend.describe()
                except Exception:
                    data["backend_info"] = {}
            return data

    @property
    def is_ready(self) -> bool:
        with self._lock:
            return self._state == "ready" and self._backend is not None

    @property
    def target(self) -> ModelTarget | None:
        with self._lock:
            return self._target

    # -- loading -----------------------------------------------------------
    def _make_backend(self, target: ModelTarget) -> InferenceBackend:
        if self._backend_factory is not None:
            return self._backend_factory(target)
        if target.kind == "local":
            return LlamaCppBackend(self.settings)
        return CloudBackend(self.settings, self.credentials)

    def load(
        self,
        target: ModelTarget,
        options: LoadOptions | None = None,
        *,
        persist: bool = True,
    ) -> LoadResult:
        options = (options or LoadOptions()).resolved(self.settings)
        with self._lock:
            if self._state == "loading":
                return LoadResult(False, "A model is already loading. Wait for it to finish.")
            self._generation += 1
            generation = self._generation
            old_backend = self._backend
            self._backend = None
            self._state = "loading"
            self._error = ""
            self._warning = ""
            self._loaded_at = None
            tokens = list(self._tokens)
            self._tokens.clear()
            self._active = 0

        for token in tokens:
            token.cancel()
        self._notify_status()

        if old_backend is not None:
            try:
                old_backend.unload()
            except Exception:
                log.exception("failed to unload previous backend")

        backend = self._make_backend(target)
        try:
            backend.load(target, options)
        except BackendError as exc:
            log.warning("model load failed: %s (%s)", exc.message, exc.kind)
            try:
                backend.unload()
            except Exception:
                log.exception("failed to clean up after load error")
            with self._lock:
                if generation == self._generation:
                    self._state = "error"
                    self._error = exc.message
                    self._target = None
                    self._options = None
            self._notify_status()
            return LoadResult(False, exc.message)
        except Exception as exc:
            log.exception("unexpected model load failure")
            try:
                backend.unload()
            except Exception:
                pass
            message = f"Unexpected error while loading the model: {exc}"
            with self._lock:
                if generation == self._generation:
                    self._state = "error"
                    self._error = message
                    self._target = None
            self._notify_status()
            return LoadResult(False, message)

        with self._lock:
            if generation != self._generation:
                # Another load/unload started while this one was running.
                self._generation += 0
                stale = True
            else:
                stale = False
                self._backend = backend
                self._target = target
                self._options = options
                self._state = "ready"
                self._warning = getattr(backend, "last_warning", "") or ""
                self._loaded_at = time.time()
        if stale:
            try:
                backend.unload()
            except Exception:
                log.exception("failed to unload stale backend")
            return LoadResult(False, "Model load was superseded by another request.")

        if persist:
            try:
                if target.kind == "cloud":
                    self.settings.set("cloud.last_target", target.to_dict(), save=False)
                else:
                    self.settings.set("cloud.last_target", target.to_dict(), save=False)
                    self.settings.set("last.local_model_path", target.path, save=False)
                self.settings.save()
            except Exception:
                log.exception("could not persist selected model")

        self._notify_status()
        if self._warning:
            self._emit({"type": "warning", "message": self._warning})
        return LoadResult(
            True,
            f"Loaded {target.display_name}",
            warning=self._warning,
            target=target,
        )

    def unload(self) -> None:
        with self._lock:
            if self._state == "idle" and self._backend is None:
                return
            self._generation += 1
            backend = self._backend
            self._backend = None
            self._target = None
            self._options = None
            self._state = "unloading"
            tokens = list(self._tokens)
            self._tokens.clear()
            self._active = 0
        for token in tokens:
            token.cancel()
        self._notify_status()
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                log.exception("failed to unload backend")
        with self._lock:
            self._state = "idle"
            self._error = ""
            self._warning = ""
            self._loaded_at = None
        self._notify_status()

    def cancel_all(self) -> None:
        with self._lock:
            tokens = list(self._tokens)
        for token in tokens:
            token.cancel()

    # -- generation --------------------------------------------------------
    @staticmethod
    def _with_system(
        messages: Iterable[dict[str, str]], system_prompt: str
    ) -> list[dict[str, str]]:
        cleaned = [
            {"role": str(m.get("role", "user")), "content": str(m.get("content", ""))}
            for m in messages
            if m.get("content")
        ]
        if system_prompt and (not cleaned or cleaned[0]["role"] != "system"):
            cleaned.insert(0, {"role": "system", "content": system_prompt})
        return cleaned

    def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        params: GenerationParams,
        *,
        system_prompt: str = "",
        cancel: CancelToken | None = None,
    ) -> Iterator[StreamEvent]:
        token = cancel or CancelToken()
        with self._lock:
            backend = self._backend
            if backend is None or self._state != "ready":
                raise BackendError(
                    "No model is loaded. Load a local GGUF model in My Models, or "
                    "configure and select a cloud model in Settings.",
                    kind="not_loaded",
                    status_code=503,
                )
            self._tokens.add(token)
            self._active += 1
        self._notify_status()
        full = self._with_system(messages, system_prompt)
        cancelled = False
        try:
            for event in backend.stream_chat(full, params, token):
                if event.type == "done" and event.finish_reason == "cancelled":
                    cancelled = True
                if event.type == "delta" and token.is_set():
                    cancelled = True
                yield event
                if cancelled:
                    break
        except BackendError:
            raise
        except GenerationCancelled:
            yield StreamEvent("done", finish_reason="cancelled")
        except Exception as exc:
            log.exception("streaming inference failed")
            raise BackendError(
                f"The inference backend failed unexpectedly: {exc}",
                kind="inference_error",
                status_code=500,
                detail=repr(exc),
            ) from exc
        finally:
            with self._lock:
                self._tokens.discard(token)
                self._active = max(0, self._active - 1)
            self._notify_status()

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        params: GenerationParams,
        *,
        system_prompt: str = "",
        cancel: CancelToken | None = None,
    ) -> tuple[str, dict[str, int] | None, str | None]:
        parts: list[str] = []
        usage: dict[str, int] | None = None
        finish_reason: str | None = None
        for event in self.stream_chat(
            messages, params, system_prompt=system_prompt, cancel=cancel
        ):
            if event.type == "delta":
                parts.append(event.text)
            elif event.type == "usage":
                usage = event.usage
            elif event.type == "done":
                finish_reason = event.finish_reason
        return "".join(parts), usage, finish_reason

    # -- misc --------------------------------------------------------------
    def shutdown(self) -> None:
        """Stop any running engine. Safe to call multiple times."""
        try:
            self.unload()
        except Exception:
            log.exception("error during inference shutdown")


def build_target_for_local(path: str, display_name: str, entry_id: str = "") -> ModelTarget:
    return ModelTarget(
        kind="local",
        display_name=display_name,
        path=path,
        entry_id=entry_id,
    )


def build_target_for_cloud(provider: str, model: str, display_name: str = "") -> ModelTarget:
    return ModelTarget(
        kind="cloud",
        display_name=display_name or model,
        provider=provider,
        model=model,
    )
