"""Lifecycle management for the embedded OpenAI-compatible HTTP server."""

from __future__ import annotations

import logging
import socket
import threading
import time
from typing import Any

import uvicorn

from app.api.routes import create_app
from app.core.inference import InferenceManager
from app.core.settings import SettingsStore

log = logging.getLogger(__name__)


class ApiServerError(RuntimeError):
    def __init__(self, message: str, kind: str = "server_error"):
        super().__init__(message)
        self.message = message
        self.kind = kind


def _is_loopback(host: str) -> bool:
    return host in {"127.0.0.1", "localhost", "::1"}


def _port_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET6 if ":" in host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((host, port))
        return True
    except OSError:
        return False


class ApiServer:
    def __init__(self, manager: InferenceManager, settings: SettingsStore):
        self.manager = manager
        self.settings = settings
        self._lock = threading.RLock()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._app = None
        self._host = ""
        self._port = 0
        self._started_at: float | None = None
        self._startup_error = ""

    @property
    def running(self) -> bool:
        with self._lock:
            return bool(self._server and self._server.started)

    def start(
        self,
        host: str | None = None,
        port: int | None = None,
        *,
        allow_external: bool | None = None,
    ) -> dict[str, Any]:
        with self._lock:
            if self._server is not None and self._server.started:
                return self.info()
            if self._thread is not None and self._thread.is_alive():
                raise ApiServerError("The server is already starting.", kind="already_starting")

            host = str(host or self.settings.get("server.host", "127.0.0.1") or "127.0.0.1")
            port = int(port or self.settings.get("server.port", 1234))
            if allow_external is None:
                allow_external = bool(self.settings.get("server.allow_external", False))
            if not _is_loopback(host) and not allow_external:
                raise ApiServerError(
                    f"Refusing to bind {host}: external access is disabled. Enable "
                    "'Allow external access' in Settings → Server first.",
                    kind="external_disabled",
                )
            if not (1024 <= port <= 65535):
                raise ApiServerError(
                    f"Port {port} is outside the allowed range 1024-65535.",
                    kind="invalid_port",
                )
            if not _port_available(host, port):
                raise ApiServerError(
                    f"Port {port} is already in use on {host}. Stop the other program "
                    "or choose a different port in Settings → Server.",
                    kind="port_in_use",
                )

            self._app = create_app(self.manager, self.settings)
            config = uvicorn.Config(
                self._app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
                log_config=None,
                lifespan="off",
                timeout_keep_alive=5,
            )
            self._server = uvicorn.Server(config)
            self._host = host
            self._port = port
            self._started_at = None
            self._startup_error = ""
            self._thread = threading.Thread(
                target=self._run, name="kestrel-api-server", daemon=True
            )
            self._thread.start()

        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            with self._lock:
                server = self._server
                thread = self._thread
                error = self._startup_error
            if error:
                self._reset_after_failure()
                raise ApiServerError(
                    f"The API server could not start: {error}", kind="startup_failed"
                )
            if server is not None and server.started:
                self._started_at = time.time()
                return self.info()
            if thread is not None and not thread.is_alive():
                self._reset_after_failure()
                raise ApiServerError(
                    "The API server exited before it became ready. Check that the port "
                    "is free and that no firewall blocks it.",
                    kind="startup_failed",
                )
            time.sleep(0.05)
        self._reset_after_failure()
        raise ApiServerError(
            "Timed out waiting for the API server to start.",
            kind="startup_timeout",
        )

    def _run(self) -> None:
        server = self._server
        if server is None:
            return
        try:
            server.run()
        except BaseException as exc:  # uvicorn may raise SystemExit on bind errors
            log.exception("API server thread crashed")
            with self._lock:
                self._startup_error = str(exc) or type(exc).__name__
        finally:
            with self._lock:
                self._started_at = None

    def _reset_after_failure(self) -> None:
        with self._lock:
            server = self._server
            if server is not None:
                server.should_exit = True
            self._server = None
            self._thread = None
            self._app = None

    def stop(self) -> None:
        with self._lock:
            server = self._server
            thread = self._thread
            self._server = None
            self._thread = None
        if server is not None:
            server.should_exit = True
            if thread is not None and thread.is_alive():
                thread.join(timeout=10.0)
                if thread.is_alive():
                    log.warning("API server did not stop within 10 seconds")
        self._started_at = None

    def info(self) -> dict[str, Any]:
        with self._lock:
            server = self._server
            running = bool(server and server.started)
            host = self._host or str(self.settings.get("server.host", "127.0.0.1"))
            port = self._port or int(self.settings.get("server.port", 1234))
            app = self._app
            started_at = self._started_at
            thread_alive = bool(self._thread and self._thread.is_alive())
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
        base_url = f"http://{display_host}:{port}"
        status = self.manager.status()
        request_count = 0
        if app is not None:
            try:
                request_count = int(app.state.request_count)
            except Exception:
                request_count = 0
        return {
            "running": running,
            "thread_alive": thread_alive,
            "host": host,
            "port": port,
            "base_url": base_url,
            "api_base_url": f"{base_url}/v1",
            "started_at": started_at,
            "request_count": request_count,
            "active_requests": status["active_requests"],
            "model": status["model_id"],
            "model_name": status["display_name"],
            "external": not _is_loopback(host),
        }

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            log.exception("error while stopping API server")
