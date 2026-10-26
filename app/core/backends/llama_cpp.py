"""llama.cpp backend: manages a llama-server child process.

The child exposes llama.cpp's OpenAI-compatible API on an ephemeral loopback
port; this backend streams from it.  Keeping the process separate means a
crash in the engine cannot take down the GUI, and it gives us one shared
backend for the chat UI and the local OpenAI-compatible server.
"""

from __future__ import annotations

import os
import socket
import subprocess
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import httpx

from app.core.backends.base import InferenceBackend
from app.core.backends.http_utils import error_from_body, iter_openai_sse
from app.core.gguf import is_probably_gguf
from app.core.paths import app_root, data_dir, logs_dir, runtime_dir
from app.core.settings import SettingsStore
from app.core.types import (
    BackendError,
    CancelToken,
    GenerationParams,
    LoadOptions,
    ModelTarget,
    StreamEvent,
)

CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail_text(lines: deque[str], limit: int = 2000) -> str:
    text = "\n".join(list(lines)[-25:]).strip()
    return text[-limit:]


class LlamaCppBackend(InferenceBackend):
    name = "llama.cpp"

    def __init__(
        self,
        settings: SettingsStore,
        *,
        server_url_override: str | None = None,
        process_factory: Callable[[list[str], dict[str, Any]], subprocess.Popen] | None = None,
    ):
        self.settings = settings
        self._server_url_override = server_url_override
        self._process_factory = process_factory
        self._proc: subprocess.Popen | None = None
        self._client: httpx.Client | None = None
        self._server_url = ""
        self._exe = Path()
        self._log_file = None
        self._log_lines: deque[str] = deque(maxlen=300)
        self._log_thread: threading.Thread | None = None
        self._model = ""
        self._api_model = "kestrel-local"
        self.last_warning = ""
        self._stopping = False

    # -- discovery ---------------------------------------------------------
    def _backend_order(self) -> list[str]:
        backend = self.settings.get("inference.backend", "auto")
        if backend == "cuda":
            return ["cuda", "cpu", ""]
        if backend == "vulkan":
            return ["vulkan", "cpu", ""]
        if backend == "cpu":
            return ["cpu", ""]
        return ["cuda", "vulkan", "cpu", ""]

    def resolve_executable(self, force_cpu: bool = False) -> Path:
        explicit = str(self.settings.get("inference.llama_server_path", "") or "").strip()
        if explicit:
            path = Path(explicit).expanduser()
            if path.is_file():
                return path
            raise BackendError(
                f"llama-server path does not exist: {path}",
                kind="invalid_config",
                status_code=400,
            )
        order = ["cpu", ""] if force_cpu else self._backend_order()
        roots = [runtime_dir(), data_dir() / "runtime", app_root() / "runtime"]
        for root in roots:
            for sub in order:
                candidate = root / sub / "llama-server.exe" if sub else root / "llama-server.exe"
                if candidate.is_file():
                    return candidate
        # Developer convenience: a llama-server installed on PATH.
        import shutil

        found = shutil.which("llama-server")
        if found:
            return Path(found)
        raise BackendError(
            "llama.cpp runtime not found. Run scripts\\fetch_llama_cpp.py "
            "(or set the llama-server path in Settings → Inference).",
            kind="runtime_missing",
            status_code=503,
        )

    # -- lifecycle ---------------------------------------------------------
    def load(self, target: ModelTarget, options: LoadOptions) -> None:
        if target.kind != "local":
            raise BackendError("LlamaCppBackend requires a local model target.", kind="invalid_target")
        model_path = Path(target.path)
        if not model_path.is_file():
            raise BackendError(
                f"Model file not found: {model_path}",
                kind="model_not_found",
                status_code=404,
            )
        if not is_probably_gguf(model_path):
            raise BackendError(
                f"{model_path.name} is not a valid GGUF file (bad header).",
                kind="invalid_model",
                status_code=400,
            )

        self._model = target.display_name
        self._api_model = "kestrel-local"
        self.last_warning = ""
        self._stopping = False
        self._log_lines.clear()

        if self._server_url_override:
            self._server_url = self._server_url_override.rstrip("/")
            self._client = self._make_client()
            return

        effective = options
        exe = self.resolve_executable()
        attempt = 0
        while True:
            attempt += 1
            port = _free_port()
            cmd = self.build_command(exe, model_path, effective, port)
            try:
                self._spawn(cmd, exe.parent)
                self._server_url = f"http://127.0.0.1:{port}"
                self._client = self._make_client()
                self._wait_until_ready(effective)
                break
            except BackendError as exc:
                self._terminate_process()
                self._close_client()
                can_fallback = (
                    attempt == 1
                    and exc.kind == "cuda_unavailable"
                    and effective.gpu_layers
                    and bool(self.settings.get("inference.auto_cpu_fallback", True))
                )
                if can_fallback:
                    self.last_warning = (
                        "GPU/CUDA could not be initialized. Fell back to CPU inference "
                        "(GPU layers = 0)."
                    )
                    try:
                        exe = self.resolve_executable(force_cpu=True)
                    except BackendError:
                        exe = exe  # A CUDA build still runs with -ngl 0.
                    effective = LoadOptions(
                        context_length=effective.context_length,
                        gpu_layers=0,
                        cpu_threads=effective.cpu_threads,
                        batch_size=effective.batch_size,
                    )
                    continue
                raise

    def build_command(
        self, exe: Path, model_path: Path, options: LoadOptions, port: int
    ) -> list[str]:
        ctx = int(options.context_length or 4096)
        gpu = int(options.gpu_layers or 0)
        threads = int(options.cpu_threads or 1)
        batch = int(options.batch_size or 512)
        return [
            str(exe),
            "-m",
            str(model_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(ctx),
            "--n-gpu-layers",
            str(gpu),
            "--threads",
            str(threads),
            "--batch-size",
            str(batch),
            "--parallel",
            "1",
            "--alias",
            self._api_model,
        ]

    def _spawn(self, cmd: list[str], cwd: Path) -> None:
        log_path = logs_dir() / "llama-server.log"
        logs_dir().mkdir(parents=True, exist_ok=True)
        try:
            self._log_file = open(log_path, "a", encoding="utf-8", errors="replace")
            self._log_file.write(
                f"\n=== {datetime.now().isoformat(timespec='seconds')} "
                f"{' '.join(cmd)}\n"
            )
            self._log_file.flush()
        except OSError:
            self._log_file = None

        env = os.environ.copy()
        env.setdefault("LLAMA_LOG_COLORS", "0")
        kwargs: dict[str, Any] = dict(
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
            creationflags=CREATE_NO_WINDOW,
        )
        try:
            if self._process_factory:
                self._proc = self._process_factory(cmd, kwargs)
            else:
                self._proc = subprocess.Popen(cmd, **kwargs)
        except OSError as exc:
            raise BackendError(
                f"Could not start llama-server: {exc}",
                kind="runtime_missing",
                status_code=503,
            ) from exc

        self._log_thread = threading.Thread(
            target=self._pump_logs, name="llama-server-log", daemon=True
        )
        self._log_thread.start()

    def _pump_logs(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = line.rstrip("\r\n")
                if not line:
                    continue
                self._log_lines.append(line)
                if self._log_file is not None:
                    try:
                        self._log_file.write(line + "\n")
                        self._log_file.flush()
                    except (OSError, ValueError):
                        pass
        except (OSError, ValueError):
            pass

    def _make_client(self) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0),
            follow_redirects=False,
        )

    def _wait_until_ready(self, options: LoadOptions) -> None:
        assert self._client is not None
        timeout = float(self.settings.get("inference.load_timeout_seconds", 180))
        deadline = time.monotonic() + max(15.0, timeout)
        health_404 = False
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                raise self._startup_error(proc.returncode)
            try:
                response = self._client.get(f"{self._server_url}/health", timeout=3.0)
                if response.status_code == 200:
                    return
                if response.status_code == 404:
                    health_404 = True
            except httpx.HTTPError:
                pass
            if health_404:
                try:
                    response = self._client.get(
                        f"{self._server_url}/v1/models", timeout=3.0
                    )
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
            time.sleep(0.3)
        raise BackendError(
            f"llama-server did not become ready within {timeout:.0f} seconds. "
            f"The model may be too large for available memory. See "
            f"{logs_dir() / 'llama-server.log'} for details.",
            kind="load_timeout",
            status_code=504,
            detail=_tail_text(self._log_lines),
        )

    def _startup_error(self, returncode: int | None) -> BackendError:
        tail = _tail_text(self._log_lines)
        lower = tail.lower()
        detail = f"exit code {returncode}\n{tail}"
        if any(
            marker in lower
            for marker in (
                "cuda error",
                "no cuda-capable device",
                "failed to initialize cuda",
                "cuda driver",
                "cudart",
                "no usable gpu",
                "ggml_cuda_init",
            )
        ):
            return BackendError(
                "llama.cpp could not initialize CUDA. Install the CUDA runtime / update "
                "your NVIDIA driver, choose the CPU backend, or load with GPU layers = 0. "
                f"Details were written to {logs_dir() / 'llama-server.log'}.",
                kind="cuda_unavailable",
                status_code=503,
                detail=detail,
            )
        if any(
            marker in lower
            for marker in ("out of memory", "failed to allocate", "cuda malloc", "alloc failed")
        ):
            return BackendError(
                "Not enough memory to load this model. Lower the context length, choose a "
                "smaller quantization, or set GPU layers = 0 and retry. "
                f"Details: {logs_dir() / 'llama-server.log'}",
                kind="out_of_memory",
                status_code=507,
                detail=detail,
            )
        if any(
            marker in lower
            for marker in (
                "invalid magic",
                "not a gguf",
                "failed to load model",
                "error loading model",
                "unknown model architecture",
                "unsupported model",
            )
        ):
            return BackendError(
                "llama.cpp could not read this GGUF model. The file may be corrupt, "
                "truncated, or use an unsupported architecture.",
                kind="invalid_model",
                status_code=400,
                detail=detail,
            )
        if "address already in use" in lower or "failed to bind" in lower:
            return BackendError(
                "llama.cpp could not bind its loopback port. Retry; if it persists, "
                "another program is blocking local ports.",
                kind="port_conflict",
                status_code=409,
                detail=detail,
            )
        return BackendError(
            "llama-server stopped while loading the model. "
            f"See {logs_dir() / 'llama-server.log'} for the full log. "
            f"Last output: {tail[-400:] or '(empty)'}",
            kind="model_load_failed",
            status_code=500,
            detail=detail,
        )

    def unload(self) -> None:
        self._stopping = True
        self._close_client()
        self._terminate_process()
        if self._log_thread is not None:
            self._log_thread.join(timeout=1.0)
            self._log_thread = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except OSError:
                pass
            self._log_file = None
        self._server_url = ""
        self._model = ""

    def _close_client(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None

    def _terminate_process(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=6)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=4)
                    except subprocess.TimeoutExpired:
                        pass
        except (OSError, ValueError):
            pass
        finally:
            try:
                if proc.stdout:
                    proc.stdout.close()
            except (OSError, ValueError):
                pass

    def describe(self) -> dict[str, str]:
        data = super().describe()
        data.update(
            {
                "server_url": self._server_url,
                "model": self._model,
                "pid": str(self._proc.pid) if self._proc and self._proc.poll() is None else "",
            }
        )
        return data

    # -- inference ---------------------------------------------------------
    def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        if not self._server_url:
            raise BackendError("No local model is loaded.", kind="not_loaded", status_code=503)
        payload: dict[str, Any] = {
            "model": self._api_model,
            "messages": [dict(m) for m in messages],
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "repeat_penalty": params.repeat_penalty,
            "max_tokens": params.max_tokens,
        }
        if params.seed >= 0:
            payload["seed"] = params.seed
        if params.stop:
            payload["stop"] = params.stop[:4]

        finish_reason: str | None = None
        try:
            with httpx.Client(
                timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=10.0)
            ) as client:
                with client.stream(
                    "POST",
                    f"{self._server_url}/v1/chat/completions",
                    json=payload,
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise error_from_body(response.status_code, body, "llama.cpp")
                    cancel.add_callback(response.close)
                    try:
                        for kind, value in iter_openai_sse(response, cancel, "llama.cpp"):
                            if kind == "delta":
                                yield StreamEvent("delta", text=str(value))
                            elif kind == "usage":
                                yield StreamEvent("usage", usage=value)
                            elif kind == "finish":
                                finish_reason = str(value)
                    finally:
                        cancel.remove_callback(response.close)
        except httpx.HTTPError as exc:
            if cancel.is_set():
                yield StreamEvent("done", finish_reason="cancelled")
                return
            raise BackendError(
                f"Lost the connection to llama-server: {exc}. "
                "The engine may have crashed; unload and reload the model.",
                kind="engine_crashed",
                status_code=502,
                detail=_tail_text(self._log_lines),
            ) from exc

        if cancel.is_set():
            yield StreamEvent("done", finish_reason="cancelled")
        else:
            yield StreamEvent("done", finish_reason=finish_reason or "stop")
