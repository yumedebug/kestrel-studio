"""Safe subprocess execution with streaming output, timeouts and tree kills.

Only this module starts processes. Every command the agent runs goes through the
tool executor, which calls :class:`ProcessManager` after the permission manager
has approved the exact command string.
"""

from __future__ import annotations

import locale
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

SHELL_LABELS = {
    "cmd": "CMD (cmd.exe)",
    "powershell": "PowerShell",
    "python": "Python",
    "git": "Git",
}

_CREATE_FLAGS = 0
if os.name == "nt":  # pragma: no cover - Windows only
    _CREATE_FLAGS = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW


@dataclass
class CommandResult:
    ok: bool
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    command: str
    shell: str
    cwd: str
    timed_out: bool = False
    cancelled: bool = False
    argv: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration": self.duration,
            "command": self.command,
            "shell": self.shell,
            "cwd": self.cwd,
            "timed_out": self.timed_out,
            "cancelled": self.cancelled,
            "argv": list(self.argv),
        }

    @property
    def summary(self) -> str:
        if self.cancelled:
            return "コマンドは停止されました。"
        if self.timed_out:
            return f"コマンドがタイムアウトしました（{self.duration:.1f}s）。"
        state = "成功" if self.ok else "失敗"
        return f"{state}: 終了コード {self.exit_code} ({self.duration:.1f}s)"


def _decode(raw: bytes) -> str:
    for encoding in ("utf-8", locale.getpreferredencoding(False), "cp932", "latin-1"):
        try:
            return raw.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", "replace")


class ProcessManager:
    """Runs commands in worker threads' context and tracks every child process."""

    def __init__(self, *, default_timeout: float = 120.0):
        self.default_timeout = float(default_timeout)
        self._lock = threading.RLock()
        self._procs: dict[str, subprocess.Popen] = {}
        self._cancelled: set[str] = set()
        self._cwd_cache: dict[str, str] = {}

    # -- command construction ---------------------------------------------
    @staticmethod
    def powershell_executable() -> str:
        found = shutil.which("pwsh") or shutil.which("pwsh.exe")
        if found:
            return found
        return shutil.which("powershell") or "powershell.exe"

    def shell_argv(self, shell: str, command: str) -> list[str]:
        """Build an argument list for the requested shell.

        Security flags never weaken the system: PowerShell runs with the
        machine's execution policy and Windows' own UAC handles elevation.
        """
        name = (shell or "cmd").casefold()
        if name == "powershell":
            return [
                self.powershell_executable(),
                "-NoLogo",
                "-NoProfile",
                "-NonInteractive",
                "-Command",
                command,
            ]
        if name == "python":
            stripped = command.strip()
            if re.match(r"^(python3?|py|pip3?|pytest)\b", stripped, re.IGNORECASE):
                return self.shell_argv("cmd", stripped)
            if not any(part in stripped for part in ("\n", "\r")) and ";" not in stripped:
                # Single expression / statement: run it in the app's interpreter.
                return [sys.executable, "-c", command]
            return [sys.executable, "-c", command]
        if name == "git":
            stripped = command.strip()
            full = stripped if re.match(r"^git\b", stripped, re.IGNORECASE) else f"git {stripped}"
            return self.shell_argv("cmd", full)
        if os.name == "nt":
            return ["cmd.exe", "/d", "/s", "/c", command]
        return ["/bin/sh", "-c", command]

    # -- execution ---------------------------------------------------------
    def run(
        self,
        command: str,
        *,
        shell: str = "cmd",
        cwd: str = "",
        timeout: float | None = None,
        on_output: Callable[[str, str], None] | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        text = str(command or "").strip()
        started = time.monotonic()
        working_dir = self._resolve_cwd(cwd)
        limit = float(timeout if timeout is not None else self.default_timeout)
        if not text:
            return CommandResult(
                False, None, "", "The command is empty.", 0.0, text, shell, working_dir
            )
        argv = self.shell_argv(shell, text)
        run_id = uuid.uuid4().hex
        started = time.monotonic()
        environment = os.environ.copy()
        environment.setdefault("PYTHONIOENCODING", "utf-8")
        environment.setdefault("PYTHONUTF8", "1")
        if env:
            environment.update({str(k): str(v) for k, v in env.items()})

        try:
            proc = subprocess.Popen(
                argv,
                cwd=working_dir or None,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=environment,
                creationflags=_CREATE_FLAGS,
            )
        except (OSError, ValueError) as exc:
            return CommandResult(
                False,
                None,
                "",
                f"Could not start the command: {exc}",
                time.monotonic() - started,
                text,
                shell,
                working_dir,
                argv=tuple(argv),
            )

        with self._lock:
            self._procs[run_id] = proc

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def pump(stream, chunks: list[str], label: str) -> None:
            try:
                for raw in iter(stream.readline, b""):
                    if not raw:
                        break
                    chunk = _decode(raw)
                    chunks.append(chunk)
                    if on_output is not None:
                        try:
                            on_output(label, chunk)
                        except Exception:
                            pass
            except (ValueError, OSError):
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        readers = [
            threading.Thread(
                target=pump, args=(proc.stdout, stdout_chunks, "stdout"), daemon=True
            ),
            threading.Thread(
                target=pump, args=(proc.stderr, stderr_chunks, "stderr"), daemon=True
            ),
        ]
        for thread in readers:
            thread.start()

        timed_out = False
        cancelled = False
        try:
            proc.wait(timeout=max(1.0, limit))
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(proc)
        finally:
            for thread in readers:
                thread.join(timeout=5.0)
            with self._lock:
                cancelled = run_id in self._cancelled
                self._procs.pop(run_id, None)
                self._cancelled.discard(run_id)
            if proc.poll() is None:
                self._kill_tree(proc)

        duration = time.monotonic() - started
        exit_code = proc.returncode
        ok = (
            exit_code == 0
            and not timed_out
            and not cancelled
        )
        return CommandResult(
            ok=ok,
            exit_code=exit_code,
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            duration=duration,
            command=text,
            shell=shell,
            cwd=working_dir,
            timed_out=timed_out,
            cancelled=cancelled,
            argv=tuple(argv),
        )

    # -- process control ---------------------------------------------------
    def _kill_tree(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return
        if os.name == "nt":  # pragma: no cover - Windows only
            try:
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                    capture_output=True,
                    creationflags=_CREATE_FLAGS,
                    timeout=15,
                )
            except (OSError, subprocess.SubprocessError):
                pass
        try:
            proc.terminate()
        except (OSError, ValueError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except (OSError, ValueError):
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass

    def stop_all(self) -> int:
        """Stop every running child process (used by Stop Task and on exit)."""
        with self._lock:
            items = list(self._procs.items())
            for run_id, _proc in items:
                self._cancelled.add(run_id)
        for _run_id, proc in items:
            self._kill_tree(proc)
        return len(items)

    def running(self) -> int:
        with self._lock:
            return sum(1 for proc in self._procs.values() if proc.poll() is None)

    # -- helpers -----------------------------------------------------------
    def _resolve_cwd(self, cwd: str) -> str:
        text = str(cwd or "").strip()
        if not text:
            return os.getcwd()
        try:
            path = Path(text).expanduser()
        except (OSError, ValueError):
            return os.getcwd()
        if path.is_dir():
            return str(path)
        return os.getcwd()

    def validate_cwd(self, cwd: str) -> bool:
        text = str(cwd or "").strip()
        if not text:
            return True
        try:
            return Path(text).expanduser().is_dir()
        except (OSError, ValueError):
            return False


def shell_label(shell: str) -> str:
    return SHELL_LABELS.get((shell or "").casefold(), shell)


__all__ = ["CommandResult", "ProcessManager", "SHELL_LABELS", "shell_label"]
