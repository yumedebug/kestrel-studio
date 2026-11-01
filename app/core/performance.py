"""Lightweight Windows performance sampling without a third-party monitor."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class PerformanceSnapshot:
    timestamp: float
    cpu_percent: float | None
    memory_percent: float | None
    memory_used_bytes: int | None
    memory_total_bytes: int | None
    disk_free_bytes: int | None
    disk_total_bytes: int | None
    process_count: int | None
    gpu_memory_used_bytes: int | None = None
    gpu_memory_total_bytes: int | None = None
    model_memory_bytes: int | None = None
    inference_tokens_per_second: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "memory_used_bytes": self.memory_used_bytes,
            "memory_total_bytes": self.memory_total_bytes,
            "disk_free_bytes": self.disk_free_bytes,
            "disk_total_bytes": self.disk_total_bytes,
            "process_count": self.process_count,
            "gpu_memory_used_bytes": self.gpu_memory_used_bytes,
            "gpu_memory_total_bytes": self.gpu_memory_total_bytes,
            "model_memory_bytes": self.model_memory_bytes,
            "inference_tokens_per_second": self.inference_tokens_per_second,
        }


class PerformanceMonitor:
    """Collects safe, read-only host metrics suitable for a polling UI."""

    def __init__(self) -> None:
        self._last_cpu: tuple[int, int, float] | None = None

    def sample(self, *, store: Any = None, task_id: str = "", inference: Any = None) -> PerformanceSnapshot:
        now = time.time()
        cpu = self._cpu_percent(now)
        memory_used, memory_total = self._memory()
        disk_free, disk_total = self._disk()
        gpu_used, gpu_total = self._gpu_memory()
        inference_data: dict[str, Any] = {}
        if inference is not None:
            try:
                status = inference.status()
                inference_data = status.get("performance") or {}
            except Exception:
                inference_data = {}
        snapshot = PerformanceSnapshot(
            timestamp=now,
            cpu_percent=cpu,
            memory_percent=(
                round(memory_used / memory_total * 100, 1)
                if memory_used is not None and memory_total
                else None
            ),
            memory_used_bytes=memory_used,
            memory_total_bytes=memory_total,
            disk_free_bytes=disk_free,
            disk_total_bytes=disk_total,
            process_count=self._process_count(),
            gpu_memory_used_bytes=gpu_used,
            gpu_memory_total_bytes=gpu_total,
            model_memory_bytes=inference_data.get("model_memory_bytes"),
            inference_tokens_per_second=inference_data.get("tokens_per_second"),
        )
        if store is not None:
            try:
                store.add_performance_sample(snapshot.as_dict(), task_id=task_id)
            except Exception:
                pass
        return snapshot

    @staticmethod
    def _gpu_memory() -> tuple[int | None, int | None]:
        if os.name != "nt":
            return None, None
        try:
            completed = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=2, check=False,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            line = completed.stdout.splitlines()[0] if completed.stdout.splitlines() else ""
            used, total = [int(value.strip()) * 1024 * 1024 for value in line.split(",")[:2]]
            return used, total
        except (OSError, ValueError, IndexError, subprocess.SubprocessError):
            return None, None

    def _cpu_percent(self, now: float) -> float | None:
        if os.name != "nt":
            return None
        try:
            kernel, user = ctypes.c_ulonglong(), ctypes.c_ulonglong()
            idle = ctypes.c_ulonglong()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                return None
            total = kernel.value + user.value
            idle_value = idle.value
            previous = self._last_cpu
            self._last_cpu = (total, idle_value, now)
            if previous is None:
                return None
            total_delta = total - previous[0]
            idle_delta = idle_value - previous[1]
            return round(max(0.0, min(100.0, (total_delta - idle_delta) / total_delta * 100)), 1)
        except (AttributeError, OSError, ZeroDivisionError):
            return None

    @staticmethod
    def _memory() -> tuple[int | None, int | None]:
        if os.name != "nt":
            return None, None
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("length", ctypes.c_ulong),
                ("memory_load", ctypes.c_ulong),
                ("total_phys", ctypes.c_ulonglong),
                ("available_phys", ctypes.c_ulonglong),
                ("total_page", ctypes.c_ulonglong),
                ("available_page", ctypes.c_ulonglong),
                ("total_virtual", ctypes.c_ulonglong),
                ("available_virtual", ctypes.c_ulonglong),
                ("available_extended", ctypes.c_ulonglong),
            ]

        try:
            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return None, None
            return status.total_phys - status.available_phys, status.total_phys
        except (AttributeError, OSError):
            return None, None

    @staticmethod
    def _disk() -> tuple[int | None, int | None]:
        try:
            usage = shutil.disk_usage(os.environ.get("SystemDrive", "C:") + "\\")
            return usage.free, usage.total
        except OSError:
            return None, None

    @staticmethod
    def _process_count() -> int | None:
        if os.name != "nt":
            return None
        try:
            return sum(1 for _ in os.popen("tasklist /fo csv /nh"))
        except OSError:
            return None


__all__ = ["PerformanceMonitor", "PerformanceSnapshot"]
