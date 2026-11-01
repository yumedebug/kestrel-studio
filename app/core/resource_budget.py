"""Low-memory resource policy used without optional monitoring dependencies."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    physical_bytes: int | None
    low_memory: bool
    context_length: int
    batch_size: int
    cpu_threads: int
    max_output_tokens: int
    history_limit: int
    terminal_output_chars: int

    @property
    def label(self) -> str:
        if self.physical_bytes is None:
            return "standard (memory unavailable)"
        return "4GB safe" if self.low_memory else "standard"


def physical_memory_bytes() -> int | None:
    if os.name != "nt":
        return None

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
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys)
    except (AttributeError, OSError):
        pass
    return None


def detect_budget(settings: object | None = None) -> ResourceBudget:
    total = physical_memory_bytes()
    low = total is not None and total <= 4 * 1024**3
    get = getattr(settings, "get", lambda _key, default=None: default)
    if low:
        return ResourceBudget(
            total, True,
            min(2048, int(get("inference.default_context_length", 2048) or 2048)),
            min(128, int(get("inference.batch_size", 128) or 128)),
            min(2, int(get("inference.cpu_threads", 2) or 2)),
            min(512, int(get("chat.max_tokens", 512) or 512)),
            40,
            8_000,
        )
    return ResourceBudget(
        total, False,
        min(8192, int(get("inference.default_context_length", 4096) or 4096)),
        min(512, int(get("inference.batch_size", 512) or 512)),
        min(8, int(get("inference.cpu_threads", 4) or 4)),
        min(2048, int(get("chat.max_tokens", 1024) or 1024)),
        100,
        12_000,
    )


__all__ = ["ResourceBudget", "detect_budget", "physical_memory_bytes"]
