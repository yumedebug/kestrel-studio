"""Low-memory resource policy used without optional monitoring dependencies."""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    physical_bytes: int | None
    low_memory: bool
    ultra_low: bool
    context_length: int
    batch_size: int
    cpu_threads: int
    max_output_tokens: int
    history_limit: int
    terminal_output_chars: int

    @property
    def label(self) -> str:
        if self.physical_bytes is None:
            return "Celeron safe" if self.ultra_low else "standard (memory unavailable)"
        if self.ultra_low:
            return "Celeron / low-memory safe"
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


def logical_cpu_count() -> int:
    """Return a conservative CPU count for systems with unusual reporting."""
    try:
        return max(1, int(os.cpu_count() or 1))
    except (TypeError, ValueError):
        return 1


def detect_budget(settings: object | None = None) -> ResourceBudget:
    total = physical_memory_bytes()
    low = total is not None and total <= 4 * 1024**3
    # Two logical CPUs is a useful proxy for Celeron-class machines.  The
    # policy is intentionally conservative: it only lowers optional limits
    # and never disables a feature.
    ultra = low or logical_cpu_count() <= 2
    get = getattr(settings, "get", lambda _key, default=None: default)
    if ultra:
        return ResourceBudget(
            total, low, ultra,
            min(1024, int(get("inference.default_context_length", 1024) or 1024)),
            min(32, int(get("inference.batch_size", 32) or 32)),
            1,
            min(256, int(get("chat.max_tokens", 256) or 256)),
            24,
            4_000,
        )
    return ResourceBudget(
        total, False, False,
        min(8192, int(get("inference.default_context_length", 4096) or 4096)),
        min(512, int(get("inference.batch_size", 512) or 512)),
        min(8, int(get("inference.cpu_threads", 4) or 4)),
        min(2048, int(get("chat.max_tokens", 1024) or 1024)),
        100,
        12_000,
    )


__all__ = [
    "ResourceBudget",
    "detect_budget",
    "logical_cpu_count",
    "physical_memory_bytes",
]
