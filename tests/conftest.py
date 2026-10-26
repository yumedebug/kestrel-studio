"""Shared pytest fixtures.

The test suite is hermetic: configuration and data directories are redirected
to a temporary folder before any application module is imported, and Qt always
runs with the offscreen platform plugin.
"""

from __future__ import annotations

import os
import struct
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import pytest

# Must be set before PySide6/app paths are imported by test modules.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("KESTREL_NO_DIALOGS", "1")
_SESSION_TMP = Path(tempfile.mkdtemp(prefix="kestrel-tests-"))
os.environ["KESTREL_CONFIG_DIR"] = str(_SESSION_TMP / "config")
os.environ["KESTREL_DATA_DIR"] = str(_SESSION_TMP / "data")

from app.core.credentials import MemoryCredentialStore  # noqa: E402
from app.core.settings import SettingsStore  # noqa: E402


@pytest.fixture()
def settings(tmp_path: Path) -> SettingsStore:
    store = SettingsStore(tmp_path / "settings.json")
    store.update(
        {
            "general": {"model_directory": str(tmp_path / "models")},
            "inference": {"load_timeout_seconds": 15},
            "server": {"host": "127.0.0.1", "port": 1234, "auto_start": False},
        },
        save=False,
    )
    return store


@pytest.fixture()
def credentials() -> MemoryCredentialStore:
    return MemoryCredentialStore()


def _write_string(fh, value: str) -> None:
    raw = value.encode("utf-8")
    fh.write(struct.pack("<Q", len(raw)))
    fh.write(raw)


def _write_value(fh, value: Any) -> None:
    if isinstance(value, bool):
        fh.write(struct.pack("<I", 7))
        fh.write(struct.pack("<B", 1 if value else 0))
    elif isinstance(value, int):
        fh.write(struct.pack("<I", 4))
        fh.write(struct.pack("<I", value))
    elif isinstance(value, float):
        fh.write(struct.pack("<I", 6))
        fh.write(struct.pack("<f", value))
    elif isinstance(value, str):
        fh.write(struct.pack("<I", 8))
        _write_string(fh, value)
    else:
        raise TypeError(f"unsupported GGUF test value: {value!r}")


def write_gguf(
    path: Path,
    metadata: dict[str, Any],
    *,
    tensors: Iterable[tuple[str, tuple[int, ...]]] | None = None,
    version: int = 3,
    magic: bytes = b"GGUF",
    truncate: int | None = None,
) -> Path:
    """Create a minimal but structurally valid GGUF file for tests."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor_list = list(tensors or [])
    with path.open("wb") as fh:
        fh.write(magic)
        fh.write(struct.pack("<I", version))
        if version == 1:
            fh.write(struct.pack("<I", len(tensor_list)))
            fh.write(struct.pack("<I", len(metadata)))
        else:
            fh.write(struct.pack("<Q", len(tensor_list)))
            fh.write(struct.pack("<Q", len(metadata)))
        for key, value in metadata.items():
            _write_string(fh, key)
            _write_value(fh, value)
        for name, dims in tensor_list:
            _write_string(fh, name)
            fh.write(struct.pack("<I", len(dims)))
            for dim in dims:
                fh.write(struct.pack("<Q", dim))
            fh.write(struct.pack("<I", 0))  # ggml type
            fh.write(struct.pack("<Q", 0))  # offset
    if truncate is not None:
        data = path.read_bytes()[:truncate]
        path.write_bytes(data)
    return path


@pytest.fixture()
def gguf_file(tmp_path: Path) -> Callable[..., Path]:
    def factory(
        name: str = "tiny-model.Q4_K_M.gguf",
        *,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Path:
        values = {
            "general.architecture": "llama",
            "general.name": "Tiny Test Model",
            "general.file_type": 15,
            "llama.context_length": 4096,
            "llama.embedding_length": 2048,
            "llama.block_count": 22,
            "llama.attention.head_count": 32,
            "llama.attention.head_count_kv": 8,
            "tokenizer.ggml.model": "gpt2",
        }
        if metadata:
            values.update(metadata)
        return write_gguf(
            tmp_path / name,
            values,
            tensors=kwargs.pop("tensors", [("token_embd.weight", (2048, 32000))]),
            **kwargs,
        )

    return factory


@pytest.fixture()
def wait_until() -> Callable[..., Any]:
    def wait(predicate: Callable[[], Any], timeout: float = 10.0, interval: float = 0.02):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            last = predicate()
            if last:
                return last
            time.sleep(interval)
        raise AssertionError(f"condition was not met within {timeout}s (last={last!r})")

    return wait


def free_port() -> int:
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def fresh_id(prefix: str = "test") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"
