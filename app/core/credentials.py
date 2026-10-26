"""Secure storage for API keys and the Hugging Face token.

On Windows the secrets are written to the Windows Credential Manager via the
Win32 credential APIs (no plaintext on disk).  If that API is unavailable (e.g.
a stripped-down CI environment) an encrypted DPAPI file is used instead.  On
non-Windows developer machines a clearly-marked local file is used; this app's
supported target is Windows.
"""

from __future__ import annotations

import base64
import ctypes
import json
import os
import sys
import threading
from ctypes import wintypes
from pathlib import Path
from typing import Protocol

from app.constants import APP_ID, APP_NAME


class CredentialError(RuntimeError):
    pass


class CredentialStore(Protocol):
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
    def delete(self, key: str) -> None: ...


class MemoryCredentialStore:
    """In-memory store used by tests."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: str) -> None:
        if not value:
            self.delete(key)
            return
        with self._lock:
            self._data[key] = value

    def delete(self, key: str) -> None:
        with self._lock:
            self._data.pop(key, None)


# ---------------------------------------------------------------------------
# Windows Credential Manager
# ---------------------------------------------------------------------------

if os.name == "nt":
    _advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    class _FILETIME(ctypes.Structure):
        _fields_ = [
            ("dwLowDateTime", wintypes.DWORD),
            ("dwHighDateTime", wintypes.DWORD),
        ]

    class _CREDENTIAL_ATTRIBUTEW(ctypes.Structure):
        _fields_ = [
            ("Keyword", wintypes.LPWSTR),
            ("Flags", wintypes.DWORD),
            ("ValueSize", wintypes.DWORD),
            ("Value", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    class _CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", _FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.POINTER(ctypes.c_ubyte)),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.POINTER(_CREDENTIAL_ATTRIBUTEW)),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    _PCREDENTIALW = ctypes.POINTER(_CREDENTIALW)
    _advapi32.CredWriteW.argtypes = [ctypes.POINTER(_CREDENTIALW), wintypes.DWORD]
    _advapi32.CredWriteW.restype = wintypes.BOOL
    _advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_PCREDENTIALW),
    ]
    _advapi32.CredReadW.restype = wintypes.BOOL
    _advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    _advapi32.CredDeleteW.restype = wintypes.BOOL
    _advapi32.CredFree.argtypes = [ctypes.c_void_p]
    _advapi32.CredFree.restype = None
    _kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    _kernel32.LocalFree.restype = ctypes.c_void_p


class WindowsCredentialStore:
    """Store secrets as generic credentials under ``KestrelStudio/<key>``."""

    CRED_TYPE_GENERIC = 1
    CRED_PERSIST_LOCAL_MACHINE = 2
    ERROR_NOT_FOUND = 1168

    def __init__(self) -> None:
        if os.name != "nt":
            raise CredentialError("Windows Credential Manager is only available on Windows")
        self._lock = threading.Lock()

    def _target(self, key: str) -> str:
        return f"{APP_ID}/{key}"

    def get(self, key: str) -> str | None:
        with self._lock:
            target = self._target(key)
            ptr = _PCREDENTIALW()
            ok = _advapi32.CredReadW(
                target, self.CRED_TYPE_GENERIC, 0, ctypes.byref(ptr)
            )
            if not ok:
                err = ctypes.get_last_error()
                if err == self.ERROR_NOT_FOUND:
                    return None
                raise CredentialError(
                    f"CredReadW failed for {key!r} (error {err})"
                )
            try:
                cred = ptr.contents
                if not cred.CredentialBlob or cred.CredentialBlobSize == 0:
                    return ""
                raw = ctypes.string_at(cred.CredentialBlob, cred.CredentialBlobSize)
                return raw.decode("utf-8")
            finally:
                _advapi32.CredFree(ptr)

    def set(self, key: str, value: str) -> None:
        if value is None:
            raise CredentialError("secret must not be None")
        with self._lock:
            encoded = value.encode("utf-8")
            blob = (ctypes.c_ubyte * len(encoded)).from_buffer_copy(encoded)
            cred = _CREDENTIALW()
            cred.Flags = 0
            cred.Type = self.CRED_TYPE_GENERIC
            cred.TargetName = ctypes.c_wchar_p(self._target(key))
            cred.Comment = ctypes.c_wchar_p(APP_NAME)
            cred.CredentialBlobSize = len(encoded)
            cred.CredentialBlob = ctypes.cast(blob, ctypes.POINTER(ctypes.c_ubyte))
            cred.Persist = self.CRED_PERSIST_LOCAL_MACHINE
            cred.AttributeCount = 0
            cred.Attributes = None
            cred.TargetAlias = None
            cred.UserName = ctypes.c_wchar_p(APP_ID)
            if not _advapi32.CredWriteW(ctypes.byref(cred), 0):
                err = ctypes.get_last_error()
                raise CredentialError(
                    f"CredWriteW failed for {key!r} (error {err}). "
                    "Secrets are limited to 2560 UTF-16 characters."
                )

    def delete(self, key: str) -> None:
        with self._lock:
            target = self._target(key)
            if not _advapi32.CredDeleteW(target, self.CRED_TYPE_GENERIC, 0):
                err = ctypes.get_last_error()
                if err not in (self.ERROR_NOT_FOUND, 0):
                    raise CredentialError(f"CredDeleteW failed (error {err})")

    @staticmethod
    def available() -> bool:
        if os.name != "nt":
            return False
        try:
            store = WindowsCredentialStore()
            probe = "__availability_probe__"
            store.set(probe, "ok")
            value = store.get(probe)
            store.delete(probe)
            return value == "ok"
        except Exception:
            return False


# ---------------------------------------------------------------------------
# DPAPI-encrypted JSON fallback
# ---------------------------------------------------------------------------

if os.name == "nt":
    _crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)

    class _DATA_BLOB(ctypes.Structure):
        _fields_ = [
            ("cbData", wintypes.DWORD),
            ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
        ]

    _crypt32.CryptProtectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        wintypes.LPCWSTR,
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    _crypt32.CryptProtectData.restype = wintypes.BOOL
    _crypt32.CryptUnprotectData.argtypes = [
        ctypes.POINTER(_DATA_BLOB),
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(_DATA_BLOB),
        ctypes.c_void_p,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(_DATA_BLOB),
    ]
    _crypt32.CryptUnprotectData.restype = wintypes.BOOL
    _crypt32.CryptProtectData  # keep linters quiet


def _blob(data: bytes) -> tuple["_DATA_BLOB", ctypes.Array]:
    buf = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
    return _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte))), buf


def _protect(data: bytes) -> bytes:
    if os.name != "nt":
        # Not a Windows deployment; obfuscated storage is used and the README
        # makes clear that only Windows gets OS-level protection.
        return b"PLAIN:" + base64.b64encode(data)
    blob_in, _buf = _blob(data)
    blob_out = _DATA_BLOB()
    if not _crypt32.CryptProtectData(
        ctypes.byref(blob_in), APP_NAME, None, None, None, 0, ctypes.byref(blob_out)
    ):
        raise CredentialError(f"CryptProtectData failed (error {ctypes.get_last_error()})")
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        _kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


def _unprotect(data: bytes) -> bytes:
    if data.startswith(b"PLAIN:"):
        return base64.b64decode(data[6:])
    if os.name != "nt":
        raise CredentialError("encrypted credentials require Windows")
    blob_in, _buf = _blob(data)
    blob_out = _DATA_BLOB()
    description = wintypes.LPWSTR()
    if not _crypt32.CryptUnprotectData(
        ctypes.byref(blob_in),
        ctypes.byref(description),
        None,
        None,
        None,
        0,
        ctypes.byref(blob_out),
    ):
        raise CredentialError(
            f"CryptUnprotectData failed (error {ctypes.get_last_error()})"
        )
    try:
        return ctypes.string_at(blob_out.pbData, blob_out.cbData)
    finally:
        if description:
            _kernel32.LocalFree(ctypes.cast(description, ctypes.c_void_p))
        _kernel32.LocalFree(ctypes.cast(blob_out.pbData, ctypes.c_void_p))


class EncryptedFileCredentialStore:
    """DPAPI-protected secrets in a single file (used when Credential Manager fails)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.RLock()

    def _read(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        try:
            raw = _unprotect(self.path.read_bytes())
            data = json.loads(raw.decode("utf-8"))
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except (OSError, ValueError, UnicodeError, CredentialError):
            # A corrupt store should not prevent the app from starting.
            try:
                self.path.replace(self.path.with_suffix(".corrupt"))
            except OSError:
                pass
        return {}

    def _write(self, data: dict[str, str]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        protected = _protect(payload)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(protected)
        os.replace(tmp, self.path)

    def get(self, key: str) -> str | None:
        with self._lock:
            return self._read().get(key)

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._read()
            if value:
                data[key] = value
            else:
                data.pop(key, None)
            self._write(data)

    def delete(self, key: str) -> None:
        with self._lock:
            data = self._read()
            if key in data:
                del data[key]
                self._write(data)


def create_credential_store(fallback_path: Path | None = None) -> CredentialStore:
    """Return the best available secure store for this machine."""
    if os.name == "nt":
        try:
            store = WindowsCredentialStore()
            # Cheap probe; CredRead of a missing key is harmless.
            store.get("__probe__")
            return store
        except Exception:
            pass
    path = fallback_path or (Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_ID / "credentials.bin")
    return EncryptedFileCredentialStore(path)


def mask_secret(value: str | None, visible: int = 4) -> str:
    if not value:
        return ""
    if len(value) <= visible:
        return "•" * len(value)
    return "•" * max(4, len(value) - visible) + value[-visible:]
