"""Application service container and Qt signal bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject, Signal

from app.api.server import ApiServer
from app.core.credentials import CredentialStore
from app.core.database import Database
from app.core.downloader import DownloadManager
from app.core.hf_client import HuggingFaceClient
from app.core.inference import InferenceManager
from app.core.model_manager import ModelManager
from app.core.settings import SettingsStore


class InferenceBridge(QObject):
    """Forwards manager events (from any thread) into the GUI thread."""

    event = Signal(object)

    def __init__(self) -> None:
        super().__init__()
        self._listener = self._on_event

    def attach(self, manager: InferenceManager) -> None:
        manager.add_listener(self._listener)

    def detach(self, manager: InferenceManager) -> None:
        manager.remove_listener(self._listener)

    def _on_event(self, event: dict[str, Any]) -> None:
        self.event.emit(event)


@dataclass
class AppServices:
    settings: SettingsStore
    credentials: CredentialStore
    database: Database
    hf: HuggingFaceClient
    library: ModelManager
    downloads: DownloadManager
    inference: InferenceManager
    api_server: ApiServer
    bridge: InferenceBridge = field(default_factory=InferenceBridge)

    def library_directory(self):
        from pathlib import Path

        return Path(self.settings.get("general.model_directory"))

    def cloud_configured(self, provider_id: str) -> bool:
        from app.core.providers import credential_key, get_provider

        spec = get_provider(provider_id)
        if spec is None:
            return False
        if not spec.key_required:
            return True
        try:
            return bool(self.credentials.get(credential_key(provider_id)))
        except Exception:
            return False
