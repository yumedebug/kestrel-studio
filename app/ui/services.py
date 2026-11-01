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
from app.core.performance import PerformanceMonitor


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
    performance: PerformanceMonitor = field(default_factory=PerformanceMonitor)
    bridge: InferenceBridge = field(default_factory=InferenceBridge)
    _agent_store: Any = field(default=None, repr=False)

    @property
    def agent_store(self):
        """Lazily opened SQLite store for tasks, approvals, grants and memories."""
        if self._agent_store is None:
            from app.agent.sessions import AgentStore
            from app.core import paths

            store = AgentStore(paths.agent_database_file())
            try:
                # Nothing that was interrupted may resume on its own.
                store.recover_interrupted()
            except Exception:
                pass
            self._agent_store = store
        return self._agent_store

    @property
    def agent_memory(self):
        from app.agent.memory import MemoryStore

        return MemoryStore(self.agent_store)

    @property
    def agent_permissions(self):
        from app.agent.permissions import PermissionManager

        return PermissionManager(self.agent_store, self.settings)

    def close_agent(self) -> None:
        if self._agent_store is not None:
            try:
                self._agent_store.close()
            except Exception:
                pass
            finally:
                self._agent_store = None

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
