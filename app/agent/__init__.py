"""Hybrid Agent: a semi-autonomous, permission-gated agent for Windows.

The package is Qt-free. The GUI talks to :class:`app.agent.controller.AgentController`
through an event listener, so the whole agent core can be tested headlessly.
"""

from __future__ import annotations

__all__ = [
    "ALLOW",
    "ASK",
    "DENY",
    "AgentController",
    "PermissionManager",
    "ProcessManager",
    "ToolExecutor",
]


def __getattr__(name: str):  # pragma: no cover - lazy re-exports
    if name in {"ALLOW", "ASK", "DENY"}:
        from app.agent import command_classifier

        return getattr(command_classifier, name)
    if name == "AgentController":
        from app.agent.controller import AgentController

        return AgentController
    if name == "PermissionManager":
        from app.agent.permissions import PermissionManager

        return PermissionManager
    if name == "ProcessManager":
        from app.agent.process_manager import ProcessManager

        return ProcessManager
    if name == "ToolExecutor":
        from app.agent.executor import ToolExecutor

        return ToolExecutor
    raise AttributeError(name)
