"""Agent page package."""

from __future__ import annotations

__all__ = ["AgentPage"]


def __getattr__(name: str):
    if name == "AgentPage":
        from app.ui.agent.agent_page import AgentPage

        return AgentPage
    raise AttributeError(name)
