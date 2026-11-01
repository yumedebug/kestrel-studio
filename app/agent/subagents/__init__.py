"""Sub-agent registry.

Sub-agents share the main agent's permission manager through a read-only view,
so they can evaluate actions but never grant, widen or revoke permissions.
"""

from __future__ import annotations

from typing import Any

from app.agent.subagents.base import SubAgent
from app.agent.subagents.coder import CoderAgent
from app.agent.subagents.debugger import DebuggerAgent
from app.agent.subagents.explorer import ExplorerAgent
from app.agent.subagents.planner import PlannerAgent
from app.agent.subagents.reviewer import ReviewerAgent
from app.agent.subagents.terminal_agent import TerminalAgent
from app.agent.subagents.validator import ValidatorAgent

SUBAGENT_CLASSES = (
    PlannerAgent,
    ExplorerAgent,
    CoderAgent,
    TerminalAgent,
    ReviewerAgent,
    DebuggerAgent,
    ValidatorAgent,
)

SUBAGENT_NAMES = tuple(cls.name for cls in SUBAGENT_CLASSES)


def build_subagents(planner: Any, permissions: Any) -> dict[str, SubAgent]:
    """Instantiate every sub-agent with a read-only permission view."""
    view = permissions.view() if hasattr(permissions, "view") else permissions
    return {cls.name: cls(planner, view) for cls in SUBAGENT_CLASSES}


def subagent_for_tool(subagents: dict[str, SubAgent], tool: str) -> SubAgent | None:
    for agent in subagents.values():
        if tool in agent.tools:
            return agent
    return None


__all__ = [
    "SUBAGENT_CLASSES",
    "SUBAGENT_NAMES",
    "SubAgent",
    "build_subagents",
    "subagent_for_tool",
]
