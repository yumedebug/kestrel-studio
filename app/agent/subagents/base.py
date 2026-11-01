"""Shared base class for Hybrid Agent sub-agents."""

from __future__ import annotations

from typing import Any

from app.agent.permissions import PermissionEscalationError, PermissionView


class SubAgent:
    """A specialist role that advises the main agent.

    Sub-agents never execute tools themselves: they analyse and return guidance
    which the main agent uses to choose its next validated tool call. They hold a
    read-only :class:`PermissionView`, so they cannot widen permissions.
    """

    name = "subagent"
    role = ""
    system_prompt = ""
    modes: tuple[str, ...] = ("plan", "build", "autonomous")
    tools: tuple[str, ...] = ()

    def __init__(self, planner: Any, permissions: PermissionView):
        if not isinstance(permissions, PermissionView):
            raise PermissionEscalationError(
                "Sub-agents must be given a read-only permission view."
            )
        self.planner = planner
        self.permissions = permissions

    # -- reasoning ---------------------------------------------------------
    @property
    def available(self) -> bool:
        try:
            return bool(self.planner is not None and self.planner.model.available)
        except Exception:
            return False

    def advise(self, goal: str, context_text: str, *, max_tokens: int = 400) -> str:
        if not self.available:
            return ""
        prompt = (
            f"Goal: {goal}\n\nContext:\n{context_text[:4000]}\n\n"
            "Reply with 1-3 short sentences of concrete guidance. No JSON, no code fences."
        )
        try:
            return self.planner.ask_role(self.system_prompt, prompt, max_tokens=max_tokens).strip()
        except Exception:
            return ""

    # -- permissions -------------------------------------------------------
    def evaluate(self, request: Any) -> Any:
        """Evaluate an action with the *same* permission manager as the main agent."""
        return self.permissions.evaluate(request)

    def escalate(self, *_args: Any, **_kwargs: Any) -> None:
        raise PermissionEscalationError(
            f"{self.name} cannot grant itself permissions."
        )


__all__ = ["SubAgent"]
