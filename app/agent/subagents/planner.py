"""Planner sub-agent: turns a goal into an ordered plan."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class PlannerAgent(SubAgent):
    name = "planner"
    role = "作業計画"
    system_prompt = (
        "You are a senior engineer who plans work for a Windows desktop agent. "
        "Split the goal into 3-6 ordered, verifiable steps. Investigate before "
        "changing anything and finish with verification."
    )
    tools = ("inspect_project", "list_directory")


__all__ = ["PlannerAgent"]
