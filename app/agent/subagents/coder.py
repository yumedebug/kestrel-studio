"""Coder sub-agent: writes and edits code."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class CoderAgent(SubAgent):
    name = "coder"
    role = "コード生成・編集"
    system_prompt = (
        "You are a careful programmer. Keep edits minimal and consistent with the "
        "existing style, preserve behaviour that is unrelated to the goal, and "
        "mention which file and function to change."
    )
    tools = ("write_file", "edit_file", "apply_patch", "create_directory", "move_file", "copy_file")


__all__ = ["CoderAgent"]
