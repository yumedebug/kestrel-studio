"""Validator sub-agent: checks whether the goal was actually reached."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class ValidatorAgent(SubAgent):
    name = "validator"
    role = "結果検証"
    system_prompt = (
        "You verify agent results. Given the goal and the actions taken, state whether "
        "the goal is met, which evidence supports that, and what is still missing. "
        "Be conservative: if evidence is missing, say so."
    )
    tools = ("run_tests", "read_file", "get_command_output", "git_diff")


__all__ = ["ValidatorAgent"]
