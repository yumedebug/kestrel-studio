"""Debugger sub-agent: analyses failures and proposes a next attempt."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class DebuggerAgent(SubAgent):
    name = "debugger"
    role = "エラー解析"
    system_prompt = (
        "You diagnose failures from command output and stack traces. State the most "
        "likely cause and one concrete next action. Never suggest repeating an "
        "identical failing command."
    )
    tools = ("execute_cmd", "execute_powershell", "execute_python", "read_file", "search_code", "run_tests")


__all__ = ["DebuggerAgent"]
