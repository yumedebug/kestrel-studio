"""Terminal sub-agent: plans shell commands and interprets their output."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class TerminalAgent(SubAgent):
    name = "terminal_agent"
    role = "コマンド操作"
    system_prompt = (
        "You are a Windows shell expert (CMD, PowerShell 7, Python, git). Suggest the "
        "smallest safe command that answers the question, and explain what its output means."
    )
    tools = ("execute_cmd", "execute_powershell", "execute_python", "get_command_output")


__all__ = ["TerminalAgent"]
