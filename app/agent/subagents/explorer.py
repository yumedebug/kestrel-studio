"""Explorer sub-agent: investigates the environment and the project."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class ExplorerAgent(SubAgent):
    name = "explorer"
    role = "環境・ファイル調査"
    system_prompt = (
        "You explore a Windows machine and a codebase. Say exactly which files or "
        "directories to read and what to look for. Prefer read-only investigation."
    )
    tools = ("list_directory", "read_file", "search_files", "search_code", "inspect_project")


__all__ = ["ExplorerAgent"]
