"""Reviewer sub-agent: reviews code and test results."""

from __future__ import annotations

from app.agent.subagents.base import SubAgent


class ReviewerAgent(SubAgent):
    name = "reviewer"
    role = "コードレビュー"
    system_prompt = (
        "You review code changes and test output. Point out concrete problems "
        "(bugs, missing error handling, broken behaviour). Be brief and specific."
    )
    tools = ("git_status", "git_diff", "run_tests", "read_file", "search_code")


__all__ = ["ReviewerAgent"]
