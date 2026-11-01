"""Tool package: builds the registry used by the agent controller."""

from __future__ import annotations

from app.agent.tools.registry import (
    ToolCall,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)


def create_registry() -> ToolRegistry:
    """Register every built-in tool exactly once."""
    from app.agent.tools import development, filesystem, memory_tools, system, terminal

    registry = ToolRegistry()
    filesystem.register(registry)
    development.register(registry)
    terminal.register(registry)
    system.register(registry)
    memory_tools.register(registry)
    return registry


__all__ = [
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
    "create_registry",
]
