"""Tool registry shared by every agent and sub-agent.

The agent never touches the OS directly: it can only name a registered tool.
The executor performs the permission check *before* a handler runs, and the
handler can only act on the command/paths that were approved.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

Handler = Callable[[dict[str, Any], "ToolContext"], dict[str, Any]]
CommandBuilder = Callable[[dict[str, Any], "ToolContext"], tuple[str, str, str]]

FILE_KINDS = {"read", "write", "create_dir", "move", "delete"}
ALL_KINDS = FILE_KINDS | {"command", "system", "memory"}


class ToolError(RuntimeError):
    """A tool failed in a way the agent can explain to the user."""


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    kind: str
    parameters: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: Modes in which the tool may be called at all.
    modes: tuple[str, ...] = ("plan", "build", "autonomous")
    command_builder: CommandBuilder | None = None
    requires_admin: bool = False

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                key: {
                    "type": value.get("type", "string"),
                    "required": bool(value.get("required", False)),
                    "description": value.get("description", ""),
                }
                for key, value in self.parameters.items()
            },
        }

    def required_arguments(self) -> tuple[str, ...]:
        return tuple(
            name for name, spec in self.parameters.items() if spec.get("required")
        )


@dataclass
class ToolContext:
    """Everything a tool handler is allowed to use for one call."""

    workspace: str = ""
    task_id: str = ""
    mode: str = "autonomous"
    full_access: bool = False
    source: str = "agent"
    store: Any = None
    memory: Any = None
    process_manager: Any = None
    emit: Callable[[str, str], None] | None = None
    timeout: float = 120.0
    save_command_history: bool = True
    approved_command: str = ""
    approved_shell: str = ""
    approved_cwd: str = ""

    def emit_output(self, stream: str, text: str) -> None:
        if self.emit is not None:
            try:
                self.emit(stream, text)
            except Exception:
                pass


@dataclass
class ToolCall:
    tool: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    ok: bool
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    denied: bool = False
    rejected: bool = False
    stopped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def observation(self) -> str:
        if self.summary:
            return self.summary
        if self.error:
            return self.error
        return "ok" if self.ok else "failed"


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Handler] = {}

    # -- registration ------------------------------------------------------
    def register(self, spec: ToolSpec, handler: Handler) -> None:
        if spec.kind not in ALL_KINDS:
            raise ValueError(f"Unknown tool kind: {spec.kind}")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec | None:
        return self._specs.get(name)

    def has(self, name: str) -> bool:
        return name in self._specs

    def specs(self) -> list[ToolSpec]:
        return list(self._specs.values())

    def names(self) -> list[str]:
        return list(self._specs)

    def describe_for_prompt(self, mode: str = "autonomous") -> str:
        tools = [
            spec.describe()
            for spec in self._specs.values()
            if mode in spec.modes
            and (mode != "plan" or spec.kind in {"read", "system", "memory"})
        ]
        return json.dumps(tools, ensure_ascii=False, indent=2)

    # -- execution ---------------------------------------------------------
    def build_command(
        self, spec: ToolSpec, arguments: dict[str, Any], context: ToolContext
    ) -> tuple[str, str, str]:
        if spec.command_builder is None:
            return (
                str(arguments.get("command") or "").strip(),
                str(arguments.get("shell") or "cmd"),
                str(arguments.get("cwd") or context.workspace or ""),
            )
        return spec.command_builder(arguments, context)

    def call(self, call: ToolCall, context: ToolContext) -> dict[str, Any]:
        spec = self._specs.get(call.tool)
        handler = self._handlers.get(call.tool)
        if spec is None or handler is None:
            return {"ok": False, "error": f"Unknown tool: {call.tool}"}
        try:
            result = handler(dict(call.arguments or {}), context)
        except ToolError as exc:
            return {"ok": False, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - reported to the agent
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        if not isinstance(result, dict):
            return {"ok": True, "summary": str(result)}
        result.setdefault("ok", True)
        return result


__all__ = [
    "FILE_KINDS",
    "ToolCall",
    "ToolContext",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "ToolSpec",
]
