"""The single gate between the agent and the operating system.

Every tool call goes through :class:`ToolExecutor`:

1. the tool name and arguments are validated against the registry schema,
2. the action is turned into a permission request and evaluated,
3. ``ask`` decisions pause for the approval dialog,
4. only then does the handler run — using the *approved* command string.

Sub-agents call :meth:`ToolExecutor.execute` too; they simply pass their
read-only `PermissionView` so they cannot grant themselves anything.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from app.agent.permissions import (
    ActionRequest,
    ApprovalOutcome,
    ApprovalRequest,
    PermissionDecision,
    PermissionManager,
)
from app.agent.tools.registry import (
    FILE_KINDS,
    ToolCall,
    ToolContext,
    ToolError,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from app.agent.validator import ToolValidationError, validate_arguments

_PATH_ARGUMENT_NAMES = ("path", "source", "destination", "root")


class ToolExecutor:
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        permissions: Any,
        store: Any = None,
        memory: Any = None,
        process_manager: Any = None,
        settings: Any = None,
    ):
        self.registry = registry
        self.permissions = permissions
        self.store = store
        self.memory = memory
        self.process_manager = process_manager
        self.settings = settings

    # -- request construction ---------------------------------------------
    def _absolute(self, value: str, workspace: str) -> str:
        raw = str(value or "").strip()
        if not raw:
            return ""
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = Path(workspace or os.getcwd()) / path
        try:
            return str(path.resolve())
        except (OSError, RuntimeError):
            return str(path)

    def _paths_for(
        self, spec: ToolSpec, arguments: dict[str, Any], context: ToolContext
    ) -> tuple[str, ...]:
        if spec.kind not in FILE_KINDS:
            return ()
        workspace = context.workspace or os.getcwd()
        found: list[str] = []
        for name in _PATH_ARGUMENT_NAMES:
            if name not in arguments:
                continue
            value = arguments.get(name)
            if not isinstance(value, str) or not value.strip():
                continue
            resolved = self._absolute(value, workspace)
            if resolved and resolved not in found:
                found.append(resolved)
        if not found and spec.kind == "read":
            found.append(self._absolute(workspace, workspace))
        return tuple(found)

    def _prepare(
        self, spec: ToolSpec, arguments: dict[str, Any], context: ToolContext
    ) -> tuple[ActionRequest, tuple[str, str, str]]:
        workspace = context.workspace or os.getcwd()
        if spec.kind == "command":
            command, shell, cwd = self.registry.build_command(spec, arguments, context)
            command = str(command or "").strip()
            if not command:
                raise ToolError("No command was produced for this tool call.")
            cwd = str(cwd or workspace)
            request = ActionRequest(
                kind="command",
                description=spec.description,
                command=command,
                shell=str(shell or "cmd"),
                cwd=cwd,
                paths=(),
                purpose=str(arguments.get("purpose") or spec.name),
                target=f"{shell}: {command}",
                task_id=context.task_id,
                source=context.source,
                mode=context.mode,
            )
            return request, (command, str(shell or "cmd"), cwd)

        if spec.kind in FILE_KINDS:
            paths = self._paths_for(spec, arguments, context)
            request = ActionRequest(
                kind=spec.kind,
                description=spec.description,
                cwd=workspace,
                paths=paths,
                purpose=str(arguments.get("purpose") or spec.name),
                target=", ".join(paths),
                task_id=context.task_id,
                source=context.source,
                mode=context.mode,
                recursive=bool(arguments.get("recursive", False)),
            )
            return request, ("", "", "")

        request = ActionRequest(
            kind=spec.kind,
            description=spec.description,
            cwd=workspace,
            purpose=spec.name,
            target=spec.name,
            task_id=context.task_id,
            source=context.source,
            mode=context.mode,
        )
        return request, ("", "", "")

    # -- execution ---------------------------------------------------------
    def execute(
        self,
        call: ToolCall,
        context: ToolContext,
        *,
        broker: Any = None,
        mode: str = "",
    ) -> ToolResult:
        mode = mode or context.mode or "autonomous"
        spec = self.registry.get(call.tool)
        if spec is None:
            return ToolResult(
                False,
                f"Unknown tool: {call.tool}",
                error=f"Unknown tool: {call.tool}",
                denied=True,
            )
        if mode not in spec.modes:
            return ToolResult(
                False,
                f"{call.tool} is not available in {mode} mode.",
                error=f"{call.tool} is not available in {mode} mode.",
                denied=True,
            )
        # Plan mode is strictly read-only and chat mode runs no tools at all.
        if mode == "chat" or (
            mode == "plan" and spec.kind not in {"read", "system", "memory"}
        ):
            return ToolResult(
                False,
                f"{call.tool} cannot run in {mode} mode.",
                error=f"{call.tool} cannot run in {mode} mode.",
                denied=True,
            )
        try:
            arguments = validate_arguments(spec, call.arguments)
        except ToolValidationError as exc:
            return ToolResult(False, "", error=str(exc))

        try:
            request, approved = self._prepare(spec, arguments, context)
        except ToolError as exc:
            return ToolResult(False, "", error=str(exc))

        decision = self.permissions.evaluate(request)
        if decision.decision == "deny":
            return ToolResult(False, decision.reason, error=decision.reason, denied=True)

        if decision.decision == "ask":
            outcome = self._ask(decision, broker)
            if outcome is None:
                return ToolResult(
                    False,
                    "承認が必要ですが、承認ダイアログを表示できませんでした。",
                    denied=True,
                )
            if outcome.stop_task:
                return ToolResult(False, "ユーザーがタスクを停止しました。", stopped=True)
            if not outcome.allowed:
                return ToolResult(False, "ユーザーがこの操作を拒否しました。", rejected=True)
            if outcome.choice == "always_allow":
                try:
                    self.permissions.add_grant(decision, origin="user")
                except Exception:
                    # A grant that the policy refuses is simply not stored.
                    pass

        tool_context = replace(
            context,
            mode=mode,
            approved_command=approved[0],
            approved_shell=approved[1],
            approved_cwd=approved[2],
            store=self.store,
            memory=self.memory,
            process_manager=self.process_manager,
        )
        raw = self.registry.call(call, tool_context)
        result = self._to_result(raw)

        executed = str(result.data.get("command") or "")
        if approved[0] and executed and executed != approved[0]:
            result = ToolResult(
                False,
                "承認された内容と実行内容が一致しないため中止しました。",
                error="The executed command did not match the approved command.",
                denied=True,
            )
        if decision.requires_admin:
            result.data.setdefault("requires_admin", True)
        if decision.grant_id:
            result.data.setdefault("grant_id", decision.grant_id)
        return result

    def _ask(self, decision: PermissionDecision, broker: Any) -> ApprovalOutcome | None:
        if broker is None:
            return None
        request = ApprovalRequest(decision)
        outcome = broker.request(request)
        if outcome.fingerprint and outcome.fingerprint != decision.fingerprint:
            # The dialog answered about a different action; never reuse it.
            outcome = broker.request(request)
        try:
            self.permissions.record_approval(decision, outcome)
        except Exception:
            pass
        return outcome

    @staticmethod
    def _to_result(raw: dict[str, Any]) -> ToolResult:
        ok = bool(raw.get("ok", True))
        summary = str(raw.get("summary") or "")
        error = str(raw.get("error") or "")
        data = raw.get("data")
        if not isinstance(data, dict):
            data = {
                key: value
                for key, value in raw.items()
                if key not in {"ok", "summary", "error", "data"}
            }
        return ToolResult(
            ok=ok,
            summary=summary or (error if not ok else ""),
            data=data,
            error=error,
        )


__all__ = ["ToolExecutor"]
