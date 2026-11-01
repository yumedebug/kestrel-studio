"""Terminal tools: CMD, PowerShell, Python and process control.

Handlers execute ``context.approved_command`` — the exact string the permission
manager approved — so an approved command can never be swapped for another one
between approval and execution.
"""

from __future__ import annotations

from typing import Any

from app.agent.command_classifier import detect_shell
from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec

MAX_INLINE_OUTPUT = 12_000


def _command_for(shell: str, arguments: dict[str, Any], context: ToolContext) -> tuple[str, str, str]:
    command = str(arguments.get("command") or "").strip()
    cwd = str(arguments.get("cwd") or context.workspace or "")
    return command, shell, cwd


def _run(arguments: dict[str, Any], context: ToolContext, shell: str) -> dict[str, Any]:
    if context.process_manager is None:
        raise ToolError("No process manager is attached.")
    command = context.approved_command or str(arguments.get("command") or "").strip()
    if not command:
        raise ToolError("A command is required.")
    if command != context.approved_command and context.approved_command:
        raise ToolError("The command changed after approval; it must be approved again.")
    cwd = context.approved_cwd or str(arguments.get("cwd") or context.workspace or "")
    if cwd and not context.process_manager.validate_cwd(cwd):
        raise ToolError(f"The working directory does not exist: {cwd}")
    timeout = arguments.get("timeout")
    try:
        limit = float(timeout) if timeout else float(context.timeout or 120)
    except (TypeError, ValueError):
        limit = float(context.timeout or 120)

    result = context.process_manager.run(
        command,
        shell=shell,
        cwd=cwd,
        timeout=limit,
        on_output=context.emit_output,
    )
    record = None
    if context.save_command_history and context.store is not None:
        try:
            record = context.store.add_command(
                shell=shell,
                command=command,
                cwd=result.cwd,
                task_id=context.task_id,
                exit_code=result.exit_code,
                stdout=result.stdout[:100_000],
                stderr=result.stderr[:100_000],
                duration=result.duration,
                status=(
                    "cancelled"
                    if result.cancelled
                    else "timeout"
                    if result.timed_out
                    else "done"
                ),
                source=context.source,
            )
        except Exception:
            record = None
    stdout = result.stdout[-MAX_INLINE_OUTPUT:]
    stderr = result.stderr[-MAX_INLINE_OUTPUT:]
    detail = {
        "command": command,
        "shell": shell,
        "cwd": result.cwd,
        "exit_code": result.exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "duration": round(result.duration, 3),
        "timed_out": result.timed_out,
        "cancelled": result.cancelled,
        "command_id": record.id if record is not None else "",
    }
    summary_lines = [result.summary]
    if stdout.strip():
        summary_lines.append(f"stdout:\n{stdout.strip()}")
    if stderr.strip():
        summary_lines.append(f"stderr:\n{stderr.strip()}")
    return {"ok": result.ok, "summary": "\n".join(summary_lines), "data": detail}


def execute_cmd(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return _run(arguments, context, "cmd")


def execute_powershell(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return _run(arguments, context, "powershell")


def execute_python(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return _run(arguments, context, "python")


def run_approved(context: ToolContext, shell: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the command that was approved for this call (built from arguments)."""
    return _run(arguments or {}, context, shell)


def stop_process(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if context.process_manager is None:
        raise ToolError("No process manager is attached.")
    count = context.process_manager.stop_all()
    return {
        "ok": True,
        "summary": f"Stopped {count} running command(s).",
        "data": {"stopped": count},
    }


def get_command_output(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if context.store is None:
        raise ToolError("Command history is not available.")
    limit = max(1, min(50, int(arguments.get("limit") or 5)))
    records = context.store.list_commands(context.task_id or None, limit=limit)
    if not records:
        return {"ok": True, "summary": "No commands have been run yet.", "data": {"commands": []}}
    lines = []
    for record in records:
        lines.append(f"$ {record.command}  [{record.shell}] exit={record.exit_code}")
        if record.stdout.strip():
            lines.append(record.stdout.strip()[-4000:])
        if record.stderr.strip():
            lines.append("stderr: " + record.stderr.strip()[-2000:])
    return {
        "ok": True,
        "summary": "\n".join(lines)[-MAX_INLINE_OUTPUT:],
        "data": {
            "commands": [
                {
                    "id": record.id,
                    "command": record.command,
                    "shell": record.shell,
                    "exit_code": record.exit_code,
                    "status": record.status,
                    "created_at": record.created_at,
                }
                for record in records
            ]
        },
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="execute_cmd",
            description="Run a CMD (cmd.exe) command and return exit code, stdout and stderr.",
            kind="command",
            parameters={
                "command": {"type": "string", "required": True, "description": "Command line to run."},
                "cwd": {"type": "string", "description": "Working directory; defaults to the workspace."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
                "purpose": {"type": "string", "description": "Why the command is needed."},
            },
            command_builder=lambda args, ctx: _command_for("cmd", args, ctx),
        ),
        execute_cmd,
    )
    registry.register(
        ToolSpec(
            name="execute_powershell",
            description="Run a PowerShell command (PowerShell 7 when installed) and return its output.",
            kind="command",
            parameters={
                "command": {"type": "string", "required": True, "description": "PowerShell command line."},
                "cwd": {"type": "string", "description": "Working directory."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
                "purpose": {"type": "string", "description": "Why the command is needed."},
            },
            command_builder=lambda args, ctx: _command_for("powershell", args, ctx),
        ),
        execute_powershell,
    )
    registry.register(
        ToolSpec(
            name="execute_python",
            description="Run Python code (or a python/pip command line) with this app's interpreter.",
            kind="command",
            parameters={
                "command": {"type": "string", "required": True, "description": "Python source or a python CLI command."},
                "cwd": {"type": "string", "description": "Working directory."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
                "purpose": {"type": "string", "description": "Why the code is needed."},
            },
            command_builder=lambda args, ctx: _command_for("python", args, ctx),
        ),
        execute_python,
    )
    registry.register(
        ToolSpec(
            name="stop_process",
            description="Stop the running command(s) started by the agent.",
            kind="system",
            parameters={},
        ),
        stop_process,
    )
    registry.register(
        ToolSpec(
            name="get_command_output",
            description="Read the most recent command output recorded for this task.",
            kind="system",
            parameters={"limit": {"type": "integer", "description": "How many commands to show."}},
        ),
        get_command_output,
    )


__all__ = [
    "detect_shell",
    "execute_cmd",
    "execute_powershell",
    "execute_python",
    "get_command_output",
    "register",
    "stop_process",
]
