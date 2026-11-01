"""System tools: machine info, processes, installed software, environment.

All of these are read-only. Commands they run are fixed by the app (never chosen
by the model) and are executed through the process manager so no stray console
windows appear.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec

_SECRET_ENV_HINTS = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
    "auth",
    "session",
    "cookie",
)

_SAFE_ENV_KEYS = (
    "USERPROFILE",
    "USERNAME",
    "COMPUTERNAME",
    "OS",
    "PROCESSOR_ARCHITECTURE",
    "NUMBER_OF_PROCESSORS",
    "TEMP",
    "APPDATA",
    "LOCALAPPDATA",
    "PROGRAMFILES",
    "SYSTEMROOT",
    "SHELL",
    "PATH",
    "VIRTUAL_ENV",
    "CONDA_DEFAULT_ENV",
    "PYTHONPATH",
)


def get_system_info(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    uname = platform.uname()
    disks = []
    for root in ("C:\\", "/") if os.name == "nt" else ("/",):
        try:
            usage = shutil.disk_usage(root)
            disks.append(
                {
                    "root": root,
                    "total_gb": round(usage.total / 1e9, 1),
                    "free_gb": round(usage.free / 1e9, 1),
                }
            )
        except OSError:
            continue
    info = {
        "os": f"{uname.system} {uname.release}",
        "machine": uname.machine,
        "computer_name": uname.node,
        "user": os.environ.get("USERNAME") or os.environ.get("USER") or "",
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "disks": disks,
        "cwd": str(Path.cwd()),
        "workspace": context.workspace,
        "powershell": shutil.which("pwsh") or shutil.which("powershell") or "",
        "git": shutil.which("git") or "",
    }
    summary = (
        f"OS: {info['os']} ({info['machine']})\n"
        f"CPU cores: {info['cpu_count']}\n"
        f"Python: {info['python']}\n"
        f"PowerShell: {info['powershell'] or 'not found'}\n"
        f"Git: {info['git'] or 'not found'}\n"
        + "\n".join(f"Disk {d['root']}: {d['free_gb']} GB free of {d['total_gb']} GB" for d in disks)
    )
    try:
        from app.agent.process_manager import ProcessManager

        info["powershell_preferred"] = Path(ProcessManager.powershell_executable()).name
    except Exception:
        pass
    return {"ok": True, "summary": summary, "data": info}


def list_processes(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    limit = max(1, min(200, int(arguments.get("limit") or 40)))
    name_filter = str(arguments.get("name") or "").casefold()
    processes: list[dict[str, Any]] = []
    if context.process_manager is not None:
        result = context.process_manager.run(
            "tasklist /fo csv /nh", shell="cmd", timeout=30
        )
        for line in result.stdout.splitlines():
            parts = [part.strip().strip('"') for part in line.split('","')]
            if len(parts) < 2:
                continue
            name = parts[0].lstrip('"')
            if name_filter and name_filter not in name.casefold():
                continue
            processes.append({"name": name, "pid": parts[1], "memory": parts[4] if len(parts) > 4 else ""})
            if len(processes) >= limit:
                break
        source = "tasklist"
    else:  # pragma: no cover - defensive
        raise ToolError("No process manager is attached.")
    summary = "\n".join(f"{p['name']} (pid {p['pid']}) {p['memory']}" for p in processes[:limit])
    return {
        "ok": True,
        "summary": f"{len(processes)} process(es) via {source}\n{summary}",
        "data": {"processes": processes, "source": source},
    }


def check_installed_software(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    name = str(arguments.get("name") or "").strip()
    if not name:
        raise ToolError("name is required.")
    needle = name.casefold()
    matches: list[dict[str, str]] = []
    which = shutil.which(name)
    if which:
        matches.append({"name": name, "location": which, "source": "PATH"})
    if os.name == "nt":
        try:
            import winreg  # type: ignore[import-not-found]

            roots = (
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            )
            for hive, path in roots:
                try:
                    key = winreg.OpenKey(hive, path)
                except OSError:
                    continue
                with key:
                    index = 0
                    while True:
                        try:
                            sub_name = winreg.EnumKey(key, index)
                        except OSError:
                            break
                        index += 1
                        try:
                            with winreg.OpenKey(key, sub_name) as sub:
                                display, _ = winreg.QueryValueEx(sub, "DisplayName")
                                if not isinstance(display, str) or needle not in display.casefold():
                                    continue
                                try:
                                    version, _ = winreg.QueryValueEx(sub, "DisplayVersion")
                                except OSError:
                                    version = ""
                                matches.append(
                                    {
                                        "name": display,
                                        "location": str(version),
                                        "source": "registry",
                                    }
                                )
                        except OSError:
                            continue
                        if len(matches) >= 40:
                            break
                if len(matches) >= 40:
                    break
        except Exception:
            pass
    summary = (
        "\n".join(f"{m['name']} — {m['location']} ({m['source']})" for m in matches)
        if matches
        else f"{name} was not found on PATH or in the installed-programs list."
    )
    return {
        "ok": True,
        "summary": summary,
        "data": {"name": name, "matches": matches, "found": bool(matches)},
    }


def get_environment_info(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    values: dict[str, str] = {}
    for key in _SAFE_ENV_KEYS:
        value = os.environ.get(key)
        if value:
            values[key] = value
    prefix = str(arguments.get("prefix") or "").strip().casefold()
    if prefix:
        for key, value in os.environ.items():
            if not key.casefold().startswith(prefix):
                continue
            if any(hint in key.casefold() for hint in _SECRET_ENV_HINTS):
                continue
            values.setdefault(key, value)
    for key in list(values):
        if any(hint in key.casefold() for hint in _SECRET_ENV_HINTS):
            values.pop(key, None)
    if "PATH" in values:
        parts = values["PATH"].split(os.pathsep)
        values["PATH"] = os.pathsep.join(parts[:12]) + (" …" if len(parts) > 12 else "")
    summary = "\n".join(f"{key}={value}" for key, value in sorted(values.items()))
    return {
        "ok": True,
        "summary": summary or "No environment variables matched.",
        "data": {"environment": values, "hidden_for_privacy": list(_SECRET_ENV_HINTS)},
    }


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="get_system_info",
            description="Read OS, CPU, disk, Python, PowerShell and Git information.",
            kind="system",
            parameters={},
        ),
        get_system_info,
    )
    registry.register(
        ToolSpec(
            name="list_processes",
            description="List running processes, optionally filtered by name.",
            kind="system",
            parameters={
                "name": {"type": "string", "description": "Only processes whose name contains this text."},
                "limit": {"type": "integer", "description": "Maximum rows (default 40)."},
            },
        ),
        list_processes,
    )
    registry.register(
        ToolSpec(
            name="check_installed_software",
            description="Check whether a program is installed (PATH and uninstall registry keys).",
            kind="system",
            parameters={"name": {"type": "string", "required": True, "description": "Program name."}},
        ),
        check_installed_software,
    )
    registry.register(
        ToolSpec(
            name="get_environment_info",
            description="Read safe environment variables (credentials are filtered out).",
            kind="system",
            parameters={"prefix": {"type": "string", "description": "Only variables starting with this prefix."}},
        ),
        get_environment_info,
    )


__all__ = [
    "check_installed_software",
    "get_environment_info",
    "get_system_info",
    "list_processes",
    "register",
]
