"""Development tools: project inspection, code search, patching, tests and git."""

from __future__ import annotations

import json
import re
import shlex
from pathlib import Path
from typing import Any

from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec
from app.agent.tools.terminal import run_approved

MANIFESTS = (
    ("package.json", "Node/JavaScript"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("setup.cfg", "Python"),
    ("requirements.txt", "Python"),
    ("Pipfile", "Python"),
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("pom.xml", "Java/Maven"),
    ("build.gradle", "Java/Gradle"),
    ("CMakeLists.txt", "C/C++ (CMake)"),
    ("Makefile", "Make"),
    ("kestrel.spec", "PyInstaller"),
    ("tsconfig.json", "TypeScript"),
)

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".mypy_cache"}


def _workspace(context: ToolContext) -> Path:
    return Path(context.workspace or ".").expanduser()


def _resolve(context: ToolContext, value: Any) -> Path:
    raw = str(value or "").strip()
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = _workspace(context) / path
    return path


def _relative(context: ToolContext, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(_workspace(context).resolve()))
    except (ValueError, OSError):
        return str(path)


# -- inspection ------------------------------------------------------------
def inspect_project(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    root = _resolve(context, arguments.get("path") or context.workspace or ".")
    if not root.is_dir():
        raise ToolError(f"Project directory not found: {root}")
    stacks: list[str] = []
    manifests: list[str] = []
    for name, stack in MANIFESTS:
        if (root / name).exists():
            manifests.append(name)
            if stack not in stacks:
                stacks.append(stack)

    directories: list[str] = []
    source_files: list[str] = []
    top_level: list[str] = []
    for child in sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())):
        if child.name in SKIP_DIRS:
            continue
        top_level.append(child.name + ("/" if child.is_dir() else ""))
        if child.is_dir():
            directories.append(child.name)

    extensions: dict[str, int] = {}
    count = 0
    for directory, dirnames, filenames in __import__("os").walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            suffix = Path(filename).suffix.casefold() or "<none>"
            extensions[suffix] = extensions.get(suffix, 0) + 1
            count += 1
            if count <= 40 and suffix in {
                ".py",
                ".js",
                ".ts",
                ".tsx",
                ".go",
                ".rs",
                ".java",
                ".cs",
                ".cpp",
                ".c",
                ".h",
                ".json",
                ".toml",
                ".yaml",
                ".yml",
            }:
                source_files.append(_relative(context, Path(directory) / filename))

    git_dir = root / ".git"
    hint = ""
    if "pyproject.toml" in manifests or "requirements.txt" in manifests:
        hint = "python -m pytest -q"
    elif "package.json" in manifests:
        hint = "npm test"
    elif "Cargo.toml" in manifests:
        hint = "cargo test"
    elif "go.mod" in manifests:
        hint = "go test ./..."

    report = {
        "root": str(root),
        "stacks": stacks,
        "manifests": manifests,
        "top_level": top_level,
        "directories": directories[:20],
        "file_count": count,
        "extensions": dict(sorted(extensions.items(), key=lambda kv: -kv[1])[:12]),
        "source_files": source_files[:25],
        "git": git_dir.exists(),
        "suggested_test_command": hint,
    }
    summary = (
        f"Project at {root}\n"
        f"Stacks: {', '.join(stacks) or 'unknown'}\n"
        f"Manifests: {', '.join(manifests) or 'none'}\n"
        f"Files: {count}\n"
        f"Top level: {', '.join(top_level[:20])}\n"
        f"Git repository: {'yes' if git_dir.exists() else 'no'}"
    )
    return {"ok": True, "summary": summary, "data": report}


def search_code(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query:
        raise ToolError("query is required.")
    root = _resolve(context, arguments.get("root") or context.workspace or ".")
    glob = str(arguments.get("glob") or "")
    use_regex = bool(arguments.get("regex", False))
    limit = max(1, min(200, int(arguments.get("max_results") or 40)))
    try:
        pattern = re.compile(query if use_regex else re.escape(query), re.IGNORECASE)
    except re.error as exc:
        raise ToolError(f"Invalid regular expression: {exc}") from exc

    matches: list[dict[str, Any]] = []
    import os

    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            if glob and not Path(filename).match(glob):
                continue
            full = Path(directory) / filename
            try:
                if full.stat().st_size > 400_000:
                    continue
                text = full.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if pattern.search(line):
                    matches.append(
                        {
                            "path": _relative(context, full),
                            "line": number,
                            "text": line.strip()[:300],
                        }
                    )
                    if len(matches) >= limit:
                        break
            if len(matches) >= limit:
                break
        if len(matches) >= limit:
            break
    listing = "\n".join(f"{m['path']}:{m['line']}: {m['text']}" for m in matches)
    return {
        "ok": True,
        "summary": f"{len(matches)} match(es)\n{listing}" if matches else "No matches.",
        "data": {"query": query, "matches": matches},
    }


def apply_patch(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    hunks = arguments.get("hunks")
    if not path.is_file():
        raise ToolError(f"File not found: {path}")
    if not isinstance(hunks, list) or not hunks:
        raise ToolError("hunks must be a non-empty list of {old, new} objects.")
    original = path.read_text(encoding="utf-8", errors="replace")
    working = original
    applied = 0
    for index, hunk in enumerate(hunks, 1):
        if not isinstance(hunk, dict):
            raise ToolError(f"Hunk {index} is not an object.")
        old = hunk.get("old")
        new = hunk.get("new")
        if not isinstance(old, str) or not isinstance(new, str) or not old:
            raise ToolError(f"Hunk {index} needs non-empty 'old' and a string 'new'.")
        occurrences = working.count(old)
        if occurrences == 0:
            raise ToolError(
                f"Hunk {index} was not found; nothing was written. Re-read the file and retry."
            )
        if occurrences > 1:
            raise ToolError(
                f"Hunk {index} matches {occurrences} places; add more context. Nothing was written."
            )
        working = working.replace(old, new, 1)
        applied += 1
    try:
        path.write_text(working, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ToolError(f"Could not write the file: {exc}") from exc
    return {
        "ok": True,
        "summary": f"Applied {applied} hunk(s) to {_relative(context, path)}",
        "data": {
            "path": str(path),
            "hunks": applied,
            "change": {"action": "patch", "path": str(path), "relative": _relative(context, path)},
        },
    }


# -- test + git command builders -------------------------------------------
def _test_command(arguments: dict[str, Any], context: ToolContext) -> tuple[str, str, str]:
    explicit = str(arguments.get("command") or "").strip()
    if explicit:
        return explicit, "cmd", str(arguments.get("cwd") or context.workspace or "")
    root = _workspace(context)
    if (root / "package.json").exists():
        return "npm test", "cmd", str(root)
    if (root / "Cargo.toml").exists():
        return "cargo test", "cmd", str(root)
    if (root / "go.mod").exists():
        return "go test ./...", "cmd", str(root)
    return "python -m pytest -q", "cmd", str(root)


def run_tests(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return run_approved(context, "cmd")


def _git_command(subcommand: str):
    def build(arguments: dict[str, Any], context: ToolContext) -> tuple[str, str, str]:
        command = f"git {subcommand}"
        if subcommand == "status":
            command = "git status --short --branch"
        elif subcommand == "diff":
            staged = bool(arguments.get("staged", False))
            command = "git diff --staged" if staged else "git diff"
        elif subcommand == "commit":
            message = str(arguments.get("message") or "").strip()
            if not message:
                raise ToolError("A commit message is required.")
            safe = message.replace('"', '\\"').replace("\n", " ")
            command = f'git commit -m "{safe}"'
        return command, "git", str(arguments.get("cwd") or context.workspace or "")

    return build


def _git_run(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    return run_approved(context, "git")


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="inspect_project",
            description="Summarise a project: stacks, manifests, top-level layout, file types, git.",
            kind="read",
            parameters={"path": {"type": "string", "description": "Project root; defaults to the workspace."}},
        ),
        inspect_project,
    )
    registry.register(
        ToolSpec(
            name="search_code",
            description="Search source code with a literal string or regular expression.",
            kind="read",
            parameters={
                "query": {"type": "string", "required": True, "description": "Text or regex to find."},
                "regex": {"type": "boolean", "description": "Treat query as a regular expression."},
                "glob": {"type": "string", "description": "Filename glob such as *.py."},
                "root": {"type": "string", "description": "Directory to search."},
                "max_results": {"type": "integer", "description": "Maximum matches."},
            },
        ),
        search_code,
    )
    registry.register(
        ToolSpec(
            name="apply_patch",
            description="Apply one or more exact search/replace hunks to a file, atomically.",
            kind="write",
            parameters={
                "path": {"type": "string", "required": True, "description": "File to patch."},
                "hunks": {
                    "type": "array",
                    "required": True,
                    "description": 'List of {"old": "...", "new": "..."} replacements.',
                },
            },
            modes=("build", "autonomous"),
        ),
        apply_patch,
    )
    registry.register(
        ToolSpec(
            name="run_tests",
            description="Run the project's test command (or an explicit one) and report the result.",
            kind="command",
            parameters={
                "command": {"type": "string", "description": "Explicit test command; auto-detected when omitted."},
                "cwd": {"type": "string", "description": "Working directory."},
                "timeout": {"type": "integer", "description": "Timeout in seconds."},
            },
            command_builder=_test_command,
        ),
        run_tests,
    )
    registry.register(
        ToolSpec(
            name="git_status",
            description="Show the repository status (read-only).",
            kind="command",
            parameters={"cwd": {"type": "string", "description": "Repository directory."}},
            command_builder=_git_command("status"),
        ),
        _git_run,
    )
    registry.register(
        ToolSpec(
            name="git_diff",
            description="Show unstaged or staged changes (read-only).",
            kind="command",
            parameters={
                "staged": {"type": "boolean", "description": "Show staged changes instead."},
                "cwd": {"type": "string", "description": "Repository directory."},
            },
            command_builder=_git_command("diff"),
        ),
        _git_run,
    )
    registry.register(
        ToolSpec(
            name="git_commit",
            description="Commit staged changes with a message (needs approval).",
            kind="command",
            parameters={
                "message": {"type": "string", "required": True, "description": "Commit message."},
                "cwd": {"type": "string", "description": "Repository directory."},
            },
            command_builder=_git_command("commit"),
        ),
        _git_run,
    )


__all__ = [
    "apply_patch",
    "inspect_project",
    "register",
    "run_tests",
    "search_code",
]
