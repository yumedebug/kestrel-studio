"""File tools: read, write, edit, list, search, create, move, copy, delete.

Every handler returns a ``change`` record for the File Changes panel. Large
read/write payloads are capped so a small model cannot flood its own context.
"""

from __future__ import annotations

import fnmatch
import os
import shutil
from pathlib import Path
from typing import Any

from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec

MAX_READ_BYTES = 200_000
MAX_WRITE_BYTES = 2_000_000
MAX_SEARCH_RESULTS = 200
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}


def _resolve(context: ToolContext, value: Any) -> Path:
    raw = str(value or "").strip().strip('"')
    if not raw:
        raise ToolError("A path is required.")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        base = context.workspace or os.getcwd()
        path = Path(base) / path
    return path


def _relative(context: ToolContext, path: Path) -> str:
    try:
        base = Path(context.workspace or os.getcwd())
        return str(path.resolve().relative_to(base.resolve()))
    except (ValueError, OSError):
        return str(path)


def _read_text(path: Path) -> str:
    if not path.exists():
        raise ToolError(f"File not found: {path}")
    if path.is_dir():
        raise ToolError(f"That path is a directory: {path}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise ToolError(f"Could not stat the file: {exc}") from exc
    if size > MAX_READ_BYTES:
        raise ToolError(
            f"The file is too large to read ({size} bytes > {MAX_READ_BYTES}). "
            "Read a slice with start_line/max_lines instead."
        )
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise ToolError(f"Could not read the file: {exc}") from exc


def _change(action: str, context: ToolContext, path: Path, **extra: Any) -> dict[str, Any]:
    return {"action": action, "path": str(path), "relative": _relative(context, path), **extra}


# -- handlers --------------------------------------------------------------
def read_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    text = _read_text(path)
    lines = text.splitlines()
    start = max(1, int(arguments.get("start_line") or 1))
    max_lines = int(arguments.get("max_lines") or 400)
    chunk = lines[start - 1 : start - 1 + max_lines]
    numbered = "\n".join(f"{start + i:>5} | {line}" for i, line in enumerate(chunk))
    truncated = start - 1 + len(chunk) < len(lines)
    return {
        "ok": True,
        "summary": f"Read {len(chunk)} line(s) from {_relative(context, path)}"
        + (" (truncated)" if truncated else ""),
        "data": {
            "path": str(path),
            "relative": _relative(context, path),
            "content": numbered,
            "total_lines": len(lines),
            "start_line": start,
            "truncated": truncated,
        },
    }


def write_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    content = arguments.get("content")
    if content is None:
        raise ToolError("content is required.")
    text = str(content)
    if len(text.encode("utf-8", "replace")) > MAX_WRITE_BYTES:
        raise ToolError("The file content is too large to write in one call.")
    existed = path.exists()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ToolError(f"Could not write the file: {exc}") from exc
    return {
        "ok": True,
        "summary": f"{'Updated' if existed else 'Created'} {_relative(context, path)} "
        f"({len(text.splitlines())} lines)",
        "data": {
            "path": str(path),
            "lines": len(text.splitlines()),
            "created": not existed,
            "change": _change("write", context, path, created=not existed),
        },
    }


def edit_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    old = arguments.get("old")
    new = arguments.get("new")
    if old is None or new is None:
        raise ToolError("old and new are required.")
    old_text = str(old)
    new_text = str(new)
    if not old_text:
        raise ToolError("old must not be empty; use write_file to create files.")
    text = _read_text(path)
    occurrences = text.count(old_text)
    if occurrences == 0:
        raise ToolError(
            "The text to replace was not found. Re-read the file and copy the exact text."
        )
    expected = int(arguments.get("count") or 1)
    if occurrences > expected and expected == 1:
        raise ToolError(
            f"The text occurs {occurrences} times; include more context so the edit is unambiguous."
        )
    replaced = text.replace(old_text, new_text, expected)
    try:
        path.write_text(replaced, encoding="utf-8", newline="\n")
    except OSError as exc:
        raise ToolError(f"Could not write the file: {exc}") from exc
    return {
        "ok": True,
        "summary": f"Edited {_relative(context, path)} ({min(occurrences, expected)} replacement(s))",
        "data": {
            "path": str(path),
            "replacements": min(occurrences, expected),
            "change": _change("edit", context, path),
        },
    }


def list_directory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path") or context.workspace or ".")
    if not path.exists():
        raise ToolError(f"Directory not found: {path}")
    if not path.is_dir():
        raise ToolError(f"That path is a file: {path}")
    depth = max(1, min(3, int(arguments.get("depth") or 1)))
    limit = max(1, min(1000, int(arguments.get("max_entries") or 200)))
    entries: list[dict[str, Any]] = []
    base_depth = len(path.parts)

    def walk(directory: Path) -> None:
        try:
            children = sorted(
                directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.casefold())
            )
        except OSError as exc:
            entries.append({"name": str(directory), "error": str(exc)})
            return
        for child in children:
            if len(entries) >= limit:
                return
            if child.name in SKIP_DIRS:
                continue
            try:
                stat = child.stat()
                size = stat.st_size
                modified = int(stat.st_mtime)
            except OSError:
                size, modified = 0, 0
            entries.append(
                {
                    "name": child.name,
                    "type": "dir" if child.is_dir() else "file",
                    "size": size,
                    "modified": modified,
                }
            )
            if child.is_dir() and len(child.parts) - base_depth < depth:
                walk(child)

    walk(path)
    listing = "\n".join(
        f"{entry.get('type', '?'):<4} {entry.get('name', '?')}"
        + (f" ({entry.get('size')} bytes)" if entry.get("type") == "file" else "")
        for entry in entries
    )
    return {
        "ok": True,
        "summary": f"Listed {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} in {path}",
        "data": {"path": str(path), "entries": entries, "listing": listing},
    }


def search_files(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    pattern = str(arguments.get("pattern") or "").strip()
    if not pattern:
        raise ToolError("pattern is required.")
    root = _resolve(context, arguments.get("root") or context.workspace or ".")
    glob = str(arguments.get("glob") or "*")
    limit = max(1, min(MAX_SEARCH_RESULTS, int(arguments.get("max_results") or 50)))
    matches: list[dict[str, Any]] = []
    needle = pattern.casefold()
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name not in SKIP_DIRS]
        for filename in filenames:
            if not fnmatch.fnmatch(filename, glob):
                continue
            full = Path(directory) / filename
            try:
                if full.stat().st_size > MAX_READ_BYTES:
                    continue
                text = full.read_text(encoding="utf-8", errors="ignore")
            except (OSError, UnicodeError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if needle in line.casefold():
                    matches.append(
                        {
                            "path": str(full),
                            "relative": _relative(context, full),
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
    return {
        "ok": True,
        "summary": f"{len(matches)} match(es) for {pattern!r} in {glob}",
        "data": {"pattern": pattern, "root": str(root), "matches": matches},
    }


def create_directory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    existed = path.exists()
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ToolError(f"Could not create the directory: {exc}") from exc
    return {
        "ok": True,
        "summary": f"{'Directory exists' if existed else 'Created directory'} {_relative(context, path)}",
        "data": {"path": str(path), "change": _change("mkdir", context, path)},
    }


def move_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    source = _resolve(context, arguments.get("source"))
    destination = _resolve(context, arguments.get("destination"))
    if not source.exists():
        raise ToolError(f"Source not found: {source}")
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    except OSError as exc:
        raise ToolError(f"Could not move the file: {exc}") from exc
    return {
        "ok": True,
        "summary": f"Moved {_relative(context, source)} → {_relative(context, destination)}",
        "data": {
            "path": str(destination),
            "change": _change("move", context, destination, source=str(source)),
        },
    }


def copy_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    source = _resolve(context, arguments.get("source"))
    destination = _resolve(context, arguments.get("destination"))
    if not source.is_file():
        raise ToolError(f"Source file not found: {source}")
    if destination.exists() and destination.is_dir():
        destination = destination / source.name
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))
    except OSError as exc:
        raise ToolError(f"Could not copy the file: {exc}") from exc
    return {
        "ok": True,
        "summary": f"Copied {_relative(context, source)} → {_relative(context, destination)}",
        "data": {"path": str(destination), "change": _change("copy", context, destination)},
    }


def delete_file(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    path = _resolve(context, arguments.get("path"))
    recursive = bool(arguments.get("recursive", False))
    if not path.exists():
        raise ToolError(f"Path not found: {path}")
    try:
        if path.is_dir():
            if recursive:
                shutil.rmtree(path)
            else:
                if any(path.iterdir()):
                    raise ToolError("The directory is not empty; pass recursive=true to remove it.")
                path.rmdir()
        else:
            path.unlink()
    except OSError as exc:
        raise ToolError(f"Could not delete: {exc}") from exc
    return {
        "ok": True,
        "summary": f"Deleted {_relative(context, path)}",
        "data": {
            "path": str(path),
            "change": _change("delete", context, path, recursive=recursive),
        },
    }


# -- registration ----------------------------------------------------------
def register(registry: ToolRegistry) -> None:
    path_arg = {"type": "string", "required": True, "description": "File or directory path (absolute or relative to the workspace)."}
    common = (
        "plan",
        "build",
        "autonomous",
    )

    registry.register(
        ToolSpec(
            name="read_file",
            description="Read a UTF-8 text file, optionally a line range, with line numbers.",
            kind="read",
            parameters={
                "path": path_arg,
                "start_line": {"type": "integer", "description": "First line (1-based)."},
                "max_lines": {"type": "integer", "description": "Maximum lines to return."},
            },
            modes=common,
        ),
        read_file,
    )
    registry.register(
        ToolSpec(
            name="list_directory",
            description="List a directory tree (folders first) up to three levels deep.",
            kind="read",
            parameters={
                "path": {"type": "string", "description": "Directory to list; defaults to the workspace."},
                "depth": {"type": "integer", "description": "Recursion depth, 1-3."},
                "max_entries": {"type": "integer", "description": "Entry cap (default 200)."},
            },
            modes=common,
        ),
        list_directory,
    )
    registry.register(
        ToolSpec(
            name="search_files",
            description="Search file contents for a literal string, filtered by a filename glob.",
            kind="read",
            parameters={
                "pattern": {"type": "string", "required": True, "description": "Literal text to find."},
                "root": {"type": "string", "description": "Directory to search; defaults to the workspace."},
                "glob": {"type": "string", "description": "Filename glob such as *.py."},
                "max_results": {"type": "integer", "description": "Maximum matches."},
            },
            modes=common,
        ),
        search_files,
    )
    registry.register(
        ToolSpec(
            name="write_file",
            description="Create or overwrite a UTF-8 text file, creating parent folders.",
            kind="write",
            parameters={
                "path": path_arg,
                "content": {"type": "string", "required": True, "description": "Full file content."},
            },
            modes=("build", "autonomous"),
        ),
        write_file,
    )
    registry.register(
        ToolSpec(
            name="edit_file",
            description="Replace an exact block of text in an existing file. Re-read first if unsure.",
            kind="write",
            parameters={
                "path": path_arg,
                "old": {"type": "string", "required": True, "description": "Exact existing text."},
                "new": {"type": "string", "required": True, "description": "Replacement text."},
                "count": {"type": "integer", "description": "Expected occurrences (default 1)."},
            },
            modes=("build", "autonomous"),
        ),
        edit_file,
    )
    registry.register(
        ToolSpec(
            name="create_directory",
            description="Create a directory and any missing parents.",
            kind="create_dir",
            parameters={"path": path_arg},
            modes=("build", "autonomous"),
        ),
        create_directory,
    )
    registry.register(
        ToolSpec(
            name="move_file",
            description="Move or rename a file or directory.",
            kind="move",
            parameters={
                "source": {"type": "string", "required": True, "description": "Existing path."},
                "destination": {"type": "string", "required": True, "description": "Target path."},
            },
            modes=("build", "autonomous"),
        ),
        move_file,
    )
    registry.register(
        ToolSpec(
            name="copy_file",
            description="Copy a file, creating the destination folder.",
            kind="write",
            parameters={
                "source": {"type": "string", "required": True, "description": "Existing file."},
                "destination": {"type": "string", "required": True, "description": "Target path."},
            },
            modes=("build", "autonomous"),
        ),
        copy_file,
    )
    registry.register(
        ToolSpec(
            name="delete_file",
            description="Delete a file, or a directory when recursive is true.",
            kind="delete",
            parameters={
                "path": path_arg,
                "recursive": {"type": "boolean", "description": "Remove a directory tree."},
            },
            modes=("build", "autonomous"),
        ),
        delete_file,
    )


__all__ = ["register", "read_file", "write_file", "edit_file", "list_directory", "search_files"]
