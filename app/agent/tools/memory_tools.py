"""Memory tools exposed to the agent.

These are the only way the agent can read or write long-term memory, and they
run through the tool executor like every other tool. Values that look like
credentials are rejected by :mod:`app.agent.memory`.
"""

from __future__ import annotations

from typing import Any

from app.agent.memory import MEMORY_CATEGORIES, MemoryRejected, MemoryStore
from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec


def _memory(context: ToolContext) -> MemoryStore:
    if context.memory is None:
        raise ToolError("Long-term memory is not available.")
    return context.memory


def save_memory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    memory = _memory(context)
    try:
        record = memory.save(
            str(arguments.get("category") or "task"),
            str(arguments.get("key") or ""),
            str(arguments.get("value") or ""),
            task_id=context.task_id,
        )
    except MemoryRejected as exc:
        raise ToolError(str(exc)) from exc
    return {
        "ok": True,
        "summary": f"Saved memory '{record.key}' ({record.category}).",
        "data": {"memory": record.to_dict()},
    }


def search_memory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    memory = _memory(context)
    query = str(arguments.get("query") or "").strip()
    category = str(arguments.get("category") or "").strip() or None
    if category and category not in MEMORY_CATEGORIES:
        raise ToolError(f"Unknown memory category: {category}")
    records = memory.search(query, category=category, limit=20) if query else memory.list(category)
    if not records:
        return {"ok": True, "summary": "No matching memories.", "data": {"memories": []}}
    lines = [f"[{r.category}] {r.key}: {r.value[:300]}" for r in records]
    return {
        "ok": True,
        "summary": "\n".join(lines),
        "data": {"memories": [r.to_dict() for r in records]},
    }


def update_memory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    memory = _memory(context)
    memory_id = str(arguments.get("id") or "").strip()
    if not memory_id:
        raise ToolError("id is required.")
    try:
        record = memory.update(
            memory_id,
            key=arguments.get("key"),
            value=arguments.get("value"),
            category=arguments.get("category"),
        )
    except MemoryRejected as exc:
        raise ToolError(str(exc)) from exc
    except Exception as exc:
        raise ToolError(str(exc)) from exc
    return {
        "ok": True,
        "summary": f"Updated memory '{record.key}'.",
        "data": {"memory": record.to_dict()},
    }


def delete_memory(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    memory = _memory(context)
    memory_id = str(arguments.get("id") or "").strip()
    if not memory_id:
        raise ToolError("id is required.")
    memory.delete(memory_id)
    return {"ok": True, "summary": "Memory deleted.", "data": {"id": memory_id}}


def register(registry: ToolRegistry) -> None:
    categories = ", ".join(sorted(MEMORY_CATEGORIES))
    registry.register(
        ToolSpec(
            name="save_memory",
            description=f"Store a durable note ({categories}). Never store secrets or API keys.",
            kind="memory",
            parameters={
                "category": {"type": "string", "required": True, "description": categories},
                "key": {"type": "string", "required": True, "description": "Short identifier."},
                "value": {"type": "string", "required": True, "description": "What to remember."},
            },
        ),
        save_memory,
    )
    registry.register(
        ToolSpec(
            name="search_memory",
            description="Find stored memories by text; omit the query to list everything.",
            kind="memory",
            parameters={
                "query": {"type": "string", "description": "Text to search for."},
                "category": {"type": "string", "description": categories},
            },
        ),
        search_memory,
    )
    registry.register(
        ToolSpec(
            name="update_memory",
            description="Update an existing memory entry by id.",
            kind="memory",
            parameters={
                "id": {"type": "string", "required": True, "description": "Memory id."},
                "key": {"type": "string", "description": "New key."},
                "value": {"type": "string", "description": "New value."},
                "category": {"type": "string", "description": categories},
            },
        ),
        update_memory,
    )
    registry.register(
        ToolSpec(
            name="delete_memory",
            description="Delete a memory entry by id.",
            kind="memory",
            parameters={"id": {"type": "string", "required": True, "description": "Memory id."}},
        ),
        delete_memory,
    )


__all__ = ["delete_memory", "register", "save_memory", "search_memory", "update_memory"]
