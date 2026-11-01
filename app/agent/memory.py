"""Long-term memory for the Hybrid Agent.

Four categories (session / project / user preference / task) live in the agent
SQLite database, so the agent can reuse previous findings instead of repeating
the same investigation. Credentials are never stored: values that look like API
keys, tokens or passwords are rejected and the rejection is reported back to the
caller.

Memory is *reference information only*. It is never consulted by the permission
manager, so a memory entry can never widen what the agent may do.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

MEMORY_CATEGORIES = {
    "session": "セッション",
    "project": "プロジェクト",
    "user_preference": "ユーザー設定",
    "task": "タスク",
}

MAX_VALUE_LENGTH = 4000

_SECRET_PATTERNS = (
    re.compile(r"\b(sk|rk|pk)-[A-Za-z0-9_\-]{12,}"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9\-]{10,}"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(
        r"\b(api[_\- ]?key|access[_\- ]?token|secret|password|passwd|pwd|"
        r"client[_\- ]?secret|bearer)\b\s*[:=]\s*\S+",
        re.IGNORECASE,
    ),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class MemoryError(RuntimeError):
    pass


class MemoryRejected(ValueError):
    """Raised when a value must not be written to long-term memory."""


@dataclass(frozen=True)
class MemoryRecord:
    id: str
    category: str
    key: str
    value: str
    task_id: str = ""
    source: str = "agent"
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def looks_secret(text: str) -> bool:
    value = str(text or "")
    return any(pattern.search(value) for pattern in _SECRET_PATTERNS)


class MemoryStore:
    """CRUD over ``agent_memories`` with secret screening."""

    def __init__(self, store: Any):
        self._store = store

    # -- write -------------------------------------------------------------
    def save(
        self,
        category: str,
        key: str,
        value: str,
        *,
        task_id: str = "",
        source: str = "agent",
    ) -> MemoryRecord:
        category = str(category or "").strip()
        if category not in MEMORY_CATEGORIES:
            raise MemoryRejected(f"Unknown memory category: {category}")
        key = str(key or "").strip()
        value = str(value or "").strip()
        if not key:
            raise MemoryRejected("A memory entry needs a key.")
        if not value:
            raise MemoryRejected("A memory entry needs a value.")
        if looks_secret(key) or looks_secret(value):
            raise MemoryRejected("Memories cannot contain credentials or API keys.")
        if len(value) > MAX_VALUE_LENGTH:
            value = value[:MAX_VALUE_LENGTH]
        now = utc_now()
        rows = self._store.query(
            "SELECT id FROM agent_memories WHERE category = ? AND key = ?", (category, key)
        )
        if rows:
            memory_id = rows[0]["id"]
            self._store.execute(
                "UPDATE agent_memories SET value = ?, updated_at = ?, source = ? WHERE id = ?",
                (value, now, source, memory_id),
            )
        else:
            memory_id = f"mem_{uuid.uuid4().hex}"
            self._store.execute(
                "INSERT INTO agent_memories (id,category,key,value,task_id,source,created_at,"
                "updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (memory_id, category, key, value, task_id, source, now, now),
            )
        record = self.get(memory_id)
        if record is None:
            raise MemoryError("The memory entry could not be saved.")
        return record

    # -- read --------------------------------------------------------------
    def get(self, memory_id: str) -> MemoryRecord | None:
        rows = self._store.query("SELECT * FROM agent_memories WHERE id = ?", (memory_id,))
        return self._from_row(rows[0]) if rows else None

    def list(self, category: str | None = None, limit: int = 500) -> list[MemoryRecord]:
        if category:
            rows = self._store.query(
                "SELECT * FROM agent_memories WHERE category = ? ORDER BY updated_at DESC LIMIT ?",
                (category, int(limit)),
            )
        else:
            rows = self._store.query(
                "SELECT * FROM agent_memories ORDER BY updated_at DESC LIMIT ?", (int(limit),)
            )
        return [self._from_row(row) for row in rows]

    def search(
        self, query: str, *, category: str | None = None, limit: int = 20
    ) -> list[MemoryRecord]:
        needle = f"%{str(query or '').strip()}%"
        if category:
            rows = self._store.query(
                "SELECT * FROM agent_memories WHERE category = ? AND (key LIKE ? OR value LIKE ?)"
                " ORDER BY updated_at DESC LIMIT ?",
                (category, needle, needle, int(limit)),
            )
        else:
            rows = self._store.query(
                "SELECT * FROM agent_memories WHERE key LIKE ? OR value LIKE ?"
                " ORDER BY updated_at DESC LIMIT ?",
                (needle, needle, int(limit)),
            )
        return [self._from_row(row) for row in rows]

    # -- update / delete ---------------------------------------------------
    def update(
        self,
        memory_id: str,
        *,
        key: str | None = None,
        value: str | None = None,
        category: str | None = None,
    ) -> MemoryRecord:
        existing = self.get(memory_id)
        if existing is None:
            raise MemoryError("That memory entry no longer exists.")
        new_key = existing.key if key is None else str(key).strip()
        new_value = existing.value if value is None else str(value).strip()
        new_category = existing.category if category is None else str(category).strip()
        if not new_key or not new_value:
            raise MemoryRejected("A memory entry needs both a key and a value.")
        if new_category not in MEMORY_CATEGORIES:
            raise MemoryRejected(f"Unknown memory category: {new_category}")
        if looks_secret(new_key) or looks_secret(new_value):
            raise MemoryRejected("Memories cannot contain credentials or API keys.")
        self._store.execute(
            "UPDATE agent_memories SET key = ?, value = ?, category = ?, updated_at = ?"
            " WHERE id = ?",
            (new_key, new_value[:MAX_VALUE_LENGTH], new_category, utc_now(), memory_id),
        )
        updated = self.get(memory_id)
        if updated is None:
            raise MemoryError("The memory entry could not be updated.")
        return updated

    def delete(self, memory_id: str) -> None:
        self._store.execute("DELETE FROM agent_memories WHERE id = ?", (memory_id,))

    def clear(self, category: str | None = None) -> int:
        if category:
            rows = self._store.query(
                "SELECT COUNT(*) AS n FROM agent_memories WHERE category = ?", (category,)
            )
            self._store.execute("DELETE FROM agent_memories WHERE category = ?", (category,))
        else:
            rows = self._store.query("SELECT COUNT(*) AS n FROM agent_memories")
            self._store.execute("DELETE FROM agent_memories")
        return int(rows[0]["n"]) if rows else 0

    @staticmethod
    def _from_row(row: Any) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            category=row["category"],
            key=row["key"],
            value=row["value"],
            task_id=row["task_id"],
            source=row["source"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


__all__ = [
    "MEMORY_CATEGORIES",
    "MemoryError",
    "MemoryRecord",
    "MemoryRejected",
    "MemoryStore",
    "looks_secret",
]
