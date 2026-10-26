"""SQLite persistence for chats and messages."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chats (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    params TEXT NOT NULL DEFAULT '{}',
    system_prompt TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    chat_id TEXT NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chats_updated ON chats(updated_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


@dataclass
class Chat:
    id: str
    title: str
    model: str = ""
    provider: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    system_prompt: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["params"] = dict(self.params)
        return data


@dataclass
class Message:
    id: str
    chat_id: str
    role: str
    content: str
    meta: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.__dict__)
        data["meta"] = dict(self.meta)
        return data


class DatabaseError(RuntimeError):
    pass


class Database:
    """A single connection guarded by a re-entrant lock.

    Chat generation runs in worker threads and the API server writes as well,
    so serialising access with one lock is simpler (and plenty fast) than
    juggling per-thread connections.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        try:
            self._conn = sqlite3.connect(
                str(self.path), check_same_thread=False, timeout=15.0
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
            with self._conn:
                self._conn.executescript(_SCHEMA)
                version = self._conn.execute("PRAGMA user_version").fetchone()[0]
                if version < SCHEMA_VERSION:
                    self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except sqlite3.Error as exc:
            raise DatabaseError(f"Could not open chat database: {exc}") from exc

    # -- chats -------------------------------------------------------------
    def create_chat(
        self,
        title: str = "New chat",
        model: str = "",
        provider: str = "",
        params: dict[str, Any] | None = None,
        system_prompt: str = "",
    ) -> Chat:
        chat = Chat(
            id=f"chat_{uuid.uuid4().hex}",
            title=title or "New chat",
            model=model,
            provider=provider,
            params=dict(params or {}),
            system_prompt=system_prompt,
        )
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO chats (id,title,model,provider,params,system_prompt,created_at,updated_at)"
                        " VALUES (?,?,?,?,?,?,?,?)",
                        (
                            chat.id,
                            chat.title,
                            chat.model,
                            chat.provider,
                            json.dumps(chat.params, ensure_ascii=False),
                            chat.system_prompt,
                            chat.created_at,
                            chat.updated_at,
                        ),
                    )
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not create chat: {exc}") from exc
        return chat

    def get_chat(self, chat_id: str) -> Chat | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chats WHERE id = ?", (chat_id,)
            ).fetchone()
        return self._chat_from_row(row) if row else None

    def list_chats(self, limit: int = 200) -> list[Chat]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self._chat_from_row(row) for row in rows]

    def update_chat(self, chat_id: str, **fields: Any) -> None:
        allowed = {"title", "model", "provider", "params", "system_prompt", "updated_at"}
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "params":
                value = json.dumps(value or {}, ensure_ascii=False)
            updates[key] = value
        updates.setdefault("updated_at", utc_now())
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        f"UPDATE chats SET {assignments} WHERE id = ?",
                        (*updates.values(), chat_id),
                    )
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not update chat: {exc}") from exc

    def delete_chat(self, chat_id: str) -> None:
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not delete chat: {exc}") from exc

    # -- messages ----------------------------------------------------------
    def add_message(
        self,
        chat_id: str,
        role: str,
        content: str,
        meta: dict[str, Any] | None = None,
    ) -> Message:
        message = Message(
            id=f"msg_{uuid.uuid4().hex}",
            chat_id=chat_id,
            role=role,
            content=content,
            meta=dict(meta or {}),
        )
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(
                        "INSERT INTO messages (id,chat_id,role,content,meta,created_at)"
                        " VALUES (?,?,?,?,?,?)",
                        (
                            message.id,
                            chat_id,
                            role,
                            content,
                            json.dumps(message.meta, ensure_ascii=False),
                            message.created_at,
                        ),
                    )
                    self._conn.execute(
                        "UPDATE chats SET updated_at = ? WHERE id = ?",
                        (message.created_at, chat_id),
                    )
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not save message: {exc}") from exc
        return message

    def update_message(self, message_id: str, content: str, meta: dict[str, Any] | None = None) -> None:
        with self._lock:
            try:
                with self._conn:
                    if meta is None:
                        self._conn.execute(
                            "UPDATE messages SET content = ? WHERE id = ?",
                            (content, message_id),
                        )
                    else:
                        self._conn.execute(
                            "UPDATE messages SET content = ?, meta = ? WHERE id = ?",
                            (content, json.dumps(meta, ensure_ascii=False), message_id),
                        )
                    row = self._conn.execute(
                        "SELECT chat_id FROM messages WHERE id = ?", (message_id,)
                    ).fetchone()
                    if row:
                        self._conn.execute(
                            "UPDATE chats SET updated_at = ? WHERE id = ?",
                            (utc_now(), row["chat_id"]),
                        )
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not update message: {exc}") from exc

    def list_messages(self, chat_id: str) -> list[Message]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM messages WHERE chat_id = ? ORDER BY created_at ASC, rowid ASC",
                (chat_id,),
            ).fetchall()
        return [self._message_from_row(row) for row in rows]

    def delete_message(self, message_id: str) -> None:
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM messages WHERE id = ?", (message_id,))
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not delete message: {exc}") from exc

    # -- maintenance -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        with self._lock:
            chats = self._conn.execute("SELECT COUNT(*) FROM chats").fetchone()[0]
            messages = self._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {"chats": chats, "messages": messages, "bytes": size, "path": str(self.path)}

    def clear_history(self) -> None:
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute("DELETE FROM messages")
                    self._conn.execute("DELETE FROM chats")
            except sqlite3.Error as exc:
                raise DatabaseError(f"Could not clear history: {exc}") from exc

    def checkpoint(self) -> None:
        with self._lock:
            try:
                self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                pass

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _chat_from_row(row: sqlite3.Row) -> Chat:
        try:
            params = json.loads(row["params"] or "{}")
        except ValueError:
            params = {}
        return Chat(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            provider=row["provider"],
            params=params if isinstance(params, dict) else {},
            system_prompt=row["system_prompt"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _message_from_row(row: sqlite3.Row) -> Message:
        try:
            meta = json.loads(row["meta"] or "{}")
        except ValueError:
            meta = {}
        return Message(
            id=row["id"],
            chat_id=row["chat_id"],
            role=row["role"],
            content=row["content"],
            meta=meta if isinstance(meta, dict) else {},
            created_at=row["created_at"],
        )
