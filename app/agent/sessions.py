"""SQLite persistence for Hybrid Agent tasks, approvals and grants.

One connection guarded by a re-entrant lock, mirroring ``app.core.database``.
Every task is resumable: the plan, the step that is running, command history,
tool results and approvals are all stored, and tasks that were interrupted are
recovered as *paused* so nothing runs again without the user pressing Resume.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_tasks (
    id TEXT PRIMARY KEY,
    goal TEXT NOT NULL,
    mode TEXT NOT NULL DEFAULT 'autonomous',
    status TEXT NOT NULL DEFAULT 'pending',
    plan TEXT NOT NULL DEFAULT '[]',
    current_step INTEGER NOT NULL DEFAULT 0,
    steps TEXT NOT NULL DEFAULT '{}',
    model TEXT NOT NULL DEFAULT '',
    workspace TEXT NOT NULL DEFAULT '',
    full_access INTEGER NOT NULL DEFAULT 0,
    summary TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_messages (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL REFERENCES agent_tasks(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'chat',
    content TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_commands (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    shell TEXT NOT NULL,
    command TEXT NOT NULL,
    cwd TEXT NOT NULL DEFAULT '',
    exit_code INTEGER,
    stdout TEXT NOT NULL DEFAULT '',
    stderr TEXT NOT NULL DEFAULT '',
    duration REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'done',
    source TEXT NOT NULL DEFAULT 'agent',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_approvals (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    kind TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    command TEXT NOT NULL DEFAULT '',
    decision TEXT NOT NULL,
    risk TEXT NOT NULL DEFAULT '',
    fingerprint TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_grants (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    scope TEXT NOT NULL DEFAULT '',
    target TEXT NOT NULL DEFAULT '',
    category TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    last_used TEXT NOT NULL DEFAULT '',
    uses INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS agent_memories (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    task_id TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT 'agent',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS computer_use_actions (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    target TEXT NOT NULL DEFAULT '',
    approved INTEGER NOT NULL DEFAULT 0,
    verified INTEGER NOT NULL DEFAULT 0,
    details TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS performance_samples (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL DEFAULT '',
    sample TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_updated ON agent_tasks(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_agent_messages_task ON agent_messages(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_commands_task ON agent_commands(task_id, created_at);
CREATE INDEX IF NOT EXISTS idx_agent_memories_category ON agent_memories(category, updated_at DESC);
"""

#: Statuses a task can be left in. Only a paused task may be resumed.
STATUSES = (
    "pending",
    "running",
    "paused",
    "awaiting_approval",
    "completed",
    "failed",
    "stopped",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _json_load(value: Any, fallback: Any) -> Any:
    try:
        parsed = json.loads(value or "null")
    except (TypeError, ValueError):
        return fallback
    return parsed if isinstance(parsed, type(fallback)) else fallback


@dataclass
class TaskRecord:
    id: str
    goal: str
    mode: str = "autonomous"
    status: str = "pending"
    plan: list[dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    steps: dict[str, str] = field(default_factory=dict)
    model: str = ""
    workspace: str = ""
    full_access: bool = False
    summary: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "goal": self.goal,
            "mode": self.mode,
            "status": self.status,
            "plan": [dict(item) for item in self.plan],
            "current_step": self.current_step,
            "steps": dict(self.steps),
            "model": self.model,
            "workspace": self.workspace,
            "full_access": self.full_access,
            "summary": self.summary,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass
class MessageRecord:
    id: str
    task_id: str
    role: str
    content: str
    kind: str = "chat"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class CommandRecord:
    id: str
    command: str
    shell: str
    cwd: str = ""
    task_id: str = ""
    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration: float = 0.0
    status: str = "done"
    source: str = "agent"
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ApprovalRecord:
    id: str
    decision: str
    task_id: str = ""
    kind: str = ""
    category: str = ""
    target: str = ""
    command: str = ""
    risk: str = ""
    fingerprint: str = ""
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class GrantRecord:
    id: str
    kind: str
    scope: str = ""
    target: str = ""
    category: str = ""
    created_at: str = field(default_factory=utc_now)
    last_used: str = ""
    uses: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @property
    def label(self) -> str:
        parts = [self.kind]
        if self.target:
            parts.append(self.target)
        if self.scope:
            parts.append(self.scope)
        return " · ".join(parts)


class AgentStoreError(RuntimeError):
    pass


class AgentStore:
    """All agent state in one SQLite file."""

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
            raise AgentStoreError(f"Could not open the agent database: {exc}") from exc

    # -- low level helpers (shared with MemoryStore) ------------------------
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self._conn.execute(sql, tuple(params)).fetchall()
            except sqlite3.Error as exc:
                raise AgentStoreError(str(exc)) from exc

    def execute(self, sql: str, params: Sequence[Any] = ()) -> None:
        with self._lock:
            try:
                with self._conn:
                    self._conn.execute(sql, tuple(params))
            except sqlite3.Error as exc:
                raise AgentStoreError(str(exc)) from exc

    # -- tasks -------------------------------------------------------------
    def create_task(
        self,
        goal: str,
        *,
        mode: str = "autonomous",
        model: str = "",
        workspace: str = "",
        full_access: bool = False,
        task_id: str = "",
    ) -> TaskRecord:
        task = TaskRecord(
            id=task_id or f"task_{uuid.uuid4().hex}",
            goal=str(goal or "").strip(),
            mode=mode if mode in {"chat", "plan", "build", "autonomous"} else "autonomous",
            model=model,
            workspace=workspace,
            full_access=bool(full_access),
        )
        self.execute(
            "INSERT INTO agent_tasks (id,goal,mode,status,plan,current_step,steps,model,"
            "workspace,full_access,summary,created_at,updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                task.id,
                task.goal,
                task.mode,
                task.status,
                "[]",
                0,
                "{}",
                task.model,
                task.workspace,
                int(task.full_access),
                "",
                task.created_at,
                task.updated_at,
            ),
        )
        return task

    def get_task(self, task_id: str) -> TaskRecord | None:
        rows = self.query("SELECT * FROM agent_tasks WHERE id = ?", (task_id,))
        return self._task_from_row(rows[0]) if rows else None

    def list_tasks(self, limit: int = 100, status: str | None = None) -> list[TaskRecord]:
        if status:
            rows = self.query(
                "SELECT * FROM agent_tasks WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                (status, int(limit)),
            )
        else:
            rows = self.query(
                "SELECT * FROM agent_tasks ORDER BY updated_at DESC LIMIT ?", (int(limit),)
            )
        return [self._task_from_row(row) for row in rows]

    def update_task(self, task_id: str, **fields: Any) -> None:
        allowed = {
            "goal",
            "mode",
            "status",
            "plan",
            "current_step",
            "steps",
            "model",
            "workspace",
            "full_access",
            "summary",
        }
        updates: dict[str, Any] = {}
        for key, value in fields.items():
            if key not in allowed:
                continue
            if key == "plan":
                value = json.dumps(list(value or []), ensure_ascii=False)
            elif key == "steps":
                value = json.dumps(dict(value or {}), ensure_ascii=False)
            elif key == "full_access":
                value = int(bool(value))
            updates[key] = value
        if not updates:
            return
        updates["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        self.execute(
            f"UPDATE agent_tasks SET {assignments} WHERE id = ?",
            (*updates.values(), task_id),
        )

    def delete_task(self, task_id: str) -> None:
        self.execute("DELETE FROM agent_tasks WHERE id = ?", (task_id,))

    def add_computer_action(
        self, *, task_id: str = "", action: str, target: str = "",
        approved: bool = False, verified: bool = False, details: dict[str, Any] | None = None
    ) -> str:
        action_id = f"computer_{uuid.uuid4().hex}"
        self.execute(
            "INSERT INTO computer_use_actions "
            "(id,task_id,action,target,approved,verified,details,created_at) VALUES (?,?,?,?,?,?,?,?)",
            (action_id, task_id, action, target, int(approved), int(verified),
             json.dumps(details or {}, ensure_ascii=False), utc_now()),
        )
        return action_id

    def list_computer_actions(self, task_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT * FROM computer_use_actions"
        params: tuple[Any, ...] = ()
        if task_id:
            sql += " WHERE task_id = ?"
            params = (task_id,)
        rows = self.query(sql + " ORDER BY created_at DESC LIMIT ?", (*params, int(limit)))
        return [
            {**dict(row), "approved": bool(row["approved"]), "verified": bool(row["verified"]),
             "details": _json_load(row["details"], {})}
            for row in rows
        ]

    def clear_computer_actions(self, task_id: str | None = None) -> None:
        if task_id:
            self.execute("DELETE FROM computer_use_actions WHERE task_id = ?", (task_id,))
        else:
            self.execute("DELETE FROM computer_use_actions")

    def add_performance_sample(self, sample: dict[str, Any], *, task_id: str = "") -> str:
        sample_id = f"perf_{uuid.uuid4().hex}"
        self.execute(
            "INSERT INTO performance_samples (id,task_id,sample,created_at) VALUES (?,?,?,?)",
            (sample_id, task_id, json.dumps(sample, ensure_ascii=False), utc_now()),
        )
        return sample_id

    def list_performance_samples(self, limit: int = 300) -> list[dict[str, Any]]:
        rows = self.query(
            "SELECT id,task_id,sample,created_at FROM performance_samples ORDER BY created_at DESC LIMIT ?",
            (int(limit),),
        )
        return [
            {"id": row["id"], "task_id": row["task_id"], "created_at": row["created_at"],
             **_json_load(row["sample"], {})}
            for row in rows
        ]

    def clear_performance_samples(self) -> None:
        self.execute("DELETE FROM performance_samples")

    def recover_interrupted(self) -> int:
        """Mark tasks that were mid-flight when the app closed as paused.

        This is what makes requirement 17 safe: after a restart nothing is
        executed until the user explicitly resumes.
        """
        rows = self.query(
            "SELECT id FROM agent_tasks WHERE status IN ('running','awaiting_approval','pending')"
        )
        for row in rows:
            self.update_task(row["id"], status="paused")
        return len(rows)

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> TaskRecord:
        return TaskRecord(
            id=row["id"],
            goal=row["goal"],
            mode=row["mode"],
            status=row["status"],
            plan=_json_load(row["plan"], []),
            current_step=int(row["current_step"]),
            steps=_json_load(row["steps"], {}),
            model=row["model"],
            workspace=row["workspace"],
            full_access=bool(row["full_access"]),
            summary=row["summary"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # -- messages ----------------------------------------------------------
    def add_message(
        self, task_id: str, role: str, content: str, *, kind: str = "chat"
    ) -> MessageRecord:
        message = MessageRecord(
            id=f"msg_{uuid.uuid4().hex}",
            task_id=task_id,
            role=role,
            content=str(content),
            kind=kind,
        )
        self.execute(
            "INSERT INTO agent_messages (id,task_id,role,kind,content,created_at)"
            " VALUES (?,?,?,?,?,?)",
            (message.id, task_id, role, kind, message.content, message.created_at),
        )
        self.update_task(task_id, status=self._status_of(task_id))
        return message

    def _status_of(self, task_id: str) -> str:
        task = self.get_task(task_id)
        return task.status if task else "running"

    def list_messages(self, task_id: str) -> list[MessageRecord]:
        rows = self.query(
            "SELECT * FROM agent_messages WHERE task_id = ? ORDER BY created_at, rowid",
            (task_id,),
        )
        return [
            MessageRecord(
                id=row["id"],
                task_id=row["task_id"],
                role=row["role"],
                content=row["content"],
                kind=row["kind"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- command history ---------------------------------------------------
    def add_command(
        self,
        *,
        shell: str,
        command: str,
        cwd: str = "",
        task_id: str = "",
        exit_code: int | None = None,
        stdout: str = "",
        stderr: str = "",
        duration: float = 0.0,
        status: str = "done",
        source: str = "agent",
    ) -> CommandRecord:
        record = CommandRecord(
            id=f"cmd_{uuid.uuid4().hex}",
            command=str(command),
            shell=str(shell),
            cwd=str(cwd),
            task_id=task_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            duration=float(duration),
            status=status,
            source=source,
        )
        self.execute(
            "INSERT INTO agent_commands (id,task_id,shell,command,cwd,exit_code,stdout,"
            "stderr,duration,status,source,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.task_id,
                record.shell,
                record.command,
                record.cwd,
                record.exit_code,
                record.stdout,
                record.stderr,
                record.duration,
                record.status,
                record.source,
                record.created_at,
            ),
        )
        return record

    def list_commands(self, task_id: str | None = None, limit: int = 200) -> list[CommandRecord]:
        if task_id:
            rows = self.query(
                "SELECT * FROM agent_commands WHERE task_id = ? ORDER BY created_at, rowid LIMIT ?",
                (task_id, int(limit)),
            )
        else:
            rows = self.query(
                "SELECT * FROM agent_commands ORDER BY created_at DESC LIMIT ?", (int(limit),)
            )
        return [
            CommandRecord(
                id=row["id"],
                command=row["command"],
                shell=row["shell"],
                cwd=row["cwd"],
                task_id=row["task_id"],
                exit_code=row["exit_code"],
                stdout=row["stdout"],
                stderr=row["stderr"],
                duration=row["duration"],
                status=row["status"],
                source=row["source"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- approvals ---------------------------------------------------------
    def add_approval(
        self,
        *,
        decision: str,
        task_id: str = "",
        kind: str = "",
        category: str = "",
        target: str = "",
        command: str = "",
        risk: str = "",
        fingerprint: str = "",
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            id=f"apr_{uuid.uuid4().hex}",
            decision=decision,
            task_id=task_id,
            kind=kind,
            category=category,
            target=target,
            command=command,
            risk=risk,
            fingerprint=fingerprint,
        )
        self.execute(
            "INSERT INTO agent_approvals (id,task_id,kind,category,target,command,decision,"
            "risk,fingerprint,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                record.id,
                record.task_id,
                record.kind,
                record.category,
                record.target,
                record.command,
                record.decision,
                record.risk,
                record.fingerprint,
                record.created_at,
            ),
        )
        return record

    def list_approvals(
        self, task_id: str | None = None, limit: int = 200
    ) -> list[ApprovalRecord]:
        if task_id:
            rows = self.query(
                "SELECT * FROM agent_approvals WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                (task_id, int(limit)),
            )
        else:
            rows = self.query(
                "SELECT * FROM agent_approvals ORDER BY created_at DESC LIMIT ?", (int(limit),)
            )
        return [
            ApprovalRecord(
                id=row["id"],
                decision=row["decision"],
                task_id=row["task_id"],
                kind=row["kind"],
                category=row["category"],
                target=row["target"],
                command=row["command"],
                risk=row["risk"],
                fingerprint=row["fingerprint"],
                created_at=row["created_at"],
            )
            for row in rows
        ]

    # -- always-allow grants ----------------------------------------------
    def add_grant(
        self, *, kind: str, scope: str = "", target: str = "", category: str = ""
    ) -> GrantRecord:
        existing = self.query(
            "SELECT * FROM agent_grants WHERE kind = ? AND scope = ? AND target = ?",
            (kind, scope, target),
        )
        if existing:
            self.execute(
                "UPDATE agent_grants SET category = ?, last_used = ? WHERE id = ?",
                (category, utc_now(), existing[0]["id"]),
            )
            return self._grant_from_row(self.query(
                "SELECT * FROM agent_grants WHERE id = ?", (existing[0]["id"],)
            )[0])
        grant = GrantRecord(
            id=f"grant_{uuid.uuid4().hex}",
            kind=kind,
            scope=scope,
            target=target,
            category=category,
        )
        self.execute(
            "INSERT INTO agent_grants (id,kind,scope,target,category,created_at,last_used,uses)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (grant.id, kind, scope, target, category, grant.created_at, "", 0),
        )
        return grant

    def list_grants(self) -> list[GrantRecord]:
        rows = self.query("SELECT * FROM agent_grants ORDER BY created_at DESC")
        return [self._grant_from_row(row) for row in rows]

    def get_grant(self, grant_id: str) -> GrantRecord | None:
        rows = self.query("SELECT * FROM agent_grants WHERE id = ?", (grant_id,))
        return self._grant_from_row(rows[0]) if rows else None

    def delete_grant(self, grant_id: str) -> None:
        self.execute("DELETE FROM agent_grants WHERE id = ?", (grant_id,))

    def touch_grant(self, grant_id: str) -> None:
        self.execute(
            "UPDATE agent_grants SET uses = uses + 1, last_used = ? WHERE id = ?",
            (utc_now(), grant_id),
        )

    @staticmethod
    def _grant_from_row(row: sqlite3.Row) -> GrantRecord:
        return GrantRecord(
            id=row["id"],
            kind=row["kind"],
            scope=row["scope"],
            target=row["target"],
            category=row["category"],
            created_at=row["created_at"],
            last_used=row["last_used"],
            uses=int(row["uses"]),
        )

    # -- maintenance -------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        def count(table: str) -> int:
            rows = self.query(f"SELECT COUNT(*) AS n FROM {table}")
            return int(rows[0]["n"]) if rows else 0

        try:
            size = self.path.stat().st_size
        except OSError:
            size = 0
        return {
            "tasks": count("agent_tasks"),
            "commands": count("agent_commands"),
            "approvals": count("agent_approvals"),
            "grants": count("agent_grants"),
            "memories": count("agent_memories"),
            "bytes": size,
            "path": str(self.path),
        }

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


def task_status_label(status: str) -> str:
    return {
        "pending": "待機中",
        "running": "実行中",
        "paused": "一時停止",
        "awaiting_approval": "承認待ち",
        "completed": "完了",
        "failed": "失敗",
        "stopped": "停止",
    }.get(status, status)
