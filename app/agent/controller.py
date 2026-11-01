"""The hybrid agent's execution loop: Plan → Execute → Observe → Verify → Replan.

The controller owns a worker thread and never touches Qt. It reports progress by
calling a listener with plain dictionaries, so the GUI can bridge those events
into the Qt thread and the loop stays testable headlessly.

Safety properties implemented here:

* hard limits on steps, wall-clock time and consecutive failures,
* a rejected approval stops the task instead of trying a substitute command,
* a resumed task continues from the persisted step without re-running anything
  automatically after an application restart,
* every tool call goes through the executor (and therefore the permission
  manager); the controller cannot execute anything directly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.agent.planner import ModelUnavailable, Plan, PlanStep, Planner, format_memory
from app.agent.subagents import build_subagents, subagent_for_tool
from app.agent.tools.registry import ToolContext, ToolResult
from app.agent.validator import ToolValidationError

log = logging.getLogger(__name__)

MAX_HISTORY = 60


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int = 20
    max_seconds: float = 600.0
    max_retries: int = 3


class _OutputBuffer:
    """Coalesces streamed command output so the GUI is not flooded."""

    def __init__(self, emit: Callable[[dict[str, Any]], None], interval: float = 0.15):
        self._emit = emit
        self._interval = interval
        self._lock = threading.Lock()
        self._buffers: dict[str, list[str]] = {"stdout": [], "stderr": []}
        self._last = time.monotonic()

    def push(self, stream: str, text: str) -> None:
        key = "stderr" if stream == "stderr" else "stdout"
        with self._lock:
            self._buffers[key].append(str(text))
            now = time.monotonic()
            due = now - self._last >= self._interval
        if due:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            payload = {key: "".join(parts) for key, parts in self._buffers.items()}
            self._buffers = {"stdout": [], "stderr": []}
            self._last = time.monotonic()
        for stream, text in payload.items():
            if text:
                self._emit({"type": "output", "stream": stream, "text": text})


class AgentController:
    def __init__(
        self,
        *,
        store: Any,
        memory: Any,
        permissions: Any,
        registry: Any,
        executor: Any,
        planner: Planner,
        settings: Any,
        process_manager: Any = None,
        computer_use: Any = None,
        performance_monitor: Any = None,
        limits: AgentLimits | None = None,
    ):
        self.store = store
        self.memory = memory
        self.permissions = permissions
        self.registry = registry
        self.executor = executor
        self.planner = planner
        self.settings = settings
        self.process_manager = process_manager
        self.computer_use = computer_use
        self.performance_monitor = performance_monitor
        self._limits = limits or AgentLimits()
        self._subagents: dict[str, Any] | None = None

        self._lock = threading.RLock()
        self._listener: Callable[[dict[str, Any]], None] | None = None
        self._broker: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._state = "idle"
        self._task: Any = None
        self._output = _OutputBuffer(self._emit)

    # -- wiring ------------------------------------------------------------
    def set_listener(self, listener: Callable[[dict[str, Any]], None] | None) -> None:
        self._listener = listener

    def set_broker(self, broker: Any) -> None:
        self._broker = broker

    @property
    def subagents(self) -> dict[str, Any]:
        if self._subagents is None:
            self._subagents = build_subagents(self.planner, self.permissions)
        return dict(self._subagents)

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def task(self) -> Any:
        with self._lock:
            return self._task

    @property
    def is_running(self) -> bool:
        thread = self._thread
        return bool(thread is not None and thread.is_alive())

    def limits(self) -> AgentLimits:
        def number(key: str, default: float) -> float:
            try:
                return float(self.settings.get(key, default))
            except (TypeError, ValueError):
                return float(default)

        return AgentLimits(
            max_steps=int(number("agent.max_steps", self._limits.max_steps)),
            max_seconds=number("agent.max_seconds", self._limits.max_seconds),
            max_retries=int(number("agent.max_retries", self._limits.max_retries)),
        )

    # -- lifecycle ---------------------------------------------------------
    def start(
        self,
        goal: str,
        *,
        mode: str | None = None,
        workspace: str = "",
    ) -> Any:
        text = str(goal or "").strip()
        if not text:
            raise ValueError("目標を入力してください。")
        with self._lock:
            if self.is_running:
                raise RuntimeError("別のタスクが実行中です。停止してから開始してください。")
            selected = mode or str(self.settings.get("agent.default_mode", "autonomous"))
            if selected == "chat":
                raise ValueError("Chat モードはツールを実行しません。")
            if selected not in {"plan", "build", "autonomous"}:
                selected = "autonomous"
            workdir = str(workspace or self.settings.get("agent.workspace", "") or os.getcwd())
            task = self.store.create_task(
                text,
                mode=selected,
                model=self._model_label(),
                workspace=workdir,
                full_access=bool(self.settings.get("agent.full_access", False)),
            )
            self._task = task
            self._start_thread(task.id)
            return task

    def resume_task(self, task_id: str) -> Any:
        with self._lock:
            if self.is_running:
                raise RuntimeError("別のタスクが実行中です。")
            task = self.store.get_task(task_id)
            if task is None:
                raise ValueError("タスクが見つかりません。")
            self._task = task
            self.store.update_task(task_id, status="running")
            self._start_thread(task_id)
            return task

    def _start_thread(self, task_id: str) -> None:
        self._stop.clear()
        self._paused.clear()
        self._set_state("running", task_id)
        thread = threading.Thread(
            target=self._run, args=(task_id,), name="hybrid-agent", daemon=True
        )
        self._thread = thread
        thread.start()

    def pause(self) -> None:
        self._paused.set()
        task = self.task
        if task is not None and self.is_running:
            try:
                self.store.update_task(task.id, status="paused")
            except Exception:
                pass
            self._emit({"type": "state", "state": "paused", "task_id": task.id, "status": "paused"})

    def resume(self) -> None:
        if not self._paused.is_set():
            return
        self._paused.clear()
        task = self.task
        if task is not None:
            try:
                self.store.update_task(task.id, status="running")
            except Exception:
                pass
            self._emit({"type": "state", "state": "running", "task_id": task.id, "status": "running"})

    def stop(self) -> None:
        self._stop.set()
        self._paused.clear()
        if self.process_manager is not None:
            try:
                self.process_manager.stop_all()
            except Exception:
                log.exception("could not stop child processes")
        task = self.task
        if task is not None:
            self._emit({"type": "state", "state": "stopping", "task_id": task.id, "status": "stopped"})
        self._emit({"type": "thought", "text": "停止要求を受け付けました。"})

    def wait(self, timeout: float = 15.0) -> bool:
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self) -> None:
        self.stop()
        if self.process_manager is not None:
            try:
                self.process_manager.stop_all()
            except Exception:
                pass
        self.wait(10.0)

    # -- events ------------------------------------------------------------
    def _emit(self, event: dict[str, Any]) -> None:
        listener = self._listener
        if listener is None:
            return
        try:
            listener(event)
        except Exception:
            log.exception("agent event listener failed")

    def _set_state(self, state: str, task_id: str = "") -> None:
        with self._lock:
            self._state = state
        self._emit({"type": "state", "state": state, "task_id": task_id})

    def _model_label(self) -> str:
        try:
            return self.planner.model.label()
        except Exception:
            return ""

    # -- main loop ---------------------------------------------------------
    def _run(self, task_id: str) -> None:
        started = time.monotonic()
        status = "completed"
        summary = ""
        steps_used = 0
        failures = 0
        step_index = 0
        plan: Plan | None = None
        history: list[str] = []
        task = None
        try:
            task = self.store.get_task(task_id)
            if task is None:
                raise ValueError("タスクが見つかりません。")
            history = self._history_from_store(task_id)
            plan, step_index = self._load_plan(task)
            self._emit({"type": "task", "task": task.to_dict()})
            self._emit({"type": "plan", "plan": plan.to_dict(), "current": step_index})
            self._persist_plan(task, plan, step_index)
            if plan.notes:
                self._emit({"type": "thought", "text": plan.notes})

            if task.mode == "plan":
                summary = self._plan_summary(plan)
                status = "completed"
                return

            context = self._context(task)
            observation = ""
            while True:
                if self._stop.is_set():
                    status, summary = "stopped", "ユーザーが停止しました。"
                    break
                if self._paused.is_set():
                    self._enter_pause(task)
                    continue
                limits = self.limits()
                if steps_used >= limits.max_steps:
                    status = "failed"
                    summary = (
                        f"最大ステップ数（{limits.max_steps}）に達したため停止しました。"
                        "Agent設定で上限を変更できます。"
                    )
                    break
                elapsed = time.monotonic() - started
                if elapsed > limits.max_seconds:
                    status = "failed"
                    summary = (
                        f"最大実行時間（{int(limits.max_seconds)}秒）を超えたため停止しました。"
                    )
                    break

                step = plan.steps[min(step_index, len(plan.steps) - 1)] if plan.steps else None
                if step is not None:
                    step.status = "active"
                self._emit({"type": "plan", "plan": plan.to_dict(), "current": step_index})
                hint = self._step_hint(step, task.goal, observation)
                if hint:
                    self._emit({"type": "thought", "text": f"{step.title}: {hint}" if step else hint})

                try:
                    action = self.planner.next_action(
                        goal=task.goal,
                        plan=plan,
                        step_index=step_index,
                        history=history,
                        observation=observation,
                        mode=task.mode,
                        memory_context=self._memory_context(),
                        workspace=task.workspace,
                    )
                except ToolValidationError as exc:
                    failures += 1
                    observation = f"モデル出力の検証エラー: {exc}"
                    self._emit({"type": "thought", "text": observation})
                    if failures > limits.max_retries:
                        status, summary = "failed", f"モデルの出力を検証できませんでした: {exc}"
                        break
                    continue

                if action.get("type") == "finish":
                    summary = str(action.get("summary") or "").strip() or "タスクを完了しました。"
                    status = "completed"
                    break
                if action.get("type") == "step_done":
                    if step is not None:
                        step.status = "completed"
                        step.note = str(action.get("note") or "")[:300]
                    if all(item.status == "completed" for item in plan.steps):
                        status = "completed"
                        summary = "計画のすべてのステップを完了しました。"
                        break
                    step_index = min(step_index + 1, max(0, len(plan.steps) - 1))
                    self._persist_plan(task, plan, step_index)
                    continue

                call = action["call"]
                steps_used += 1
                thought = str(action.get("thought") or "").strip()
                if thought:
                    self._emit({"type": "thought", "text": thought})
                self._emit(
                    {
                        "type": "tool_start",
                        "tool": call.tool,
                        "arguments": call.arguments,
                        "step": step_index,
                    }
                )
                result = self.executor.execute(call, context, broker=self._broker, mode=task.mode)
                self._output.flush()
                self._emit_tool_result(call.tool, result)
                history.append(self._history_line(call.tool, result, call.arguments))
                observation = self._observation_text(result)
                self._persist_plan(task, plan, step_index)

                if result.stopped:
                    status, summary = "stopped", "ユーザーがタスクを停止しました。"
                    break
                if result.rejected:
                    status = "stopped"
                    summary = (
                        "承認が拒否されたためタスクを停止しました。"
                        "拒否された操作を別のコマンドで置き換えて実行することはしません。"
                    )
                    break
                if result.denied:
                    failures += 1
                    if failures > limits.max_retries:
                        status, summary = "failed", f"ポリシーにより実行できません: {result.summary}"
                        break
                    continue
                if not result.ok:
                    failures += 1
                    advice = self._consult(
                        "debugger",
                        task.goal,
                        f"Tool: {call.tool}\nArguments: {call.arguments}\nError: {observation}",
                    )
                    if advice:
                        observation = f"{observation}\n\nDebugger suggestion: {advice}"
                        self._emit({"type": "thought", "text": f"エラー解析: {advice}"})
                    if failures > limits.max_retries:
                        status = "failed"
                        summary = (
                            f"{failures}回連続で失敗したため停止しました。\n{observation[:2000]}"
                        )
                        break
                else:
                    failures = 0
        except ModelUnavailable as exc:
            status, summary = "failed", str(exc)
        except ToolValidationError as exc:
            status, summary = "failed", f"モデルの出力を検証できませんでした: {exc}"
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            log.exception("agent loop failed")
            status, summary = "failed", f"内部エラー: {exc}"
        finally:
            self._output.flush()
            if task is not None:
                try:
                    self.store.update_task(
                        task.id,
                        status=status,
                        summary=summary,
                        current_step=step_index,
                        plan=[step.to_dict() for step in plan.steps] if plan else [],
                        steps={str(s.index): s.status for s in plan.steps} if plan else {},
                    )
                    self.store.add_message(task.id, "assistant", summary or status, kind="summary")
                except Exception:
                    log.exception("could not persist the finished task")
                self._save_memory(task, summary, status)
                if status == "completed" and plan is not None:
                    verdict = self._consult(
                        "validator",
                        task.goal,
                        "Actions:\n" + "\n".join(history[-12:]) + f"\n\nReported result:\n{summary}",
                    )
                    if verdict:
                        self._emit({"type": "verification", "text": verdict})
            self._emit(
                {
                    "type": "finished",
                    "status": status,
                    "summary": summary,
                    "task_id": task_id,
                }
            )
            with self._lock:
                self._thread = None
                self._task = self.store.get_task(task_id) if task is not None else None
            self._set_state("idle", task_id)

    # -- helpers -----------------------------------------------------------
    def _enter_pause(self, task: Any) -> None:
        self._emit({"type": "state", "state": "paused", "task_id": task.id, "status": "paused"})
        while self._paused.is_set() and not self._stop.is_set():
            self._stop.wait(0.1)
        if not self._stop.is_set():
            self._emit({"type": "state", "state": "running", "task_id": task.id, "status": "running"})

    def _load_plan(self, task: Any) -> tuple[Plan, int]:
        steps: list[PlanStep] = []
        for item in task.plan or []:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if not title:
                continue
            steps.append(
                PlanStep(
                    index=int(item.get("index", len(steps))),
                    title=title,
                    tool=str(item.get("tool") or ""),
                    status=str(item.get("status") or "pending"),
                    note=str(item.get("note") or ""),
                )
            )
        if steps:
            plan = Plan(goal=task.goal, mode=task.mode, steps=steps)
        else:
            plan = self.planner.make_plan(
                task.goal,
                task.mode,
                workspace=task.workspace,
                memory_context=self._memory_context(),
            )
        index = 0
        for position, step in enumerate(plan.steps):
            if step.status in {"pending", "active"}:
                index = position
                break
        else:
            index = max(0, len(plan.steps) - 1)
        return plan, index

    def _persist_plan(self, task: Any, plan: Plan, step_index: int) -> None:
        try:
            self.store.update_task(
                task.id,
                plan=[step.to_dict() for step in plan.steps],
                steps={str(step.index): step.status for step in plan.steps},
                current_step=step_index,
                status="paused" if self._paused.is_set() else "running",
            )
        except Exception:
            log.exception("could not persist the plan")

    def _context(self, task: Any) -> ToolContext:
        timeout = 120.0
        try:
            timeout = float(self.settings.get("agent.command_timeout", 120))
        except (TypeError, ValueError):
            pass
        return ToolContext(
            workspace=task.workspace,
            task_id=task.id,
            mode=task.mode,
            full_access=bool(self.settings.get("agent.full_access", False)),
            source="agent",
            store=self.store,
            memory=self.memory,
            process_manager=self.process_manager,
            emit=self._output.push,
            timeout=timeout,
            save_command_history=bool(self.settings.get("agent.save_command_history", True)),
            computer_use=self.computer_use,
            performance_monitor=self.performance_monitor,
        )

    def _memory_context(self) -> str:
        if self.memory is None:
            return ""
        try:
            return format_memory(self.memory.list(limit=12))
        except Exception:
            return ""

    def _plan_summary(self, plan: Plan) -> str:
        lines = [f"「{plan.goal}」の作業計画を作成しました。", ""]
        for step in plan.steps:
            lines.append(f"{step.index + 1}. {step.title}")
        if plan.notes:
            lines.extend(["", plan.notes])
        lines.extend(
            [
                "",
                "このモードではファイル変更やコマンド実行は行いません。",
                "実行するには Build または Autonomous モードで開始してください。",
            ]
        )
        return "\n".join(lines)

    def _step_hint(self, step: PlanStep | None, goal: str, observation: str) -> str:
        if step is None:
            return ""
        agent = subagent_for_tool(self.subagents, step.tool) if step.tool else None
        if agent is None or not agent.available:
            return ""
        return self._consult(
            agent.name, goal, f"Current step: {step.title}\nLast observation: {observation[:1500]}"
        )

    def _consult(self, name: str, goal: str, context_text: str) -> str:
        agent = self.subagents.get(name)
        if agent is None:
            return ""
        return agent.advise(goal, context_text)

    def _emit_tool_result(self, tool: str, result: ToolResult) -> None:
        self._emit(
            {
                "type": "tool_result",
                "tool": tool,
                "ok": result.ok,
                "denied": result.denied,
                "rejected": result.rejected,
                "summary": result.summary or result.error,
                "data": result.data,
            }
        )
        command = result.data.get("command") if isinstance(result.data, dict) else None
        if command:
            self._emit(
                {
                    "type": "command",
                    "record": {
                        "command": command,
                        "shell": result.data.get("shell", ""),
                        "cwd": result.data.get("cwd", ""),
                        "exit_code": result.data.get("exit_code"),
                        "duration": result.data.get("duration", 0),
                        "stdout": result.data.get("stdout", ""),
                        "stderr": result.data.get("stderr", ""),
                        "timed_out": result.data.get("timed_out", False),
                        "cancelled": result.data.get("cancelled", False),
                        "source": "agent",
                    },
                }
            )

    @staticmethod
    def _observation_text(result: ToolResult) -> str:
        parts = []
        if result.summary:
            parts.append(result.summary)
        elif result.error:
            parts.append(result.error)
        data = result.data or {}
        if data.get("exit_code") is not None:
            parts.append(f"exit code: {data.get('exit_code')}")
        for key in ("listing", "content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                parts.append(value[:3000])
                break
        return "\n".join(parts)[:6000]

    @staticmethod
    def _history_line(tool: str, result: ToolResult, arguments: dict[str, Any]) -> str:
        detail = ""
        data = result.data or {}
        if data.get("command"):
            detail = str(data.get("command"))
        elif data.get("path"):
            detail = str(data.get("path"))
        elif arguments.get("path"):
            detail = str(arguments.get("path"))
        state = "ok" if result.ok else "failed"
        first_line = (result.summary or result.error or "").splitlines()[0][:200]
        return f"- {tool}({detail[:120]}) → {state}: {first_line}"

    def _history_from_store(self, task_id: str) -> list[str]:
        lines: list[str] = []
        try:
            for record in self.store.list_commands(task_id, limit=12):
                lines.append(
                    f"- {record.shell}: {record.command} → exit {record.exit_code}"
                )
        except Exception:
            pass
        try:
            for message in self.store.list_messages(task_id)[-6:]:
                if message.role == "user":
                    lines.append(f"- user: {message.content[:200]}")
        except Exception:
            pass
        return lines[-MAX_HISTORY:]

    def _save_memory(self, task: Any, summary: str, status: str) -> None:
        if self.memory is None:
            return
        if status not in {"completed", "failed"} or not summary:
            return
        try:
            self.memory.save(
                "task",
                key=f"task: {task.goal[:60]}",
                value=f"[{status}] {summary[:900]}",
                task_id=task.id,
            )
        except Exception:
            pass


__all__ = ["AgentController", "AgentLimits"]
