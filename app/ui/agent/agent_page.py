"""Hybrid Agent page: three-column workspace, terminal, approvals and memory."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.agent.controller import AgentController
from app.agent.computer_use import ComputerUse
from app.agent.executor import ToolExecutor
from app.agent.memory import MemoryStore
from app.agent.permissions import ApprovalOutcome, ApprovalRequest, PermissionManager
from app.agent.planner import ModelClient, Planner
from app.agent.process_manager import ProcessManager
from app.agent.sessions import task_status_label
from app.agent.tools import ToolCall, ToolContext, create_registry
from app.agent.tools.registry import ToolResult
from app.core.types import GenerationParams
from app.ui.agent.approval_dialog import ApprovalDialog
from app.ui.agent.memory_page import MemoryPage
from app.ui.agent.progress_panel import ProgressPanel
from app.ui.agent.task_panel import EventLogPanel, PlanPanel, TaskListPanel
from app.ui.agent.terminal_panel import TerminalPanel
from app.ui.i18n import ja
from app.ui.services import AppServices
from app.ui.theme import current_palette
from app.ui.widgets import Card, MessageBubble, muted_label
from app.workers.workers import ChatWorker, TaskWorker

log = logging.getLogger(__name__)

MODE_LABELS = (
    ("chat", "Chat Mode"),
    ("plan", "Plan Mode"),
    ("build", "Build Mode"),
    ("autonomous", "Autonomous Mode"),
)

MANUAL_TOOL = {
    "cmd": "execute_cmd",
    "powershell": "execute_powershell",
    "python": "execute_python",
    "git": "execute_cmd",
}


class _PendingApproval:
    def __init__(self, request: ApprovalRequest):
        self.request = request
        self.outcome: ApprovalOutcome | None = None
        self.event = threading.Event()


class QtApprovalBroker(QObject):
    """Bridges a blocking approval request (agent thread) to the Qt dialog."""

    requested = Signal(object)

    def __init__(self, timeout: float = 600.0, parent: QObject | None = None):
        super().__init__(parent)
        self.timeout = float(timeout)
        self._lock = threading.Lock()
        self._pending: _PendingApproval | None = None

    @property
    def has_pending(self) -> bool:
        with self._lock:
            return self._pending is not None

    def request(self, request: ApprovalRequest) -> ApprovalOutcome:
        pending = _PendingApproval(request)
        with self._lock:
            self._pending = pending
        try:
            self.requested.emit(request)
        except Exception:
            log.exception("could not show the approval dialog")
        answered = pending.event.wait(self.timeout)
        with self._lock:
            self._pending = None
        if not answered or pending.outcome is None:
            # No answer: refuse. Never default to allowing an action.
            return ApprovalOutcome("reject", request.fingerprint)
        return pending.outcome

    def resolve(self, outcome: ApprovalOutcome) -> None:
        with self._lock:
            pending = self._pending
        if pending is None:
            return
        pending.outcome = outcome
        pending.event.set()


class AgentBridge(QObject):
    event = Signal(object)


class AgentPage(QWidget):
    show_status_message = Signal(str)
    output_received = Signal(str, str)

    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        self.settings = services.settings
        try:
            self.store = services.agent_store
        except Exception:
            log.exception("agent database is unavailable; using a recovery database")
            from app.agent.sessions import AgentStore
            from app.core import paths

            self.store = AgentStore(paths.data_dir() / "agent-recovered.db")
        self.memory = MemoryStore(self.store)

        self.bridge = AgentBridge(self)
        self.broker = QtApprovalBroker(600.0, self)

        self._controller: AgentController | None = None
        self._executor: ToolExecutor | None = None
        self._permissions: PermissionManager | None = None
        self._process_manager: ProcessManager | None = None
        self._task_id = ""
        self._paused = False
        self._chat_worker: ChatWorker | None = None
        self._manual_worker: TaskWorker | None = None
        self._chat_text = ""
        self._chat_bubble: MessageBubble | None = None
        self._bubbles: list[QWidget] = []

        self._build()
        self._wire()
        self._load_settings()
        self.refresh()

    # -- construction ------------------------------------------------------
    def _build(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 14, 20, 8)
        header_layout.setSpacing(10)
        title = QLabel("Hybrid Agent")
        title.setProperty("variant", "title")
        header_layout.addWidget(title)

        self.mode_combo = QComboBox()
        for key, label in MODE_LABELS:
            self.mode_combo.addItem(ja(label), key)
        self.mode_combo.setFixedWidth(190)
        header_layout.addWidget(self.mode_combo)

        self.mode_hint = muted_label("")
        self.mode_hint.setWordWrap(False)
        header_layout.addWidget(self.mode_hint, 1)

        self.state_label = muted_label("待機中")
        self.state_label.setProperty("variant", "chip")
        header_layout.addWidget(self.state_label)

        self.start_button = QPushButton(ja("Start"))
        self.start_button.setProperty("variant", "primary")
        self.start_button.setCursor(Qt.PointingHandCursor)
        header_layout.addWidget(self.start_button)

        self.pause_button = QPushButton(ja("Pause"))
        self.resume_button = QPushButton(ja("Resume"))
        self.stop_button = QPushButton(ja("Stop"))
        self.stop_button.setProperty("variant", "danger")
        for button in (self.pause_button, self.resume_button, self.stop_button):
            button.setCursor(Qt.PointingHandCursor)
            header_layout.addWidget(button)
        root.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(10)

        self.task_panel = TaskListPanel()
        self.task_panel.setMinimumWidth(220)
        self.task_panel.setMaximumWidth(300)
        splitter.addWidget(self.task_panel)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(16, 10, 16, 10)
        center_layout.setSpacing(12)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_chat_tab(), ja("Agent Chat"))
        self.plan_panel = PlanPanel()
        self.tabs.addTab(self.plan_panel, ja("Task Plan"))

        activity_tabs = QTabWidget()
        activity_tabs.setDocumentMode(True)
        activity_tabs.setTabPosition(QTabWidget.North)
        self.tool_log = EventLogPanel("ツールの実行結果がここに表示されます。")
        activity_tabs.addTab(self.tool_log, ja("Tools"))
        self.file_log = EventLogPanel("ファイルの変更がここに表示されます。")
        activity_tabs.addTab(self.file_log, ja("Files"))
        self.thought_log = EventLogPanel("作業方針と検証結果の要約が表示されます。")
        activity_tabs.addTab(self.thought_log, ja("Reasoning"))
        self.tabs.addTab(activity_tabs, ja("Activity"))

        self.terminal = TerminalPanel()
        self.tabs.addTab(self.terminal, ja("Terminal"))
        self.memory_page = MemoryPage(self.memory)
        self.tabs.addTab(self.memory_page, ja("Memory"))
        center_layout.addWidget(self.tabs, 1)
        splitter.addWidget(center)

        self.progress = ProgressPanel()
        self.progress.setMinimumWidth(250)
        self.progress.setMaximumWidth(320)
        splitter.addWidget(self.progress)
        splitter.setSizes([250, 780, 280])
        root.addWidget(splitter, 1)

        self._update_buttons()

    def _build_chat_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        self.chat_scroll = scroll
        container = QWidget()
        self.chat_layout = QVBoxLayout(container)
        self.chat_layout.setContentsMargins(4, 4, 4, 4)
        self.chat_layout.setSpacing(10)
        self.chat_layout.setAlignment(Qt.AlignTop)
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        self.empty_hint = muted_label(
            "目標を入力して Start を押すと、エージェントが計画を立てて作業を進めます。"
            "Chat Mode ではツールを実行せず、通常のチャットとして応答します。"
        )
        self.empty_hint.setStyleSheet("font-size: 8.5pt;")
        self.chat_layout.addWidget(self.empty_hint)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.goal_input = QLineEdit()
        self.goal_input.setPlaceholderText(ja("目標またはメッセージを入力…"))
        row.addWidget(self.goal_input, 1)
        self.send_button = QPushButton(ja("Send"))
        self.send_button.setProperty("variant", "primary")
        self.send_button.setCursor(Qt.PointingHandCursor)
        row.addWidget(self.send_button)
        layout.addLayout(row)
        return tab

    def _wire(self) -> None:
        self.bridge.event.connect(self._on_event)
        self.broker.requested.connect(self._on_approval_requested)
        self.output_received.connect(self.terminal.append_output)

        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        self.start_button.clicked.connect(self._primary_action)
        self.pause_button.clicked.connect(self._pause)
        self.resume_button.clicked.connect(self._resume)
        self.stop_button.clicked.connect(self._stop)
        self.send_button.clicked.connect(self._primary_action)
        self.goal_input.returnPressed.connect(self._primary_action)

        self.task_panel.new_task_requested.connect(self._new_task)
        self.task_panel.task_selected.connect(self._load_task)
        self.task_panel.memory_requested.connect(self._show_memory)
        self.task_panel.workspace_changed.connect(self._workspace_changed)
        self.task_panel.full_access_changed.connect(self._full_access_changed)

        self.terminal.command_submitted.connect(self._run_manual)
        self.terminal.stop_requested.connect(self._stop_processes)
        self.terminal.show_status_message.connect(self.show_status_message.emit)
        self.memory_page.show_status_message.connect(self.show_status_message.emit)

    # -- settings ----------------------------------------------------------
    def _load_settings(self) -> None:
        mode = str(self.settings.get("agent.default_mode", "autonomous"))
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        workspace = str(self.settings.get("agent.workspace", "")) or os.getcwd()
        self.task_panel.set_workspace(workspace)
        self.task_panel.set_full_access(bool(self.settings.get("agent.full_access", False)))
        self.progress.set_workspace(workspace)
        shell = str(self.settings.get("agent.default_shell", "cmd"))
        shell_index = self.terminal.shell_combo.findData(shell)
        if shell_index >= 0:
            self.terminal.shell_combo.setCurrentIndex(shell_index)
        self._mode_changed()

    def refresh(self) -> None:
        """Called when the page is shown and after every finished task."""
        try:
            tasks = self.store.list_tasks(limit=100)
        except Exception:
            tasks = []
        self.task_panel.refresh(tasks, self._task_id)
        self.task_panel.set_workspace(self._workspace())
        self.task_panel.set_full_access(bool(self.settings.get("agent.full_access", False)))
        self.progress.set_workspace(self._workspace())
        controller = self._controller
        if controller is not None:
            self.progress.set_limits(controller.limits())
        else:
            from app.agent.controller import AgentLimits

            self.progress.set_limits(AgentLimits())
        try:
            status = self.services.inference.status()
            ready = status.get("state") == "ready"
            name = status.get("display_name") or status.get("model_id") or "モデル未選択"
            self.progress.set_model_status(name if ready else "モデル未選択")
        except Exception:
            pass
        self._update_buttons()

    def update_inference_status(self, status: dict[str, Any]) -> None:
        ready = status.get("state") == "ready"
        name = status.get("display_name") or status.get("model_id") or ""
        self.progress.set_model_status(name if ready and name else "モデル未選択")

    # -- runtime -----------------------------------------------------------
    def _ensure_runtime(self) -> AgentController:
        if self._controller is not None:
            return self._controller
        try:
            timeout = float(self.settings.get("agent.command_timeout", 120))
        except (TypeError, ValueError):
            timeout = 120.0
        registry = create_registry()
        permissions = PermissionManager(self.store, self.settings)
        process_manager = ProcessManager(default_timeout=timeout)
        executor = ToolExecutor(
            registry=registry,
            permissions=permissions,
            store=self.store,
            memory=self.memory,
            process_manager=process_manager,
            settings=self.settings,
        )
        computer_use = ComputerUse(permissions, self._approve_computer_action)
        planner = Planner(ModelClient(self.services.inference, self.settings), registry)
        controller = AgentController(
            store=self.store,
            memory=self.memory,
            permissions=permissions,
            registry=registry,
            executor=executor,
            planner=planner,
            settings=self.settings,
            process_manager=process_manager,
            computer_use=computer_use,
            performance_monitor=self.services.performance,
        )
        controller.set_listener(self.bridge.event.emit)
        controller.set_broker(self.broker)
        self._controller = controller
        self._executor = executor
        self._permissions = permissions
        self._process_manager = process_manager
        self.progress.set_limits(controller.limits())
        return controller

    def _approve_computer_action(self, request):
        permissions = self._permissions
        if permissions is None:
            from app.agent.permissions import ApprovalOutcome
            return ApprovalOutcome("reject", request.fingerprint())
        decision = permissions.evaluate(request)
        if decision.decision != "ask":
            from app.agent.permissions import ApprovalOutcome
            return ApprovalOutcome("reject", request.fingerprint())
        return self.broker.request(ApprovalRequest(decision))

    def _workspace(self) -> str:
        text = str(self.settings.get("agent.workspace", "") or os.getcwd())
        try:
            Path(text).mkdir(parents=True, exist_ok=True)
        except OSError:
            return os.getcwd()
        return text

    # -- actions -----------------------------------------------------------
    def _primary_action(self) -> None:
        if str(self.mode_combo.currentData()) == "chat":
            self._send_chat()
        else:
            self._start_task()

    def _start_task(self) -> None:
        goal = self.goal_input.text().strip()
        if not goal:
            self.show_status_message.emit("目標を入力してください。")
            return
        preferred = str(self.settings.get("agent.default_model", "") or "")
        if preferred:
            try:
                loaded = str(self.services.inference.status().get("display_name") or "")
            except Exception:
                loaded = ""
            if loaded and loaded != preferred:
                self.show_status_message.emit(
                    f"既定モデル「{preferred}」ではなく「{loaded}」が読み込まれています。"
                )
        mode = str(self.mode_combo.currentData())
        controller = self._ensure_runtime()
        try:
            task = controller.start(goal, mode=mode, workspace=self._workspace())
        except Exception as exc:
            self.show_status_message.emit(str(exc))
            return
        self.goal_input.clear()
        self._task_id = task.id
        self._paused = False
        self._reset_views()
        self.plan_panel.clear()
        self.tool_log.clear()
        self.file_log.clear()
        self.thought_log.clear()
        self.terminal.clear_output()
        self._append_message("user", goal)
        self.tabs.setCurrentIndex(0)
        self.progress.set_task(task.to_dict())
        self.progress.set_state("running")
        self.show_status_message.emit("タスクを開始しました")
        self._update_buttons()
        self.task_panel.refresh(self.store.list_tasks(limit=100), self._task_id)

    def _pause(self) -> None:
        if self._controller is None or not self._controller.is_running:
            return
        self._controller.pause()
        self._paused = True
        self.progress.set_state("paused")
        self.state_label.setText(ja("一時停止中"))
        self._update_buttons()

    def _resume(self) -> None:
        controller = self._ensure_runtime()
        if controller.is_running:
            controller.resume()
        else:
            if not self._task_id:
                self.show_status_message.emit("再開するタスクを選択してください。")
                return
            status = self._task_status()
            if status not in {"paused", "awaiting_approval"}:
                self.show_status_message.emit(
                    "再開できるのは一時停止中のタスクだけです。新しい目標は Start で開始してください。"
                )
                return
            try:
                task = controller.resume_task(self._task_id)
            except Exception as exc:
                self.show_status_message.emit(str(exc))
                return
            self.progress.set_task(task.to_dict())
            self.show_status_message.emit("タスクを再開しました")
        self._paused = False
        self.progress.set_state("running")
        self._update_buttons()

    def _stop(self) -> None:
        if self._controller is not None:
            self._controller.stop()
        if self._process_manager is not None:
            self._process_manager.stop_all()
        self._paused = False
        self.progress.set_state("stopped")
        self.state_label.setText(ja("停止"))
        self._update_buttons()

    def _stop_processes(self) -> None:
        if self._process_manager is not None:
            count = self._process_manager.stop_all()
            self.terminal.append_info(f"{count} 個のプロセスを停止しました。")
        self.terminal.append_result(status="cancelled", cancelled=True)

    def _new_task(self) -> None:
        self._task_id = ""
        self._paused = False
        self.progress.set_task(None)
        self.plan_panel.clear()
        self.tool_log.clear()
        self.file_log.clear()
        self.thought_log.clear()
        self.terminal.clear_output()
        self._reset_views()
        self.goal_input.clear()
        self.goal_input.setFocus()
        self.state_label.setText(ja("待機中"))
        self._update_buttons()
        self.task_panel.refresh(self.store.list_tasks(limit=100), "")

    def _load_task(self, task_id: str) -> None:
        task = self.store.get_task(task_id)
        if task is None:
            return
        self._task_id = task.id
        self.progress.set_task(task.to_dict())
        self.progress.set_state(task.status)
        self.plan_panel.set_plan(
            {"steps": task.plan, "notes": ""}, int(task.current_step)
        )
        self.goal_input.setText(task.goal)
        index = self.mode_combo.findData(task.mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        self.terminal.clear_output()
        try:
            for record in self.store.list_commands(task.id, limit=100):
                self.terminal.append_command(
                    record.command, record.shell, record.cwd, record.source
                )
                if record.stdout:
                    self.terminal.append_output(record.stdout, "stdout")
                if record.stderr:
                    self.terminal.append_output(record.stderr, "stderr")
                self.terminal.append_result(
                    record.exit_code,
                    record.duration,
                    cancelled=record.status == "cancelled",
                    timed_out=record.status == "timeout",
                )
        except Exception:
            log.exception("could not restore command history")
        self._reset_views()
        try:
            for message in self.store.list_messages(task.id, limit=200):
                role = "user" if message.role == "user" else "assistant"
                self._append_message(role, message.content)
        except Exception:
            log.exception("could not restore task messages")
        self.progress.set_plan_progress(task.plan, int(task.current_step))
        self._paused = task.status == "paused"
        self._update_buttons()

    def _workspace_changed(self, path: str) -> None:
        try:
            Path(path).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self.show_status_message.emit(f"フォルダーを作成できませんでした: {exc}")
            return
        self.settings.set("agent.workspace", path)
        self.progress.set_workspace(path)
        self.show_status_message.emit("作業ディレクトリを変更しました")

    def _full_access_changed(self, enabled: bool) -> None:
        self.settings.set("agent.full_access", bool(enabled))
        self.show_status_message.emit(
            "Full Access Mode を有効にしました。インストールや管理者権限の操作は引き続き承認が必要です。"
            if enabled
            else "Full Access Mode を無効にしました。"
        )

    def _show_memory(self) -> None:
        self.memory_page.refresh()
        self.tabs.setCurrentWidget(self.memory_page)

    def _mode_changed(self) -> None:
        mode = str(self.mode_combo.currentData())
        hints = {
            "chat": "ツールは実行しません。モデルとの通常の会話になります。",
            "plan": "計画のみ作成します。ファイル変更やコマンド実行は行いません。",
            "build": "開発作業向け。危険な操作の前に承認を求めます。",
            "autonomous": "目標に向かって半自律的に実行します。危険な操作は承認が必要です。",
        }
        self.mode_hint.setText(ja(hints.get(mode, "")))
        self.start_button.setText(ja("Send" if mode == "chat" else "Start"))
        self.goal_input.setPlaceholderText(
            ja(
                "メッセージを入力…"
                if mode == "chat"
                else "目標を入力（例: このPythonプロジェクトを調査してエラーを直して）"
            )
        )
        self._update_buttons()

    def _update_buttons(self) -> None:
        running = bool(self._controller is not None and self._controller.is_running)
        mode = str(self.mode_combo.currentData())
        self.start_button.setEnabled(not running)
        self.start_button.setText(ja("Send" if mode == "chat" else "Start"))
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        resumable = running or (
            bool(self._task_id) and not running and self._task_status() in {"paused", "awaiting_approval"}
        )
        self.resume_button.setEnabled(resumable)
        self.state_label.setText(
            ja(
                "実行中"
                if running and not self._paused
                else "一時停止中"
                if running and self._paused
                else "待機中"
            )
        )

    def _task_status(self) -> str:
        if not self._task_id:
            return ""
        task = self.store.get_task(self._task_id)
        return task.status if task is not None else ""

    # -- events ------------------------------------------------------------
    def _on_event(self, event: dict[str, Any]) -> None:
        kind = event.get("type")
        if kind == "task":
            task = event.get("task") or {}
            self._task_id = str(task.get("id") or self._task_id)
            self.progress.set_task(task)
        elif kind == "state":
            state = str(event.get("state") or "idle")
            self.progress.set_state(state if state != "idle" else self._task_status())
            if state == "idle":
                self._paused = False
                self.refresh()
            self._update_buttons()
        elif kind == "plan":
            plan = event.get("plan") or {}
            current = int(event.get("current") or 0)
            self.plan_panel.set_plan(plan, current)
            self.progress.set_plan_progress(list(plan.get("steps") or []), current)
        elif kind == "thought":
            self.thought_log.append(str(event.get("text") or ""))
        elif kind == "verification":
            self.thought_log.append("検証: " + str(event.get("text") or ""))
        elif kind == "tool_start":
            tool = str(event.get("tool") or "")
            self.progress.set_active_tools([tool])
            self.state_label.setText(ja(f"実行中: {tool}"))
        elif kind == "tool_result":
            self.progress.set_active_tools([])
            tool = str(event.get("tool") or "")
            ok = bool(event.get("ok"))
            summary = str(event.get("summary") or "")
            self.tool_log.append(f"[{'OK' if ok else 'NG'}] {tool}: {summary}")
            data = event.get("data") or {}
            change = data.get("change") if isinstance(data, dict) else None
            if isinstance(change, dict):
                self.file_log.append(
                    f"{change.get('action', 'change')}: "
                    f"{change.get('relative') or change.get('path')}"
                )
        elif kind == "command":
            self._show_command(event.get("record") or {})
        elif kind == "output":
            if not bool(self.settings.get("agent.show_command_output", True)):
                return
            stream = str(event.get("stream") or "stdout")
            self.terminal.append_output(str(event.get("text") or ""), stream)
        elif kind == "finished":
            status = str(event.get("status") or "")
            summary = str(event.get("summary") or "")
            if summary:
                self._append_message("assistant", summary)
            self.progress.set_state(status)
            self.progress.set_pending(None)
            self.progress.set_active_tools([])
            finished_task = self.store.get_task(self._task_id) if self._task_id else None
            self.progress.set_task(finished_task.to_dict() if finished_task else None)
            self._paused = False
            self.refresh()
            label = task_status_label(status)
            self.show_status_message.emit(f"タスク{ja(label)}: {summary.splitlines()[0] if summary else ''}")
            if status in {"completed", "failed", "stopped"} and self._task_id:
                self.tabs.setCurrentIndex(0)

    def _show_command(self, record: dict[str, Any]) -> None:
        self.terminal.append_command(
            str(record.get("command") or ""),
            str(record.get("shell") or ""),
            str(record.get("cwd") or ""),
            str(record.get("source") or "agent"),
        )
        if bool(self.settings.get("agent.show_command_output", True)):
            if record.get("stdout"):
                self.terminal.append_output(str(record["stdout"]), "stdout")
            if record.get("stderr"):
                self.terminal.append_output(str(record["stderr"]), "stderr")
        self.terminal.append_result(
            record.get("exit_code"),
            float(record.get("duration") or 0),
            cancelled=bool(record.get("cancelled")),
            timed_out=bool(record.get("timed_out")),
        )

    # -- approvals ---------------------------------------------------------
    def _on_approval_requested(self, request: ApprovalRequest) -> None:
        self.progress.set_pending(f"{request.risk}: {request.target}"[:300])
        self.progress.set_state("awaiting_approval")
        self.state_label.setText(ja("承認待ち"))
        self.tabs.setCurrentWidget(self.terminal)
        dialog = ApprovalDialog(request, self)
        dialog.exec()
        outcome = dialog.outcome or ApprovalOutcome("reject", request.fingerprint)
        self.broker.resolve(outcome)
        self.progress.set_pending(None)
        if outcome.stop_task:
            self.progress.set_state("stopped")
        elif outcome.allowed:
            self.progress.set_state("running")
        else:
            self.progress.set_state("paused")
        self.thought_log.append(
            {
                "allow_once": "承認: Allow Once",
                "always_allow": "承認: Always Allow（保存しました）",
                "reject": "拒否: Reject",
                "stop": "停止: Stop Task",
            }.get(outcome.choice, outcome.choice)
        )
        self._update_buttons()

    # -- chat --------------------------------------------------------------
    def _send_chat(self) -> None:
        text = self.goal_input.text().strip()
        if not text:
            return
        if self._chat_worker is not None and self._chat_worker.isRunning():
            self.show_status_message.emit("応答を生成中です。")
            return
        try:
            ready = self.services.inference.is_ready
        except Exception:
            ready = False
        if not ready:
            self.show_status_message.emit("モデルが読み込まれていません。「モデル管理」で読み込んでください。")
            return
        self.goal_input.clear()
        self._append_message("user", text)
        self._chat_text = ""
        self._chat_bubble = self._append_message("assistant", "", streaming=True)
        params = GenerationParams.from_settings(self.settings.get("chat", {}) or {})
        try:
            model_label = str(self.services.inference.status().get("display_name") or "")
        except Exception:
            model_label = ""
        worker = ChatWorker(
            self.services.inference,
            [{"role": "user", "content": text}],
            params,
            system_prompt=str(self.settings.get("chat.system_prompt", "")),
            model_label=model_label,
            parent=self,
        )
        worker.delta.connect(self._chat_delta)
        worker.completed.connect(self._chat_completed)
        worker.failed.connect(self._chat_failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._chat_finished)
        self._chat_worker = worker
        self.state_label.setText(ja("生成中…"))
        worker.start()

    def _chat_delta(self, chunk: str) -> None:
        self._chat_text += chunk
        if self._chat_bubble is not None:
            self._chat_bubble.set_streaming_text(self._chat_text)
        self._scroll_chat()

    def _chat_completed(self, result: dict[str, Any]) -> None:
        text = str(result.get("content") or self._chat_text)
        if self._chat_bubble is not None:
            self._chat_bubble.finalize(text or "(空の応答)")
        self._chat_bubble = None
        self._chat_text = ""
        self._scroll_chat()

    def _chat_failed(self, message: str, detail: str) -> None:
        if self._chat_bubble is not None:
            self._chat_bubble.set_error(f"生成に失敗しました: {message}")
            self._chat_bubble.setToolTip(detail)
        self._chat_bubble = None
        self._chat_text = ""
        self.show_status_message.emit(f"生成に失敗しました: {message}")

    def _chat_finished(self) -> None:
        self._chat_worker = None
        self.state_label.setText(ja("待機中"))

    # -- manual terminal ---------------------------------------------------
    def _run_manual(self, command: str, shell: str) -> None:
        if self._manual_worker is not None and self._manual_worker.isRunning():
            self.show_status_message.emit("手動コマンドが実行中です。")
            return
        self._ensure_runtime()
        executor = self._executor
        if executor is None:
            self.show_status_message.emit("実行環境を準備できませんでした。")
            return
        actual = command.strip()
        if shell == "git" and not actual.lower().startswith("git "):
            actual = f"git {actual}"
        tool = MANUAL_TOOL.get(shell, "execute_cmd")
        call = ToolCall(tool, {"command": actual, "cwd": self._workspace()})
        context = ToolContext(
            workspace=self._workspace(),
            task_id=self._task_id,
            mode="autonomous",
            full_access=bool(self.settings.get("agent.full_access", False)),
            source="user",
            store=self.store,
            memory=self.memory,
            process_manager=self._process_manager,
            emit=self._manual_emit,
            timeout=float(self.settings.get("agent.command_timeout", 120)),
            save_command_history=bool(self.settings.get("agent.save_command_history", True)),
        )
        self.terminal.append_command(actual, shell, self._workspace(), source="user")

        def work() -> ToolResult:
            return executor.execute(call, context, broker=self.broker, mode="autonomous")

        worker = TaskWorker(work, self)
        worker.succeeded.connect(self._manual_done)
        worker.failed.connect(self._manual_failed)
        worker.finished.connect(worker.deleteLater)
        worker.finished.connect(self._manual_finished)
        self._manual_worker = worker
        worker.start()

    def _manual_emit(self, stream: str, text: str) -> None:
        self.output_received.emit(stream, text)

    def _manual_done(self, result: ToolResult) -> None:
        data = result.data or {}
        if data.get("timed_out") or data.get("cancelled"):
            self.terminal.append_result(
                data.get("exit_code"),
                float(data.get("duration") or 0),
                cancelled=bool(data.get("cancelled")),
                timed_out=bool(data.get("timed_out")),
            )
        elif data.get("exit_code") is not None or data.get("command"):
            self.terminal.append_result(
                data.get("exit_code"), float(data.get("duration") or 0)
            )
        if not result.ok and (result.rejected or result.denied or result.stopped):
            self.terminal.append_info(
                result.summary or result.error or "実行されませんでした。"
            )

    def _manual_failed(self, message: str, detail: str) -> None:
        self.terminal.append_info(f"エラー: {message}")
        log.warning("manual command failed: %s (%s)", message, detail)

    def _manual_finished(self) -> None:
        self._manual_worker = None

    # -- chat transcript ---------------------------------------------------
    def _reset_views(self) -> None:
        for bubble in self._bubbles:
            self.chat_layout.removeWidget(bubble)
            bubble.deleteLater()
        self._bubbles.clear()
        self.chat_layout.addWidget(self.empty_hint)
        self.empty_hint.show()

    def _append_message(
        self, role: str, text: str, *, streaming: bool = False
    ) -> MessageBubble:
        self.empty_hint.hide()
        bubble = MessageBubble(role, text, streaming=streaming)
        self.chat_layout.addWidget(bubble)
        self._bubbles.append(bubble)
        QTimer.singleShot(0, self._scroll_chat)
        return bubble

    def _scroll_chat(self) -> None:
        bar = self.chat_scroll.verticalScrollBar()
        bar.setValue(bar.maximum())

    # -- shutdown ----------------------------------------------------------
    def shutdown(self) -> None:
        if self._controller is not None:
            try:
                self._controller.shutdown()
            except Exception:
                log.exception("could not shut down the agent controller")
        for worker in (self._chat_worker, self._manual_worker):
            try:
                if worker is not None and worker.isRunning():
                    worker.wait(5000)
            except RuntimeError:
                continue
        if self._process_manager is not None:
            try:
                self._process_manager.stop_all()
            except Exception:
                pass


__all__ = ["AgentBridge", "AgentPage", "QtApprovalBroker"]
