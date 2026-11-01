"""Tests for the Hybrid Agent core (permissions, tools, memory, persistence, loop).

These tests never touch real system settings: commands only run small Python
snippets, and every file operation happens inside pytest's temporary folders.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from app.agent import command_classifier as cc
from app.agent.controller import AgentController
from app.agent.executor import ToolExecutor
from app.agent.memory import MemoryRejected, MemoryStore, looks_secret
from app.agent.permissions import (
    ActionRequest,
    ApprovalOutcome,
    PermissionEscalationError,
    PermissionManager,
    PermissionView,
)
from app.agent.planner import Plan, PlanStep
from app.agent.process_manager import ProcessManager
from app.agent.sessions import AgentStore
from app.agent.subagents import SUBAGENT_NAMES, build_subagents
from app.agent.tools.registry import ToolCall, ToolContext, ToolRegistry, ToolSpec
from app.agent.validator import (
    ToolValidationError,
    extract_json_object,
    parse_model_action,
    validate_plan,
    validate_tool_call,
)
from app.core import paths as app_paths


# -- fixtures --------------------------------------------------------------
@pytest.fixture()
def store(tmp_path: Path):
    db = AgentStore(tmp_path / "agent.db")
    yield db
    db.close()


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    path = tmp_path / "work"
    path.mkdir()
    return path


@pytest.fixture()
def permissions(store, settings, workspace: Path) -> PermissionManager:
    settings.set("agent.workspace", str(workspace))
    return PermissionManager(store, settings)


class Broker:
    """Stand-in for the Qt approval dialog."""

    def __init__(self, choice: str = "allow_once"):
        self.choice = choice
        self.requests: list = []

    def request(self, request):
        self.requests.append(request)
        return ApprovalOutcome(self.choice, request.fingerprint)


class FakeModel:
    available = True

    def label(self) -> str:
        return "fake-model"


class FakePlanner:
    """Deterministic stand-in for the model-backed planner."""

    def __init__(self, actions: list[dict] | None = None):
        self.actions = list(actions or [])
        self.model = FakeModel()
        self.next_action_calls = 0
        self.plan_calls = 0

    def make_plan(self, goal, mode, *, workspace="", memory_context=""):
        self.plan_calls += 1
        return Plan(
            goal=goal,
            mode=mode,
            steps=[PlanStep(0, "調査する"), PlanStep(1, "実行する")],
        )

    def next_action(self, **_kwargs):
        self.next_action_calls += 1
        if self.actions:
            return self.actions.pop(0)
        return {"type": "finish", "summary": "完了しました。"}

    def ask_role(self, system_prompt, prompt, *, max_tokens=400):
        return ""


def build_registry() -> tuple[ToolRegistry, list[tuple[dict, str]]]:
    registry = ToolRegistry()
    executed: list[tuple[dict, str]] = []

    def handler(arguments, context):
        executed.append((arguments, context.approved_command))
        return {
            "ok": True,
            "summary": "done",
            "data": {"command": context.approved_command},
        }

    registry.register(
        ToolSpec(
            name="write_note",
            description="Write a note.",
            kind="write",
            parameters={"path": {"type": "string", "required": True}},
            modes=("build", "autonomous"),
        ),
        handler,
    )
    registry.register(
        ToolSpec(
            name="say",
            description="Run a command.",
            kind="command",
            parameters={"command": {"type": "string", "required": True}},
            command_builder=lambda arguments, context: (
                arguments["command"],
                "cmd",
                context.workspace,
            ),
        ),
        handler,
    )
    return registry, executed


def build_executor(store, settings, workspace: Path):
    registry, executed = build_registry()
    manager = PermissionManager(store, settings)
    settings.update({"agent": {"workspace": str(workspace)}}, save=False)
    executor = ToolExecutor(
        registry=registry,
        permissions=manager,
        store=store,
        memory=MemoryStore(store),
        process_manager=ProcessManager(default_timeout=10),
        settings=settings,
    )
    return executor, manager, executed, registry


def context_for(workspace: Path, **overrides) -> ToolContext:
    values = dict(
        workspace=str(workspace),
        task_id="",
        mode="autonomous",
        source="agent",
        timeout=10,
    )
    values.update(overrides)
    return ToolContext(**values)


# -- command classification ------------------------------------------------
@pytest.mark.parametrize(
    "command",
    ["dir", "git status", "python --version", "where python", "Get-ChildItem"],
)
def test_read_only_commands_are_allowed(command: str, workspace: Path) -> None:
    result = cc.classify_command(
        command, shell="cmd", cwd=str(workspace), allowed_roots=(str(workspace),)
    )
    assert result.decision == cc.ALLOW, (command, result.reason)


@pytest.mark.parametrize(
    "command",
    [
        "pip install requests",
        "npm install left-pad",
        "winget install Git.Git",
        "curl https://example.com/x.exe -o x.exe",
    ],
)
def test_install_and_download_commands_ask(command: str, workspace: Path) -> None:
    result = cc.classify_command(
        command, shell="cmd", cwd=str(workspace), allowed_roots=(str(workspace),)
    )
    assert result.decision == cc.ASK
    assert result.grantable is False


@pytest.mark.parametrize(
    "command",
    [
        "Set-MpPreference -DisableRealtimeMonitoring $true",
        "reg save HKLM\\SAM sam.hiv",
        "powershell -enc SQBFAFgAIAAoAE4AZQB3AC0ATwBiAGoAZQBjAHQA",
        "Invoke-Expression $payload",
    ],
)
def test_dangerous_and_opaque_commands_are_denied(command: str, workspace: Path) -> None:
    result = cc.classify_command(command, shell="powershell", cwd=str(workspace))
    assert result.decision == cc.DENY, (command, result.reason)


def test_chained_commands_take_the_strictest_segment(workspace: Path) -> None:
    result = cc.classify_command(
        "dir && del C:\\Windows\\System32\\drivers\\etc\\hosts",
        shell="cmd",
        cwd=str(workspace),
        allowed_roots=(str(workspace),),
    )
    assert result.decision == cc.DENY


def test_unknown_commands_are_not_auto_run(workspace: Path) -> None:
    result = cc.classify_command("frobnicate --danger", shell="cmd", cwd=str(workspace))
    assert result.decision == cc.ASK
    assert result.grantable is False


def test_delete_inside_workspace_is_allowed_but_recursive_outside_asks(
    workspace: Path, tmp_path: Path
) -> None:
    inside = cc.classify_command(
        f"del {workspace / 'old.txt'}",
        shell="cmd",
        cwd=str(workspace),
        allowed_roots=(str(workspace),),
    )
    assert inside.decision == cc.ALLOW
    outside = cc.classify_command(
        f"rmdir /s {tmp_path / 'other'}",
        shell="cmd",
        cwd=str(workspace),
        allowed_roots=(str(workspace),),
    )
    assert outside.decision == cc.ASK
    assert outside.grantable is False


# -- permission manager ----------------------------------------------------
def test_file_write_inside_workspace_is_allowed(permissions, workspace: Path) -> None:
    decision = permissions.evaluate(
        ActionRequest(kind="write", paths=(str(workspace / "app.py"),), cwd=str(workspace))
    )
    assert decision.decision == cc.ALLOW
    assert decision.grantable is False


def test_write_outside_workspace_asks_and_grant_does_not_leak(
    permissions, workspace: Path, tmp_path: Path
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    request = ActionRequest(
        kind="write",
        paths=(str(outside / "notes.txt"),),
        cwd=str(workspace),
    )
    decision = permissions.evaluate(request)
    assert decision.decision == cc.ASK
    assert decision.grantable is True

    with pytest.raises(PermissionEscalationError):
        permissions.add_grant(decision, origin="agent")

    grant = permissions.add_grant(decision, origin="user")
    assert grant is not None
    assert permissions.evaluate(request).decision == cc.ALLOW

    other = tmp_path / "other"
    other.mkdir()
    other_request = ActionRequest(
        kind="write", paths=(str(other / "x.txt"),), cwd=str(workspace)
    )
    assert permissions.evaluate(other_request).decision == cc.ASK


def test_installation_is_never_persistently_grantable(permissions, workspace: Path) -> None:
    decision = permissions.evaluate(
        ActionRequest(kind="command", command="pip install requests", shell="cmd", cwd=str(workspace))
    )
    assert decision.decision == cc.ASK
    assert decision.category == "install"
    assert decision.grantable is False
    with pytest.raises(PermissionEscalationError):
        permissions.add_grant(decision, origin="user")


def test_full_access_still_requires_approval_for_install_and_delete(
    permissions, settings, workspace: Path, tmp_path: Path
) -> None:
    settings.set("agent.full_access", True)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    write = permissions.evaluate(
        ActionRequest(kind="write", paths=(str(outside / "a.txt"),), cwd=str(workspace))
    )
    assert write.decision == cc.ALLOW
    delete = permissions.evaluate(
        ActionRequest(kind="delete", paths=(str(outside / "a.txt"),), cwd=str(workspace))
    )
    assert delete.decision == cc.ASK
    install = permissions.evaluate(
        ActionRequest(kind="command", command="pip install requests", shell="cmd", cwd=str(workspace))
    )
    assert install.decision == cc.ASK
    assert install.grantable is False


def test_application_settings_and_secrets_are_protected(permissions, workspace: Path) -> None:
    protected = app_paths.config_dir() / "settings.json"
    decision = permissions.evaluate(
        ActionRequest(kind="write", paths=(str(protected),), cwd=str(workspace))
    )
    assert decision.decision == cc.DENY


def test_command_execution_can_be_disabled(permissions, settings, workspace: Path) -> None:
    settings.set("agent.command_execution", False)
    decision = permissions.evaluate(
        ActionRequest(kind="command", command="dir", shell="cmd", cwd=str(workspace))
    )
    assert decision.decision == cc.DENY


# -- persistence -----------------------------------------------------------
def test_task_persistence_and_interrupted_tasks_pause(store) -> None:
    task = store.create_task("fix the bug", mode="autonomous", workspace="C:\\work")
    store.update_task(
        task.id, plan=[{"index": 0, "title": "a", "status": "pending"}], current_step=0
    )
    store.add_command(
        shell="cmd", command="dir", cwd="C:\\work", task_id=task.id, exit_code=0, stdout="x"
    )
    store.add_approval(decision="allow_once", task_id=task.id, command="dir", fingerprint="f")

    loaded = store.get_task(task.id)
    assert loaded is not None
    assert loaded.goal == "fix the bug"
    assert loaded.plan[0]["title"] == "a"
    assert store.list_commands(task.id)[0].command == "dir"
    assert store.list_approvals(task.id)[0].decision == "allow_once"

    store.update_task(task.id, status="running")
    assert store.recover_interrupted() == 1
    assert store.get_task(task.id).status == "paused"


def test_stored_grants_are_listed_and_revoked(store, permissions, workspace: Path, tmp_path: Path) -> None:
    outside = tmp_path / "granted"
    outside.mkdir()
    decision = permissions.evaluate(
        ActionRequest(kind="write", paths=(str(outside / "a.txt"),), cwd=str(workspace))
    )
    permissions.add_grant(decision, origin="user")
    grants = store.list_grants()
    assert len(grants) == 1
    with pytest.raises(PermissionEscalationError):
        permissions.revoke_grant(grants[0].id, origin="agent")
    permissions.revoke_grant(grants[0].id, origin="user")
    assert store.list_grants() == []


# -- memory ----------------------------------------------------------------
def test_memory_rejects_credentials_and_supports_crud(store) -> None:
    memory = MemoryStore(store)
    assert looks_secret("password = hunter2")
    with pytest.raises(MemoryRejected):
        memory.save("task", "api key", "sk-abcdefghijklmnopqrstuvwx")
    record = memory.save("project", "test command", "python -m pytest -q")
    assert memory.search("pytest")[0].id == record.id
    updated = memory.update(record.id, value="python -m pytest tests -q")
    assert updated.value == "python -m pytest tests -q"
    memory.delete(record.id)
    assert memory.search("pytest") == []


def test_memory_rejects_unknown_categories(store) -> None:
    memory = MemoryStore(store)
    with pytest.raises(MemoryRejected):
        memory.save("nonsense", "k", "v")


# -- process manager -------------------------------------------------------
def test_python_shell_runs_and_reports_output() -> None:
    manager = ProcessManager(default_timeout=30)
    result = manager.run("print('hello from agent')", shell="python", timeout=30)
    assert result.ok
    assert "hello from agent" in result.stdout
    assert manager.running() == 0


def test_timeout_terminates_the_process() -> None:
    manager = ProcessManager(default_timeout=30)
    started = time.monotonic()
    result = manager.run("import time\ntime.sleep(30)", shell="python", timeout=1)
    assert result.timed_out is True
    assert result.ok is False
    assert time.monotonic() - started < 25


# -- validator -------------------------------------------------------------
def test_tool_call_validation_rejects_bad_model_output() -> None:
    registry, _executed = build_registry()
    call = validate_tool_call(
        {"tool": "write_note", "arguments": {"path": "a.txt"}}, registry
    )
    assert call.tool == "write_note"
    with pytest.raises(ToolValidationError):
        validate_tool_call({"tool": "format_disk", "arguments": {}}, registry)
    with pytest.raises(ToolValidationError):
        validate_tool_call({"tool": "write_note", "arguments": {}}, registry)
    with pytest.raises(ToolValidationError):
        validate_tool_call(
            {"tool": "write_note", "arguments": {"path": "a", "mode": "x"}}, registry
        )
    with pytest.raises(ToolValidationError):
        validate_tool_call(
            {"tool": "write_note", "arguments": {"path": "a"}}, registry, mode="plan"
        )


def test_model_json_and_plans_are_parsed() -> None:
    registry, _executed = build_registry()
    assert extract_json_object('note\n```json\n{"a": 1}\n```\nend')["a"] == 1
    assert parse_model_action('{"finish": true, "summary": "ok"}', registry)["type"] == "finish"
    action = parse_model_action(
        '{"tool": "say", "arguments": {"command": "echo hi"}, "thought": "probe"}',
        registry,
    )
    assert action["type"] == "tool"
    assert action["call"].tool == "say"
    plan = validate_plan({"steps": ["first", {"title": "second"}]})
    assert [item["title"] for item in plan] == ["first", "second"]


# -- executor --------------------------------------------------------------
def test_executor_runs_the_approved_command(store, settings, workspace: Path) -> None:
    executor, _manager, executed, _registry = build_executor(store, settings, workspace)
    result = executor.execute(
        ToolCall("say", {"command": "echo hello"}),
        context_for(workspace),
        broker=Broker(),
        mode="autonomous",
    )
    assert result.ok
    assert executed == [({"command": "echo hello"}, "echo hello")]


def test_rejected_action_is_not_executed(store, settings, workspace: Path, tmp_path: Path) -> None:
    executor, _manager, executed, _registry = build_executor(store, settings, workspace)
    outside = tmp_path / "rejected"
    outside.mkdir()
    result = executor.execute(
        ToolCall("write_note", {"path": str(outside / "a.txt")}),
        context_for(workspace),
        broker=Broker("reject"),
        mode="autonomous",
    )
    assert result.rejected is True
    assert result.ok is False
    assert executed == []
    assert store.list_approvals()[-1].decision == "reject"


def test_always_allow_is_stored_and_reused(
    store, settings, workspace: Path, tmp_path: Path
) -> None:
    executor, manager, executed, _registry = build_executor(store, settings, workspace)
    outside = tmp_path / "granted"
    outside.mkdir()
    call = ToolCall("write_note", {"path": str(outside / "a.txt")})
    first = executor.execute(call, context_for(workspace), broker=Broker("always_allow"), mode="autonomous")
    assert first.ok
    assert manager.list_grants()
    second = executor.execute(call, context_for(workspace), broker=None, mode="autonomous")
    assert second.ok
    assert len(executed) == 2


def test_plan_mode_blocks_write_tools(store, settings, workspace: Path, tmp_path: Path) -> None:
    executor, _manager, executed, _registry = build_executor(store, settings, workspace)
    result = executor.execute(
        ToolCall("write_note", {"path": str(tmp_path / "x.txt")}),
        context_for(workspace),
        broker=Broker(),
        mode="plan",
    )
    assert result.denied is True
    assert executed == []


# -- sub-agents ------------------------------------------------------------
def test_subagents_share_a_read_only_permission_view(store, settings, workspace: Path) -> None:
    manager = PermissionManager(store, settings)
    agents = build_subagents(FakePlanner(), manager)
    assert set(agents) == set(SUBAGENT_NAMES)
    assert len(agents) == 7
    for agent in agents.values():
        assert isinstance(agent.permissions, PermissionView)
        with pytest.raises(PermissionEscalationError):
            agent.permissions.add_grant(None)
        with pytest.raises(PermissionEscalationError):
            agent.permissions.escalate()
        with pytest.raises(PermissionEscalationError):
            agent.escalate()
    decision = agents["explorer"].evaluate(
        ActionRequest(kind="read", paths=(str(workspace / "a.py"),), cwd=str(workspace))
    )
    assert decision.decision == cc.ALLOW


# -- controller loop -------------------------------------------------------
def test_agent_loop_completes_and_does_not_block_the_caller(
    store, settings, workspace: Path
) -> None:
    executor, manager, executed, registry = build_executor(store, settings, workspace)
    planner = FakePlanner(
        [
            {
                "type": "tool",
                "call": ToolCall("say", {"command": "echo hello"}),
                "thought": "checking",
            }
        ]
    )
    controller = AgentController(
        store=store,
        memory=MemoryStore(store),
        permissions=manager,
        registry=registry,
        executor=executor,
        planner=planner,
        settings=settings,
        process_manager=ProcessManager(default_timeout=10),
    )
    events: list[dict] = []
    controller.set_listener(events.append)
    controller.set_broker(Broker())

    started_at = time.monotonic()
    task = controller.start("say hello", mode="autonomous", workspace=str(workspace))
    assert time.monotonic() - started_at < 5
    assert controller.wait(30)
    stored = store.get_task(task.id)
    assert stored is not None
    assert stored.status == "completed"
    assert any(event.get("type") == "finished" for event in events)
    assert any(event.get("type") == "plan" for event in events)
    assert controller.state == "idle"
    assert len(executed) == 1


def test_plan_mode_only_creates_a_plan(store, settings, workspace: Path) -> None:
    executor, manager, _executed, registry = build_executor(store, settings, workspace)
    planner = FakePlanner([])
    controller = AgentController(
        store=store,
        memory=MemoryStore(store),
        permissions=manager,
        registry=registry,
        executor=executor,
        planner=planner,
        settings=settings,
        process_manager=ProcessManager(default_timeout=10),
    )
    controller.set_listener(lambda _event: None)
    task = controller.start("plan the work", mode="plan", workspace=str(workspace))
    assert controller.wait(20)
    stored = store.get_task(task.id)
    assert stored.status == "completed"
    assert "作業計画" in stored.summary
    assert planner.next_action_calls == 0


def test_step_limit_stops_the_loop(store, settings, workspace: Path) -> None:
    settings.set("agent.max_steps", 2)
    executor, manager, _executed, registry = build_executor(store, settings, workspace)
    repeated = [
        {"type": "tool", "call": ToolCall("say", {"command": "echo loop"}), "thought": ""}
        for _ in range(6)
    ]
    planner = FakePlanner(repeated)
    controller = AgentController(
        store=store,
        memory=MemoryStore(store),
        permissions=manager,
        registry=registry,
        executor=executor,
        planner=planner,
        settings=settings,
        process_manager=ProcessManager(default_timeout=10),
    )
    controller.set_listener(lambda _event: None)
    task = controller.start("loop forever", mode="autonomous", workspace=str(workspace))
    assert controller.wait(30)
    stored = store.get_task(task.id)
    assert stored.status == "failed"
    assert "最大ステップ数" in stored.summary


def test_rejection_stops_the_task_without_substitution(
    store, settings, workspace: Path, tmp_path: Path
) -> None:
    executor, manager, executed, registry = build_executor(store, settings, workspace)
    outside = tmp_path / "denied"
    outside.mkdir()
    planner = FakePlanner(
        [
            {
                "type": "tool",
                "call": ToolCall("write_note", {"path": str(outside / "a.txt")}),
                "thought": "writing",
            },
            {
                "type": "tool",
                "call": ToolCall("say", {"command": "echo should-not-run"}),
                "thought": "retry differently",
            },
        ]
    )
    controller = AgentController(
        store=store,
        memory=MemoryStore(store),
        permissions=manager,
        registry=registry,
        executor=executor,
        planner=planner,
        settings=settings,
        process_manager=ProcessManager(default_timeout=10),
    )
    controller.set_listener(lambda _event: None)
    controller.set_broker(Broker("reject"))
    task = controller.start("write outside", mode="autonomous", workspace=str(workspace))
    assert controller.wait(30)
    stored = store.get_task(task.id)
    assert stored.status == "stopped"
    assert "拒否" in stored.summary
    assert executed == []


def test_resume_is_required_after_restart(store, settings, workspace: Path) -> None:
    executor, manager, _executed, registry = build_executor(store, settings, workspace)
    task = store.create_task("interrupted work", mode="autonomous", workspace=str(workspace))
    store.update_task(task.id, status="running", plan=[{"index": 0, "title": "a", "status": "pending"}])
    assert store.recover_interrupted() == 1
    assert store.get_task(task.id).status == "paused"
    controller = AgentController(
        store=store,
        memory=MemoryStore(store),
        permissions=manager,
        registry=registry,
        executor=executor,
        planner=FakePlanner([]),
        settings=settings,
        process_manager=ProcessManager(default_timeout=10),
    )
    controller.set_listener(lambda _event: None)
    assert controller.is_running is False
    resumed = controller.resume_task(task.id)
    assert resumed.id == task.id
    assert controller.wait(20)
    assert store.get_task(task.id).status == "completed"
