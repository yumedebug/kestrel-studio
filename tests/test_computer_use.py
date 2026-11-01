from __future__ import annotations

from pathlib import Path

from app.agent.permissions import ActionRequest, PermissionManager
from app.agent.tools import create_registry
from app.core.performance import PerformanceMonitor
from app.agent.sessions import AgentStore


def test_computer_use_always_requires_individual_approval(tmp_path: Path, settings) -> None:
    store = AgentStore(tmp_path / "agent.db")
    try:
        manager = PermissionManager(store, settings)
        decision = manager.evaluate(
            ActionRequest(kind="computer", description="click", source="user")
        )
        assert decision.decision == "ask"
        assert decision.category == "computer"
        assert decision.grantable is False
    finally:
        store.close()


def test_performance_monitor_returns_stable_snapshot() -> None:
    snapshot = PerformanceMonitor().sample()
    values = snapshot.as_dict()
    assert values["timestamp"] > 0
    assert set(values) == {
        "timestamp",
        "cpu_percent",
        "memory_percent",
        "memory_used_bytes",
        "memory_total_bytes",
        "disk_free_bytes",
        "disk_total_bytes",
        "process_count",
        "gpu_memory_used_bytes",
        "gpu_memory_total_bytes",
        "model_memory_bytes",
        "inference_tokens_per_second",
        "memory_budget_bytes",
        "memory_profile",
    }


def test_computer_tools_are_registered() -> None:
    registry = create_registry()
    assert {
        "computer_observe",
        "computer_click_element",
        "computer_click",
        "computer_type",
        "performance_sample",
    }.issubset(set(registry.names()))


def test_computer_and_performance_history_is_persistent(tmp_path: Path) -> None:
    store = AgentStore(tmp_path / "agent.db")
    try:
        store.add_computer_action(
            task_id="task", action="click", target="10,20", approved=True, verified=True
        )
        assert store.list_computer_actions("task")[0]["verified"] is True
        store.add_performance_sample({"cpu_percent": 12.5}, task_id="task")
        assert store.list_performance_samples(1)[0]["cpu_percent"] == 12.5
        store.clear_computer_actions()
        store.clear_performance_samples()
        assert store.list_computer_actions() == []
        assert store.list_performance_samples() == []
    finally:
        store.close()
