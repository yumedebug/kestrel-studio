"""Registry handlers for permission-gated Computer Use and host metrics."""

from __future__ import annotations

from typing import Any

from app.agent.computer_use import ComputerUseError, ComputerUseLoop
from app.agent.tools.registry import ToolContext, ToolError, ToolRegistry, ToolSpec


def _computer(context: ToolContext):
    if context.computer_use is None:
        raise ToolError("Computer Use is not available in this session.")
    return context.computer_use


def observe(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        observation = _computer(context).observe(int(arguments["hwnd"]) if arguments.get("hwnd") else None)
    except (ComputerUseError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    data = observation.to_dict()
    if context.store is not None:
        context.store.add_computer_action(
            task_id=context.task_id, action="observe", target=observation.window_title,
            approved=True, verified=True, details=data,
        )
    return {"ok": True, "summary": f"Observed {len(observation.elements)} UI element(s).", "data": data}


def click_element(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        observation = _computer(context).click_element(
            name=str(arguments.get("name") or ""),
            automation_id=str(arguments.get("automation_id") or ""),
        )
    except ComputerUseError as exc:
        raise ToolError(str(exc)) from exc
    data = observation.to_dict()
    if context.store is not None:
        context.store.add_computer_action(
            task_id=context.task_id, action="click_element",
            target=str(arguments.get("name") or arguments.get("automation_id") or ""),
            approved=True, verified=True, details=data,
        )
    return {"ok": True, "summary": "UI element clicked and re-observed.", "data": data}


def click_coordinate(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        computer = _computer(context)
        baseline = computer.observe()
        loop = ComputerUseLoop(computer, max_retries=2)
        observation = loop.run(
            lambda: (
                computer.click(
                    int(arguments["x"]),
                    int(arguments["y"]),
                    str(arguments.get("button") or "left"),
                ),
                computer.observe(),
            )[1],
            lambda current: current.to_dict() != baseline.to_dict(),
        )
    except (ComputerUseError, KeyError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    data = {**observation.to_dict(), "verification": loop.history}
    if context.store is not None:
        context.store.add_computer_action(
            task_id=context.task_id, action="click",
            target=f"{arguments['x']},{arguments['y']}",
            approved=True, verified=True, details=data,
        )
    return {
        "ok": True,
        "summary": "Coordinate clicked and verified through bounded observe/retry.",
        "data": data,
    }


def type_text(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    try:
        computer = _computer(context)
        computer.type_text(str(arguments.get("text") or ""), float(arguments.get("interval") or 0))
        observation = computer.observe()
    except (ComputerUseError, ValueError) as exc:
        raise ToolError(str(exc)) from exc
    data = observation.to_dict()
    if context.store is not None:
        context.store.add_computer_action(
            task_id=context.task_id, action="type", target=str(arguments.get("text") or "")[:120],
            approved=True, verified=True, details=data,
        )
    return {"ok": True, "summary": "Text entered and UI re-observed.", "data": data}


def performance_sample(arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
    if context.performance_monitor is None:
        raise ToolError("Performance monitoring is not available in this session.")
    snapshot = context.performance_monitor.sample(store=context.store, task_id=context.task_id)
    return {"ok": True, "summary": "Read-only performance sample collected.", "data": snapshot.as_dict()}


def register(registry: ToolRegistry) -> None:
    registry.register(
        ToolSpec(
            name="computer_observe",
            description="Observe the active Windows UI and return screenshot-independent UI metadata.",
            kind="computer",
            parameters={"hwnd": {"type": "integer", "description": "Optional window handle."}},
            modes=("build", "autonomous"),
        ),
        observe,
    )
    registry.register(
        ToolSpec(
            name="computer_click_element",
            description="Click an enabled Windows UI Automation element by name or automation id.",
            kind="computer",
            parameters={
                "name": {"type": "string", "description": "UI element name."},
                "automation_id": {"type": "string", "description": "UI Automation id."},
            },
            modes=("build", "autonomous"),
        ),
        click_element,
    )
    registry.register(
        ToolSpec(
            name="computer_click",
            description="Click a Windows screen coordinate.",
            kind="computer",
            parameters={
                "x": {"type": "integer", "required": True},
                "y": {"type": "integer", "required": True},
                "button": {"type": "string", "description": "left, right or middle."},
            },
            modes=("build", "autonomous"),
        ),
        click_coordinate,
    )
    registry.register(
        ToolSpec(
            name="computer_type",
            description="Type approved text into the focused Windows control.",
            kind="computer",
            parameters={"text": {"type": "string", "required": True}, "interval": {"type": "number"}},
            modes=("build", "autonomous"),
        ),
        type_text,
    )
    registry.register(
        ToolSpec(
            name="performance_sample",
            description="Collect a read-only CPU, memory, disk, process and inference performance sample.",
            kind="performance",
            parameters={},
        ),
        performance_sample,
    )


__all__ = ["register"]
