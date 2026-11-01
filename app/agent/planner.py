"""Planning and next-action selection for the Hybrid Agent.

The planner talks to the *same* inference backend the chat page uses (a model is
never loaded twice) and only ever returns data: a plan, or one validated tool
call. Execution happens in :mod:`app.agent.executor`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from app.agent.tools.registry import ToolRegistry
from app.agent.validator import (
    ToolValidationError,
    extract_json_object,
    parse_model_action,
    validate_plan,
)

PLANNER_SYSTEM = """You are the planning module of a Windows desktop agent.
You break a goal into a short, concrete plan. You never execute anything.
Reply with JSON only, in this shape:
{"steps": [{"title": "short imperative step", "tool": "tool_name or empty"}], "notes": "one short sentence"}
Rules:
- 2 to 6 steps, ordered, each verifiable.
- Prefer investigation before modification, and verification last.
- Use only these tools when naming one: {tools}
"""

ACTOR_SYSTEM = """You are the acting module of a Windows desktop agent.
You choose exactly ONE next action, as JSON only.
Shapes you may return:
{"thought": "short user-facing reason", "tool": "tool_name", "arguments": {...}}
{"thought": "why this step is finished", "step_done": true}
{"summary": "final report", "finish": true}
Rules:
- One tool call per reply, arguments must match the tool schema exactly.
- Prefer read-only investigation before changing files.
- If a command failed, read the error and try a different approach; never repeat an identical failing call.
- Never invent tools. Never claim you did something you did not do.
- Allowed tools in this mode: {tools}
"""


class ModelUnavailable(RuntimeError):
    """No model is loaded, so the agent cannot reason about the goal."""


@dataclass
class PlanStep:
    index: int
    title: str
    tool: str = ""
    status: str = "pending"
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Plan:
    goal: str
    mode: str = "autonomous"
    steps: list[PlanStep] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "mode": self.mode,
            "notes": self.notes,
            "steps": [step.to_dict() for step in self.steps],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Plan":
        steps = []
        for item in data.get("steps") or []:
            if not isinstance(item, dict):
                continue
            steps.append(
                PlanStep(
                    index=int(item.get("index", len(steps))),
                    title=str(item.get("title") or ""),
                    tool=str(item.get("tool") or ""),
                    status=str(item.get("status") or "pending"),
                    note=str(item.get("note") or ""),
                )
            )
        return cls(
            goal=str(data.get("goal") or ""),
            mode=str(data.get("mode") or "autonomous"),
            steps=steps,
            notes=str(data.get("notes") or ""),
        )


class ModelClient:
    """Thin wrapper over the shared inference manager."""

    def __init__(self, inference: Any, settings: Any):
        self.inference = inference
        self.settings = settings

    @property
    def available(self) -> bool:
        try:
            return bool(self.inference is not None and self.inference.is_ready)
        except Exception:
            return False

    def label(self) -> str:
        try:
            status = self.inference.status()
            return str(status.get("display_name") or status.get("model_id") or "モデル未選択")
        except Exception:
            return "モデル未選択"

    def ask(
        self,
        system_prompt: str,
        prompt: str,
        *,
        max_tokens: int = 900,
        temperature: float = 0.2,
    ) -> str:
        if not self.available:
            raise ModelUnavailable("モデルが読み込まれていません。")
        from app.core.types import GenerationParams

        params = GenerationParams(
            temperature=float(temperature),
            top_p=0.95,
            top_k=40,
            repeat_penalty=1.05,
            max_tokens=int(max_tokens),
            seed=-1,
        )
        text, _usage, _finish = self.inference.complete(
            [{"role": "user", "content": prompt}],
            params,
            system_prompt=system_prompt,
        )
        return str(text or "")


class Planner:
    def __init__(self, model_client: ModelClient, registry: ToolRegistry):
        self.model = model_client
        self.registry = registry

    # -- plan --------------------------------------------------------------
    def make_plan(
        self,
        goal: str,
        mode: str,
        *,
        workspace: str = "",
        memory_context: str = "",
    ) -> Plan:
        tools = self.registry.describe_for_prompt(mode if mode != "plan" else "plan")
        prompt = (
            f"Goal: {goal}\n"
            f"Mode: {mode}\n"
            f"Working directory: {workspace or '(none)'}\n"
            f"Relevant memory:\n{memory_context or '(none)'}\n\n"
            "Return the JSON plan now."
        )
        if self.model.available:
            for attempt in range(2):
                try:
                    text = self.model.ask(
                        PLANNER_SYSTEM.format(tools=tools), prompt, max_tokens=700
                    )
                    payload = extract_json_object(text)
                    steps = validate_plan(payload)
                    return Plan(
                        goal=goal,
                        mode=mode,
                        steps=[
                            PlanStep(
                                index=item["index"],
                                title=item["title"],
                                tool=item.get("tool", ""),
                            )
                            for item in steps
                        ],
                        notes=str(payload.get("notes") or "")[:500],
                    )
                except (ToolValidationError, ValueError):
                    if attempt == 0:
                        prompt += "\n\nYour previous reply was not valid JSON. Reply with JSON only."
                        continue
                except Exception:
                    break
        return self._fallback_plan(goal, mode)

    def _fallback_plan(self, goal: str, mode: str) -> Plan:
        note = (
            "モデルが利用できないため、既定の4段階の計画を使用しています。"
            if not self.model.available
            else "モデルの計画を解釈できなかったため、既定の4段階の計画を使用しています。"
        )
        titles = [
            "作業ディレクトリと開発環境を調査する",
            "目標に必要なファイル変更やコマンドを実行する",
            "テストやコマンドで結果を検証する",
            "結果を要約して報告する",
        ]
        return Plan(
            goal=goal,
            mode=mode,
            steps=[PlanStep(index=i, title=title) for i, title in enumerate(titles)],
            notes=note,
        )

    # -- next action -------------------------------------------------------
    def next_action(
        self,
        *,
        goal: str,
        plan: Plan,
        step_index: int,
        history: list[str],
        observation: str,
        mode: str,
        memory_context: str = "",
        workspace: str = "",
    ) -> dict[str, Any]:
        if not self.model.available:
            raise ModelUnavailable(
                "モデルが読み込まれていないため、エージェントは次の操作を決定できません。"
                "「モデル管理」でGGUFモデルを読み込んでください。"
            )
        tools = self.registry.describe_for_prompt(mode)
        plan_text = "\n".join(
            f"{'→' if step.index == step_index else ' '} {step.index + 1}. {step.title} [{step.status}]"
            for step in plan.steps
        )
        recent = "\n".join(history[-10:]) or "(no actions yet)"
        prompt = (
            f"Goal: {goal}\nMode: {mode}\nWorking directory: {workspace or '(none)'}\n"
            f"Plan:\n{plan_text}\n\nRecent actions:\n{recent}\n\n"
            f"Last observation:\n{(observation or '(none)')[:4000]}\n\n"
            f"Relevant memory:\n{(memory_context or '(none)')[:1200]}\n\n"
            "Choose the single next action. Reply with JSON only."
        )
        last_error = ""
        for attempt in range(2):
            text = self.model.ask(ACTOR_SYSTEM.format(tools=tools), prompt, max_tokens=700)
            try:
                return parse_model_action(text, self.registry, mode)
            except ToolValidationError as exc:
                last_error = str(exc)
                prompt += (
                    f"\n\nYour previous reply was rejected: {last_error}\n"
                    "Reply again with JSON only, using a registered tool and valid arguments."
                )
        raise ToolValidationError(last_error or "モデルのツール呼び出しを検証できませんでした。")

    # -- sub-agent guidance ------------------------------------------------
    def ask_role(self, system_prompt: str, prompt: str, *, max_tokens: int = 500) -> str:
        if not self.model.available:
            raise ModelUnavailable("モデルが読み込まれていません。")
        return self.model.ask(system_prompt, prompt, max_tokens=max_tokens)


def format_memory(records: list[Any], limit: int = 8) -> str:
    if not records:
        return ""
    lines = []
    for record in records[:limit]:
        lines.append(f"- [{record.category}] {record.key}: {record.value[:300]}")
    return "\n".join(lines)


__all__ = [
    "ModelClient",
    "ModelUnavailable",
    "Plan",
    "PlanStep",
    "Planner",
    "format_memory",
]
