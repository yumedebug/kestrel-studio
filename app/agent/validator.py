"""Validation of model output: JSON extraction, tool calls and plans.

Model output is never executed as-is. It is parsed, checked against the tool's
declared schema and rejected with a readable reason when it does not fit, so the
planner can ask the model to try again.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.tools.registry import ToolCall, ToolRegistry

MAX_TOOL_ARGUMENT_CHARS = 200_000


class ToolValidationError(ValueError):
    """The model produced a tool call or plan that cannot be used."""


_TYPE_MAP = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull the first JSON object out of a model response."""
    raw = str(text or "").strip()
    if not raw:
        raise ToolValidationError("The model returned an empty response.")
    candidates: list[str] = []
    fenced = re.findall(r"```(?:json)?\s*(.+?)```", raw, re.DOTALL | re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced)
    candidates.append(raw)
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
    # Brace matching that respects strings and escapes.
    start = raw.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(raw)):
            char = raw[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = raw[start : index + 1]
                    try:
                        parsed = json.loads(snippet)
                    except (ValueError, TypeError):
                        break
                    if isinstance(parsed, dict):
                        return parsed
                    break
        start = raw.find("{", start + 1)
    raise ToolValidationError("The response did not contain a JSON object.")


def _coerce(value: Any, type_name: str, field: str) -> Any:
    if type_name == "integer":
        if isinstance(value, bool):
            raise ToolValidationError(f"{field} must be an integer.")
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{field} must be an integer.") from exc
    if type_name == "number":
        if isinstance(value, bool):
            raise ToolValidationError(f"{field} must be a number.")
        try:
            return float(value)
        except (TypeError, ValueError) as exc:
            raise ToolValidationError(f"{field} must be a number.") from exc
    if type_name == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().casefold()
            if lowered in {"true", "yes", "1"}:
                return True
            if lowered in {"false", "no", "0"}:
                return False
        raise ToolValidationError(f"{field} must be true or false.")
    if type_name == "string":
        if isinstance(value, (dict, list)):
            raise ToolValidationError(f"{field} must be a string.")
        return str(value)
    if type_name == "array":
        if not isinstance(value, list):
            raise ToolValidationError(f"{field} must be a JSON array.")
        return value
    if type_name == "object":
        if not isinstance(value, dict):
            raise ToolValidationError(f"{field} must be a JSON object.")
        return value
    return value


def validate_arguments(spec, arguments: Any) -> dict[str, Any]:
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ToolValidationError("arguments must be a JSON object.")
    unknown = [key for key in arguments if key not in spec.parameters]
    if unknown:
        raise ToolValidationError(
            f"Unknown argument(s) for {spec.name}: {', '.join(sorted(unknown))}."
        )
    cleaned: dict[str, Any] = {}
    for name, definition in spec.parameters.items():
        required = bool(definition.get("required", False))
        if name not in arguments or arguments[name] is None:
            if required:
                raise ToolValidationError(f"{spec.name} requires '{name}'.")
            continue
        value = _coerce(arguments[name], str(definition.get("type", "string")), name)
        enum = definition.get("enum")
        if enum and value not in enum:
            raise ToolValidationError(
                f"'{name}' must be one of: {', '.join(str(item) for item in enum)}."
            )
        if isinstance(value, str) and len(value) > MAX_TOOL_ARGUMENT_CHARS:
            raise ToolValidationError(f"'{name}' is too large.")
        cleaned[name] = value
    return cleaned


def validate_tool_call(
    payload: Any, registry: ToolRegistry, mode: str = "autonomous"
) -> ToolCall:
    if not isinstance(payload, dict):
        raise ToolValidationError("A tool call must be a JSON object.")
    name = payload.get("tool") or payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ToolValidationError("The tool call is missing a 'tool' name.")
    spec = registry.get(name.strip())
    if spec is None:
        raise ToolValidationError(
            f"'{name}' is not a registered tool. Use one of: {', '.join(registry.names())}."
        )
    if mode not in spec.modes:
        raise ToolValidationError(f"'{spec.name}' is not available in {mode} mode.")
    arguments = validate_arguments(spec, payload.get("arguments") or {})
    return ToolCall(spec.name, arguments)


def parse_model_action(
    text: str, registry: ToolRegistry, mode: str = "autonomous"
) -> dict[str, Any]:
    """Turn a model response into either a tool call or a control action."""
    payload = extract_json_object(text)
    if payload.get("finish") is True:
        return {"type": "finish", "summary": str(payload.get("summary") or "").strip()}
    if payload.get("step_done") is True:
        return {
            "type": "step_done",
            "note": str(payload.get("note") or payload.get("thought") or "").strip(),
        }
    if not payload.get("tool") and not payload.get("name"):
        raise ToolValidationError(
            'Respond with {"tool": "...", "arguments": {...}} or {"finish": true}.'
        )
    call = validate_tool_call(payload, registry, mode)
    thought = str(payload.get("thought") or payload.get("reason") or "").strip()
    return {"type": "tool", "call": call, "thought": thought}


def validate_plan(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        steps = payload.get("steps")
    else:
        steps = payload
    if not isinstance(steps, list) or not steps:
        raise ToolValidationError("A plan needs a non-empty 'steps' list.")
    cleaned: list[dict[str, Any]] = []
    for index, step in enumerate(steps, 1):
        if isinstance(step, str):
            title = step.strip()
            hint = ""
        elif isinstance(step, dict):
            title = str(step.get("title") or step.get("step") or "").strip()
            hint = str(step.get("tool") or step.get("tool_hint") or "").strip()
        else:
            raise ToolValidationError(f"Step {index} is not a string or object.")
        if not title:
            raise ToolValidationError(f"Step {index} has no title.")
        cleaned.append({"index": index - 1, "title": title[:200], "tool": hint, "status": "pending"})
        if len(cleaned) >= 20:
            break
    return cleaned


def verify_tool_result(result: Any) -> tuple[bool, str]:
    """Turn a tool result into a verification verdict."""
    if result is None:
        return False, "No result was produced."
    if result.stopped:
        return False, "The task was stopped by the user."
    if result.rejected:
        return False, "The user rejected this action."
    if result.denied:
        return False, f"Denied by policy: {result.summary or result.error}"
    if result.ok:
        return True, result.summary or "ok"
    return False, result.error or result.summary or "The tool reported a failure."


__all__ = [
    "ToolValidationError",
    "extract_json_object",
    "parse_model_action",
    "validate_arguments",
    "validate_plan",
    "validate_tool_call",
    "verify_tool_result",
]
