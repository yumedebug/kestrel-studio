"""HTTP error mapping and SSE helpers used by all HTTP-based backends."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator

import httpx

from app.core.types import BackendError, CancelToken


def extract_error_message(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return "The provider returned an empty error response."
    try:
        data = json.loads(text)
    except ValueError:
        return text[:800]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail")
            if message:
                return str(message)[:800]
        if isinstance(error, str):
            return error[:800]
        if isinstance(data.get("message"), str):
            return data["message"][:800]
        if isinstance(data.get("detail"), str):
            return data["detail"][:800]
    return text[:800]


def error_from_response(
    response: httpx.Response, provider_name: str = "Provider"
) -> BackendError:
    return error_from_body(response.status_code, response.text, provider_name)


def error_from_body(
    status: int, body: str, provider_name: str = "Provider"
) -> BackendError:
    message = extract_error_message(body)
    if status in (401, 403):
        return BackendError(
            f"{provider_name} rejected the API key: {message}",
            kind="auth_error",
            status_code=401,
            detail=f"HTTP {status}",
        )
    if status == 404:
        return BackendError(
            f"{provider_name} could not find that model or endpoint: {message}",
            kind="model_not_found",
            status_code=404,
            detail=f"HTTP {status}",
        )
    if status == 429:
        return BackendError(
            f"{provider_name} rate limit reached. Wait a moment and retry: {message}",
            kind="rate_limit",
            status_code=429,
            detail=f"HTTP {status}",
        )
    if status in (400, 422):
        return BackendError(
            f"{provider_name} rejected the request: {message}",
            kind="invalid_request",
            status_code=400,
            detail=f"HTTP {status}",
        )
    if status >= 500:
        return BackendError(
            f"{provider_name} is having trouble (HTTP {status}): {message}",
            kind="provider_error",
            status_code=502,
            detail=f"HTTP {status}",
        )
    return BackendError(
        f"{provider_name} request failed (HTTP {status}): {message}",
        kind="provider_error",
        status_code=502,
        detail=f"HTTP {status}",
    )


@dataclass
class SSEFrame:
    event: str
    data: str


def iter_sse(response: httpx.Response) -> Iterator[SSEFrame]:
    event = "message"
    try:
        for line in response.iter_lines():
            if line == "":
                event = "message"
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event = line[6:].strip()
                continue
            if line.startswith("data:"):
                yield SSEFrame(event=event, data=line[5:].strip())
    except httpx.HTTPError as exc:
        raise BackendError(
            f"Connection to the provider was interrupted: {exc}",
            kind="network_error",
            status_code=502,
        ) from exc


def iter_openai_sse(
    response: httpx.Response,
    cancel: CancelToken,
    provider_name: str,
) -> Iterator[tuple[str, Any]]:
    """Yield ('delta', text), ('usage', dict), ('finish', reason) tuples."""
    for frame in iter_sse(response):
        if cancel.is_set():
            return
        if frame.data == "[DONE]":
            return
        try:
            payload = json.loads(frame.data)
        except ValueError:
            continue
        if not isinstance(payload, dict):
            continue
        error = payload.get("error")
        if error:
            message = (
                error.get("message") if isinstance(error, dict) else str(error)
            ) or "Unknown provider error"
            raise BackendError(
                f"{provider_name} reported an error: {message}",
                kind="provider_error",
                status_code=502,
            )
        usage = payload.get("usage")
        if isinstance(usage, dict):
            normalized = _normalize_usage(usage)
            if normalized:
                yield "usage", normalized
        choices = payload.get("choices")
        if not isinstance(choices, list):
            continue
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta")
            if not isinstance(delta, dict):
                delta = {}
            content = delta.get("content")
            if isinstance(content, list):
                # Some compatible providers return structured content parts.
                text = "".join(
                    str(part.get("text", ""))
                    for part in content
                    if isinstance(part, dict)
                )
                if text:
                    yield "delta", text
            elif isinstance(content, str) and content:
                yield "delta", content
            reason = choice.get("finish_reason")
            if reason:
                yield "finish", str(reason)


def _normalize_usage(usage: dict[str, Any]) -> dict[str, int]:
    prompt = usage.get("prompt_tokens", usage.get("input_tokens"))
    completion = usage.get("completion_tokens", usage.get("output_tokens"))
    total = usage.get("total_tokens")
    result: dict[str, int] = {}
    if isinstance(prompt, int):
        result["prompt_tokens"] = prompt
    if isinstance(completion, int):
        result["completion_tokens"] = completion
    if isinstance(total, int):
        result["total_tokens"] = total
    elif "prompt_tokens" in result and "completion_tokens" in result:
        result["total_tokens"] = result["prompt_tokens"] + result["completion_tokens"]
    return result


def merge_consecutive(messages: list[dict[str, str]], allowed: set[str]) -> list[dict[str, str]]:
    """Merge adjacent same-role messages (required by Anthropic/Gemini)."""
    merged: list[dict[str, str]] = []
    for message in messages:
        role = message.get("role", "user")
        if role not in allowed:
            continue
        content = str(message.get("content", ""))
        if not content:
            continue
        if merged and merged[-1]["role"] == role:
            merged[-1]["content"] += "\n\n" + content
        else:
            merged.append({"role": role, "content": content})
    return merged
