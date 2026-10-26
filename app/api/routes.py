"""OpenAI-compatible FastAPI routes backed by the shared InferenceManager.

Only the documented subset is advertised.  Unsupported OpenAI features
(tools, n>1, logprobs, multimodal content, response_format) return a clear 400
instead of being silently accepted.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from typing import Any, AsyncIterator, Iterator

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool

from app.core.inference import InferenceManager
from app.core.settings import SettingsStore
from app.core.types import BackendError, CancelToken, GenerationParams

log = logging.getLogger(__name__)


class ApiError(Exception):
    def __init__(self, message: str, *, status_code: int, kind: str, param: str | None = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.kind = kind
        self.param = param


def _error_response(
    message: str, *, status_code: int, kind: str, param: str | None = None
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": kind,
                "param": param,
                "code": kind,
            }
        },
    )


def create_app(manager: InferenceManager, settings: SettingsStore) -> FastAPI:
    app = FastAPI(
        title="Kestrel Studio Local API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.manager = manager
    app.state.settings = settings
    app.state.request_count = 0
    app.state.started_at = time.time()
    app.state.counter_lock = threading.Lock()

    @app.exception_handler(ApiError)
    async def handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return _error_response(
            exc.message, status_code=exc.status_code, kind=exc.kind, param=exc.param
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return _error_response(
            "Invalid request body.", status_code=400, kind="invalid_request"
        )

    @app.exception_handler(BackendError)
    async def handle_backend_error(_request: Request, exc: BackendError) -> JSONResponse:
        return _error_response(
            exc.message, status_code=exc.status_code or 500, kind=exc.kind
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(_request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled API error")
        return _error_response(
            f"Internal server error: {exc}", status_code=500, kind="internal_error"
        )

    def count_request() -> None:
        with app.state.counter_lock:
            app.state.request_count += 1

    def require_model(body_model: Any) -> str:
        status = manager.status()
        if status["state"] != "ready" or not status["model_id"]:
            raise ApiError(
                "No model is loaded. Load a model in Kestrel Studio first.",
                status_code=503,
                kind="model_not_loaded",
            )
        current = status["model_id"]
        if body_model is not None and str(body_model) != current:
            raise ApiError(
                f"The model {body_model!r} is not loaded. The active model is {current!r}. "
                "Call GET /v1/models for the available ID.",
                status_code=404,
                kind="model_not_found",
                param="model",
            )
        return current

    @app.get("/health")
    def health() -> dict[str, Any]:
        status = manager.status()
        return {
            "status": "ok",
            "model_loaded": status["state"] == "ready",
            "model": status["model_id"],
            "active_requests": status["active_requests"],
            "server": "kestrel-studio",
        }

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        status = manager.status()
        data: list[dict[str, Any]] = []
        if status["state"] == "ready" and status["model_id"]:
            created = int(status.get("loaded_at") or time.time())
            data.append(
                {
                    "id": status["model_id"],
                    "object": "model",
                    "created": created,
                    "owned_by": status.get("provider") or "local",
                    "permission": [],
                    "root": status["model_id"],
                }
            )
        return {"object": "list", "data": data}

    @app.get("/v1/models/{model_id:path}")
    def get_model(model_id: str) -> dict[str, Any]:
        status = manager.status()
        if status["state"] != "ready" or status["model_id"] != model_id:
            raise ApiError(
                f"Unknown model {model_id!r}.",
                status_code=404,
                kind="model_not_found",
            )
        return {
            "id": model_id,
            "object": "model",
            "created": int(status.get("loaded_at") or time.time()),
            "owned_by": status.get("provider") or "local",
            "permission": [],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> Any:
        count_request()
        body = await _json_object(request)
        _reject_unsupported_chat(body)
        model_id = require_model(body.get("model"))
        messages = _parse_messages(body.get("messages"))
        params = _parse_params(body, settings, default_max_tokens=1024)
        stream = bool(body.get("stream", False))
        created = int(time.time())
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        cancel = CancelToken()

        if not stream:
            try:
                text, usage, finish_reason = await run_in_threadpool(
                    manager.complete, messages, params, system_prompt="", cancel=cancel
                )
            except BackendError as exc:
                raise ApiError(
                    exc.message, status_code=exc.status_code or 502, kind=exc.kind
                ) from exc
            response: dict[str, Any] = {
                "id": completion_id,
                "object": "chat.completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": text},
                        "finish_reason": finish_reason or "stop",
                    }
                ],
            }
            if usage:
                response["usage"] = usage
            return response

        def generate() -> Iterator[str]:
            sent_role = False
            finish_reason = None
            usage: dict[str, int] | None = None
            try:
                for event in manager.stream_chat(
                    messages, params, system_prompt="", cancel=cancel
                ):
                    if event.type == "delta":
                        if not sent_role:
                            yield _sse(
                                {
                                    "id": completion_id,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": model_id,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": {"role": "assistant", "content": ""},
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                            )
                            sent_role = True
                        yield _sse(
                            {
                                "id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": event.text},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    elif event.type == "usage":
                        usage = event.usage
                    elif event.type == "done":
                        finish_reason = event.finish_reason
                if not sent_role:
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_id,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                final: dict[str, Any] = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                }
                if usage:
                    final["usage"] = usage
                yield _sse(final)
                yield "data: [DONE]\n\n"
            except BackendError as exc:
                yield _sse(
                    {
                        "error": {
                            "message": exc.message,
                            "type": exc.kind,
                            "code": exc.kind,
                        }
                    }
                )
                yield "data: [DONE]\n\n"
            finally:
                cancel.cancel()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=headers
        )

    @app.post("/v1/completions")
    async def completions(request: Request) -> Any:
        count_request()
        body = await _json_object(request)
        _reject_unsupported_completion(body)
        model_id = require_model(body.get("model"))
        prompt = _parse_prompt(body.get("prompt"))
        params = _parse_params(body, settings, default_max_tokens=256)
        stream = bool(body.get("stream", False))
        messages = [{"role": "user", "content": prompt}]
        created = int(time.time())
        completion_id = f"cmpl-{uuid.uuid4().hex[:24]}"
        cancel = CancelToken()

        if not stream:
            try:
                text, usage, finish_reason = await run_in_threadpool(
                    manager.complete, messages, params, system_prompt="", cancel=cancel
                )
            except BackendError as exc:
                raise ApiError(
                    exc.message, status_code=exc.status_code or 502, kind=exc.kind
                ) from exc
            response: dict[str, Any] = {
                "id": completion_id,
                "object": "text_completion",
                "created": created,
                "model": model_id,
                "choices": [
                    {
                        "index": 0,
                        "text": text,
                        "logprobs": None,
                        "finish_reason": finish_reason or "stop",
                    }
                ],
            }
            if usage:
                response["usage"] = usage
            return response

        def generate() -> Iterator[str]:
            usage: dict[str, int] | None = None
            finish_reason: str | None = None
            try:
                for event in manager.stream_chat(
                    messages, params, system_prompt="", cancel=cancel
                ):
                    if event.type == "delta":
                        yield _sse(
                            {
                                "id": completion_id,
                                "object": "text_completion",
                                "created": created,
                                "model": model_id,
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": event.text,
                                        "logprobs": None,
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    elif event.type == "usage":
                        usage = event.usage
                    elif event.type == "done":
                        finish_reason = event.finish_reason
                final: dict[str, Any] = {
                    "id": completion_id,
                    "object": "text_completion",
                    "created": created,
                    "model": model_id,
                    "choices": [
                        {
                            "index": 0,
                            "text": "",
                            "logprobs": None,
                            "finish_reason": finish_reason or "stop",
                        }
                    ],
                }
                if usage:
                    final["usage"] = usage
                yield _sse(final)
                yield "data: [DONE]\n\n"
            except BackendError as exc:
                yield _sse(
                    {
                        "error": {
                            "message": exc.message,
                            "type": exc.kind,
                            "code": exc.kind,
                        }
                    }
                )
                yield "data: [DONE]\n\n"
            finally:
                cancel.cancel()

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
        return StreamingResponse(
            generate(), media_type="text/event-stream", headers=headers
        )

    return app


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

async def _json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise ApiError(
            f"Request body must be valid JSON: {exc}",
            status_code=400,
            kind="invalid_request",
        ) from exc
    if not isinstance(body, dict):
        raise ApiError(
            "Request body must be a JSON object.",
            status_code=400,
            kind="invalid_request",
        )
    return body


def _reject_unsupported_chat(body: dict[str, Any]) -> None:
    if body.get("tools") or body.get("tool_choice") or body.get("functions"):
        raise ApiError(
            "Tool/function calling is not implemented by this server.",
            status_code=400,
            kind="unsupported_feature",
            param="tools",
        )
    if body.get("response_format") is not None:
        raise ApiError(
            "response_format is not implemented. Use prompt instructions instead.",
            status_code=400,
            kind="unsupported_feature",
            param="response_format",
        )
    if body.get("logprobs") or body.get("top_logprobs"):
        raise ApiError(
            "Logprobs are not implemented.",
            status_code=400,
            kind="unsupported_feature",
            param="logprobs",
        )
    n = body.get("n", 1)
    if n not in (None, 1):
        raise ApiError(
            "Only n=1 is supported.",
            status_code=400,
            kind="unsupported_feature",
            param="n",
        )
    for field_name in ("frequency_penalty", "presence_penalty"):
        value = body.get(field_name)
        if value not in (None, 0, 0.0):
            raise ApiError(
                f"{field_name} is not supported by this backend.",
                status_code=400,
                kind="unsupported_feature",
                param=field_name,
            )


def _reject_unsupported_completion(body: dict[str, Any]) -> None:
    if body.get("logprobs") is not None:
        raise ApiError(
            "Logprobs are not implemented.",
            status_code=400,
            kind="unsupported_feature",
            param="logprobs",
        )
    n = body.get("n", 1)
    if n not in (None, 1):
        raise ApiError(
            "Only n=1 is supported.",
            status_code=400,
            kind="unsupported_feature",
            param="n",
        )


def _parse_messages(raw: Any) -> list[dict[str, str]]:
    if not isinstance(raw, list) or not raw:
        raise ApiError(
            "messages must be a non-empty array.",
            status_code=400,
            kind="invalid_request",
            param="messages",
        )
    parsed: list[dict[str, str]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ApiError(
                f"messages[{index}] must be an object.",
                status_code=400,
                kind="invalid_request",
                param="messages",
            )
        role = item.get("role")
        content = item.get("content")
        if role not in {"system", "user", "assistant"}:
            raise ApiError(
                f"messages[{index}].role must be system, user or assistant.",
                status_code=400,
                kind="invalid_request",
                param="messages",
            )
        if isinstance(content, list):
            raise ApiError(
                "Multimodal content is not supported; send a text string.",
                status_code=400,
                kind="unsupported_feature",
                param=f"messages[{index}].content",
            )
        if content is None:
            content = ""
        if not isinstance(content, str):
            raise ApiError(
                f"messages[{index}].content must be a string.",
                status_code=400,
                kind="invalid_request",
                param="messages",
            )
        parsed.append({"role": role, "content": content})
    if not any(m["role"] == "user" for m in parsed):
        raise ApiError(
            "messages must include at least one user message.",
            status_code=400,
            kind="invalid_request",
            param="messages",
        )
    return parsed


def _parse_prompt(raw: Any) -> str:
    if isinstance(raw, str) and raw.strip():
        return raw
    if isinstance(raw, list):
        if len(raw) == 1 and isinstance(raw[0], str) and raw[0].strip():
            return raw[0]
        raise ApiError(
            "prompt arrays with multiple prompts are not supported (n=1 only).",
            status_code=400,
            kind="unsupported_feature",
            param="prompt",
        )
    raise ApiError(
        "prompt must be a non-empty string.",
        status_code=400,
        kind="invalid_request",
        param="prompt",
    )


def _number(
    body: dict[str, Any], key: str, default: float, minimum: float, maximum: float
) -> float:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(
            f"{key} must be a number.", status_code=400, kind="invalid_request", param=key
        )
    if value < minimum or value > maximum:
        raise ApiError(
            f"{key} must be between {minimum} and {maximum}.",
            status_code=400,
            kind="invalid_request",
            param=key,
        )
    return float(value)


def _integer(
    body: dict[str, Any], key: str, default: int, minimum: int, maximum: int
) -> int:
    value = body.get(key)
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(
            f"{key} must be an integer.",
            status_code=400,
            kind="invalid_request",
            param=key,
        )
    if value < minimum or value > maximum:
        raise ApiError(
            f"{key} must be between {minimum} and {maximum}.",
            status_code=400,
            kind="invalid_request",
            param=key,
        )
    return value


def _parse_params(
    body: dict[str, Any], settings: SettingsStore, *, default_max_tokens: int
) -> GenerationParams:
    chat = settings.data.get("chat", {})
    max_tokens = body.get("max_tokens")
    if max_tokens is None:
        max_tokens = body.get("max_completion_tokens")
    if max_tokens is not None:
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int):
            raise ApiError(
                "max_tokens must be an integer.",
                status_code=400,
                kind="invalid_request",
                param="max_tokens",
            )
        if max_tokens < 1 or max_tokens > 131072:
            raise ApiError(
                "max_tokens must be between 1 and 131072.",
                status_code=400,
                kind="invalid_request",
                param="max_tokens",
            )
        effective_max = max_tokens
    else:
        effective_max = int(chat.get("max_tokens") or default_max_tokens)

    stop_raw = body.get("stop")
    stop: list[str] = []
    if isinstance(stop_raw, str):
        stop = [stop_raw]
    elif isinstance(stop_raw, list) and all(isinstance(item, str) for item in stop_raw):
        stop = list(stop_raw)

    return GenerationParams(
        temperature=_number(body, "temperature", float(chat.get("temperature", 0.7)), 0.0, 2.0),
        top_p=_number(body, "top_p", float(chat.get("top_p", 0.95)), 0.0, 1.0),
        top_k=_integer(body, "top_k", int(chat.get("top_k", 40)), 0, 1000),
        repeat_penalty=_number(
            body, "repeat_penalty", float(chat.get("repeat_penalty", 1.1)), 0.0, 4.0
        ),
        max_tokens=effective_max,
        seed=_integer(body, "seed", int(chat.get("seed", -1)), -1, 2_147_483_647),
        stop=stop,
    )


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
