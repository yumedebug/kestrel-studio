"""Cloud inference backend (OpenAI-compatible, Anthropic, Gemini)."""

from __future__ import annotations

import json
from contextlib import contextmanager
from typing import Any, Iterable, Iterator
from urllib.parse import quote

import httpx

from app.core.backends.base import InferenceBackend
from app.core.backends.http_utils import (
    error_from_body,
    iter_openai_sse,
    iter_sse,
    merge_consecutive,
    _normalize_usage,
)
from app.core.credentials import CredentialStore
from app.core.providers import ProviderSpec, credential_key, get_provider
from app.core.settings import SettingsStore
from app.core.types import (
    BackendError,
    CancelToken,
    GenerationCancelled,
    GenerationParams,
    LoadOptions,
    ModelTarget,
    StreamEvent,
)

DEFAULT_TIMEOUT = httpx.Timeout(connect=20.0, read=300.0, write=60.0, pool=20.0)


class CloudBackend(InferenceBackend):
    name = "cloud"

    def __init__(
        self,
        settings: SettingsStore,
        credentials: CredentialStore,
        transport: httpx.BaseTransport | None = None,
    ):
        self.settings_store = settings
        self.credentials = credentials
        self._transport = transport
        self._spec: ProviderSpec | None = None
        self._base_url = ""
        self._api_key = ""
        self._model = ""
        self._warning = ""
        self.last_warning = ""

    # -- lifecycle ---------------------------------------------------------
    def load(self, target: ModelTarget, options: LoadOptions) -> None:
        if target.kind != "cloud":
            raise BackendError("CloudBackend requires a cloud model target.", kind="invalid_target")
        spec = get_provider(target.provider)
        if spec is None:
            raise BackendError(
                f"Unknown cloud provider {target.provider!r}. Add it in Settings.",
                kind="invalid_request",
                status_code=400,
            )
        if not target.model.strip():
            raise BackendError("No cloud model ID was selected.", kind="invalid_request", status_code=400)

        config = self.settings_store.get(f"cloud.providers.{spec.id}", {}) or {}
        if not isinstance(config, dict):
            config = {}
        base_url = str(config.get("base_url") or spec.base_url).strip().rstrip("/")
        if not base_url:
            raise BackendError(
                f"{spec.name} needs a base URL. Set it in Settings → Cloud Providers.",
                kind="invalid_request",
                status_code=400,
            )
        if not (base_url.startswith("https://") or base_url.startswith("http://")):
            raise BackendError(
                f"{spec.name} base URL must start with http:// or https://",
                kind="invalid_request",
                status_code=400,
            )

        api_key = self.credentials.get(credential_key(spec.id)) or ""
        if spec.key_required and not api_key:
            raise BackendError(
                f"No API key stored for {spec.name}. Add one in Settings → Cloud Providers, "
                f"or choose a local model.",
                kind="auth_error",
                status_code=401,
            )

        self._spec = spec
        self._base_url = base_url
        self._api_key = api_key
        self._model = target.model.strip()
        self._warning = ""
        if spec.id == "lmstudio" and base_url.startswith("http://127.0.0.1:1234"):
            self._warning = (
                "LM Studio uses port 1234, which is also this app's default Local Server port. "
                "Change one of them to avoid a conflict."
            )
        self.last_warning = self._warning

    def unload(self) -> None:
        self._spec = None
        self._base_url = ""
        self._api_key = ""
        self._model = ""

    def describe(self) -> dict[str, str]:
        data = super().describe()
        if self._spec:
            data.update(
                {
                    "provider": self._spec.id,
                    "provider_name": self._spec.name,
                    "base_url": self._base_url,
                    "model": self._model,
                }
            )
        return data

    # -- request helpers ---------------------------------------------------
    def _client(self) -> httpx.Client:
        kwargs: dict[str, Any] = {
            "timeout": DEFAULT_TIMEOUT,
            "follow_redirects": True,
        }
        if self._transport is not None:
            kwargs["transport"] = self._transport
        return httpx.Client(**kwargs)

    def _headers(self) -> dict[str, str]:
        assert self._spec is not None
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._spec.kind == "anthropic":
            headers["x-api-key"] = self._api_key
            headers["anthropic-version"] = "2023-06-01"
        elif self._spec.kind == "gemini":
            if self._api_key:
                headers["x-goog-api-key"] = self._api_key
        else:
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _openai_payload(
        self, messages: list[dict[str, str]], params: GenerationParams
    ) -> dict[str, Any]:
        assert self._spec is not None
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
            "max_tokens": params.max_tokens,
        }
        if params.seed >= 0 and self._spec.supports_seed:
            payload["seed"] = params.seed
        if self._spec.supports_top_k and params.top_k > 0:
            payload["top_k"] = params.top_k
        if self._spec.supports_repeat_penalty and params.repeat_penalty:
            payload["repeat_penalty"] = params.repeat_penalty
        if params.stop:
            payload["stop"] = params.stop[:4]
        if self._spec.include_usage:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def _anthropic_payload(
        self, messages: list[dict[str, str]], params: GenerationParams
    ) -> dict[str, Any]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        conversation = merge_consecutive(messages, {"user", "assistant"})
        if not conversation:
            conversation = [{"role": "user", "content": ""}]
        payload: dict[str, Any] = {
            "model": self._model,
            "max_tokens": params.max_tokens,
            "messages": conversation,
            "stream": True,
            "temperature": params.temperature,
            "top_p": params.top_p,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if params.top_k > 0:
            payload["top_k"] = params.top_k
        if params.stop:
            payload["stop_sequences"] = params.stop[:4]
        return payload

    def _gemini_payload(
        self, messages: list[dict[str, str]], params: GenerationParams
    ) -> dict[str, Any]:
        system_parts = [m["content"] for m in messages if m.get("role") == "system"]
        contents = []
        for message in merge_consecutive(messages, {"user", "assistant"}):
            role = "model" if message["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": message["content"]}]})
        config: dict[str, Any] = {
            "temperature": params.temperature,
            "topP": params.top_p,
            "maxOutputTokens": params.max_tokens,
        }
        if params.top_k > 0:
            config["topK"] = params.top_k
        if params.seed >= 0:
            config["seed"] = params.seed
        if params.stop:
            config["stopSequences"] = params.stop[:5]
        payload: dict[str, Any] = {"contents": contents, "generationConfig": config}
        if system_parts:
            payload["systemInstruction"] = {
                "parts": [{"text": "\n\n".join(system_parts)}]
            }
        return payload

    @contextmanager
    def _stream_request(
        self, url: str, payload: dict[str, Any], cancel: CancelToken
    ) -> Iterator[httpx.Response]:
        provider_name = self._spec.name if self._spec else "Provider"
        with self._client() as client:
            try:
                with client.stream(
                    "POST", url, json=payload, headers=self._headers()
                ) as response:
                    if response.status_code >= 400:
                        body = response.read().decode("utf-8", errors="replace")
                        raise error_from_body(response.status_code, body, provider_name)
                    cancel.add_callback(response.close)
                    try:
                        yield response
                    finally:
                        cancel.remove_callback(response.close)
            except httpx.HTTPError as exc:
                if cancel.is_set():
                    raise GenerationCancelled() from exc
                raise BackendError(
                    f"Could not reach {provider_name}: {exc}",
                    kind="network_error",
                    status_code=502,
                ) from exc

    # -- streaming ---------------------------------------------------------
    def stream_chat(
        self,
        messages: Iterable[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        if self._spec is None:
            raise BackendError("No cloud provider is loaded.", kind="not_loaded", status_code=503)
        message_list = [dict(m) for m in messages]
        if self._spec.kind == "anthropic":
            yield from self._stream_anthropic(message_list, params, cancel)
        elif self._spec.kind == "gemini":
            yield from self._stream_gemini(message_list, params, cancel)
        else:
            yield from self._stream_openai(message_list, params, cancel)

    def _stream_openai(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        assert self._spec is not None
        url = f"{self._base_url}/chat/completions"
        payload = self._openai_payload(messages, params)
        finish_reason: str | None = None
        with self._stream_request(url, payload, cancel) as response:
            for kind, value in iter_openai_sse(response, cancel, self._spec.name):
                if kind == "delta":
                    yield StreamEvent("delta", text=str(value))
                elif kind == "usage":
                    yield StreamEvent("usage", usage=value)
                elif kind == "finish":
                    finish_reason = str(value)
        if cancel.is_set():
            yield StreamEvent("done", finish_reason="cancelled")
        else:
            yield StreamEvent("done", finish_reason=finish_reason or "stop")

    def _stream_anthropic(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        assert self._spec is not None
        payload = self._anthropic_payload(messages, params)
        url = f"{self._base_url}/v1/messages"
        if self._base_url.endswith("/v1"):
            url = f"{self._base_url}/messages"
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        with self._stream_request(url, payload, cancel) as response:
            for frame in iter_sse(response):
                if cancel.is_set():
                    break
                try:
                    data = json.loads(frame.data)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                event_type = data.get("type") or frame.event
                if event_type == "error":
                    error = data.get("error") or {}
                    raise BackendError(
                        f"{self._spec.name} reported an error: {error.get('message', 'unknown')}",
                        kind="provider_error",
                        status_code=502,
                    )
                if event_type == "message_start":
                    message = data.get("message") or {}
                    raw_usage = message.get("usage") or {}
                    if isinstance(raw_usage.get("input_tokens"), int):
                        usage["prompt_tokens"] = raw_usage["input_tokens"]
                elif event_type == "content_block_delta":
                    delta = data.get("delta") or {}
                    if delta.get("type") == "text_delta":
                        text = delta.get("text")
                        if isinstance(text, str) and text:
                            yield StreamEvent("delta", text=text)
                elif event_type == "message_delta":
                    delta = data.get("delta") or {}
                    if delta.get("stop_reason"):
                        finish_reason = str(delta["stop_reason"])
                    raw_usage = data.get("usage") or {}
                    if isinstance(raw_usage.get("output_tokens"), int):
                        usage["completion_tokens"] = raw_usage["output_tokens"]
                    if (
                        "prompt_tokens" in usage
                        and "completion_tokens" in usage
                    ):
                        usage["total_tokens"] = (
                            usage["prompt_tokens"] + usage["completion_tokens"]
                        )
        if usage:
            yield StreamEvent("usage", usage=usage)
        if cancel.is_set():
            yield StreamEvent("done", finish_reason="cancelled")
        else:
            mapping = {
                "end_turn": "stop",
                "max_tokens": "length",
                "stop_sequence": "stop",
                "tool_use": "tool_calls",
            }
            yield StreamEvent(
                "done", finish_reason=mapping.get(finish_reason or "", finish_reason or "stop")
            )

    def _stream_gemini(
        self,
        messages: list[dict[str, str]],
        params: GenerationParams,
        cancel: CancelToken,
    ) -> Iterator[StreamEvent]:
        assert self._spec is not None
        model = quote(self._model, safe="-._/")
        url = f"{self._base_url}/models/{model}:streamGenerateContent?alt=sse"
        payload = self._gemini_payload(messages, params)
        usage: dict[str, int] = {}
        finish_reason: str | None = None
        with self._stream_request(url, payload, cancel) as response:
            for frame in iter_sse(response):
                if cancel.is_set():
                    break
                try:
                    data = json.loads(frame.data)
                except ValueError:
                    continue
                if not isinstance(data, dict):
                    continue
                if "error" in data:
                    error = data.get("error") or {}
                    raise BackendError(
                        f"{self._spec.name} reported an error: {error.get('message', 'unknown')}",
                        kind="provider_error",
                        status_code=502,
                    )
                prompt_feedback = data.get("promptFeedback") or {}
                if prompt_feedback.get("blockReason"):
                    raise BackendError(
                        "Gemini blocked the prompt "
                        f"({prompt_feedback.get('blockReason')}). Adjust the content and retry.",
                        kind="content_filter",
                        status_code=400,
                    )
                candidates = data.get("candidates") or []
                if candidates:
                    candidate = candidates[0] or {}
                    content = candidate.get("content") or {}
                    for part in content.get("parts") or []:
                        text = part.get("text") if isinstance(part, dict) else None
                        if isinstance(text, str) and text:
                            yield StreamEvent("delta", text=text)
                    raw_reason = candidate.get("finishReason")
                    if raw_reason:
                        finish_reason = str(raw_reason)
                raw_usage = data.get("usageMetadata") or {}
                if isinstance(raw_usage, dict):
                    normalized = _normalize_usage(
                        {
                            "prompt_tokens": raw_usage.get("promptTokenCount"),
                            "completion_tokens": raw_usage.get("candidatesTokenCount"),
                            "total_tokens": raw_usage.get("totalTokenCount"),
                        }
                    )
                    if normalized:
                        usage.update(normalized)
        if usage:
            yield StreamEvent("usage", usage=usage)
        if cancel.is_set():
            yield StreamEvent("done", finish_reason="cancelled")
        else:
            mapping = {"STOP": "stop", "MAX_TOKENS": "length", "SAFETY": "content_filter"}
            yield StreamEvent(
                "done", finish_reason=mapping.get(finish_reason or "", finish_reason or "stop")
            )



