"""Fetch model lists from cloud providers for the model picker."""

from __future__ import annotations

from typing import Any

import httpx

from app.core.backends.http_utils import error_from_body, extract_error_message
from app.core.credentials import CredentialStore
from app.core.providers import credential_key, get_provider
from app.core.settings import SettingsStore
from app.core.types import BackendError


def provider_config(settings: SettingsStore, provider_id: str) -> dict[str, str]:
    cfg = settings.get(f"cloud.providers.{provider_id}", {}) or {}
    if not isinstance(cfg, dict):
        cfg = {}
    return {
        "base_url": str(cfg.get("base_url") or ""),
        "default_model": str(cfg.get("default_model") or ""),
    }


def fetch_models(
    provider_id: str,
    settings: SettingsStore,
    credentials: CredentialStore,
    *,
    transport: httpx.BaseTransport | None = None,
) -> list[str]:
    spec = get_provider(provider_id)
    if spec is None:
        raise BackendError(f"Unknown provider {provider_id!r}.", kind="invalid_request")
    cfg = provider_config(settings, provider_id)
    base_url = (cfg["base_url"] or spec.base_url).rstrip("/")
    if not base_url:
        raise BackendError("Set a base URL for this provider first.", kind="invalid_request")
    api_key = credentials.get(credential_key(provider_id)) or ""
    if spec.key_required and not api_key:
        raise BackendError(
            f"No API key saved for {spec.name}. Add it in Settings → Cloud Providers.",
            kind="auth_error",
            status_code=401,
        )

    headers: dict[str, str] = {}
    url = ""
    if spec.kind == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        url = f"{base_url}/v1/models" if not base_url.endswith("/v1") else f"{base_url}/models"
    elif spec.kind == "gemini":
        headers = {"x-goog-api-key": api_key}
        url = f"{base_url}/models?pageSize=200"
    else:
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{base_url}/models"

    try:
        with httpx.Client(timeout=20.0, transport=transport, follow_redirects=True) as client:
            response = client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise BackendError(
            f"Could not reach {spec.name}: {exc}", kind="network_error", status_code=502
        ) from exc
    if response.status_code >= 400:
        raise error_from_body(response.status_code, response.text, spec.name)
    try:
        payload = response.json()
    except ValueError as exc:
        raise BackendError(
            f"{spec.name} returned an unexpected response.",
            kind="provider_error",
            status_code=502,
            detail=extract_error_message(response.text),
        ) from exc

    models: list[str] = []
    if spec.kind == "gemini":
        for item in payload.get("models", []) if isinstance(payload, dict) else []:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods") or []
            if methods and "generateContent" not in methods:
                continue
            name = str(item.get("name") or "")
            if name.startswith("models/"):
                name = name[len("models/") :]
            if name:
                models.append(name)
    else:
        data = payload.get("data", []) if isinstance(payload, dict) else []
        for item in data:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
    # Local providers sometimes return entries with prefixes like "models/".
    cleaned = []
    seen = set()
    for model in models:
        if model.startswith("models/"):
            model = model[len("models/") :]
        if model and model not in seen:
            seen.add(model)
            cleaned.append(model)
    cleaned.sort()
    return cleaned
