"""Registry of cloud inference providers.

All providers are reached through their own documented HTTP APIs; the app
never routes requests through a third-party gateway.  Model lists are hints
only and every UI field is editable + refreshable from the provider's API.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProviderSpec:
    id: str
    name: str
    kind: str  # openai | anthropic | gemini
    base_url: str
    default_model: str
    models: tuple[str, ...] = ()
    key_required: bool = True
    key_url: str = ""
    docs_url: str = ""
    supports_top_k: bool = False
    supports_repeat_penalty: bool = False
    supports_seed: bool = True
    include_usage: bool = False
    notes: str = ""
    local: bool = False

    @property
    def display_base_url(self) -> str:
        return self.base_url


PROVIDERS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        id="openai",
        name="OpenAI",
        kind="openai",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-mini",
        models=("gpt-4o-mini", "gpt-4o", "gpt-4.1", "gpt-4.1-mini", "o4-mini"),
        key_url="https://platform.openai.com/api-keys",
        docs_url="https://platform.openai.com/docs/api-reference",
        include_usage=True,
        notes="OpenAI does not accept top_k, repeat_penalty or every seed value.",
    ),
    "anthropic": ProviderSpec(
        id="anthropic",
        name="Anthropic (Claude)",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        default_model="claude-sonnet-4-5",
        models=("claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"),
        key_url="https://console.anthropic.com/settings/keys",
        docs_url="https://docs.anthropic.com/en/api/messages",
        supports_top_k=True,
        supports_seed=False,
        notes="The Messages API requires max_tokens and does not support seed or repeat_penalty.",
    ),
    "gemini": ProviderSpec(
        id="gemini",
        name="Google Gemini",
        kind="gemini",
        base_url="https://generativelanguage.googleapis.com/v1beta",
        default_model="gemini-2.5-flash",
        models=("gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"),
        key_url="https://aistudio.google.com/app/apikey",
        docs_url="https://ai.google.dev/api/generate-content",
        supports_top_k=True,
        supports_seed=True,
        notes="Gemini uses the API key as a query parameter; keys are stored securely.",
    ),
    "openrouter": ProviderSpec(
        id="openrouter",
        name="OpenRouter",
        kind="openai",
        base_url="https://openrouter.ai/api/v1",
        default_model="openai/gpt-4o-mini",
        models=("openai/gpt-4o-mini", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-flash"),
        key_url="https://openrouter.ai/keys",
        docs_url="https://openrouter.ai/docs/api-reference/overview",
        supports_top_k=True,
        supports_seed=True,
        include_usage=True,
        notes="One key, many upstream providers. Model IDs use the provider/model form.",
    ),
    "groq": ProviderSpec(
        id="groq",
        name="Groq",
        kind="openai",
        base_url="https://api.groq.com/openai/v1",
        default_model="llama-3.3-70b-versatile",
        models=("llama-3.3-70b-versatile", "llama-3.1-8b-instant"),
        key_url="https://console.groq.com/keys",
        docs_url="https://console.groq.com/docs/openai",
        include_usage=True,
        notes="Very fast OpenAI-compatible endpoint.",
    ),
    "mistral": ProviderSpec(
        id="mistral",
        name="Mistral AI",
        kind="openai",
        base_url="https://api.mistral.ai/v1",
        default_model="mistral-large-latest",
        models=("mistral-large-latest", "mistral-small-latest", "open-mistral-nemo"),
        key_url="https://console.mistral.ai/api-keys",
        docs_url="https://docs.mistral.ai/api/",
        supports_top_k=True,
        supports_seed=True,
        notes="OpenAI-compatible chat completions endpoint.",
    ),
    "deepseek": ProviderSpec(
        id="deepseek",
        name="DeepSeek",
        kind="openai",
        base_url="https://api.deepseek.com/v1",
        default_model="deepseek-chat",
        models=("deepseek-chat", "deepseek-reasoner"),
        key_url="https://platform.deepseek.com/api_keys",
        docs_url="https://api-docs.deepseek.com/",
        notes="Reasoning content is not displayed; only the final answer is streamed.",
    ),
    "xai": ProviderSpec(
        id="xai",
        name="xAI (Grok)",
        kind="openai",
        base_url="https://api.x.ai/v1",
        default_model="grok-4",
        models=("grok-4", "grok-3", "grok-3-mini"),
        key_url="https://console.x.ai/",
        docs_url="https://docs.x.ai/docs/api-reference",
        supports_top_k=True,
        supports_seed=True,
        notes="OpenAI-compatible endpoint.",
    ),
    "together": ProviderSpec(
        id="together",
        name="Together AI",
        kind="openai",
        base_url="https://api.together.xyz/v1",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        models=(
            "meta-llama/Llama-3.3-70B-Instruct-Turbo",
            "Qwen/Qwen2.5-72B-Instruct-Turbo",
        ),
        key_url="https://api.together.xyz/settings/api-keys",
        docs_url="https://docs.together.ai/reference/chat-completions-1",
        supports_top_k=True,
        supports_seed=True,
        notes="OpenAI-compatible endpoint.",
    ),
    "ollama": ProviderSpec(
        id="ollama",
        name="Ollama (local)",
        kind="openai",
        base_url="http://127.0.0.1:11434/v1",
        default_model="llama3.2",
        models=("llama3.2", "qwen2.5", "mistral"),
        key_required=False,
        docs_url="https://github.com/ollama/ollama/blob/main/docs/openai.md",
        supports_top_k=True,
        supports_seed=True,
        local=True,
        notes="Talks to an Ollama daemon on this machine through its OpenAI-compatible API.",
    ),
    "lmstudio": ProviderSpec(
        id="lmstudio",
        name="LM Studio (local)",
        kind="openai",
        base_url="http://127.0.0.1:1234/v1",
        default_model="local-model",
        models=(),
        key_required=False,
        docs_url="https://lmstudio.ai/docs/api",
        supports_top_k=True,
        supports_repeat_penalty=True,
        supports_seed=True,
        local=True,
        notes="Optional. Uses LM Studio's server; port 1234 collides with this app's default API port.",
    ),
    "custom": ProviderSpec(
        id="custom",
        name="Custom OpenAI-compatible",
        kind="openai",
        base_url="",
        default_model="",
        models=(),
        key_required=False,
        docs_url="",
        supports_top_k=True,
        supports_repeat_penalty=True,
        supports_seed=True,
        notes="Any endpoint that implements POST /chat/completions with SSE streaming. Great for local mock servers.",
    ),
}

# Providers shown in the settings list, in order.
PROVIDER_ORDER = (
    "openai",
    "anthropic",
    "gemini",
    "openrouter",
    "groq",
    "mistral",
    "deepseek",
    "xai",
    "together",
    "ollama",
    "lmstudio",
    "custom",
)

KIND_LABELS = {
    "openai": "OpenAI-compatible",
    "anthropic": "Anthropic Messages",
    "gemini": "Google Gemini",
}


def get_provider(provider_id: str) -> ProviderSpec | None:
    return PROVIDERS.get(provider_id)


def credential_key(provider_id: str) -> str:
    return f"provider:{provider_id}:api_key"


def provider_models(provider_id: str, config: dict | None = None) -> tuple[str, ...]:
    spec = PROVIDERS.get(provider_id)
    if not spec:
        return ()
    default_model = (config or {}).get("default_model", "")
    models = list(spec.models)
    if default_model and default_model not in models:
        models.insert(0, default_model)
    return tuple(models)
