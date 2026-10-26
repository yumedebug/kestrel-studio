from __future__ import annotations

import json
from pathlib import Path

from app.core.settings import DEFAULT_SYSTEM_PROMPT, SettingsStore, normalize


def test_defaults_are_complete(settings: SettingsStore) -> None:
    assert settings.get("general.theme") == "light"
    assert settings.get("inference.backend") == "auto"
    assert settings.get("server.host") == "127.0.0.1"
    assert settings.get("server.port") == 1234
    assert settings.get("chat.system_prompt") == DEFAULT_SYSTEM_PROMPT
    assert settings.get("chat.seed") == -1
    assert str(settings.get("general.model_directory")).endswith("models")


def test_unknown_keys_are_preserved() -> None:
    data, warnings = normalize({"window": {"geometry": "abc"}, "custom": {"x": 1}})
    assert warnings == []
    assert data["window"]["geometry"] == "abc"
    assert data["custom"]["x"] == 1


def test_invalid_values_are_clamped_or_reset() -> None:
    data, warnings = normalize(
        {
            "general": {"theme": "neon", "font_size": "huge"},
            "inference": {"default_context_length": 5, "backend": "tpu"},
            "chat": {"temperature": 9.5, "top_p": -2, "max_tokens": 0},
            "server": {"port": 99999, "auto_start": "yes"},
        }
    )
    assert data["general"]["theme"] == "light"
    assert data["general"]["font_size"] == 10
    assert data["inference"]["default_context_length"] == 128
    assert data["inference"]["backend"] == "auto"
    assert data["chat"]["temperature"] == 2.0
    assert data["chat"]["top_p"] == 0.0
    assert data["chat"]["max_tokens"] == 1
    assert data["server"]["port"] == 65535
    assert data["server"]["auto_start"] is False
    assert len(warnings) >= 8


def test_set_validates_and_persists(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    store.set("chat.temperature", 1.25)
    store.set("chat.temperature", 5.0)  # clamped, not an exception
    assert store.get("chat.temperature") == 2.0
    store.set("window.geometry", "persisted")
    assert store.get("window.geometry") == "persisted"

    reloaded = SettingsStore(path)
    assert reloaded.get("chat.temperature") == 2.0
    assert reloaded.get("window.geometry") == "persisted"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["chat"]["temperature"] == 2.0


def test_cloud_provider_config_is_sanitized(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "cloud": {
                    "providers": {
                        "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini", "extra": 1},
                        "broken": "not-an-object",
                    },
                    "last_target": {"kind": "nonsense"},
                }
            }
        ),
        encoding="utf-8",
    )
    store = SettingsStore(path)
    providers = store.get("cloud.providers")
    assert providers == {
        "openai": {"base_url": "https://api.openai.com/v1", "default_model": "gpt-4o-mini"}
    }
    assert store.get("cloud.last_target") is None


def test_broken_json_is_backed_up_and_replaced(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = SettingsStore(path)
    assert store.get("general.theme") == "light"
    assert store.load_warnings and "could not be read" in store.load_warnings[0]
    assert path.with_suffix(".broken.json").exists()
    assert json.loads(path.read_text(encoding="utf-8"))["version"] == 1


def test_reset_restores_defaults(tmp_path: Path) -> None:
    store = SettingsStore(tmp_path / "settings.json")
    store.set("chat.temperature", 1.9)
    store.set("general.theme", "light")
    store.reset()
    assert store.get("chat.temperature") == 0.7
    assert store.get("general.theme") == "light"
