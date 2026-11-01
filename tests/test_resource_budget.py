from app.core import resource_budget
from app.core.types import GenerationParams, LoadOptions


class Settings:
    def get(self, key: str, default=None):
        return {
            "inference.default_context_length": 8192,
            "inference.batch_size": 512,
            "inference.cpu_threads": 8,
        }.get(key, default)


def test_four_gb_budget_clamps_inference_options(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 4 * 1024**3)
    options = LoadOptions(
        context_length=8192, cpu_threads=8, batch_size=512
    ).resolved(Settings())
    assert options.context_length == 2048
    assert options.cpu_threads == 2
    assert options.batch_size == 128


def test_standard_budget_preserves_safe_defaults(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 16 * 1024**3)
    budget = resource_budget.detect_budget(Settings())
    assert budget.low_memory is False
    assert budget.context_length == 8192


def test_four_gb_budget_caps_generation(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 4 * 1024**3)
    params = GenerationParams.from_settings({"max_tokens": 4096})
    assert params.max_tokens == 512
