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
    assert options.context_length == 1024
    assert options.cpu_threads == 1
    assert options.batch_size == 32


def test_standard_budget_preserves_safe_defaults(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 16 * 1024**3)
    monkeypatch.setattr(resource_budget, "logical_cpu_count", lambda: 8)
    budget = resource_budget.detect_budget(Settings())
    assert budget.low_memory is False
    assert budget.context_length == 8192


def test_four_gb_budget_caps_generation(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 4 * 1024**3)
    params = GenerationParams.from_settings({"max_tokens": 4096})
    assert params.max_tokens == 256


def test_two_cpu_budget_is_celeron_safe(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(resource_budget, "logical_cpu_count", lambda: 2)
    budget = resource_budget.detect_budget(Settings())
    assert budget.ultra_low is True
    assert budget.cpu_threads == 1
    assert budget.context_length == 1024
    assert budget.batch_size == 32


def test_two_cpu_load_options_are_clamped(monkeypatch):
    monkeypatch.setattr(resource_budget, "physical_memory_bytes", lambda: 8 * 1024**3)
    monkeypatch.setattr(resource_budget, "logical_cpu_count", lambda: 2)
    options = LoadOptions(context_length=8192, cpu_threads=8, batch_size=512).resolved(Settings())
    assert options.context_length == 1024
    assert options.cpu_threads == 1
    assert options.batch_size == 32
