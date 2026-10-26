from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.core.model_manager import ModelManager, ModelManagerError
from app.core.types import LoadOptions


def test_add_and_persist_model(gguf_file, tmp_path: Path) -> None:
    path = gguf_file("Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf")
    library_path = tmp_path / "library.json"
    manager = ModelManager(library_path, tmp_path / "models")

    entry = manager.add_file(path, source_repo="acme/tiny", source_file=path.name)
    assert entry.status == "ready"
    assert entry.architecture == "llama"
    assert entry.quant == "Q4_K_M"
    assert entry.context_length == 4096
    assert entry.size == path.stat().st_size
    assert entry.has_chat_template is False
    assert entry.source_repo == "acme/tiny"

    reloaded = ModelManager(library_path)
    assert reloaded.count() == 1
    saved = reloaded.entries()[0]
    assert saved.path == str(path.resolve())
    assert saved.name == "Tiny Test Model"


def test_duplicate_add_updates_existing_entry(gguf_file, tmp_path: Path) -> None:
    path = gguf_file("dupe.Q8_0.gguf")
    manager = ModelManager(tmp_path / "library.json", tmp_path / "models")
    first = manager.add_file(path)
    second = manager.add_file(path)
    assert first.id == second.id
    assert manager.count() == 1


def test_non_gguf_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")
    manager = ModelManager(tmp_path / "library.json")
    with pytest.raises(ModelManagerError, match=".gguf"):
        manager.add_file(path)


def test_corrupt_gguf_is_rejected_unless_allowed(tmp_path: Path) -> None:
    path = tmp_path / "broken.gguf"
    path.write_bytes(b"not a model")
    manager = ModelManager(tmp_path / "library.json")
    with pytest.raises(ModelManagerError):
        manager.add_file(path)

    entry = manager.add_file(path, allow_invalid=True)
    assert entry.status == "invalid"
    assert "GGUF" in entry.metadata_error


def test_refresh_scans_directory_and_skips_part(gguf_file, tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    good = gguf_file("nested/model.Q4_K_M.gguf")
    # gguf_file writes into tmp_path; move it under the managed directory.
    target = model_dir / "nested" / good.name
    target.parent.mkdir(parents=True, exist_ok=True)
    good.replace(target)
    (model_dir / "still-downloading.gguf.part").write_bytes(b"GGUF")

    manager = ModelManager(tmp_path / "library.json", model_dir)
    summary = manager.refresh()
    assert summary.added == 1
    assert summary.invalid == 0
    assert [entry.name for entry in manager.entries()] == ["Tiny Test Model"]

    untouched = manager.refresh()
    assert untouched.added == 0
    assert untouched.updated == 0


def test_refresh_reports_invalid_models(gguf_file, tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "bad.gguf").write_bytes(b"GGUF\x03\x00")
    manager = ModelManager(tmp_path / "library.json", model_dir)
    summary = manager.refresh()
    assert summary.invalid == 1
    entry = manager.entries()[0]
    assert entry.status == "invalid"


def test_remove_from_library_keeps_file(gguf_file, tmp_path: Path) -> None:
    path = gguf_file()
    manager = ModelManager(tmp_path / "library.json")
    entry = manager.add_file(path)
    manager.remove_from_library(entry.id)
    assert manager.count() == 0
    assert path.exists()
    assert path.read_bytes().startswith(b"GGUF")


def test_delete_file_removes_file_and_entry(gguf_file, tmp_path: Path) -> None:
    path = gguf_file()
    manager = ModelManager(tmp_path / "library.json")
    entry = manager.add_file(path)
    manager.delete_file(entry.id)
    assert manager.count() == 0
    assert not path.exists()


def test_load_options_roundtrip(gguf_file, tmp_path: Path) -> None:
    path = gguf_file()
    library_path = tmp_path / "library.json"
    manager = ModelManager(library_path)
    entry = manager.add_file(path)
    options = LoadOptions(context_length=8192, gpu_layers=12, cpu_threads=6, batch_size=256)
    manager.save_load_options(entry.id, options, base={"note": "keep"})

    reloaded = ModelManager(library_path).get(entry.id)
    assert reloaded is not None
    resolved = reloaded.resolved_options()
    assert resolved.context_length == 8192
    assert resolved.gpu_layers == 12
    assert resolved.cpu_threads == 6
    assert resolved.batch_size == 256
    assert reloaded.load_options["note"] == "keep"


def test_corrupt_library_json_is_backed_up(gguf_file, tmp_path: Path) -> None:
    library_path = tmp_path / "library.json"
    library_path.write_text("{ nope", encoding="utf-8")
    manager = ModelManager(library_path)
    assert manager.count() == 0
    assert library_path.with_suffix(".broken.json").exists()


def test_missing_status_and_remove_missing(gguf_file, tmp_path: Path) -> None:
    path = gguf_file()
    manager = ModelManager(tmp_path / "library.json")
    entry = manager.add_file(path)
    path.unlink()
    assert manager.get(entry.id).status == "missing"
    assert manager.remove_missing() == 1
    assert manager.count() == 0


def test_library_file_shape(gguf_file, tmp_path: Path) -> None:
    path = gguf_file()
    library_path = tmp_path / "library.json"
    ModelManager(library_path).add_file(path)
    raw = json.loads(library_path.read_text(encoding="utf-8"))
    assert raw["version"] == 1
    assert raw["models"][0]["name"] == "Tiny Test Model"
