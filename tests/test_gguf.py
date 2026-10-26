from __future__ import annotations

from pathlib import Path

from app.core.gguf import (
    GGUF_MAGIC,
    is_probably_gguf,
    quant_from_filename,
    read_gguf,
)
from tests.conftest import write_gguf


def test_reads_metadata_and_parameter_count(gguf_file) -> None:
    path = gguf_file(tensors=[("token_embd.weight", (2048, 1000)), ("blk.0.attn_q.weight", (2048, 2048))])
    info = read_gguf(path)
    assert info.ok, info.error
    assert info.version == 3
    assert info.architecture == "llama"
    assert info.name == "Tiny Test Model"
    assert info.context_length == 4096
    assert info.embedding_length == 2048
    assert info.block_count == 22
    assert info.quant == "Q4_K_M"  # from filename
    assert info.has_chat_template is False
    assert info.parameter_count == 2048 * 1000 + 2048 * 2048
    assert info.parameters_display.endswith("M")
    assert info.context_display == "4,096"


def test_file_type_supplies_quant_when_filename_lacks_it(gguf_file) -> None:
    path = gguf_file("model.gguf")
    info = read_gguf(path)
    assert info.ok
    assert info.quant == "Q4_K_M"


def test_chat_template_is_detected_but_not_exposed(gguf_file) -> None:
    template = "{% for message in messages %}" + ("{{ message.content }}" * 400)
    path = gguf_file("chatty.Q8_0.gguf", metadata={"tokenizer.chat_template": template})
    info = read_gguf(path)
    assert info.ok
    assert info.has_chat_template is True
    assert info.chat_template_length == len(template)
    assert info.selected_metadata["tokenizer.chat_template"] == "<present>"


def test_rejects_non_gguf(tmp_path: Path) -> None:
    path = tmp_path / "text.gguf"
    path.write_bytes(b"this is not a model")
    info = read_gguf(path)
    assert not info.ok
    assert "GGUF" in info.error
    assert is_probably_gguf(path) is False


def test_rejects_unsupported_version(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "future.gguf", {"general.architecture": "llama"}, version=99)
    info = read_gguf(path)
    assert not info.ok
    assert "Unsupported GGUF version" in info.error


def test_rejects_truncated_file(tmp_path: Path) -> None:
    path = tmp_path / "truncated.gguf"
    path.write_bytes(GGUF_MAGIC + b"\x03\x00\x00\x00" + b"\x00" * 4)
    info = read_gguf(path)
    assert not info.ok
    assert "Corrupt or truncated" in info.error


def test_rejects_missing_architecture(tmp_path: Path) -> None:
    path = write_gguf(tmp_path / "noarch.gguf", {"general.name": "Mystery"})
    info = read_gguf(path)
    assert not info.ok
    assert "no architecture" in info.error


def test_implausible_metadata_count_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bomb.gguf"
    path.write_bytes(GGUF_MAGIC + b"\x03\x00\x00\x00" + b"\x00" * 8 + (10**9).to_bytes(8, "little"))
    info = read_gguf(path)
    assert not info.ok
    assert "Implausible metadata count" in info.error


def test_quant_from_filename_variants() -> None:
    assert quant_from_filename("Llama-3.1-8B-Instruct.Q4_K_M.gguf") == "Q4_K_M"
    assert quant_from_filename("model-iq2_xxs.gguf") == "IQ2_XXS"
    assert quant_from_filename("weights-bf16.gguf") == "BF16"
    assert quant_from_filename("plain-model.gguf") == ""


def test_unreadable_path_returns_error(tmp_path: Path) -> None:
    info = read_gguf(tmp_path / "does-not-exist.gguf")
    assert not info.ok
    assert "not accessible" in info.error
