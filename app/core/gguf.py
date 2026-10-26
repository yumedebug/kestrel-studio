"""Minimal, defensive GGUF header/metadata reader.

Only the metadata header and tensor index are read; tensor data is never
loaded.  The parser is deliberately tolerant: a damaged or unsupported file
returns ``GGUFInfo(ok=False, error=...)`` instead of raising.
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

GGUF_MAGIC = b"GGUF"

# metadata value type -> struct format / special
_SIZES = {
    0: 1,   # uint8
    1: 1,   # int8
    2: 2,   # uint16
    3: 2,   # int16
    4: 4,   # uint32
    5: 4,   # int32
    6: 4,   # float32
    7: 1,   # bool
    8: None,  # string
    9: None,  # array
    10: 8,  # uint64
    11: 8,  # int64
    12: 8,  # float64
}

_QUANT_RE = re.compile(
    r"(?i)(?<![A-Z0-9])(I?Q\d+(?:_[A-Z0-9]+)*|BF16|F16|F32|FP16)(?![A-Z0-9])"
)
_FILE_TYPE_QUANT = {
    0: "F32",
    1: "F16",
    2: "Q4_0",
    3: "Q4_1",
    7: "Q8_0",
    8: "Q5_0",
    9: "Q5_1",
    10: "Q2_K",
    11: "Q3_K_S",
    12: "Q3_K_M",
    13: "Q3_K_L",
    14: "Q4_K_S",
    15: "Q4_K_M",
    16: "Q5_K_S",
    17: "Q5_K_M",
    18: "Q6_K",
    19: "IQ2_XXS",
    20: "IQ2_XS",
    21: "Q2_K_S",
    22: "IQ3_XS",
    23: "IQ3_XXS",
    24: "IQ1_S",
    25: "IQ4_NL",
    26: "IQ3_S",
    27: "IQ3_M",
    28: "IQ2_S",
    29: "IQ2_M",
    30: "IQ4_XS",
    31: "IQ1_M",
    32: "BF16",
    36: "TQ1_0",
    37: "TQ2_0",
}

_WANTED_SUFFIXES = (
    ".context_length",
    ".embedding_length",
    ".block_count",
    ".feed_forward_length",
    ".attention.head_count",
    ".attention.head_count_kv",
)


class _Reader:
    def __init__(self, fh: BinaryIO):
        self.fh = fh
        self.pos = 0

    def read(self, n: int) -> bytes:
        if n < 0:
            raise ValueError("negative read")
        data = self.fh.read(n)
        self.pos += len(data)
        if len(data) != n:
            raise EOFError("unexpected end of file")
        return data

    def skip(self, n: int) -> None:
        if n < 0:
            raise ValueError("negative skip")
        self.fh.seek(n, 1)
        self.pos += n

    def u32(self) -> int:
        return struct.unpack("<I", self.read(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.read(8))[0]

    def string(self, max_bytes: int = 64 * 1024 * 1024) -> str:
        length = self.u64()
        if length > max_bytes:
            raise ValueError(f"string too large ({length} bytes)")
        return self.read(length).decode("utf-8", errors="replace")


def _skip_array(reader: _Reader, elem_type: int, count: int, depth: int = 0) -> None:
    """Advance past an array without materialising it."""
    if depth > 4:
        raise ValueError("arrays nested too deeply")
    if count < 0 or count > 1_000_000_000:
        raise ValueError("implausible array length")
    if elem_type == 8:  # string
        for _ in range(count):
            length = reader.u64()
            if length > 64 * 1024 * 1024:
                raise ValueError("string in array too large")
            reader.skip(length)
    elif elem_type == 9:
        for _ in range(count):
            child_type = reader.u32()
            child_count = reader.u64()
            _skip_array(reader, child_type, child_count, depth + 1)
    else:
        size = _SIZES.get(elem_type)
        if size is None:
            raise ValueError(f"unknown GGUF array type {elem_type}")
        reader.skip(size * count)


def _read_scalar(reader: _Reader, value_type: int) -> object:
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        raise ValueError("unexpected array")
    if value_type == 7:
        return bool(reader.read(1)[0])
    fmt = {
        0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
        6: "<f", 10: "<Q", 11: "<q", 12: "<d",
    }.get(value_type)
    if fmt is None:
        raise ValueError(f"unknown GGUF value type {value_type}")
    size = _SIZES[value_type]
    return struct.unpack(fmt, reader.read(size))[0]


@dataclass
class GGUFInfo:
    path: str = ""
    ok: bool = False
    error: str = ""
    version: int = 0
    tensor_count: int = 0
    metadata_count: int = 0
    architecture: str = ""
    name: str = ""
    context_length: int | None = None
    embedding_length: int | None = None
    block_count: int | None = None
    head_count: int | None = None
    parameter_count: int | None = None
    quant: str = ""
    has_chat_template: bool = False
    chat_template_length: int = 0
    file_size: int = 0
    file_type: int | None = None
    selected_metadata: dict[str, object] = field(default_factory=dict)

    @property
    def context_display(self) -> str:
        return f"{self.context_length:,}" if self.context_length else ""

    @property
    def parameters_display(self) -> str:
        count = self.parameter_count
        if not count:
            return ""
        if count >= 1_000_000_000:
            return f"{count / 1_000_000_000:.1f}B"
        if count >= 1_000_000:
            return f"{count / 1_000_000:.0f}M"
        return f"{count:,}"


def quant_from_filename(filename: str) -> str:
    stem = Path(filename).name
    matches = _QUANT_RE.findall(stem)
    if not matches:
        return ""
    # Prefer the last match; names often include base quant first.
    return matches[-1].upper()


def read_gguf(path: Path | str, max_metadata: int = 100_000) -> GGUFInfo:
    path = Path(path)
    info = GGUFInfo(path=str(path), quant=quant_from_filename(path.name))
    try:
        info.file_size = path.stat().st_size
    except OSError as exc:
        info.error = f"File is not accessible: {exc}"
        return info

    try:
        with path.open("rb") as fh:
            reader = _Reader(fh)
            magic = reader.read(4)
            if magic != GGUF_MAGIC:
                info.error = "Not a GGUF file (missing GGUF magic bytes)"
                return info
            info.version = reader.u32()
            if info.version not in (1, 2, 3):
                info.error = f"Unsupported GGUF version {info.version}"
                return info
            if info.version == 1:
                info.tensor_count = reader.u32()
                info.metadata_count = reader.u32()
            else:
                info.tensor_count = reader.u64()
                info.metadata_count = reader.u64()

            if info.metadata_count > max_metadata:
                info.error = (
                    f"Implausible metadata count ({info.metadata_count:,}); file may be corrupt"
                )
                return info

            raw: dict[str, object] = {}
            for _ in range(info.metadata_count):
                key = reader.string(max_bytes=4096)
                value_type = reader.u32()
                if value_type == 9:
                    elem_type = reader.u32()
                    count = reader.u64()
                    # We are interested in scalar metadata only; skip arrays.
                    _skip_array(reader, elem_type, count)
                    continue
                value = _read_scalar(reader, value_type)
                if isinstance(value, str) and len(value) > 4096:
                    # Chat templates are useful to detect but not to display.
                    if key == "tokenizer.chat_template":
                        raw[key] = value
                    continue
                raw[key] = value

            architecture = str(raw.get("general.architecture", "") or "")
            info.architecture = architecture
            info.name = str(
                raw.get("general.name")
                or raw.get("general.basename")
                or path.stem
            )
            info.file_type = raw.get("general.file_type") if isinstance(
                raw.get("general.file_type"), int
            ) else None

            def find_stat(suffix: str) -> int | None:
                for candidate_key in (f"{architecture}{suffix}", suffix.lstrip(".")):
                    value = raw.get(candidate_key)
                    if isinstance(value, int):
                        return value
                for key, value in raw.items():
                    if key.endswith(suffix) and isinstance(value, int):
                        return value
                return None

            info.context_length = find_stat(".context_length")
            info.embedding_length = find_stat(".embedding_length")
            info.block_count = find_stat(".block_count")
            info.head_count = find_stat(".attention.head_count")
            template = raw.get("tokenizer.chat_template")
            if isinstance(template, str):
                info.has_chat_template = True
                info.chat_template_length = len(template)
            if not info.quant and info.file_type in _FILE_TYPE_QUANT:
                info.quant = _FILE_TYPE_QUANT[info.file_type]

            # Tensor index follows metadata.  It lets us compute parameter count
            # without reading any weights.
            total = 0
            tensors_read = 0
            try:
                for _ in range(min(info.tensor_count, 200_000)):
                    reader.string(max_bytes=4096)  # tensor name
                    n_dims = reader.u32()
                    if n_dims > 8:
                        raise ValueError("implausible tensor rank")
                    dims = [reader.u64() for _ in range(n_dims)]
                    reader.u32()  # ggml type
                    reader.u64()  # offset
                    elements = 1
                    for dim in dims:
                        elements *= max(1, dim)
                    total += elements
                    tensors_read += 1
                if tensors_read:
                    info.parameter_count = total
            except (EOFError, ValueError, struct.error):
                # A truncated tensor index does not invalidate the metadata.
                pass

            info.selected_metadata = {
                key: value
                for key, value in raw.items()
                if key in {
                    "general.architecture",
                    "general.name",
                    "general.basename",
                    "general.size_label",
                    "general.license",
                    "general.license.name",
                    "general.license.link",
                    "general.file_type",
                    "tokenizer.ggml.model",
                    "tokenizer.ggml.pre",
                }
                or key.endswith(_WANTED_SUFFIXES)
            }
            if "tokenizer.chat_template" in raw:
                info.selected_metadata["tokenizer.chat_template"] = "<present>"
            if not info.architecture:
                info.error = "GGUF metadata has no architecture"
                return info
            info.ok = True
            return info
    except (OSError, EOFError, ValueError, struct.error, OverflowError) as exc:
        info.error = f"Corrupt or truncated GGUF file: {exc}"
        return info


def is_probably_gguf(path: Path | str) -> bool:
    try:
        with Path(path).open("rb") as fh:
            return fh.read(4) == GGUF_MAGIC
    except OSError:
        return False
