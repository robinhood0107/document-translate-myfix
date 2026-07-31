from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO


GGUF_MAGIC = b"GGUF"
GGUF_SUPPORTED_VERSIONS = frozenset({2, 3})
GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

_SCALAR_SIZES = {
    GGUF_VALUE_UINT8: 1,
    GGUF_VALUE_INT8: 1,
    GGUF_VALUE_UINT16: 2,
    GGUF_VALUE_INT16: 2,
    GGUF_VALUE_UINT32: 4,
    GGUF_VALUE_INT32: 4,
    GGUF_VALUE_FLOAT32: 4,
    GGUF_VALUE_BOOL: 1,
    GGUF_VALUE_UINT64: 8,
    GGUF_VALUE_INT64: 8,
    GGUF_VALUE_FLOAT64: 8,
}


class GGUFMetadataError(ValueError):
    pass


@dataclass(frozen=True)
class GGUFMetadataEntry:
    key: str
    value_type: int
    value_offset: int
    value: object


def _read_exact(stream: BinaryIO, size: int) -> bytes:
    payload = stream.read(size)
    if len(payload) != size:
        raise GGUFMetadataError("Unexpected end of GGUF metadata.")
    return payload


def _read_u32(stream: BinaryIO) -> int:
    return int(struct.unpack("<I", _read_exact(stream, 4))[0])


def _read_u64(stream: BinaryIO) -> int:
    return int(struct.unpack("<Q", _read_exact(stream, 8))[0])


def _read_string(stream: BinaryIO) -> str:
    length = _read_u64(stream)
    if length > 16 * 1024 * 1024:
        raise GGUFMetadataError("GGUF metadata string is unreasonably large.")
    try:
        return _read_exact(stream, length).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GGUFMetadataError("GGUF metadata contains invalid UTF-8.") from exc


def _skip_array_value(stream: BinaryIO, element_type: int, length: int) -> None:
    if length > 100_000_000:
        raise GGUFMetadataError("GGUF metadata array is unreasonably large.")
    if element_type in _SCALAR_SIZES:
        stream.seek(_SCALAR_SIZES[element_type] * length, os.SEEK_CUR)
        return
    if element_type == GGUF_VALUE_STRING:
        for _ in range(length):
            string_length = _read_u64(stream)
            stream.seek(string_length, os.SEEK_CUR)
        return
    if element_type == GGUF_VALUE_ARRAY:
        raise GGUFMetadataError("Nested GGUF metadata arrays are unsupported.")
    raise GGUFMetadataError(
        f"Unknown GGUF array element type: {element_type}"
    )


def _read_or_skip_value(
    stream: BinaryIO,
    value_type: int,
    *,
    capture: bool,
) -> object:
    if value_type in _SCALAR_SIZES:
        payload = _read_exact(stream, _SCALAR_SIZES[value_type])
        if not capture:
            return None
        formats = {
            GGUF_VALUE_UINT8: "<B",
            GGUF_VALUE_INT8: "<b",
            GGUF_VALUE_UINT16: "<H",
            GGUF_VALUE_INT16: "<h",
            GGUF_VALUE_UINT32: "<I",
            GGUF_VALUE_INT32: "<i",
            GGUF_VALUE_FLOAT32: "<f",
            GGUF_VALUE_BOOL: "<?",
            GGUF_VALUE_UINT64: "<Q",
            GGUF_VALUE_INT64: "<q",
            GGUF_VALUE_FLOAT64: "<d",
        }
        return struct.unpack(formats[value_type], payload)[0]
    if value_type == GGUF_VALUE_STRING:
        value = _read_string(stream)
        return value if capture else None
    if value_type == GGUF_VALUE_ARRAY:
        element_type = _read_u32(stream)
        length = _read_u64(stream)
        _skip_array_value(stream, element_type, length)
        return None
    raise GGUFMetadataError(f"Unknown GGUF metadata value type: {value_type}")


def find_metadata_entry(path: Path, key: str) -> GGUFMetadataEntry:
    target_key = str(key)
    with Path(path).open("rb") as stream:
        if _read_exact(stream, 4) != GGUF_MAGIC:
            raise GGUFMetadataError("Input is not a GGUF file.")
        version = _read_u32(stream)
        if version not in GGUF_SUPPORTED_VERSIONS:
            raise GGUFMetadataError(
                f"Unsupported GGUF version: {version}"
            )
        _tensor_count = _read_u64(stream)
        metadata_count = _read_u64(stream)
        if metadata_count > 1_000_000:
            raise GGUFMetadataError(
                "GGUF metadata registry is unreasonably large."
            )
        for _ in range(metadata_count):
            current_key = _read_string(stream)
            value_type = _read_u32(stream)
            value_offset = stream.tell()
            capture = current_key == target_key
            value = _read_or_skip_value(
                stream,
                value_type,
                capture=capture,
            )
            if capture:
                return GGUFMetadataEntry(
                    key=current_key,
                    value_type=value_type,
                    value_offset=value_offset,
                    value=value,
                )
    raise GGUFMetadataError(f"GGUF metadata key was not found: {target_key}")


def patch_uint32_metadata(
    path: Path,
    *,
    key: str,
    expected_value: int,
    replacement_value: int,
) -> None:
    entry = find_metadata_entry(path, key)
    if entry.value_type != GGUF_VALUE_UINT32:
        raise GGUFMetadataError(
            f"GGUF metadata {key!r} is not uint32."
        )
    if int(entry.value) != int(expected_value):
        raise GGUFMetadataError(
            f"GGUF metadata {key!r} is {entry.value}, expected "
            f"{expected_value}."
        )
    with Path(path).open("r+b") as stream:
        stream.seek(entry.value_offset)
        stream.write(struct.pack("<I", int(replacement_value)))
        stream.flush()
        os.fsync(stream.fileno())
    verified = find_metadata_entry(path, key)
    if int(verified.value) != int(replacement_value):
        raise GGUFMetadataError(
            f"Failed to verify patched GGUF metadata {key!r}."
        )
