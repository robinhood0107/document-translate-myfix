from __future__ import annotations

import struct
import tempfile
import unittest
from pathlib import Path

from modules.ocr.paddleocr_vl_spotting.gguf_metadata import (
    GGUFMetadataError,
    GGUF_VALUE_STRING,
    GGUF_VALUE_UINT32,
    find_metadata_entry,
    patch_uint32_metadata,
)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return struct.pack("<Q", len(encoded)) + encoded


def _fixture() -> bytes:
    entries = (
        (
            "general.name",
            GGUF_VALUE_STRING,
            _gguf_string("PaddleOCR-VL"),
        ),
        (
            "clip.vision.image_max_pixels",
            GGUF_VALUE_UINT32,
            struct.pack("<I", 1_003_520),
        ),
    )
    payload = bytearray(b"GGUF")
    payload += struct.pack("<I", 3)
    payload += struct.pack("<Q", 0)
    payload += struct.pack("<Q", len(entries))
    for key, value_type, value in entries:
        payload += _gguf_string(key)
        payload += struct.pack("<I", value_type)
        payload += value
    return bytes(payload)


class PaddleSpottingGGUFMetadataTests(unittest.TestCase):
    def test_reads_and_patches_only_the_uint32_metadata_value(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmproj.gguf"
            path.write_bytes(_fixture())
            original_size = path.stat().st_size

            entry = find_metadata_entry(
                path,
                "clip.vision.image_max_pixels",
            )
            self.assertEqual(entry.value, 1_003_520)
            patch_uint32_metadata(
                path,
                key="clip.vision.image_max_pixels",
                expected_value=1_003_520,
                replacement_value=1_605_632,
            )
            updated = find_metadata_entry(
                path,
                "clip.vision.image_max_pixels",
            )

            self.assertEqual(updated.value, 1_605_632)
            self.assertEqual(path.stat().st_size, original_size)
            self.assertEqual(
                find_metadata_entry(path, "general.name").value,
                "PaddleOCR-VL",
            )

    def test_refuses_wrong_expected_value_and_missing_key(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "mmproj.gguf"
            path.write_bytes(_fixture())
            with self.assertRaisesRegex(GGUFMetadataError, "expected"):
                patch_uint32_metadata(
                    path,
                    key="clip.vision.image_max_pixels",
                    expected_value=123,
                    replacement_value=1_605_632,
                )
            with self.assertRaisesRegex(GGUFMetadataError, "not found"):
                find_metadata_entry(path, "missing.key")


if __name__ == "__main__":
    unittest.main()
