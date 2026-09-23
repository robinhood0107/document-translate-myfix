from __future__ import annotations

import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from modules.utils import image_safety


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _write_header_only_png(path: Path, width: int, height: int) -> None:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IEND", b"")
    )


class ImageSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        image_safety._inspect_cached.cache_clear()

    def test_one_hundred_million_pixel_page_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, "page.png")
            _write_header_only_png(path, 10_000, 10_000)
            self.assertEqual(image_safety.inspect_image_dimensions(path), (10_000, 10_000))

    def test_two_hundred_million_pixel_boundary_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, "page.png")
            _write_header_only_png(path, 10_000, 20_000)
            self.assertEqual(image_safety.inspect_image_dimensions(path), (10_000, 20_000))

    def test_page_above_two_hundred_million_pixels_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, "page.png")
            _write_header_only_png(path, 10_001, 20_000)
            with self.assertRaises(image_safety.ImageResourceLimitError):
                image_safety.inspect_image_dimensions(path)

    def test_aggregate_size_does_not_create_a_workflow_signal(self) -> None:
        resources = [
            (f"page-{index}.png", 2_400, 1_700)
            for index in range(93)
        ]
        plan = image_safety.build_image_resource_plan(
            resources,
            available_memory_bytes=8 * 1024**3,
        )

        self.assertIsInstance(plan, image_safety.ImageResourcePlan)
        self.assertEqual(plan.page_count, 93)
        self.assertEqual(plan.largest_pixels, 4_080_000)
        self.assertEqual(plan.largest_page_peak_bytes, 61_200_000)
        self.assertTrue(plan.hard_cap_passed)
        self.assertFalse(hasattr(plan, "streaming"))

    def test_single_page_above_available_ram_threshold_is_rejected(self) -> None:
        with self.assertRaises(image_safety.ImageResourceLimitError):
            image_safety.build_image_resource_plan(
                [("page.png", 10_000, 10_000)],
                available_memory_bytes=2_000_000_000,
            )
