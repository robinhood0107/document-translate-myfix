from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from modules.utils.automatic_output import (
    build_output_file_name,
    default_global_output_settings,
    estimate_output_for_pages,
    format_estimate_ratio_text,
    format_estimate_seconds_text,
    resolve_automatic_output_settings,
    resolve_encode_settings,
    sanitize_series_folder_name,
    write_output_image,
)


class AutomaticOutputTests(unittest.TestCase):
    def test_series_folder_name_strips_trailing_version_suffix(self) -> None:
        self.assertEqual(
            sanitize_series_folder_name("My Series c12 v03"),
            "My Series",
        )
        self.assertEqual(
            sanitize_series_folder_name("Title[v2] c001"),
            "Title[v2]",
        )

    def test_resolved_settings_apply_project_override(self) -> None:
        settings = resolve_automatic_output_settings(
            default_global_output_settings(),
            {
                "output_format_override_mode": "project",
                "output_format_override_value": "webp",
                "output_preset_override_mode": "project",
                "output_preset_override_value": "small",
            },
        )
        self.assertEqual(settings["resolved_automatic_output_format"], "webp")
        self.assertEqual(settings["resolved_automatic_output_preset"], "small")

    def test_build_output_file_name_falls_back_to_png_for_unknown_same_as_source(self) -> None:
        file_name = build_output_file_name(
            "page001",
            "translated",
            "/tmp/page001.tiff",
            {"resolved_automatic_output_format": "same_as_source"},
        )
        self.assertEqual(file_name, "page001_translated.png")

    def test_write_output_image_creates_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "result.png")
            image = np.zeros((20, 10, 3), dtype=np.uint8)
            write_output_image(
                output_path,
                image,
                source_path="/tmp/source.png",
                resolved_settings={
                    "resolved_automatic_output_format": "png",
                    "resolved_automatic_output_preset": "balanced",
                    "automatic_output_png_compression_level": 6,
                    "automatic_output_jpg_quality": 90,
                    "automatic_output_webp_quality": 90,
                },
            )
            self.assertTrue(os.path.isfile(output_path))

    def test_estimate_output_for_pages_formats_text(self) -> None:
        estimate = estimate_output_for_pages(
            [
                {
                    "source_path": "/tmp/page1.png",
                    "byte_size": 1_000_000,
                    "megapixels": 2.5,
                }
            ],
            {
                "resolved_automatic_output_format": "png",
                "resolved_automatic_output_preset": "balanced",
                "automatic_output_png_compression_level": 6,
                "automatic_output_jpg_quality": 90,
                "automatic_output_webp_quality": 90,
            },
        )
        self.assertEqual(estimate["page_count"], 1)
        self.assertTrue(format_estimate_ratio_text(estimate).endswith("%"))
        self.assertTrue(format_estimate_seconds_text(estimate["seconds"]).endswith(("s", "m")))

    def test_resolve_encode_settings_uses_same_as_source_bmp(self) -> None:
        encode = resolve_encode_settings(
            {
                "resolved_automatic_output_format": "same_as_source",
                "resolved_automatic_output_preset": "balanced",
                "automatic_output_png_compression_level": 6,
                "automatic_output_jpg_quality": 90,
                "automatic_output_webp_quality": 90,
            },
            "/tmp/page.bmp",
        )
        self.assertEqual(encode["effective_format"], "bmp")
        self.assertEqual(encode["extension"], ".bmp")


if __name__ == "__main__":
    unittest.main()
