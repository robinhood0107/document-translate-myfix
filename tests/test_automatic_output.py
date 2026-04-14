from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from modules.utils.automatic_output import (
    OUTPUT_ARCHIVE_FORMAT_CBZ,
    OUTPUT_ARCHIVE_FORMAT_ZIP,
    OUTPUT_IMAGE_FORMAT_JPG,
    OUTPUT_IMAGE_FORMAT_PNG,
    OUTPUT_IMAGE_FORMAT_SAME,
    OUTPUT_IMAGE_FORMAT_WEBP,
    OUTPUT_TARGET_ARCHIVE,
    build_archive_file_name,
    build_archive_page_file_name,
    build_output_file_name,
    default_global_output_settings,
    estimate_archive_options_for_pages,
    format_estimate_ratio_text,
    format_estimate_seconds_text,
    format_estimate_size_text,
    resolve_automatic_output_settings,
    sanitize_series_folder_name,
    write_archive_image,
    write_output_image,
)


class AutomaticOutputTests(unittest.TestCase):
    def test_series_folder_name_strips_trailing_version_suffix(self) -> None:
        self.assertEqual(sanitize_series_folder_name("My Series c12 v03"), "My Series")
        self.assertEqual(sanitize_series_folder_name("Title[v2] c001"), "Title[v2]")

    def test_resolved_settings_apply_project_override(self) -> None:
        settings = resolve_automatic_output_settings(
            default_global_output_settings(),
            {
                "output_use_global": False,
                "output_target": OUTPUT_TARGET_ARCHIVE,
                "output_archive_format": OUTPUT_ARCHIVE_FORMAT_ZIP,
                "output_archive_image_format": OUTPUT_IMAGE_FORMAT_WEBP,
                "output_archive_compression_level": 9,
            },
        )
        self.assertEqual(settings["resolved_automatic_output_target"], OUTPUT_TARGET_ARCHIVE)
        self.assertEqual(settings["resolved_automatic_output_archive_format"], OUTPUT_ARCHIVE_FORMAT_ZIP)
        self.assertEqual(
            settings["resolved_automatic_output_archive_image_format"],
            OUTPUT_IMAGE_FORMAT_WEBP,
        )
        self.assertEqual(settings["resolved_automatic_output_archive_compression_level"], 9)

    def test_build_output_file_name_falls_back_to_png_for_unknown_same_as_source(self) -> None:
        file_name = build_output_file_name(
            "page001",
            "translated",
            "/tmp/page001.tiff",
            {"resolved_automatic_output_image_format": OUTPUT_IMAGE_FORMAT_SAME},
        )
        self.assertEqual(file_name, "page001_translated.png")

    def test_build_output_file_name_keeps_same_as_source_bmp(self) -> None:
        file_name = build_output_file_name(
            "page001",
            "translated",
            "/tmp/page001.bmp",
            {"resolved_automatic_output_image_format": OUTPUT_IMAGE_FORMAT_SAME},
        )
        self.assertEqual(file_name, "page001_translated.bmp")

    def test_write_output_image_creates_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "result.jpg")
            image = np.zeros((20, 10, 3), dtype=np.uint8)
            write_output_image(
                output_path,
                image,
                source_path="/tmp/source.png",
                resolved_settings={
                    "resolved_automatic_output_image_format": OUTPUT_IMAGE_FORMAT_JPG,
                },
            )
            self.assertTrue(os.path.isfile(output_path))

    def test_write_archive_image_creates_requested_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, "result.webp")
            image = np.zeros((20, 10, 3), dtype=np.uint8)
            write_archive_image(
                output_path,
                image,
                resolved_settings={
                    "resolved_automatic_output_archive_image_format": OUTPUT_IMAGE_FORMAT_WEBP,
                },
            )
            self.assertTrue(os.path.isfile(output_path))

    def test_archive_estimates_cover_all_formats(self) -> None:
        estimates = estimate_archive_options_for_pages(
            [
                {
                    "source_path": "/tmp/page1.png",
                    "byte_size": 1_000_000,
                    "megapixels": 2.5,
                }
            ],
            6,
        )
        self.assertEqual(set(estimates.keys()), {OUTPUT_IMAGE_FORMAT_PNG, OUTPUT_IMAGE_FORMAT_JPG, OUTPUT_IMAGE_FORMAT_WEBP})
        self.assertTrue(format_estimate_ratio_text(estimates[OUTPUT_IMAGE_FORMAT_PNG]).endswith("%"))
        self.assertTrue(format_estimate_seconds_text(estimates[OUTPUT_IMAGE_FORMAT_JPG]["seconds"]))
        self.assertTrue(any(unit in format_estimate_size_text(estimates[OUTPUT_IMAGE_FORMAT_WEBP]["output_bytes"]) for unit in ("B", "KB", "MB", "GB")))

    def test_archive_file_names_match_selected_formats(self) -> None:
        self.assertEqual(
            build_archive_page_file_name(0, 27, "Page 01", OUTPUT_IMAGE_FORMAT_PNG),
            "001_Page 01.png",
        )
        self.assertEqual(
            build_archive_file_name("Series c01 v02", OUTPUT_ARCHIVE_FORMAT_CBZ),
            "Series_translated.cbz",
        )


if __name__ == "__main__":
    unittest.main()
