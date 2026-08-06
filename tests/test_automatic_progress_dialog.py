from __future__ import annotations

import unittest

from app.ui.automatic_progress_dialog import _display_service_name


class AutomaticProgressDialogTests(unittest.TestCase):
    def test_spotting_startup_uses_distinct_service_label(self) -> None:
        self.assertEqual(
            _display_service_name("paddleocr_vl_spotting"),
            "PaddleOCR VL Spotting",
        )

    def test_crop_ocr_label_remains_unchanged(self) -> None:
        self.assertEqual(
            _display_service_name("paddleocr_vl"),
            "PaddleOCR VL",
        )

    def test_mangalmm_is_marked_experimental_and_slow(self) -> None:
        self.assertEqual(
            _display_service_name("mangalmm"),
            "MangaLMM (Experimental, Slow)",
        )
