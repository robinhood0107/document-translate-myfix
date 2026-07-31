from __future__ import annotations

import unittest

from controller import _runtime_service_event_key


class ControllerRuntimeServiceRoutingTests(unittest.TestCase):
    def test_spotting_runtime_is_not_reported_as_crop_ocr(self) -> None:
        self.assertEqual(
            _runtime_service_event_key("PaddleOCR VL Spotting"),
            "paddleocr_vl_spotting",
        )

    def test_known_runtime_names_have_distinct_event_keys(self) -> None:
        self.assertEqual(
            _runtime_service_event_key("PaddleOCR VL"),
            "paddleocr_vl",
        )
        self.assertEqual(
            _runtime_service_event_key("HunyuanOCR"),
            "hunyuanocr",
        )
        self.assertEqual(
            _runtime_service_event_key("MangaLMM"),
            "mangalmm",
        )
        self.assertEqual(_runtime_service_event_key("Gemma"), "gemma")
