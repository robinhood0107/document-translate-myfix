from __future__ import annotations

import importlib
import unittest

import numpy as np

from modules.ocr.common.geometry import ImageCoordinateTransform
from modules.ocr.mangalmm_full_page.image_policy import (
    MANGALMM_OFFICIAL_MAX_PIXELS,
    official_smart_resize,
)
from modules.ocr.paddle_crop.response_parser import (
    extract_text_from_response,
)
from modules.ocr.paddle_spotting.image_policy import (
    preprocess_spotting_image,
)


class OCRStrategyModuleBoundaryTests(unittest.TestCase):
    def test_legacy_engine_modules_alias_new_implementations(self) -> None:
        legacy_paddle = importlib.import_module("modules.ocr.ocr_paddle_VL")
        current_paddle = importlib.import_module(
            "modules.ocr.paddle_crop.engine"
        )
        legacy_manga = importlib.import_module("modules.ocr.mangalmm_ocr")
        current_manga = importlib.import_module(
            "modules.ocr.mangalmm_full_page.engine"
        )

        self.assertIs(legacy_paddle, current_paddle)
        self.assertIs(legacy_manga, current_manga)

    def test_legacy_contract_modules_alias_new_implementations(self) -> None:
        aliases = (
            (
                "modules.ocr.paddle_llamacpp_runtime_contract",
                "modules.ocr.paddle_crop.runtime",
            ),
            (
                "modules.ocr.mangalmm_llamacpp_runtime_contract",
                "modules.ocr.mangalmm_full_page.runtime",
            ),
            (
                "modules.ocr.mangalmm_response_contract",
                "modules.ocr.mangalmm_full_page.response_parser",
            ),
            (
                "modules.ocr.paddleocr_vl_spotting.response_contract",
                "modules.ocr.paddle_spotting.response_parser",
            ),
            (
                "modules.ocr.result_contract",
                "modules.ocr.common.result_contract",
            ),
        )

        for legacy_name, current_name in aliases:
            with self.subTest(legacy_name=legacy_name):
                self.assertIs(
                    importlib.import_module(legacy_name),
                    importlib.import_module(current_name),
                )

    def test_shared_coordinate_transform_is_reversible_and_clipped(self) -> None:
        transform = ImageCoordinateTransform(
            original_width=101,
            original_height=203,
            request_width=202,
            request_height=406,
        )

        self.assertEqual(
            transform.request_to_original_point((100.0, 200.0)),
            (50.0, 100.0),
        )
        self.assertEqual(
            transform.request_to_original_point((-4.0, 9999.0)),
            (0.0, 203.0),
        )

    def test_paddle_crop_parser_keeps_existing_layout_priority(self) -> None:
        payload = {
            "result": {
                "layoutParsingResults": [
                    {"markdown": {"text": "**첫째**"}},
                    {"text": "둘째"},
                ]
            }
        }

        self.assertEqual(extract_text_from_response(payload), "첫째")

    def test_paddle_crop_engine_keeps_parser_override_hooks(self) -> None:
        engine_module = importlib.import_module(
            "modules.ocr.paddle_crop.engine"
        )
        engine = engine_module.PaddleOCRVLEngine()
        engine._normalize_output_text = lambda text: f"normalized:{text}"

        self.assertEqual(
            engine._extract_text_from_layout_item(
                {"markdown": {"text": "**첫째**"}}
            ),
            "normalized:첫째",
        )

    def test_spotting_low_resolution_policy_preserves_aspect_ratio(self) -> None:
        image = np.zeros((101, 203, 3), dtype=np.uint8)

        resized, metadata = preprocess_spotting_image(image)

        self.assertEqual(resized.shape[:2], (202, 406))
        self.assertTrue(metadata["low_resolution_doubled"])
        self.assertTrue(metadata["aspect_ratio_preserved"])

    def test_mangalmm_official_resize_stays_inside_pixel_budget(self) -> None:
        height, width = official_smart_resize(
            3001,
            2003,
            max_pixels=MANGALMM_OFFICIAL_MAX_PIXELS,
        )

        self.assertLessEqual(height * width, MANGALMM_OFFICIAL_MAX_PIXELS)
        self.assertEqual(height % 28, 0)
        self.assertEqual(width % 28, 0)


if __name__ == "__main__":
    unittest.main()
