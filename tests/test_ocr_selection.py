from __future__ import annotations

import unittest

from modules.ocr.selection import (
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_DEFAULT,
    OCR_OPTIMAL_LABEL,
    STAGE_BATCHED_WORKFLOW_MODE,
    normalize_ocr_mode,
    resolve_ocr_engine,
    resolve_stage_batched_ocr_policy,
)


class OCRSelectionTests(unittest.TestCase):
    def test_normalize_legacy_default_value(self) -> None:
        self.assertEqual(normalize_ocr_mode("Default"), OCR_MODE_DEFAULT)

    def test_normalize_optimal_label(self) -> None:
        self.assertEqual(normalize_ocr_mode(OCR_OPTIMAL_LABEL), OCR_MODE_BEST_LOCAL)

    def test_normalize_legacy_optimal_plus_value_to_optimal(self) -> None:
        self.assertEqual(normalize_ocr_mode("best_local_plus"), OCR_MODE_BEST_LOCAL)

    def test_normalize_legacy_optimal_plus_label_to_optimal(self) -> None:
        self.assertEqual(
            normalize_ocr_mode("Optimal+ (HunyuanOCR / MangaLMM / PaddleOCR VL)"),
            OCR_MODE_BEST_LOCAL,
        )

    def test_best_local_routes_chinese_to_hunyuan(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese"), "HunyuanOCR")

    def test_best_local_routes_parenthesized_chinese_variants_to_hunyuan(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese (Simplified)"), "HunyuanOCR")
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese (Traditional)"), "HunyuanOCR")

    def test_best_local_routes_japanese_to_paddle(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Japanese"), "PaddleOCR VL")

    def test_best_local_routes_other_languages_to_paddle(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "English"), "PaddleOCR VL")

    def test_stage_batched_accepts_legacy_gemma_alias(self) -> None:
        policy = resolve_stage_batched_ocr_policy(
            STAGE_BATCHED_WORKFLOW_MODE,
            "paddleocr_vl",
            "Japanese",
            "gemma_local",
        )

        self.assertTrue(policy.stage_batched_supported)
        self.assertEqual(policy.translator, "Custom Local Server(Gemma)")
        self.assertEqual(policy.primary_ocr_engine, "PaddleOCR VL")

if __name__ == "__main__":
    unittest.main()
