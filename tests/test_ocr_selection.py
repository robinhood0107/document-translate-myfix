from __future__ import annotations

import unittest

from modules.ocr.selection import (
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_BEST_LOCAL_PLUS,
    OCR_MODE_DEFAULT,
    OCR_OPTIMAL_LABEL,
    OCR_OPTIMAL_PLUS_LABEL,
    normalize_ocr_mode,
    resolve_ocr_engine,
)


class OCRSelectionTests(unittest.TestCase):
    def test_normalize_legacy_default_value(self) -> None:
        self.assertEqual(normalize_ocr_mode("Default"), OCR_MODE_DEFAULT)

    def test_normalize_optimal_label(self) -> None:
        self.assertEqual(normalize_ocr_mode(OCR_OPTIMAL_LABEL), OCR_MODE_BEST_LOCAL)

    def test_normalize_optimal_plus_label(self) -> None:
        self.assertEqual(normalize_ocr_mode(OCR_OPTIMAL_PLUS_LABEL), OCR_MODE_BEST_LOCAL_PLUS)

    def test_best_local_routes_chinese_to_hunyuan(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese"), "HunyuanOCR")

    def test_best_local_routes_parenthesized_chinese_variants_to_hunyuan(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese (Simplified)"), "HunyuanOCR")
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Chinese (Traditional)"), "HunyuanOCR")

    def test_best_local_routes_japanese_to_paddle(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "Japanese"), "PaddleOCR VL")

    def test_best_local_routes_other_languages_to_paddle(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL, "English"), "PaddleOCR VL")

    def test_best_local_plus_routes_japanese_to_mangalmm(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL_PLUS, "Japanese"), "MangaLMM")

    def test_best_local_plus_routes_chinese_to_hunyuan(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL_PLUS, "Chinese"), "HunyuanOCR")

    def test_best_local_plus_routes_other_languages_to_paddle(self) -> None:
        self.assertEqual(resolve_ocr_engine(OCR_MODE_BEST_LOCAL_PLUS, "English"), "PaddleOCR VL")


if __name__ == "__main__":
    unittest.main()
