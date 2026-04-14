from __future__ import annotations

import json
import unittest

import numpy as np

from modules.ocr.selection import OCR_MODE_BEST_LOCAL_PLUS, OCR_MODE_PADDLE_VL
from modules.translation.llm.base import BaseLLMTranslation
from modules.utils.text_normalization import normalize_decorative_ocr_text
from modules.utils.textblock import TextBlock
from modules.utils.translator_utils import normalize_text_for_translation


def _make_block(text: str) -> TextBlock:
    blk = TextBlock(
        text_bbox=np.array([0, 0, 10, 10], dtype=np.int32),
        source_lang="ja",
        direction="vertical",
    )
    blk.text = text
    return blk


class _FakeSettings:
    def __init__(self, ocr_mode: str) -> None:
        self._ocr_mode = ocr_mode

    def get_llm_settings(self) -> dict:
        return {"image_input_enabled": False}

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type != "ocr":
            raise KeyError(tool_type)
        return self._ocr_mode


class _StubLLMTranslation(BaseLLMTranslation):
    def _perform_translation(self, user_prompt: str, system_prompt: str, image):
        return "{}"


class TranslationInputNormalizationTests(unittest.TestCase):
    def test_normalize_decorative_ocr_text_strips_known_noise_glyphs(self) -> None:
        self.assertEqual(normalize_decorative_ocr_text("⌒テ✺スト︸"), "テスト")

    def test_normalize_text_for_translation_strips_noise_only_for_mangalmm(self) -> None:
        self.assertEqual(
            normalize_text_for_translation("⌒テ✺スト︸", "ja", ocr_engine="MangaLMM"),
            "テスト",
        )
        self.assertEqual(
            normalize_text_for_translation("⌒テ✺スト︸", "ja", ocr_engine="PaddleOCR VL"),
            "⌒テ✺スト︸",
        )
        self.assertEqual(
            normalize_text_for_translation("⌒テ✺スト︸", "ja"),
            "⌒テ✺スト︸",
        )

    def test_normalize_text_for_translation_preserves_render_handled_glyphs(self) -> None:
        sample = '「テスト」♡❤♥'
        self.assertEqual(
            normalize_text_for_translation(sample, "ja", ocr_engine="MangaLMM"),
            sample,
        )
        self.assertEqual(
            normalize_text_for_translation(sample, "ja", ocr_engine="PaddleOCR VL"),
            sample,
        )

    def test_llm_translation_payload_resolves_mangalmm_under_optimal_plus(self) -> None:
        engine = _StubLLMTranslation()
        engine.initialize(_FakeSettings(OCR_MODE_BEST_LOCAL_PLUS), "Japanese", "Korean")
        blk = _make_block("⌒テ✺スト︸")

        raw_json, normalized_json = engine._build_translation_input_payloads([blk])

        self.assertEqual(json.loads(raw_json)["block_0"], "⌒テ✺スト︸")
        self.assertEqual(json.loads(normalized_json)["block_0"], "テスト")

    def test_llm_translation_payload_keeps_noise_for_non_mangalmm_ocr(self) -> None:
        engine = _StubLLMTranslation()
        engine.initialize(_FakeSettings(OCR_MODE_PADDLE_VL), "Japanese", "Korean")
        blk = _make_block("⌒テ✺スト︸")

        _raw_json, normalized_json = engine._build_translation_input_payloads([blk])

        self.assertEqual(json.loads(normalized_json)["block_0"], "⌒テ✺スト︸")


if __name__ == "__main__":
    unittest.main()
