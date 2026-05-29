from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_PROMPT_PROFILE,
    CustomLocalGemmaTranslation,
)
from modules.utils.textblock import TextBlock


def _response(payload: dict[str, str]) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(payload, ensure_ascii=False)},
            }
        ],
        "usage": {},
    }


class CustomLocalGemmaRepetitionGuardTests(unittest.TestCase):
    def _engine(self) -> CustomLocalGemmaTranslation:
        engine = CustomLocalGemmaTranslation()
        engine.source_lang = "Japanese"
        engine.target_lang = "Korean"
        engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
        engine.chunk_size = 6
        engine.max_tokens = 512
        engine.timeout = 1
        return engine

    def test_severe_output_repetition_is_collapsed_after_json_parse(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(
                text_bbox=np.array([0, 0, 100, 100]),
                text="んばあ" + ("ぐ" * 80),
            )
        ]

        with mock.patch.object(engine, "_request_translation", return_value=_response({"block_0": "으" * 80})):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "으으으으...")
        self.assertEqual(engine.last_benchmark_stats["gemma_repetition_guard_count"], 1)
        self.assertTrue(getattr(blocks[0], "_translation_repetition_guard")["changed"])

    def test_short_sfx_output_is_preserved(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(
                text_bbox=np.array([0, 0, 100, 100]),
                text="プラブブブブ",
            )
        ]

        with mock.patch.object(engine, "_request_translation", return_value=_response({"block_0": "푸르르르르"})):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "푸르르르르")
        self.assertEqual(engine.last_benchmark_stats["gemma_repetition_guard_count"], 0)
        self.assertFalse(hasattr(blocks[0], "_translation_repetition_guard"))


if __name__ == "__main__":
    unittest.main()
