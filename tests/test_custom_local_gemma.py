from __future__ import annotations

import json
import unittest
from unittest import mock

import numpy as np

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_PROMPT_PROFILE,
    CustomLocalGemmaTranslation,
    GemmaLocalServerResponseError,
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


def _raw_response(content: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": content},
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

    def test_contextual_merge_prompt_wraps_chunk_but_applies_block_response(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Ah..."),
            TextBlock(text_bbox=np.array([0, 100, 100, 200]), text="I'm dizzy."),
        ]
        requests = []

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            requests.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "expected_keys": expected_keys,
                }
            )
            return _response({"block_0": "아...", "block_1": "어지러워."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "아...")
        self.assertEqual(blocks[1].translation, "어지러워.")
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["expected_keys"], ["block_0", "block_1"])

        payload = json.loads(requests[0]["user_prompt"])
        self.assertEqual(list(payload.keys()), ["merged_context"])
        self.assertIn("[[block_0]] Ah...", payload["merged_context"])
        self.assertIn("[[block_1]] I'm dizzy.", payload["merged_context"])
        self.assertIn("continuous comic passage", requests[0]["system_prompt"])

        schema = engine._build_response_format(
            requests[0]["user_prompt"],
            expected_keys=requests[0]["expected_keys"],
        )
        self.assertEqual(schema["type"], "json_schema")
        self.assertEqual(
            schema["json_schema"]["schema"]["required"],
            ["block_0", "block_1"],
        )

    def test_channel_tokens_are_removed_before_translation_assignment(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]
        content = '<|channel>thought\n<channel|>{"block_0": "<|channel>thought\\n<channel|>안녕."}'

        with mock.patch.object(engine, "_request_translation", return_value=_raw_response(content)):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "안녕.")

    def test_contextual_merge_failure_raises_without_retry_or_fallback(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="However, a cock conquered me."),
        ]
        prompts = []

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            prompts.append(json.loads(user_prompt))
            return _response({"merged_context": "bad shape"})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            with self.assertRaises(GemmaLocalServerResponseError):
                engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(len(prompts), 1)
        self.assertEqual(list(prompts[0].keys()), ["merged_context"])
        self.assertEqual(blocks[0].translation, "")
        self.assertEqual(engine.last_benchmark_stats["gemma_retry_count"], 0)
        self.assertEqual(engine.last_benchmark_stats["gemma_fallback_count"], 0)


if __name__ == "__main__":
    unittest.main()
