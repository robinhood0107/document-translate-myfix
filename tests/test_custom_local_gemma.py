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

        with mock.patch.object(engine, "_request_translation", return_value=_response({"translation": "으" * 80})):
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

        with mock.patch.object(engine, "_request_translation", return_value=_response({"translation": "푸르르르르"})):
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
            if expected_keys == ["translation"]:
                payload = json.loads(user_prompt)
                if payload["target_block"] == "block_0":
                    return _response({"translation": "아..."})
                return _response({"translation": "어지러워."})
            return _response({"block_0": "아...", "block_1": "어지러워."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "아...")
        self.assertEqual(blocks[1].translation, "어지러워.")
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["expected_keys"], ["translation"])
        self.assertEqual(requests[1]["expected_keys"], ["translation"])

        payload = json.loads(requests[0]["user_prompt"])
        self.assertEqual(list(payload.keys()), ["merged_context", "target_block", "target_text"])
        self.assertIn("[[block_0]] Ah...", payload["merged_context"])
        self.assertIn("[[block_1]] I'm dizzy.", payload["merged_context"])
        self.assertEqual(payload["target_block"], "block_0")
        self.assertEqual(payload["target_text"], "Ah...")
        self.assertIn("continuous comic passage", requests[0]["system_prompt"])

        schema = engine._build_response_format(
            requests[0]["user_prompt"],
            expected_keys=requests[0]["expected_keys"],
        )
        self.assertEqual(schema["type"], "json_schema")
        self.assertEqual(
            schema["json_schema"]["schema"]["required"],
            ["translation"],
        )

    def test_channel_tokens_are_removed_before_translation_assignment(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]
        content = '<|channel>thought\n<channel|>{"translation": "<|channel>thought\\n<channel|>안녕."}'

        with mock.patch.object(engine, "_request_translation", return_value=_raw_response(content)):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "안녕.")

    def test_contextual_merge_failure_falls_back_to_per_block_json(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="However, a cock conquered me."),
        ]
        prompts = []

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            prompts.append(json.loads(user_prompt))
            if len(prompts) < 3:
                return _response({"merged_context": "bad shape"})
            return _response({"block_0": "하지만 자지가 나를 정복했어."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(list(prompts[0].keys()), ["merged_context", "target_block", "target_text"])
        self.assertEqual(list(prompts[1].keys()), ["merged_context", "target_block", "target_text"])
        self.assertIn("block_0", prompts[2])
        self.assertNotIn("merged_context", prompts[2])
        self.assertEqual(blocks[0].translation, "하지만 자지가 나를 정복했어.")
        self.assertEqual(engine.last_benchmark_stats["gemma_contextual_merge_fallback_count"], 1)


if __name__ == "__main__":
    unittest.main()
