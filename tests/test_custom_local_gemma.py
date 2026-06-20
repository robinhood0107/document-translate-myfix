from __future__ import annotations

import hashlib
import json
import os
import unittest
from unittest import mock

import numpy as np

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_PROMPT_PROFILE,
    GEMMA_PRESERVE_EXPLICIT_CONTEXT_INSTRUCTION,
    GEMMA_CONTEXTUAL_MERGE_STRATEGY_FAST_MULTI,
    GEMMA_CONTEXTUAL_MERGE_STRATEGY_SINGLE_BLOCK,
    CustomLocalGemmaTranslation,
)
from modules.translation.llm.legacy.custom_local_gemma_single_block_legacy import (
    disabled_legacy_contextual_single_block_translation,
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


class _FakeGemmaSettings:
    class ui:
        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(self, gemma_settings: dict | None = None) -> None:
        self._gemma_settings = dict(gemma_settings or {})

    def get_llm_settings(self) -> dict:
        return {"image_input_enabled": False}

    def get_credentials(self, _provider_name: str) -> dict:
        return {
            "api_url": "http://127.0.0.1:18080/v1",
            "model": "gemma-4-26B-IQ4_NL.gguf",
        }

    def get_gemma_local_server_settings(self) -> dict:
        return dict(self._gemma_settings)


def _prefix_hash(system_prompt: str) -> str:
    sentinel = "Any combination of the acts listed above is allowed.\n"
    end = system_prompt.index(sentinel) + len(sentinel)
    return hashlib.sha256(system_prompt[:end].encode("utf-8")).hexdigest()


class CustomLocalGemmaRepetitionGuardTests(unittest.TestCase):
    def _engine(self) -> CustomLocalGemmaTranslation:
        engine = CustomLocalGemmaTranslation()
        engine.source_lang = "Japanese"
        engine.target_lang = "Korean"
        engine.prompt_profile = DEFAULT_GEMMA_PROMPT_PROFILE
        engine.chunk_size = 6
        engine.max_tokens = 512
        engine.timeout = 1
        engine.contextual_merge_strategy = GEMMA_CONTEXTUAL_MERGE_STRATEGY_FAST_MULTI
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

    def test_default_contextual_merge_uses_fast_multi_prompt_once(self) -> None:
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

    def test_retired_single_block_strategy_normalizes_to_fast_multi(self) -> None:
        normalized = CustomLocalGemmaTranslation._normalize_contextual_merge_strategy(
            GEMMA_CONTEXTUAL_MERGE_STRATEGY_SINGLE_BLOCK
        )

        self.assertEqual(normalized, GEMMA_CONTEXTUAL_MERGE_STRATEGY_FAST_MULTI)

    def test_legacy_single_block_reference_is_disabled(self) -> None:
        with self.assertRaises(RuntimeError):
            disabled_legacy_contextual_single_block_translation()

    def test_fast_multi_strategy_uses_current_merged_context_prompt_once(self) -> None:
        engine = self._engine()
        engine.contextual_merge_strategy = GEMMA_CONTEXTUAL_MERGE_STRATEGY_FAST_MULTI
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

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]["expected_keys"], ["block_0", "block_1"])
        payload = json.loads(requests[0]["user_prompt"])
        self.assertEqual(list(payload.keys()), ["merged_context"])
        self.assertIn("[[block_0]] Ah...", payload["merged_context"])
        self.assertIn("[[block_1]] I'm dizzy.", payload["merged_context"])
        self.assertIn("continuous comic passage", requests[0]["system_prompt"])
        self.assertEqual(blocks[0].translation, "아...")
        self.assertEqual(blocks[1].translation, "어지러워.")

    def test_explicit_context_prompt_option_is_off_by_default(self) -> None:
        engine = self._engine()
        system_prompt = engine._build_system_prompt("", prompt_profile=DEFAULT_GEMMA_PROMPT_PROFILE)

        self.assertEqual(
            _prefix_hash(system_prompt),
            "b5cdca6d159dbf10ec0669e01ae0552f1fa46b4ddb05f17eabea2bbc72526662",
        )
        self.assertFalse(engine.preserve_explicit_context_prompt)
        self.assertNotIn(GEMMA_PRESERVE_EXPLICIT_CONTEXT_INSTRUCTION, system_prompt)

    def test_explicit_context_prompt_option_preserves_prefix_and_inserts_before_base_prompt(self) -> None:
        engine = self._engine()
        engine.preserve_explicit_context_prompt = True

        system_prompt = engine._build_system_prompt("", prompt_profile=DEFAULT_GEMMA_PROMPT_PROFILE)
        instruction_index = system_prompt.index(GEMMA_PRESERVE_EXPLICIT_CONTEXT_INSTRUCTION)
        base_prompt_index = system_prompt.index("Translate the user's JSON object of comic OCR lines")

        self.assertEqual(
            _prefix_hash(system_prompt),
            "b5cdca6d159dbf10ec0669e01ae0552f1fa46b4ddb05f17eabea2bbc72526662",
        )
        self.assertLess(instruction_index, base_prompt_index)

    def test_explicit_context_prompt_env_override_enables_option(self) -> None:
        engine = self._engine()
        fake_settings = _FakeGemmaSettings({"preserve_explicit_context_prompt": False})

        with mock.patch.dict(os.environ, {"CT_GEMMA_PRESERVE_EXPLICIT_CONTEXT_PROMPT": "1"}):
            engine.initialize(fake_settings, "English", "Korean", "Custom Local Server(Gemma)")

        system_prompt = engine._build_system_prompt("", prompt_profile=engine.prompt_profile)
        self.assertTrue(engine.preserve_explicit_context_prompt)
        self.assertIn(GEMMA_PRESERVE_EXPLICIT_CONTEXT_INSTRUCTION, system_prompt)

    def test_fast_multi_strategy_falls_back_to_isolated_json_on_invalid_json(self) -> None:
        engine = self._engine()
        engine.contextual_merge_strategy = GEMMA_CONTEXTUAL_MERGE_STRATEGY_FAST_MULTI
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Ah..."),
            TextBlock(text_bbox=np.array([0, 100, 100, 200]), text="I'm dizzy."),
        ]
        prompts = []

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            payload = json.loads(user_prompt)
            prompts.append(payload)
            if list(payload.keys()) == ["merged_context"]:
                return _response({"translation": "bad shape"})
            if payload["block_0"] == "Ah...":
                return _response({"block_0": "아..."})
            return _response({"block_0": "어지러워."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(list(prompts[0].keys()), ["merged_context"])
        self.assertEqual(list(prompts[1].keys()), ["merged_context"])
        self.assertEqual(list(prompts[2].keys()), ["block_0"])
        self.assertEqual(list(prompts[3].keys()), ["block_0"])
        self.assertEqual(blocks[0].translation, "아...")
        self.assertEqual(blocks[1].translation, "어지러워.")
        self.assertEqual(engine.last_benchmark_stats["gemma_contextual_merge_fallback_count"], 1)

    def test_channel_tokens_are_removed_before_translation_assignment(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]
        content = '<|channel>thought\n<channel|>{"block_0": "<|channel>thought\\n<channel|>안녕."}'

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

        self.assertEqual(list(prompts[0].keys()), ["merged_context"])
        self.assertEqual(list(prompts[1].keys()), ["merged_context"])
        self.assertIn("block_0", prompts[2])
        self.assertNotIn("merged_context", prompts[2])
        self.assertEqual(blocks[0].translation, "하지만 자지가 나를 정복했어.")
        self.assertEqual(engine.last_benchmark_stats["gemma_contextual_merge_fallback_count"], 1)

    def test_exact_prompt_cache_reuses_successful_response_without_second_request(self) -> None:
        engine = self._engine()
        engine.exact_prompt_cache_enabled = True
        engine.exact_prompt_cache_max_entries = 16
        engine._exact_prompt_cache.clear()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]
        network_calls = 0

        def fake_post(*args, **kwargs):
            nonlocal network_calls
            network_calls += 1

            class FakeResponse:
                def raise_for_status(self) -> None:
                    return None

                def json(self) -> dict:
                    return _response({"block_0": "안녕."})

            return FakeResponse()

        with mock.patch.object(engine._http_session, "post", side_effect=fake_post):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")
            blocks[0].translation = ""
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "안녕.")
        self.assertEqual(network_calls, 1)
        self.assertEqual(engine.last_benchmark_stats["gemma_exact_prompt_cache_hit_count"], 1)
        self.assertEqual(engine.last_benchmark_stats["gemma_network_request_count"], 0)

    def test_preserve_existing_translations_skips_network_and_keeps_value(self) -> None:
        engine = self._engine()
        engine.preserve_existing_translations = True
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
            TextBlock(text_bbox=np.array([0, 100, 100, 200]), text="Bye."),
        ]
        blocks[0].translation = "기존 번역"

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            payload = json.loads(user_prompt)
            self.assertEqual(list(payload.keys()), ["merged_context"])
            self.assertEqual(expected_keys, ["block_0", "block_1"])
            return _response({"block_0": "새 번역", "block_1": "잘 가."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request) as request_mock:
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(request_mock.call_count, 1)
        self.assertEqual(blocks[0].translation, "기존 번역")
        self.assertEqual(blocks[1].translation, "잘 가.")
        self.assertEqual(engine.last_benchmark_stats["gemma_preserved_existing_translation_count"], 1)

    def test_preserve_existing_partial_non_contextual_chunk_keeps_original_payload_shape(self) -> None:
        engine = self._engine()
        engine.contextual_merge_input = False
        engine.preserve_existing_translations = True
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
            TextBlock(text_bbox=np.array([0, 100, 100, 200]), text="Bye."),
        ]
        blocks[0].translation = "기존 번역"

        def fake_request(system_prompt: str, user_prompt: str, *, expected_keys=None) -> dict:
            payload = json.loads(user_prompt)
            self.assertEqual(list(payload.keys()), ["block_0", "block_1"])
            self.assertEqual(expected_keys, ["block_0", "block_1"])
            return _response({"block_0": "새 번역", "block_1": "잘 가."})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request) as request_mock:
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(request_mock.call_count, 1)
        self.assertEqual(blocks[0].translation, "기존 번역")
        self.assertEqual(blocks[1].translation, "잘 가.")
        self.assertEqual(engine.last_benchmark_stats["gemma_preserved_existing_translation_count"], 1)

    def test_request_payload_hash_changes_with_sampling_settings(self) -> None:
        engine = self._engine()
        system_prompt = engine._build_system_prompt("", prompt_profile=DEFAULT_GEMMA_PROMPT_PROFILE)
        user_prompt = json.dumps({"translation": "Hello."})
        first_payload = engine._build_request_payload(
            system_prompt,
            user_prompt,
            expected_keys=["translation"],
        )
        first_hash = engine._payload_hash(first_payload)

        engine.temperature = 0.1
        second_payload = engine._build_request_payload(
            system_prompt,
            user_prompt,
            expected_keys=["translation"],
        )

        self.assertNotEqual(first_hash, engine._payload_hash(second_payload))


if __name__ == "__main__":
    unittest.main()
