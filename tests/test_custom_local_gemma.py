from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

from modules.translation.llm.custom_local_gemma import (
    DEFAULT_GEMMA_PROMPT_PROFILE,
    GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED,
    STRICT_GEMMA_PROMPT_PROFILE,
    CustomLocalGemmaTranslation,
    GemmaLocalServerContextCapacityError,
    GemmaLocalServerResponseError,
)
from modules.utils.exceptions import LocalServiceConnectionError
from modules.utils.textblock import TextBlock
from modules.utils.correction_dictionary import apply_translation_result_dictionary
from modules.translation.translation_memory import TranslationMemoryStore


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


class _FakeGemmaHTTPResponse:
    def __init__(self, payload: object, status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            response_error = requests.exceptions.HTTPError(f"{self.status_code} error")
            response_error.response = self
            raise response_error

    def json(self) -> dict:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _FakeSettings:
    class _UI:
        @staticmethod
        def tr(value: str) -> str:
            return value

    ui = _UI()

    def __init__(self, gemma_settings: dict | None = None) -> None:
        self._gemma_settings = gemma_settings or {}

    @staticmethod
    def get_llm_settings() -> dict:
        return {}

    @staticmethod
    def get_credentials(_name: str) -> dict:
        return {
            "api_url": "http://127.0.0.1:18080/v1",
            "model": "example-model.gguf",
        }

    def get_gemma_local_server_settings(self) -> dict:
        return dict(self._gemma_settings)


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

    @staticmethod
    def _blocks(*texts: str) -> list[TextBlock]:
        return [
            TextBlock(
                text_bbox=np.array([0, index * 100, 100, (index + 1) * 100]),
                text=text,
            )
            for index, text in enumerate(texts)
        ]

    def _assert_grouped_telemetry_zero(
        self,
        engine: CustomLocalGemmaTranslation,
    ) -> None:
        for key in (
            "gemma_max_requested_group_size",
            "gemma_contextual_grouped_request_count",
            "gemma_contextual_grouped_wall_ms",
            "gemma_direct_grouped_request_count",
            "gemma_direct_grouped_wall_ms",
            "gemma_strict_grouped_retry_count",
            "gemma_partial_response_count",
            "gemma_partial_fallback_block_count",
            "gemma_split_count",
            "gemma_context_capacity_split_count",
        ):
            self.assertEqual(engine.last_benchmark_stats[key], 0, key)

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

    def test_safe_retranslation_clears_stale_repetition_guard_metadata(self) -> None:
        engine = self._engine()
        blocks = self._blocks("もう一度")
        setattr(
            blocks[0],
            "_translation_repetition_guard",
            {"changed": True, "reason": "stale"},
        )

        with mock.patch.object(
            engine,
            "_request_translation",
            return_value=_response({"translation": "다시 한번"}),
        ):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "다시 한번")
        self.assertFalse(hasattr(blocks[0], "_translation_repetition_guard"))

    def test_contextual_merge_prompt_wraps_chunk_but_applies_block_response(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Ah..."),
            TextBlock(text_bbox=np.array([0, 100, 100, 200]), text="I'm dizzy."),
        ]
        requests = []

        def fake_request(
            system_prompt: str,
            user_prompt: str,
            *,
            expected_keys=None,
            request_mode="",
        ) -> dict:
            requests.append(
                {
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "expected_keys": expected_keys,
                    "request_mode": request_mode,
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
        self.assertEqual(
            requests[0]["request_mode"],
            GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        )

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

    def test_unsafe_control_chars_are_removed_before_translation_assignment(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]

        with mock.patch.object(
            engine,
            "_request_translation",
            return_value=_response({"translation": "안\u200b녕\u2066�\ufffc\ue000\t끝"}),
        ):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, "안녕 끝")

    def test_empty_translation_for_nonempty_source_is_rejected(self) -> None:
        engine = self._engine()
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="Hello."),
        ]

        with mock.patch.object(engine, "_request_translation", return_value=_response({"translation": ""})):
            with self.assertRaises(GemmaLocalServerResponseError):
                engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

    def test_contextual_merge_failure_falls_back_to_per_block_json(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        blocks = [
            TextBlock(text_bbox=np.array([0, 0, 100, 100]), text="However, a cock conquered me."),
        ]
        prompts = []

        def fake_request(
            system_prompt: str,
            user_prompt: str,
            *,
            expected_keys=None,
            request_mode="",
        ) -> dict:
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

    def test_retired_grouped_attribute_cannot_dispatch_grouped(self) -> None:
        engine = self._engine()
        engine.source_lang = "English"
        engine.request_mode = RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
        blocks = self._blocks("Ah...", "I'm dizzy.")
        requests_seen: list[tuple[dict, list[str], str]] = []

        def fake_request(
            _system_prompt: str,
            user_prompt: str,
            *,
            expected_keys=None,
            request_mode="",
        ) -> dict:
            payload = json.loads(user_prompt)
            requests_seen.append((payload, expected_keys, request_mode))
            translation = (
                "아..."
                if payload["target_block"] == "block_0"
                else "어지러워."
            )
            return _response({"translation": translation})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(
            [block.translation for block in blocks],
            ["아...", "어지러워."],
        )
        self.assertEqual(len(requests_seen), 2)
        for payload, expected_keys, request_mode in requests_seen:
            self.assertEqual(expected_keys, ["translation"])
            self.assertEqual(
                request_mode,
                GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            )
            self.assertIn("[[block_0]] Ah...", payload["merged_context"])
            self.assertIn(
                "[[block_1]] I'm dizzy.",
                payload["merged_context"],
            )
        self._assert_grouped_telemetry_zero(engine)

    def test_single_decoder_retries_invalid_value_shapes_strictly(self) -> None:
        cases = (
            (
                '{"translation":{"value":"하나"}}',
                "gemma_nested_value_count",
            ),
            ('{"translation":123}', "gemma_invalid_value_count"),
            ('{"translation":null}', "gemma_invalid_value_count"),
        )
        for invalid_content, expected_counter in cases:
            with self.subTest(invalid_content=invalid_content):
                engine = self._engine()
                responses = [
                    _raw_response(invalid_content),
                    _response({"translation": "하나"}),
                ]
                with mock.patch.object(
                    engine,
                    "_request_translation",
                    side_effect=responses,
                ) as request:
                    blocks = self._blocks("一")
                    engine.translate(
                        blocks,
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        "",
                    )

                self.assertEqual(blocks[0].translation, "하나")
                self.assertEqual(request.call_count, 2)
                self.assertEqual(
                    engine.last_benchmark_stats[expected_counter],
                    1,
                )
                self.assertEqual(
                    engine.last_benchmark_stats[
                        "gemma_strict_single_retry_count"
                    ],
                    1,
                )
                self._assert_grouped_telemetry_zero(engine)

    def test_single_decoder_retries_duplicate_trailing_array_and_extra_keys(
        self,
    ) -> None:
        cases = (
            (
                '{"translation":"하나","translation":"중복"}',
                "gemma_duplicate_key_count",
            ),
            (
                '{"translation":"하나"}{"extra":true}',
                "gemma_trailing_content_count",
            ),
            ('["하나"]', "gemma_top_level_type_error_count"),
            (
                '{"translation":"하나","explanation":"not allowed"}',
                "gemma_unexpected_key_count",
            ),
        )
        for invalid_content, expected_counter in cases:
            with self.subTest(invalid_content=invalid_content):
                engine = self._engine()
                responses = [
                    _raw_response(invalid_content),
                    _response({"translation": "하나"}),
                ]
                with mock.patch.object(
                    engine,
                    "_request_translation",
                    side_effect=responses,
                ) as request:
                    blocks = self._blocks("一")
                    engine.translate(
                        blocks,
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        "",
                    )

                self.assertEqual(blocks[0].translation, "하나")
                self.assertEqual(request.call_count, 2)
                self.assertEqual(
                    engine.last_benchmark_stats[expected_counter],
                    1,
                )
                self.assertEqual(
                    engine.last_benchmark_stats[
                        "gemma_strict_single_retry_count"
                    ],
                    1,
                )
                self._assert_grouped_telemetry_zero(engine)

    def test_literal_json_source_may_be_copied_by_contextual_single(self) -> None:
        engine = self._engine()
        blocks = self._blocks('{"code":1}')

        with mock.patch.object(
            engine,
            "_request_translation",
            return_value=_response({"translation": '{"code":1}'}),
        ) as request:
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertEqual(blocks[0].translation, '{"code":1}')
        self.assertEqual(request.call_count, 1)
        self.assertEqual(engine.last_benchmark_stats["gemma_nested_value_count"], 0)
        self._assert_grouped_telemetry_zero(engine)

    def test_literal_block_marker_in_source_is_preserved_as_text(self) -> None:
        engine = self._engine()
        blocks = self._blocks("literal [[block_7]] text")
        prompts = []

        def fake_request(
            _system_prompt: str,
            user_prompt: str,
            *,
            expected_keys=None,
            request_mode="",
        ) -> dict:
            prompts.append(json.loads(user_prompt))
            return _response({"translation": "리터럴 [[block_7]] 텍스트"})

        with mock.patch.object(engine, "_request_translation", side_effect=fake_request):
            engine.translate(blocks, np.zeros((1, 1, 3), dtype=np.uint8), "")

        self.assertTrue(prompts[0]["merged_context"].startswith("[[block_0]] "))
        self.assertIn("[[block_7]]", prompts[0]["merged_context"])
        self.assertEqual(blocks[0].translation, "리터럴 [[block_7]] 텍스트")
        self._assert_grouped_telemetry_zero(engine)

    def test_single_truncation_uses_strict_single_retry_without_grouped_split(
        self,
    ) -> None:
        engine = self._engine()
        truncated = _raw_response('{"translation":"하나"')
        truncated["choices"][0]["finish_reason"] = "length"
        responses = [
            truncated,
            _response({"translation": "하나"}),
        ]

        with mock.patch.object(
            engine,
            "_request_translation",
            side_effect=responses,
        ) as request:
            blocks = self._blocks("一")
            engine.translate(
                blocks,
                np.zeros((1, 1, 3), dtype=np.uint8),
                "",
            )

        self.assertEqual(blocks[0].translation, "하나")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(engine.last_benchmark_stats["gemma_truncated_count"], 1)
        self.assertEqual(
            engine.last_benchmark_stats["gemma_strict_single_retry_count"],
            1,
        )
        self._assert_grouped_telemetry_zero(engine)

    def test_single_invalid_response_envelope_uses_strict_single_retry(
        self,
    ) -> None:
        engine = self._engine()
        responses = [
            {"choices": {"bad": "shape"}},
            _response({"translation": "하나"}),
        ]

        with mock.patch.object(
            engine,
            "_request_translation",
            side_effect=responses,
        ) as request:
            blocks = self._blocks("一")
            engine.translate(
                blocks,
                np.zeros((1, 1, 3), dtype=np.uint8),
                "",
            )

        self.assertEqual(blocks[0].translation, "하나")
        self.assertEqual(request.call_count, 2)
        self.assertEqual(engine.last_benchmark_stats["gemma_parser_error_count"], 1)
        self.assertEqual(
            engine.last_benchmark_stats["gemma_strict_single_retry_count"],
            1,
        )
        self._assert_grouped_telemetry_zero(engine)

    def test_single_invalid_http_json_envelope_uses_strict_single_retry(
        self,
    ) -> None:
        engine = self._engine()
        invalid = _FakeGemmaHTTPResponse(ValueError("invalid JSON"))
        success = _FakeGemmaHTTPResponse(
            _response({"translation": "하나"})
        )

        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            side_effect=[invalid, success],
        ) as post:
            blocks = self._blocks("一")
            engine.translate(
                blocks,
                np.zeros((1, 1, 3), dtype=np.uint8),
                "",
            )

        self.assertEqual(blocks[0].translation, "하나")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(
            engine.last_benchmark_stats["gemma_logical_request_count"],
            2,
        )
        self.assertEqual(
            engine.last_benchmark_stats["gemma_http_attempt_count"],
            2,
        )
        self.assertEqual(
            engine.last_benchmark_stats["gemma_strict_single_retry_count"],
            1,
        )
        self._assert_grouped_telemetry_zero(engine)

    def test_single_final_failure_preserves_all_original_translations(
        self,
    ) -> None:
        engine = self._engine()
        blocks = self._blocks("一", "二")
        blocks[0].translation = "old-0"
        blocks[1].translation = "old-1"

        def fake_request(
            _system_prompt: str,
            user_prompt: str,
            *,
            expected_keys=None,
            request_mode="",
        ) -> dict:
            payload = json.loads(user_prompt)
            source = str(
                payload.get("target_text")
                or payload.get("block_0")
                or ""
            )
            if source == "一":
                return _response(
                    {"translation": "하나"}
                    if expected_keys == ["translation"]
                    else {"block_0": "하나"}
                )
            return _response(
                {"translation": ""}
                if expected_keys == ["translation"]
                else {"block_0": ""}
            )

        with mock.patch.object(
            engine,
            "_request_translation",
            side_effect=fake_request,
        ):
            with self.assertRaises(GemmaLocalServerResponseError):
                engine.translate(
                    blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )

        self.assertEqual(
            [block.translation for block in blocks],
            ["old-0", "old-1"],
        )
        self._assert_grouped_telemetry_zero(engine)

    def test_context_capacity_http_400_is_split_retryable_without_transport_retry(self) -> None:
        engine = self._engine()
        response = _FakeGemmaHTTPResponse(
            {"error": {"message": "prompt exceeds the context window limit"}},
            status_code=400,
        )

        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            return_value=response,
        ) as post:
            with self.assertRaises(GemmaLocalServerContextCapacityError):
                engine._request_translation(
                    "system",
                    "user",
                    expected_keys=["translation"],
                    request_mode=GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
                )

        self.assertEqual(post.call_count, 1)
        self.assertEqual(engine._current_benchmark_stats["gemma_http_attempt_count"], 1)
        self.assertEqual(engine._current_benchmark_stats["gemma_request_retry_count"], 0)

    def test_general_http_400_fails_immediately(self) -> None:
        engine = self._engine()
        response = _FakeGemmaHTTPResponse(
            {"error": {"message": "invalid model name"}},
            status_code=400,
        )

        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            return_value=response,
        ) as post:
            with self.assertRaises(LocalServiceConnectionError):
                engine._request_translation(
                    "system",
                    "user",
                    expected_keys=["translation"],
                )

        self.assertEqual(post.call_count, 1)

    def test_request_translation_retries_transient_read_timeout(self) -> None:
        engine = self._engine()
        engine.request_retry_backoff_seconds = (0.0, 0.0)
        timeout = requests.exceptions.ReadTimeout("read timed out")
        success = _FakeGemmaHTTPResponse(_response({"translation": "성공"}))

        with mock.patch("modules.translation.llm.custom_local_gemma.requests.post", side_effect=[timeout, success]) as post:
            response = engine._request_translation("system", "user", expected_keys=["translation"])

        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(engine._current_benchmark_stats["gemma_request_retry_count"], 1)
        self.assertEqual(engine._current_benchmark_stats["gemma_http_attempt_count"], 2)
        self.assertEqual(engine._current_benchmark_stats["gemma_http_retry_count"], 1)

    def test_request_translation_retries_transient_http_503(self) -> None:
        engine = self._engine()
        engine.request_retry_backoff_seconds = (0.0, 0.0)
        unavailable = _FakeGemmaHTTPResponse(
            {"error": {"message": "temporarily unavailable"}},
            status_code=503,
        )
        success = _FakeGemmaHTTPResponse(_response({"translation": "성공"}))

        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            side_effect=[unavailable, success],
        ) as post:
            response = engine._request_translation(
                "system",
                "user",
                expected_keys=["translation"],
            )

        self.assertEqual(response["choices"][0]["finish_reason"], "stop")
        self.assertEqual(post.call_count, 2)
        self.assertEqual(engine._current_benchmark_stats["gemma_request_retry_count"], 1)

    def test_request_telemetry_separates_logical_request_http_attempts_and_usage(self) -> None:
        engine = self._engine()
        response_payload = _response({"translation": "성공"})
        response_payload["usage"] = {
            "prompt_tokens": 40,
            "completion_tokens": 5,
            "total_tokens": 45,
            "prompt_tokens_details": {"cached_tokens": 12},
        }
        response_payload["timings"] = {"prompt_ms": 20.5, "predicted_ms": 30.25}

        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            return_value=_FakeGemmaHTTPResponse(response_payload),
        ):
            engine._request_translation(
                "system",
                "user",
                expected_keys=["translation"],
                request_mode=GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            )

        stats = engine._current_benchmark_stats
        self.assertEqual(stats["gemma_logical_request_count"], 1)
        self.assertEqual(stats["gemma_http_attempt_count"], 1)
        self.assertEqual(stats["gemma_prompt_tokens"], 40)
        self.assertEqual(stats["gemma_completion_tokens"], 5)
        self.assertEqual(stats["gemma_total_tokens"], 45)
        self.assertEqual(stats["gemma_cached_prompt_tokens"], 12)
        self.assertEqual(stats["gemma_prompt_eval_ms"], 20.5)
        self.assertEqual(stats["gemma_decode_ms"], 30.25)
        self.assertEqual(stats["gemma_contextual_single_request_count"], 1)

    def test_retired_request_mode_setting_and_environment_force_single(
        self,
    ) -> None:
        with self.assertLogs(
            "modules.translation.llm.custom_local_gemma",
            level="WARNING",
        ) as configured_logs:
            configured = CustomLocalGemmaTranslation()
            configured.initialize(
                _FakeSettings(
                    {
                        "request_mode": (
                            RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
                        )
                    }
                ),
                "Japanese",
                "Korean",
                "Custom Local Server(Gemma)",
            )
        self.assertEqual(
            configured.request_mode,
            GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        )
        self.assertIn("retired or unsupported", "\n".join(configured_logs.output))

        with mock.patch.dict(
            "os.environ",
            {
                "CT_GEMMA_REQUEST_MODE": (
                    RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
                )
            },
        ):
            with self.assertLogs(
                "modules.translation.llm.custom_local_gemma",
                level="WARNING",
            ) as environment_logs:
                overridden = CustomLocalGemmaTranslation()
                overridden.initialize(
                    _FakeSettings(
                        {
                            "request_mode": (
                                GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE
                            )
                        }
                    ),
                    "Japanese",
                    "Korean",
                    "Custom Local Server(Gemma)",
                )
        self.assertEqual(
            overridden.request_mode,
            GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
        )
        self.assertIn(
            "CT_GEMMA_REQUEST_MODE is retired and ignored",
            "\n".join(environment_logs.output),
        )

    @staticmethod
    def _runtime_identity(fingerprint: str = "runtime-a") -> dict:
        return {
            "model_sha256": "a" * 64,
            "runtime_fingerprint": fingerprint,
            "runtime_image_id": "sha256:" + ("b" * 64),
            "runtime_command_sha256": "c" * 64,
            "runtime_preparation_version": 1,
        }

    def test_persistent_all_hit_skips_runtime_and_http(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            first = self._engine()
            first.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": True,
                },
            )
            first_ensure = mock.Mock()
            first.configure_runtime_hooks(
                ensure_runtime=first_ensure,
                runtime_identity_provider=self._runtime_identity,
            )
            first_blocks = self._blocks("first", "second")
            with mock.patch.object(
                first,
                "_request_translation",
                side_effect=[
                    _response({"translation": "첫째"}),
                    _response({"translation": "둘째"}),
                ],
            ):
                first.translate(
                    first_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "same context",
                )
            first_ensure.assert_called_once_with()

            second = self._engine()
            second.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": True,
                },
            )
            second_ensure = mock.Mock()
            second.configure_runtime_hooks(
                ensure_runtime=second_ensure,
                runtime_identity_provider=self._runtime_identity,
            )
            second_blocks = self._blocks("first", "second")
            setattr(
                second_blocks[0],
                "_translation_repetition_guard",
                {"changed": True, "reason": "stale"},
            )
            with mock.patch.object(second, "_request_translation") as request:
                runtime_required = second.prepare_translation(
                    second_blocks,
                    "same context",
                )
                second.translate(
                    second_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "same context",
                )

            self.assertFalse(runtime_required)
            second_ensure.assert_not_called()
            request.assert_not_called()
            self.assertEqual(
                [block.translation for block in second_blocks],
                ["첫째", "둘째"],
            )
            self.assertFalse(
                hasattr(second_blocks[0], "_translation_repetition_guard")
            )
            self.assertEqual(
                second.last_benchmark_stats["gemma_tm_result_cache_hit_count"],
                2,
            )
            self.assertEqual(
                second.last_benchmark_stats["gemma_tm_runtime_skipped_count"],
                1,
            )
            self._assert_grouped_telemetry_zero(second)

    def test_mixed_exact_tm_hit_requests_only_missing_key_with_full_context(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            store.record_tm_candidate("first", "승인됨", "Japanese", "Korean")
            entry_id = store.list_tm_entries()[0]["id"]
            store.set_approved([entry_id], True)

            engine = self._engine()
            engine.chunk_size = 2
            engine.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": False,
                    "exact_tm_enabled": True,
                },
            )
            ensure_runtime = mock.Mock()
            engine.configure_runtime_hooks(
                ensure_runtime=ensure_runtime,
                runtime_identity_provider=self._runtime_identity,
            )
            blocks = self._blocks("first", "second")
            captured: list[dict] = []

            def fake_request(
                _system_prompt: str,
                user_prompt: str,
                *,
                expected_keys=None,
                request_mode="",
            ) -> dict:
                captured.append(
                    {
                        "payload": json.loads(user_prompt),
                        "expected_keys": expected_keys,
                        "request_mode": request_mode,
                    }
                )
                return _response({"translation": "새 번역"})

            with mock.patch.object(
                engine,
                "_request_translation",
                side_effect=fake_request,
            ):
                engine.translate(
                    blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "context",
                )

            ensure_runtime.assert_called_once_with()
            self.assertEqual(
                [block.translation for block in blocks],
                ["승인됨", "새 번역"],
            )
            self.assertEqual(len(captured), 1)
            self.assertEqual(captured[0]["expected_keys"], ["translation"])
            self.assertEqual(
                captured[0]["payload"]["target_block"],
                "block_1",
            )
            self.assertNotIn("requested_blocks", captured[0]["payload"])
            self.assertEqual(
                captured[0]["request_mode"],
                GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            )
            self.assertIn("[[block_0]] first", captured[0]["payload"]["merged_context"])
            self.assertIn("[[block_1]] second", captured[0]["payload"]["merged_context"])
            self.assertEqual(
                engine.last_benchmark_stats["gemma_tm_exact_hit_count"],
                1,
            )
            self._assert_grouped_telemetry_zero(engine)

    def test_partial_result_cache_population_requests_only_remaining_key(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            first = self._engine()
            first.chunk_size = 2
            first.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            first.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            first_blocks = self._blocks("first", "second")
            with mock.patch.object(
                first,
                "_request_translation",
                return_value=_response({"translation": "첫째"}),
            ):
                first.translate(
                    first_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "context",
                    requested_indices=[0],
                )

            second = self._engine()
            second.chunk_size = 2
            second.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            second.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            second_blocks = self._blocks("first", "second")
            captured: list[dict] = []

            def fake_request(
                _system_prompt: str,
                user_prompt: str,
                *,
                expected_keys=None,
                request_mode="",
            ) -> dict:
                captured.append(
                    {
                        "payload": json.loads(user_prompt),
                        "expected_keys": expected_keys,
                        "request_mode": request_mode,
                    }
                )
                return _response({"translation": "둘째"})

            with mock.patch.object(
                second,
                "_request_translation",
                side_effect=fake_request,
            ):
                second.translate(
                    second_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "context",
                )

            self.assertEqual(
                [block.translation for block in second_blocks],
                ["첫째", "둘째"],
            )
            self.assertEqual(
                second.last_benchmark_stats["gemma_tm_result_cache_hit_count"],
                1,
            )
            self.assertEqual(
                second.last_benchmark_stats["gemma_tm_result_cache_miss_count"],
                1,
            )
            self.assertEqual(captured[0]["expected_keys"], ["translation"])
            self.assertEqual(
                captured[0]["payload"]["target_block"],
                "block_1",
            )
            self.assertNotIn("requested_blocks", captured[0]["payload"])
            self.assertEqual(
                captured[0]["request_mode"],
                GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            )
            self.assertIn("[[block_0]] first", captured[0]["payload"]["merged_context"])
            self.assertIn("[[block_1]] second", captured[0]["payload"]["merged_context"])
            self._assert_grouped_telemetry_zero(second)
            self.assertEqual(store.stats()["result_cache_entries"], 2)

    def test_changed_runtime_identity_rejects_stale_result(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            first = self._engine()
            first.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            first.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=lambda: self._runtime_identity("runtime-a"),
            )
            with mock.patch.object(
                first,
                "_request_translation",
                return_value=_response({"translation": "old"}),
            ):
                first.translate(
                    self._blocks("source"),
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )

            second = self._engine()
            second.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            second.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=lambda: self._runtime_identity("runtime-b"),
            )
            blocks = self._blocks("source")
            with mock.patch.object(
                second,
                "_request_translation",
                return_value=_response({"translation": "new"}),
            ) as request:
                second.translate(
                    blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )

            request.assert_called_once()
            self.assertEqual(blocks[0].translation, "new")
            self.assertEqual(
                second.last_benchmark_stats["gemma_tm_stale_reject_count"],
                1,
            )

    def test_result_cache_rejects_prompt_model_sampler_context_and_tm_changes(
        self,
    ) -> None:
        variants = {
            "prompt": lambda engine, _store: setattr(
                engine,
                "prompt_profile",
                STRICT_GEMMA_PROMPT_PROFILE,
            ),
            "model": lambda engine, _store: setattr(
                engine,
                "model",
                "different-model.gguf",
            ),
            "sampler": lambda engine, _store: setattr(
                engine,
                "temperature",
                0.2,
            ),
            "context": lambda _engine, _store: None,
            "tm_revision": lambda _engine, store: (
                store.record_tm_candidate(
                    "unrelated",
                    "entry",
                    "Japanese",
                    "Korean",
                ),
                store.set_approved(
                    [store.list_tm_entries()[0]["id"]],
                    True,
                ),
            ),
        }

        for variant, mutate in variants.items():
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp_dir:
                with TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store:
                    first = self._engine()
                    first.configure_translation_memory(
                        store,
                        {
                            "persistent_cache_enabled": True,
                            "exact_tm_enabled": False,
                        },
                    )
                    first.configure_runtime_hooks(
                        ensure_runtime=mock.Mock(),
                        runtime_identity_provider=self._runtime_identity,
                    )
                    with mock.patch.object(
                        first,
                        "_request_translation",
                        return_value=_response({"translation": "old"}),
                    ):
                        first.translate(
                            self._blocks("source"),
                            np.zeros((1, 1, 3), dtype=np.uint8),
                            "base context",
                        )

                    second = self._engine()
                    second.configure_translation_memory(
                        store,
                        {
                            "persistent_cache_enabled": True,
                            "exact_tm_enabled": False,
                        },
                    )
                    second.configure_runtime_hooks(
                        ensure_runtime=mock.Mock(),
                        runtime_identity_provider=self._runtime_identity,
                    )
                    mutate(second, store)
                    second_context = (
                        "changed context"
                        if variant == "context"
                        else "base context"
                    )
                    blocks = self._blocks("source")
                    with mock.patch.object(
                        second,
                        "_request_translation",
                        return_value=_response({"translation": "new"}),
                    ) as request:
                        second.translate(
                            blocks,
                            np.zeros((1, 1, 3), dtype=np.uint8),
                            second_context,
                        )

                    request.assert_called_once()
                    self.assertEqual(blocks[0].translation, "new")

    def test_result_dictionary_is_applied_once_after_miss_and_hit(self) -> None:
        rules = [
            {
                "keyword": "cat",
                "sub": "cat!",
                "use_reg": False,
                "case_sens": True,
            }
        ]
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            first = self._engine()
            first.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            first.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            first_blocks = self._blocks("source")
            with mock.patch.object(
                first,
                "_request_translation",
                return_value=_response({"translation": "cat"}),
            ):
                first.translate(
                    first_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )
            apply_translation_result_dictionary(first_blocks, rules)
            self.assertEqual(first_blocks[0].translation, "cat!")

            second = self._engine()
            second.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            second.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            second_blocks = self._blocks("source")
            with mock.patch.object(second, "_request_translation") as request:
                second.translate(
                    second_blocks,
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )
            apply_translation_result_dictionary(second_blocks, rules)

            request.assert_not_called()
            self.assertEqual(second_blocks[0].translation, "cat!")

    def test_enabling_exact_tm_collects_candidates_from_result_cache_hits(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            first = self._engine()
            first.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            first.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            with mock.patch.object(
                first,
                "_request_translation",
                return_value=_response({"translation": "cached raw"}),
            ):
                first.translate(
                    self._blocks("source"),
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )
            self.assertEqual(store.stats()["candidate_tm_entries"], 0)

            second = self._engine()
            second.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": True,
                },
            )
            second.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            with mock.patch.object(second, "_request_translation") as request:
                second.translate(
                    self._blocks("source"),
                    np.zeros((1, 1, 3), dtype=np.uint8),
                    "",
                )

            request.assert_not_called()
            self.assertEqual(store.stats()["candidate_tm_entries"], 1)

    def test_result_cache_identity_contains_every_output_affecting_contract(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temp_dir,
            TranslationMemoryStore(Path(temp_dir) / "tm.sqlite3") as store,
        ):
            engine = self._engine()
            engine.settings = mock.Mock(
                get_translation_result_dictionary_rules=mock.Mock(
                    return_value=[
                        {
                            "keyword": "old",
                            "sub": "new",
                            "use_reg": False,
                            "case_sens": True,
                        }
                    ]
                )
            )
            engine.request_mode = (
                RETIRED_GEMMA_REQUEST_MODE_CONTEXTUAL_GROUPED
            )
            engine.chunk_size = 2
            engine.temperature = 0.7
            engine.top_k = 64
            engine.top_p = 0.95
            engine.min_p = 0.0
            engine.max_tokens = 512
            engine.configure_translation_memory(
                store,
                {
                    "persistent_cache_enabled": True,
                    "exact_tm_enabled": False,
                },
            )
            engine.configure_runtime_hooks(
                ensure_runtime=mock.Mock(),
                runtime_identity_provider=self._runtime_identity,
            )
            blocks = self._blocks("first", "second")

            self.assertTrue(engine.prepare_translation(blocks, "extra context"))
            plan = engine._pending_translation_plan
            self.assertIsNotNone(plan)
            identity = json.loads(plan.targets[1].identity_json)

            self.assertEqual(identity["ordered_raw_group_context"], ["first", "second"])
            self.assertEqual(identity["ordered_full_group_context"], ["first", "second"])
            self.assertEqual(identity["target_index"], 1)
            self.assertEqual(identity["target_key"], "block_1")
            self.assertEqual(identity["source_lang"], "Japanese")
            self.assertEqual(identity["target_lang"], "Korean")
            self.assertEqual(identity["extra_context"], "extra context")
            self.assertEqual(identity["configured_group_size"], 2)
            self.assertEqual(
                identity["request_mode"],
                GEMMA_REQUEST_MODE_CONTEXTUAL_SINGLE,
            )
            self.assertEqual(
                identity["sampler"],
                {
                    "temperature": 0.7,
                    "top_k": 64,
                    "top_p": 0.95,
                    "min_p": 0.0,
                    "max_completion_tokens": 512,
                },
            )
            self.assertEqual(identity["runtime"]["model_sha256"], "a" * 64)
            self.assertEqual(identity["runtime"]["runtime_fingerprint"], "runtime-a")
            self.assertEqual(identity["tm_revision"], 0)
            self.assertTrue(identity["dictionary_version"])
            self.assertGreaterEqual(identity["prompt_contract_version"], 1)
            self.assertGreaterEqual(identity["translation_input_normalizer_version"], 1)
            self.assertGreaterEqual(identity["output_sanitizer_version"], 1)
            self.assertGreaterEqual(identity["repetition_guard_version"], 1)

    def test_corrupt_cache_fails_open_and_translates_normally(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "tm.sqlite3"
            db_path.write_bytes(b"not sqlite")
            with TranslationMemoryStore(db_path) as store:
                engine = self._engine()
                engine.configure_translation_memory(
                    store,
                    {
                        "persistent_cache_enabled": True,
                        "exact_tm_enabled": True,
                    },
                )
                ensure_runtime = mock.Mock()
                engine.configure_runtime_hooks(
                    ensure_runtime=ensure_runtime,
                    runtime_identity_provider=self._runtime_identity,
                )
                blocks = self._blocks("source")

                with mock.patch.object(
                    engine,
                    "_request_translation",
                    return_value=_response({"translation": "translated"}),
                ) as request:
                    engine.translate(
                        blocks,
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        "",
                    )

                ensure_runtime.assert_called_once_with()
                request.assert_called_once()
                self.assertEqual(blocks[0].translation, "translated")
                self.assertEqual(
                    engine.last_benchmark_stats["gemma_tm_cache_disabled_count"],
                    1,
                )

    def test_disabled_cache_features_do_not_open_a_corrupt_database(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "tm.sqlite3"
            original_bytes = b"not sqlite"
            db_path.write_bytes(original_bytes)
            with TranslationMemoryStore(db_path) as store:
                engine = self._engine()
                engine.configure_translation_memory(
                    store,
                    {
                        "persistent_cache_enabled": False,
                        "exact_tm_enabled": False,
                    },
                )
                engine.configure_runtime_hooks(
                    ensure_runtime=mock.Mock(),
                    runtime_identity_provider=self._runtime_identity,
                )
                blocks = self._blocks("source")

                with mock.patch.object(
                    engine,
                    "_request_translation",
                    return_value=_response({"translation": "translated"}),
                ):
                    engine.translate(
                        blocks,
                        np.zeros((1, 1, 3), dtype=np.uint8),
                        "",
                    )

                self.assertTrue(store.enabled)
                self.assertEqual(db_path.read_bytes(), original_bytes)
                self.assertEqual(
                    engine.last_benchmark_stats["gemma_tm_cache_disabled_count"],
                    0,
                )


if __name__ == "__main__":
    unittest.main()
