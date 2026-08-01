from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import SettingsPage
from modules.ocr.factory import OCRFactory
from modules.ocr.ocr_paddle_VL import PaddleOCRVLEngine
from modules.ocr.persistent_cache import OCRPersistentResultCache
from modules.utils.exceptions import OperationCancelledError
from modules.utils.ocr_debug import (
    OCR_EMPTY_REASON_NON_TEXT_RESPONSE,
    OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE,
    drop_layout_schema_only_ocr_blocks,
    drop_rejected_empty_ocr_blocks,
)
from modules.utils import gpu_metrics as gpu_metrics_module
from modules.utils.textblock import TextBlock
from pipeline.cache_manager import CacheManager


def _make_block(x1: int, y1: int, x2: int, y2: int) -> TextBlock:
    return TextBlock(
        text_bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
        text_class="text_bubble",
        source_lang="ja",
        direction="vertical",
    )


def _make_text_free_block(x1: int, y1: int, x2: int, y2: int) -> TextBlock:
    return TextBlock(
        text_bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
        bubble_bbox=None,
        text_class="text_free",
        source_lang="en",
        direction="horizontal",
    )


class _FakeSettings:
    class ui:
        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(
        self,
        *,
        scheduler_mode: str | None = None,
        parallel_workers: int = 8,
        max_new_tokens: int = 1024,
        server_url: str = (
            "http://127.0.0.1:18000/v1/chat/completions"
        ),
    ) -> None:
        self._scheduler_mode = scheduler_mode
        self._parallel_workers = parallel_workers
        self._max_new_tokens = max_new_tokens
        self._server_url = server_url

    def get_paddleocr_vl_settings(self) -> dict:
        return {
            "server_url": self._server_url,
            "parallel_workers": self._parallel_workers,
            "max_new_tokens": self._max_new_tokens,
            "prettify_markdown": False,
            "visualize": False,
        }

    def get_ocr_generic_settings(self) -> dict:
        payload = {
            "manga_expansion_percentage": 7,
            "crop_padding_ratio": 0.05,
            "ppocr_retry_crop_ratio_x": 0.06,
            "ppocr_retry_crop_ratio_y": 0.10,
        }
        if self._scheduler_mode is not None:
            payload["paddleocr_vl_scheduler_mode"] = self._scheduler_mode
        return payload

    def get_credentials(self, _provider_name: str) -> dict:
        return {}

    def is_gpu_enabled(self) -> bool:
        return False


class _SettingsPageOverlayProbe:
    pass


class _FakeHTTPResponse:
    def __init__(self, status_code: int, payload: dict | None = None) -> None:
        self.status_code = status_code
        self._payload = payload or {}

    def json(self) -> dict:
        return self._payload


class PaddleOCRVLEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        OCRFactory._engines.clear()
        gpu_metrics_module._GPU_METRICS_CACHE_VALUE = None
        gpu_metrics_module._GPU_METRICS_CACHE_EXPIRES_AT = 0.0

    def test_bubble_ocr_uses_expanded_text_bbox_clamped_to_bubble(
        self,
    ) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        image = np.zeros((320, 320, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=np.asarray([80, 100, 140, 180], dtype=np.int32),
            bubble_bbox=np.asarray([80, 80, 180, 220], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )

        with mock.patch.object(
            engine,
            "_request_ocr_text",
            return_value="テスト",
        ):
            engine.process_image(image, [block])

        request_record = engine.last_page_profile["request_records"][0]
        self.assertEqual(request_record["bbox"], [80, 97, 142, 183])
        self.assertEqual(request_record["crop_source"], "xyxy")
        self.assertEqual(
            block.ocr_effective_crop_xyxy,
            [80, 97, 142, 183],
        )
        self.assertEqual(block.ocr_strategy, "paddle_crop")
        self.assertEqual(
            block.ocr_geometry_provenance["strategy"],
            "text_first_bubble_clamp",
        )
        self.assertNotEqual(
            request_record["bbox"],
            [80, 80, 180, 220],
        )

    def test_missing_text_bbox_uses_one_bubble_fallback_crop(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        image = np.zeros((260, 260, 3), dtype=np.uint8)
        block = TextBlock(
            text_bbox=None,
            bubble_bbox=np.asarray([60, 70, 180, 210], dtype=np.int32),
            text_class="text_bubble",
            source_lang="ja",
            direction="vertical",
        )

        with mock.patch.object(
            engine,
            "_request_ocr_text",
            return_value="テスト",
        ) as request:
            engine.process_image(image, [block])

        request.assert_called_once()
        request_record = engine.last_page_profile["request_records"][0]
        self.assertEqual(request_record["bbox"], [60, 70, 180, 210])
        self.assertEqual(request_record["crop_source"], "bubble_fallback")
        self.assertEqual(
            block.ocr_geometry_provenance["crop_source"],
            "bubble_fallback",
        )

    def test_settings_page_generic_settings_overlays_benchmark_values(self) -> None:
        probe = _SettingsPageOverlayProbe()
        probe._benchmark_ocr_generic_settings = {
            "paddleocr_vl_scheduler_mode": "auto_v1",
            "crop_padding_ratio": 0.11,
        }

        merged = SettingsPage.get_ocr_generic_settings(probe)

        self.assertEqual(merged["paddleocr_vl_scheduler_mode"], "auto_v1")
        self.assertEqual(merged["crop_padding_ratio"], 0.11)
        self.assertEqual(merged["manga_expansion_percentage"], 7)

    def test_persistent_cache_exact_hit_skips_http_request(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        img = np.full((240, 320, 3), 220, dtype=np.uint8)
        runtime_identity = {
            "managed": True,
            "model_name": "PaddleOCR-VL-1.6-0.9B",
            "image_digest": "sha256:test",
            "runtime_fingerprint": "runtime-test",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with OCRPersistentResultCache(
                Path(temp_dir) / "ocr.sqlite3"
            ) as store:
                first_block = _make_block(40, 50, 220, 95)
                first_plan = engine.prepare_persistent_cache(
                    img,
                    [first_block],
                    store,
                    runtime_identity,
                )
                self.assertTrue(first_plan.requires_runtime)
                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                    return_value="テスト",
                ) as request:
                    engine.process_persistent_cache_plan(first_plan)
                request.assert_called_once()
                self.assertTrue(
                    store.store_records(
                        engine.build_persistent_cache_records(first_plan)
                    )
                )

                second_block = _make_block(40, 50, 220, 95)
                second_plan = engine.prepare_persistent_cache(
                    img.copy(),
                    [second_block],
                    store,
                    runtime_identity,
                )
                self.assertTrue(second_plan.all_hit)
                self.assertFalse(second_plan.requires_runtime)
                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                ) as request:
                    engine.process_persistent_cache_plan(second_plan)
                request.assert_not_called()
                self.assertEqual(second_block.text, "テスト")
                self.assertEqual(second_block.ocr_raw_text, "テスト")
                self.assertEqual(second_block.ocr_strategy, "paddle_crop")
                self.assertEqual(
                    second_block.ocr_geometry_provenance["strategy"],
                    "text_first_bubble_clamp",
                )
                self.assertTrue(second_block.ocr_runtime_identity)
                self.assertEqual(
                    engine.last_page_profile["performance"]["http_attempt_count"],
                    0,
                )
                self.assertEqual(
                    engine.last_page_profile["performance"]["request_bytes"],
                    0,
                )

    def test_persistent_cache_key_changes_with_exact_crop_pixels(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        identity = {
            "managed": True,
            "model_name": "PaddleOCR-VL-1.6-0.9B",
            "image_digest": "sha256:test",
            "runtime_fingerprint": "runtime-test",
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with OCRPersistentResultCache(
                Path(temp_dir) / "ocr.sqlite3"
            ) as store:
                first_img = np.zeros((200, 200, 3), dtype=np.uint8)
                second_img = first_img.copy()
                second_img[50, 50, 0] = 1
                first_plan = engine.prepare_persistent_cache(
                    first_img,
                    [_make_block(40, 40, 100, 100)],
                    store,
                    identity,
                    lookup=False,
                )
                second_plan = engine.prepare_persistent_cache(
                    second_img,
                    [_make_block(40, 40, 100, 100)],
                    store,
                    identity,
                    lookup=False,
                )

                self.assertNotEqual(
                    first_plan.jobs[0]["cache_key"],
                    second_plan.jobs[0]["cache_key"],
                )
                cache_identity = first_plan.jobs[0]["cache_identity"]
                self.assertEqual(
                    len(cache_identity["raw_crop_pixel_sha256"]),
                    64,
                )
                self.assertEqual(
                    len(cache_identity["request_jpeg_sha256"]),
                    64,
                )

    def test_prepared_worker_returns_outcome_without_mutating_block(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        img = np.full((240, 320, 3), 220, dtype=np.uint8)
        block = _make_block(40, 50, 220, 95)
        with tempfile.TemporaryDirectory() as temp_dir:
            with OCRPersistentResultCache(
                Path(temp_dir) / "ocr.sqlite3"
            ) as store:
                plan = engine.prepare_persistent_cache(
                    img,
                    [block],
                    store,
                    {"runtime_fingerprint": "runtime-test"},
                    lookup=False,
                )

                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                    return_value="テスト",
                ):
                    outcome = engine._process_prepared_job(plan.runtime_jobs[0])

        self.assertEqual(block.text, "")
        self.assertEqual(outcome["text"], "テスト")
        self.assertEqual(outcome["raw_text"], "テスト")

    def test_prepared_page_results_commit_only_after_all_workers_succeed(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        img = np.full((240, 320, 3), 220, dtype=np.uint8)
        blocks = [
            _make_block(20, 30, 120, 80),
            _make_block(160, 120, 280, 180),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            with OCRPersistentResultCache(
                Path(temp_dir) / "ocr.sqlite3"
            ) as store:
                plan = engine.prepare_persistent_cache(
                    img,
                    blocks,
                    store,
                    {"runtime_fingerprint": "runtime-test"},
                    lookup=False,
                )

                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                    side_effect=["first", RuntimeError("second failed")],
                ), self.assertRaises(RuntimeError):
                    engine.process_persistent_cache_plan(plan)

        self.assertEqual([block.text for block in blocks], ["", ""])
        self.assertEqual(
            engine.build_persistent_cache_records(plan),
            [],
        )

    def test_partial_cache_hit_is_not_committed_when_page_miss_fails(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(scheduler_mode="fixed", parallel_workers=1)
        )
        img = np.full((240, 320, 3), 220, dtype=np.uint8)
        first_block = _make_block(20, 30, 120, 80)
        second_block = _make_block(160, 120, 280, 180)
        runtime_identity = {"runtime_fingerprint": "runtime-test"}
        with tempfile.TemporaryDirectory() as temp_dir:
            with OCRPersistentResultCache(
                Path(temp_dir) / "ocr.sqlite3"
            ) as store:
                cached_plan = engine.prepare_persistent_cache(
                    img,
                    [first_block],
                    store,
                    runtime_identity,
                    lookup=False,
                )
                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                    return_value="cached",
                ):
                    engine.process_persistent_cache_plan(cached_plan)
                self.assertTrue(
                    store.store_records(
                        engine.build_persistent_cache_records(cached_plan)
                    )
                )
                hit_block = _make_block(20, 30, 120, 80)
                partial_plan = engine.prepare_persistent_cache(
                    img,
                    [hit_block, second_block],
                    store,
                    runtime_identity,
                )

                self.assertEqual(hit_block.text, "")
                self.assertEqual(partial_plan.hit_count, 1)
                self.assertEqual(partial_plan.miss_count, 1)
                with mock.patch.object(
                    engine,
                    "_request_ocr_text_from_encoded",
                    side_effect=RuntimeError("miss failed"),
                ), self.assertRaises(RuntimeError):
                    engine.process_persistent_cache_plan(partial_plan)

        self.assertEqual(hit_block.text, "")
        self.assertEqual(second_block.text, "")

    def test_default_scheduler_mode_is_fixed_area_desc_without_override(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings())

        self.assertEqual(engine.scheduler_mode, "fixed_area_desc")

    def test_fixed_mode_preserves_original_order_and_uses_fixed_worker_cap(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=8))
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _make_block(10, 10, 60, 60),
            _make_block(10, 10, 210, 210),
            _make_block(10, 10, 140, 140),
        ]

        with mock.patch.object(engine, "_request_ocr_text", return_value="テスト"):
            engine.process_image(img, blocks)

        profile = engine.last_page_profile
        self.assertEqual(profile["scheduler_mode"], "fixed")
        self.assertEqual(profile["chosen_workers"], 3)
        self.assertEqual(profile["job_order"], "original")
        self.assertEqual([item["job_index"] for item in profile["request_records"]], [0, 1, 2])

    def test_fixed_area_desc_sorts_jobs_by_crop_area_desc(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed_area_desc", parallel_workers=1))
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _make_block(10, 10, 60, 60),
            _make_block(10, 10, 210, 210),
            _make_block(10, 10, 140, 140),
        ]

        with mock.patch.object(engine, "_request_ocr_text", return_value="テスト"):
            engine.process_image(img, blocks)

        self.assertEqual(
            [item["job_index"] for item in engine.last_page_profile["request_records"]],
            [1, 2, 0],
        )
        self.assertEqual(engine.last_page_profile["chosen_workers"], 1)

    def test_auto_v1_local_mode_uses_gpu_headroom_and_penalties(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="auto_v1", parallel_workers=8))
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _make_block(10, 10, 210, 210),
            _make_block(250, 10, 430, 190),
            _make_block(500, 10, 620, 130),
        ]

        gpu_payload = {
            "available": True,
            "gpu_count": 1,
            "sampled_at": 1.0,
            "primary": {
                "index": 0,
                "name": "GPU",
                "memory_total_mb": 12288,
                "memory_used_mb": 5120,
                "memory_free_mb": 7000,
                "gpu_util_percent": 90,
                "memory_util_percent": 65,
            },
        }
        with mock.patch("modules.ocr.ocr_paddle_VL.query_gpu_metrics_cached", return_value=gpu_payload), \
             mock.patch.object(engine, "_request_ocr_text", return_value="テスト"):
            engine.process_image(img, blocks)

        profile = engine.last_page_profile
        self.assertEqual(profile["scheduler_mode"], "auto_v1")
        self.assertTrue(profile["local_server"])
        self.assertEqual(profile["chosen_workers"], 1)
        self.assertGreaterEqual(profile["p90_area_ratio"], 0.03)
        self.assertGreaterEqual(profile["large_crop_ratio"], 0.35)

    def test_auto_v1_remote_fallback_skips_gpu_probe_and_caps_by_crop_stats(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(
            _FakeSettings(
                scheduler_mode="auto_v1",
                parallel_workers=8,
                server_url="http://192.168.0.10:28118/layout-parsing",
            )
        )
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _make_block(10, 10, 210, 210),
            _make_block(250, 10, 430, 190),
            _make_block(500, 10, 620, 130),
        ]

        with mock.patch("modules.ocr.ocr_paddle_VL.query_gpu_metrics_cached") as gpu_query, \
             mock.patch.object(engine, "_request_ocr_text", return_value="テスト"):
            engine.process_image(img, blocks)

        gpu_query.assert_not_called()
        profile = engine.last_page_profile
        self.assertFalse(profile["local_server"])
        self.assertEqual(profile["chosen_workers"], 2)

    def test_request_records_capture_minimum_fields(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        blocks = [_make_block(10, 10, 110, 110)]

        with mock.patch.object(engine, "_request_ocr_text", return_value="テスト"):
            engine.process_image(img, blocks)

        record = engine.last_page_profile["request_records"][0]
        self.assertEqual(record["job_index"], 0)
        self.assertEqual(record["bbox"], [7, 7, 113, 113])
        self.assertGreater(record["crop_area_px"], 0)
        self.assertIsNotNone(record["enqueue_ts"])
        self.assertIsNotNone(record["start_ts"])
        self.assertIsNotNone(record["end_ts"])
        self.assertIsNotNone(record["elapsed_ms"])
        self.assertGreaterEqual(record["queue_wait_ms"], 0.0)
        self.assertGreaterEqual(record["crop_ms"], 0.0)
        self.assertGreaterEqual(record["text_guard_ms"], 0.0)
        self.assertEqual(record["status"], "ok")
        performance = engine.last_page_profile["performance"]
        self.assertEqual(performance["schema_version"], 1)
        self.assertEqual(performance["job_count"], 1)
        self.assertGreaterEqual(performance["queue_wait_ms"], 0.0)

    def test_schema_only_layout_labels_are_marked_empty(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        blocks = [_make_block(10, 10, 110, 110)]
        schema_text = "\n".join(
            [
                "number",
                "footnote",
                "header",
                "header_image",
                "footer",
                "footer_image",
                "aside_text",
                "ocr",
            ]
        )

        with mock.patch.object(engine, "_request_ocr_text", return_value=schema_text):
            engine.process_image(img, blocks)

        self.assertEqual(blocks[0].text, "")
        self.assertEqual(blocks[0].ocr_status, "empty_initial")
        self.assertEqual(blocks[0].ocr_raw_text, schema_text)
        self.assertEqual(blocks[0].ocr_sanitized_text, "")
        self.assertIn("layout schema labels", blocks[0].ocr_empty_reason)
        self.assertEqual(engine.last_page_profile["request_records"][0]["status"], "schema_only")

    def test_schema_only_layout_blocks_are_dropped_before_masking(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        schema_block = _make_block(10, 10, 110, 110)
        valid_block = _make_block(130, 10, 230, 110)
        schema_text = "\n".join(
            [
                "number",
                "footnote",
                "header",
                "header_image",
                "footer",
                "footer_image",
                "aside_text",
                "ocr",
            ]
        )

        with mock.patch.object(engine, "_request_ocr_text", side_effect=[schema_text, "PATREON.COM/YTSNOW"]):
            engine.process_image(img, [schema_block, valid_block])

        kept, dropped = drop_layout_schema_only_ocr_blocks([schema_block, valid_block])

        self.assertEqual(kept, [valid_block])
        self.assertEqual(dropped, [schema_block])

    def test_text_free_without_visual_text_evidence_is_rejected_before_request(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((240, 320, 3), 128, dtype=np.uint8)
        block = _make_text_free_block(40, 50, 220, 95)

        with mock.patch.object(engine, "_request_ocr_text", return_value="aunt in a") as request:
            engine.process_image(img, [block])

        request.assert_not_called()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["status"], "rejected_no_text_evidence")

    def test_text_free_title_like_crop_still_goes_to_paddleocr_vl(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((240, 420, 3), 232, dtype=np.uint8)
        cv2.putText(img, "LATER", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
        block = _make_text_free_block(20, 50, 280, 130)

        with mock.patch.object(engine, "_request_ocr_text", return_value="LATER") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "LATER")
        self.assertEqual(block.ocr_status, "ok")

    def test_text_bubble_is_not_rejected_by_text_free_visual_gate(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((240, 320, 3), 128, dtype=np.uint8)
        block = _make_block(40, 50, 220, 95)

        with mock.patch.object(engine, "_request_ocr_text", return_value="テスト") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "テスト")
        self.assertEqual(block.ocr_status, "ok")

    def test_non_text_model_response_is_marked_empty_and_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((240, 420, 3), 232, dtype=np.uint8)
        cv2.putText(img, "NEXT WEEK", (25, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (20, 20, 20), 3, cv2.LINE_AA)
        block = _make_text_free_block(20, 50, 360, 130)
        raw_response = "The image is too blurry to determine readable text."

        with mock.patch.object(engine, "_request_ocr_text", return_value=raw_response):
            engine.process_image(img, [block])

        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_raw_text, raw_response)
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        kept, dropped = drop_rejected_empty_ocr_blocks([block])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [block])

    def test_text_free_symbol_only_response_is_marked_empty_and_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 260, 3), 235, dtype=np.uint8)
        cv2.rectangle(img, (40, 60), (120, 95), (30, 30, 30), 2)
        block = _make_text_free_block(30, 50, 150, 110)

        with mock.patch.object(engine, "_request_ocr_text", return_value="☐☐☐☐") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["status"], "rejected_non_text_response")
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "symbol_only")
        kept, dropped = drop_rejected_empty_ocr_blocks([block])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [block])

    def test_text_free_warm_texture_numeric_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((180, 360, 3), dtype=np.uint8)
        rng = np.random.default_rng(3)
        for x in range(20, 340, 18):
            height = int(rng.integers(25, 75))
            color = (int(rng.integers(190, 255)), int(rng.integers(70, 165)), int(rng.integers(0, 35)))
            cv2.ellipse(img, (x, 110), (10, height), 0, 180, 360, color, -1, cv2.LINE_AA)
        block = _make_text_free_block(10, 35, 350, 150)

        with mock.patch.object(engine, "_request_ocr_text", return_value="2024") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "numeric_warm_texture")

    def test_text_free_warm_texture_date_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((180, 360, 3), dtype=np.uint8)
        rng = np.random.default_rng(7)
        for x in range(20, 340, 16):
            height = int(rng.integers(25, 80))
            color = (int(rng.integers(190, 255)), int(rng.integers(75, 170)), int(rng.integers(0, 40)))
            cv2.ellipse(img, (x, 110), (10, height), 0, 180, 360, color, -1, cv2.LINE_AA)
        block = _make_text_free_block(10, 35, 350, 150)

        with mock.patch.object(engine, "_request_ocr_text", return_value="2024年1月1日") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "numeric_warm_texture")

    def test_text_free_saturated_texture_mask_is_rejected_before_ocr_request(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 460, 3), 80, dtype=np.uint8)
        block = _make_text_free_block(10, 35, 450, 150)
        evidence = {
            "width": 460,
            "height": 127,
            "contrast_std": 36.249,
            "contrast_spread": 114.0,
            "edge_density": 0.05612,
            "transition_density": 0.00415,
            "component_count": 28,
            "component_area_ratio": 0.024213,
            "max_component_area_ratio": 0.01442,
            "saturated_ratio": 0.831856,
            "warm_saturated_ratio": 0.0,
            "low_saturation_ratio": 0.001162,
        }

        with (
            mock.patch.object(engine, "_analyze_text_free_crop", return_value=evidence),
            mock.patch.object(engine, "_request_ocr_text", return_value="127") as request,
        ):
            engine.process_image(img, [block])

        request.assert_not_called()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE)
        self.assertEqual(block.ocr_reject_reason, "saturated_texture_without_text_mask")
        self.assertEqual(engine.last_page_profile["request_records"][0]["status"], "rejected_no_text_evidence")
        self.assertEqual(
            engine.last_page_profile["request_records"][0]["non_text_reason"],
            "saturated_texture_without_text_mask",
        )

    def test_text_free_short_saturated_numeric_texture_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 460, 3), (25, 30, 120), dtype=np.uint8)
        rng = np.random.default_rng(11)
        for _ in range(55):
            x = int(rng.integers(18, 440))
            y = int(rng.integers(45, 145))
            cv2.ellipse(
                img,
                (x, y),
                (int(rng.integers(4, 9)), int(rng.integers(10, 25))),
                0,
                180,
                360,
                (30, 190, 255),
                2,
                cv2.LINE_AA,
            )
        block = _make_text_free_block(10, 35, 450, 150)

        with mock.patch.object(engine, "_request_ocr_text", return_value="127") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "short_saturated_texture")

    def test_text_free_short_saturated_icon_letter_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((120, 220, 3), (28, 34, 42), dtype=np.uint8)
        for x in range(45, 160, 22):
            cv2.circle(img, (x, 60), 9, (210, 220, 250), -1, cv2.LINE_AA)
            cv2.circle(img, (x, 60), 12, (80, 20, 220), 2, cv2.LINE_AA)
        cv2.line(img, (20, 30), (195, 95), (230, 40, 210), 3, cv2.LINE_AA)
        block = _make_text_free_block(20, 25, 195, 95)

        with mock.patch.object(engine, "_request_ocr_text", return_value="C") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "short_saturated_texture")

    def test_text_free_source_script_mismatch_response_is_dropped_without_strong_typography(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 360, 3), 210, dtype=np.uint8)
        for x in range(20, 340, 12):
            cv2.line(img, (x, 45), (x, 135), (90, 90, 90), 1)
        block = _make_text_free_block(10, 35, 350, 150)

        with mock.patch.object(engine, "_request_ocr_text", return_value="ཡོད་པ་ཡིན།་༡༩༠༠ལོ") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "source_script_mismatch")

    def test_short_text_free_foreign_sfx_is_not_dropped_by_script_mismatch_only(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 360, 3), 240, dtype=np.uint8)
        cv2.putText(img, "SFX", (70, 105), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (20, 20, 20), 4, cv2.LINE_AA)
        block = _make_text_free_block(40, 50, 260, 130)

        with mock.patch.object(engine, "_request_ocr_text", return_value="摔倒！") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "摔倒！")
        self.assertEqual(block.ocr_status, "ok")

    def test_text_free_book_spine_index_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((1000, 500, 3), 205, dtype=np.uint8)
        for x in range(0, 410, 30):
            cv2.line(img, (x, 20), (x, 940), (80, 80, 80), 2)
        block = _make_text_free_block(3, 16, 438, 942)
        ocr_text = "欽定四庫全書\n卷一\n卷二\n卷三\n卷四\n卷五"

        with mock.patch.object(engine, "_request_ocr_text", return_value=ocr_text) as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "book_spine_or_index")

    def test_text_free_generic_catalog_index_response_is_dropped(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((1000, 500, 3), 205, dtype=np.uint8)
        for x in range(0, 410, 30):
            cv2.line(img, (x, 20), (x, 940), (80, 80, 80), 2)
        block = _make_text_free_block(2, 18, 92, 950)
        ocr_text = "Volume 1\nChapter 2\nChapter 3\nChapter 4\nPage 120"

        with mock.patch.object(engine, "_request_ocr_text", return_value=ocr_text) as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(block.ocr_reject_reason, "book_spine_or_index")
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "book_spine_or_index")

    def test_text_free_watermark_credit_response_is_dropped_before_inpaint(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.full((180, 520, 3), 240, dtype=np.uint8)
        cv2.putText(img, "PATREON.COM/YTSNOW", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2, cv2.LINE_AA)
        block = _make_text_free_block(20, 45, 480, 115)

        with mock.patch.object(engine, "_request_ocr_text", return_value="PATREON.COM/YTSNOW YTSNOW.FANBOX.CC") as request:
            engine.process_image(img, [block])

        request.assert_called_once()
        self.assertEqual(block.text, "")
        self.assertEqual(block.ocr_status, "empty_initial")
        self.assertEqual(block.ocr_empty_reason, OCR_EMPTY_REASON_NON_TEXT_RESPONSE)
        self.assertEqual(block.ocr_reject_reason, "watermark_or_credit")
        self.assertEqual(engine.last_page_profile["request_records"][0]["non_text_reason"], "watermark_or_credit")
        kept, dropped = drop_rejected_empty_ocr_blocks([block])
        self.assertEqual(kept, [])
        self.assertEqual(dropped, [block])

    def test_schema_words_inside_real_paddleocr_vl_text_are_preserved(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        valid_samples = [
            "PATREON.COM/YTSNOW YTSNOW.FANBOX.CC",
            "number 4",
            "footer note",
            "OCR settings",
            "これは本文です",
            "header\nHello world",
        ]

        for sample in valid_samples:
            with self.subTest(sample=sample):
                blocks = [_make_block(10, 10, 110, 110)]
                with mock.patch.object(engine, "_request_ocr_text", return_value=sample):
                    engine.process_image(img, blocks)

                self.assertEqual(blocks[0].text, sample)
                self.assertEqual(blocks[0].ocr_status, "ok")
                self.assertEqual(engine.last_page_profile["request_records"][0]["status"], "ok")

    def test_parallel_ocr_stops_waiting_when_cancelled(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=2))
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [
            _make_block(10, 10, 210, 210),
            _make_block(250, 10, 430, 190),
        ]
        started = threading.Event()
        release = threading.Event()
        cancelled = False

        def cancel_checker() -> bool:
            return cancelled

        def slow_ocr(_crop) -> str:
            started.set()
            release.wait(timeout=5)
            return "テスト"

        def request_cancel_after_worker_starts() -> None:
            nonlocal cancelled
            started.wait(timeout=1)
            cancelled = True

        engine.set_cancel_checker(cancel_checker)
        canceller = threading.Thread(target=request_cancel_after_worker_starts)
        canceller.start()
        started_at = time.perf_counter()
        try:
            with mock.patch.object(engine, "_request_ocr_text", side_effect=slow_ocr):
                with self.assertRaises(OperationCancelledError):
                    engine.process_image(img, blocks)
            self.assertLess(time.perf_counter() - started_at, 1.5)
        finally:
            release.set()
            canceller.join(timeout=1)

    def test_single_worker_ocr_stops_waiting_when_cancelled(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)
        blocks = [_make_block(10, 10, 210, 210)]
        started = threading.Event()
        release = threading.Event()
        cancelled = False

        def cancel_checker() -> bool:
            return cancelled

        def slow_ocr(_crop) -> str:
            started.set()
            release.wait(timeout=5)
            return "テスト"

        def request_cancel_after_worker_starts() -> None:
            nonlocal cancelled
            started.wait(timeout=1)
            cancelled = True

        engine.set_cancel_checker(cancel_checker)
        canceller = threading.Thread(target=request_cancel_after_worker_starts)
        canceller.start()
        started_at = time.perf_counter()
        try:
            with mock.patch.object(engine, "_request_ocr_text", side_effect=slow_ocr):
                with self.assertRaises(OperationCancelledError):
                    engine.process_image(img, blocks)
            self.assertLess(time.perf_counter() - started_at, 1.5)
        finally:
            release.set()
            canceller.join(timeout=1)

    def test_query_gpu_metrics_cached_reuses_recent_sample(self) -> None:
        payload = {"available": True, "sampled_at": 1.0}

        with mock.patch("modules.utils.gpu_metrics.query_gpu_metrics", return_value=payload) as query:
            first = gpu_metrics_module.query_gpu_metrics_cached(ttl_sec=10.0)
            second = gpu_metrics_module.query_gpu_metrics_cached(ttl_sec=10.0)

        self.assertEqual(first, second)
        query.assert_called_once()

    def test_ocr_factory_cache_key_changes_with_generic_scheduler_mode(self) -> None:
        fixed_key = OCRFactory._create_cache_key(
            "PaddleOCR VL",
            "Japanese",
            _FakeSettings(scheduler_mode="fixed"),
            backend="onnx",
        )
        auto_key = OCRFactory._create_cache_key(
            "PaddleOCR VL",
            "Japanese",
            _FakeSettings(scheduler_mode="auto_v1"),
            backend="onnx",
        )

        self.assertNotEqual(fixed_key, auto_key)

    def test_paddleocr_vl_ocr_cache_key_includes_text_guard_version(self) -> None:
        cache_manager = CacheManager()
        img = np.zeros((40, 40, 3), dtype=np.uint8)

        paddle_key = cache_manager._get_ocr_cache_key(img, "English", "PaddleOCR VL", "cpu")
        other_key = cache_manager._get_ocr_cache_key(img, "English", "MangaLMM", "cpu")

        self.assertIn(CacheManager.PADDLEOCR_VL_CACHE_VERSION, paddle_key)
        self.assertNotIn(CacheManager.PADDLEOCR_VL_CACHE_VERSION, other_key)

    def test_send_request_retries_transient_http_failure(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings())
        engine.REQUEST_RETRY_BACKOFF_SECONDS = (0.0, 0.0)
        payload = {"file": "encoded", "fileType": 1}
        responses = [
            _FakeHTTPResponse(500),
            _FakeHTTPResponse(200, {"errorCode": 0, "result": {"markdown": {"text": "OK"}}}),
        ]

        with mock.patch("modules.ocr.ocr_paddle_VL.requests.post", side_effect=responses) as post:
            data = engine._send_request(payload)

        self.assertEqual(data["errorCode"], 0)
        self.assertEqual(post.call_count, 2)

    def test_request_telemetry_separates_logical_http_and_retry_counts(self) -> None:
        engine = PaddleOCRVLEngine()
        engine.initialize(_FakeSettings(scheduler_mode="fixed", parallel_workers=1))
        engine.REQUEST_RETRY_BACKOFF_SECONDS = (0.0, 0.0)
        engine._supports_max_new_tokens = False
        img = np.zeros((300, 300, 3), dtype=np.uint8)
        blocks = [_make_block(10, 10, 110, 110)]
        responses = [
            _FakeHTTPResponse(500),
            _FakeHTTPResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "テスト"},
                        }
                    ],
                },
            ),
        ]

        with mock.patch(
            "modules.ocr.ocr_paddle_VL.requests.post",
            side_effect=responses,
        ):
            engine.process_image(img, blocks)

        performance = engine.last_page_profile["performance"]
        self.assertEqual(performance["logical_request_count"], 1)
        self.assertEqual(performance["http_attempt_count"], 2)
        self.assertEqual(performance["http_retry_count"], 1)
        self.assertGreater(performance["request_bytes"], 0)
        self.assertGreater(performance["base64_chars"], 0)
        self.assertGreaterEqual(performance["encode_ms"], 0.0)
        self.assertGreaterEqual(performance["base64_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
