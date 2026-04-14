from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import SettingsPage
from modules.ocr.factory import OCRFactory
from modules.ocr.ocr_paddle_VL import PaddleOCRVLEngine
from modules.utils import gpu_metrics as gpu_metrics_module
from modules.utils.textblock import TextBlock


def _make_block(x1: int, y1: int, x2: int, y2: int) -> TextBlock:
    return TextBlock(
        text_bbox=np.array([x1, y1, x2, y2], dtype=np.int32),
        text_class="text_bubble",
        source_lang="ja",
        direction="vertical",
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
        server_url: str = "http://127.0.0.1:28118/layout-parsing",
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


class PaddleOCRVLEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        OCRFactory._engines.clear()
        gpu_metrics_module._GPU_METRICS_CACHE_VALUE = None
        gpu_metrics_module._GPU_METRICS_CACHE_EXPIRES_AT = 0.0

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
        self.assertEqual(record["bbox"], [5, 5, 115, 115])
        self.assertGreater(record["crop_area_px"], 0)
        self.assertIsNotNone(record["enqueue_ts"])
        self.assertIsNotNone(record["start_ts"])
        self.assertIsNotNone(record["end_ts"])
        self.assertIsNotNone(record["elapsed_ms"])
        self.assertEqual(record["status"], "ok")

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


if __name__ == "__main__":
    unittest.main()
