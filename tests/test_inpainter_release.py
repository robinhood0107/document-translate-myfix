from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pipeline.inpainting import InpaintingHandler
from pipeline.stage_batched_processor import StageBatchedProcessor, StagePageContext


class InpainterReleaseTests(unittest.TestCase):
    def test_targeted_release_preserves_materialized_edit_mask(self) -> None:
        handler = InpaintingHandler(SimpleNamespace())
        cached_model = SimpleNamespace()
        handler.inpainter_cache = SimpleNamespace(
            runtime_device="cuda",
            model=cached_model,
            session=None,
        )
        handler.cached_inpainter_key = "lama_large_512px"
        edit_mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        handler.last_inpaint_edit_mask = edit_mask
        before = {
            "process": {
                "available": True,
                "allocated_mb": 1024.0,
                "reserved_mb": 1280.0,
            },
            "driver": {"available": False, "primary": None},
        }
        gate = {
            "required": True,
            "measurement_available": True,
            "observed": True,
            "status": "observed",
        }

        with mock.patch(
            "pipeline.inpainting.query_cuda_handoff_metrics",
            return_value=before,
        ), mock.patch(
            "pipeline.inpainting.release_source_lama_cache",
            return_value={
                "cache_entry_count": 1,
                "loaded_model_count": 1,
                "gpu_loaded_model_count": 1,
                "gpu_release_expected": True,
            },
        ), mock.patch(
            "pipeline.inpainting.cleanup_python_cuda_memory",
            return_value={"gc_collected": 1, "errors": []},
        ), mock.patch(
            "pipeline.inpainting.wait_for_vram_release",
            return_value=gate,
        ) as wait_for_release:
            report = handler.release_inpainter_resources()

        self.assertIsNone(handler.inpainter_cache)
        self.assertIsNone(handler.cached_inpainter_key)
        self.assertIs(handler.last_inpaint_edit_mask, edit_mask)
        np.testing.assert_array_equal(handler.last_inpaint_edit_mask, edit_mask)
        self.assertTrue(report["gpu_release_expected"])
        self.assertEqual(report["vram_release_gate"], gate)
        self.assertEqual(
            report["python_native_cleanup"],
            {"gc_collected": 1, "errors": []},
        )
        wait_for_release.assert_called_once_with(
            before,
            gpu_release_expected=True,
            timeout_sec=5.0,
            poll_interval_sec=0.1,
            min_drop_mb=16.0,
        )

    def test_stage_handoff_preserves_page_outputs_before_starting_gemma(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        events: list[str] = []
        release_report = {
            "gpu_release_expected": True,
            "vram_release_gate": {
                "required": True,
                "observed": True,
                "status": "observed",
                "elapsed_sec": 0.2,
            },
        }
        processor.inpainting = SimpleNamespace(
            release_inpainter_resources=lambda: (
                events.append("release") or release_report
            )
        )
        processor._emit_benchmark_event = lambda *_args, **_kwargs: events.append("telemetry")
        processor._raise_if_cancelled = lambda: events.append("cancel-check")
        processor._start_gemma_prewarm = lambda: events.append("gemma-start")

        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        patch_image = image[:1, :1].copy()
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
            inpaint_input_img=image,
            mask=mask,
            patches=[{"bbox": [0, 0, 1, 1], "image": patch_image}],
        )
        image_before = image.copy()
        mask_before = mask.copy()
        patch_before = patch_image.copy()

        processor._release_inpainter_before_gemma([page])

        self.assertEqual(
            events,
            ["release", "telemetry", "cancel-check", "gemma-start"],
        )
        np.testing.assert_array_equal(page.inpaint_input_img, image_before)
        np.testing.assert_array_equal(page.mask, mask_before)
        np.testing.assert_array_equal(page.patches[0]["image"], patch_before)

    def test_failed_vram_gate_blocks_gemma_start(self) -> None:
        processor = object.__new__(StageBatchedProcessor)
        started = False
        processor.inpainting = SimpleNamespace(
            release_inpainter_resources=lambda: {
                "gpu_release_expected": True,
                "vram_release_gate": {
                    "required": True,
                    "observed": False,
                    "status": "timeout",
                    "elapsed_sec": 5.0,
                },
            }
        )
        processor._emit_benchmark_event = lambda *_args, **_kwargs: None

        def start_gemma() -> None:
            nonlocal started
            started = True

        processor._start_gemma_prewarm = start_gemma
        page = StagePageContext(
            image_path="example.png",
            image_name="example.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        with self.assertRaisesRegex(RuntimeError, "VRAM release was not observed"):
            processor._release_inpainter_before_gemma([page])

        self.assertFalse(started)


if __name__ == "__main__":
    unittest.main()
