from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pipeline.batch_processor import BatchProcessor
from modules.ocr.selection import STAGE_BATCHED_WORKFLOW_MODE

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

try:
    import benchmark_pipeline  # noqa: E402
except ModuleNotFoundError:
    benchmark_pipeline = None


class _PreviewMain:
    def __init__(self) -> None:
        self.events = []
        self._intermediate_preview_disabled_notices = set()

    def report_runtime_progress(self, payload):
        self.events.append(dict(payload))


class _BenchmarkMain:
    def __init__(self) -> None:
        self.events = []
        self._current_batch_run_type = "batch"
        self.settings_page = SimpleNamespace(
            get_workflow_mode=lambda: STAGE_BATCHED_WORKFLOW_MODE,
        )

    def emit_memlog(self, tag, **payload):
        self.events.append((tag, dict(payload)))


class BatchDebugOutputTests(unittest.TestCase):
    def test_inpainted_debug_image_stays_out_of_final_output_folder(self) -> None:
        processor = object.__new__(BatchProcessor)
        with tempfile.TemporaryDirectory() as temp_dir:
            export_root = os.path.join(temp_dir, "comic_translate_run")
            source_path = os.path.join(temp_dir, "92.png")
            with open(source_path, "wb") as fh:
                fh.write(b"source")

            output_path = processor._write_inpainted_debug_image(
                export_root=export_root,
                archive_bname="",
                image_path=source_path,
                cleaned_image=np.zeros((8, 8, 3), dtype=np.uint8),
                export_settings={
                    "export_inpainted_image": True,
                    "resolved_automatic_output_image_format": "png",
                },
            )

            self.assertTrue(os.path.isfile(output_path))
            self.assertIn(os.path.join("comic_translate_run", "inpainted_images"), output_path)
            self.assertTrue(output_path.endswith("92_cleaned.png"))
            self.assertFalse(os.path.exists(os.path.join(temp_dir, "92_cleaned.png")))

    def test_raw_text_export_does_not_implicitly_create_ocr_diagnostics(
        self,
    ) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "92.png")
            processor._write_json_exports(
                temp_dir,
                "run-token",
                "",
                image_path,
                np.zeros((8, 8, 3), dtype=np.uint8),
                [],
                {"processing_summary": {"ocr_engine": "PaddleOCR VL"}},
                "Japanese",
                {
                    "export_raw_text": True,
                    "export_translated_text": False,
                    "export_ocr_debug": False,
                },
            )

            raw_path = os.path.join(
                temp_dir,
                "comic_translate_run-token",
                "raw_texts",
                "92_raw.json",
            )
            self.assertTrue(os.path.isfile(raw_path))
            self.assertFalse(
                any(
                    "ocr-debug" in path.name
                    for path in Path(temp_dir).rglob("*")
                )
            )

    def test_preview_is_not_generated_when_debug_checkbox_is_off(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "92.png")
            preview_path = os.path.join(temp_dir, "92_raw_mask.png")
            with open(image_path, "wb") as fh:
                fh.write(b"source")
            with open(preview_path, "wb") as fh:
                fh.write(b"preview")

            processor._maybe_emit_preview_image(
                index=0,
                total=1,
                image_path=image_path,
                stage_key="raw_mask",
                stage_label="원본 마스크",
                export_settings={"export_raw_mask": False},
                preferred_path=preview_path,
            )

            self.assertEqual(len(processor.main_page.events), 1)
            event = processor.main_page.events[0]
            self.assertEqual(event["preview_disabled_reason"], "intermediate_preview_disabled")
            self.assertNotIn("preview_path", event)

    def test_preview_uses_existing_debug_export_when_checkbox_is_on(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = os.path.join(temp_dir, "92.png")
            preview_path = os.path.join(temp_dir, "92_raw_mask.png")
            with open(image_path, "wb") as fh:
                fh.write(b"source")
            with open(preview_path, "wb") as fh:
                fh.write(b"preview")

            processor._maybe_emit_preview_image(
                index=0,
                total=1,
                image_path=image_path,
                stage_key="raw_mask",
                stage_label="원본 마스크",
                export_settings={"export_raw_mask": True},
                preferred_path=preview_path,
            )

            self.assertEqual(len(processor.main_page.events), 1)
            event = processor.main_page.events[0]
            self.assertEqual(event["preview_path"], preview_path)
            self.assertFalse(event["temporary_preview"])

    def test_detector_debug_write_failure_is_fail_open(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _PreviewMain()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch(
            "pipeline.batch_processor.active_debug_page_directory",
            return_value=temp_dir,
        ), mock.patch(
            "pipeline.batch_processor.atomic_debug_image",
            side_effect=OSError("blocked"),
        ), mock.patch(
            "pipeline.batch_processor.logger.warning",
        ):
            output = processor._write_detector_overlay_debug_image(
                export_root=temp_dir,
                archive_bname="",
                image_path=os.path.join(temp_dir, "page.png"),
                image=np.zeros((8, 8, 3), dtype=np.uint8),
                blk_list=[],
                export_settings={"export_detector_overlay": True},
            )

        self.assertEqual(output, "")

    def test_benchmark_events_expose_product_pipeline_entrypoint_contract(self) -> None:
        processor = object.__new__(BatchProcessor)
        processor.main_page = _BenchmarkMain()

        processor._emit_benchmark_event("batch_run_start", total_images=3)

        tag, event = processor.main_page.events[0]
        self.assertEqual(tag, "batch_run_start")
        self.assertTrue(event["product_pipeline_entrypoint"])
        self.assertEqual(event["workflow_mode"], STAGE_BATCHED_WORKFLOW_MODE)
        self.assertEqual(event["alignment_id"], 1)
        self.assertEqual(event["vertical_alignment_id"], 1)
        self.assertEqual(event["runner_render_mode"], "product")

    @unittest.skipIf(
        benchmark_pipeline is None,
        "benchmark runner is available only on benchmarking/lab",
    )
    def test_page_snapshot_block_includes_mask_and_review_diagnostics(self) -> None:
        block = SimpleNamespace(
            xyxy=[1, 2, 11, 22],
            bubble_xyxy=None,
            angle=0,
            text_class="text_free",
            text="フーー",
            translation="후우",
            _render_text="",
            _text_fit_status="needs_review_text_free_mask",
            _render_normalization_reasons=["render_without_erase_mask"],
            block_final_mask_pixel_count=0,
            block_mask_iou=0.0,
            block_mask_span_coverage=0.0,
            block_mask_bbox=None,
            block_mask_source="none",
            block_mask_decision="review",
            _render_restore_applied=True,
            ui_panel_mode="preserve_original",
            ui_panel_preview_path="",
            mask_decision="review",
            mask_reject_reason="render_without_erase_mask",
        )

        payload = benchmark_pipeline._serialize_page_snapshot_block(block)

        self.assertEqual(payload["block_final_mask_pixel_count"], 0)
        self.assertEqual(payload["block_mask_source"], "none")
        self.assertEqual(payload["block_mask_decision"], "review")
        self.assertTrue(payload["render_restore_applied"])
        self.assertEqual(payload["ui_panel_mode"], "preserve_original")
        self.assertEqual(payload["mask_reject_reason"], "render_without_erase_mask")


if __name__ == "__main__":
    unittest.main()
