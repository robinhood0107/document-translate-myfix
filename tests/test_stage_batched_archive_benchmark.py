from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_stage_batched_archive_pipeline as runner  # noqa: E402
from modules.ocr.selection import (  # noqa: E402
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_BEST_LOCAL_PLUS,
    OCR_MODE_HUNYUAN,
    OCR_MODE_PADDLE_VL,
)


class StageBatchedArchiveBenchmarkTests(unittest.TestCase):
    def test_resolve_ocr_mode_value_includes_optimal_plus(self) -> None:
        self.assertEqual(runner.resolve_ocr_mode_value("fastest"), OCR_MODE_BEST_LOCAL)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal"), OCR_MODE_BEST_LOCAL)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal+"), OCR_MODE_BEST_LOCAL_PLUS)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal-plus"), OCR_MODE_BEST_LOCAL_PLUS)
        self.assertEqual(runner.resolve_ocr_mode_value("paddleocr-vl"), OCR_MODE_PADDLE_VL)
        self.assertEqual(runner.resolve_ocr_mode_value("hunyuanocr"), OCR_MODE_HUNYUAN)

    def test_default_preset_tracks_optimal_plus(self) -> None:
        self.assertEqual(
            runner.default_preset_for_ocr_mode("optimal-plus"),
            runner.DEFAULT_OPTIMAL_PLUS_PRESET,
        )
        self.assertEqual(
            runner.default_preset_for_ocr_mode("optimal"),
            runner.DEFAULT_FAST_PRESET,
        )

    def test_reserve_unique_path_does_not_overwrite_existing_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "example_source_translated.cbz"
            base.write_text("existing", encoding="utf-8")
            candidate = runner.reserve_unique_path(base)
            self.assertEqual(candidate.name, "example_source_translated_001.cbz")
            self.assertEqual(base.read_text(encoding="utf-8"), "existing")

    def test_archive_file_name_preserves_source_version_suffix(self) -> None:
        self.assertEqual(
            runner.build_preserved_archive_file_name("source chapter v01 c14 (E)", "cbz"),
            "source chapter v01 c14 (E)_translated.cbz",
        )

    def test_collect_render_fit_summary_flags_tiny_fonts_without_raw_text(self) -> None:
        summary = runner.collect_render_fit_summary(
            [
                {
                    "image_path": "/tmp/page-001.png",
                    "viewer_state": {
                        "text_items_state": [
                            {
                                "font_size": 9.5,
                                "source_rect": [1, 2, 30, 20],
                                "width": 28,
                                "height": 18,
                                "translation_raw": "sensitive source text",
                                "render_text": "sensitive translated text",
                            },
                            {
                                "font_size": 18,
                                "source_rect": [4, 5, 60, 30],
                                "translation_raw": "normal source text",
                                "render_text": "normal translated text",
                            },
                        ]
                    },
                }
            ],
            tiny_font_threshold=12,
        )

        self.assertEqual(summary["item_count"], 2)
        self.assertEqual(summary["tiny_item_count"], 1)
        self.assertEqual(summary["min_font_size"], 9.5)
        self.assertEqual(summary["tiny_items"][0]["render_text_length"], len("sensitive translated text"))
        self.assertNotIn("sensitive source text", json.dumps(summary, ensure_ascii=False))
        self.assertNotIn("sensitive translated text", json.dumps(summary, ensure_ascii=False))

    def test_transient_ocr_errors_include_service_warmup_failures(self) -> None:
        self.assertTrue(runner.is_transient_ocr_runtime_error(RuntimeError("PaddleOCR VL service returned HTTP 500.")))
        self.assertTrue(runner.is_transient_ocr_runtime_error(RuntimeError("Unable to reach the local PaddleOCR VL service.")))
        self.assertFalse(runner.is_transient_ocr_runtime_error(RuntimeError("OCR quality too low after retry.")))

    def test_patch_preset_for_run_copies_and_sets_ocr(self) -> None:
        preset = {
            "app": {
                "ocr": "PaddleOCR VL",
                "translator": "Custom Local Server(Gemma)",
                "use_gpu": False,
            }
        }
        patched = runner.patch_preset_for_run(preset, ocr_mode="optimal-plus")
        self.assertEqual(patched["app"]["ocr"], OCR_MODE_BEST_LOCAL_PLUS)
        self.assertTrue(patched["app"]["use_gpu"])
        self.assertEqual(preset["app"]["ocr"], "PaddleOCR VL")
        self.assertFalse(preset["app"]["use_gpu"])

    def test_patch_preset_for_run_can_disable_line_protect(self) -> None:
        preset = {"mask_refiner_settings": {"keep_existing_lines": True}}
        patched = runner.patch_preset_for_run(
            preset,
            ocr_mode="optimal-plus",
            disable_line_protect=True,
        )

        self.assertFalse(patched["mask_refiner_settings"]["keep_existing_lines"])
        self.assertTrue(preset["mask_refiner_settings"]["keep_existing_lines"])

    def test_patch_preset_for_run_can_override_ctd_mask_dilation(self) -> None:
        preset = {"mask_refiner_settings": {"ctd_mask_dilate_size": 2}}
        patched = runner.patch_preset_for_run(
            preset,
            ocr_mode="optimal-plus",
            ctd_mask_dilate_size=4,
        )

        self.assertEqual(patched["mask_refiner_settings"]["ctd_mask_dilate_size"], 4)
        self.assertEqual(preset["mask_refiner_settings"]["ctd_mask_dilate_size"], 2)


if __name__ == "__main__":
    unittest.main()
