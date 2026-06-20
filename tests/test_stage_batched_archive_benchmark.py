from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_stage_batched_archive_pipeline as runner  # noqa: E402
from modules.ocr.selection import (  # noqa: E402
    OCR_MODE_BEST_LOCAL,
    OCR_MODE_HUNYUAN,
    OCR_MODE_PADDLE_VL,
)


class StageBatchedArchiveBenchmarkTests(unittest.TestCase):
    def test_resolve_ocr_mode_value_includes_optimal_plus(self) -> None:
        self.assertEqual(runner.resolve_ocr_mode_value("fastest"), OCR_MODE_BEST_LOCAL)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal"), OCR_MODE_BEST_LOCAL)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal+"), OCR_MODE_BEST_LOCAL)
        self.assertEqual(runner.resolve_ocr_mode_value("optimal-plus"), OCR_MODE_BEST_LOCAL)
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

    def test_attach_running_gemma_endpoint_uses_unmanaged_localhost(self) -> None:
        widget = mock.Mock()
        window = mock.Mock()
        window.settings_page.ui.credential_widgets = {"Custom Local Server(Gemma)_api_url": widget}

        runner.apply_attach_running_gemma_endpoint(window)

        widget.setText.assert_called_once_with("http://localhost:18080/v1")

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
        self.assertEqual(patched["app"]["ocr"], OCR_MODE_BEST_LOCAL)
        self.assertTrue(patched["app"]["use_gpu"])
        self.assertEqual(preset["app"]["ocr"], "PaddleOCR VL")
        self.assertFalse(preset["app"]["use_gpu"])

    def test_build_gemma_runtime_overrides_uses_only_explicit_values(self) -> None:
        self.assertEqual(
            runner.build_gemma_runtime_overrides(
                context_size=3072,
                threads=None,
                n_gpu_layers=23,
                n_parallel=1,
                predict=512,
                batch_size=1024,
                ubatch_size=512,
                cache_type_k="q8_0",
                cache_type_v="q8_0",
                flash_attn=True,
                no_warmup=True,
            ),
            {
                "context_size": 3072,
                "n_gpu_layers": 23,
                "n_parallel": 1,
                "predict": 512,
                "batch_size": 1024,
                "ubatch_size": 512,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": True,
                "no_warmup": True,
            },
        )

    def test_patch_preset_for_run_can_override_gemma_runtime(self) -> None:
        preset = {"gemma": {"context_size": 4096, "threads": 10, "n_gpu_layers": 23}}
        patched = runner.patch_preset_for_run(
            preset,
            ocr_mode="optimal-plus",
            gemma_runtime_overrides={
                "context_size": 3072,
                "threads": 8,
                "n_parallel": 1,
                "cache_type_k": "q8_0",
                "flash_attn": True,
            },
        )

        self.assertEqual(patched["gemma"]["context_size"], 3072)
        self.assertEqual(patched["gemma"]["threads"], 8)
        self.assertEqual(patched["gemma"]["n_gpu_layers"], 23)
        self.assertEqual(patched["gemma"]["n_parallel"], 1)
        self.assertEqual(patched["gemma"]["cache_type_k"], "q8_0")
        self.assertTrue(patched["gemma"]["flash_attn"])
        self.assertEqual(preset["gemma"]["context_size"], 4096)

    def test_speed_profile_expands_runtime_and_archive_defaults(self) -> None:
        profile = runner.resolve_speed_profile("ctx3072-fast-archive")

        self.assertEqual(profile["gemma_ctx_size"], 3072)
        self.assertEqual(profile["compression_level"], 0)
        self.assertEqual(profile["runtime_reuse_mode"], "signature")

        ctx2560_profile = runner.resolve_speed_profile("ctx2560-fast-archive")
        self.assertEqual(ctx2560_profile["gemma_ctx_size"], 2560)
        self.assertEqual(ctx2560_profile["compression_level"], 0)

        ctx2048_profile = runner.resolve_speed_profile("ctx2048-gpu23-fast")
        self.assertEqual(ctx2048_profile["gemma_ctx_size"], 2048)
        self.assertEqual(ctx2048_profile["gemma_gpu_layers"], 23)

    def test_extreme_speed_profile_can_raise_gpu_layers(self) -> None:
        profile = runner.resolve_speed_profile("ctx2560-gpu25-danger")

        self.assertEqual(profile["gemma_ctx_size"], 2560)
        self.assertEqual(profile["gemma_gpu_layers"], 25)
        self.assertEqual(profile["compression_level"], 0)
        self.assertTrue(profile["allow_failure"])

        gpu24_profile = runner.resolve_speed_profile("ctx3072-gpu24-extreme")
        self.assertTrue(gpu24_profile["allow_failure"])

        gpu26_profile = runner.resolve_speed_profile("ctx2048-gpu26-danger")
        self.assertEqual(gpu26_profile["gemma_gpu_layers"], 26)
        self.assertTrue(gpu26_profile["allow_failure"])

    def test_gemma_command_signature_accepts_matching_overrides(self) -> None:
        command = [
            "--model",
            "/models/gemma.gguf",
            "--ctx-size",
            "3072",
            "--threads",
            "10",
            "--n-gpu-layers",
            "23",
            "--parallel",
            "1",
            "--n-predict",
            "512",
            "-b",
            "1024",
            "-ub",
            "512",
            "--cache-type-k",
            "q8_0",
            "--cache-type-v",
            "q8_0",
            "--flash-attn",
            "--no-warmup",
        ]

        self.assertTrue(
            runner.gemma_command_matches_overrides(
                command,
                {
                    "context_size": 3072,
                    "threads": 10,
                    "n_gpu_layers": 23,
                    "n_parallel": 1,
                    "predict": 512,
                    "batch_size": 1024,
                    "ubatch_size": 512,
                    "cache_type_k": "q8_0",
                    "cache_type_v": "q8_0",
                    "flash_attn": True,
                    "no_warmup": True,
                },
            )
        )

    def test_gemma_command_signature_rejects_mismatched_context(self) -> None:
        command = ["--ctx-size", "4096", "--threads", "10", "--n-gpu-layers", "23", "--parallel", "1"]

        self.assertFalse(
            runner.gemma_command_matches_overrides(
                command,
                {"context_size": 3072, "threads": 10, "n_gpu_layers": 23, "n_parallel": 1},
            )
        )

    def test_stage_isolation_stops_existing_gemma_containers(self) -> None:
        def fake_inspect(name: str) -> list[str] | None:
            return ["--ctx-size", "4096"] if name == "gemma-local-server" else None

        with (
            mock.patch.object(runner, "inspect_container_command", side_effect=fake_inspect),
            mock.patch.object(runner, "remove_containers") as remove_containers,
        ):
            self.assertTrue(runner.stop_gemma_for_stage_isolation())

        remove_containers.assert_called_once_with(["gemma-local-server"])

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
