from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

from modules.utils.debug_artifacts import DebugArtifactError
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    export_inpaint_debug_artifacts,
)
from pipeline.batch_processor import BatchProcessor


@dataclass
class _Block:
    xyxy: list[int]
    bubble_xyxy: list[int] | None = None
    text_class: str = "text_bubble"
    translation: str = ""
    _render_translation_raw: str = ""
    _render_text: str = ""
    _render_html_applied: bool = False
    _render_fallback_font_family: str = ""
    _render_normalization_applied: bool = False
    _render_normalization_reasons: list[str] = field(default_factory=list)
    _render_original_xyxy: list[int] | None = None
    _render_bubble_xyxy: list[int] | None = None
    _render_area_xyxy: list[int] | None = None
    _render_area_source: str = ""
    _mask_anchor_xyxy: list[int] | None = None
    _mask_anchor_source: str = ""
    _mask_anchor_relation: str = ""
    inpaint_bboxes: list[list[int]] = field(default_factory=list)
    _hard_box_applied: bool = False
    _hard_box_reason_codes: list[str] = field(default_factory=list)
    _legacy_fill_ratio: float = 0.0
    _rescue_fill_ratio: float = 0.0
    _legacy_mask_pixel_count: int = 0
    _rescue_mask_pixel_count: int = 0
    _final_mask_pixel_count: int = 0
    _erase_mode: str = ""
    _erase_edit_pixel_count: int = 0
    _erase_protect_pixel_count: int = 0
    _erase_skipped_reason: str = ""
    ui_panel_mode: str = ""
    ui_panel_preview_path: str = ""
    mask_decision: str = ""
    mask_reject_reason: str = ""
    semantic_role: str = ""
    processing_action: str = ""
    processing_decision_source: str = ""
    processing_decision_reasons: list[str] = field(default_factory=list)
    canonical_block_id: str = ""
    duplicate_alias_block_ids: list[str] = field(default_factory=list)
    duplicate_alias_count: int = 0
    merge_split_diagnostics: dict = field(default_factory=dict)
    mask_strategy: str = ""
    mask_strategy_reason: str = ""
    mask_actual_bbox: list[int] | None = None
    mask_actual_pixel_count: int = 0


class InpaintDebugTests(unittest.TestCase):
    @staticmethod
    def _load_export_module():
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_inpaint_debug.py"
        spec = importlib.util.spec_from_file_location("export_inpaint_debug_for_test", script_path)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_export_inpaint_debug_script_imports_blockwise_lama_runner(self) -> None:
        module = self._load_export_module()

        self.assertTrue(callable(getattr(module, "source_lama_blockwise_inpaint", None)))

    def test_export_inpaint_debug_defaults_to_original_and_parses_private_inputs(self) -> None:
        module = self._load_export_module()
        settings = module._SettingsStub(inpainter="AOT", use_gpu=False)

        self.assertEqual(settings.get_hd_strategy_settings()["strategy"], "Original")
        self.assertFalse(
            settings.get_hd_strategy_settings()["developer_performance_mode"]
        )
        self.assertEqual(str(module.get_config(settings).hd_strategy), "Original")
        parsed = module._build_argument_parser().parse_args(
            [
                "--corpus",
                "japan",
                "--input",
                "one.png",
                "--input",
                "two.png",
                "--auto-max-font-profile",
                "strong",
            ]
        )
        self.assertEqual(parsed.corpus, "japan")
        self.assertEqual(parsed.input, [Path("one.png"), Path("two.png")])
        self.assertEqual(parsed.auto_max_font_profile, "strong")
        self.assertFalse(parsed.require_cuda_lama)
        self.assertFalse(parsed.require_rounded_bubble_gate)

        resize_settings = module._SettingsStub(
            inpainter="AOT",
            use_gpu=False,
            hd_strategy="Resize",
            developer_performance_mode=True,
            resize_limit=1440,
        )
        resize_config = module.get_config(resize_settings)
        self.assertEqual(str(resize_config.hd_strategy), "Resize")
        self.assertEqual(resize_config.hd_strategy_resize_limit, 1440)

        crop_settings = module._SettingsStub(
            inpainter="AOT",
            use_gpu=False,
            hd_strategy="Crop",
            developer_performance_mode=True,
            crop_margin=320,
            crop_trigger_size=768,
        )
        crop_config = module.get_config(crop_settings)
        self.assertEqual(str(crop_config.hd_strategy), "Crop")
        self.assertEqual(crop_config.hd_strategy_crop_margin, 320)
        self.assertEqual(crop_config.hd_strategy_crop_trigger_size, 768)

    def test_export_required_gates_fail_closed(self) -> None:
        module = self._load_export_module()
        summary = {
            "inpainter": "lama_large_512px",
            "use_gpu": True,
            "hd_strategy": "Original",
            "inpainter_runtime": {
                "actual_device": "cuda",
                "device_verified_from_model": True,
                "cpu_fallback_used": False,
            },
            "cpu_fallback_count": 0,
            "non_cuda_refiner_count": 0,
            "zero_block_count": 0,
            "empty_final_mask_count": 0,
            "image_count": 1,
            "success_count": 1,
            "runtime_inference_call_count": 1,
        }
        record = {
            "image": "page.png",
            "bubble_block_count": 1,
            "bubble_silhouette_fallback_count": 0,
            "protected_corner_mask_pixel_count": 100,
            "protected_corner_final_mask_pixel_count": 0,
            "protected_corner_changed_pixel_count": 0,
            "text_anchor_final_mask_pixel_count": 10,
            "text_anchor_changed_pixel_count": 10,
            "changed_outside_final_mask_pixel_count_exact": 0,
        }

        self.assertEqual(
            module._required_gate_failures(
                summary,
                {"private": [record]},
                require_cuda_lama=True,
                require_rounded_bubble_gate=True,
                required_image_count=1,
            ),
            [],
        )
        record["protected_corner_changed_pixel_count"] = 1
        failures = module._required_gate_failures(
            summary,
            {"private": [record]},
            require_cuda_lama=True,
            require_rounded_bubble_gate=True,
            required_image_count=1,
        )
        self.assertIn("private/page.png:protected_corner_changed", failures)

        summary["image_count"] = 0
        summary["success_count"] = 0
        summary["runtime_inference_call_count"] = 0
        failures = module._required_gate_failures(
            summary,
            {"private": []},
            require_cuda_lama=True,
            require_rounded_bubble_gate=False,
            required_image_count=1,
        )
        self.assertIn("no_input_images", failures)
        self.assertIn("no_inpaint_inference", failures)
        self.assertIn("image_count_mismatch:0!=1", failures)

    def test_export_inpaint_debug_collects_lama_runtime_diagnostics(self) -> None:
        module = self._load_export_module()
        settings = module._SettingsStub(inpainter="lama_large_512px", use_gpu=True)
        block = SimpleNamespace(
            xyxy=np.asarray([8, 8, 24, 24], dtype=np.int32),
            bubble_xyxy=None,
            text_class="text_free",
            text="demo",
            translation="",
            source_lang="ja",
            inpaint_bboxes=None,
        )
        detector = SimpleNamespace(
            detect=lambda _image: [block],
            last_engine_name="detector",
            last_device="cuda",
        )
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[8:24, 8:24] = 255
        details = {
            "raw_mask": mask.copy(),
            "final_mask": mask.copy(),
            "final_mask_pre_expand": mask.copy(),
            "final_mask_post_expand": mask.copy(),
            "refiner_backend": "torch",
            "refiner_device": "cuda",
        }
        diagnostic = {
            "actual_device": "cuda:0",
            "cpu_fallback_used": False,
            "status": "completed",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "sample.png"
            Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(image_path)
            with mock.patch.object(
                module,
                "generate_mask",
                return_value=details,
            ), mock.patch.object(
                module,
                "source_lama_blockwise_inpaint",
                return_value=(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    mask.copy(),
                    [diagnostic],
                ),
            ) as run_lama, mock.patch.object(
                module,
                "refine_bubble_residue_inpaint",
                return_value=(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    mask.copy(),
                    {"applied": False, "component_count": 0, "block_count": 0},
                ),
            ), mock.patch.object(
                module,
                "apply_duplicate_bubble_inner_fill",
                return_value=(
                    np.zeros((32, 32, 3), dtype=np.uint8),
                    mask.copy(),
                    {
                        "applied": False,
                        "component_count": 0,
                        "block_count": 0,
                        "duplicate_bubble_inner_fill": {"applied": False},
                    },
                ),
            ) as duplicate_fill, mock.patch.object(
                module,
                "build_inpaint_debug_metadata",
                return_value={},
            ), mock.patch.object(
                module,
                "export_inpaint_debug_artifacts",
            ), mock.patch.object(
                module,
                "_write_image",
            ):
                record = module._process_image(
                    image_path,
                    Path(temp_dir),
                    detector,
                    SimpleNamespace(),
                    settings,
                )

        self.assertTrue(run_lama.call_args.kwargs["return_diagnostics"])
        duplicate_fill.assert_called_once()
        self.assertEqual(record["hd_strategy"], "Original")
        self.assertEqual(record["refiner_device"], "cuda")
        self.assertEqual(record["inpaint_runtime_inference_call_count"], 1)
        self.assertEqual(record["inpaint_runtime_cpu_fallback_count"], 0)

    def test_build_metadata_counts_masks_and_blocks(self) -> None:
        raw_mask = np.zeros((8, 8), dtype=np.uint8)
        raw_mask[1:3, 1:3] = 255
        cleanup_delta = np.zeros((8, 8), dtype=np.uint8)
        cleanup_delta[4:5, 4:6] = 255
        block = _Block(
            xyxy=[1, 1, 4, 5],
            bubble_xyxy=[0, 0, 7, 7],
            text_class="text_bubble",
            translation='본명 「나나세 아야카」 25세♥',
            _render_translation_raw='본명 「나나세 아야카」 25세♥',
            _render_text='본명 "나나세 아야카" 25세',
            _render_html_applied=True,
            _render_fallback_font_family="Malgun Gothic",
            _render_normalization_applied=True,
            _render_normalization_reasons=["quote-to-ascii", "heart-dropped"],
            _render_original_xyxy=[1, 1, 4, 5],
            _render_bubble_xyxy=[0, 0, 7, 7],
            _render_area_xyxy=[0, 0, 7, 7],
            _render_area_source="detected_bubble",
            _mask_anchor_xyxy=[1, 1, 4, 5],
            _mask_anchor_source="render_original",
            _mask_anchor_relation="render_area",
            inpaint_bboxes=[[1, 1, 4, 5]],
            _hard_box_applied=True,
            _hard_box_reason_codes=["edge_dense"],
            _legacy_fill_ratio=0.05,
            _rescue_fill_ratio=0.10,
            _legacy_mask_pixel_count=4,
            _rescue_mask_pixel_count=2,
            _final_mask_pixel_count=6,
            _erase_mode="bubble_flat_fill",
            _erase_edit_pixel_count=12,
            _erase_protect_pixel_count=3,
            ui_panel_mode="preserve_original",
            ui_panel_preview_path="previews/page_block_0.png",
            mask_decision="review",
            mask_reject_reason="embedded_ui_panel_layout_review",
            semantic_role="ui_or_sign",
            processing_action="preserve",
            processing_decision_source="embedded_ui_cluster",
            processing_decision_reasons=["embedded_device_ui_cluster"],
            canonical_block_id="canonical-1",
            duplicate_alias_block_ids=["alias-1"],
            duplicate_alias_count=1,
            merge_split_diagnostics={"automatic_merge": False},
            mask_strategy="preserve_original",
            mask_strategy_reason="processing_action_preserve",
            mask_actual_bbox=[1, 1, 4, 5],
            mask_actual_pixel_count=12,
        )

        metadata = build_inpaint_debug_metadata(
            image_path="page.png",
            run_type="batch",
            detector_key="RT-DETR-v2",
            detector_engine="RTDetrV2ONNXDetection",
            device="cpu",
            inpainter="AOT",
            hd_strategy="Resize",
            blocks=[block],
            raw_mask=raw_mask,
            cleanup_delta=cleanup_delta,
            mask_quality_policy="lama_safe_bubble_and_text_free",
            mask_policy_bubble_clamp_applied_count=2,
            mask_policy_text_free_glyph_applied_count=1,
            mask_policy_removed_pixel_count=17,
            mask_policy_outside_bubble_removed_pixel_count=5,
            ctd_legacy_rectangle_rescue_disabled=True,
            text_free_image_glyph_rescue_count=3,
            text_free_image_glyph_rescue_mask_pixel_count=41,
            mask_policy_version="ctd_lama_mask_policy_v2",
            mask_candidate_source="ctd_refined",
            mask_decision="review",
            mask_reject_reason="ambiguous_candidate_scores",
            mask_score_outside_change=0.01,
            mask_score_outline_damage=0.02,
            mask_score_residue=0.03,
            mask_score_color_delta=0.04,
            ui_panel_mode="preserve_original",
            ui_panel_preview_path="previews/page.png",
            cleanup_stats={
                "applied": True,
                "component_count": 2,
                "block_count": 1,
                "duplicate_bubble_inner_fill": {
                    "applied": True,
                    "duplicate_bubble_inner_fill_pixel_count": 25,
                    "duplicate_bubble_inner_fill_backend": "bubble_flat_fill",
                },
            },
        )

        self.assertEqual(metadata["block_count"], 1)
        self.assertEqual(metadata["raw_mask_pixel_count"], 4)
        self.assertEqual(metadata["cleanup_delta_pixel_count"], 2)
        self.assertTrue(metadata["cleanup_applied"])
        self.assertEqual(metadata["hard_box_applied_count"], 1)
        self.assertEqual(metadata["blocks"][0]["text_class"], "text_bubble")
        self.assertEqual(metadata["blocks"][0]["inpaint_bboxes"], [[1, 1, 4, 5]])
        self.assertEqual(metadata["blocks"][0]["mask_anchor_xyxy"], [1, 1, 4, 5])
        self.assertEqual(metadata["blocks"][0]["mask_anchor_source"], "render_original")
        self.assertEqual(metadata["blocks"][0]["mask_anchor_relation"], "render_area")
        self.assertEqual(metadata["blocks"][0]["render_original_xyxy"], [1, 1, 4, 5])
        self.assertEqual(metadata["blocks"][0]["render_area_xyxy"], [0, 0, 7, 7])
        self.assertEqual(metadata["blocks"][0]["render_bubble_xyxy"], [0, 0, 7, 7])
        self.assertEqual(metadata["blocks"][0]["render_area_source"], "detected_bubble")
        self.assertTrue(metadata["blocks"][0]["hard_box_applied"])
        self.assertEqual(metadata["blocks"][0]["hard_box_reason_codes"], ["edge_dense"])
        self.assertEqual(
            metadata["blocks"][0]["translation_raw"],
            '본명 「나나세 아야카」 25세♥',
        )
        self.assertEqual(
            metadata["blocks"][0]["render_text"],
            '본명 "나나세 아야카" 25세',
        )
        self.assertTrue(metadata["blocks"][0]["render_html_applied"])
        self.assertEqual(
            metadata["blocks"][0]["render_fallback_font_family"],
            "Malgun Gothic",
        )
        self.assertTrue(metadata["blocks"][0]["render_normalization_applied"])
        self.assertEqual(
            metadata["blocks"][0]["render_normalization_reasons"],
            ["quote-to-ascii", "heart-dropped"],
        )
        self.assertEqual(metadata["blocks"][0]["erase_mode"], "bubble_flat_fill")
        self.assertEqual(metadata["blocks"][0]["erase_edit_pixel_count"], 12)
        self.assertEqual(metadata["blocks"][0]["erase_protect_pixel_count"], 3)
        self.assertTrue(metadata["duplicate_bubble_inner_fill_applied"])
        self.assertEqual(metadata["duplicate_bubble_inner_fill_pixel_count"], 25)
        self.assertEqual(metadata["duplicate_bubble_inner_fill_backend"], "bubble_flat_fill")
        self.assertEqual(metadata["mask_quality_policy"], "lama_safe_bubble_and_text_free")
        self.assertEqual(metadata["mask_policy_bubble_clamp_applied_count"], 2)
        self.assertEqual(metadata["mask_policy_text_free_glyph_applied_count"], 1)
        self.assertEqual(metadata["mask_policy_removed_pixel_count"], 17)
        self.assertEqual(metadata["mask_policy_outside_bubble_removed_pixel_count"], 5)
        self.assertTrue(metadata["ctd_legacy_rectangle_rescue_disabled"])
        self.assertEqual(metadata["text_free_image_glyph_rescue_count"], 3)
        self.assertEqual(metadata["text_free_image_glyph_rescue_mask_pixel_count"], 41)
        self.assertEqual(metadata["mask_policy_version"], "ctd_lama_mask_policy_v2")
        self.assertEqual(metadata["mask_candidate_source"], "ctd_refined")
        self.assertEqual(metadata["mask_decision"], "review")
        self.assertEqual(metadata["mask_reject_reason"], "ambiguous_candidate_scores")
        self.assertEqual(metadata["mask_score_outside_change"], 0.01)
        self.assertEqual(metadata["mask_score_outline_damage"], 0.02)
        self.assertEqual(metadata["mask_score_residue"], 0.03)
        self.assertEqual(metadata["mask_score_color_delta"], 0.04)
        self.assertEqual(metadata["ui_panel_mode"], "preserve_original")
        self.assertEqual(metadata["ui_panel_preview_path"], "previews/page.png")
        self.assertEqual(metadata["blocks"][0]["ui_panel_mode"], "preserve_original")
        self.assertEqual(metadata["blocks"][0]["ui_panel_preview_path"], "previews/page_block_0.png")
        self.assertEqual(metadata["blocks"][0]["mask_decision"], "review")
        self.assertEqual(metadata["blocks"][0]["mask_reject_reason"], "embedded_ui_panel_layout_review")
        self.assertEqual(
            metadata["blocks"][0]["semantic_role"],
            "ui_or_sign",
        )
        self.assertEqual(
            metadata["blocks"][0]["processing_action"],
            "preserve",
        )
        self.assertEqual(
            metadata["blocks"][0]["canonical_block_id"],
            "canonical-1",
        )
        self.assertEqual(
            metadata["blocks"][0]["duplicate_alias_block_ids"],
            ["alias-1"],
        )
        self.assertEqual(
            metadata["blocks"][0]["mask_strategy"],
            "preserve_original",
        )
        self.assertEqual(
            metadata["blocks"][0]["mask_actual_bbox"],
            [1, 1, 4, 5],
        )
        self.assertEqual(
            metadata["blocks"][0]["mask_actual_pixel_count"],
            12,
        )

    def test_export_artifacts_only_writes_selected_debug_outputs(self) -> None:
        image = np.full((10, 12, 3), 255, dtype=np.uint8)
        raw_mask = np.zeros((10, 12), dtype=np.uint8)
        raw_mask[2:6, 3:7] = 255
        cleanup_delta = np.zeros((10, 12), dtype=np.uint8)
        cleanup_delta[6:8, 4:5] = 255
        block = _Block(xyxy=[3, 2, 7, 6], bubble_xyxy=[1, 1, 10, 9])

        with tempfile.TemporaryDirectory() as tmp_dir:
            export_inpaint_debug_artifacts(
                export_root=tmp_dir,
                archive_bname="",
                page_base_name="page",
                image=image,
                blocks=[block],
                export_settings={
                    "export_detector_overlay": True,
                    "export_raw_mask": True,
                    "export_mask_overlay": False,
                    "export_cleanup_mask_delta": False,
                    "export_debug_metadata": True,
                },
                raw_mask=raw_mask,
                cleanup_delta=cleanup_delta,
                metadata={"hello": "world"},
            )

            root = Path(tmp_dir)
            self.assertTrue((root / "detector_overlays" / "page_detector_overlay.png").exists())
            self.assertTrue((root / "raw_masks" / "page_raw_mask.png").exists())
            self.assertTrue((root / "debug_metadata" / "page_debug.json").exists())
            self.assertFalse((root / "mask_overlays" / "page_mask_overlay.png").exists())
            self.assertFalse((root / "cleanup_mask_delta" / "page_cleanup_delta.png").exists())
            payload = json.loads((root / "debug_metadata" / "page_debug.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["hello"], "world")

    def test_cleanup_delta_only_contains_new_pixels(self) -> None:
        processor = BatchProcessor.__new__(BatchProcessor)
        raw_mask = np.zeros((5, 5), dtype=np.uint8)
        raw_mask[1:3, 1:3] = 255
        final_mask = raw_mask.copy()
        final_mask[3:5, 2:4] = 255

        cleanup_delta = processor._build_cleanup_delta_mask(raw_mask, final_mask)

        expected = np.zeros((5, 5), dtype=np.uint8)
        expected[3:5, 2:4] = 255
        np.testing.assert_array_equal(cleanup_delta, expected)

    def test_sidecar_mode_uses_fixed_names_and_rejects_symlink_targets(
        self,
    ) -> None:
        image = np.full((8, 8, 3), 255, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        block = _Block(xyxy=[1, 1, 6, 6], bubble_xyxy=[0, 0, 7, 7])
        with tempfile.TemporaryDirectory() as tmp_dir:
            page_dir = Path(tmp_dir) / "page-0001_test"
            page_dir.mkdir()
            written = export_inpaint_debug_artifacts(
                export_root=tmp_dir,
                archive_bname="",
                page_base_name="ignored",
                image=image,
                blocks=[block],
                export_settings={
                    "export_raw_mask": True,
                    "export_debug_metadata": True,
                },
                raw_mask=mask,
                metadata={"safe": True},
                page_output_dir=str(page_dir),
            )
            self.assertEqual(
                Path(written["raw_mask"]).name,
                "inpaint-raw-mask.png",
            )
            self.assertEqual(
                Path(written["debug_metadata"]).name,
                "debug-metadata.json",
            )

            outside = Path(tmp_dir) / "outside.json"
            outside.write_text("unchanged", encoding="utf-8")
            target = page_dir / "debug-metadata.json"
            target.unlink()
            try:
                target.symlink_to(outside)
            except OSError:
                self.skipTest("symlink creation is unavailable")
            with self.assertRaises(DebugArtifactError):
                export_inpaint_debug_artifacts(
                    export_root=tmp_dir,
                    archive_bname="",
                    page_base_name="ignored",
                    image=image,
                    blocks=[block],
                    export_settings={"export_debug_metadata": True},
                    metadata={"safe": False},
                    page_output_dir=str(page_dir),
                )
            self.assertEqual(
                outside.read_text(encoding="utf-8"),
                "unchanged",
            )


if __name__ == "__main__":
    unittest.main()
