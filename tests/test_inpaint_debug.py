from __future__ import annotations

import json
import importlib.util
import sys
import tempfile
import unittest
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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


class InpaintDebugTests(unittest.TestCase):
    def test_export_inpaint_debug_script_imports_blockwise_lama_runner(self) -> None:
        script_path = Path(__file__).resolve().parents[1] / "scripts" / "export_inpaint_debug.py"
        spec = importlib.util.spec_from_file_location("export_inpaint_debug_for_test", script_path)
        self.assertIsNotNone(spec)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)

        self.assertTrue(callable(getattr(module, "source_lama_blockwise_inpaint", None)))

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
