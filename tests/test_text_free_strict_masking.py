from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.masking.ctd_refiner import (
    CTDRefiner,
    _filter_candidate_mask,
    _text_bubble_polarity_glyph_mask,
    _text_free_glyph_color_mask,
)
from modules.utils.inpaint_cleanup import (
    _cap_residue_mask_to_source_mask,
    apply_duplicate_bubble_inner_fill,
    fill_duplicate_bubble_inner_regions,
    refine_bubble_residue_inpaint,
)
from modules.utils.image_utils import generate_mask
from modules.utils.mask_roi import resolve_block_ctd_roi
from modules.inpainting.source_lama_blockwise import _split_bubble_source_mask
from modules.utils.textblock import TextBlock


def _block(*, xyxy, text_class="text_free") -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=None,
        text_class=text_class,
        text="demo",
    )


def _bubble_block(*, xyxy, bubble_xyxy) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32),
        text_class="text_bubble",
        text="demo",
    )


class TextFreeStrictMaskingTests(unittest.TestCase):
    def test_text_free_ctd_roi_uses_modest_erase_envelope(self) -> None:
        block = _block(xyxy=[40, 20, 50, 70])

        roi = resolve_block_ctd_roi(block, (100, 100, 3))

        self.assertEqual(roi, (38, 13, 52, 77))

    def test_structure_protect_bubble_uses_tight_text_roi(self) -> None:
        block = _bubble_block(
            xyxy=[40, 40, 55, 55],
            bubble_xyxy=[10, 10, 90, 90],
        )
        block.mask_strategy = "glyph_only_structure_protect"

        roi = resolve_block_ctd_roi(block, (100, 100, 3))

        self.assertEqual(roi, (36, 36, 59, 59))

    def test_regular_glyph_only_keeps_existing_text_free_envelope(self) -> None:
        block = _block(xyxy=[40, 20, 50, 70])
        block.mask_strategy = "glyph_only"

        roi = resolve_block_ctd_roi(block, (100, 100, 3))

        self.assertEqual(roi, (38, 13, 52, 77))

    def test_structure_protect_bubble_uses_thin_final_dilation(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        block = _bubble_block(
            xyxy=[40, 40, 55, 55],
            bubble_xyxy=[10, 10, 90, 90],
        )
        block.mask_strategy = "glyph_only_structure_protect"
        ctd_mask = np.zeros((100, 100), dtype=np.uint8)
        ctd_mask[47, 47] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros((100, 100), dtype=np.uint8),
            ),
        ):
            refiner_cls.return_value.refine.return_value = mock.Mock(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                    "text_free_final_mask_dilate_size": 1,
                },
                return_details=True,
            )

        self.assertEqual(int(details["final_mask"][47, 48]), 255)
        self.assertEqual(int(details["final_mask"][47, 55]), 0)

    def test_protected_bubble_hard_excludes_neighboring_final_mask(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        inpaint_block = _bubble_block(
            xyxy=[40, 40, 55, 55],
            bubble_xyxy=[10, 10, 90, 90],
        )
        protected = _bubble_block(
            xyxy=[42, 42, 58, 58],
            bubble_xyxy=[10, 10, 90, 90],
        )
        protected.processing_action = "review"
        protected.mask_strategy = "preserve_original"
        protected._inpaint_protected_reason = "bubble_panel_text_candidate"
        ctd_mask = np.zeros((100, 100), dtype=np.uint8)
        ctd_mask[47, 47] = 255

        with (
            mock.patch("modules.utils.image_utils.CTDRefiner") as refiner_cls,
            mock.patch(
                "modules.utils.image_utils.build_protect_mask",
                return_value=np.zeros((100, 100), dtype=np.uint8),
            ),
        ):
            refiner_cls.return_value.refine.return_value = mock.Mock(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )
            details = generate_mask(
                image,
                [inpaint_block],
                settings={
                    "mask_refiner": "ctd",
                    "keep_existing_lines": False,
                    "final_mask_dilate_size": 8,
                    "text_free_final_mask_dilate_size": 1,
                },
                protected_blocks=[protected],
                return_details=True,
            )

        self.assertEqual(int(np.count_nonzero(details["final_mask"])), 0)
        self.assertEqual(
            int(details["mask_policy_protected_region_block_count"]),
            1,
        )
        self.assertGreater(
            int(details["mask_policy_protected_region_removed_pixel_count"]),
            0,
        )

    def test_residue_cleanup_skips_structure_protect_bubble(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        block = _bubble_block(
            xyxy=[12, 12, 20, 20],
            bubble_xyxy=[4, 4, 28, 28],
        )
        block.mask_strategy = "glyph_only_structure_protect"

        def fail_inpainter(_image, _mask, _config):
            raise AssertionError("structure-protect cleanup must not invoke a broad pass2")

        cleaned, merged_mask, stats = refine_bubble_residue_inpaint(
            image,
            mask,
            [block],
            fail_inpainter,
            object(),
        )

        self.assertIs(cleaned, image)
        self.assertIs(merged_mask, mask)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["pass2_glyph_only_skipped_count"], 1)

    def test_structure_protect_bubble_bypasses_broad_bubble_erase_route(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        protected = _bubble_block(
            xyxy=[12, 12, 20, 20],
            bubble_xyxy=[4, 4, 28, 28],
        )
        protected.mask_strategy = "glyph_only_structure_protect"
        ordinary = _bubble_block(
            xyxy=[12, 12, 20, 20],
            bubble_xyxy=[4, 4, 28, 28],
        )

        protected_mask, protected_bubbles, protected_lama = (
            _split_bubble_source_mask(mask, [protected], image.shape)
        )
        ordinary_mask, ordinary_bubbles, ordinary_lama = (
            _split_bubble_source_mask(mask, [ordinary], image.shape)
        )

        self.assertEqual(int(np.count_nonzero(protected_mask)), 0)
        self.assertEqual(protected_bubbles, [])
        self.assertEqual(protected_lama, [protected])
        self.assertGreater(int(np.count_nonzero(ordinary_mask)), 0)
        self.assertEqual(ordinary_bubbles, [ordinary])
        self.assertEqual(ordinary_lama, [])

    def test_shared_bubble_with_structure_protect_bypasses_broad_erase_for_all_blocks(self) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[12:20, 12:20] = 255
        mask[21:28, 12:20] = 255
        protected = _bubble_block(
            xyxy=[12, 12, 20, 20],
            bubble_xyxy=[4, 4, 32, 32],
        )
        protected.mask_strategy = "glyph_only_structure_protect"
        ordinary = _bubble_block(
            xyxy=[12, 21, 20, 28],
            bubble_xyxy=[4, 4, 32, 32],
        )

        bubble_mask, bubble_blocks, lama_blocks = _split_bubble_source_mask(
            mask,
            [protected, ordinary],
            image.shape,
        )

        self.assertEqual(int(np.count_nonzero(bubble_mask)), 0)
        self.assertEqual(bubble_blocks, [])
        self.assertEqual(lama_blocks, [protected, ordinary])

    def test_text_free_filter_keeps_large_edge_touching_glyph_component(self) -> None:
        candidate = np.zeros((50, 50), dtype=np.uint8)
        candidate[5:45, 30:50] = 255
        prior = candidate.copy()

        text_free = _filter_candidate_mask(candidate, prior, "text_free")
        text_bubble = _filter_candidate_mask(candidate, prior, "text_bubble")

        self.assertEqual(int(np.count_nonzero(text_free)), int(np.count_nonzero(candidate)))
        self.assertEqual(int(np.count_nonzero(text_bubble)), 0)

    def test_text_free_filter_rejects_obvious_rule_line_component(self) -> None:
        candidate = np.zeros((50, 50), dtype=np.uint8)
        candidate[20:23, 0:50] = 255
        prior = candidate.copy()

        result = _filter_candidate_mask(candidate, prior, "text_free")

        self.assertEqual(int(np.count_nonzero(result)), 0)

    def test_text_free_glyph_color_mask_adds_bright_fill_inside_prior(self) -> None:
        image = np.full((40, 40, 3), 80, dtype=np.uint8)
        image[12:24, 14:28] = [245, 245, 238]
        prior = np.zeros((40, 40), dtype=np.uint8)
        prior[8:30, 8:32] = 255
        anchor = np.zeros_like(prior)
        anchor[12:24, 14:17] = 255

        result = _text_free_glyph_color_mask(image, prior, anchor)

        self.assertEqual(int(result[18, 22]), 255)
        self.assertGreater(int(np.count_nonzero(result)), int(np.count_nonzero(anchor)))

    def test_text_free_glyph_color_mask_rejects_large_edge_background(self) -> None:
        image = np.full((40, 40, 3), 245, dtype=np.uint8)
        prior = np.full((40, 40), 255, dtype=np.uint8)
        anchor = np.zeros_like(prior)

        result = _text_free_glyph_color_mask(image, prior, anchor)

        self.assertEqual(int(np.count_nonzero(result)), 0)

    def test_text_free_glyph_color_mask_limits_warm_shadow_to_anchor_area(self) -> None:
        image = np.full((50, 50, 3), 70, dtype=np.uint8)
        image[16:26, 16:26] = [245, 245, 238]
        image[16:26, 27:37] = [190, 155, 88]
        image[38:44, 38:44] = [190, 155, 88]
        prior = np.full((50, 50), 255, dtype=np.uint8)
        anchor = np.zeros((50, 50), dtype=np.uint8)
        anchor[16:26, 16:26] = 255

        result = _text_free_glyph_color_mask(image, prior, anchor)

        self.assertEqual(int(result[20, 31]), 255)
        self.assertEqual(int(result[41, 41]), 0)

    def test_text_bubble_polarity_mask_adds_dark_glyph_and_white_outline(self) -> None:
        image = np.full((70, 60, 3), 118, dtype=np.uint8)
        image[8:10, 8:10] = [245, 245, 245]
        image[18:52, 24:42] = [245, 245, 245]
        image[21:49, 27:39] = [12, 12, 12]
        search = np.zeros((70, 60), dtype=np.uint8)
        search[14:56, 18:48] = 255

        result = _text_bubble_polarity_glyph_mask(image, search)

        self.assertEqual(int(result[30, 32]), 255)
        self.assertEqual(int(result[20, 27]), 255)
        self.assertEqual(int(result[8, 8]), 0)

    def test_text_bubble_polarity_mask_rejects_bright_bubble_background(self) -> None:
        image = np.full((64, 64, 3), 244, dtype=np.uint8)
        search = np.full((64, 64), 255, dtype=np.uint8)

        result = _text_bubble_polarity_glyph_mask(image, search)

        self.assertEqual(int(np.count_nonzero(result)), 0)

    def test_text_bubble_refiner_merges_polarity_mask_outside_sparse_prior(self) -> None:
        image = np.full((80, 80, 3), 122, dtype=np.uint8)
        image[28:54, 31:47] = [246, 246, 246]
        image[31:51, 34:44] = [14, 14, 14]
        block = _bubble_block(xyxy=[26, 24, 51, 58], bubble_xyxy=[8, 8, 72, 72])
        block.inpaint_bboxes = [[12, 12, 16, 16]]
        refiner = CTDRefiner()

        def fake_raw(crop: np.ndarray) -> np.ndarray:
            raw = np.zeros(crop.shape[:2], dtype=np.uint8)
            raw[4:8, 4:8] = 255
            return raw

        with (
            mock.patch.object(refiner, "_infer_raw_mask", side_effect=fake_raw),
            mock.patch("modules.masking.ctd_refiner._expand_final_mask_crop", side_effect=lambda mask, _text_class: mask),
        ):
            result = refiner.refine(image, [block])

        self.assertEqual(int(result.final_mask[40, 38]), 255)
        self.assertEqual(int(result.final_mask[29, 32]), 255)

    def test_text_free_refiner_does_not_merge_refined_roi_fill(self) -> None:
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        block = _block(xyxy=[10, 10, 18, 24])
        refiner = CTDRefiner()

        def fake_raw(crop: np.ndarray) -> np.ndarray:
            raw = np.zeros(crop.shape[:2], dtype=np.uint8)
            raw[5:9, 5:9] = 255
            return raw

        with (
            mock.patch.object(refiner, "_infer_raw_mask", side_effect=fake_raw),
            mock.patch("modules.masking.ctd_refiner._refine_mask_roi", return_value=np.full((20, 14), 255, dtype=np.uint8)),
            mock.patch("modules.masking.ctd_refiner._expand_final_mask_crop", side_effect=lambda mask, _text_class: mask),
        ):
            result = refiner.refine(image, [block])

        self.assertEqual(int(np.count_nonzero(result.refined_mask)), 0)
        self.assertEqual(int(np.count_nonzero(result.final_mask)), 16)

    def test_residue_mask_is_capped_to_source_mask_dilation(self) -> None:
        source_mask = np.zeros((20, 20), dtype=np.uint8)
        source_mask[10, 10] = 255
        residue_mask = np.zeros_like(source_mask)
        residue_mask[9:12, 9:12] = 255
        residue_mask[0:5, 0:5] = 255

        capped = _cap_residue_mask_to_source_mask(residue_mask, source_mask, dilate_px=2)

        self.assertEqual(int(np.count_nonzero(capped[0:5, 0:5])), 0)
        self.assertEqual(int(np.count_nonzero(capped[9:12, 9:12])), 9)

    def test_duplicate_bubble_inner_fill_only_changes_duplicate_mask(self) -> None:
        image = np.full((72, 72, 3), 224, dtype=np.uint8)
        image[24:48, 24:28] = 20
        image[24:48, 34:38] = 20
        image[24:48, 44:48] = 20
        duplicate_mask = np.zeros((72, 72), dtype=np.uint8)
        duplicate_mask[20:52, 20:52] = 255

        filled, stats = fill_duplicate_bubble_inner_regions(image, duplicate_mask)

        self.assertTrue(stats["applied"])
        self.assertEqual(stats["pass_name"], "duplicate_bubble_inner_fill")
        self.assertGreater(stats["duplicate_bubble_inner_fill_pixel_count"], 0)
        self.assertFalse(np.array_equal(filled[24:48, 24:48], image[24:48, 24:48]))
        self.assertEqual(int(np.count_nonzero(filled[duplicate_mask == 0] != image[duplicate_mask == 0])), 0)
        self.assertGreater(float(np.mean(filled[24:48, 24:48])), 180.0)

    def test_duplicate_bubble_inner_fill_merges_mask_and_cleanup_stats(self) -> None:
        image = np.full((48, 48, 3), 230, dtype=np.uint8)
        image[18:30, 20:28] = 15
        base_mask = np.zeros((48, 48), dtype=np.uint8)
        base_mask[4:8, 4:8] = 255
        duplicate_mask = np.zeros((48, 48), dtype=np.uint8)
        duplicate_mask[16:32, 16:32] = 255
        cleanup_stats = {"applied": False, "component_count": 0, "block_count": 0}

        filled, merged_mask, merged_stats = apply_duplicate_bubble_inner_fill(
            image,
            base_mask,
            {"duplicate_bubble_inner_mask": duplicate_mask},
            cleanup_stats,
        )

        self.assertTrue(merged_stats["duplicate_bubble_inner_fill"]["applied"])
        self.assertEqual(int(np.count_nonzero(merged_mask[4:8, 4:8])), 16)
        self.assertEqual(int(np.count_nonzero(merged_mask[16:32, 16:32])), 256)
        self.assertEqual(int(np.count_nonzero(filled[duplicate_mask == 0] != image[duplicate_mask == 0])), 0)
        self.assertIsNot(merged_stats, cleanup_stats)

    def test_residue_cleanup_skips_text_free_blocks(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[6:12, 6:12] = 255
        block = _block(xyxy=[6, 6, 12, 12])

        def fail_inpainter(_image, _mask, _config):
            raise AssertionError("text-free cleanup must not invoke pass2 inpainting")

        cleaned, merged_mask, stats = refine_bubble_residue_inpaint(
            image,
            mask,
            [block],
            fail_inpainter,
            object(),
        )

        self.assertIs(cleaned, image)
        self.assertIs(merged_mask, mask)
        self.assertFalse(stats["applied"])
        self.assertEqual(stats["pass2_text_free_candidate_count"], 0)

    def test_residue_cleanup_does_not_count_text_free_in_mixed_pages(self) -> None:
        image = np.zeros((24, 24, 3), dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        mask[14:20, 14:20] = 255
        bubble = _bubble_block(xyxy=[2, 2, 8, 8], bubble_xyxy=[0, 0, 10, 10])
        text_free = _block(xyxy=[14, 14, 20, 20])

        def fake_inpainter(input_image, input_mask, _config):
            raise AssertionError("bubble cleanup must use the safe fill backend, not LaMa pass2")

        with mock.patch(
            "modules.utils.inpaint_cleanup.detect_content_in_bbox",
            return_value=[(2, 2, 4, 4)],
        ):
            cleaned, merged_mask, stats = refine_bubble_residue_inpaint(
                image,
                mask,
                [bubble, text_free],
                fake_inpainter,
                object(),
            )

        self.assertIsNot(cleaned, image)
        self.assertIsNot(merged_mask, mask)
        self.assertTrue(stats["applied"])
        self.assertEqual(stats["pass2_backend"], "bubble_flat_fill")
        self.assertGreater(stats["pass2_bubble_candidate_count"], 0)
        self.assertEqual(stats["pass2_text_free_candidate_count"], 0)
        self.assertEqual(stats["pass2_text_free_kept_count"], 0)


if __name__ == "__main__":
    unittest.main()
