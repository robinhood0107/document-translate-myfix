from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.masking.ctd_refiner import CTDRefiner, _filter_candidate_mask
from modules.utils.inpaint_cleanup import _cap_residue_mask_to_source_mask, refine_bubble_residue_inpaint
from modules.utils.mask_roi import resolve_block_ctd_roi
from modules.utils.textblock import TextBlock


def _block(*, xyxy, text_class="text_free") -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=None,
        text_class=text_class,
        text="demo",
    )


class TextFreeStrictMaskingTests(unittest.TestCase):
    def test_text_free_ctd_roi_uses_modest_erase_envelope(self) -> None:
        block = _block(xyxy=[40, 20, 50, 70])

        roi = resolve_block_ctd_roi(block, (100, 100, 3))

        self.assertEqual(roi, (38, 13, 52, 77))

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


if __name__ == "__main__":
    unittest.main()
