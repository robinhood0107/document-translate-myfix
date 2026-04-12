from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.masking.legacy_bbox_mask import (
    _build_legacy_base_block_mask,
    build_legacy_bbox_mask_details,
    merge_legacy_and_rescue,
)
from modules.masking.legacy_bbox_rescue import build_block_rescue_mask
from modules.utils.textblock import TextBlock


def _block(*, xyxy, text_class="text_bubble", bubble_xyxy=None, carrier_kind=None, carrier_mask_roi_xyxy=None) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
        carrier_kind=carrier_kind,
        carrier_mask_roi_xyxy=carrier_mask_roi_xyxy,
        text="demo",
    )


class LegacyBBoxMaskTests(unittest.TestCase):
    def test_hard_box_trigger_applies_when_low_fill_and_edges_exist(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        block = _block(xyxy=[20, 20, 36, 36], text_class="text_bubble", bubble_xyxy=[16, 16, 40, 40])
        legacy_block_mask = np.zeros((80, 80), dtype=np.uint8)

        def fake_feature_masks(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
            h, w = crop_rgb.shape[:2]
            center = np.zeros((h, w), dtype=np.uint8)
            cy1 = max(0, h // 2 - 2)
            cy2 = min(h, cy1 + 4)
            cx1 = max(0, w // 2 - 2)
            cx2 = min(w, cx1 + 4)
            center[cy1:cy2, cx1 + 1:cx2 - 1] = 255
            center[cy1 + 1:cy2 - 1, cx1:cx2] = 255
            return {
                "bright_core": center.copy(),
                "dark_fringe": np.zeros((h, w), dtype=np.uint8),
                "color_core": np.zeros((h, w), dtype=np.uint8),
                "outline_ring": center.copy(),
                "canny": np.full((h, w), 255, dtype=np.uint8),
            }

        with mock.patch("modules.masking.legacy_bbox_rescue._build_feature_masks", side_effect=fake_feature_masks):
            result = build_block_rescue_mask(image, block, legacy_block_mask)

        self.assertTrue(result["applied"])
        self.assertIn("low_legacy_fill_ratio", result["reason_codes"])
        self.assertIn("edge_dense", result["reason_codes"])
        self.assertGreater(int(np.count_nonzero(result["rescue_mask"])), 0)

    def test_hard_box_rejects_large_bbox(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        block = _block(xyxy=[0, 0, 40, 40], text_class="text_bubble", bubble_xyxy=[0, 0, 60, 60])
        legacy_block_mask = np.zeros((100, 100), dtype=np.uint8)

        result = build_block_rescue_mask(image, block, legacy_block_mask)

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_codes"], ["bbox_too_large"])

    def test_hard_box_rejects_dense_legacy_mask(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        block = _block(xyxy=[16, 16, 32, 32], text_class="text_free")
        legacy_block_mask = np.zeros((64, 64), dtype=np.uint8)
        legacy_block_mask[16:32, 16:32] = 255

        result = build_block_rescue_mask(image, block, legacy_block_mask)

        self.assertFalse(result["applied"])
        self.assertEqual(result["reason_codes"], ["legacy_mask_already_dense"])

    @mock.patch("modules.masking.legacy_bbox_mask.build_block_rescue_mask")
    @mock.patch("modules.masking.legacy_bbox_mask._build_legacy_base_block_mask")
    def test_bubble_final_mask_is_clamped_to_bubble_box(self, mock_base_mask, mock_rescue) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        block = _block(xyxy=[10, 10, 22, 22], text_class="text_bubble", bubble_xyxy=[12, 12, 18, 18])

        base_mask = np.zeros((40, 40), dtype=np.uint8)
        base_mask[8:24, 8:24] = 255
        rescue_mask = np.zeros((40, 40), dtype=np.uint8)
        rescue_mask[6:26, 6:26] = 255
        mock_base_mask.return_value = (base_mask, None)
        mock_rescue.return_value = {
            "applied": True,
            "reason_codes": ["low_legacy_fill_ratio", "edge_dense"],
            "legacy_fill_ratio": 0.0,
            "rescue_fill_ratio": 0.25,
            "rescue_mask": rescue_mask,
            "rescue_roi_xyxy": (6, 6, 26, 26),
            "metrics": {},
        }

        details = build_legacy_bbox_mask_details(image, [block])
        final_mask = details["final_mask"]

        self.assertEqual(int(np.count_nonzero(final_mask[:12, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, :12])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[18:, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, 18:])), 0)

    @mock.patch("modules.masking.legacy_bbox_mask.build_block_rescue_mask")
    @mock.patch("modules.masking.legacy_bbox_mask._build_legacy_base_block_mask")
    def test_text_free_final_mask_is_clamped_to_local_roi_and_merges_base_or_rescue(self, mock_base_mask, mock_rescue) -> None:
        image = np.zeros((40, 40, 3), dtype=np.uint8)
        block = _block(xyxy=[10, 10, 20, 20], text_class="text_free")

        base_mask = np.zeros((40, 40), dtype=np.uint8)
        base_mask[11:15, 11:15] = 255
        rescue_mask = np.zeros((40, 40), dtype=np.uint8)
        rescue_mask[15:18, 15:18] = 255
        mock_base_mask.return_value = (base_mask, None)
        mock_rescue.return_value = {
            "applied": True,
            "reason_codes": ["low_legacy_fill_ratio", "edge_dense"],
            "legacy_fill_ratio": 0.05,
            "rescue_fill_ratio": 0.10,
            "rescue_mask": rescue_mask,
            "rescue_roi_xyxy": (10, 10, 18, 18),
            "metrics": {},
        }

        details = build_legacy_bbox_mask_details(image, [block])
        final_mask = details["final_mask"]

        self.assertEqual(int(np.count_nonzero(final_mask[:10, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, :10])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[18:, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, 18:])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[11:15, 11:15])), 16)
        self.assertEqual(int(np.count_nonzero(final_mask[15:18, 15:18])), 9)
        self.assertEqual(int(np.count_nonzero(final_mask)), 25)

    def test_caption_plate_final_mask_is_clamped_to_local_roi(self) -> None:
        image_shape = (40, 40, 3)
        block = _block(
            xyxy=[10, 10, 20, 20],
            text_class="text_bubble",
            bubble_xyxy=[2, 2, 32, 32],
            carrier_kind="caption_plate",
            carrier_mask_roi_xyxy=[8, 8, 22, 22],
        )
        legacy_mask = np.zeros((40, 40), dtype=np.uint8)
        legacy_mask[4:28, 4:28] = 255
        rescue_mask = np.zeros((40, 40), dtype=np.uint8)
        rescue_mask[6:26, 6:26] = 255

        base_mask, rescue_mask, final_mask = merge_legacy_and_rescue(
            image_shape,
            block,
            legacy_mask,
            rescue_mask,
            (6, 6, 26, 26),
        )

        self.assertEqual(int(np.count_nonzero(base_mask[:8, :])), 0)
        self.assertEqual(int(np.count_nonzero(base_mask[:, :8])), 0)
        self.assertEqual(int(np.count_nonzero(base_mask[22:, :])), 0)
        self.assertEqual(int(np.count_nonzero(base_mask[:, 22:])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:8, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, :8])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[22:, :])), 0)
        self.assertEqual(int(np.count_nonzero(final_mask[:, 22:])), 0)

    @mock.patch("modules.masking.legacy_bbox_mask.get_inpaint_bboxes")
    def test_caption_plate_base_mask_uses_conservative_union_instead_of_blob(self, mock_get_inpaint_bboxes) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        block = _block(
            xyxy=[20, 20, 60, 60],
            text_class="text_bubble",
            bubble_xyxy=[10, 10, 70, 70],
            carrier_kind="caption_plate",
            carrier_mask_roi_xyxy=[16, 16, 64, 64],
        )
        mock_get_inpaint_bboxes.return_value = [
            [22, 24, 34, 32],
            [37, 24, 50, 32],
            [24, 38, 32, 46],
            [36, 38, 48, 46],
        ]

        block_mask, _ = _build_legacy_base_block_mask(
            image,
            block,
            roi_override=(16, 16, 64, 64),
        )

        roi_area = (64 - 16) * (64 - 16)
        fill_ratio = float(np.count_nonzero(block_mask[16:64, 16:64])) / float(roi_area)
        self.assertLess(fill_ratio, 0.25)
        self.assertGreater(int(np.count_nonzero(block_mask[24:32, 22:50])), 0)
        self.assertEqual(int(np.count_nonzero(block_mask[16:22, 16:64])), 0)
        self.assertEqual(int(np.count_nonzero(block_mask[48:64, 16:64])), 0)


if __name__ == "__main__":
    unittest.main()
