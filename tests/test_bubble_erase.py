from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.utils.bubble_erase import (
    BubbleEraseBlockStats,
    ERASE_MODE_BUBBLE_FLAT_FILL,
    ERASE_MODE_BUBBLE_GRADIENT_FILL,
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    ERASE_MODE_BUBBLE_SKIPPED,
    ERASE_MODE_BUBBLE_TELEA,
    build_bubble_residual_edit_mask,
    erase_text_bubble_regions,
    mask_pixel_count,
    set_block_erase_metadata,
    _bubble_interior_cap_mask,
)
from modules.utils.bubble_silhouette import extract_bubble_interior_cap_crop
from modules.utils.textblock import TextBlock


def _block(*, xyxy, bubble_xyxy=None, text_class="text_bubble") -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
    )


class BubbleEraseMetadataTests(unittest.TestCase):
    def test_set_block_erase_metadata_persists_debug_fields(self) -> None:
        class Block:
            pass

        block = Block()

        set_block_erase_metadata(
            block,
            BubbleEraseBlockStats(
                mode="bubble_flat_fill",
                edit_pixel_count=42,
                protect_pixel_count=7,
                skipped_reason="",
            ),
        )

        self.assertEqual(block._erase_mode, "bubble_flat_fill")
        self.assertEqual(block._erase_edit_pixel_count, 42)
        self.assertEqual(block._erase_protect_pixel_count, 7)
        self.assertEqual(block._erase_skipped_reason, "")

    def test_mask_pixel_count_counts_binary_pixels(self) -> None:
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255

        self.assertEqual(mask_pixel_count(mask), 4)

    def test_interior_cap_keeps_the_existing_erase_policy(self) -> None:
        crop = np.zeros((8, 8, 3), dtype=np.uint8)
        seed = np.zeros((8, 8), dtype=np.uint8)
        seed[3:5, 3:5] = 255
        detected = np.full((8, 8), 255, dtype=np.uint8)

        with mock.patch(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            return_value=detected,
        ) as extract:
            result = _bubble_interior_cap_mask(crop, seed)

        np.testing.assert_array_equal(result, detected)
        self.assertEqual(extract.call_args.kwargs["min_area_ratio"], 0.20)
        self.assertEqual(extract.call_args.kwargs["max_area_ratio"], 1.0)
        self.assertEqual(extract.call_args.kwargs["min_seed_coverage"], 0.0)
        self.assertFalse(extract.call_args.kwargs["preserve_seed_after_erode"])

    def test_silhouette_cap_accepts_only_area_and_seed_coverage_bounds(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        seed = np.zeros((10, 10), dtype=np.uint8)
        seed[4:6, 4:6] = 255
        accepted = np.zeros((10, 10), dtype=np.uint8)
        accepted[1:9, 1:9] = 255
        too_small = np.zeros((10, 10), dtype=np.uint8)
        too_small[4:6, 4:6] = 255
        misses_seed = np.zeros((10, 10), dtype=np.uint8)
        misses_seed[0:5, 0:5] = 255
        misses_seed[4:6, 4:6] = 0

        vendor_path = (
            "modules.source_parity_vendor.utils.textblock_mask.extract_ballon_mask"
        )
        with mock.patch(vendor_path, return_value=(accepted, None)):
            result = extract_bubble_interior_cap_crop(crop, seed)
        self.assertIsNotNone(result)
        self.assertTrue(np.all(result[seed > 0] == 255))

        for rejected in (too_small, np.full((10, 10), 255, dtype=np.uint8), misses_seed):
            with self.subTest(nonzero=int(np.count_nonzero(rejected))):
                with mock.patch(vendor_path, return_value=(rejected, None)):
                    self.assertIsNone(
                        extract_bubble_interior_cap_crop(crop, seed)
                    )

    def test_silhouette_cap_fails_closed_when_vendor_detection_raises(self) -> None:
        crop = np.zeros((10, 10, 3), dtype=np.uint8)
        seed = np.zeros((10, 10), dtype=np.uint8)
        seed[4:6, 4:6] = 255

        with mock.patch(
            "modules.source_parity_vendor.utils.textblock_mask.extract_ballon_mask",
            side_effect=RuntimeError("detector failed"),
        ):
            result = extract_bubble_interior_cap_crop(crop, seed)

        self.assertIsNone(result)


class BubbleResidualMaskTests(unittest.TestCase):
    def test_residual_mask_expands_text_like_pixels_near_seed(self) -> None:
        image = np.full((48, 48, 3), 120, dtype=np.uint8)
        image[20:24, 24:27] = 245
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 30, 28], bubble_xyxy=[8, 8, 40, 40])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(stats.edit_pixel_count, mask_pixel_count(source_mask))
        self.assertGreater(np.count_nonzero(edit_mask[20:24, 24:27]), 0)

    def test_residual_mask_captures_orphan_dark_glyphs_inside_text_prior(self) -> None:
        image = np.full((64, 64, 3), 135, dtype=np.uint8)
        image[28:38, 36:42] = 15
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 46, 44], bubble_xyxy=[8, 8, 56, 56])

        edit_mask, stats = build_bubble_residual_edit_mask(
            image,
            source_mask,
            block,
            seed_dilate_px=1,
        )

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(np.count_nonzero(edit_mask[28:38, 36:42]), 0)

    def test_residual_mask_captures_orphan_bright_glyphs_inside_text_prior(self) -> None:
        image = np.full((64, 64, 3), 135, dtype=np.uint8)
        image[28:38, 36:42] = 245
        source_mask = np.zeros((64, 64), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 46, 44], bubble_xyxy=[8, 8, 56, 56])

        edit_mask, stats = build_bubble_residual_edit_mask(
            image,
            source_mask,
            block,
            seed_dilate_px=1,
        )

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertGreater(np.count_nonzero(edit_mask[28:38, 36:42]), 0)

    def test_residual_mask_does_not_copy_boxy_source_seed_wholesale(self) -> None:
        image = np.full((72, 72, 3), 142, dtype=np.uint8)
        image[28:46, 30:35] = 20
        image[28:46, 42:47] = 245
        source_mask = np.zeros((72, 72), dtype=np.uint8)
        source_mask[24:50, 24:54] = 255
        block = _block(xyxy=[22, 20, 56, 54], bubble_xyxy=[12, 12, 62, 62])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertLess(mask_pixel_count(edit_mask), mask_pixel_count(source_mask))
        self.assertGreater(np.count_nonzero(edit_mask[28:46, 30:47]), 0)

    def test_residual_mask_rejects_long_rule_like_components(self) -> None:
        image = np.full((48, 48, 3), 120, dtype=np.uint8)
        image[22:24, 10:38] = 245
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[8, 18, 40, 28], bubble_xyxy=[4, 4, 44, 44])

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertEqual(np.count_nonzero(edit_mask[22:24, 32:38]), 0)

    def test_non_bubble_blocks_are_skipped(self) -> None:
        image = np.full((32, 32, 3), 120, dtype=np.uint8)
        source_mask = np.zeros((32, 32), dtype=np.uint8)
        source_mask[10:14, 10:14] = 255
        block = _block(xyxy=[8, 8, 16, 16], text_class="text_free")

        edit_mask, stats = build_bubble_residual_edit_mask(image, source_mask, block)

        self.assertEqual(stats.mode, ERASE_MODE_BUBBLE_SKIPPED)
        self.assertEqual(stats.skipped_reason, "not_text_bubble")
        self.assertEqual(mask_pixel_count(edit_mask), 0)


class BubbleFillBackendTests(unittest.TestCase):
    def test_flat_bubble_fill_changes_only_edit_mask(self) -> None:
        original = np.full((48, 48, 3), 128, dtype=np.uint8)
        original[20:24, 20:24] = 245
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((48, 48), dtype=np.uint8)
        source_mask[20:24, 20:24] = 255
        block = _block(xyxy=[18, 18, 28, 28], bubble_xyxy=[8, 8, 40, 40])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_FLAT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertLess(int(np.mean(result.image[20:24, 20:24])), 180)

    def test_complex_bubble_uses_telea_and_preserves_outside_mask(self) -> None:
        original = np.full((56, 56, 3), 128, dtype=np.uint8)
        for x in range(8, 48, 4):
            original[8:48, x:x + 2] = 80
        original[24:28, 24:28] = 245
        current = original.copy()
        current[1, 1] = [9, 10, 11]
        source_mask = np.zeros((56, 56), dtype=np.uint8)
        source_mask[24:28, 24:28] = 255
        block = _block(xyxy=[20, 20, 34, 34], bubble_xyxy=[8, 8, 48, 48])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_TELEA)
        self.assertTrue(np.all(result.image[1, 1] == current[1, 1]))

    def test_smooth_bubble_roi_prefers_flat_fill_for_white_text_ghosts(self) -> None:
        original = np.full((72, 72, 3), 142, dtype=np.uint8)
        original[24:42, 28:34] = 248
        original[28:46, 40:46] = 248
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((72, 72), dtype=np.uint8)
        source_mask[24:42, 28:34] = 255
        source_mask[28:46, 40:46] = 255
        block = _block(xyxy=[22, 20, 50, 50], bubble_xyxy=[12, 12, 60, 60])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_FLAT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertLess(abs(int(np.mean(result.image[24:46, 28:46])) - 142), 4)

    def test_smooth_gradient_bubble_uses_gradient_fill_for_boxy_source_masks(self) -> None:
        gradient = np.tile(np.linspace(120, 170, 80, dtype=np.uint8), (80, 1))
        expected_background = np.repeat(gradient[:, :, None], 3, axis=2)
        original = expected_background.copy()
        original[28:48, 30:36] = 248
        original[28:48, 46:52] = 248
        current = original.copy()
        current[0, 0] = [5, 6, 7]
        source_mask = np.zeros((80, 80), dtype=np.uint8)
        source_mask[24:52, 24:58] = 255
        block = _block(xyxy=[22, 20, 60, 56], bubble_xyxy=[10, 10, 70, 70])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(result.stats["changed_outside_edit_mask_pixel_count"], 0)
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_GRADIENT_FILL)
        self.assertTrue(np.all(result.image[0, 0] == current[0, 0]))
        self.assertLess(mask_pixel_count(result.edit_mask), mask_pixel_count(source_mask))
        self.assertLess(
            abs(int(np.mean(result.image[28:48, 30:52])) - int(np.mean(expected_background[28:48, 30:52]))),
            12,
        )

    def test_line_art_bubble_defers_to_lama_fallback_without_flattening(self) -> None:
        original = np.full((96, 96, 3), 150, dtype=np.uint8)
        original[46:49, 8:88] = 20
        original[12:84, 68:71] = 30
        original[30:52, 32:38] = 245
        original[30:52, 48:54] = 245
        current = original.copy()
        source_mask = np.zeros((96, 96), dtype=np.uint8)
        source_mask[30:52, 32:38] = 255
        source_mask[30:52, 48:54] = 255
        block = _block(xyxy=[26, 24, 60, 58], bubble_xyxy=[8, 8, 88, 88])

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertTrue(result.stats["applied"])
        self.assertEqual(block._erase_mode, ERASE_MODE_BUBBLE_LAMA_FALLBACK)
        self.assertEqual(result.stats["fallback_block_count"], 1)
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertGreater(mask_pixel_count(result.fallback_mask), 0)
        self.assertTrue(np.array_equal(result.image, current))

    def test_text_free_blocks_are_not_modified_by_bubble_erase(self) -> None:
        original = np.full((32, 32, 3), 128, dtype=np.uint8)
        current = original.copy()
        current[10:14, 10:14] = 64
        source_mask = np.zeros((32, 32), dtype=np.uint8)
        source_mask[10:14, 10:14] = 255
        block = _block(xyxy=[8, 8, 16, 16], text_class="text_free")

        result = erase_text_bubble_regions(original, current, source_mask, [block])

        self.assertFalse(result.stats["applied"])
        self.assertEqual(mask_pixel_count(result.edit_mask), 0)
        self.assertTrue(np.array_equal(result.image, current))
        self.assertEqual(block._erase_mode, "text_free_lama")


if __name__ == "__main__":
    unittest.main()
