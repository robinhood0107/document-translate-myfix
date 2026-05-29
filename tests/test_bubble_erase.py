from __future__ import annotations

import unittest

import numpy as np

from modules.utils.bubble_erase import (
    BubbleEraseBlockStats,
    ERASE_MODE_BUBBLE_FLAT_FILL,
    ERASE_MODE_BUBBLE_SKIPPED,
    ERASE_MODE_BUBBLE_TELEA,
    build_bubble_residual_edit_mask,
    erase_text_bubble_regions,
    mask_pixel_count,
    set_block_erase_metadata,
)
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
