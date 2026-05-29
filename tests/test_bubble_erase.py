from __future__ import annotations

import unittest

import numpy as np

from modules.utils.bubble_erase import (
    BubbleEraseBlockStats,
    mask_pixel_count,
    set_block_erase_metadata,
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


if __name__ == "__main__":
    unittest.main()
