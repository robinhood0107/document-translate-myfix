from __future__ import annotations

import unittest

import numpy as np

from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
    normalize_edit_mask,
)


class InpaintCompositeTests(unittest.TestCase):
    def test_composite_restores_pixels_outside_edit_mask(self) -> None:
        original = np.zeros((6, 6, 3), dtype=np.uint8)
        original[:, :] = [10, 20, 30]
        edited = np.full_like(original, 200)
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[2:4, 2:4] = 255

        composited = composite_with_edit_mask(original, edited, mask)

        np.testing.assert_array_equal(composited[mask <= 0], original[mask <= 0])
        np.testing.assert_array_equal(composited[mask > 0], edited[mask > 0])
        self.assertEqual(count_changed_outside_edit_mask(original, composited, mask), 0)

    def test_normalize_edit_mask_crops_oversized_masks(self) -> None:
        mask = np.ones((8, 8), dtype=np.uint8) * 255

        normalized = normalize_edit_mask(mask, (4, 6, 3))

        self.assertEqual(normalized.shape, (4, 6))
        self.assertEqual(int(np.count_nonzero(normalized)), 24)

    def test_changed_outside_mask_counts_only_unprotected_pixels(self) -> None:
        original = np.zeros((5, 5), dtype=np.uint8)
        edited = original.copy()
        edited[1, 1] = 5
        edited[4, 4] = 5
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 1] = 255

        self.assertEqual(count_changed_outside_edit_mask(original, edited, mask), 1)


if __name__ == "__main__":
    unittest.main()
