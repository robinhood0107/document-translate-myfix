from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint
from modules.utils.textblock import TextBlock


def _block(*, text_class: str) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray([10, 10, 30, 30], dtype=np.int32),
        text_class=text_class,
        text="demo",
        translation="demo",
    )


class InpaintCleanupTests(unittest.TestCase):
    def test_text_free_edge_touching_residue_is_skipped(self) -> None:
        image = np.full((50, 50, 3), 128, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        block = _block(text_class="text_free")
        prior = np.full((20, 20), 255, dtype=np.uint8)
        inpainter = mock.Mock(return_value=image.copy())

        with (
            mock.patch("modules.utils.inpaint_cleanup.build_text_prior_mask", return_value=prior),
            mock.patch("modules.utils.inpaint_cleanup.detect_content_in_bbox", return_value=np.asarray([[0, 5, 8, 15]])),
        ):
            refined, merged, stats = refine_bubble_residue_inpaint(
                image,
                mask,
                [block],
                inpainter,
                {},
            )

        inpainter.assert_not_called()
        self.assertFalse(stats["applied"])
        np.testing.assert_array_equal(refined, image)
        np.testing.assert_array_equal(merged, mask)


if __name__ == "__main__":
    unittest.main()
