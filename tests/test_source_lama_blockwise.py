from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from modules.inpainting.source_lama_blockwise import (
    _resolve_caption_plate_erase_mask,
    _try_caption_plate_local_inpaint,
)


class CaptionPlateLocalInpaintTests(unittest.TestCase):
    def test_caption_plate_erase_mask_expands_beyond_text_mask(self) -> None:
        image = np.full((120, 120, 3), 210, dtype=np.uint8)
        image[15:105, 15:105] = np.array([125, 90, 72], dtype=np.uint8)
        image[15:18, 15:105] = 240
        image[102:105, 15:105] = 240
        image[15:105, 15:18] = 240
        image[15:105, 102:105] = 240
        image[36:42, 28:92] = 255
        image[56:62, 28:92] = 255

        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[36:42, 28:92] = 255
        mask[56:62, 28:92] = 255

        block = SimpleNamespace(
            carrier_kind="caption_plate",
            carrier_mask_roi_xyxy=[20, 20, 100, 100],
        )

        result = _resolve_caption_plate_erase_mask(image, mask, block)

        self.assertIsNotNone(result)
        result = np.asarray(result)
        self.assertGreater(int(np.count_nonzero(result)), int(np.count_nonzero(mask)) * 2)
        self.assertEqual(int(result[60, 60]), 255)

    def test_caption_plate_local_inpaint_replaces_bright_text_with_local_plate_colors(self) -> None:
        image = np.zeros((80, 80, 3), dtype=np.uint8)
        image[:, :] = np.array([150, 105, 88], dtype=np.uint8)
        image[20:24, 18:58] = 255
        image[32:36, 18:58] = 255

        mask = np.zeros((80, 80), dtype=np.uint8)
        mask[20:24, 18:58] = 255
        mask[32:36, 18:58] = 255

        block = SimpleNamespace(
            carrier_kind="caption_plate",
            carrier_mask_roi_xyxy=[10, 10, 70, 70],
        )

        result = _try_caption_plate_local_inpaint(image, mask, block, [0, 0, 80, 80])

        self.assertIsNotNone(result)
        result = np.asarray(result)
        self.assertLess(float(np.mean(result[20:24, 18:58])), 220.0)
        self.assertLess(float(np.mean(result[32:36, 18:58])), 220.0)
        self.assertGreater(float(np.mean(result[20:24, 18:58])), 90.0)

    def test_non_caption_plate_returns_none(self) -> None:
        image = np.full((24, 24, 3), 128, dtype=np.uint8)
        mask = np.zeros((24, 24), dtype=np.uint8)
        mask[8:12, 8:12] = 255
        block = SimpleNamespace(
            carrier_kind="speech_bubble",
            carrier_mask_roi_xyxy=[4, 4, 20, 20],
        )

        self.assertIsNone(_resolve_caption_plate_erase_mask(image, mask, block))
        self.assertIsNone(_try_caption_plate_local_inpaint(image, mask, block, [0, 0, 24, 24]))


if __name__ == "__main__":
    unittest.main()
