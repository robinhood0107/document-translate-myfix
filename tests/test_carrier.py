from __future__ import annotations

import unittest

import numpy as np

from modules.utils.carrier import (
    CARRIER_KIND_CAPTION_PLATE,
    CARRIER_KIND_SPEECH_BUBBLE,
    annotate_text_block_carriers,
    classify_text_block_carrier,
    resolve_effective_text_bubble_roi,
)
from modules.utils.mask_roi import resolve_block_cleanup_roi, resolve_block_ctd_roi
from modules.utils.textblock import TextBlock


def _block(*, xyxy, bubble_xyxy=None, text_class="text_bubble") -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
        text="demo",
    )


def _white_bubble_image() -> np.ndarray:
    image = np.full((96, 96, 3), 235, dtype=np.uint8)
    image[12:84, 12:84] = 252
    image[12:15, 12:84] = 0
    image[81:84, 12:84] = 0
    image[12:84, 12:15] = 0
    image[12:84, 81:84] = 0
    return image


def _caption_plate_image(*, color: tuple[int, int, int], with_gradient: bool = True) -> np.ndarray:
    image = np.full((96, 96, 3), 90, dtype=np.uint8)
    image[12:84, 12:84] = np.asarray(color, dtype=np.uint8)
    if with_gradient:
        gradient = np.linspace(-18, 18, 72, dtype=np.int16)[None, :, None]
        plate = image[12:84, 12:84].astype(np.int16)
        image[12:84, 12:84] = np.clip(plate + gradient, 0, 255).astype(np.uint8)
    return image


class CarrierClassificationTests(unittest.TestCase):
    def test_white_bubble_stays_speech_bubble(self) -> None:
        image = _white_bubble_image()
        block = _block(xyxy=[34, 28, 62, 68], bubble_xyxy=[12, 12, 84, 84])

        carrier_kind, metrics, carrier_roi = classify_text_block_carrier(image, block)

        self.assertEqual(carrier_kind, CARRIER_KIND_SPEECH_BUBBLE)
        self.assertIsNone(carrier_roi)
        self.assertGreater(metrics["white_ratio"], 0.80)

    def test_colored_caption_plate_is_detected(self) -> None:
        for color in ((150, 105, 88), (166, 122, 144), (140, 140, 140)):
            with self.subTest(color=color):
                image = _caption_plate_image(color=color)
                block = _block(xyxy=[34, 28, 62, 68], bubble_xyxy=[12, 12, 84, 84])

                carrier_kind, metrics, carrier_roi = classify_text_block_carrier(image, block)

                self.assertEqual(carrier_kind, CARRIER_KIND_CAPTION_PLATE)
                self.assertIsNotNone(carrier_roi)
                self.assertLessEqual(metrics["white_ratio"], 0.55)

    def test_small_or_ambiguous_analysis_area_falls_back_to_speech_bubble(self) -> None:
        image = _caption_plate_image(color=(150, 105, 88))
        block = _block(xyxy=[14, 14, 24, 24], bubble_xyxy=[12, 12, 34, 34])

        carrier_kind, metrics, carrier_roi = classify_text_block_carrier(image, block)

        self.assertEqual(carrier_kind, CARRIER_KIND_SPEECH_BUBBLE)
        self.assertLess(metrics["analysis_area"], 400.0)
        self.assertIsNone(carrier_roi)

    def test_caption_plate_roi_is_used_by_mask_roi_helpers(self) -> None:
        image = _caption_plate_image(color=(150, 105, 88))
        block = _block(xyxy=[34, 28, 62, 68], bubble_xyxy=[12, 12, 84, 84])

        annotate_text_block_carriers(image, [block])
        ctd_roi = resolve_block_ctd_roi(block, image.shape)
        cleanup_roi = resolve_block_cleanup_roi(block, image.shape)
        effective_roi = resolve_effective_text_bubble_roi(block, image.shape)

        self.assertEqual(block.carrier_kind, CARRIER_KIND_CAPTION_PLATE)
        self.assertIsNotNone(block.carrier_mask_roi_xyxy)
        self.assertEqual(ctd_roi, tuple(block.carrier_mask_roi_xyxy))
        self.assertEqual(cleanup_roi, tuple(block.carrier_mask_roi_xyxy))
        self.assertEqual(effective_roi, tuple(block.carrier_mask_roi_xyxy))


if __name__ == "__main__":
    unittest.main()
