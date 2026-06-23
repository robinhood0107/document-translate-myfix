from __future__ import annotations

import unittest

import numpy as np

from modules.detection.base import DetectionEngine
from modules.detection.utils.bubble_text_rescue import detect_bubble_text_rescue_boxes


class _DummyDetectionEngine(DetectionEngine):
    def initialize(self, **kwargs) -> None:
        del kwargs

    def detect(self, image: np.ndarray):
        del image
        return []


def _gray_bubble_with_outlined_text() -> np.ndarray:
    image = np.full((100, 120, 3), 240, dtype=np.uint8)
    image[10:90, 20:100] = [122, 122, 122]
    image[26:76, 50:72] = [246, 246, 246]
    image[30:72, 54:68] = [14, 14, 14]
    return image


class BubbleTextRescueTests(unittest.TestCase):
    def test_rescue_detects_outlined_text_inside_unmatched_bubble(self) -> None:
        image = _gray_bubble_with_outlined_text()

        rescued = detect_bubble_text_rescue_boxes(
            image,
            bubble_boxes=np.asarray([[20, 10, 100, 90]], dtype=np.int32),
            text_boxes=np.asarray([], dtype=np.int32),
        )

        self.assertEqual(len(rescued), 1)
        text_box, bubble_box = rescued[0]
        self.assertEqual(bubble_box, (20, 10, 100, 90))
        self.assertLessEqual(text_box[0], 50)
        self.assertLessEqual(text_box[1], 26)
        self.assertGreaterEqual(text_box[2], 72)
        self.assertGreaterEqual(text_box[3], 76)

    def test_create_text_blocks_adds_rescued_bubble_block(self) -> None:
        image = _gray_bubble_with_outlined_text()
        engine = _DummyDetectionEngine()

        blocks = engine.create_text_blocks(
            image,
            text_boxes=np.asarray([], dtype=np.int32),
            bubble_boxes=np.asarray([[20, 10, 100, 90]], dtype=np.int32),
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text_class, "text_bubble")
        self.assertEqual(blocks[0].direction, "vertical")
        self.assertTrue(np.array_equal(blocks[0].bubble_xyxy, np.asarray([20, 10, 100, 90], dtype=np.int32)))

    def test_create_text_blocks_does_not_duplicate_existing_text_box(self) -> None:
        image = _gray_bubble_with_outlined_text()
        engine = _DummyDetectionEngine()

        blocks = engine.create_text_blocks(
            image,
            text_boxes=np.asarray([[48, 24, 74, 78]], dtype=np.int32),
            bubble_boxes=np.asarray([[20, 10, 100, 90]], dtype=np.int32),
        )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].text_class, "text_bubble")

    def test_rescue_ignores_empty_bubble(self) -> None:
        image = np.full((100, 120, 3), 240, dtype=np.uint8)
        image[10:90, 20:100] = [122, 122, 122]

        rescued = detect_bubble_text_rescue_boxes(
            image,
            bubble_boxes=np.asarray([[20, 10, 100, 90]], dtype=np.int32),
            text_boxes=np.asarray([], dtype=np.int32),
        )

        self.assertEqual(rescued, [])


if __name__ == "__main__":
    unittest.main()
