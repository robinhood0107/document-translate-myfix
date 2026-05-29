from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

from modules.masking.text_free_rescue import (
    apply_text_free_rescue_mask,
    mark_text_free_inpaint_residuals,
)
from modules.utils.inpaint_envelope import build_text_free_erase_envelope
from modules.utils.textblock import TextBlock


def _block(xyxy: list[int]) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        text_class="text_free",
        text="demo",
        translation="demo",
    )


def _candidate(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        mask[y1:y2, x1:x2] = 255
    return mask


class TextFreeRescueTests(unittest.TestCase):
    def test_low_coverage_text_free_gets_contour_rescue_inside_envelope(self) -> None:
        image = np.full((140, 140, 3), 128, dtype=np.uint8)
        block = _block([30, 20, 45, 70])
        base_mask = np.zeros((140, 140), dtype=np.uint8)
        base_mask[1:3, 1:3] = 255
        envelope = build_text_free_erase_envelope(block, image.shape)
        assert envelope is not None

        def fake_candidate(crop):
            return _candidate(crop.shape[:2], [(8, 12, 14, 46)])

        with mock.patch("modules.masking.text_free_rescue._candidate_stroke_mask", side_effect=fake_candidate):
            final_mask, details = apply_text_free_rescue_mask(image, [block], base_mask)

        self.assertEqual(details["text_free_rescue_applied_count"], 1)
        self.assertTrue(block._text_free_rescue_applied)
        self.assertIn("contour_rescue_applied", block._text_free_rescue_reason_codes)
        self.assertGreater(int(np.count_nonzero(final_mask[20:70, 30:45])), 0)
        outside = final_mask.copy()
        x1, y1, x2, y2 = envelope
        outside[y1:y2, x1:x2] = 0
        self.assertEqual(int(np.count_nonzero(outside)), int(np.count_nonzero(base_mask)))

    def test_sufficient_text_free_coverage_does_not_rescue(self) -> None:
        image = np.full((70, 70, 3), 128, dtype=np.uint8)
        block = _block([20, 20, 40, 50])
        base_mask = np.zeros((70, 70), dtype=np.uint8)
        base_mask[20:30, 20:30] = 255

        with mock.patch("modules.masking.text_free_rescue._candidate_stroke_mask") as candidate:
            final_mask, details = apply_text_free_rescue_mask(image, [block], base_mask)

        candidate.assert_not_called()
        self.assertEqual(details["text_free_rescue_applied_count"], 0)
        self.assertFalse(block._text_free_rescue_applied)
        np.testing.assert_array_equal(final_mask, base_mask)

    def test_dense_text_free_rescue_is_rejected(self) -> None:
        image = np.full((150, 150, 3), 128, dtype=np.uint8)
        block = _block([20, 20, 50, 60])
        base_mask = np.zeros((150, 150), dtype=np.uint8)

        def fake_candidate(crop):
            return np.full(crop.shape[:2], 255, dtype=np.uint8)

        with mock.patch("modules.masking.text_free_rescue._candidate_stroke_mask", side_effect=fake_candidate):
            final_mask, details = apply_text_free_rescue_mask(image, [block], base_mask)

        self.assertEqual(details["text_free_rescue_applied_count"], 0)
        self.assertFalse(block._text_free_rescue_applied)
        self.assertIn("rescue_too_dense", block._text_free_rescue_reason_codes)
        self.assertEqual(int(np.count_nonzero(final_mask)), 0)

    def test_residual_gate_marks_rescued_text_free_for_review(self) -> None:
        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        block = _block([20, 20, 60, 60])
        block._text_free_rescue_applied = True

        def fake_candidate(crop):
            boxes = [(idx * 4, idx * 4, idx * 4 + 2, idx * 4 + 2) for idx in range(8)]
            return _candidate(crop.shape[:2], boxes)

        with mock.patch("modules.masking.text_free_rescue._candidate_stroke_mask", side_effect=fake_candidate):
            stats = mark_text_free_inpaint_residuals(image, [block])

        self.assertEqual(stats["inpaint_residual_checked_count"], 1)
        self.assertEqual(stats["inpaint_needs_review_count"], 1)
        self.assertTrue(block._inpaint_needs_review)
        self.assertEqual(block._inpaint_residual_status, "needs_review")
        self.assertIn("residual_component_count", block._inpaint_residual_reason_codes)

    def test_residual_gate_checks_covered_text_free_without_rescue(self) -> None:
        image = np.full((80, 80, 3), 128, dtype=np.uint8)
        block = _block([20, 20, 60, 60])
        block._text_free_rescue_applied = False

        def fake_candidate(crop):
            boxes = [(idx * 4, idx * 4, idx * 4 + 2, idx * 4 + 2) for idx in range(8)]
            return _candidate(crop.shape[:2], boxes)

        with mock.patch("modules.masking.text_free_rescue._candidate_stroke_mask", side_effect=fake_candidate):
            stats = mark_text_free_inpaint_residuals(image, [block])

        self.assertEqual(stats["inpaint_residual_checked_count"], 1)
        self.assertEqual(stats["inpaint_needs_review_count"], 1)
        self.assertTrue(block._inpaint_needs_review)
        self.assertEqual(block._inpaint_residual_status, "needs_review")


if __name__ == "__main__":
    unittest.main()
