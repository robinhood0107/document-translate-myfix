from __future__ import annotations

import unittest

import numpy as np

from modules.utils.inpaint_evidence import BlockInpaintEvidence, MaskPatch
from modules.utils.inpaint_positive_evidence import (
    build_detector_positive_text_evidence,
)
from modules.utils.textblock import TextBlock


class DetectorPositiveTextEvidenceTests(unittest.TestCase):
    def _block(self, *, origin: str, detector_box=None) -> TextBlock:
        block = TextBlock(
            text_bbox=np.asarray([8, 8, 28, 28], dtype=np.int32),
            bubble_bbox=np.asarray([4, 4, 32, 32], dtype=np.int32),
            text_class="text_bubble",
            inpaint_bboxes=[[12, 12, 16, 16]],
            detector_origin=origin,
            detector_text_bbox=detector_box,
            detector_provider="RTDetrV2ONNXDetection",
        )
        block._erase_skipped_reason = "microtexture_source_seed_unavailable"
        return block

    @staticmethod
    def _evidence(*, structure=None, ownership=None):
        return (
            BlockInpaintEvidence(
                block_id="block-0",
                block_index=0,
                skipped_reason="microtexture_source_seed_unavailable",
                structure_protect=structure,
                ownership_protect=ownership,
            ),
        )

    def test_direct_text_box_widens_ownership_but_never_creates_claim(self) -> None:
        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[10:26, 10:26] = 255
        block = self._block(
            origin="direct_text",
            detector_box=[9, 9, 27, 27],
        )

        result = build_detector_positive_text_evidence(
            [block],
            raw,
            self._evidence(),
            image_shape=(40, 40, 3),
        )

        self.assertEqual(int(np.count_nonzero(result.positive_claim)), 16 * 16)
        self.assertTrue(np.all(result.positive_claim[raw == 0] == 0))
        self.assertIn("rtdetr_raw_text_box", result.block_claim_providers[0])
        self.assertIn(
            "block_detector:RTDetrV2ONNXDetection",
            result.block_claim_providers[0],
        )

    def test_rescue_stays_inside_content_component_ownership(self) -> None:
        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[10:26, 10:26] = 255
        block = self._block(origin="bubble_text_rescue")

        result = build_detector_positive_text_evidence(
            [block],
            raw,
            self._evidence(),
            image_shape=(40, 40, 3),
        )

        self.assertEqual(int(np.count_nonzero(result.positive_claim)), 16)
        self.assertTrue(np.all(result.positive_claim[12:16, 12:16] == 255))
        self.assertNotIn("rtdetr_raw_text_box", result.block_claim_providers[0])

    def test_exact_protection_and_existing_edit_are_subtracted(self) -> None:
        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[12:16, 12:16] = 255
        existing = np.zeros_like(raw)
        existing[12:14, 12:14] = 255
        structure = MaskPatch((12, 12, 16, 16), np.asarray(
            [[0, 0, 0, 0], [0, 0, 0, 0], [255, 255, 0, 0], [255, 255, 0, 0]],
            dtype=np.uint8,
        ))
        ownership = MaskPatch((12, 12, 16, 16), np.asarray(
            [[0, 0, 255, 255], [0, 0, 255, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
            dtype=np.uint8,
        ))
        corner = np.zeros_like(raw)
        corner[14:15, 14:16] = 255

        result = build_detector_positive_text_evidence(
            [self._block(origin="bubble_text_rescue")],
            raw,
            self._evidence(structure=structure, ownership=ownership),
            image_shape=(40, 40, 3),
            existing_edit_mask=existing,
            protected_corner_mask=corner,
        )

        self.assertEqual(int(np.count_nonzero(result.positive_claim)), 16)
        self.assertEqual(int(np.count_nonzero(result.positive_edit)), 2)
        self.assertTrue(np.all(result.positive_edit[14:16, 12:14] == 0))
        self.assertTrue(np.all(result.positive_edit[12:14, 14:16] == 0))

    def test_non_recoverable_block_is_fail_closed(self) -> None:
        raw = np.full((40, 40), 255, dtype=np.uint8)
        block = self._block(origin="direct_text", detector_box=[8, 8, 28, 28])
        block._erase_skipped_reason = "microtexture_intrusion"
        evidence = (
            BlockInpaintEvidence(
                block_id="block-0",
                block_index=0,
                skipped_reason="microtexture_intrusion",
            ),
        )

        result = build_detector_positive_text_evidence(
            [block],
            raw,
            evidence,
            image_shape=(40, 40, 3),
        )

        self.assertEqual(int(np.count_nonzero(result.positive_claim)), 0)
        self.assertEqual(int(np.count_nonzero(result.positive_edit)), 0)

    def test_crop_edge_and_sfx_claims_never_fill_the_ownership_box(self) -> None:
        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[0:3, 1:4] = 255
        raw[18:21, 18:25] = 255
        block = TextBlock(
            text_bbox=np.asarray([0, 0, 30, 30], dtype=np.int32),
            bubble_bbox=np.asarray([0, 0, 34, 34], dtype=np.int32),
            text_class="text_bubble",
            inpaint_bboxes=[[0, 0, 6, 6], [16, 16, 28, 24]],
            detector_origin="bubble_text_rescue",
        )
        block._erase_skipped_reason = "microtexture_source_seed_unavailable"

        result = build_detector_positive_text_evidence(
            [block],
            raw,
            self._evidence(),
            image_shape=(40, 40, 3),
        )

        self.assertEqual(int(np.count_nonzero(result.positive_edit)), 30)
        self.assertTrue(np.array_equal(result.positive_edit > 0, raw > 0))
        self.assertEqual(int(np.count_nonzero(result.positive_edit[3:18])), 0)

    def test_invalid_or_out_of_bounds_provenance_is_fail_closed(self) -> None:
        raw = np.full((40, 40), 255, dtype=np.uint8)
        for detector_box in (None, [-20, -20, -1, -1], [12, 12, 12, 20]):
            block = TextBlock(
                text_bbox=np.asarray([8, 8, 28, 28], dtype=np.int32),
                bubble_bbox=np.asarray([4, 4, 32, 32], dtype=np.int32),
                text_class="text_bubble",
                inpaint_bboxes=[],
                detector_origin="direct_text",
                detector_text_bbox=detector_box,
            )
            block._erase_skipped_reason = (
                "microtexture_source_seed_unavailable"
            )

            result = build_detector_positive_text_evidence(
                [block],
                raw,
                self._evidence(),
                image_shape=(40, 40, 3),
            )

            self.assertEqual(int(np.count_nonzero(result.positive_claim)), 0)
            self.assertEqual(int(np.count_nonzero(result.positive_edit)), 0)

    def test_overlapping_claims_union_once_and_preserve_each_provider(self) -> None:
        raw = np.zeros((40, 40), dtype=np.uint8)
        raw[12:18, 12:18] = 255
        blocks = [
            self._block(origin="bubble_text_rescue"),
            self._block(origin="bubble_text_rescue"),
        ]
        evidence = tuple(
            BlockInpaintEvidence(
                block_id=f"block-{index}",
                block_index=index,
                skipped_reason="microtexture_source_seed_unavailable",
            )
            for index in range(2)
        )

        result = build_detector_positive_text_evidence(
            blocks,
            raw,
            evidence,
            image_shape=(40, 40, 3),
        )

        self.assertEqual(int(np.count_nonzero(result.positive_edit)), 16)
        self.assertEqual(set(result.block_edit_patches), {0, 1})
        self.assertEqual(
            result.block_claim_providers[0],
            result.block_claim_providers[1],
        )


if __name__ == "__main__":
    unittest.main()
