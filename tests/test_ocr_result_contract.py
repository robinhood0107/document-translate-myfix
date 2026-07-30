from __future__ import annotations

import unittest

import numpy as np

from modules.ocr.result_contract import (
    canonicalize_exact_duplicate_blocks,
)
from modules.utils.textblock import TextBlock


def _block(
    text_bbox: list[int] | None,
    *,
    bubble_bbox: list[int] | None,
    text_class: str = "text_bubble",
    direction: str = "vertical",
) -> TextBlock:
    return TextBlock(
        text_bbox=(
            np.asarray(text_bbox, dtype=np.int32)
            if text_bbox is not None
            else None
        ),
        bubble_bbox=(
            np.asarray(bubble_bbox, dtype=np.int32)
            if bubble_bbox is not None
            else None
        ),
        text_class=text_class,
        direction=direction,
    )


class OCRResultContractTests(unittest.TestCase):
    def test_exact_same_source_geometry_keeps_one_canonical_block(self) -> None:
        first = _block(
            [100, 120, 180, 260],
            bubble_bbox=[80, 90, 210, 290],
        )
        duplicate = _block(
            [100, 120, 180, 260],
            bubble_bbox=[80, 90, 210, 290],
        )

        canonical, summary = canonicalize_exact_duplicate_blocks(
            [first, duplicate],
            source_identity="source-sha",
        )

        self.assertEqual(canonical, [first])
        self.assertEqual(summary["input_block_count"], 2)
        self.assertEqual(summary["canonical_block_count"], 1)
        self.assertEqual(summary["duplicate_alias_count"], 1)
        self.assertEqual(first.canonical_block_id, first.block_id)
        self.assertEqual(
            first.duplicate_alias_block_ids,
            [duplicate.block_id],
        )
        self.assertEqual(duplicate.canonical_block_id, first.block_id)

    def test_distinct_fragments_inside_same_bubble_are_preserved(self) -> None:
        first = _block(
            [100, 120, 180, 180],
            bubble_bbox=[80, 90, 240, 300],
        )
        second = _block(
            [100, 200, 180, 260],
            bubble_bbox=[80, 90, 240, 300],
        )

        canonical, summary = canonicalize_exact_duplicate_blocks(
            [first, second],
            source_identity="source-sha",
        )

        self.assertEqual(canonical, [first, second])
        self.assertEqual(summary["duplicate_alias_count"], 0)

    def test_same_geometry_with_different_class_is_preserved(self) -> None:
        first = _block(
            [20, 30, 80, 120],
            bubble_bbox=[10, 20, 90, 130],
        )
        second = _block(
            [20, 30, 80, 120],
            bubble_bbox=[10, 20, 90, 130],
            text_class="text_free",
        )

        canonical, summary = canonicalize_exact_duplicate_blocks(
            [first, second],
            source_identity="source-sha",
        )

        self.assertEqual(canonical, [first, second])
        self.assertEqual(summary["duplicate_alias_count"], 0)

    def test_invalid_geometry_is_preserved_for_existing_error_handling(
        self,
    ) -> None:
        invalid = _block(None, bubble_bbox=None)
        duplicate_invalid = _block(None, bubble_bbox=None)

        canonical, summary = canonicalize_exact_duplicate_blocks(
            [invalid, duplicate_invalid],
        )

        self.assertEqual(canonical, [invalid, duplicate_invalid])
        self.assertEqual(summary["duplicate_alias_count"], 0)

    def test_invalid_bubble_or_angle_is_preserved_fail_open(self) -> None:
        invalid_bubble = _block(
            [20, 30, 80, 120],
            bubble_bbox=[10, 20, 10, 130],
        )
        invalid_bubble_duplicate = _block(
            [20, 30, 80, 120],
            bubble_bbox=[10, 20, 10, 130],
        )
        invalid_angle = _block(
            [100, 130, 160, 220],
            bubble_bbox=[90, 110, 180, 240],
        )
        invalid_angle.angle = "not-a-number"
        invalid_angle_duplicate = invalid_angle.deep_copy()

        canonical, summary = canonicalize_exact_duplicate_blocks(
            [
                invalid_bubble,
                invalid_bubble_duplicate,
                invalid_angle,
                invalid_angle_duplicate,
            ],
            source_identity="source-sha",
        )

        self.assertEqual(
            canonical,
            [
                invalid_bubble,
                invalid_bubble_duplicate,
                invalid_angle,
                invalid_angle_duplicate,
            ],
        )
        self.assertEqual(summary["duplicate_alias_count"], 0)


if __name__ == "__main__":
    unittest.main()
