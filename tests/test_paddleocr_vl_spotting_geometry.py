from __future__ import annotations

import unittest

import numpy as np

from modules.ocr.paddleocr_vl_spotting.geometry import (
    assign_spotting_regions,
    map_normalized_region,
)
from modules.ocr.paddleocr_vl_spotting.response_contract import PaddleSpottingRegion
from modules.utils.textblock import TextBlock


def _region(
    text: str,
    points: tuple[
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
        tuple[int, int],
    ],
    line: int = 1,
) -> PaddleSpottingRegion:
    return PaddleSpottingRegion(
        text=text,
        normalized_points=points,
        source_line=line,
    )


def _block(
    bbox: tuple[int, int, int, int],
    *,
    block_id: str,
    direction: str = "",
) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(bbox, dtype=np.int32),
        text_class="text_bubble",
        block_id=block_id,
        direction=direction,
    )


class PaddleSpottingGeometryTests(unittest.TestCase):
    def test_maps_normalized_coordinates_to_original_image_space(self) -> None:
        mapped = map_normalized_region(
            _region(
                "scaled",
                ((250, 100), (750, 100), (750, 900), (250, 900)),
            ),
            image_width=1920,
            image_height=1080,
        )

        self.assertEqual(
            mapped.points,
            ((480, 108), (1440, 108), (1440, 972), (480, 972)),
        )
        self.assertEqual(mapped.bbox_xyxy, (480, 108, 1440, 972))

    def test_ambiguous_overlap_invalidates_all_involved_blocks(self) -> None:
        blocks = [
            _block((100, 100, 300, 300), block_id="left"),
            _block((250, 100, 450, 300), block_id="right"),
        ]
        result = assign_spotting_regions(
            (
                _region(
                    "safe",
                    ((110, 120), (220, 120), (220, 180), (110, 180)),
                    line=1,
                ),
                _region(
                    "ambiguous",
                    ((260, 120), (290, 120), (290, 180), (260, 180)),
                    line=2,
                ),
            ),
            blocks,
            image_width=1000,
            image_height=1000,
        )

        self.assertFalse(result.assignments[0])
        self.assertFalse(result.assignments[1])
        self.assertEqual(len(result.ambiguous_regions), 1)
        self.assertEqual(result.ambiguous_block_indices, (0, 1))
        self.assertEqual(
            result.ambiguous_regions[0]["reason"],
            "one_spot_multiple_detector_blocks",
        )

    def test_refuses_a_native_region_that_covers_multiple_detector_blocks(
        self,
    ) -> None:
        blocks = [
            _block((100, 100, 300, 300), block_id="top"),
            _block((100, 320, 300, 520), block_id="bottom"),
        ]
        result = assign_spotting_regions(
            (
                _region(
                    "merged text",
                    ((90, 90), (310, 90), (310, 530), (90, 530)),
                ),
            ),
            blocks,
            image_width=1000,
            image_height=1000,
        )

        self.assertFalse(result.assignments[0])
        self.assertFalse(result.assignments[1])
        self.assertEqual(result.ambiguous_block_indices, (0, 1))
        self.assertEqual(
            result.ambiguous_regions[0]["candidate_block_ids"],
            ["top", "bottom"],
        )

    def test_ambiguous_region_invalidates_an_earlier_safe_assignment(
        self,
    ) -> None:
        blocks = [
            _block((100, 100, 300, 300), block_id="top"),
            _block((100, 320, 300, 520), block_id="bottom"),
        ]
        result = assign_spotting_regions(
            (
                _region(
                    "safe first",
                    ((120, 120), (280, 120), (280, 200), (120, 200)),
                    line=1,
                ),
                _region(
                    "merged later",
                    ((90, 90), (310, 90), (310, 530), (90, 530)),
                    line=2,
                ),
            ),
            blocks,
            image_width=1000,
            image_height=1000,
        )

        self.assertFalse(result.assignments[0])
        self.assertFalse(result.assignments[1])
        self.assertEqual(result.ambiguous_block_indices, (0, 1))

    def test_preserves_multiple_lines_and_detector_reading_order(self) -> None:
        horizontal = _block((0, 0, 1000, 1000), block_id="horizontal")
        vertical = _block(
            (0, 0, 1000, 1000),
            block_id="vertical",
            direction="vertical",
        )
        regions = (
            _region(
                "bottom-left",
                ((100, 700), (300, 700), (300, 800), (100, 800)),
                line=1,
            ),
            _region(
                "top-right",
                ((700, 100), (900, 100), (900, 200), (700, 200)),
                line=2,
            ),
        )

        horizontal_result = assign_spotting_regions(
            regions,
            [horizontal],
            image_width=1000,
            image_height=1000,
        )
        vertical_result = assign_spotting_regions(
            regions,
            [vertical],
            image_width=1000,
            image_height=1000,
        )

        self.assertEqual(
            [item.region.text for item in horizontal_result.assignments[0]],
            ["top-right", "bottom-left"],
        )
        self.assertEqual(
            [item.region.text for item in vertical_result.assignments[0]],
            ["top-right", "bottom-left"],
        )

    def test_leaves_non_overlapping_native_spot_as_shadow_region(self) -> None:
        result = assign_spotting_regions(
            (
                _region(
                    "outside",
                    ((700, 700), (900, 700), (900, 900), (700, 900)),
                ),
            ),
            [_block((0, 0, 200, 200), block_id="detector")],
            image_width=1000,
            image_height=1000,
        )

        self.assertFalse(result.assignments[0])
        self.assertEqual(
            [region.text for region in result.unmatched_regions],
            ["outside"],
        )


if __name__ == "__main__":
    unittest.main()
