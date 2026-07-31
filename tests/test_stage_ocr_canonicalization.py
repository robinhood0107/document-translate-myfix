from __future__ import annotations

import unittest

import numpy as np

from modules.utils.textblock import TextBlock
from pipeline.stage_batched_processor import (
    StageBatchedProcessor,
    StagePageContext,
)


class StageOCRCanonicalizationTests(unittest.TestCase):
    def test_exact_duplicate_leaves_one_block_for_all_downstream_stages(
        self,
    ) -> None:
        first = TextBlock(
            text_bbox=np.asarray([40, 50, 120, 180], dtype=np.int32),
            bubble_bbox=np.asarray([20, 30, 150, 210], dtype=np.int32),
            text_class="text_bubble",
            direction="vertical",
        )
        duplicate = TextBlock(
            text_bbox=np.asarray([40, 50, 120, 180], dtype=np.int32),
            bubble_bbox=np.asarray([20, 30, 150, 210], dtype=np.int32),
            text_class="text_bubble",
            direction="vertical",
        )
        context = StagePageContext(
            image_path="synthetic-page.png",
            image_name="synthetic-page.png",
            source_lang="Japanese",
            target_lang="Korean",
            blk_list=[first, duplicate],
            source_decoded_sha256="source-sha",
        )

        StageBatchedProcessor._canonicalize_ocr_inputs([context])

        self.assertEqual(context.blk_list, [first])
        self.assertEqual(
            context.ocr_canonicalization_summary["duplicate_alias_count"],
            1,
        )
        self.assertEqual(first.duplicate_alias_block_ids, [duplicate.block_id])

    def test_same_bubble_distinct_text_boxes_remain_independent(self) -> None:
        first = TextBlock(
            text_bbox=np.asarray([40, 50, 120, 100], dtype=np.int32),
            bubble_bbox=np.asarray([20, 30, 150, 210], dtype=np.int32),
            text_class="text_bubble",
            direction="vertical",
        )
        second = TextBlock(
            text_bbox=np.asarray([40, 120, 120, 180], dtype=np.int32),
            bubble_bbox=np.asarray([20, 30, 150, 210], dtype=np.int32),
            text_class="text_bubble",
            direction="vertical",
        )
        context = StagePageContext(
            image_path="synthetic-page.png",
            image_name="synthetic-page.png",
            source_lang="Japanese",
            target_lang="Korean",
            blk_list=[first, second],
        )

        StageBatchedProcessor._canonicalize_ocr_inputs([context])

        self.assertEqual(context.blk_list, [first, second])
        self.assertEqual(
            context.ocr_canonicalization_summary["duplicate_alias_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
