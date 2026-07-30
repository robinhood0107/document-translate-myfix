from __future__ import annotations

import unittest

import numpy as np

from modules.ocr.result_contract import (
    MASK_STRATEGY_BUBBLE_SAFE,
    MASK_STRATEGY_GLYPH_ONLY,
    MASK_STRATEGY_GLYPH_ONLY_STRUCTURE_PROTECT,
    MASK_STRATEGY_PRESERVE_ORIGINAL,
    PROCESSING_ACTION_PRESERVE,
    PROCESSING_ACTION_REVIEW,
    PROCESSING_ACTION_TRANSLATE_INPAINT,
    SEMANTIC_ROLE_AMBIGUOUS,
    SEMANTIC_ROLE_DIALOGUE_BUBBLE,
    SEMANTIC_ROLE_DIALOGUE_FREE,
    SEMANTIC_ROLE_SFX,
    assign_ocr_processing_contract,
    canonicalize_exact_duplicate_blocks,
    finalize_ocr_processing_contract,
    finalize_ocr_processing_contracts,
    select_translate_inpaint_blocks,
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
    def test_bubble_defaults_to_translate_with_safe_bubble_mask(self) -> None:
        block = _block(
            [10, 20, 80, 140],
            bubble_bbox=[5, 10, 90, 150],
        )

        finalize_ocr_processing_contract(block)

        self.assertEqual(
            block.semantic_role,
            SEMANTIC_ROLE_DIALOGUE_BUBBLE,
        )
        self.assertEqual(
            block.processing_action,
            PROCESSING_ACTION_TRANSLATE_INPAINT,
        )
        self.assertEqual(block.mask_strategy, MASK_STRATEGY_BUBBLE_SAFE)

    def test_text_free_defaults_to_translate_with_glyph_mask(self) -> None:
        block = _block(
            [10, 20, 80, 140],
            bubble_bbox=None,
            text_class="text_free",
        )

        finalize_ocr_processing_contract(block)

        self.assertEqual(block.semantic_role, SEMANTIC_ROLE_DIALOGUE_FREE)
        self.assertEqual(
            block.processing_action,
            PROCESSING_ACTION_TRANSLATE_INPAINT,
        )
        self.assertEqual(block.mask_strategy, MASK_STRATEGY_GLYPH_ONLY)

    def test_explicit_sfx_is_preserved_without_destructive_mask(self) -> None:
        block = _block(
            [10, 20, 80, 140],
            bubble_bbox=None,
            text_class="sfx",
        )

        finalize_ocr_processing_contract(block)

        self.assertEqual(block.semantic_role, SEMANTIC_ROLE_SFX)
        self.assertEqual(
            block.processing_action,
            PROCESSING_ACTION_PRESERVE,
        )
        self.assertEqual(
            block.mask_strategy,
            MASK_STRATEGY_PRESERVE_ORIGINAL,
        )
        self.assertEqual(select_translate_inpaint_blocks([block]), [])

    def test_invalid_explicit_contract_fails_closed_to_review(self) -> None:
        block = _block(
            [10, 20, 80, 140],
            bubble_bbox=[5, 10, 90, 150],
        )

        assign_ocr_processing_contract(
            block,
            semantic_role="not-a-role",
            processing_action="delete-everything",
            decision_source="test",
        )

        self.assertEqual(block.semantic_role, SEMANTIC_ROLE_AMBIGUOUS)
        self.assertEqual(block.processing_action, PROCESSING_ACTION_REVIEW)
        self.assertEqual(
            block.mask_strategy,
            MASK_STRATEGY_PRESERVE_ORIGINAL,
        )
        self.assertIn(
            "invalid_semantic_role",
            block.processing_decision_reasons,
        )
        self.assertIn(
            "invalid_processing_action",
            block.processing_decision_reasons,
        )

    def test_structure_risk_selects_structure_protection_mask(self) -> None:
        block = _block(
            [10, 20, 80, 140],
            bubble_bbox=[5, 10, 90, 150],
        )
        block.bubble_transparency_risk = True

        finalize_ocr_processing_contract(block)

        self.assertEqual(
            block.mask_strategy,
            MASK_STRATEGY_GLYPH_ONLY_STRUCTURE_PROTECT,
        )

    def test_summary_and_action_selection_preserve_original_order(self) -> None:
        dialogue = _block(
            [10, 20, 80, 140],
            bubble_bbox=[5, 10, 90, 150],
        )
        sfx = _block(
            [100, 120, 180, 240],
            bubble_bbox=None,
            text_class="sfx",
        )
        free = _block(
            [200, 220, 280, 340],
            bubble_bbox=None,
            text_class="text_free",
        )

        summary = finalize_ocr_processing_contracts(
            [dialogue, sfx, free]
        )

        self.assertEqual(summary["block_count"], 3)
        self.assertEqual(
            summary["processing_action_counts"],
            {
                PROCESSING_ACTION_TRANSLATE_INPAINT: 2,
                PROCESSING_ACTION_PRESERVE: 1,
            },
        )
        self.assertEqual(
            select_translate_inpaint_blocks([dialogue, sfx, free]),
            [dialogue, free],
        )

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
