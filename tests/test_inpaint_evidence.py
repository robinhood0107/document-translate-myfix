from __future__ import annotations

import numpy as np
import pytest

from modules.utils.inpaint_evidence import (
    BlockInpaintEvidence,
    MaskPatch,
    combine_evidence_patches,
)


def test_positive_claim_patches_union_without_losing_provider_provenance() -> None:
    first = np.zeros((6, 6), dtype=np.uint8)
    first[1:4, 1:4] = 255
    second = np.zeros((6, 6), dtype=np.uint8)
    second[2:5, 2:5] = 255
    evidence = (
        BlockInpaintEvidence(
            block_id="b0",
            block_index=0,
            positive_claim=MaskPatch((2, 3, 8, 9), first),
            claim_providers=("ctd_raw", "ctd_raw"),
        ),
        BlockInpaintEvidence(
            block_id="b0",
            block_index=0,
            positive_claim=MaskPatch((2, 3, 8, 9), second),
            claim_providers=("ctbd",),
        ),
    )

    combined = combine_evidence_patches(
        evidence,
        "positive_claim",
        (12, 12, 3),
    )

    assert np.count_nonzero(combined) == 14
    assert evidence[0].claim_providers == ("ctd_raw",)
    assert evidence[1].claim_providers == ("ctbd",)


def test_bubble_route_evidence_keeps_sparse_interior_and_deduplicates_reasons() -> None:
    interior = np.zeros((5, 7), dtype=np.uint8)
    interior[1:4, 1:6] = 255
    item = BlockInpaintEvidence(
        block_id="b0",
        block_index=0,
        bubble_interior=MaskPatch((3, 4, 10, 9), interior),
        route_decision="broad",
        route_reasons=("validated_interior", "validated_interior", "clean"),
    )

    combined = combine_evidence_patches(
        (item,),
        "bubble_interior",
        (16, 16, 3),
    )

    assert np.count_nonzero(combined) == 15
    assert item.route_decision == "broad"
    assert item.route_reasons == ("validated_interior", "clean")


def test_sparse_evidence_rejects_invalid_or_out_of_bounds_roi() -> None:
    with pytest.raises(ValueError, match="inpaint_evidence_roi_invalid"):
        MaskPatch((2, 2, 2, 4), np.zeros((2, 0), dtype=np.uint8))

    evidence = (
        BlockInpaintEvidence(
            block_id="b0",
            block_index=0,
            positive_edit=MaskPatch(
                (8, 8, 12, 12),
                np.full((4, 4), 255, dtype=np.uint8),
            ),
        ),
    )
    with pytest.raises(ValueError, match="inpaint_evidence_roi_out_of_bounds"):
        combine_evidence_patches(evidence, "positive_edit", (10, 10, 3))
