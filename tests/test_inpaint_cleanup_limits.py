from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint


def test_autonomous_residue_cleanup_is_a_compatibility_noop() -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[24:28, 24:28] = 255
    block = SimpleNamespace(
        xyxy=[20, 20, 32, 32],
        bubble_xyxy=[8, 8, 56, 56],
        cleanup_roi_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
    )

    cleaned, final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [block],
        None,
        None,
    )

    assert cleaned is image
    assert final_mask is mask
    assert stats["autonomous_residue_cleanup"] == "disabled"
    assert stats["applied"] is False
    assert stats["pass2_applied_pixel_count"] == 0
    assert np.count_nonzero(stats["residue_mask"]) == 0


def test_retired_cleanup_never_expands_an_empty_mask() -> None:
    image = np.full((32, 32, 3), 210, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)

    cleaned, final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [],
        None,
        None,
    )

    np.testing.assert_array_equal(cleaned, image)
    np.testing.assert_array_equal(final_mask, mask)
    assert stats["autonomous_residue_cleanup"] == "disabled"
    assert stats["pass2_candidate_count"] == 0
