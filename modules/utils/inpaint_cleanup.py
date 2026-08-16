from __future__ import annotations

from typing import Iterable

import imkit as imk
import numpy as np

from modules.utils.bubble_erase import (
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    fill_bubble_edit_mask,
)
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    normalize_edit_mask,
)
from modules.utils.inpaint_evidence import BlockInpaintEvidence
from modules.utils.textblock import TextBlock

def _empty_pass2_stats(mask_shape: tuple[int, int]) -> dict:
    """Return the legacy telemetry shape with autonomous cleanup disabled."""

    return {
        "applied": False,
        "component_count": 0,
        "block_count": 0,
        "pass_name": "residue_pass2",
        "residue_mask": np.zeros(mask_shape, dtype=np.uint8),
        "pass2_candidate_count": 0,
        "pass2_bubble_candidate_count": 0,
        "pass2_bubble_kept_count": 0,
        "pass2_text_free_candidate_count": 0,
        "pass2_text_free_kept_count": 0,
        "residue_mask_pre_cap_pixel_count": 0,
        "residue_mask_cap_pixel_count": 0,
        "residue_mask_cap_dilate_px": 0,
        "pass2_backend": "",
        "pass2_backend_distribution": {},
        "pass2_applied_block_count": 0,
        "pass2_fallback_block_count": 0,
        "pass2_applied_pixel_count": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_pass_cap_dropped_candidate_count": 0,
        "residue_pass_structure_guard_block_count": 0,
    }


def _empty_duplicate_bubble_inner_fill_stats(
    mask_shape: tuple[int, int],
) -> dict:
    return {
        "applied": False,
        "pass_name": "duplicate_bubble_inner_fill",
        "duplicate_bubble_inner_fill_mask": np.zeros(
            mask_shape,
            dtype=np.uint8,
        ),
        "duplicate_bubble_inner_fill_pixel_count": 0,
        "duplicate_bubble_inner_fill_backend": "",
    }


def fill_duplicate_bubble_inner_regions(
    inpainted_image: np.ndarray,
    duplicate_bubble_inner_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    if inpainted_image is None:
        shape = (
            duplicate_bubble_inner_mask.shape
            if duplicate_bubble_inner_mask is not None
            else (0, 0)
        )
        return (
            inpainted_image,
            _empty_duplicate_bubble_inner_fill_stats(shape),
        )

    edit_mask = normalize_edit_mask(
        duplicate_bubble_inner_mask,
        inpainted_image.shape,
    )
    if edit_mask.size == 0 or not np.any(edit_mask):
        return (
            inpainted_image,
            _empty_duplicate_bubble_inner_fill_stats(
                inpainted_image.shape[:2]
            ),
        )

    filled_image, backend = fill_bubble_edit_mask(
        inpainted_image,
        edit_mask,
    )
    if backend == ERASE_MODE_BUBBLE_LAMA_FALLBACK:
        stats = _empty_duplicate_bubble_inner_fill_stats(
            inpainted_image.shape[:2]
        )
        stats["duplicate_bubble_inner_fill_backend"] = backend
        return inpainted_image, stats

    filled_image = imk.convert_scale_abs(filled_image)
    filled_image = composite_with_edit_mask(
        inpainted_image,
        filled_image,
        edit_mask,
    )
    return filled_image, {
        "applied": True,
        "pass_name": "duplicate_bubble_inner_fill",
        "duplicate_bubble_inner_fill_mask": edit_mask,
        "duplicate_bubble_inner_fill_pixel_count": int(
            np.count_nonzero(edit_mask)
        ),
        "duplicate_bubble_inner_fill_backend": backend,
    }


def apply_duplicate_bubble_inner_fill(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    mask_details: dict | None,
    cleanup_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    merged_stats = dict(cleanup_stats or {})
    duplicate_mask = (mask_details or {}).get(
        "duplicate_bubble_inner_mask"
    )
    filled_image, fill_stats = fill_duplicate_bubble_inner_regions(
        inpainted_image,
        duplicate_mask,
    )
    merged_stats["duplicate_bubble_inner_fill"] = fill_stats
    if not fill_stats.get("applied"):
        return inpainted_image, mask, merged_stats

    fill_mask = normalize_edit_mask(
        fill_stats.get("duplicate_bubble_inner_fill_mask"),
        filled_image.shape,
    )
    merged_mask = np.where(
        (normalize_edit_mask(mask, filled_image.shape) > 0)
        | (fill_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    return filled_image, merged_mask, merged_stats


def refine_bubble_residue_inpaint(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    blk_list: Iterable[TextBlock],
    inpainter,
    config,
    page_label: str = "",
    *,
    protected_corner_mask: np.ndarray | None = None,
    routing_evidence: tuple[BlockInpaintEvidence, ...] = (),
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compatibility no-op for the retired autonomous residue pass.

    Text pixels may only be added by detector-positive evidence.  The legacy
    arguments remain accepted so older integrations fail safe during the
    stacked-PR migration.
    """

    del (
        blk_list,
        inpainter,
        config,
        page_label,
        protected_corner_mask,
        routing_evidence,
    )
    shape = mask.shape if mask is not None else inpainted_image.shape[:2]
    stats = _empty_pass2_stats(shape)
    stats["autonomous_residue_cleanup"] = "disabled"
    return inpainted_image, mask, stats
