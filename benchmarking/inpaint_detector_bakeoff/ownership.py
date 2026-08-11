from __future__ import annotations

from typing import Iterable, Mapping

import cv2
import numpy as np


REQUIRED_SOURCE_SEED_UNAVAILABLE_REASONS = frozenset(
    {
        "bubble_interior_cap_source_seed_unavailable",
        "bubble_protected_source_seed_unavailable",
        "bubble_residual_source_seed_unavailable",
        "line_art_source_seed_unavailable",
        "microtexture_source_seed_unavailable",
        "residual_source_seed_unavailable",
        "text_prior_unavailable_source_seed_unavailable",
    }
)


def _normalized_xyxy(value: object, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    height, width = shape
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in value[:4])
    except (TypeError, ValueError):
        return None
    x1, x2 = sorted((max(0, min(width, x1)), max(0, min(width, x2))))
    y1, y2 = sorted((max(0, min(height, y1)), max(0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_existing_ownership_mask(
    blocks: Iterable[Mapping[str, object]],
    shape: tuple[int, int],
    *,
    scope: str = "region",
) -> np.ndarray:
    """Reconstruct only existing block/bubble ownership, never a text claim."""

    normalized_scope = str(scope).strip().lower()
    if normalized_scope not in {
        "region",
        "text_prior",
        "content_components",
        "content_prior",
        "required_skip_text_prior",
        "required_skip_components",
    }:
        raise ValueError(f"unsupported ownership scope: {scope}")
    mask = np.zeros(shape, dtype=np.uint8)
    for block in blocks:
        if normalized_scope.startswith("required_skip_") and str(
            block.get("erase_skipped_reason") or ""
        ) not in REQUIRED_SOURCE_SEED_UNAVAILABLE_REASONS:
            continue
        if normalized_scope in {
            "content_components",
            "content_prior",
            "required_skip_components",
        }:
            boxes = block.get("inpaint_bboxes")
            if not isinstance(boxes, (list, tuple)):
                continue
            for candidate in boxes:
                box = _normalized_xyxy(candidate, shape)
                if box is None:
                    continue
                x1, y1, x2, y2 = box
                mask[y1:y2, x1:x2] = 255
            continue
        text_class = str(block.get("text_class") or "")
        if normalized_scope in {"text_prior", "required_skip_text_prior"}:
            candidates = (
                block.get("mask_anchor_xyxy"),
                block.get("xyxy"),
                block.get("mask_actual_bbox"),
            )
        elif text_class == "text_bubble":
            candidates = (
                block.get("bubble_xyxy"),
                block.get("cleanup_roi_xyxy"),
                block.get("ctd_roi_xyxy"),
                block.get("mask_anchor_xyxy"),
            )
        else:
            candidates = (
                block.get("text_free_erase_envelope_xyxy"),
                block.get("mask_actual_bbox"),
                block.get("mask_anchor_xyxy"),
            )
        box = next(
            (
                normalized
                for candidate in candidates
                if (normalized := _normalized_xyxy(candidate, shape)) is not None
            ),
            None,
        )
        if box is None:
            continue
        x1, y1, x2, y2 = box
        mask[y1:y2, x1:x2] = 255
    if normalized_scope == "content_prior" and np.any(mask):
        mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return np.where(mask > 0, 255, 0).astype(np.uint8)
