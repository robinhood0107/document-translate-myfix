from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np

from modules.detection.utils.content import get_inpaint_bboxes
from modules.utils.carrier import (
    CARRIER_KIND_CAPTION_PLATE,
    normalize_xyxy as normalize_carrier_xyxy,
    resolve_effective_text_bubble_roi,
)


def normalize_xyxy(box, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    return normalize_carrier_xyxy(box, image_shape)


def _expand_bbox(
    text_xyxy,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    width_ratio: float,
    height_ratio: float,
    min_pad: int,
    max_pad: int,
) -> tuple[int, int, int, int] | None:
    norm = normalize_xyxy(text_xyxy, image_shape)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    img_h, img_w = image_shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = min(max_pad, max(min_pad, int(round(width * width_ratio))))
    pad_y = min(max_pad, max(min_pad, int(round(height * height_ratio))))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(img_w, x2 + pad_x)
    ey2 = min(img_h, y2 + pad_y)
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return ex1, ey1, ex2, ey2


def resolve_block_ctd_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    explicit = normalize_xyxy(getattr(block, 'ctd_roi_xyxy', None), image_shape)
    if explicit is not None:
        return explicit

    text_class = getattr(block, 'text_class', '') or ''
    if text_class == 'text_bubble':
        bubble_roi = resolve_effective_text_bubble_roi(block, image_shape)
        if bubble_roi is not None:
            return bubble_roi

    mask_alias = normalize_xyxy(getattr(block, 'mask_roi_xyxy', None), image_shape)
    if mask_alias is not None:
        return mask_alias

    return _expand_bbox(
        getattr(block, 'xyxy', None),
        image_shape,
        width_ratio=0.04,
        height_ratio=0.04,
        min_pad=4,
        max_pad=8,
    )


def resolve_block_cleanup_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    explicit = normalize_xyxy(getattr(block, 'cleanup_roi_xyxy', None), image_shape)
    if explicit is not None:
        return explicit

    text_class = getattr(block, 'text_class', '') or ''
    if text_class == 'text_bubble':
        bubble_roi = resolve_effective_text_bubble_roi(block, image_shape)
        if bubble_roi is not None:
            return bubble_roi

    ctd_roi = normalize_xyxy(getattr(block, 'ctd_roi_xyxy', None), image_shape)
    if ctd_roi is not None:
        return ctd_roi

    return _expand_bbox(
        getattr(block, 'xyxy', None),
        image_shape,
        width_ratio=0.10,
        height_ratio=0.10,
        min_pad=8,
        max_pad=16,
    )


def resolve_block_mask_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    return resolve_block_ctd_roi(block, image_shape)


def resolve_block_residue_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    text_class = getattr(block, 'text_class', '') or ''
    if text_class == 'text_bubble':
        return resolve_block_cleanup_roi(block, image_shape)
    return resolve_block_ctd_roi(block, image_shape)


def assign_mask_rois(blocks: Iterable, image_shape: tuple[int, int] | tuple[int, int, int]) -> None:
    for block in blocks or []:
        ctd_roi = resolve_block_ctd_roi(block, image_shape)
        cleanup_roi = resolve_block_cleanup_roi(block, image_shape)
        block.ctd_roi_xyxy = list(ctd_roi) if ctd_roi is not None else None
        block.cleanup_roi_xyxy = list(cleanup_roi) if cleanup_roi is not None else None
        block.mask_roi_xyxy = list(ctd_roi) if ctd_roi is not None else None


def _clip_box_to_roi(box, roi: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
    rx1, ry1, rx2, ry2 = roi
    x1 = max(rx1, min(x1, rx2))
    y1 = max(ry1, min(y1, ry2))
    x2 = max(rx1, min(x2, rx2))
    y2 = max(ry1, min(y2, ry2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def build_text_prior_mask(
    image_rgb: np.ndarray,
    block,
    roi: tuple[int, int, int, int],
    *,
    dilate_iterations: int = 1,
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    prior = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
    if getattr(block, 'xyxy', None) is None:
        return prior

    prior_boxes = getattr(block, 'inpaint_bboxes', None)
    if prior_boxes is None:
        prior_boxes = get_inpaint_bboxes(block.xyxy, image_rgb, bubble_bbox=roi)

    for box in prior_boxes or []:
        clipped = _clip_box_to_roi(box, roi)
        if clipped is None:
            continue
        bx1, by1, bx2, by2 = clipped
        prior[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = 255

    if not np.any(prior):
        fallback = _clip_box_to_roi(
            [
                int(block.xyxy[0]) - 2,
                int(block.xyxy[1]) - 2,
                int(block.xyxy[2]) + 2,
                int(block.xyxy[3]) + 2,
            ],
            roi,
        )
        if fallback is not None:
            bx1, by1, bx2, by2 = fallback
            prior[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1] = 255

    if np.any(prior) and dilate_iterations > 0:
        prior = cv2.dilate(prior, np.ones((3, 3), np.uint8), iterations=int(dilate_iterations))
    return np.where(prior > 0, 255, 0).astype(np.uint8)


def get_mask_roi_type(block) -> str:
    roi = getattr(block, 'ctd_roi_xyxy', None) or getattr(block, 'mask_roi_xyxy', None)
    if roi is None:
        roi = getattr(block, 'carrier_mask_roi_xyxy', None)
    if roi is None:
        return 'none'
    text_class = getattr(block, 'text_class', '') or ''
    if text_class == 'text_bubble' and (getattr(block, 'carrier_kind', '') or '') == CARRIER_KIND_CAPTION_PLATE:
        return 'caption_plate_local'
    if text_class == 'text_bubble' and getattr(block, 'bubble_xyxy', None) is not None:
        return 'bubble'
    return 'synthetic_text_free'
