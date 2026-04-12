from __future__ import annotations

from typing import Any

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import get_inpaint_bboxes
from modules.masking.legacy_bbox_rescue import build_block_rescue_mask
from modules.utils.textblock import TextBlock


LONG_EDGE = 2048


def _build_exact_legacy_block_mask(
    img: np.ndarray,
    blk: TextBlock,
    *,
    default_padding: int = 5,
    roi_override: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w, _ = img.shape
    if getattr(blk, "xyxy", None) is None:
        return np.zeros((h, w), dtype=np.uint8), None

    roi = roi_override if roi_override is not None else getattr(blk, "bubble_xyxy", None)
    bboxes = get_inpaint_bboxes(
        blk.xyxy,
        img,
        bubble_bbox=roi,
    )
    blk.inpaint_bboxes = bboxes
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((h, w), dtype=np.uint8), bboxes

    xs = [x for x1, _, x2, _ in bboxes for x in (x1, x2)]
    ys = [y for _, y1, _, y2 in bboxes for y in (y1, y2)]
    min_x, max_x = int(min(xs)), int(max(xs))
    min_y, max_y = int(min(ys)), int(max(ys))
    roi_w, roi_h = max_x - min_x + 1, max_y - min_y + 1

    ds = max(1.0, max(roi_w, roi_h) / LONG_EDGE)
    mw, mh = int(roi_w / ds) + 2, int(roi_h / ds) + 2
    pad_offset = 1

    small = np.zeros((mh, mw), dtype=np.uint8)
    for x1, y1, x2, y2 in bboxes:
        x1i = int((x1 - min_x) / ds) + pad_offset
        y1i = int((y1 - min_y) / ds) + pad_offset
        x2i = int((x2 - min_x) / ds) + pad_offset
        y2i = int((y2 - min_y) / ds) + pad_offset
        small = imk.rectangle(small, (x1i, y1i), (x2i, y2i), 255, -1)

    kernel = imk.get_structuring_element(imk.MORPH_RECT, (15, 15))
    closed = imk.morphology_ex(small, imk.MORPH_CLOSE, kernel)
    contours, _ = imk.find_contours(closed)
    if not contours:
        return np.zeros((h, w), dtype=np.uint8), bboxes

    polys = []
    for cnt in contours:
        pts = cnt.squeeze(1)
        if pts.ndim != 2 or pts.shape[0] < 3:
            continue
        pts_f = (pts.astype(np.float32) - pad_offset) * ds
        pts_f[:, 0] += min_x
        pts_f[:, 1] += min_y
        polys.append(pts_f.astype(np.int32))
    if not polys:
        return np.zeros((h, w), dtype=np.uint8), bboxes

    block_mask = np.zeros((h, w), dtype=np.uint8)
    block_mask = imk.fill_poly(block_mask, polys, 255)

    kernel_size = max(default_padding, 5)
    widths = [max(1, x2 - x1) for x1, _, x2, _ in bboxes]
    heights = [max(1, y2 - y1) for _, y1, _, y2 in bboxes]
    median_span = max(float(np.median(widths)), float(np.median(heights)))
    dynamic_padding = int(round(median_span * 0.08))
    kernel_size = max(kernel_size, min(9, max(3, dynamic_padding)))

    if getattr(blk, "text_class", None) == "text_bubble" and roi is not None:
        rx1, ry1, rx2, ry2 = [int(float(v)) for v in roi[:4]]
        valid = [
            p for p in polys
            if (p[:, 0] >= rx1).all() and (p[:, 0] <= rx2).all() and (p[:, 1] >= ry1).all() and (p[:, 1] <= ry2).all()
        ]
        if valid:
            dists = []
            for p in valid:
                dists.extend([
                    p[:, 0].min() - rx1,
                    rx2 - p[:, 0].max(),
                    p[:, 1].min() - ry1,
                    ry2 - p[:, 1].max(),
                ])
            min_dist = min(dists)
            if kernel_size >= min_dist:
                kernel_size = max(1, int(min_dist * 0.8))

    dil_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = imk.dilate(block_mask, dil_kernel, iterations=4)
    if roi is not None:
        rx1, ry1, rx2, ry2 = [int(float(v)) for v in roi[:4]]
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        roi_mask[ry1:ry2, rx1:rx2] = 255
        dilated = cv2.bitwise_and(dilated, roi_mask)
    return np.where(dilated > 0, 255, 0).astype(np.uint8), bboxes



def build_exact_legacy_bbox_mask(img: np.ndarray, blk_list: list[TextBlock], default_padding: int = 5) -> np.ndarray:
    """Exact bbox-only mask flow from commit 1aec275."""
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    for blk in blk_list:
        block_mask, _bboxes = _build_exact_legacy_block_mask(img, blk, default_padding=default_padding)
        mask = np.bitwise_or(mask, block_mask)

    return np.where(mask > 0, 255, 0).astype(np.uint8)



def build_rtdetr_legacy_bbox_mask(
    img: np.ndarray,
    blk_list: list[TextBlock],
    cfg: dict[str, Any] | None = None,
    *,
    default_padding: int = 5,
) -> dict[str, Any]:
    h, w, _ = img.shape
    legacy_base_mask = np.zeros((h, w), dtype=np.uint8)
    hard_box_rescue_mask = np.zeros((h, w), dtype=np.uint8)
    hard_box_applied_count = 0
    hard_box_reason_totals: dict[str, int] = {}

    for index, blk in enumerate(blk_list):
        block_base_mask, _bboxes = _build_exact_legacy_block_mask(img, blk, default_padding=default_padding)
        legacy_base_mask = np.bitwise_or(legacy_base_mask, block_base_mask)

        rescue = build_block_rescue_mask(img, blk, block_base_mask)
        setattr(blk, "_hard_box_applied", bool(rescue["applied"]))
        setattr(blk, "_hard_box_reason_codes", list(rescue["reason_codes"]))
        setattr(blk, "_legacy_fill_ratio", float(rescue["legacy_fill_ratio"]))
        setattr(blk, "_rescue_fill_ratio", float(rescue["rescue_fill_ratio"]))
        setattr(blk, "_hard_box_rescue_roi_xyxy", rescue.get("rescue_roi_xyxy"))
        setattr(blk, "_hard_box_index", int(index))
        setattr(blk, "_hard_box_metrics", dict(rescue.get("metrics", {})))

        for code in rescue["reason_codes"]:
            hard_box_reason_totals[code] = hard_box_reason_totals.get(code, 0) + 1

        if rescue["applied"]:
            hard_box_applied_count += 1
            hard_box_rescue_mask = np.bitwise_or(hard_box_rescue_mask, rescue["rescue_mask"])

    final_mask = np.bitwise_or(legacy_base_mask, hard_box_rescue_mask)
    zeros = np.zeros_like(final_mask, dtype=np.uint8)
    mode = str((cfg or {}).get("mask_inpaint_mode", "rtdetr_legacy_bbox_source_lama") or "rtdetr_legacy_bbox_source_lama")
    return {
        "raw_mask": final_mask.copy(),
        "refined_mask": final_mask.copy(),
        "protect_mask": zeros,
        "final_mask_pre_expand": final_mask.copy(),
        "final_mask_post_expand": final_mask.copy(),
        "final_mask": final_mask.copy(),
        "legacy_base_mask": legacy_base_mask.copy(),
        "hard_box_rescue_mask": hard_box_rescue_mask.copy(),
        "hard_box_applied_count": int(hard_box_applied_count),
        "hard_box_reason_totals": dict(hard_box_reason_totals),
        "mask_refiner": "legacy_bbox",
        "keep_existing_lines": False,
        "refiner_backend": "legacy_bbox_rescue",
        "refiner_device": "cpu",
        "fallback_used": False,
        "mask_inpaint_mode": mode,
    }
