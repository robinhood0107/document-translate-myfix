from __future__ import annotations

from typing import Any

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import get_inpaint_bboxes
from modules.masking.legacy_bbox_rescue import build_block_rescue_mask
from modules.utils.carrier import (
    CARRIER_KIND_CAPTION_PLATE,
    resolve_caption_plate_mask_roi,
    resolve_effective_text_bubble_roi,
)
from modules.utils.mask_inpaint_mode import DEFAULT_MASK_INPAINT_MODE
from modules.utils.textblock import TextBlock


LONG_EDGE = 2048


def _normalize_xyxy(xyxy, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    if xyxy is None or len(xyxy) < 4:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in xyxy[:4]]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _mask_for_xyxy(image_shape: tuple[int, ...], xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if xyxy is None:
        return mask
    x1, y1, x2, y2 = xyxy
    mask[y1:y2, x1:x2] = 255
    return mask


def _clip_box_to_roi(
    box: tuple[int, int, int, int] | list[int] | np.ndarray | None,
    roi: tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
) -> tuple[int, int, int, int] | None:
    norm = _normalize_xyxy(box, image_shape)
    if norm is None:
        return None
    if roi is None:
        return norm
    x1, y1, x2, y2 = norm
    rx1, ry1, rx2, ry2 = roi
    ix1 = max(x1, rx1)
    iy1 = max(y1, ry1)
    ix2 = min(x2, rx2)
    iy2 = min(y2, ry2)
    if ix2 <= ix1 or iy2 <= iy1:
        return None
    return ix1, iy1, ix2, iy2


def _build_caption_plate_base_block_mask(
    img: np.ndarray,
    blk: TextBlock,
    *,
    roi: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w, _ = img.shape
    if getattr(blk, "xyxy", None) is None:
        return np.zeros((h, w), dtype=np.uint8), None

    bboxes = get_inpaint_bboxes(
        blk.xyxy,
        img,
        bubble_bbox=roi,
    )
    blk.inpaint_bboxes = bboxes
    if bboxes is None or len(bboxes) == 0:
        return np.zeros((h, w), dtype=np.uint8), bboxes

    raw_mask = np.zeros((h, w), dtype=np.uint8)
    spans: list[int] = []
    valid_boxes = 0
    for box in bboxes:
        clipped = _clip_box_to_roi(box, roi, img.shape)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        spans.extend((bw, bh))
        pad = 1 if max(bw, bh) <= 16 else 2
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(w, x2 + pad)
        py2 = min(h, y2 + pad)
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
            px1 = max(px1, rx1)
            py1 = max(py1, ry1)
            px2 = min(px2, rx2)
            py2 = min(py2, ry2)
        if px2 <= px1 or py2 <= py1:
            continue
        raw_mask[py1:py2, px1:px2] = 255
        valid_boxes += 1

    if valid_boxes <= 0 or not np.any(raw_mask):
        return np.zeros((h, w), dtype=np.uint8), bboxes

    median_span = float(np.median(spans)) if spans else 6.0
    kernel_side = 3 if median_span < 18.0 else 5
    soft_mask = cv2.morphologyEx(
        raw_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_side, kernel_side)),
        iterations=1,
    )
    soft_mask = cv2.dilate(
        soft_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )

    if roi is not None:
        roi_mask = _mask_for_xyxy(img.shape, roi)
        raw_mask = cv2.bitwise_and(raw_mask, roi_mask)
        soft_mask = cv2.bitwise_and(soft_mask, roi_mask)
        roi_area = max(1, (roi[2] - roi[0]) * (roi[3] - roi[1]))
    else:
        roi_area = max(1, h * w)

    raw_fill_ratio = float(np.count_nonzero(raw_mask)) / float(roi_area)
    soft_fill_ratio = float(np.count_nonzero(soft_mask)) / float(roi_area)
    if soft_fill_ratio > 0.22 or (soft_fill_ratio > raw_fill_ratio * 1.65 and soft_fill_ratio > 0.14):
        return np.where(raw_mask > 0, 255, 0).astype(np.uint8), bboxes
    return np.where(soft_mask > 0, 255, 0).astype(np.uint8), bboxes


def _build_legacy_base_block_mask(
    img: np.ndarray,
    blk: TextBlock,
    *,
    default_padding: int = 5,
    roi_override: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    h, w, _ = img.shape
    if getattr(blk, "xyxy", None) is None:
        return np.zeros((h, w), dtype=np.uint8), None

    roi = roi_override
    if roi is None and getattr(blk, "text_class", None) == "text_bubble":
        roi = resolve_effective_text_bubble_roi(blk, img.shape)
    if (getattr(blk, "carrier_kind", "") or "") == CARRIER_KIND_CAPTION_PLATE:
        return _build_caption_plate_base_block_mask(
            img,
            blk,
            roi=roi,
        )
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
        rx1, ry1, rx2, ry2 = roi
        valid = [
            poly
            for poly in polys
            if (poly[:, 0] >= rx1).all()
            and (poly[:, 0] <= rx2).all()
            and (poly[:, 1] >= ry1).all()
            and (poly[:, 1] <= ry2).all()
        ]
        if valid:
            dists = []
            for poly in valid:
                dists.extend(
                    [
                        poly[:, 0].min() - rx1,
                        rx2 - poly[:, 0].max(),
                        poly[:, 1].min() - ry1,
                        ry2 - poly[:, 1].max(),
                    ]
                )
            min_dist = min(dists)
            if kernel_size >= min_dist:
                kernel_size = max(1, int(min_dist * 0.8))

    dil_kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = imk.dilate(block_mask, dil_kernel, iterations=4)
    if roi is not None:
        dilated = cv2.bitwise_and(dilated, _mask_for_xyxy(img.shape, roi))
    return np.where(dilated > 0, 255, 0).astype(np.uint8), bboxes


def _resolve_text_free_local_roi(
    block: TextBlock,
    image_shape: tuple[int, ...],
    rescue_roi_xyxy: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if rescue_roi_xyxy is not None:
        return rescue_roi_xyxy
    return _normalize_xyxy(getattr(block, "xyxy", None), image_shape)


def merge_legacy_and_rescue(
    image_shape: tuple[int, ...],
    block: TextBlock,
    legacy_block_mask: np.ndarray,
    rescue_mask: np.ndarray,
    rescue_roi_xyxy: tuple[int, int, int, int] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    legacy_block_mask = np.where(np.asarray(legacy_block_mask) > 0, 255, 0).astype(np.uint8)
    rescue_mask = np.where(np.asarray(rescue_mask) > 0, 255, 0).astype(np.uint8)
    final_block_mask = np.bitwise_or(legacy_block_mask, rescue_mask)

    text_class = str(getattr(block, "text_class", "") or "")
    clamp_xyxy: tuple[int, int, int, int] | None = None
    if text_class == "text_bubble":
        clamp_xyxy = resolve_effective_text_bubble_roi(block, image_shape)
    elif text_class == "text_free":
        clamp_xyxy = _resolve_text_free_local_roi(block, image_shape, rescue_roi_xyxy)

    if clamp_xyxy is not None:
        clamp_mask = _mask_for_xyxy(image_shape, clamp_xyxy)
        legacy_block_mask = cv2.bitwise_and(legacy_block_mask, clamp_mask)
        rescue_mask = cv2.bitwise_and(rescue_mask, clamp_mask)
        final_block_mask = cv2.bitwise_and(final_block_mask, clamp_mask)

    return legacy_block_mask, rescue_mask, final_block_mask


def build_legacy_bbox_mask_details(
    img: np.ndarray,
    blk_list: list[TextBlock],
    cfg: dict[str, Any] | None = None,
    *,
    default_padding: int = 5,
) -> dict[str, Any]:
    h, w, _ = img.shape
    legacy_base_mask = np.zeros((h, w), dtype=np.uint8)
    hard_box_rescue_mask = np.zeros((h, w), dtype=np.uint8)
    final_mask = np.zeros((h, w), dtype=np.uint8)
    hard_box_applied_count = 0
    hard_box_reason_totals: dict[str, int] = {}

    for index, blk in enumerate(blk_list):
        carrier_mask_roi_xyxy = resolve_caption_plate_mask_roi(blk, img.shape)
        if (getattr(blk, "carrier_kind", "") or "") == CARRIER_KIND_CAPTION_PLATE:
            setattr(blk, "carrier_mask_roi_xyxy", list(carrier_mask_roi_xyxy) if carrier_mask_roi_xyxy is not None else None)
        block_base_mask, _bboxes = _build_legacy_base_block_mask(
            img,
            blk,
            default_padding=default_padding,
            roi_override=carrier_mask_roi_xyxy if carrier_mask_roi_xyxy is not None else None,
        )
        rescue = build_block_rescue_mask(img, blk, block_base_mask)
        block_base_mask, block_rescue_mask, final_block_mask = merge_legacy_and_rescue(
            img.shape,
            blk,
            block_base_mask,
            rescue["rescue_mask"],
            rescue.get("rescue_roi_xyxy"),
        )

        legacy_base_mask = np.bitwise_or(legacy_base_mask, block_base_mask)
        hard_box_rescue_mask = np.bitwise_or(hard_box_rescue_mask, block_rescue_mask)
        final_mask = np.bitwise_or(final_mask, final_block_mask)

        setattr(blk, "_hard_box_applied", bool(rescue["applied"]))
        setattr(blk, "_hard_box_reason_codes", list(rescue["reason_codes"]))
        setattr(blk, "_legacy_fill_ratio", float(rescue["legacy_fill_ratio"]))
        setattr(blk, "_rescue_fill_ratio", float(rescue["rescue_fill_ratio"]))
        setattr(blk, "_hard_box_rescue_roi_xyxy", rescue.get("rescue_roi_xyxy"))
        setattr(blk, "_hard_box_index", int(index))
        setattr(blk, "_hard_box_metrics", dict(rescue.get("metrics", {})))
        setattr(blk, "_legacy_mask_pixel_count", int(np.count_nonzero(block_base_mask)))
        setattr(blk, "_rescue_mask_pixel_count", int(np.count_nonzero(block_rescue_mask)))
        setattr(blk, "_final_mask_pixel_count", int(np.count_nonzero(final_block_mask)))

        for code in rescue["reason_codes"]:
            hard_box_reason_totals[code] = hard_box_reason_totals.get(code, 0) + 1
        if rescue["applied"]:
            hard_box_applied_count += 1

    mode = str((cfg or {}).get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE) or DEFAULT_MASK_INPAINT_MODE)
    zeros = np.zeros_like(final_mask, dtype=np.uint8)
    return {
        "raw_mask": legacy_base_mask.copy(),
        "refined_mask": final_mask.copy(),
        "protect_mask": zeros,
        "final_mask_pre_expand": final_mask.copy(),
        "final_mask_post_expand": final_mask.copy(),
        "final_mask": final_mask.copy(),
        "legacy_base_mask": legacy_base_mask.copy(),
        "hard_box_rescue_mask": hard_box_rescue_mask.copy(),
        "hard_box_applied_count": int(hard_box_applied_count),
        "hard_box_reason_totals": dict(hard_box_reason_totals),
        "legacy_base_mask_pixel_count": int(np.count_nonzero(legacy_base_mask)),
        "hard_box_rescue_mask_pixel_count": int(np.count_nonzero(hard_box_rescue_mask)),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
        "mask_refiner": "legacy_bbox",
        "keep_existing_lines": False,
        "refiner_backend": "legacy_bbox_rescue",
        "refiner_device": "cpu",
        "fallback_used": False,
        "mask_inpaint_mode": mode,
    }
