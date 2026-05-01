from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from modules.utils.inpaint_envelope import (
    build_text_free_erase_envelope,
    mask_for_xyxy,
    merge_close_components_within_envelope,
)
from modules.utils.textblock import TextBlock


def _clip_xyxy(xyxy, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
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


def _expand_xyxy(
    xyxy: tuple[int, int, int, int] | None,
    image_shape: tuple[int, ...],
    *,
    ratio: float,
    min_px: int,
    max_px: int,
) -> tuple[int, int, int, int] | None:
    if xyxy is None:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    pad_x = max(min_px, min(max_px, int(round(bw * ratio))))
    pad_y = max(min_px, min(max_px, int(round(bh * ratio))))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )


def _intersect_xyxy(
    lhs: tuple[int, int, int, int] | None,
    rhs: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    x1 = max(lhs[0], rhs[0])
    y1 = max(lhs[1], rhs[1])
    x2 = min(lhs[2], rhs[2])
    y2 = min(lhs[3], rhs[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bbox_mask_local(
    roi_shape: tuple[int, int],
    bbox: tuple[int, int, int, int] | None,
    roi_xyxy: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(roi_shape, dtype=np.uint8)
    if bbox is None:
        return mask
    rx1, ry1, rx2, ry2 = roi_xyxy
    bx1, by1, bx2, by2 = bbox
    ix1 = max(rx1, bx1)
    iy1 = max(ry1, by1)
    ix2 = min(rx2, bx2)
    iy2 = min(ry2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return mask
    mask[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = 255
    return mask


def _build_feature_masks(crop_rgb: np.ndarray) -> dict[str, np.ndarray]:
    rgb = crop_rgb.astype(np.uint8, copy=False)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    l_chan = lab[:, :, 0]
    high_l = np.percentile(l_chan, 80) if l_chan.size else 255
    bright_seed = np.where(l_chan >= high_l, 255, 0).astype(np.uint8)
    bright_adapt = cv2.adaptiveThreshold(
        clahe,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -5,
    )
    bright_core = cv2.bitwise_or(bright_seed, bright_adapt)

    inv_gray = 255 - gray
    dark_adapt = cv2.adaptiveThreshold(
        inv_gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        21,
        -3,
    )
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    nonzero = blackhat[blackhat > 0]
    blackhat_threshold = max(10, int(np.percentile(nonzero, 75))) if nonzero.size else 255
    dark_blackhat = np.where(blackhat >= blackhat_threshold, 255, 0).astype(np.uint8)
    dark_fringe = cv2.bitwise_or(dark_adapt, dark_blackhat)

    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_core = np.where(
        (saturation > 35)
        & (value > 60)
        & (
            (np.abs(lab[:, :, 1].astype(np.int16) - 128) > 10)
            | (np.abs(lab[:, :, 2].astype(np.int16) - 128) > 10)
        ),
        255,
        0,
    ).astype(np.uint8)

    seed = cv2.bitwise_or(bright_core, color_core)
    ring = cv2.subtract(
        cv2.dilate(seed, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1),
        seed,
    )
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    outline_support = cv2.bitwise_or(dark_fringe, np.where(gradient > 12, 255, 0).astype(np.uint8))
    outline_ring = cv2.bitwise_and(ring, outline_support)

    canny = cv2.Canny(clahe, 70, 140)
    return {
        "bright_core": bright_core,
        "dark_fringe": dark_fringe,
        "color_core": color_core,
        "outline_ring": outline_ring,
        "canny": np.where(canny > 0, 255, 0).astype(np.uint8),
    }


def _component_count(mask: np.ndarray) -> int:
    binary = np.where(mask > 0, 1, 0).astype(np.uint8)
    if not np.any(binary):
        return 0
    count, _labels = cv2.connectedComponents(binary, connectivity=8)
    return max(0, count - 1)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def build_block_rescue_mask(
    image_rgb: np.ndarray,
    block: TextBlock,
    legacy_block_mask: np.ndarray,
) -> dict[str, Any]:
    bbox = _clip_xyxy(getattr(block, "xyxy", None), image_rgb.shape)
    bubble = _clip_xyxy(getattr(block, "bubble_xyxy", None), image_rgb.shape)
    text_class = str(getattr(block, "text_class", "") or "")

    empty_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    result = {
        "applied": False,
        "reason_codes": [],
        "legacy_fill_ratio": 0.0,
        "rescue_fill_ratio": 0.0,
        "rescue_mask": empty_mask,
        "rescue_roi_xyxy": None,
        "metrics": {},
    }
    if bbox is None:
        return result

    x1, y1, x2, y2 = bbox
    bbox_area = max(1, (x2 - x1) * (y2 - y1))
    image_area = max(1, image_rgb.shape[0] * image_rgb.shape[1])
    aspect_ratio = max((x2 - x1) / max(1, y2 - y1), (y2 - y1) / max(1, x2 - x1))
    if _ratio(bbox_area, image_area) > 0.08:
        result["reason_codes"] = ["bbox_too_large"]
        return result
    if text_class == "text_free" and aspect_ratio > 12.0:
        result["reason_codes"] = ["text_free_extreme_aspect"]
        return result

    legacy_bbox_crop = np.where(legacy_block_mask[y1:y2, x1:x2] > 0, 255, 0).astype(np.uint8)
    legacy_fill_ratio = _ratio(np.count_nonzero(legacy_bbox_crop), bbox_area)
    result["legacy_fill_ratio"] = legacy_fill_ratio
    if legacy_fill_ratio >= 0.35:
        result["reason_codes"] = ["legacy_mask_already_dense"]
        return result

    bbox_crop = image_rgb[y1:y2, x1:x2]
    features = _build_feature_masks(bbox_crop)
    edge_density = _ratio(np.count_nonzero(features["canny"]), bbox_area)
    bright_core_density = _ratio(np.count_nonzero(features["bright_core"]), bbox_area)
    color_core_density = _ratio(np.count_nonzero(features["color_core"]), bbox_area)
    legacy_component_count = _component_count(legacy_bbox_crop)

    thresholds = {
        "fill_ratio": 0.12 if text_class == "text_bubble" else 0.08,
        "edge_density": 0.025 if text_class == "text_bubble" else 0.030,
        "bright_density": 0.018 if text_class == "text_bubble" else 0.020,
        "color_density": 0.012 if text_class == "text_bubble" else 0.015,
    }

    reasons: list[str] = []
    if legacy_fill_ratio < thresholds["fill_ratio"]:
        reasons.append("low_legacy_fill_ratio")
    if edge_density > thresholds["edge_density"]:
        reasons.append("edge_dense")
    if bright_core_density > thresholds["bright_density"]:
        reasons.append("bright_core_detected")
    if color_core_density > thresholds["color_density"]:
        reasons.append("color_core_detected")
    if legacy_component_count <= 2 and min(x2 - x1, y2 - y1) >= 18:
        reasons.append("few_legacy_components")

    hard_box = (
        "low_legacy_fill_ratio" in reasons
        and any(
            code in reasons
            for code in ("edge_dense", "bright_core_detected", "color_core_detected", "few_legacy_components")
        )
    )

    result["metrics"] = {
        "bbox_area": bbox_area,
        "image_area": image_area,
        "aspect_ratio": aspect_ratio,
        "edge_density": edge_density,
        "bright_core_density": bright_core_density,
        "color_core_density": color_core_density,
        "legacy_component_count": legacy_component_count,
    }
    result["reason_codes"] = reasons
    if not hard_box:
        return result

    if text_class == "text_bubble":
        rescue_roi = _expand_xyxy(bbox, image_rgb.shape, ratio=0.08, min_px=4, max_px=12)
        rescue_roi = _intersect_xyxy(rescue_roi, bubble)
    elif text_class == "text_free":
        rescue_roi = build_text_free_erase_envelope(block, image_rgb.shape, residue_risk=True)
    else:
        rescue_roi = _expand_xyxy(bbox, image_rgb.shape, ratio=0.06, min_px=3, max_px=8)
    if rescue_roi is None:
        return result

    rx1, ry1, rx2, ry2 = rescue_roi
    rescue_crop = image_rgb[ry1:ry2, rx1:rx2]
    rescue_features = _build_feature_masks(rescue_crop)
    rescue_raw = cv2.bitwise_or(
        cv2.bitwise_or(rescue_features["bright_core"], rescue_features["color_core"]),
        rescue_features["outline_ring"],
    )
    legacy_local = np.where(legacy_block_mask[ry1:ry2, rx1:rx2] > 0, 255, 0).astype(np.uint8)
    bbox_local = _bbox_mask_local(rescue_raw.shape, bbox, rescue_roi)
    prior_mask = cv2.bitwise_or(legacy_local, bbox_local)

    binary = np.where(rescue_raw > 0, 1, 0).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(rescue_raw, dtype=np.uint8)
    roi_area = max(1, rescue_raw.shape[0] * rescue_raw.shape[1])

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < 6:
            continue

        cx = int(stats[label, cv2.CC_STAT_LEFT])
        cy = int(stats[label, cv2.CC_STAT_TOP])
        cw = int(stats[label, cv2.CC_STAT_WIDTH])
        ch = int(stats[label, cv2.CC_STAT_HEIGHT])
        bbox_fill_ratio = _ratio(area, max(1, cw * ch))
        if bbox_fill_ratio < 0.03 or bbox_fill_ratio > 0.85:
            continue

        component = labels == label
        overlap_pixels = int(np.count_nonzero(component & (prior_mask > 0)))
        if overlap_pixels < 10:
            continue

        comp_bbox_area = max(1, cw * ch)
        if cw > rescue_raw.shape[1] * 0.70 or ch > rescue_raw.shape[0] * 0.70:
            continue

        touches_edge = cx == 0 or cy == 0 or (cx + cw) >= rescue_raw.shape[1] or (cy + ch) >= rescue_raw.shape[0]
        comp_aspect_ratio = max(cw / max(1, ch), ch / max(1, cw))
        if text_class == "text_bubble":
            if _ratio(comp_bbox_area, roi_area) > 0.35:
                continue
            if touches_edge and _ratio(overlap_pixels, area) < 0.20:
                continue
        else:
            if _ratio(comp_bbox_area, roi_area) > 0.18:
                continue
            if touches_edge:
                continue
            if comp_aspect_ratio > 10.0:
                continue

        if touches_edge and _ratio(comp_bbox_area, roi_area) > 0.10:
            continue

        filtered[component] = 255

    if text_class == "text_bubble" and np.any(filtered):
        filtered = cv2.dilate(
            filtered,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
            iterations=1,
        )

    if text_class == "text_free":
        envelope_local = mask_for_xyxy(rescue_raw.shape, (0, 0, rx2 - rx1, ry2 - ry1))
        filtered = cv2.bitwise_or(filtered, envelope_local)
        filtered = merge_close_components_within_envelope(
            filtered,
            (0, 0, rx2 - rx1, ry2 - ry1),
            kernel_size=5,
        )

    rescue_mask = np.zeros_like(empty_mask, dtype=np.uint8)
    rescue_mask[ry1:ry2, rx1:rx2] = filtered
    if text_class == "text_bubble" and bubble is not None:
        bx1, by1, bx2, by2 = bubble
        bubble_mask = np.zeros_like(empty_mask, dtype=np.uint8)
        bubble_mask[by1:by2, bx1:bx2] = 255
        rescue_mask = cv2.bitwise_and(rescue_mask, bubble_mask)

    rescue_fill_ratio = _ratio(np.count_nonzero(rescue_mask[y1:y2, x1:x2]), bbox_area)
    result.update(
        {
            "applied": bool(np.any(rescue_mask)),
            "rescue_fill_ratio": rescue_fill_ratio,
            "rescue_mask": rescue_mask,
            "rescue_roi_xyxy": rescue_roi,
        }
    )
    return result
