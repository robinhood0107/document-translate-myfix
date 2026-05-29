from __future__ import annotations

from typing import Any, Iterable

import cv2
import imkit as imk
import numpy as np

from modules.utils.inpaint_envelope import build_text_free_erase_envelope, normalize_xyxy

TEXT_FREE_COVERAGE_THRESHOLD = 0.08
TEXT_FREE_MAX_IMAGE_AREA_RATIO = 0.08
TEXT_FREE_MAX_ASPECT_RATIO = 12.0
TEXT_FREE_MAX_RESCUE_FILL_RATIO = 0.35
TEXT_FREE_MIN_RESCUE_FILL_RATIO = 0.01


def _empty_mask_like(mask: np.ndarray) -> np.ndarray:
    return np.zeros(mask.shape[:2], dtype=np.uint8)


def _ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _bbox_area(xyxy: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = xyxy
    return max(1, int(x2 - x1) * int(y2 - y1))


def _bbox_aspect_ratio(xyxy: tuple[int, int, int, int]) -> float:
    x1, y1, x2, y2 = xyxy
    width = max(1, int(x2 - x1))
    height = max(1, int(y2 - y1))
    return max(width / height, height / width)


def _touches_image_edge(xyxy: tuple[int, int, int, int], image_shape: tuple[int, ...]) -> bool:
    h, w = image_shape[:2]
    x1, y1, x2, y2 = xyxy
    return x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h


def _local_bbox_mask(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    rx1, ry1, rx2, ry2 = roi
    x1, y1, x2, y2 = bbox
    ix1 = max(rx1, min(x1, rx2))
    iy1 = max(ry1, min(y1, ry2))
    ix2 = max(rx1, min(x2, rx2))
    iy2 = max(ry1, min(y2, ry2))
    if ix2 <= ix1 or iy2 <= iy1:
        return mask
    mask[iy1 - ry1:iy2 - ry1, ix1 - rx1:ix2 - rx1] = 255
    return mask


def _candidate_stroke_mask(crop_rgb: np.ndarray) -> np.ndarray:
    if crop_rgb is None or crop_rgb.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    gray = imk.to_gray(crop_rgb)
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)
    gray = gray.astype(np.uint8, copy=False)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gradient = cv2.morphologyEx(
        clahe,
        cv2.MORPH_GRADIENT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    edges = cv2.Canny(clahe, 70, 150)
    edge_support = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)

    bright_cut = max(180, int(np.percentile(gray, 82)))
    dark_cut = min(90, int(np.percentile(gray, 18)))
    bright = np.where(gray >= bright_cut, 255, 0).astype(np.uint8)
    dark = np.where(gray <= dark_cut, 255, 0).astype(np.uint8)
    contrast = cv2.bitwise_or(bright, dark)
    contrast = cv2.bitwise_and(
        contrast,
        np.where((edge_support > 0) | (gradient > 12), 255, 0).astype(np.uint8),
    )
    contrast = cv2.morphologyEx(
        contrast,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )

    binary = np.where(contrast > 0, 1, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(gray, dtype=np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(gray, dtype=np.uint8)
    crop_area = max(1, int(gray.shape[0]) * int(gray.shape[1]))
    for label in range(1, num_labels):
        x, y, w, h, area = [int(v) for v in stats[label]]
        if area < 6 or w <= 0 or h <= 0:
            continue
        comp_bbox_area = max(1, w * h)
        if _ratio(comp_bbox_area, crop_area) > 0.24:
            continue
        fill_ratio = _ratio(area, comp_bbox_area)
        if fill_ratio < 0.02 or fill_ratio > 0.85:
            continue
        aspect = max(w / max(1, h), h / max(1, w))
        if aspect > 12.0:
            continue
        touches_edge = x <= 0 or y <= 0 or (x + w) >= gray.shape[1] or (y + h) >= gray.shape[0]
        if touches_edge and _ratio(comp_bbox_area, crop_area) > 0.08:
            continue
        filtered[labels == label] = 255

    if not np.any(filtered):
        return filtered
    filtered = cv2.morphologyEx(
        filtered,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1,
    )
    filtered = cv2.dilate(
        filtered,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        iterations=1,
    )
    return np.where(filtered > 0, 255, 0).astype(np.uint8)


def _restrict_components_to_bbox_prior(
    candidate_mask: np.ndarray,
    bbox_local: np.ndarray,
) -> np.ndarray:
    if candidate_mask is None or candidate_mask.size == 0 or not np.any(candidate_mask):
        return np.zeros_like(candidate_mask, dtype=np.uint8)
    if bbox_local is None or bbox_local.size == 0 or not np.any(bbox_local):
        return np.zeros_like(candidate_mask, dtype=np.uint8)

    binary = np.where(candidate_mask > 0, 1, 0).astype(np.uint8)
    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    kept = np.zeros_like(candidate_mask, dtype=np.uint8)
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area <= 0:
            continue
        component = labels == label
        overlap = int(np.count_nonzero(component & (bbox_local > 0)))
        if overlap < 5:
            continue
        if _ratio(overlap, area) < 0.05:
            continue
        kept[component] = 255
    return kept


def _small_bbox_fallback_mask(
    shape: tuple[int, int],
    bbox_local: np.ndarray,
) -> np.ndarray:
    fallback = np.where(bbox_local > 0, 255, 0).astype(np.uint8)
    if not np.any(fallback):
        return fallback
    return cv2.dilate(
        fallback,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )[: shape[0], : shape[1]]


def apply_text_free_rescue_mask(
    image_rgb: np.ndarray,
    blocks: Iterable,
    final_mask: np.ndarray,
    *,
    coverage_threshold: float = TEXT_FREE_COVERAGE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, Any]]:
    mask = np.where(np.asarray(final_mask) > 0, 255, 0).astype(np.uint8)
    rescue_mask = _empty_mask_like(mask)
    reason_totals: dict[str, int] = {}
    applied_count = 0
    image_area = max(1, int(image_rgb.shape[0]) * int(image_rgb.shape[1]))

    for block in blocks or []:
        if str(getattr(block, "text_class", "") or "") != "text_free":
            continue

        reason_codes: list[str] = []
        bbox = normalize_xyxy(getattr(block, "xyxy", None), image_rgb.shape)
        setattr(block, "_text_free_mask_coverage", 0.0)
        setattr(block, "_text_free_rescue_applied", False)
        setattr(block, "_text_free_rescue_reason_codes", [])
        setattr(block, "_text_free_rescue_pixel_count", 0)
        setattr(block, "_text_free_rescue_fill_ratio", 0.0)
        setattr(block, "_text_free_rescue_roi_xyxy", None)
        setattr(block, "_text_free_rescue_metrics", {})
        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox
        bbox_area = _bbox_area(bbox)
        bbox_mask_pixels = int(np.count_nonzero(mask[y1:y2, x1:x2] > 0))
        coverage = _ratio(bbox_mask_pixels, bbox_area)
        setattr(block, "_text_free_mask_coverage", coverage)

        area_ratio = _ratio(bbox_area, image_area)
        aspect_ratio = _bbox_aspect_ratio(bbox)
        metrics = {
            "bbox_area": bbox_area,
            "image_area": image_area,
            "bbox_area_ratio": area_ratio,
            "aspect_ratio": aspect_ratio,
            "initial_mask_pixels": bbox_mask_pixels,
        }

        if coverage >= coverage_threshold:
            reason_codes.append("coverage_ok")
        if area_ratio > TEXT_FREE_MAX_IMAGE_AREA_RATIO:
            reason_codes.append("bbox_too_large")
        if aspect_ratio > TEXT_FREE_MAX_ASPECT_RATIO:
            reason_codes.append("text_free_extreme_aspect")
        if _touches_image_edge(bbox, image_rgb.shape):
            reason_codes.append("touches_image_edge")
        if reason_codes:
            setattr(block, "_text_free_rescue_reason_codes", reason_codes)
            setattr(block, "_text_free_rescue_metrics", metrics)
            for code in reason_codes:
                reason_totals[code] = reason_totals.get(code, 0) + 1
            continue

        reason_codes.append("low_ctd_coverage")
        envelope = build_text_free_erase_envelope(block, image_rgb.shape, residue_risk=False)
        if envelope is None:
            reason_codes.append("missing_envelope")
            setattr(block, "_text_free_rescue_reason_codes", reason_codes)
            setattr(block, "_text_free_rescue_metrics", metrics)
            for code in reason_codes:
                reason_totals[code] = reason_totals.get(code, 0) + 1
            continue

        rx1, ry1, rx2, ry2 = envelope
        crop = image_rgb[ry1:ry2, rx1:rx2]
        candidate = _candidate_stroke_mask(crop)
        bbox_local = _local_bbox_mask(candidate.shape, bbox, envelope)
        candidate = _restrict_components_to_bbox_prior(candidate, bbox_local)
        if not np.any(candidate) and coverage < 0.02:
            candidate = _small_bbox_fallback_mask(candidate.shape, bbox_local)
            reason_codes.append("small_bbox_fallback")

        local_bbox_pixels = int(np.count_nonzero(candidate[bbox_local > 0] > 0)) if np.any(candidate) else 0
        rescue_fill_ratio = _ratio(local_bbox_pixels, bbox_area)
        metrics.update(
            {
                "rescue_fill_ratio": rescue_fill_ratio,
                "rescue_pixel_count": int(np.count_nonzero(candidate > 0)),
            }
        )
        if rescue_fill_ratio <= 0.0:
            reason_codes.append("no_rescue_candidate")
        elif rescue_fill_ratio < TEXT_FREE_MIN_RESCUE_FILL_RATIO:
            reason_codes.append("rescue_too_sparse")
        elif rescue_fill_ratio > TEXT_FREE_MAX_RESCUE_FILL_RATIO:
            reason_codes.append("rescue_too_dense")

        if any(code in reason_codes for code in ("no_rescue_candidate", "rescue_too_sparse", "rescue_too_dense")):
            setattr(block, "_text_free_rescue_reason_codes", reason_codes)
            setattr(block, "_text_free_rescue_fill_ratio", rescue_fill_ratio)
            setattr(block, "_text_free_rescue_metrics", metrics)
            for code in reason_codes:
                reason_totals[code] = reason_totals.get(code, 0) + 1
            continue

        block_rescue = np.zeros_like(mask, dtype=np.uint8)
        block_rescue[ry1:ry2, rx1:rx2] = np.where(candidate > 0, 255, 0).astype(np.uint8)
        rescue_mask = cv2.bitwise_or(rescue_mask, block_rescue)
        mask = cv2.bitwise_or(mask, block_rescue)
        applied_count += 1
        reason_codes.append("contour_rescue_applied")
        setattr(block, "_text_free_rescue_applied", True)
        setattr(block, "_text_free_rescue_reason_codes", reason_codes)
        setattr(block, "_text_free_rescue_pixel_count", int(np.count_nonzero(block_rescue > 0)))
        setattr(block, "_text_free_rescue_fill_ratio", rescue_fill_ratio)
        setattr(block, "_text_free_rescue_roi_xyxy", list(envelope))
        setattr(block, "_text_free_rescue_metrics", metrics)
        for code in reason_codes:
            reason_totals[code] = reason_totals.get(code, 0) + 1

    return mask, {
        "text_free_rescue_mask": rescue_mask,
        "text_free_rescue_applied_count": int(applied_count),
        "text_free_rescue_reason_totals": reason_totals,
        "text_free_rescue_mask_pixel_count": int(np.count_nonzero(rescue_mask > 0)),
    }


def mark_text_free_inpaint_residuals(
    image_rgb: np.ndarray,
    blocks: Iterable,
    *,
    component_threshold: int = 8,
    area_ratio_threshold: float = 0.035,
) -> dict[str, Any]:
    checked_count = 0
    needs_review_count = 0
    reason_totals: dict[str, int] = {}

    for block in blocks or []:
        if str(getattr(block, "text_class", "") or "") != "text_free":
            continue
        setattr(block, "_inpaint_residual_status", "not_checked")
        setattr(block, "_inpaint_needs_review", False)
        setattr(block, "_inpaint_residual_component_count", 0)
        setattr(block, "_inpaint_residual_area_ratio", 0.0)
        setattr(block, "_inpaint_residual_reason_codes", [])
        bbox = normalize_xyxy(getattr(block, "xyxy", None), image_rgb.shape)
        if bbox is None:
            continue

        checked_count += 1
        x1, y1, x2, y2 = bbox
        crop = image_rgb[y1:y2, x1:x2]
        candidate = _candidate_stroke_mask(crop)
        binary = np.where(candidate > 0, 1, 0).astype(np.uint8)
        component_count = 0
        if np.any(binary):
            labels_count, _labels = cv2.connectedComponents(binary, connectivity=8)
            component_count = max(0, int(labels_count) - 1)
        area_ratio = _ratio(int(np.count_nonzero(candidate > 0)), _bbox_area(bbox))
        reason_codes: list[str] = []
        if component_count >= component_threshold:
            reason_codes.append("residual_component_count")
        if area_ratio >= area_ratio_threshold:
            reason_codes.append("residual_area_ratio")
        needs_review = bool(reason_codes)
        setattr(block, "_inpaint_residual_status", "needs_review" if needs_review else "clean")
        setattr(block, "_inpaint_needs_review", needs_review)
        setattr(block, "_inpaint_residual_component_count", int(component_count))
        setattr(block, "_inpaint_residual_area_ratio", float(area_ratio))
        setattr(block, "_inpaint_residual_reason_codes", reason_codes)
        if needs_review:
            needs_review_count += 1
            for code in reason_codes:
                reason_totals[code] = reason_totals.get(code, 0) + 1

    return {
        "inpaint_residual_checked_count": int(checked_count),
        "inpaint_needs_review_count": int(needs_review_count),
        "inpaint_residual_reason_totals": reason_totals,
    }
