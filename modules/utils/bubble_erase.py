from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
    normalize_edit_mask,
)
from modules.utils.mask_roi import build_text_prior_mask, normalize_xyxy


ERASE_MODE_TEXT_FREE_LAMA = "text_free_lama"
ERASE_MODE_BUBBLE_FLAT_FILL = "bubble_flat_fill"
ERASE_MODE_BUBBLE_GRADIENT_FILL = "bubble_gradient_fill"
ERASE_MODE_BUBBLE_TELEA = "bubble_telea"
ERASE_MODE_BUBBLE_LAMA_FALLBACK = "bubble_lama_fallback"
ERASE_MODE_BUBBLE_SKIPPED = "bubble_skipped"


@dataclass(slots=True)
class BubbleEraseBlockStats:
    mode: str
    edit_pixel_count: int = 0
    protect_pixel_count: int = 0
    skipped_reason: str = ""


@dataclass(slots=True)
class BubbleEraseResult:
    image: np.ndarray
    edit_mask: np.ndarray
    fallback_mask: np.ndarray
    expanded_bubble_mask: np.ndarray
    stats: dict


def set_block_erase_metadata(block, stats: BubbleEraseBlockStats) -> None:
    block._erase_mode = str(stats.mode or "")
    block._erase_edit_pixel_count = int(stats.edit_pixel_count or 0)
    block._erase_protect_pixel_count = int(stats.protect_pixel_count or 0)
    block._erase_skipped_reason = str(stats.skipped_reason or "")


def mask_pixel_count(mask: np.ndarray | None) -> int:
    if mask is None:
        return 0
    return int(np.count_nonzero(np.asarray(mask) > 0))


def _bubble_border_protect_mask(shape: tuple[int, int], width: int = 3) -> np.ndarray:
    h, w = shape
    protect = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return protect
    cv2.rectangle(protect, (0, 0), (w - 1, h - 1), 255, thickness=max(1, int(width)))
    return protect


def _bubble_interior_cap_mask(crop: np.ndarray, seed_mask: np.ndarray) -> np.ndarray:
    if crop.size == 0 or seed_mask.size == 0:
        return np.zeros(seed_mask.shape, dtype=np.uint8)
    try:
        from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask

        balloon_mask, _non_text_mask = extract_ballon_mask(crop, seed_mask)
    except Exception:
        return np.full(seed_mask.shape, 255, dtype=np.uint8)
    if balloon_mask is None or balloon_mask.size == 0:
        return np.full(seed_mask.shape, 255, dtype=np.uint8)
    if balloon_mask.shape[:2] != seed_mask.shape[:2]:
        balloon_mask = cv2.resize(balloon_mask, (seed_mask.shape[1], seed_mask.shape[0]), interpolation=cv2.INTER_NEAREST)
    cap = np.where(balloon_mask > 0, 255, 0).astype(np.uint8)
    area_ratio = float(np.count_nonzero(cap)) / float(max(1, cap.size))
    if area_ratio < 0.20:
        return np.full(seed_mask.shape, 255, dtype=np.uint8)
    if area_ratio < 0.995:
        cap = cv2.erode(cap, np.ones((3, 3), np.uint8), iterations=1)
    return np.where(cap > 0, 255, 0).astype(np.uint8)


def _line_art_protect_mask(crop: np.ndarray) -> np.ndarray:
    if crop.size == 0:
        return np.zeros(crop.shape[:2], dtype=np.uint8)
    gray = _to_gray(crop)
    h, w = gray.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((h, w), dtype=np.uint8)
    edges = cv2.Canny(gray, 40, 120)
    min_line_length = max(28, int(round(min(h, w) * 0.18)))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=max(18, min_line_length // 2),
        minLineLength=min_line_length,
        maxLineGap=6,
    )
    protect = np.zeros((h, w), dtype=np.uint8)
    if lines is None:
        return protect
    for line in lines[:, 0, :]:
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        if length < min_line_length:
            continue
        cv2.line(protect, (x1, y1), (x2, y2), 255, thickness=4, lineType=cv2.LINE_AA)
    return np.where(protect > 0, 255, 0).astype(np.uint8)


def _edit_mask_near_line_art(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
) -> bool:
    if bubble_roi is None:
        return False
    x1, y1, x2, y2 = bubble_roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    mask_crop = normalize_edit_mask(edit_mask, image_rgb.shape)[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(mask_crop):
        return False
    line_mask = _line_art_protect_mask(crop)
    if not np.any(line_mask):
        return False
    near_mask = cv2.dilate(mask_crop, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)), iterations=1)
    overlap = np.count_nonzero((line_mask > 0) & (near_mask > 0))
    return overlap >= max(8, int(np.count_nonzero(mask_crop) * 0.002))


def _line_art_protect_mask_for_roi(
    image_rgb: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
) -> np.ndarray:
    protect = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if bubble_roi is None:
        return protect
    x1, y1, x2, y2 = bubble_roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    if crop.size == 0:
        return protect
    protect[y1:y2, x1:x2] = _line_art_protect_mask(crop)
    return np.where(protect > 0, 255, 0).astype(np.uint8)


def _rule_like_component(width: int, height: int, area: int) -> bool:
    long_side = max(int(width), int(height))
    short_side = min(int(width), int(height))
    if long_side < 16 or short_side > 5:
        return False
    aspect = float(long_side) / float(max(1, short_side))
    fill_ratio = float(area) / float(max(1, int(width) * int(height)))
    return aspect >= 10.0 and fill_ratio >= 0.30


def _component_filtered_mask(
    candidate: np.ndarray,
    seed_gate: np.ndarray,
    *,
    max_bbox_ratio: float = 0.22,
    min_area: int = 3,
) -> np.ndarray:
    if candidate.size == 0 or not np.any(candidate) or not np.any(seed_gate):
        return np.zeros_like(candidate, dtype=np.uint8)
    roi_h, roi_w = candidate.shape[:2]
    roi_area = max(1, roi_h * roi_w)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    output = np.zeros_like(candidate, dtype=np.uint8)
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) < int(min_area) or w <= 0 or h <= 0:
            continue
        if int(w * h) > int(round(roi_area * max_bbox_ratio)):
            continue
        if _rule_like_component(int(w), int(h), int(area)):
            continue
        component = labels[y:y + h, x:x + w] == label_idx
        if not np.any(seed_gate[y:y + h, x:x + w][component] > 0):
            continue
        output[y:y + h, x:x + w][component] = 255
    return np.where(output > 0, 255, 0).astype(np.uint8)


def _non_boxy_seed_mask(seed: np.ndarray) -> np.ndarray:
    if seed.size == 0 or not np.any(seed):
        return np.zeros_like(seed, dtype=np.uint8)
    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (seed > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    output = np.zeros_like(seed, dtype=np.uint8)
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        bbox_area = max(1, int(w) * int(h))
        fill_ratio = float(area) / float(bbox_area)
        if bbox_area >= 64 and fill_ratio >= 0.78:
            continue
        component = labels[y:y + h, x:x + w] == label_idx
        output[y:y + h, x:x + w][component] = 255
    return np.where(output > 0, 255, 0).astype(np.uint8)


def build_bubble_residual_edit_mask(
    image_rgb: np.ndarray,
    source_mask: np.ndarray,
    block,
    *,
    seed_dilate_px: int = 8,
    final_dilate_px: int = 4,
    protect_line_art: bool = True,
) -> tuple[np.ndarray, BubbleEraseBlockStats]:
    edit_mask = np.zeros(image_rgb.shape[:2], dtype=np.uint8)
    if getattr(block, "text_class", "") != "text_bubble":
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="not_text_bubble")

    bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_rgb.shape)
    if bubble_roi is None:
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="missing_bubble_roi")

    x1, y1, x2, y2 = bubble_roi
    source = normalize_edit_mask(source_mask, image_rgb.shape)
    seed = source[y1:y2, x1:x2]
    if not np.any(seed):
        return edit_mask, BubbleEraseBlockStats(mode=ERASE_MODE_BUBBLE_SKIPPED, skipped_reason="empty_seed")

    crop = image_rgb[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop.astype(np.uint8)
    prior = build_text_prior_mask(image_rgb, block, bubble_roi, dilate_iterations=2)
    if not np.any(prior):
        prior = seed.copy()

    px = max(1, int(seed_dilate_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1), (px, px))
    seed_gate = cv2.dilate(seed, kernel, iterations=1)
    candidate_gate = np.where((seed_gate > 0) & (prior > 0), 255, 0).astype(np.uint8)
    interior_cap = _bubble_interior_cap_mask(crop, seed)
    line_protect = _line_art_protect_mask(crop) if bool(protect_line_art) else np.zeros_like(seed, dtype=np.uint8)
    protect = np.where(
        (_bubble_border_protect_mask(seed.shape, width=3) > 0)
        | (interior_cap <= 0)
        | (line_protect > 0),
        255,
        0,
    ).astype(np.uint8)

    safe_bg = gray[(candidate_gate <= 0) & (protect <= 0)]
    if safe_bg.size == 0:
        safe_bg = gray[protect <= 0]
    if safe_bg.size == 0:
        safe_bg = gray.reshape(-1)
    bg_median = float(np.median(safe_bg))
    bg_std = float(np.std(safe_bg))
    bright_threshold = min(245.0, bg_median + max(18.0, bg_std * 1.25))
    dark_threshold = max(10.0, bg_median - max(24.0, bg_std * 1.50))

    bright = np.where(gray >= bright_threshold, 255, 0).astype(np.uint8)
    dark = np.where(gray <= dark_threshold, 255, 0).astype(np.uint8)
    candidates = np.where(((bright > 0) | (dark > 0)) & (candidate_gate > 0) & (protect <= 0), 255, 0).astype(np.uint8)
    residual = _component_filtered_mask(candidates, seed_gate)
    source_glyphs = _non_boxy_seed_mask(seed)
    prior_candidates = np.where(((bright > 0) | (dark > 0)) & (prior > 0) & (protect <= 0), 255, 0).astype(np.uint8)
    if np.any(prior_candidates):
        prior_candidates = cv2.morphologyEx(
            prior_candidates,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
    orphan_glyphs = _component_filtered_mask(
        prior_candidates,
        prior,
        max_bbox_ratio=0.45,
        min_area=4,
    )
    merged = np.where(
        ((source_glyphs > 0) | (residual > 0) | (orphan_glyphs > 0)) & (protect <= 0),
        255,
        0,
    ).astype(np.uint8)
    if np.any(merged):
        final_px = max(1, int(final_dilate_px))
        merged = cv2.dilate(
            merged,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * final_px + 1, 2 * final_px + 1), (final_px, final_px)),
            iterations=1,
        )
        merged = np.where((merged > 0) & (protect <= 0), 255, 0).astype(np.uint8)

    edit_mask[y1:y2, x1:x2] = merged
    stats = BubbleEraseBlockStats(
        mode=ERASE_MODE_BUBBLE_TELEA,
        edit_pixel_count=mask_pixel_count(merged),
        protect_pixel_count=mask_pixel_count(protect),
    )
    return edit_mask, stats


def _ring_mask(mask: np.ndarray, *, radius: int) -> np.ndarray:
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.uint8)
    px = max(1, int(radius))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1), (px, px))
    dilated = cv2.dilate(binary, kernel, iterations=1)
    return np.where((dilated > 0) & (binary <= 0), 255, 0).astype(np.uint8)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.shape[2] >= 3:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    return image[:, :, 0].astype(np.uint8)


def _should_use_flat_fill(image_rgb: np.ndarray, edit_mask: np.ndarray) -> bool:
    ring = _ring_mask(edit_mask, radius=6)
    if not np.any(ring):
        return True
    gray = _to_gray(image_rgb)
    ring_pixels = gray[ring > 0]
    if ring_pixels.size == 0:
        return True
    spread = float(np.percentile(ring_pixels, 95) - np.percentile(ring_pixels, 5))
    std = float(np.std(ring_pixels))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges[ring > 0])) / float(max(1, ring_pixels.size))
    return std <= 12.0 and spread <= 36.0 and edge_density <= 0.08


def _local_median_fill(image_rgb: np.ndarray, edit_mask: np.ndarray) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    if not np.any(mask):
        return output

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        component = np.zeros_like(mask, dtype=np.uint8)
        component[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label_idx] = 255
        ring = _ring_mask(component, radius=5)
        ring_pixels = output[ring > 0]
        if ring_pixels.size == 0:
            ring_pixels = output[mask <= 0]
        if ring_pixels.size == 0:
            continue
        fill_value = np.median(ring_pixels, axis=0)
        if output.ndim == 2:
            output[component > 0] = np.uint8(np.clip(round(float(fill_value)), 0, 255))
        else:
            output[component > 0] = np.clip(np.round(fill_value), 0, 255).astype(output.dtype)
    return composite_with_edit_mask(image_rgb, output, mask)


def _trimmed_background_pixels(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, x2, y2 = roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    mask_crop = normalize_edit_mask(edit_mask, image_rgb.shape)[y1:y2, x1:x2]
    protect = _bubble_border_protect_mask(mask_crop.shape, width=4)
    bg_pixels = crop[(mask_crop <= 0) & (protect <= 0)]
    if bg_pixels.size == 0:
        return bg_pixels
    gray = _to_gray(bg_pixels.reshape((-1, 1, bg_pixels.shape[-1]))).reshape(-1) if crop.ndim == 3 else bg_pixels.reshape(-1)
    low = float(np.percentile(gray, 15))
    high = float(np.percentile(gray, 85))
    keep = (gray >= low) & (gray <= high)
    if np.count_nonzero(keep) < 8:
        return bg_pixels
    return bg_pixels[keep]


def _bubble_roi_background_metrics(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray, float]:
    x1, y1, x2, y2 = roi
    crop = np.asarray(image_rgb)[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((0,), dtype=np.uint8), np.zeros((0, 0), dtype=bool), 1.0
    mask_crop = normalize_edit_mask(edit_mask, image_rgb.shape)[y1:y2, x1:x2]
    protect = _bubble_border_protect_mask(mask_crop.shape, width=4)
    gray = _to_gray(crop)
    bg_region = (mask_crop <= 0) & (protect <= 0)
    bg_pixels = gray[bg_region]
    if bg_pixels.size == 0:
        return bg_pixels, bg_region, 1.0
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges[bg_region])) / float(max(1, int(np.count_nonzero(bg_region))))
    horizontal_pairs = bg_region[:, 1:] & bg_region[:, :-1]
    vertical_pairs = bg_region[1:, :] & bg_region[:-1, :]
    horizontal_jumps = np.abs(gray[:, 1:].astype(np.int16) - gray[:, :-1].astype(np.int16)) >= 18
    vertical_jumps = np.abs(gray[1:, :].astype(np.int16) - gray[:-1, :].astype(np.int16)) >= 18
    pair_count = int(np.count_nonzero(horizontal_pairs)) + int(np.count_nonzero(vertical_pairs))
    jump_count = int(np.count_nonzero(horizontal_jumps & horizontal_pairs)) + int(np.count_nonzero(vertical_jumps & vertical_pairs))
    texture_density = float(jump_count) / float(max(1, pair_count))
    edge_density = max(edge_density, texture_density)
    return bg_pixels, bg_region, edge_density


def _should_use_bubble_roi_flat_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
) -> bool:
    if bubble_roi is None:
        return False
    if _edit_mask_near_line_art(image_rgb, edit_mask, bubble_roi):
        return False
    bg_pixels = _trimmed_background_pixels(image_rgb, edit_mask, bubble_roi)
    if bg_pixels.size == 0:
        return False
    gray = (
        _to_gray(bg_pixels.reshape((-1, 1, bg_pixels.shape[-1]))).reshape(-1)
        if bg_pixels.ndim == 2
        else bg_pixels.reshape(-1)
    )
    if gray.size == 0:
        return False
    iqr = float(np.percentile(gray, 75) - np.percentile(gray, 25))
    spread = float(np.percentile(gray, 90) - np.percentile(gray, 10))
    _bg_pixels, _bg_region, edge_density = _bubble_roi_background_metrics(image_rgb, edit_mask, bubble_roi)
    return iqr <= 10.0 and spread <= 28.0 and edge_density <= 0.08


def _should_use_bubble_roi_gradient_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int] | None,
) -> bool:
    if bubble_roi is None:
        return False
    if _edit_mask_near_line_art(image_rgb, edit_mask, bubble_roi):
        return False
    bg_pixels, bg_region, edge_density = _bubble_roi_background_metrics(image_rgb, edit_mask, bubble_roi)
    if bg_pixels.size < 64 or np.count_nonzero(bg_region) < 64:
        return False
    return edge_density <= 0.09


def _bubble_roi_median_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int],
) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    bg_pixels = _trimmed_background_pixels(image_rgb, mask, bubble_roi)
    if bg_pixels.size == 0:
        return _local_median_fill(image_rgb, mask)
    fill_value = np.median(bg_pixels, axis=0)
    if output.ndim == 2:
        output[mask > 0] = np.uint8(np.clip(round(float(fill_value)), 0, 255))
    else:
        output[mask > 0] = np.clip(np.round(fill_value), 0, 255).astype(output.dtype)
    return composite_with_edit_mask(image_rgb, output, mask)


def _bubble_roi_gradient_fill(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    bubble_roi: tuple[int, int, int, int],
) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    output = np.asarray(image_rgb).copy()
    x1, y1, x2, y2 = bubble_roi
    crop = output[y1:y2, x1:x2].copy()
    mask_crop = mask[y1:y2, x1:x2]
    if crop.size == 0 or not np.any(mask_crop):
        return composite_with_edit_mask(image_rgb, output, mask)

    protect = _bubble_border_protect_mask(mask_crop.shape, width=5)
    gray = _to_gray(crop)

    labels_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (mask_crop > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    if labels_count <= 1:
        return _bubble_roi_median_fill(image_rgb, mask, bubble_roi)

    for label_idx in range(1, labels_count):
        x, y, w, h, area = stats[label_idx]
        if int(area) <= 0 or w <= 0 or h <= 0:
            continue
        component = np.zeros_like(mask_crop, dtype=np.uint8)
        component[y:y + h, x:x + w][labels[y:y + h, x:x + w] == label_idx] = 255
        fit_region = np.zeros_like(mask_crop, dtype=bool)
        for radius in (10, 16, 24, 32):
            ring = _ring_mask(component, radius=radius)
            candidate_region = (ring > 0) & (mask_crop <= 0) & (protect <= 0)
            candidate_values = gray[candidate_region]
            if candidate_values.size >= max(32, min(256, int(area) // 2)):
                low = float(np.percentile(candidate_values, 15))
                high = float(np.percentile(candidate_values, 85))
                fit_region = candidate_region & (gray >= low) & (gray <= high)
                if np.count_nonzero(fit_region) >= 32:
                    break
                fit_region = candidate_region
                break
        yy, xx = np.nonzero(fit_region)
        if yy.size < 32:
            continue
        if yy.size > 3000:
            sample_indices = np.linspace(0, yy.size - 1, 3000).astype(np.int32)
            yy = yy[sample_indices]
            xx = xx[sample_indices]

        target_y, target_x = np.nonzero(component > 0)
        design = np.stack(
            [
                xx.astype(np.float64),
                yy.astype(np.float64),
                np.ones_like(xx, dtype=np.float64),
            ],
            axis=1,
        )
        target_design = np.stack(
            [
                target_x.astype(np.float64),
                target_y.astype(np.float64),
                np.ones_like(target_x, dtype=np.float64),
            ],
            axis=1,
        )
        if crop.ndim == 2:
            coeffs, *_ = np.linalg.lstsq(design, crop[yy, xx].astype(np.float64), rcond=None)
            crop[target_y, target_x] = np.clip(np.round(target_design @ coeffs), 0, 255).astype(crop.dtype)
        else:
            channel_count = min(3, crop.shape[2])
            for channel in range(channel_count):
                coeffs, *_ = np.linalg.lstsq(design, crop[yy, xx, channel].astype(np.float64), rcond=None)
                crop[target_y, target_x, channel] = np.clip(np.round(target_design @ coeffs), 0, 255).astype(crop.dtype)
    output[y1:y2, x1:x2] = crop
    return composite_with_edit_mask(image_rgb, output, mask)


def _telea_fill(image_rgb: np.ndarray, edit_mask: np.ndarray, *, radius: int = 3) -> np.ndarray:
    mask = normalize_edit_mask(edit_mask, image_rgb.shape)
    if not np.any(mask):
        return np.asarray(image_rgb).copy()
    image = np.asarray(image_rgb)
    if image.ndim == 3 and image.shape[2] > 3:
        rgb = image[:, :, :3]
        filled_rgb = cv2.inpaint(rgb, mask, max(1, int(radius)), cv2.INPAINT_TELEA)
        filled = image.copy()
        filled[:, :, :3] = filled_rgb
    else:
        filled = cv2.inpaint(image, mask, max(1, int(radius)), cv2.INPAINT_TELEA)
    return composite_with_edit_mask(image_rgb, filled, mask)


def _fill_bubble_mask(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    bubble_roi: tuple[int, int, int, int] | None = None,
) -> tuple[np.ndarray, str]:
    if _should_use_bubble_roi_flat_fill(image_rgb, edit_mask, bubble_roi):
        return _bubble_roi_median_fill(image_rgb, edit_mask, bubble_roi), ERASE_MODE_BUBBLE_FLAT_FILL
    if _should_use_bubble_roi_gradient_fill(image_rgb, edit_mask, bubble_roi):
        return _bubble_roi_gradient_fill(image_rgb, edit_mask, bubble_roi), ERASE_MODE_BUBBLE_GRADIENT_FILL
    if _should_use_flat_fill(image_rgb, edit_mask):
        return _local_median_fill(image_rgb, edit_mask), ERASE_MODE_BUBBLE_FLAT_FILL
    return _telea_fill(image_rgb, edit_mask, radius=2), ERASE_MODE_BUBBLE_TELEA


def fill_bubble_edit_mask(image_rgb: np.ndarray, edit_mask: np.ndarray) -> tuple[np.ndarray, str]:
    return _fill_bubble_mask(image_rgb, edit_mask)


def erase_text_bubble_regions(
    original_image: np.ndarray,
    current_cleaned: np.ndarray,
    source_mask: np.ndarray,
    blocks: list,
    config=None,
) -> BubbleEraseResult:
    if original_image is None or current_cleaned is None:
        empty_shape = (0, 0) if original_image is None else original_image.shape[:2]
        empty_mask = np.zeros(empty_shape, dtype=np.uint8)
        return BubbleEraseResult(
            image=current_cleaned,
            edit_mask=empty_mask,
            fallback_mask=empty_mask.copy(),
            expanded_bubble_mask=empty_mask.copy(),
            stats={
                "applied": False,
                "block_count": 0,
                "applied_block_count": 0,
                "fallback_block_count": 0,
                "edit_pixel_count": 0,
                "fallback_pixel_count": 0,
                "changed_outside_edit_mask_pixel_count": 0,
                "blocks": [],
            },
        )

    result = np.asarray(current_cleaned).copy()
    source = normalize_edit_mask(source_mask, original_image.shape)
    union_edit_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    fallback_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    bubble_roi_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    block_entries: list[dict] = []
    applied_blocks = 0
    fallback_blocks = 0

    for index, block in enumerate(list(blocks or [])):
        if getattr(block, "text_class", "") != "text_bubble":
            if getattr(block, "text_class", "") == "text_free":
                set_block_erase_metadata(block, BubbleEraseBlockStats(mode=ERASE_MODE_TEXT_FREE_LAMA))
            continue

        bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), original_image.shape)
        if bubble_roi is not None:
            x1, y1, x2, y2 = bubble_roi
            bubble_roi_mask[y1:y2, x1:x2] = 255

        edit_mask, mask_stats = build_bubble_residual_edit_mask(original_image, source, block)
        line_art_intrusion = (
            _edit_mask_near_line_art(original_image, source, bubble_roi)
            or _edit_mask_near_line_art(original_image, edit_mask, bubble_roi)
        )
        if line_art_intrusion:
            fallback_edit_mask, fallback_stats = build_bubble_residual_edit_mask(
                original_image,
                source,
                block,
                protect_line_art=False,
            )
            line_protect = _line_art_protect_mask_for_roi(original_image, bubble_roi)
            fallback_edit_mask = np.where(
                (fallback_edit_mask > 0) & ((line_protect <= 0) | (source > 0)),
                255,
                0,
            ).astype(np.uint8)
            if not np.any(fallback_edit_mask):
                fallback_edit_mask = edit_mask
            if np.any(fallback_edit_mask):
                fallback_blocks += 1
                fallback_mask = np.where((fallback_mask > 0) | (fallback_edit_mask > 0), 255, 0).astype(np.uint8)
                block_stats = BubbleEraseBlockStats(
                    mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
                    edit_pixel_count=mask_pixel_count(fallback_edit_mask),
                    protect_pixel_count=max(mask_stats.protect_pixel_count, fallback_stats.protect_pixel_count),
                    skipped_reason="line_art_intrusion",
                )
                set_block_erase_metadata(block, block_stats)
                block_entries.append(
                    {
                        "index": index,
                        "mode": block_stats.mode,
                        "edit_pixel_count": block_stats.edit_pixel_count,
                        "protect_pixel_count": block_stats.protect_pixel_count,
                        "skipped_reason": block_stats.skipped_reason,
                    }
                )
                continue

        if not np.any(edit_mask):
            set_block_erase_metadata(block, mask_stats)
            block_entries.append(
                {
                    "index": index,
                    "mode": mask_stats.mode,
                    "edit_pixel_count": 0,
                    "protect_pixel_count": mask_stats.protect_pixel_count,
                    "skipped_reason": mask_stats.skipped_reason,
                }
            )
            continue

        filled, mode = _fill_bubble_mask(original_image, edit_mask, bubble_roi=bubble_roi)
        result = composite_with_edit_mask(result, filled, edit_mask)
        union_edit_mask = np.where((union_edit_mask > 0) | (edit_mask > 0), 255, 0).astype(np.uint8)
        applied_blocks += 1
        block_stats = BubbleEraseBlockStats(
            mode=mode,
            edit_pixel_count=mask_pixel_count(edit_mask),
            protect_pixel_count=mask_stats.protect_pixel_count,
        )
        set_block_erase_metadata(block, block_stats)
        block_entries.append(
            {
                "index": index,
                "mode": mode,
                "edit_pixel_count": block_stats.edit_pixel_count,
                "protect_pixel_count": block_stats.protect_pixel_count,
                "skipped_reason": "",
            }
        )

    result = composite_with_edit_mask(current_cleaned, result, union_edit_mask)
    outside_changed = count_changed_outside_edit_mask(current_cleaned, result, union_edit_mask)
    return BubbleEraseResult(
        image=result,
        edit_mask=union_edit_mask,
        fallback_mask=fallback_mask,
        expanded_bubble_mask=bubble_roi_mask,
        stats={
            "applied": bool(applied_blocks or fallback_blocks),
            "block_count": len(list(blocks or [])),
            "applied_block_count": applied_blocks,
            "fallback_block_count": fallback_blocks,
            "edit_pixel_count": mask_pixel_count(union_edit_mask),
            "fallback_pixel_count": mask_pixel_count(fallback_mask),
            "changed_outside_edit_mask_pixel_count": int(outside_changed),
            "blocks": block_entries,
        },
    )
