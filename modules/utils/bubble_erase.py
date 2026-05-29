from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from modules.utils.inpaint_composite import normalize_edit_mask
from modules.utils.mask_roi import build_text_prior_mask, normalize_xyxy


ERASE_MODE_TEXT_FREE_LAMA = "text_free_lama"
ERASE_MODE_BUBBLE_FLAT_FILL = "bubble_flat_fill"
ERASE_MODE_BUBBLE_TELEA = "bubble_telea"
ERASE_MODE_BUBBLE_LAMA_FALLBACK = "bubble_lama_fallback"
ERASE_MODE_BUBBLE_SKIPPED = "bubble_skipped"


@dataclass(slots=True)
class BubbleEraseBlockStats:
    mode: str
    edit_pixel_count: int = 0
    protect_pixel_count: int = 0
    skipped_reason: str = ""


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


def _rule_like_component(width: int, height: int, area: int) -> bool:
    long_side = max(int(width), int(height))
    short_side = min(int(width), int(height))
    if long_side < 16 or short_side > 5:
        return False
    aspect = float(long_side) / float(max(1, short_side))
    fill_ratio = float(area) / float(max(1, int(width) * int(height)))
    return aspect >= 10.0 and fill_ratio >= 0.30


def _component_filtered_mask(candidate: np.ndarray, seed_gate: np.ndarray, *, max_bbox_ratio: float = 0.22) -> np.ndarray:
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
        if int(area) < 3 or w <= 0 or h <= 0:
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


def build_bubble_residual_edit_mask(
    image_rgb: np.ndarray,
    source_mask: np.ndarray,
    block,
    *,
    seed_dilate_px: int = 8,
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
    protect = _bubble_border_protect_mask(seed.shape, width=3)

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
    merged = np.where(((seed > 0) | (residual > 0)) & (protect <= 0), 255, 0).astype(np.uint8)
    if np.any(merged):
        merged = cv2.dilate(merged, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        merged = np.where((merged > 0) & (protect <= 0), 255, 0).astype(np.uint8)

    edit_mask[y1:y2, x1:x2] = merged
    stats = BubbleEraseBlockStats(
        mode=ERASE_MODE_BUBBLE_TELEA,
        edit_pixel_count=mask_pixel_count(merged),
        protect_pixel_count=mask_pixel_count(protect),
    )
    return edit_mask, stats
