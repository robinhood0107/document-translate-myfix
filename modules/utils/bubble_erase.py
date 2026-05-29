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
    prefer_flat: bool = False,
) -> tuple[np.ndarray, str]:
    if prefer_flat:
        return _local_median_fill(image_rgb, edit_mask), ERASE_MODE_BUBBLE_FLAT_FILL
    if _should_use_flat_fill(image_rgb, edit_mask):
        return _local_median_fill(image_rgb, edit_mask), ERASE_MODE_BUBBLE_FLAT_FILL
    return _telea_fill(image_rgb, edit_mask, radius=3), ERASE_MODE_BUBBLE_TELEA


def fill_bubble_edit_mask(
    image_rgb: np.ndarray,
    edit_mask: np.ndarray,
    *,
    prefer_flat: bool = False,
) -> tuple[np.ndarray, str]:
    return _fill_bubble_mask(image_rgb, edit_mask, prefer_flat=prefer_flat)


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
            expanded_bubble_mask=empty_mask.copy(),
            stats={
                "applied": False,
                "block_count": 0,
                "applied_block_count": 0,
                "edit_pixel_count": 0,
                "changed_outside_edit_mask_pixel_count": 0,
                "blocks": [],
            },
        )

    result = np.asarray(current_cleaned).copy()
    source = normalize_edit_mask(source_mask, original_image.shape)
    union_edit_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    bubble_roi_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    block_entries: list[dict] = []
    applied_blocks = 0

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

        filled, mode = _fill_bubble_mask(original_image, edit_mask, prefer_flat=True)
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
        expanded_bubble_mask=bubble_roi_mask,
        stats={
            "applied": bool(applied_blocks),
            "block_count": len(list(blocks or [])),
            "applied_block_count": applied_blocks,
            "edit_pixel_count": mask_pixel_count(union_edit_mask),
            "changed_outside_edit_mask_pixel_count": int(outside_changed),
            "blocks": block_entries,
        },
    )
