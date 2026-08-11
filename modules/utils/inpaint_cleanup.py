from __future__ import annotations

import logging
from typing import Iterable

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import detect_content_in_bbox
from modules.utils.bubble_erase import (
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    fill_bubble_edit_mask,
)
from modules.utils.inpaint_composite import composite_with_edit_mask, normalize_edit_mask
from modules.utils.mask_roi import (
    build_text_prior_mask,
    normalize_xyxy,
    resolve_block_residue_roi,
    resolve_inpaint_text_xyxy,
)
from modules.utils.textblock import TextBlock

logger = logging.getLogger(__name__)
RESIDUE_SOURCE_MASK_DILATE_PX = 2
PASS2_BACKEND_MIXED = "mixed"


def _is_structure_guarded_cleanup_block(block: TextBlock) -> bool:
    reason = str(getattr(block, "_erase_skipped_reason", "") or "")
    return bool(reason)


def _build_cleanup_priority_protection(
    source_mask: np.ndarray,
    blocks: list[TextBlock],
    image_shape: tuple[int, ...],
) -> np.ndarray:
    protection = np.zeros(image_shape[:2], dtype=np.uint8)
    for block in blocks:
        text_class = getattr(block, "text_class", "")
        missing_bubble_roi = (
            text_class == "text_bubble"
            and normalize_xyxy(
                getattr(block, "bubble_xyxy", None),
                image_shape,
            )
            is None
        )
        if (
            text_class != "text_free"
            and not _is_structure_guarded_cleanup_block(block)
            and not missing_bubble_roi
        ):
            continue
        reason = str(getattr(block, "_erase_skipped_reason", "") or "")
        delegated_reason = reason in {"lama_priority_owned", "missing_bubble_roi"}
        roi = None
        if reason and not delegated_reason:
            roi = normalize_xyxy(
                getattr(block, "bubble_xyxy", None),
                image_shape,
            )
        if roi is None:
            roi = resolve_inpaint_text_xyxy(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        if reason and not delegated_reason:
            protection[y1:y2, x1:x2] = 255
            continue
        owned = source_mask[y1:y2, x1:x2]
        if not np.any(owned):
            continue
        local_protection = cv2.dilate(
            owned,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
            iterations=1,
        )
        protection[y1:y2, x1:x2] = np.where(
            (protection[y1:y2, x1:x2] > 0)
            | (local_protection > 0),
            255,
            0,
        ).astype(np.uint8)
    return protection


def _empty_pass2_stats(mask_shape: tuple[int, int]) -> dict:
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
        "residue_mask_cap_dilate_px": RESIDUE_SOURCE_MASK_DILATE_PX,
        "pass2_backend": "",
        "pass2_backend_distribution": {},
        "pass2_applied_block_count": 0,
        "pass2_fallback_block_count": 0,
        "pass2_applied_pixel_count": 0,
        "residue_pass_truncated_block_count": 0,
        "residue_pass_cap_dropped_candidate_count": 0,
        "residue_pass_structure_guard_block_count": 0,
    }


def _empty_duplicate_bubble_inner_fill_stats(mask_shape: tuple[int, int]) -> dict:
    return {
        "applied": False,
        "pass_name": "duplicate_bubble_inner_fill",
        "duplicate_bubble_inner_fill_mask": np.zeros(mask_shape, dtype=np.uint8),
        "duplicate_bubble_inner_fill_pixel_count": 0,
        "duplicate_bubble_inner_fill_backend": "",
    }


def fill_duplicate_bubble_inner_regions(
    inpainted_image: np.ndarray,
    duplicate_bubble_inner_mask: np.ndarray | None,
) -> tuple[np.ndarray, dict]:
    if inpainted_image is None:
        shape = duplicate_bubble_inner_mask.shape if duplicate_bubble_inner_mask is not None else (0, 0)
        return inpainted_image, _empty_duplicate_bubble_inner_fill_stats(shape)

    edit_mask = normalize_edit_mask(duplicate_bubble_inner_mask, inpainted_image.shape)
    if edit_mask.size == 0 or not np.any(edit_mask):
        return inpainted_image, _empty_duplicate_bubble_inner_fill_stats(inpainted_image.shape[:2])

    filled_image, backend = fill_bubble_edit_mask(inpainted_image, edit_mask)
    if backend == ERASE_MODE_BUBBLE_LAMA_FALLBACK:
        stats = _empty_duplicate_bubble_inner_fill_stats(
            inpainted_image.shape[:2]
        )
        stats["duplicate_bubble_inner_fill_backend"] = backend
        return inpainted_image, stats
    filled_image = imk.convert_scale_abs(filled_image)
    filled_image = composite_with_edit_mask(inpainted_image, filled_image, edit_mask)

    return filled_image, {
        "applied": True,
        "pass_name": "duplicate_bubble_inner_fill",
        "duplicate_bubble_inner_fill_mask": edit_mask,
        "duplicate_bubble_inner_fill_pixel_count": int(np.count_nonzero(edit_mask)),
        "duplicate_bubble_inner_fill_backend": backend,
    }


def apply_duplicate_bubble_inner_fill(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    mask_details: dict | None,
    cleanup_stats: dict | None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    merged_stats = dict(cleanup_stats or {})
    duplicate_mask = (mask_details or {}).get("duplicate_bubble_inner_mask")
    filled_image, fill_stats = fill_duplicate_bubble_inner_regions(inpainted_image, duplicate_mask)
    merged_stats["duplicate_bubble_inner_fill"] = fill_stats
    if not fill_stats.get("applied"):
        return inpainted_image, mask, merged_stats

    fill_mask = fill_stats.get("duplicate_bubble_inner_fill_mask")
    fill_mask = normalize_edit_mask(fill_mask, filled_image.shape)
    merged_mask = np.where((normalize_edit_mask(mask, filled_image.shape) > 0) | (fill_mask > 0), 255, 0).astype(np.uint8)
    return filled_image, merged_mask, merged_stats


def _dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    deduped: list[tuple[int, int, int, int]] = []
    for box in boxes:
        norm = tuple(int(v) for v in box[:4])
        if norm in seen:
            continue
        seen.add(norm)
        deduped.append(norm)
    return deduped


def _component_boxes_from_mask(mask: np.ndarray, *, min_area: int) -> list[tuple[int, int, int, int]]:
    if mask is None or mask.size == 0 or not np.any(mask):
        return []
    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8, cv2.CV_32S)
    boxes: list[tuple[int, int, int, int]] = []
    for label in range(1, num_labels):
        x, y, w, h, area = stats[label]
        if int(area) < int(min_area) or w <= 0 or h <= 0:
            continue
        boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def _residue_source_cap(mask: np.ndarray, *, dilate_px: int = RESIDUE_SOURCE_MASK_DILATE_PX) -> np.ndarray:
    if mask is None or mask.size == 0 or not np.any(mask):
        return np.zeros_like(mask, dtype=np.uint8)
    binary = np.where(mask > 0, 255, 0).astype(np.uint8)
    px = max(0, int(dilate_px))
    if px <= 0:
        return binary
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1), (px, px))
    return np.where(cv2.dilate(binary, kernel, iterations=1) > 0, 255, 0).astype(np.uint8)


def _cap_residue_mask_to_source_mask(
    residue_mask: np.ndarray,
    source_mask: np.ndarray,
    *,
    dilate_px: int = RESIDUE_SOURCE_MASK_DILATE_PX,
) -> np.ndarray:
    if residue_mask is None or residue_mask.size == 0 or not np.any(residue_mask):
        return np.zeros_like(residue_mask, dtype=np.uint8)
    cap = _residue_source_cap(source_mask, dilate_px=dilate_px)
    return np.where((residue_mask > 0) & (cap > 0), 255, 0).astype(np.uint8)


def _build_bubble_faint_boxes(crop: np.ndarray, prior_mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    if crop is None or crop.size == 0 or prior_mask is None or not np.any(prior_mask):
        return []
    gray = imk.to_gray(crop)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    prior_pixels = blackhat[prior_mask > 0]
    if prior_pixels.size == 0:
        return []
    threshold = max(8, int(np.percentile(prior_pixels, 70)))
    binary = np.where((blackhat >= threshold) & (prior_mask > 0), 255, 0).astype(np.uint8)
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)
    return _component_boxes_from_mask(binary, min_area=4)


def refine_bubble_residue_inpaint(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    blk_list: Iterable[TextBlock],
    inpainter,
    config,
    page_label: str = "",
    *,
    protected_corner_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    block_list = list(blk_list or [])
    if mask is not None and inpainted_image is not None:
        protected = normalize_edit_mask(
            protected_corner_mask,
            inpainted_image.shape,
        )
        if np.any(protected):
            mask = np.where(
                (normalize_edit_mask(mask, inpainted_image.shape) > 0)
                & (protected <= 0),
                255,
                0,
            ).astype(np.uint8)
    else:
        protected = np.zeros((0, 0), dtype=np.uint8)
    if (
        inpainted_image is None
        or mask is None
        or not block_list
        or not np.any(mask)
    ):
        return inpainted_image, mask, _empty_pass2_stats(mask.shape if mask is not None else inpainted_image.shape[:2])

    block_residue_masks: dict[int, np.ndarray] = {}
    block_residue_rois: dict[int, tuple[int, int, int, int]] = {}
    block_bubble_rois: dict[int, tuple[int, int, int, int] | None] = {}
    touched_blocks: list[int] = []
    component_count = 0
    pass2_candidate_count = 0
    bubble_candidate_count = 0
    bubble_kept_count = 0
    text_free_candidate_count = 0
    text_free_kept_count = 0
    truncated_block_indices: set[int] = set()
    dropped_candidate_count = 0
    structure_guard_block_count = 0
    priority_protected = _build_cleanup_priority_protection(
        mask,
        block_list,
        inpainted_image.shape,
    )
    cleanup_source_mask = np.where(
        (mask > 0) & (priority_protected <= 0),
        255,
        0,
    ).astype(np.uint8)
    source_cap = _residue_source_cap(
        cleanup_source_mask,
        dilate_px=RESIDUE_SOURCE_MASK_DILATE_PX,
    )
    cleanup_protected = np.where(
        (protected > 0) | (priority_protected > 0),
        255,
        0,
    ).astype(np.uint8)

    for idx, blk in enumerate(block_list):
        if getattr(blk, "xyxy", None) is None:
            continue

        text_class = getattr(blk, "text_class", "") or ""
        if text_class != "text_bubble":
            continue
        if _is_structure_guarded_cleanup_block(blk):
            structure_guard_block_count += 1
            continue

        residue_roi = normalize_xyxy(getattr(blk, "cleanup_roi_xyxy", None), inpainted_image.shape)
        if residue_roi is None:
            residue_roi = resolve_block_residue_roi(blk, inpainted_image.shape)
        if residue_roi is None:
            continue

        rx1, ry1, rx2, ry2 = residue_roi
        block_residue_rois[idx] = residue_roi
        bubble_roi = normalize_xyxy(
            getattr(blk, "bubble_xyxy", None),
            inpainted_image.shape,
        )
        if bubble_roi is None:
            structure_guard_block_count += 1
            continue
        block_bubble_rois[idx] = bubble_roi
        crop = inpainted_image[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        prior_mask = build_text_prior_mask(
            inpainted_image,
            blk,
            residue_roi,
            dilate_iterations=3,
        )
        if not np.any(prior_mask):
            continue

        detected_boxes = detect_content_in_bbox(
            crop,
            min_area=4,
            margin=0,
            inclusive_min_area=True,
        )
        residual_boxes = list(detected_boxes) if detected_boxes is not None else []
        if text_class == "text_bubble":
            residual_boxes.extend(_build_bubble_faint_boxes(crop, prior_mask))
        residual_boxes = _dedupe_boxes(residual_boxes)
        if len(residual_boxes) == 0:
            continue

        roi_area = max(1, (rx2 - rx1) * (ry2 - ry1))
        max_bbox_ratio = 0.20 if text_class == "text_bubble" else 0.16
        edge_bbox_ratio = 0.10
        gray = imk.to_gray(crop)
        local_components = 0
        max_local_components = 35

        for residual_index, (lx1, ly1, lx2, ly2) in enumerate(residual_boxes):
            if local_components >= max_local_components:
                truncated_block_indices.add(idx)
                dropped_candidate_count += len(residual_boxes) - residual_index
                break
            lx1 = max(0, min(int(lx1), crop.shape[1]))
            ly1 = max(0, min(int(ly1), crop.shape[0]))
            lx2 = max(0, min(int(lx2), crop.shape[1]))
            ly2 = max(0, min(int(ly2), crop.shape[0]))
            w = int(lx2 - lx1)
            h = int(ly2 - ly1)
            bbox_area = int(w * h)
            if bbox_area <= 0:
                continue
            if bbox_area > int(round(roi_area * max_bbox_ratio)):
                continue

            pass2_candidate_count += 1
            if text_class == "text_bubble":
                bubble_candidate_count += 1
            else:
                text_free_candidate_count += 1

            touches_edge = lx1 <= 0 or ly1 <= 0 or lx2 >= crop.shape[1] or ly2 >= crop.shape[0]
            prior_crop = prior_mask[ly1:ly2, lx1:lx2]
            if prior_crop.size == 0 or not np.any(prior_crop > 0):
                continue
            prior_overlap_ratio = float(np.count_nonzero(prior_crop > 0)) / float(max(1, bbox_area))
            if prior_overlap_ratio <= 0.0:
                continue
            if touches_edge and prior_overlap_ratio < 0.20 and bbox_area > int(round(roi_area * edge_bbox_ratio)):
                continue

            comp_gray = gray[ly1:ly2, lx1:lx2]
            if comp_gray.size == 0:
                continue
            comp_mean = float(np.mean(comp_gray))
            comp_p35 = float(np.percentile(comp_gray, 35))
            if text_class == "text_bubble":
                if comp_mean > 245 and comp_p35 > 230:
                    continue
            else:
                if comp_mean > 228 and comp_p35 > 205:
                    continue

            gx1, gy1, gx2, gy2 = rx1 + int(lx1), ry1 + int(ly1), rx1 + int(lx2), ry1 + int(ly2)
            if gx2 <= gx1 or gy2 <= gy1:
                continue

            allowed_crop = source_cap[gy1:gy2, gx1:gx2] > 0
            if allowed_crop.size == 0 or not np.any(allowed_crop):
                continue
            block_mask = block_residue_masks.setdefault(
                idx,
                np.zeros((ry2 - ry1, rx2 - rx1), dtype=np.uint8),
            )
            block_mask_crop = block_mask[
                gy1 - ry1 : gy2 - ry1,
                gx1 - rx1 : gx2 - rx1,
            ]
            block_mask_crop[allowed_crop] = 255
            local_components += 1
            component_count += 1
            if text_class == "text_bubble":
                bubble_kept_count += 1
            else:
                text_free_kept_count += 1
        if local_components > 0:
            touched_blocks.append(idx)

    if component_count <= 0:
        empty_stats = _empty_pass2_stats(mask.shape)
        empty_stats["residue_pass_structure_guard_block_count"] = int(
            structure_guard_block_count
        )
        return inpainted_image, mask, empty_stats

    residue_mask_pre_cap = np.zeros_like(mask, dtype=np.uint8)
    residue_mask = np.zeros_like(mask, dtype=np.uint8)
    refined_image = np.asarray(inpainted_image).copy()
    applied_residue_mask = np.zeros_like(mask, dtype=np.uint8)
    backend_distribution: dict[str, int] = {}
    applied_block_count = 0
    fallback_block_count = 0
    for idx in touched_blocks:
        block_local_mask = block_residue_masks.get(idx)
        residue_roi = block_residue_rois.get(idx)
        if (
            block_local_mask is None
            or residue_roi is None
            or not np.any(block_local_mask)
        ):
            continue
        block_local_mask = imk.dilate(
            block_local_mask,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )
        rx1, ry1, rx2, ry2 = residue_roi
        residue_mask_pre_cap[ry1:ry2, rx1:rx2] = np.where(
            (residue_mask_pre_cap[ry1:ry2, rx1:rx2] > 0)
            | (block_local_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        block_local_mask = np.where(
            (block_local_mask > 0)
            & (source_cap[ry1:ry2, rx1:rx2] > 0),
            255,
            0,
        ).astype(np.uint8)
        if np.any(cleanup_protected):
            block_local_mask = np.where(
                (block_local_mask > 0)
                & (cleanup_protected[ry1:ry2, rx1:rx2] <= 0),
                255,
                0,
            ).astype(np.uint8)
        if not np.any(block_local_mask):
            continue
        bubble_roi = block_bubble_rois.get(idx)
        fill_roi = bubble_roi or residue_roi
        fx1, fy1, fx2, fy2 = fill_roi
        ix1 = max(rx1, fx1)
        iy1 = max(ry1, fy1)
        ix2 = min(rx2, fx2)
        iy2 = min(ry2, fy2)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        fill_local_mask = np.zeros((fy2 - fy1, fx2 - fx1), dtype=np.uint8)
        fill_local_mask[
            iy1 - fy1 : iy2 - fy1,
            ix1 - fx1 : ix2 - fx1,
        ] = block_local_mask[
            iy1 - ry1 : iy2 - ry1,
            ix1 - rx1 : ix2 - rx1,
        ]
        if not np.any(fill_local_mask):
            continue
        fill_local_mask = np.where(
            (fill_local_mask > 0)
            & (applied_residue_mask[fy1:fy2, fx1:fx2] <= 0),
            255,
            0,
        ).astype(np.uint8)
        if not np.any(fill_local_mask):
            continue
        residue_mask[fy1:fy2, fx1:fx2] = np.where(
            (residue_mask[fy1:fy2, fx1:fx2] > 0)
            | (fill_local_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        fill_source = refined_image[fy1:fy2, fx1:fx2].copy()
        local_bubble_roi = (
            (0, 0, fx2 - fx1, fy2 - fy1)
            if bubble_roi is not None
            else None
        )
        local_background_exclude = source_cap[fy1:fy2, fx1:fx2].copy()
        if np.any(cleanup_protected):
            local_background_exclude = np.where(
                (local_background_exclude > 0)
                | (cleanup_protected[fy1:fy2, fx1:fx2] > 0),
                255,
                0,
            ).astype(np.uint8)
        block_filled, backend = fill_bubble_edit_mask(
            fill_source,
            fill_local_mask,
            bubble_roi=local_bubble_roi,
            background_exclude_mask=local_background_exclude,
        )
        backend_distribution[backend] = backend_distribution.get(backend, 0) + 1
        if backend == ERASE_MODE_BUBBLE_LAMA_FALLBACK:
            fallback_block_count += 1
            continue
        refined_image[fy1:fy2, fx1:fx2] = composite_with_edit_mask(
            refined_image[fy1:fy2, fx1:fx2],
            imk.convert_scale_abs(block_filled),
            fill_local_mask,
        )
        applied_residue_mask[fy1:fy2, fx1:fx2] = np.where(
            (applied_residue_mask[fy1:fy2, fx1:fx2] > 0)
            | (fill_local_mask > 0),
            255,
            0,
        ).astype(np.uint8)
        applied_block_count += 1

    residue_mask_pre_cap_pixel_count = int(
        np.count_nonzero(residue_mask_pre_cap)
    )
    residue_pass_truncated_block_count = len(truncated_block_indices)
    residue_mask_cap_pixel_count = int(np.count_nonzero(residue_mask))
    if not np.any(residue_mask):
        empty_stats = _empty_pass2_stats(mask.shape)
        empty_stats["residue_pass_truncated_block_count"] = int(
            residue_pass_truncated_block_count
        )
        empty_stats["residue_pass_cap_dropped_candidate_count"] = int(
            dropped_candidate_count
        )
        empty_stats["residue_pass_structure_guard_block_count"] = int(
            structure_guard_block_count
        )
        return inpainted_image, mask, empty_stats

    pass2_backend = ""
    if len(backend_distribution) == 1:
        pass2_backend = next(iter(backend_distribution))
    elif len(backend_distribution) > 1:
        pass2_backend = PASS2_BACKEND_MIXED

    if not np.any(applied_residue_mask):
        fallback_stats = _empty_pass2_stats(mask.shape)
        fallback_stats.update(
            {
                "component_count": int(component_count),
                "block_count": len(touched_blocks),
                "pass2_candidate_count": int(pass2_candidate_count),
                "pass2_bubble_candidate_count": int(
                    bubble_candidate_count
                ),
                "pass2_bubble_kept_count": int(bubble_kept_count),
                "pass2_text_free_candidate_count": int(
                    text_free_candidate_count
                ),
                "pass2_text_free_kept_count": int(text_free_kept_count),
                "residue_mask_pre_cap_pixel_count": int(
                    residue_mask_pre_cap_pixel_count
                ),
                "residue_mask_cap_pixel_count": int(
                    residue_mask_cap_pixel_count
                ),
                "pass2_backend": pass2_backend,
                "pass2_backend_distribution": dict(backend_distribution),
                "pass2_applied_block_count": int(applied_block_count),
                "pass2_fallback_block_count": int(fallback_block_count),
                "pass2_applied_pixel_count": 0,
                "residue_pass_truncated_block_count": int(
                    residue_pass_truncated_block_count
                ),
                "residue_pass_cap_dropped_candidate_count": int(
                    dropped_candidate_count
                ),
                "residue_pass_structure_guard_block_count": int(
                    structure_guard_block_count
                ),
            }
        )
        return inpainted_image, mask, fallback_stats
    residue_mask = applied_residue_mask
    merged_mask = np.where(
        (mask > 0) | (residue_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    if protected.shape == merged_mask.shape and np.any(protected):
        merged_mask = np.where(
            (merged_mask > 0) & (protected <= 0),
            255,
            0,
        ).astype(np.uint8)

    logger.info(
        "[%s] inpaint-residue-cleanup: 인페인팅 후처리(잔여 텍스트 재정리) blocks=%s components=%d "
        "bubble_kept=%d text_free_kept=%d",
        page_label or "?/?",
        touched_blocks,
        component_count,
        bubble_kept_count,
        text_free_kept_count,
    )
    return refined_image, merged_mask, {
        "applied": True,
        "component_count": component_count,
        "block_count": len(touched_blocks),
        "pass_name": "residue_pass2",
        "residue_mask": residue_mask,
        "pass2_candidate_count": pass2_candidate_count,
        "pass2_bubble_candidate_count": bubble_candidate_count,
        "pass2_bubble_kept_count": bubble_kept_count,
        "pass2_text_free_candidate_count": text_free_candidate_count,
        "pass2_text_free_kept_count": text_free_kept_count,
        "residue_mask_pre_cap_pixel_count": residue_mask_pre_cap_pixel_count,
        "residue_mask_cap_pixel_count": residue_mask_cap_pixel_count,
        "residue_mask_cap_dilate_px": RESIDUE_SOURCE_MASK_DILATE_PX,
        "pass2_backend": pass2_backend,
        "pass2_backend_distribution": dict(backend_distribution),
        "pass2_applied_block_count": int(applied_block_count),
        "pass2_fallback_block_count": int(fallback_block_count),
        "pass2_applied_pixel_count": int(
            np.count_nonzero(applied_residue_mask)
        ),
        "residue_pass_truncated_block_count": int(
            residue_pass_truncated_block_count
        ),
        "residue_pass_cap_dropped_candidate_count": int(
            dropped_candidate_count
        ),
        "residue_pass_structure_guard_block_count": int(
            structure_guard_block_count
        ),
    }
