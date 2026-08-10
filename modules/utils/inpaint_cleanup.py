from __future__ import annotations

import logging
from typing import Iterable

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import detect_content_in_bbox
from modules.utils.bubble_erase import fill_bubble_edit_mask
from modules.utils.inpaint_composite import composite_with_edit_mask, normalize_edit_mask
from modules.utils.mask_roi import build_text_prior_mask, normalize_xyxy, resolve_block_residue_roi
from modules.utils.textblock import TextBlock

logger = logging.getLogger(__name__)
RESIDUE_SOURCE_MASK_DILATE_PX = 2


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
        "residue_pass_truncated_block_count": 0,
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

    residue_mask = np.zeros_like(mask, dtype=np.uint8)
    residue_roi_union = np.zeros_like(mask, dtype=np.uint8)
    touched_blocks: list[int] = []
    component_count = 0
    pass2_candidate_count = 0
    bubble_candidate_count = 0
    bubble_kept_count = 0
    text_free_candidate_count = 0
    text_free_kept_count = 0
    page_cap_hit = False
    truncated_block_indices: set[int] = set()
    source_cap = _residue_source_cap(mask, dilate_px=RESIDUE_SOURCE_MASK_DILATE_PX)

    for idx, blk in enumerate(block_list):
        if getattr(blk, "xyxy", None) is None:
            continue

        text_class = getattr(blk, "text_class", "") or ""
        if text_class != "text_bubble":
            continue

        residue_roi = normalize_xyxy(getattr(blk, "cleanup_roi_xyxy", None), inpainted_image.shape)
        if residue_roi is None:
            residue_roi = resolve_block_residue_roi(blk, inpainted_image.shape)
        if residue_roi is None:
            continue

        rx1, ry1, rx2, ry2 = residue_roi
        residue_roi_union[ry1:ry2, rx1:rx2] = 255
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

        detected_boxes = detect_content_in_bbox(crop, min_area=4, margin=0)
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
                break
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
            residue_mask_crop = residue_mask[gy1:gy2, gx1:gx2]
            residue_mask_crop[allowed_crop] = 255
            local_components += 1
            component_count += 1
            if text_class == "text_bubble":
                bubble_kept_count += 1
            else:
                text_free_kept_count += 1
            if component_count >= 120:
                page_cap_hit = True
                if residual_index < len(residual_boxes) - 1:
                    truncated_block_indices.add(idx)
                truncated_block_indices.update(
                    later_index
                    for later_index, later_block in enumerate(
                        block_list[idx + 1 :],
                        start=idx + 1,
                    )
                    if getattr(later_block, "xyxy", None) is not None
                    and str(getattr(later_block, "text_class", "") or "")
                    == "text_bubble"
                )
                logger.info(
                    "[%s] inpaint-residue-cleanup: 인페인팅 후처리 컴포넌트 상한(%d) 도달, 수집된 마스크 사용",
                    page_label or "?/?",
                    component_count,
                )
                break

        if local_components > 0:
            touched_blocks.append(idx)
        if page_cap_hit:
            break

    if component_count <= 0 or not np.any(residue_mask):
        return inpainted_image, mask, _empty_pass2_stats(mask.shape)

    residue_mask = imk.dilate(residue_mask, np.ones((3, 3), np.uint8), iterations=1)
    residue_mask = np.where((residue_mask > 0) & (residue_roi_union > 0), 255, 0).astype(np.uint8)
    residue_mask_pre_cap_pixel_count = int(np.count_nonzero(residue_mask))
    residue_mask = _cap_residue_mask_to_source_mask(
        residue_mask,
        mask,
        dilate_px=RESIDUE_SOURCE_MASK_DILATE_PX,
    )
    if protected.shape == residue_mask.shape and np.any(protected):
        residue_mask = np.where(
            (residue_mask > 0) & (protected <= 0),
            255,
            0,
        ).astype(np.uint8)
    residue_pass_truncated_block_count = len(truncated_block_indices)
    residue_mask_cap_pixel_count = int(np.count_nonzero(residue_mask))
    if not np.any(residue_mask):
        empty_stats = _empty_pass2_stats(mask.shape)
        empty_stats["residue_pass_truncated_block_count"] = int(
            residue_pass_truncated_block_count
        )
        return inpainted_image, mask, empty_stats

    refined_image, pass2_backend = fill_bubble_edit_mask(inpainted_image, residue_mask)
    refined_image = imk.convert_scale_abs(refined_image)
    refined_image = composite_with_edit_mask(inpainted_image, refined_image, residue_mask)
    merged_mask = np.where((mask > 0) | (residue_mask > 0), 255, 0).astype(np.uint8)
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
        "residue_pass_truncated_block_count": int(
            residue_pass_truncated_block_count
        ),
    }
