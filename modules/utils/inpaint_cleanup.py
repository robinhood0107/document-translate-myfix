from __future__ import annotations

import logging
from typing import Iterable

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import detect_content_in_bbox
from modules.utils.mask_roi import build_text_prior_mask, normalize_xyxy, resolve_block_residue_roi
from modules.utils.textblock import TextBlock

logger = logging.getLogger(__name__)


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
    }


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
) -> tuple[np.ndarray, np.ndarray, dict]:
    block_list = list(blk_list or [])
    if (
        inpainted_image is None
        or mask is None
        or not block_list
        or inpainter is None
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

    for idx, blk in enumerate(block_list):
        if getattr(blk, "xyxy", None) is None:
            continue

        residue_roi = normalize_xyxy(getattr(blk, "cleanup_roi_xyxy", None), inpainted_image.shape)
        if (getattr(blk, "text_class", "") or "") != "text_bubble":
            residue_roi = normalize_xyxy(getattr(blk, "ctd_roi_xyxy", None), inpainted_image.shape)
        if residue_roi is None:
            residue_roi = resolve_block_residue_roi(blk, inpainted_image.shape)
        if residue_roi is None:
            continue

        rx1, ry1, rx2, ry2 = residue_roi
        residue_roi_union[ry1:ry2, rx1:rx2] = 255
        crop = inpainted_image[ry1:ry2, rx1:rx2]
        if crop.size == 0:
            continue

        text_class = getattr(blk, "text_class", "") or ""
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

        for lx1, ly1, lx2, ly2 in residual_boxes:
            if local_components >= max_local_components:
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

            residue_mask[gy1:gy2, gx1:gx2] = 255
            local_components += 1
            component_count += 1
            if text_class == "text_bubble":
                bubble_kept_count += 1
            else:
                text_free_kept_count += 1
            if component_count >= 120:
                page_cap_hit = True
                logger.info("residue pass reached page component cap (%d); using collected mask", component_count)
                break

        if local_components > 0:
            touched_blocks.append(idx)
        if page_cap_hit:
            break

    if component_count <= 0 or not np.any(residue_mask):
        return inpainted_image, mask, _empty_pass2_stats(mask.shape)

    residue_mask = imk.dilate(residue_mask, np.ones((3, 3), np.uint8), iterations=1)
    residue_mask = np.where((residue_mask > 0) & (residue_roi_union > 0), 255, 0).astype(np.uint8)
    if not np.any(residue_mask):
        return inpainted_image, mask, _empty_pass2_stats(mask.shape)

    refined_image = inpainter(inpainted_image, residue_mask, config)
    refined_image = imk.convert_scale_abs(refined_image)
    merged_mask = np.where((mask > 0) | (residue_mask > 0), 255, 0).astype(np.uint8)

    logger.info(
        "residue pass applied: blocks=%s components=%d bubble_kept=%d text_free_kept=%d",
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
    }
