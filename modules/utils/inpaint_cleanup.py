from __future__ import annotations

import logging
from typing import Iterable

import imkit as imk
import numpy as np

from modules.detection.utils.content import detect_content_in_bbox
from modules.utils.inpaint_quality import (
    InpaintMaskBundle,
    REGION_COLOR_ART_OVERLAY,
    REGION_MONO_ART_OVERLAY,
    apply_local_mask_to_global,
    build_retry_mask,
    choose_retry_config,
    score_inpaint_artifact,
)
from modules.utils.textblock import TextBlock

logger = logging.getLogger(__name__)


def _expand_bbox(
    xyxy: list[float] | tuple[float, float, float, float],
    image_shape: tuple[int, int],
    width_ratio: float = 0.18,
    height_ratio: float = 0.26,
    min_pad: int = 12,
    clamp_xyxy: list[float] | tuple[float, float, float, float] | None = None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    img_h, img_w = image_shape[:2]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    pad_x = max(min_pad, int(round(width * width_ratio / 2.0)))
    pad_y = max(min_pad, int(round(height * height_ratio / 2.0)))

    ex1 = max(0, int(np.floor(x1 - pad_x)))
    ey1 = max(0, int(np.floor(y1 - pad_y)))
    ex2 = min(img_w, int(np.ceil(x2 + pad_x)))
    ey2 = min(img_h, int(np.ceil(y2 + pad_y)))

    if clamp_xyxy is not None and len(clamp_xyxy) >= 4:
        cx1, cy1, cx2, cy2 = [int(v) for v in clamp_xyxy[:4]]
        ex1 = max(ex1, cx1)
        ey1 = max(ey1, cy1)
        ex2 = min(ex2, cx2)
        ey2 = min(ey2, cy2)

    return ex1, ey1, ex2, ey2


def build_bubble_cleanup_mask(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    blk_list: Iterable[TextBlock],
) -> tuple[np.ndarray, dict]:
    block_list = list(blk_list or [])
    if (
        inpainted_image is None
        or mask is None
        or not block_list
        or not np.any(mask)
    ):
        return np.zeros_like(mask, dtype=np.uint8), {"applied": False, "component_count": 0, "block_count": 0}

    cleanup_zone = imk.dilate(
        (mask > 0).astype(np.uint8) * 255,
        np.ones((9, 9), np.uint8),
        iterations=2,
    )
    cleanup_mask = np.zeros_like(mask, dtype=np.uint8)
    touched_blocks: list[int] = []
    component_count = 0

    for idx, blk in enumerate(block_list):
        if getattr(blk, "text_class", "") != "text_bubble":
            continue
        if getattr(blk, "xyxy", None) is None:
            continue

        roi = _expand_bbox(
            blk.xyxy,
            inpainted_image.shape[:2],
            clamp_xyxy=getattr(blk, "bubble_xyxy", None),
        )
        x1, y1, x2, y2 = roi
        if x2 <= x1 or y2 <= y1:
            continue

        crop = inpainted_image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        residual_boxes = detect_content_in_bbox(crop, min_area=6, margin=0)
        if residual_boxes is None or len(residual_boxes) == 0:
            continue

        gray = imk.to_gray(crop)
        roi_h, roi_w = crop.shape[:2]
        roi_area = max(1, roi_w * roi_h)
        max_component_area = max(400, min(5000, int(round(roi_area * 0.08))))
        local_components = 0

        for lx1, ly1, lx2, ly2 in residual_boxes:
            w = int(lx2 - lx1)
            h = int(ly2 - ly1)
            area = w * h
            if area < 8 or area > max_component_area:
                continue
            if lx1 <= 2 or ly1 <= 2 or lx2 >= roi_w - 2 or ly2 >= roi_h - 2:
                continue
            if w > int(roi_w * 0.55) or h > int(roi_h * 0.55):
                continue

            gx1, gy1, gx2, gy2 = x1 + int(lx1), y1 + int(ly1), x1 + int(lx2), y1 + int(ly2)
            if gx2 <= gx1 or gy2 <= gy1:
                continue
            if not np.any(cleanup_zone[gy1:gy2, gx1:gx2] > 0):
                continue

            comp_gray = gray[ly1:ly2, lx1:lx2]
            if comp_gray.size == 0:
                continue
            if float(np.mean(comp_gray)) > 205 and float(np.percentile(comp_gray, 25)) > 180:
                continue

            cleanup_mask[gy1:gy2, gx1:gx2] = 255
            local_components += 1

        if local_components > 0:
            touched_blocks.append(idx)
            component_count += local_components

    if component_count <= 0 or not np.any(cleanup_mask):
        return np.zeros_like(mask, dtype=np.uint8), {"applied": False, "component_count": 0, "block_count": 0}

    cleanup_mask = imk.dilate(cleanup_mask, np.ones((3, 3), np.uint8), iterations=2)
    return cleanup_mask, {
        "applied": True,
        "component_count": component_count,
        "block_count": len(touched_blocks),
        "block_indexes": touched_blocks,
    }


def refine_bubble_residue_inpaint(
    inpainted_image: np.ndarray,
    mask: np.ndarray,
    blk_list: Iterable[TextBlock],
    inpainter,
    config,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cleanup_mask, stats = build_bubble_cleanup_mask(inpainted_image, mask, blk_list)
    if not stats.get("applied") or inpainter is None:
        return inpainted_image, mask, {"applied": False, "component_count": 0, "block_count": 0, "cleanup_mask": cleanup_mask}

    refined_image = inpainter(inpainted_image, cleanup_mask, config)
    refined_image = imk.convert_scale_abs(refined_image)
    merged_mask = np.where((mask > 0) | (cleanup_mask > 0), 255, 0).astype(np.uint8)

    logger.info(
        "inpaint residue cleanup applied: blocks=%s components=%d",
        stats.get("block_indexes", []),
        stats.get("component_count", 0),
    )
    return refined_image, merged_mask, {
        "applied": True,
        "component_count": stats.get("component_count", 0),
        "block_count": stats.get("block_count", 0),
        "block_indexes": stats.get("block_indexes", []),
        "cleanup_mask": cleanup_mask,
    }


def refine_automatic_inpaint_result(
    *,
    original_image: np.ndarray,
    inpainted_image: np.ndarray,
    bundle: InpaintMaskBundle,
    blk_list: Iterable[TextBlock],
    primary_inpainter,
    base_config,
    lama_inpainter=None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    refined_image, merged_mask, cleanup_stats = refine_bubble_residue_inpaint(
        inpainted_image,
        bundle.mask,
        blk_list,
        primary_inpainter,
        base_config,
    )

    retry_records: list[dict] = []
    review_flagged_blocks: list[int] = []
    cleanup_blocks = {
        int(idx)
        for idx in cleanup_stats.get("block_indexes", [])
        if isinstance(idx, (int, np.integer))
    }

    for plan in bundle.plans:
        x1, y1, x2, y2 = plan.roi
        crop = refined_image[y1:y2, x1:x2]
        artifact_score = score_inpaint_artifact(
            crop,
            plan.erase_mask_local,
            protect_mask_local=plan.protect_mask_local,
        )
        severe_artifact = (
            float(artifact_score.get("residual_component_count", 0.0)) > 0.0
            or (
                float(artifact_score.get("inner_std", 0.0))
                > max(
                    float(artifact_score.get("ring_std", 0.0)) * 1.6,
                    float(artifact_score.get("ring_std", 0.0)) + 12.0,
                )
                and float(artifact_score.get("inner_edge_density", 0.0))
                > max(
                    float(artifact_score.get("ring_edge_density", 0.0)) * 2.1,
                    float(artifact_score.get("ring_edge_density", 0.0)) + 0.08,
                )
            )
        )
        residual_component_count = int(artifact_score.get("residual_component_count", 0.0))
        should_retry = (
            plan.block_index in cleanup_blocks
            and severe_artifact
            and 0 < residual_component_count <= 3
        )
        if plan.review_flag:
            review_flagged_blocks.append(plan.block_index)

        if plan.region_kind in {REGION_MONO_ART_OVERLAY, REGION_COLOR_ART_OVERLAY}:
            should_retry = False

        if not should_retry:
            retry_records.append(
                {
                    "block_index": plan.block_index,
                    "region_kind": plan.region_kind,
                    "roi": list(plan.roi),
                    "artifact_score": artifact_score,
                    "applied": False,
                }
            )
            continue

        retry_inpainter = lama_inpainter or primary_inpainter
        local_retry_mask = build_retry_mask(plan)
        global_retry_mask = apply_local_mask_to_global(bundle.mask.shape, local_retry_mask, plan.roi)
        protect_mask = apply_local_mask_to_global(bundle.mask.shape, plan.protect_mask_local, plan.roi)
        retry_config = choose_retry_config(base_config, plan.region_kind, protect_mask=protect_mask)
        retried = retry_inpainter(refined_image, global_retry_mask, retry_config)
        refined_image = imk.convert_scale_abs(retried)
        merged_mask = np.where((merged_mask > 0) | (global_retry_mask > 0), 255, 0).astype(np.uint8)
        retry_records.append(
            {
                "block_index": plan.block_index,
                "region_kind": plan.region_kind,
                "roi": list(plan.roi),
                "artifact_score": artifact_score,
                "applied": True,
                "local_mask": local_retry_mask,
            }
        )

        if plan.region_kind in {REGION_MONO_ART_OVERLAY, REGION_COLOR_ART_OVERLAY}:
            review_flagged_blocks.append(plan.block_index)

    if review_flagged_blocks:
        review_flagged_blocks = sorted(set(review_flagged_blocks))

    summary = {
        "bubble_cleanup_applied": bool(cleanup_stats.get("applied")),
        "artifact_retry_count": int(sum(1 for record in retry_records if record.get("applied"))),
        "artifact_retry_blocks": [
            {
                "block_index": int(record["block_index"]),
                "region_kind": str(record["region_kind"]),
            }
            for record in retry_records
            if record.get("applied")
        ],
        "protect_mask_violation_ratio": float(
            max((plan.protect_overlap_ratio for plan in bundle.plans), default=0.0)
        ),
        "residual_text_component_count": int(
            sum(int(record.get("artifact_score", {}).get("residual_component_count", 0)) for record in retry_records)
        ),
        "review_flag_count": int(len(review_flagged_blocks)),
        "review_flagged_blocks": review_flagged_blocks,
        "inpaint_region_kind_counts": bundle.summary.get("inpaint_region_kind_counts", {}),
        "cleanup_stats": cleanup_stats,
        "retry_records": retry_records,
        "cleanup_mask": cleanup_stats.get("cleanup_mask"),
    }
    return refined_image, merged_mask, summary
