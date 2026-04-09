from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import copy
import json
from typing import Iterable

import cv2
import imkit as imk
import numpy as np

from modules.detection.utils.content import get_inpaint_bboxes, detect_content_in_bbox
from modules.utils.textblock import TextBlock

REGION_MONO_TONED_BUBBLE = "mono_toned_bubble"
REGION_MONO_FLAT_BUBBLE = "mono_flat_bubble"
REGION_COLOR_FLAT_BUBBLE = "color_flat_bubble"
REGION_MONO_ART_OVERLAY = "mono_art_overlay"
REGION_COLOR_ART_OVERLAY = "color_art_overlay"


@dataclass
class InpaintRegionPlan:
    block_index: int
    region_kind: str
    text_class: str
    roi: tuple[int, int, int, int]
    bbox: tuple[int, int, int, int]
    raw_bboxes: list[list[int]]
    raw_mask_local: np.ndarray
    erase_mask_local: np.ndarray
    protect_mask_local: np.ndarray
    coverage_ratio: float
    protect_overlap_ratio: float
    metrics: dict[str, float] = field(default_factory=dict)
    review_flag: bool = False


@dataclass
class InpaintMaskBundle:
    mask: np.ndarray
    raw_mask: np.ndarray
    protect_mask: np.ndarray
    plans: list[InpaintRegionPlan]
    summary: dict[str, object]


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 == 1 else value + 1


def _clip_bbox(x1: int, y1: int, x2: int, y2: int, image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x1 = max(0, min(int(x1), w))
    y1 = max(0, min(int(y1), h))
    x2 = max(0, min(int(x2), w))
    y2 = max(0, min(int(y2), h))
    return x1, y1, x2, y2


def _bbox_from_boxes(
    boxes: Iterable[Iterable[int]],
    image_shape: tuple[int, int],
    fallback_bbox: Iterable[int] | None = None,
) -> tuple[int, int, int, int] | None:
    valid = []
    for box in boxes or []:
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, image_shape)
        if x2 > x1 and y2 > y1:
            valid.append((x1, y1, x2, y2))

    if not valid and fallback_bbox is not None:
        x1, y1, x2, y2 = [int(v) for v in fallback_bbox]
        x1, y1, x2, y2 = _clip_bbox(x1, y1, x2, y2, image_shape)
        if x2 > x1 and y2 > y1:
            valid.append((x1, y1, x2, y2))

    if not valid:
        return None

    xs = [x for box in valid for x in (box[0], box[2])]
    ys = [y for box in valid for y in (box[1], box[3])]
    return min(xs), min(ys), max(xs), max(ys)


def _fill_boxes_local(
    shape: tuple[int, int],
    boxes: Iterable[Iterable[int]],
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    x1, y1, _, _ = roi
    mask = np.zeros(shape, dtype=np.uint8)
    for bx1, by1, bx2, by2 in boxes or []:
        lx1 = max(0, int(bx1) - x1)
        ly1 = max(0, int(by1) - y1)
        lx2 = min(shape[1], int(bx2) - x1)
        ly2 = min(shape[0], int(by2) - y1)
        if lx2 > lx1 and ly2 > ly1:
            mask[ly1:ly2, lx1:lx2] = 255
    return mask


def _is_color_region(crop: np.ndarray) -> bool:
    if crop is None or crop.size == 0 or crop.ndim != 3:
        return False
    hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    return float(np.mean(saturation)) > 18.0 and float(np.std(saturation)) > 8.0


def _edge_density(gray: np.ndarray, region_mask: np.ndarray | None = None) -> float:
    if gray is None or gray.size == 0:
        return 0.0
    edges = cv2.Canny(gray, 50, 150)
    if region_mask is None:
        return float(np.mean(edges > 0))
    selector = region_mask > 0
    if not np.any(selector):
        return 0.0
    return float(np.mean(edges[selector] > 0))


def _bubble_is_toned(crop: np.ndarray) -> bool:
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    std = float(np.std(gray))
    return std > 18.0 or _edge_density(gray) > 0.075


def classify_inpaint_region(crop: np.ndarray, blk: TextBlock) -> str:
    is_bubble = getattr(blk, "text_class", "") == "text_bubble" and getattr(blk, "bubble_xyxy", None) is not None
    is_color = _is_color_region(crop)
    if is_bubble:
        if is_color:
            return REGION_COLOR_FLAT_BUBBLE
        return REGION_MONO_TONED_BUBBLE if _bubble_is_toned(crop) else REGION_MONO_FLAT_BUBBLE
    return REGION_COLOR_ART_OVERLAY if is_color else REGION_MONO_ART_OVERLAY


def _region_kernel_policy(region_kind: str, median_span: float, default_padding: int) -> tuple[int, int, int]:
    span = max(3.0, float(median_span))
    if region_kind == REGION_MONO_TONED_BUBBLE:
        close_k = _odd(min(7, max(3, int(round(span * 0.08)))))
        dilate_k = _odd(max(default_padding, min(5, max(3, int(round(span * 0.05))))))
        iterations = 1
    elif region_kind == REGION_MONO_FLAT_BUBBLE:
        close_k = _odd(min(9, max(3, int(round(span * 0.10)))))
        dilate_k = _odd(max(default_padding, min(7, max(3, int(round(span * 0.08))))))
        iterations = 2
    elif region_kind == REGION_COLOR_FLAT_BUBBLE:
        close_k = _odd(min(11, max(5, int(round(span * 0.12)))))
        dilate_k = _odd(max(default_padding, min(9, max(5, int(round(span * 0.10))))))
        iterations = 2
    else:
        close_k = _odd(min(5, max(3, int(round(span * 0.05)))))
        dilate_k = _odd(min(3, max(1, int(round(span * 0.04)))))
        iterations = 1
    return close_k, dilate_k, iterations


def _local_bubble_bbox(
    blk: TextBlock,
    roi: tuple[int, int, int, int],
    shape: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    bubble_bbox = getattr(blk, "bubble_xyxy", None)
    if bubble_bbox is None or len(bubble_bbox) < 4:
        return None
    x1, y1, _, _ = roi
    bx1, by1, bx2, by2 = [int(v) for v in bubble_bbox[:4]]
    local = (
        max(0, bx1 - x1),
        max(0, by1 - y1),
        min(shape[1], bx2 - x1),
        min(shape[0], by2 - y1),
    )
    if local[2] <= local[0] or local[3] <= local[1]:
        return None
    return local


def _build_protect_mask_local(
    crop: np.ndarray,
    raw_mask_local: np.ndarray,
    region_kind: str,
    local_bubble_bbox: tuple[int, int, int, int] | None,
) -> np.ndarray:
    protect_mask = np.zeros(raw_mask_local.shape, dtype=np.uint8)

    if local_bubble_bbox is not None:
        bx1, by1, bx2, by2 = local_bubble_bbox
        thickness = max(2, min(6, int(round(min(bx2 - bx1, by2 - by1) * 0.03))))
        cv2.rectangle(protect_mask, (bx1, by1), (max(bx1, bx2 - 1), max(by1, by2 - 1)), 255, thickness=thickness)

    if region_kind in {REGION_MONO_ART_OVERLAY, REGION_COLOR_ART_OVERLAY}:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
        edges = cv2.Canny(gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        exclusion = cv2.dilate(raw_mask_local, np.ones((5, 5), np.uint8), iterations=1)
        protect_mask = np.where((edges > 0) & (exclusion == 0), 255, protect_mask).astype(np.uint8)

    return protect_mask


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask > 0)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _merge_local_mask(
    target: np.ndarray,
    local_mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = roi
    target[y1:y2, x1:x2] = np.maximum(target[y1:y2, x1:x2], local_mask)


def build_inpaint_mask_bundle(
    image: np.ndarray,
    blk_list: list[TextBlock],
    default_padding: int = 5,
) -> InpaintMaskBundle:
    h, w = image.shape[:2]
    merged_mask = np.zeros((h, w), dtype=np.uint8)
    merged_raw_mask = np.zeros((h, w), dtype=np.uint8)
    merged_protect_mask = np.zeros((h, w), dtype=np.uint8)
    plans: list[InpaintRegionPlan] = []
    region_counts: dict[str, int] = {}

    for idx, blk in enumerate(blk_list or []):
        if getattr(blk, "xyxy", None) is None:
            continue

        bboxes = get_inpaint_bboxes(
            blk.xyxy,
            image,
            bubble_bbox=getattr(blk, "bubble_xyxy", None),
        )
        if (bboxes is None or len(bboxes) == 0) and getattr(blk, "xyxy", None) is not None:
            bboxes = [list(map(int, blk.xyxy))]

        blk.inpaint_bboxes = np.array(bboxes, dtype=np.int32) if bboxes else None
        roi = _bbox_from_boxes(bboxes, image.shape[:2], fallback_bbox=getattr(blk, "xyxy", None))
        if roi is None:
            continue

        x1, y1, x2, y2 = roi
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        region_kind = classify_inpaint_region(crop, blk)
        raw_mask_local = _fill_boxes_local((y2 - y1, x2 - x1), bboxes, roi)
        bbox_widths = [max(1, int(bx2) - int(bx1)) for bx1, _by1, bx2, _by2 in bboxes]
        bbox_heights = [max(1, int(by2) - int(by1)) for _bx1, by1, _bx2, by2 in bboxes]
        median_span = max(float(np.median(bbox_widths)), float(np.median(bbox_heights)))
        close_k, dilate_k, iterations = _region_kernel_policy(region_kind, median_span, default_padding)

        if np.any(raw_mask_local):
            erase_mask_local = cv2.morphologyEx(
                raw_mask_local,
                cv2.MORPH_CLOSE,
                np.ones((close_k, close_k), np.uint8),
            )
            erase_mask_local = cv2.dilate(
                erase_mask_local,
                np.ones((dilate_k, dilate_k), np.uint8),
                iterations=iterations,
            )
        else:
            erase_mask_local = raw_mask_local.copy()

        local_bubble_bbox = _local_bubble_bbox(blk, roi, raw_mask_local.shape)
        if local_bubble_bbox is not None:
            bubble_mask = np.zeros_like(erase_mask_local)
            bx1, by1, bx2, by2 = local_bubble_bbox
            bubble_mask[by1:by2, bx1:bx2] = 255
            erase_mask_local = np.where(bubble_mask > 0, erase_mask_local, 0).astype(np.uint8)

        protect_mask_local = _build_protect_mask_local(crop, raw_mask_local, region_kind, local_bubble_bbox)
        pre_overlap = int(np.count_nonzero((erase_mask_local > 0) & (protect_mask_local > 0)))
        erase_mask_local = np.where(protect_mask_local > 0, 0, erase_mask_local).astype(np.uint8)
        if not np.any(erase_mask_local):
            erase_mask_local = raw_mask_local.copy()

        coverage_ratio = float(np.count_nonzero(erase_mask_local)) / float(max(1, (y2 - y1) * (x2 - x1)))
        protect_overlap_ratio = float(pre_overlap) / float(max(1, np.count_nonzero(raw_mask_local)))
        review_flag = region_kind in {REGION_MONO_ART_OVERLAY, REGION_COLOR_ART_OVERLAY} and protect_overlap_ratio > 0.08

        bbox = _mask_bbox(erase_mask_local)
        bbox = bbox or (0, 0, erase_mask_local.shape[1], erase_mask_local.shape[0])
        bbox = (bbox[0] + x1, bbox[1] + y1, bbox[2] + x1, bbox[3] + y1)
        metrics = {
            "median_span": float(median_span),
            "close_kernel": float(close_k),
            "dilate_kernel": float(dilate_k),
            "coverage_ratio": float(coverage_ratio),
            "protect_overlap_ratio": float(protect_overlap_ratio),
        }
        plan = InpaintRegionPlan(
            block_index=idx,
            region_kind=region_kind,
            text_class=str(getattr(blk, "text_class", "") or ""),
            roi=roi,
            bbox=bbox,
            raw_bboxes=[list(map(int, box)) for box in (bboxes or [])],
            raw_mask_local=raw_mask_local,
            erase_mask_local=erase_mask_local,
            protect_mask_local=protect_mask_local,
            coverage_ratio=coverage_ratio,
            protect_overlap_ratio=protect_overlap_ratio,
            metrics=metrics,
            review_flag=review_flag,
        )
        plans.append(plan)
        region_counts[region_kind] = region_counts.get(region_kind, 0) + 1
        _merge_local_mask(merged_mask, erase_mask_local, roi)
        _merge_local_mask(merged_raw_mask, raw_mask_local, roi)
        _merge_local_mask(merged_protect_mask, protect_mask_local, roi)

    summary = {
        "inpaint_region_kind_counts": region_counts,
        "mask_coverage_ratio": float(np.count_nonzero(merged_mask)) / float(max(1, h * w)),
        "review_flag_count": int(sum(1 for plan in plans if plan.review_flag)),
    }
    return InpaintMaskBundle(
        mask=merged_mask,
        raw_mask=merged_raw_mask,
        protect_mask=merged_protect_mask,
        plans=plans,
        summary=summary,
    )


def _build_alpha(mask: np.ndarray, feather_radius: int) -> np.ndarray:
    alpha = mask.astype(np.float32) / 255.0
    if feather_radius > 0 and np.any(mask > 0):
        kernel = _odd(max(1, feather_radius * 2 + 1))
        alpha = cv2.GaussianBlur(alpha, (kernel, kernel), feather_radius / 2.0)
        alpha = np.clip(alpha, 0.0, 1.0)
    return alpha


def composite_inpaint_result(
    inpainted: np.ndarray,
    original: np.ndarray,
    mask: np.ndarray,
    *,
    feather_radius: int = 0,
    protect_mask: np.ndarray | None = None,
) -> np.ndarray:
    mask_2d = np.array(mask, copy=True).astype(np.uint8)
    if protect_mask is not None and protect_mask.shape == mask_2d.shape:
        mask_2d = np.where(protect_mask > 0, 0, mask_2d).astype(np.uint8)

    alpha = _build_alpha(mask_2d, feather_radius)
    alpha_3d = alpha[:, :, np.newaxis]
    result = (inpainted.astype(np.float32) * alpha_3d) + (original.astype(np.float32) * (1.0 - alpha_3d))
    result = np.clip(result, 0, 255).astype(np.uint8)

    untouched = mask_2d <= 0
    result[untouched] = original[untouched]
    if protect_mask is not None and protect_mask.shape == mask_2d.shape:
        protected = protect_mask > 0
        result[protected] = original[protected]
    return result


def build_automatic_inpaint_config(base_config, bundle: InpaintMaskBundle):
    config = copy.deepcopy(base_config)
    setattr(config, "mask_feather_radius", choose_feather_radius(bundle.mask))
    setattr(config, "protect_mask", bundle.protect_mask)
    if (
        float(bundle.summary.get("mask_coverage_ratio", 0.0) or 0.0) < 0.04
        and len(bundle.plans) <= 3
        and getattr(config, "hd_strategy", "") != "Crop"
    ):
        config.hd_strategy = "Crop"
        config.hd_strategy_crop_margin = 320
        config.hd_strategy_crop_trigger_size = 384
    return config


def choose_retry_config(base_config, region_kind: str, protect_mask: np.ndarray | None = None):
    config = copy.deepcopy(base_config)
    config.hd_strategy = "Crop"
    config.hd_strategy_crop_margin = 320
    config.hd_strategy_crop_trigger_size = 384
    config.hd_strategy_resize_limit = getattr(base_config, "hd_strategy_resize_limit", 512)
    setattr(config, "mask_feather_radius", 6 if "bubble" in region_kind else 4)
    setattr(config, "protect_mask", protect_mask)
    return config


def choose_feather_radius(mask: np.ndarray) -> int:
    bbox = _mask_bbox(mask)
    if bbox is None:
        return 0
    width = max(1, bbox[2] - bbox[0])
    height = max(1, bbox[3] - bbox[1])
    span = max(width, height)
    return max(2, min(6, int(round(span * 0.015))))


def score_inpaint_artifact(
    crop: np.ndarray,
    erase_mask_local: np.ndarray,
    *,
    protect_mask_local: np.ndarray | None = None,
) -> dict[str, float]:
    if crop is None or crop.size == 0 or erase_mask_local is None or not np.any(erase_mask_local):
        return {
            "residual_component_count": 0.0,
            "inner_std": 0.0,
            "ring_std": 0.0,
            "inner_edge_density": 0.0,
            "ring_edge_density": 0.0,
            "protect_violation_ratio": 0.0,
            "should_retry": 0.0,
        }

    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if crop.ndim == 3 else crop
    inner_mask = erase_mask_local > 0
    ring_mask = cv2.dilate(erase_mask_local, np.ones((9, 9), np.uint8), iterations=1) > 0
    ring_mask = np.logical_and(ring_mask, np.logical_not(inner_mask))

    residual_boxes = detect_content_in_bbox(crop, min_area=6, margin=0)
    residual_count = 0
    roi_h, roi_w = gray.shape[:2]
    for x1, y1, x2, y2 in residual_boxes:
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(roi_w, int(x2))
        y2 = min(roi_h, int(y2))
        if x2 <= x1 or y2 <= y1:
            continue
        box_mask = np.zeros_like(erase_mask_local, dtype=np.uint8)
        box_mask[y1:y2, x1:x2] = 255
        if np.any((box_mask > 0) & inner_mask):
            comp = gray[y1:y2, x1:x2]
            if comp.size and float(np.mean(comp)) < 215:
                residual_count += 1

    inner_std = float(np.std(gray[inner_mask])) if np.any(inner_mask) else 0.0
    ring_std = float(np.std(gray[ring_mask])) if np.any(ring_mask) else inner_std
    inner_edge = _edge_density(gray, erase_mask_local)
    ring_edge = _edge_density(gray, ring_mask.astype(np.uint8) * 255) if np.any(ring_mask) else inner_edge
    protect_violation = 0.0
    if protect_mask_local is not None and protect_mask_local.shape == erase_mask_local.shape:
        protect_violation = float(
            np.count_nonzero((protect_mask_local > 0) & inner_mask)
        ) / float(max(1, np.count_nonzero(inner_mask)))

    should_retry = (
        residual_count > 0
        or inner_std > max(ring_std * 1.35, ring_std + 8.0)
        or inner_edge > max(ring_edge * 1.40, ring_edge + 0.02)
        or protect_violation > 0.0
    )
    return {
        "residual_component_count": float(residual_count),
        "inner_std": inner_std,
        "ring_std": ring_std,
        "inner_edge_density": float(inner_edge),
        "ring_edge_density": float(ring_edge),
        "protect_violation_ratio": float(protect_violation),
        "should_retry": 1.0 if should_retry else 0.0,
    }


def build_retry_mask(plan: InpaintRegionPlan) -> np.ndarray:
    local_mask = plan.raw_mask_local if "art_overlay" in plan.region_kind else plan.erase_mask_local
    mask = np.zeros((plan.roi[3] - plan.roi[1], plan.roi[2] - plan.roi[0]), dtype=np.uint8)
    mask[:] = local_mask
    return mask


def apply_local_mask_to_global(
    shape: tuple[int, int],
    local_mask: np.ndarray,
    roi: tuple[int, int, int, int],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    _merge_local_mask(mask, local_mask, roi)
    return mask


def build_overlay_summary(
    image: np.ndarray,
    erase_mask: np.ndarray,
    raw_mask: np.ndarray,
    protect_mask: np.ndarray,
) -> np.ndarray:
    overlay = image.copy()
    overlay = np.clip(overlay.astype(np.int16), 0, 255).astype(np.uint8)
    overlay[raw_mask > 0] = ((overlay[raw_mask > 0].astype(np.float32) * 0.45) + np.array([255, 191, 0], dtype=np.float32) * 0.55).astype(np.uint8)
    overlay[erase_mask > 0] = ((overlay[erase_mask > 0].astype(np.float32) * 0.35) + np.array([255, 64, 64], dtype=np.float32) * 0.65).astype(np.uint8)
    overlay[protect_mask > 0] = ((overlay[protect_mask > 0].astype(np.float32) * 0.35) + np.array([64, 160, 255], dtype=np.float32) * 0.65).astype(np.uint8)
    return overlay


def export_automatic_inpaint_debug_artifacts(
    root_dir: str,
    page_base_name: str,
    original_image: np.ndarray,
    cleaned_image: np.ndarray,
    bundle: InpaintMaskBundle,
    cleanup_mask: np.ndarray | None,
    retry_records: list[dict],
    summary: dict[str, object],
) -> None:
    def _json_safe(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_json_safe(v) for v in value]
        return value

    page_dir = Path(root_dir) / page_base_name
    roi_dir = page_dir / "rois"
    roi_dir.mkdir(parents=True, exist_ok=True)

    imk.write_image(str(page_dir / "source.png"), original_image)
    imk.write_image(str(page_dir / "cleaned.png"), cleaned_image)
    imk.write_image(str(page_dir / "mask_raw.png"), bundle.raw_mask)
    imk.write_image(str(page_dir / "mask_erase.png"), bundle.mask)
    imk.write_image(str(page_dir / "mask_protect.png"), bundle.protect_mask)
    if cleanup_mask is not None:
        imk.write_image(str(page_dir / "mask_cleanup.png"), cleanup_mask)

    retry_mask = np.zeros_like(bundle.mask, dtype=np.uint8)
    for record in retry_records or []:
        roi = tuple(record.get("roi", (0, 0, 0, 0)))
        local_mask = record.get("local_mask")
        if local_mask is not None:
            _merge_local_mask(retry_mask, local_mask, roi)
    imk.write_image(str(page_dir / "mask_retry.png"), retry_mask)
    overlay = build_overlay_summary(original_image, bundle.mask, bundle.raw_mask, bundle.protect_mask)
    imk.write_image(str(page_dir / "overlay_summary.png"), overlay)
    diff = cv2.absdiff(cleaned_image, original_image)
    imk.write_image(str(page_dir / "diff.png"), diff)

    for plan in bundle.plans:
        x1, y1, x2, y2 = plan.roi
        block_dir = roi_dir / f"block_{plan.block_index:03d}_{plan.region_kind}"
        block_dir.mkdir(parents=True, exist_ok=True)
        imk.write_image(str(block_dir / "source.png"), original_image[y1:y2, x1:x2])
        imk.write_image(str(block_dir / "cleaned.png"), cleaned_image[y1:y2, x1:x2])
        imk.write_image(str(block_dir / "raw_mask.png"), plan.raw_mask_local)
        imk.write_image(str(block_dir / "erase_mask.png"), plan.erase_mask_local)
        imk.write_image(str(block_dir / "protect_mask.png"), plan.protect_mask_local)
        record = next((item for item in retry_records or [] if int(item.get("block_index", -1)) == plan.block_index), None)
        if record is not None and record.get("local_mask") is not None:
            imk.write_image(str(block_dir / "retry_mask.png"), record["local_mask"])
        with open(block_dir / "summary.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "block_index": plan.block_index,
                    "region_kind": plan.region_kind,
                    "text_class": plan.text_class,
                    "roi": list(plan.roi),
                    "bbox": list(plan.bbox),
                    "coverage_ratio": plan.coverage_ratio,
                    "protect_overlap_ratio": plan.protect_overlap_ratio,
                    "review_flag": plan.review_flag,
                    "metrics": plan.metrics,
                    "artifact_score": record.get("artifact_score", {}) if record else {},
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    summary_payload = {}
    for key, value in (summary or {}).items():
        if key == "cleanup_mask":
            continue
        if key == "retry_records":
            summary_payload[key] = [
                {sub_key: sub_value for sub_key, sub_value in record.items() if sub_key != "local_mask"}
                for record in (value or [])
            ]
            continue
        summary_payload[key] = value

    payload = {
        "summary": summary_payload,
        "region_counts": bundle.summary.get("inpaint_region_kind_counts", {}),
        "review_flag_count": bundle.summary.get("review_flag_count", 0),
    }
    with open(page_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(_json_safe(payload), handle, ensure_ascii=False, indent=2)
