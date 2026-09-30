from __future__ import annotations

import base64
from typing import Any

import cv2
import imkit as imk
import numpy as np
from PySide6.QtGui import QColor

from modules.masking import (
    CTDRefiner,
    CTDRefinerSettings,
    build_legacy_bbox_mask_details,
    build_protect_mask,
)
from modules.masking.protect_mask import ProtectMaskSettings
from modules.masking.ctd_positive_claim import CTDPositiveClaimProvider
from modules.ocr.common.result_contract import (
    OCR_STRATEGY_PADDLE_SPOTTING,
    PROCESSING_ACTION_TRANSLATE_INPAINT,
)
from modules.utils.bubble_silhouette import extract_bubble_interior_cap_crop
from modules.utils.inpaint_composite import normalize_edit_mask
from modules.utils.inpainting_runtime import normalized_mask_refiner_settings
from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
    normalize_mask_inpaint_mode,
)
from modules.utils.mask_roi import resolve_block_ctd_roi, resolve_inpaint_text_xyxy

MASK_POLICY_VERSION = "ctd_lama_mask_policy_v5_spotting_source_glyph"
MASK_DECISION_ACCEPTED = "accepted"
MASK_DECISION_REVIEW = "review"
MASK_CANDIDATE_SOURCE_CTD_REFINED = "ctd_refined"
MASK_CANDIDATE_SOURCE_CTD_OR = "ctd_raw_refined_final_or"
MASK_CANDIDATE_SOURCE_TEXT_FREE_GLYPH_THIN = "text_free_glyph_thin"
MASK_CANDIDATE_SOURCE_NONE = "none"
MASK_REJECT_LEGACY_WINDOW_ONLY_NO_CTD_MASK = "legacy_bbox_window_only_no_ctd_mask"
MASK_REJECT_RENDER_WITHOUT_ERASE_MASK = "render_without_erase_mask"
BUBBLE_VERIFIED_INTERIOR_FINAL_DILATE_SIZE = 4
BUBBLE_VERIFIED_INTERIOR_FINAL_DILATE_SIZE = 4


def rgba2hex(rgba_list):
    r, g, b, a = [int(num) for num in rgba_list]
    return "#{:02x}{:02x}{:02x}{:02x}".format(r, g, b, a)


def encode_image_array(img_array: np.ndarray):
    img_bytes = imk.encode_image(img_array, ".png")
    return base64.b64encode(img_bytes).decode("utf-8")


def get_smart_text_color(detected_rgb: tuple, setting_color: QColor) -> QColor:
    if not detected_rgb:
        return setting_color
    try:
        detected_color = QColor(*detected_rgb)
        if not detected_color.isValid():
            return setting_color
        return detected_color
    except Exception:
        return setting_color


def _legacy_details(
    img: np.ndarray,
    blk_list,
    cfg: dict[str, Any],
    *,
    default_padding: int,
) -> dict[str, Any]:
    return build_legacy_bbox_mask_details(
        img,
        list(blk_list or []),
        cfg,
        default_padding=default_padding,
    )


def _ctd_settings_from_cfg(cfg: dict[str, Any]) -> CTDRefinerSettings:
    return CTDRefinerSettings(
        detect_size=int(cfg.get("ctd_detect_size", 1280) or 1280),
        det_rearrange_max_batches=int(cfg.get("ctd_det_rearrange_max_batches", 4) or 4),
        device=str(cfg.get("ctd_device", "cuda") or "cuda"),
        font_size_multiplier=float(cfg.get("ctd_font_size_multiplier", 1.0) or 1.0),
        font_size_max=int(cfg.get("ctd_font_size_max", -1) or -1),
        font_size_min=int(cfg.get("ctd_font_size_min", -1) or -1),
        mask_dilate_size=int(cfg.get("ctd_mask_dilate_size", 2) or 2),
    )


def _allows_ctd_hard_box_rescue(block) -> bool:
    return str(getattr(block, "text_class", "") or "") != "text_free"


def release_protected_mask_for_explicit_additions(
    protected_mask: np.ndarray | None,
    automatic_mask: np.ndarray | None,
    merged_mask: np.ndarray | None,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, int]:
    """Let persisted positive brush input override automatic corner protection."""
    protected = normalize_edit_mask(protected_mask, image_shape)
    if not np.any(protected):
        return protected, 0
    automatic = normalize_edit_mask(automatic_mask, image_shape)
    merged = normalize_edit_mask(merged_mask, image_shape)
    explicit_additions = (merged > 0) & (automatic <= 0)
    released = int(np.count_nonzero((protected > 0) & explicit_additions))
    if released <= 0:
        return protected, 0
    updated = np.where(
        (protected > 0) & ~explicit_additions,
        255,
        0,
    ).astype(np.uint8)
    return updated, released


def _build_candidate_window_mask(
    image_rgb: np.ndarray,
    block_list,
    *,
    bubble_seed_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, int, int]:
    image_shape = image_rgb.shape
    window_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_cap_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_cap_roi_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    protected_exclusion_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_window_count = 0
    bubble_silhouette_applied_count = 0
    bubble_silhouette_fallback_count = 0
    for block in list(block_list or []):
        roi = resolve_block_ctd_roi(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 <= x1 or y2 <= y1:
            continue
        text_anchor = resolve_inpaint_text_xyxy(block, image_shape)
        if text_anchor is not None:
            tx1, ty1, tx2, ty2 = text_anchor
            protected_exclusion_mask[ty1:ty2, tx1:tx2] = 255
        is_bubble = (
            str(getattr(block, "text_class", "") or "") == "text_bubble"
            and getattr(block, "bubble_xyxy", None) is not None
        )
        if is_bubble:
            bubble_window_count += 1
            bubble_seed = _build_block_bubble_seed_crop(
                bubble_seed_mask,
                block,
                image_shape,
            )
            cap_crop = None
            if bubble_seed is not None:
                bubble_roi, seed_crop = bubble_seed
                bx1, by1, bx2, by2 = bubble_roi
                cap_crop = extract_bubble_interior_cap_crop(
                    np.ascontiguousarray(image_rgb[by1:by2, bx1:bx2]),
                    seed_crop,
                )
            if cap_crop is not None:
                cap_crop = np.where(cap_crop > 0, 255, 0).astype(np.uint8)
                window_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
                    window_mask[by1:by2, bx1:bx2],
                    cap_crop,
                )
                bubble_cap_mask[by1:by2, bx1:bx2] = cv2.bitwise_or(
                    bubble_cap_mask[by1:by2, bx1:bx2],
                    cap_crop,
                )
                bubble_cap_roi_mask[by1:by2, bx1:bx2] = 255
                bubble_silhouette_applied_count += 1
                continue
            bubble_silhouette_fallback_count += 1
        window_mask[y1:y2, x1:x2] = 255
        protected_exclusion_mask[y1:y2, x1:x2] = 255
    protected_corner_mask = np.where(
        (bubble_cap_roi_mask > 0)
        & (bubble_cap_mask <= 0)
        & (protected_exclusion_mask <= 0),
        255,
        0,
    ).astype(np.uint8)
    return (
        window_mask,
        bubble_window_count,
        bubble_cap_mask,
        protected_corner_mask,
        bubble_silhouette_applied_count,
        bubble_silhouette_fallback_count,
    )


def _build_block_bubble_seed_crop(
    seed_mask: np.ndarray | None,
    block,
    image_shape: tuple[int, ...],
) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
    if seed_mask is None:
        return None
    source = np.asarray(seed_mask)
    if source.ndim == 3:
        source = source[:, :, 0]
    if source.shape[:2] != image_shape[:2]:
        return None
    bubble_roi = _normalize_xyxy_for_shape(
        getattr(block, "bubble_xyxy", None),
        image_shape,
    )
    if bubble_roi is None:
        return None
    x1, y1, x2, y2 = bubble_roi
    crop = np.where(
        source[y1:y2, x1:x2] > 0,
        255,
        0,
    ).astype(np.uint8)
    return bubble_roi, np.ascontiguousarray(crop)


def _dilate_final_mask(mask: np.ndarray, size: int) -> np.ndarray:
    mask_arr = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if int(size) <= 0 or mask_arr.size == 0 or not np.any(mask_arr):
        return mask_arr
    radius = int(size)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (radius * 2 + 1, radius * 2 + 1),
        (radius, radius),
    )
    return np.where(cv2.dilate(mask_arr, kernel, iterations=1) > 0, 255, 0).astype(np.uint8)


def _normalize_xyxy_for_shape(box, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    try:
        x1, y1, x2, y2 = [int(float(v)) for v in list(box)[:4]]
    except Exception:
        return None
    img_h, img_w = image_shape[:2]
    x1 = max(0, min(img_w, x1))
    x2 = max(0, min(img_w, x2))
    y1 = max(0, min(img_h, y1))
    y2 = max(0, min(img_h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _mask_bbox(mask: np.ndarray, *, offset_x: int = 0, offset_y: int = 0) -> tuple[int, int, int, int] | None:
    coords = cv2.findNonZero(np.where(mask > 0, 255, 0).astype(np.uint8))
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w <= 0 or h <= 0:
        return None
    return offset_x + int(x), offset_y + int(y), offset_x + int(x + w), offset_y + int(y + h)


def _build_text_free_window_mask(
    image_shape: tuple[int, ...],
    block_list,
) -> tuple[np.ndarray, int]:
    window_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    count = 0
    for block in list(block_list or []):
        if str(getattr(block, "text_class", "") or "") != "text_free":
            continue
        roi = resolve_block_ctd_roi(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 <= x1 or y2 <= y1:
            continue
        window_mask[y1:y2, x1:x2] = 255
        count += 1
    return window_mask, count


def _dilate_ctd_final_mask_by_block_policy(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    block_list,
    *,
    final_dilate_size: int,
    text_free_dilate_size: int,
) -> tuple[np.ndarray, int]:
    mask_arr = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if int(final_dilate_size) <= 0 or mask_arr.size == 0 or not np.any(mask_arr):
        return mask_arr, 0
    text_free_window, text_free_window_count = _build_text_free_window_mask(image_shape, block_list)
    if not np.any(text_free_window):
        return _dilate_final_mask(mask_arr, final_dilate_size), 0

    text_free_mask = np.where((mask_arr > 0) & (text_free_window > 0), 255, 0).astype(np.uint8)
    other_mask = np.where((mask_arr > 0) & (text_free_window <= 0), 255, 0).astype(np.uint8)
    dilated_other = _dilate_final_mask(other_mask, final_dilate_size)
    dilated_text_free = _dilate_final_mask(text_free_mask, max(0, int(text_free_dilate_size)))
    merged = np.where((dilated_other > 0) | (dilated_text_free > 0), 255, 0).astype(np.uint8)
    return merged, text_free_window_count if np.any(text_free_mask) else 0


def annotate_block_mask_attribution(
    block_list,
    final_mask: np.ndarray | None,
    image_shape: tuple[int, ...],
    *,
    candidate_source: str = MASK_CANDIDATE_SOURCE_CTD_REFINED,
) -> None:
    if final_mask is None:
        return
    mask = np.where(np.asarray(final_mask) > 0, 255, 0).astype(np.uint8)
    for block in list(block_list or []):
        roi = resolve_block_ctd_roi(block, image_shape)
        setattr(block, "_final_mask_pixel_count", 0)
        setattr(block, "block_final_mask_pixel_count", 0)
        setattr(block, "block_mask_iou", 0.0)
        setattr(block, "block_mask_span_coverage", 0.0)
        setattr(block, "block_mask_bbox", None)
        setattr(block, "block_mask_source", MASK_CANDIDATE_SOURCE_NONE)
        setattr(block, "block_mask_decision", MASK_DECISION_REVIEW)
        setattr(block, "mask_actual_pixel_count", 0)
        setattr(block, "mask_actual_bbox", None)
        setattr(block, "mask_strategy_reason", "no_final_mask_attribution")
        if roi is None:
            if str(getattr(block, "text_class", "") or "") == "text_free":
                setattr(block, "mask_decision", MASK_DECISION_REVIEW)
                setattr(block, "mask_reject_reason", MASK_REJECT_RENDER_WITHOUT_ERASE_MASK)
            continue
        x1, y1, x2, y2 = roi
        crop = mask[y1:y2, x1:x2]
        count = int(np.count_nonzero(crop))
        bbox = _mask_bbox(crop, offset_x=x1, offset_y=y1)
        roi_area = max(1, (x2 - x1) * (y2 - y1))
        setattr(block, "_final_mask_pixel_count", count)
        setattr(block, "block_final_mask_pixel_count", count)
        setattr(block, "block_mask_iou", float(count) / float(roi_area))
        setattr(block, "block_mask_bbox", [int(v) for v in bbox] if bbox is not None else None)
        setattr(block, "mask_actual_pixel_count", count)
        setattr(
            block,
            "mask_actual_bbox",
            [int(v) for v in bbox] if bbox is not None else None,
        )
        if count > 0:
            setattr(block, "block_mask_source", candidate_source or MASK_CANDIDATE_SOURCE_CTD_REFINED)
            setattr(block, "block_mask_decision", MASK_DECISION_ACCEPTED)
            setattr(
                block,
                "mask_strategy_reason",
                candidate_source or MASK_CANDIDATE_SOURCE_CTD_REFINED,
            )
        if bool(getattr(block, "bubble_panel_text_candidate", False)):
            setattr(block, "bubble_panel_mask_pixel_count", count)
            setattr(
                block,
                "bubble_panel_mask_source",
                candidate_source or (MASK_CANDIDATE_SOURCE_CTD_REFINED if count > 0 else MASK_CANDIDATE_SOURCE_NONE),
            )
        source_box = resolve_inpaint_text_xyxy(block, image_shape)
        if bbox is not None and source_box is not None:
            source_w = max(1, source_box[2] - source_box[0])
            source_h = max(1, source_box[3] - source_box[1])
            mask_w = max(1, bbox[2] - bbox[0])
            mask_h = max(1, bbox[3] - bbox[1])
            if source_h >= source_w * 1.25:
                span = min(1.0, float(mask_h) / float(source_h))
            else:
                span = min(1.0, float(mask_w) / float(source_w))
            setattr(block, "block_mask_span_coverage", span)
        if str(getattr(block, "text_class", "") or "") == "text_free":
            if count <= 0:
                setattr(block, "mask_decision", MASK_DECISION_REVIEW)
                setattr(block, "mask_reject_reason", MASK_REJECT_RENDER_WITHOUT_ERASE_MASK)
            elif not str(getattr(block, "mask_decision", "") or ""):
                setattr(block, "mask_decision", MASK_DECISION_ACCEPTED)
                setattr(block, "mask_reject_reason", "")


def restore_original_for_block_masks(
    original_image: np.ndarray,
    cleaned_image: np.ndarray,
    final_mask: np.ndarray | None,
    blocks,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if original_image is None or cleaned_image is None or final_mask is None:
        return cleaned_image, final_mask, {"applied": False, "block_count": 0, "pixel_count": 0, "block_indices": []}
    restore_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
    restored_indices: list[int] = []
    mask = np.where(np.asarray(final_mask) > 0, 255, 0).astype(np.uint8)
    for index, block in enumerate(list(blocks or [])):
        roi = resolve_block_ctd_roi(block, original_image.shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        block_mask = mask[y1:y2, x1:x2]
        if not np.any(block_mask):
            continue
        restore_mask[y1:y2, x1:x2] = np.where(block_mask > 0, 255, restore_mask[y1:y2, x1:x2]).astype(np.uint8)
        setattr(block, "_render_restore_applied", True)
        restored_indices.append(index)
    if not np.any(restore_mask):
        return cleaned_image, final_mask, {"applied": False, "block_count": 0, "pixel_count": 0, "block_indices": []}
    restored = np.asarray(cleaned_image).copy()
    restored[restore_mask > 0] = np.asarray(original_image)[restore_mask > 0]
    updated_mask = np.where((mask > 0) & (restore_mask <= 0), 255, 0).astype(np.uint8)
    pixel_count = int(np.count_nonzero(restore_mask))
    return restored, updated_mask, {
        "applied": True,
        "block_count": len(restored_indices),
        "pixel_count": pixel_count,
        "block_indices": restored_indices,
    }


def _ctd_details(
    img: np.ndarray,
    blk_list,
    cfg: dict[str, Any],
    *,
    default_padding: int,
) -> dict[str, Any]:
    block_list = list(blk_list or [])
    ctd = CTDRefiner(_ctd_settings_from_cfg(cfg))
    ctd_result = ctd.refine(img, block_list)
    raw_mask = np.where(np.asarray(ctd_result.raw_mask) > 0, 255, 0).astype(np.uint8)
    refined_mask = np.where(np.asarray(ctd_result.refined_mask) > 0, 255, 0).astype(np.uint8)
    ctd_final_mask = np.where(np.asarray(ctd_result.final_mask) > 0, 255, 0).astype(np.uint8)
    ctd_or_mask = np.where(
        (raw_mask > 0) | (refined_mask > 0) | (ctd_final_mask > 0),
        255,
        0,
    ).astype(np.uint8)
    positive_claim_raw_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    positive_claim_runtime: dict[str, Any] = {
        "status": "disabled",
        "provider": "ctd_fixed1280_onnx",
    }
    has_detector_provenance = any(
        str(getattr(block, "detector_origin", "") or "")
        in {"direct_text", "bubble_text_rescue"}
        for block in block_list
    )
    if bool(cfg.get("positive_text_evidence_enabled", True)) and has_detector_provenance:
        try:
            positive_provider = CTDPositiveClaimProvider(
                device=str(cfg.get("ctd_device", "cuda") or "cuda"),
                detect_size=int(cfg.get("positive_claim_detect_size", 1280) or 1280),
                max_batch_size=int(
                    cfg.get("ctd_det_rearrange_max_batches", 4) or 4
                ),
            )
            positive_result = positive_provider.infer(img)
            positive_claim_raw_mask = positive_result.raw_mask
            positive_claim_runtime = {
                "status": "completed",
                "provider": "ctd_fixed1280_onnx",
                "providers": list(positive_result.providers),
                "detect_size": int(positive_result.detect_size),
                "model_sha256": positive_result.model_sha256,
                "model_opset": int(positive_result.model_opset),
                "pixel_count": int(np.count_nonzero(positive_result.raw_mask)),
            }
        except Exception as exc:
            positive_claim_runtime = {
                "status": "failed",
                "provider": "ctd_fixed1280_onnx",
                "error_type": type(exc).__name__,
            }
    protect_mask = build_protect_mask(
        img,
        block_list,
        ProtectMaskSettings(
            keep_existing_lines=bool(cfg.get("keep_existing_lines", True)),
        ),
    )
    protected_mask = np.where(
        (ctd_or_mask > 0) & (np.asarray(protect_mask) <= 0),
        255,
        0,
    ).astype(np.uint8)
    fallback_used = bool(ctd_result.fallback_used)
    refiner_backend = str(ctd_result.backend or "ctd")
    final_mask = protected_mask
    legacy_fallback_details: dict[str, Any] | None = None
    legacy_rescue_details: dict[str, Any] | None = None
    hard_box_rescue_used = False
    mask_candidate_source = MASK_CANDIDATE_SOURCE_NONE
    mask_decision = MASK_DECISION_REVIEW
    mask_reject_reason = MASK_REJECT_LEGACY_WINDOW_ONLY_NO_CTD_MASK

    hard_box_rescue_blocks = [block for block in block_list if _allows_ctd_hard_box_rescue(block)]
    if hard_box_rescue_blocks:
        legacy_rescue_details = _legacy_details(
            img,
            hard_box_rescue_blocks,
            cfg,
            default_padding=default_padding,
        )

    if not np.any(final_mask) and np.any(ctd_or_mask):
        final_mask = ctd_or_mask.copy()
        fallback_used = True
        refiner_backend = f"{refiner_backend}+protect_fallback"

    if not np.any(final_mask) and block_list:
        legacy_fallback_details = legacy_rescue_details or _legacy_details(
            img,
            block_list,
            cfg,
            default_padding=default_padding,
        )
        fallback_used = True
        refiner_backend = f"{refiner_backend}+legacy_bbox_window_only"

    if np.any(final_mask):
        mask_candidate_source = MASK_CANDIDATE_SOURCE_CTD_OR
        mask_decision = MASK_DECISION_ACCEPTED
        mask_reject_reason = ""

    details = {
        "raw_mask": raw_mask,
        "positive_claim_raw_mask": positive_claim_raw_mask,
        "positive_claim_runtime": positive_claim_runtime,
        "refined_mask": refined_mask,
        "protect_mask": np.where(np.asarray(protect_mask) > 0, 255, 0).astype(np.uint8),
        "ctd_or_mask_pixel_count": int(np.count_nonzero(ctd_or_mask)),
        "final_mask_pre_expand": final_mask.copy(),
        "final_mask_post_expand": final_mask.copy(),
        "final_mask": final_mask.copy(),
        "legacy_base_mask": None,
        "hard_box_rescue_mask": None,
        "hard_box_applied_count": 0,
        "hard_box_reason_totals": {},
        "legacy_base_mask_pixel_count": 0,
        "hard_box_rescue_mask_pixel_count": 0,
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
        "mask_refiner": "ctd",
        "keep_existing_lines": bool(cfg.get("keep_existing_lines", True)),
        "refiner_backend": refiner_backend,
        "refiner_device": str(cfg.get("ctd_device", "cuda") or "cuda"),
        "fallback_used": fallback_used,
        "hard_box_rescue_used": hard_box_rescue_used,
        "mask_inpaint_mode": str(cfg.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE) or DEFAULT_MASK_INPAINT_MODE),
        "mask_policy_version": MASK_POLICY_VERSION,
        "mask_candidate_source": mask_candidate_source,
        "mask_decision": mask_decision,
        "mask_reject_reason": mask_reject_reason,
        "mask_score_outside_change": 0.0,
        "mask_score_outline_damage": 0.0,
        "mask_score_residue": 0.0,
        "mask_score_color_delta": 0.0,
        "mask_policy_bubble_clamp_applied_count": 0,
        "mask_policy_removed_pixel_count": 0,
        "mask_policy_outside_bubble_removed_pixel_count": 0,
        "mask_policy_protected_pixel_removed_count": 0,
        "legacy_bbox_role": "window_only",
        "legacy_bbox_direct_erase_disabled": True,
        "ctd_legacy_rectangle_rescue_disabled": True,
    }
    legacy_details = legacy_fallback_details or legacy_rescue_details
    if legacy_details:
        details["legacy_base_mask"] = legacy_details.get("legacy_base_mask")
        details["hard_box_rescue_mask"] = legacy_details.get("hard_box_rescue_mask")
        details["hard_box_applied_count"] = int(legacy_details.get("hard_box_applied_count", 0) or 0)
        details["hard_box_reason_totals"] = dict(legacy_details.get("hard_box_reason_totals", {}) or {})
        details["legacy_base_mask_pixel_count"] = int(legacy_details.get("legacy_base_mask_pixel_count", 0) or 0)
        details["hard_box_rescue_mask_pixel_count"] = int(legacy_details.get("hard_box_rescue_mask_pixel_count", 0) or 0)
    return details


def apply_spotting_source_glyph_extension(
    img: np.ndarray,
    blk_list,
    details: dict[str, Any],
) -> dict[str, Any]:
    # Only native Spotting quadrilaterals can authorize this narrow addition.
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "disabled",
        "geometry_source": "paddle_spotting_normalized_quad",
        "blocks_seen": 0,
        "blocks_extended": 0,
        "lines_seen": 0,
        "lines_extended": 0,
        "lines_rejected": 0,
        "added_pixels": 0,
        "inside_line_added_pixels": 0,
        "protected_candidate_pixels": 0,
        "line_diagnostics": [],
    }
    if (
        str(details.get("mask_refiner", "") or "") != "ctd"
        or img is None
        or not isinstance(img, np.ndarray)
        or img.ndim < 2
    ):
        details["source_glyph_extension"] = summary
        return summary

    base_mask = normalize_edit_mask(details.get("final_mask"), img.shape)
    if not np.any(base_mask):
        summary["status"] = "missing_base_mask"
        details["source_glyph_extension"] = summary
        return summary
    protect = normalize_edit_mask(details.get("protect_mask"), img.shape)
    protected_corner = normalize_edit_mask(
        details.get("protected_corner_mask"), img.shape
    )
    protected = np.where(
        (protect > 0) | (protected_corner > 0), 255, 0
    ).astype(np.uint8)
    bubble_cap = normalize_edit_mask(
        details.get("bubble_interior_cap_mask"), img.shape
    )
    source = np.asarray(img[:, :, :3] if img.ndim == 3 else img)
    if source.ndim == 2:
        source = np.repeat(source[:, :, None], 3, axis=2)
    if source.dtype != np.uint8:
        source = np.clip(source, 0, 255).astype(np.uint8)
    height, width = source.shape[:2]
    selected = np.zeros((height, width), dtype=np.uint8)
    allowed_source_bands = np.zeros((height, width), dtype=np.uint8)
    line_diagnostics: list[dict[str, Any]] = []

    for block in list(blk_list or []):
        if (
            str(getattr(block, "ocr_strategy", "") or "")
            != OCR_STRATEGY_PADDLE_SPOTTING
            or str(getattr(block, "processing_action", "") or "")
            != PROCESSING_ACTION_TRANSLATE_INPAINT
        ):
            continue
        provenance = dict(getattr(block, "ocr_geometry_provenance", {}) or {})
        if provenance.get("source") != "paddle_spotting_normalized_quad":
            continue
        regions = list(getattr(block, "ocr_regions", []) or [])
        if not regions:
            continue
        block_id = str(getattr(block, "block_id", "") or "")
        is_bubble = str(getattr(block, "text_class", "") or "").lower() == "text_bubble"
        if is_bubble and not np.any(bubble_cap):
            for region in regions:
                summary["lines_seen"] += 1
                summary["lines_rejected"] += 1
                line_diagnostics.append({
                    "block_id": block_id,
                    "source_line": region.get("source_line"),
                    "status": "verified_bubble_interior_unavailable",
                    "added_pixels": 0,
                })
            continue
        summary["blocks_seen"] += 1
        block_added_before = int(np.count_nonzero(selected))

        for region in regions:
            summary["lines_seen"] += 1
            line_no = region.get("source_line")
            bbox = region.get("bbox_xyxy")
            points = region.get("points")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                summary["lines_rejected"] += 1
                line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "invalid_bbox", "added_pixels": 0})
                continue
            x1, y1, x2, y2 = [int(round(float(v))) for v in bbox]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width, x2), min(height, y2)
            if x2 <= x1 or y2 <= y1:
                summary["lines_rejected"] += 1
                line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "empty_bbox", "added_pixels": 0})
                continue

            orientation = ""
            if isinstance(points, (list, tuple)) and len(points) >= 4:
                try:
                    quad = np.asarray(points, dtype=np.float64).reshape(-1, 2)
                    vectors = [quad[(i + 1) % len(quad)] - quad[i] for i in range(len(quad))]
                    major = max(vectors, key=lambda v: float(np.linalg.norm(v)))
                    angle = float(np.degrees(np.arctan2(abs(float(major[1])), max(abs(float(major[0])), 1e-6))))
                    if angle <= 15.0:
                        orientation = "horizontal"
                    elif angle >= 75.0:
                        orientation = "vertical"
                    else:
                        summary["lines_rejected"] += 1
                        line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "oblique_geometry_review", "angle_degrees": round(angle, 3), "added_pixels": 0})
                        continue
                except (TypeError, ValueError):
                    orientation = ""
            if not orientation:
                direction = str(getattr(block, "direction", "") or "").lower()
                if direction.startswith("vertical"):
                    orientation = "vertical"
                elif direction.startswith(("horizontal", "ltr", "rtl")):
                    orientation = "horizontal"
                else:
                    summary["lines_rejected"] += 1
                    line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "missing_axis_geometry", "added_pixels": 0})
                    continue

            cross_extent = (y2 - y1) if orientation == "horizontal" else (x2 - x1)
            cap = min(64, max(24, int(round(0.75 * cross_extent))))
            if orientation == "horizontal":
                ylo, yhi = max(0, y1 - 2), min(height, y2 + 2)
                xlo, xhi = max(0, x1 - cap), min(width, x2 + cap)
                left_width, right_start = x1 - xlo, x2 - xlo
                side = np.zeros((yhi - ylo, xhi - xlo), dtype=bool)
                if left_width > 0:
                    side[:, :left_width] = True
                if right_start < side.shape[1]:
                    side[:, right_start:] = True
            else:
                xlo, xhi = max(0, x1 - 2), min(width, x2 + 2)
                ylo, yhi = max(0, y1 - cap), min(height, y2 + cap)
                top_height, bottom_start = y1 - ylo, y2 - ylo
                side = np.zeros((yhi - ylo, xhi - xlo), dtype=bool)
                if top_height > 0:
                    side[:top_height, :] = True
                if bottom_start < side.shape[0]:
                    side[bottom_start:, :] = True

            roi_source = source[ylo:yhi, xlo:xhi]
            roi_base = base_mask[ylo:yhi, xlo:xhi] > 0
            roi_protect = protected[ylo:yhi, xlo:xhi] > 0
            roi_allowed = side.copy()
            if is_bubble:
                roi_allowed &= bubble_cap[ylo:yhi, xlo:xhi] > 0
            sample_mask = roi_allowed & ~roi_base & ~roi_protect
            samples = roi_source[sample_mask].astype(np.float32)
            if samples.shape[0] < 100:
                summary["lines_rejected"] += 1
                line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "insufficient_background_samples", "sample_pixels": int(samples.shape[0]), "added_pixels": 0})
                continue
            background = np.median(samples, axis=0)
            contrast = np.max(np.abs(roi_source.astype(np.float32) - background[None, None, :]), axis=2)
            selected_band = np.zeros_like(side, dtype=bool)
            extents: dict[str, int] = {}
            if orientation == "horizontal":
                if left_width > 0:
                    left_fg = contrast[:, :left_width] >= 16
                    extent = left_width - int(np.where(left_fg)[1].min()) if np.any(left_fg) else 0
                    selected_width = min(left_width, cap, extent + 2) if extent else 0
                    extents["left"] = extent
                    if selected_width:
                        selected_band[:, left_width - selected_width:left_width] = True
                if right_start < contrast.shape[1]:
                    right_fg = contrast[:, right_start:] >= 16
                    extent = int(np.where(right_fg)[1].max() + 1) if np.any(right_fg) else 0
                    selected_width = min(contrast.shape[1] - right_start, cap, extent + 2) if extent else 0
                    extents["right"] = extent
                    if selected_width:
                        selected_band[:, right_start:right_start + selected_width] = True
            else:
                if top_height > 0:
                    top_fg = contrast[:top_height, :] >= 16
                    extent = top_height - int(np.where(top_fg)[0].min()) if np.any(top_fg) else 0
                    selected_height = min(top_height, cap, extent + 2) if extent else 0
                    extents["top"] = extent
                    if selected_height:
                        selected_band[top_height - selected_height:top_height, :] = True
                if bottom_start < contrast.shape[0]:
                    bottom_fg = contrast[bottom_start:, :] >= 16
                    extent = int(np.where(bottom_fg)[0].max() + 1) if np.any(bottom_fg) else 0
                    selected_height = min(contrast.shape[0] - bottom_start, cap, extent + 2) if extent else 0
                    extents["bottom"] = extent
                    if selected_height:
                        selected_band[bottom_start:bottom_start + selected_height, :] = True

            before_protect = selected_band & side & ~roi_base & roi_allowed
            potential_protect = int(np.count_nonzero(before_protect & roi_protect))
            summary["protected_candidate_pixels"] += potential_protect
            near_mask = before_protect & ~roi_protect
            near_pixels = roi_source[near_mask].astype(np.float32)
            near_ratio = float(np.mean(np.max(np.abs(near_pixels - background[None, :]), axis=1) <= 8)) if near_pixels.shape[0] else 1.0
            if near_ratio < 0.55:
                summary["lines_rejected"] += 1
                line_diagnostics.append({"block_id": block_id, "source_line": line_no, "status": "source_background_not_separable", "near_background_ratio": round(near_ratio, 6), "added_pixels": 0})
                continue
            local_foreground = (contrast >= 8) & near_mask
            side_added = int(np.count_nonzero(local_foreground))

            # A native OCR quad can include real source glyph pixels that a broader
            # detector mask missed. Recover only foreground at the line's two ends,
            # inside the quad, near the base mask, and outside every protect mask.
            # The previous path inspected only pixels outside the quad and left such
            # clipped edge glyphs untouched while reporting the line as complete.
            inside_endpoint = np.zeros_like(side, dtype=bool)
            native_quad_valid = False
            if isinstance(points, (list, tuple)) and len(points) >= 4:
                try:
                    quad = np.asarray(points, dtype=np.float64).reshape(-1, 2)
                    if len(quad) >= 4 and np.all(np.isfinite(quad)):
                        quad_roi = np.rint(quad - np.asarray([xlo, ylo])).astype(np.int32)
                        polygon = np.zeros(side.shape, dtype=np.uint8)
                        cv2.fillPoly(polygon, [quad_roi], 255)
                        native_quad_valid = bool(np.any(polygon))
                        if native_quad_valid:
                            if orientation == "horizontal":
                                left_edge = x1 - xlo
                                right_edge = x2 - xlo
                                inner_width = min(cap, max(1, x2 - x1))
                                inside_endpoint[:, left_edge:min(side.shape[1], left_edge + inner_width)] = True
                                inside_endpoint[:, max(0, right_edge - inner_width):right_edge] = True
                            else:
                                top_edge = y1 - ylo
                                bottom_edge = y2 - ylo
                                inner_height = min(cap, max(1, y2 - y1))
                                inside_endpoint[top_edge:min(side.shape[0], top_edge + inner_height), :] = True
                                inside_endpoint[max(0, bottom_edge - inner_height):bottom_edge, :] = True
                            inside_endpoint &= polygon > 0
                except (TypeError, ValueError):
                    native_quad_valid = False

            if is_bubble:
                inside_endpoint &= bubble_cap[ylo:yhi, xlo:xhi] > 0
            inside_candidates = (
                (contrast >= 16)
                & inside_endpoint
                & ~roi_base
                & ~roi_protect
            ) if native_quad_valid else np.zeros_like(side, dtype=bool)
            inside_added = int(np.count_nonzero(inside_candidates))
            added = side_added + inside_added
            if added:
                selected_roi = selected[ylo:yhi, xlo:xhi]
                selected_roi[:] = np.maximum(
                    selected_roi,
                    np.where(local_foreground | inside_candidates, 255, 0).astype(np.uint8),
                )
                allowed_roi = allowed_source_bands[ylo:yhi, xlo:xhi]
                allowed_roi[:] = np.maximum(
                    allowed_roi,
                    np.where(before_protect | inside_endpoint, 255, 0).astype(np.uint8),
                )
                summary["lines_extended"] += 1
            line_diagnostics.append({
                "block_id": block_id,
                "source_line": line_no,
                "orientation": orientation,
                "bbox_xyxy": [x1, y1, x2, y2],
                "search_cap_px": cap,
                "foreground_extents_px": extents,
                "near_background_ratio": round(near_ratio, 6),
                "source_foreground_pixels": added,
                "outside_quad_source_foreground_pixels": side_added,
                "inside_quad_endpoint_source_foreground_pixels": inside_added,
                "status": "extended" if added else "no_unmasked_source_foreground",
                "added_pixels": added,
            })
            summary["inside_line_added_pixels"] += inside_added
        if int(np.count_nonzero(selected)) > block_added_before:
            summary["blocks_extended"] += 1

    dilated = cv2.dilate(selected, np.ones((3, 3), np.uint8), iterations=1) > 0
    addition = dilated & (allowed_source_bands > 0) & (base_mask == 0) & (protected == 0)
    added_count = int(np.count_nonzero(addition))
    if added_count:
        final_mask = np.where((base_mask > 0) | addition, 255, 0).astype(np.uint8)
        details["final_mask"] = final_mask
        details["final_mask_post_expand"] = final_mask.copy()
        details["final_mask_pixel_count"] = int(np.count_nonzero(final_mask))
        existing_source = str(details.get("mask_candidate_source", "") or "ctd_refined")
        details["mask_candidate_source"] = existing_source + "+spotting_source_glyph"
        summary["status"] = "completed"
    elif summary["lines_rejected"]:
        summary["status"] = "review_required"
    else:
        summary["status"] = "no_source_supported_additions"
    summary["added_pixels"] = added_count
    details["source_glyph_extension"] = summary
    return summary

def generate_mask(
    img: np.ndarray,
    blk_list,
    default_padding: int = 5,
    settings: dict[str, Any] | None = None,
    return_details: bool = False,
    precomputed_mask_details: dict[str, Any] | None = None,
):
    del precomputed_mask_details

    cfg = normalized_mask_refiner_settings(settings)
    cfg["mask_inpaint_mode"] = normalize_mask_inpaint_mode(
        cfg.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE)
    )
    try:
        if str(cfg.get("mask_refiner", "ctd") or "ctd") == "legacy_bbox":
            details = _legacy_details(
                img,
                blk_list,
                cfg,
                default_padding=default_padding,
            )
        else:
            details = _ctd_details(
                img,
                blk_list,
                cfg,
                default_padding=default_padding,
            )
    except Exception:
        if str(cfg.get("mask_refiner", "ctd") or "ctd") == "legacy_bbox":
            raise
        legacy_details = _legacy_details(
            img,
            blk_list,
            cfg,
            default_padding=default_padding,
        )
        details = dict(legacy_details)
        empty_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        details["raw_mask"] = empty_mask.copy()
        details["positive_claim_raw_mask"] = empty_mask.copy()
        details["positive_claim_runtime"] = {
            "status": "unavailable",
            "provider": "ctd_fixed1280_onnx",
        }
        details["refined_mask"] = empty_mask.copy()
        details["protect_mask"] = empty_mask.copy()
        details["final_mask_pre_expand"] = empty_mask.copy()
        details["final_mask_post_expand"] = empty_mask.copy()
        details["final_mask"] = empty_mask.copy()
        details["final_mask_pixel_count"] = 0
        details["mask_refiner"] = "ctd"
        details["keep_existing_lines"] = bool(cfg.get("keep_existing_lines", True))
        details["refiner_backend"] = "ctd+legacy_bbox_exception_window_only"
        details["refiner_device"] = str(cfg.get("ctd_device", "cuda") or "cuda")
        details["fallback_used"] = True
        details["hard_box_rescue_used"] = False
        details["mask_inpaint_mode"] = cfg["mask_inpaint_mode"]
        details["mask_policy_version"] = MASK_POLICY_VERSION
        details["mask_candidate_source"] = MASK_CANDIDATE_SOURCE_NONE
        details["mask_decision"] = MASK_DECISION_REVIEW
        details["mask_reject_reason"] = "ctd_exception_legacy_bbox_window_only"
        details["mask_score_outside_change"] = 0.0
        details["mask_score_outline_damage"] = 0.0
        details["mask_score_residue"] = 0.0
        details["mask_score_color_delta"] = 0.0
        details["mask_policy_bubble_clamp_applied_count"] = 0
        details["mask_policy_removed_pixel_count"] = 0
        details["mask_policy_outside_bubble_removed_pixel_count"] = 0
        details["legacy_bbox_role"] = "window_only"
        details["legacy_bbox_direct_erase_disabled"] = True
        details["ctd_legacy_rectangle_rescue_disabled"] = True

    final_dilate_size = int(cfg.get("final_mask_dilate_size", 8) or 0)
    if final_dilate_size > 0:
        if str(details.get("mask_refiner", "") or "") == "ctd":
            final_mask, text_free_glyph_count = _dilate_ctd_final_mask_by_block_policy(
                details.get("final_mask"),
                img.shape,
                blk_list,
                final_dilate_size=final_dilate_size,
                text_free_dilate_size=int(cfg.get("text_free_final_mask_dilate_size", 1) or 1),
            )
            if text_free_glyph_count:
                details["mask_policy_text_free_glyph_applied_count"] = int(
                    details.get("mask_policy_text_free_glyph_applied_count", 0) or 0
                ) + int(text_free_glyph_count)
        else:
            final_mask = _dilate_final_mask(details.get("final_mask"), final_dilate_size)
        details["final_mask_post_expand"] = final_mask.copy()
        details["final_mask"] = final_mask
        details["final_mask_pixel_count"] = int(np.count_nonzero(final_mask))
    if str(details.get("mask_refiner", "") or "") == "ctd":
        (
            window_mask,
            bubble_window_count,
            bubble_cap_mask,
            protected_corner_mask,
            bubble_silhouette_applied_count,
            bubble_silhouette_fallback_count,
        ) = _build_candidate_window_mask(
            img,
            blk_list,
            bubble_seed_mask=details.get("raw_mask"),
        )
        if np.any(window_mask):
            current_mask = np.where(np.asarray(details.get("final_mask")) > 0, 255, 0).astype(np.uint8)
            clamped_mask = np.where((current_mask > 0) & (window_mask > 0), 255, 0).astype(np.uint8)
            removed = int(np.count_nonzero(current_mask)) - int(np.count_nonzero(clamped_mask))
            if removed > 0:
                details["final_mask_post_expand"] = clamped_mask.copy()
                details["final_mask"] = clamped_mask
                details["final_mask_pixel_count"] = int(np.count_nonzero(clamped_mask))
                details["mask_policy_removed_pixel_count"] = int(details.get("mask_policy_removed_pixel_count", 0) or 0) + removed
                details["mask_policy_outside_bubble_removed_pixel_count"] = (
                    int(details.get("mask_policy_outside_bubble_removed_pixel_count", 0) or 0) + removed
                )
        details["mask_policy_bubble_clamp_applied_count"] = int(bubble_window_count)
        details["bubble_interior_cap_mask"] = bubble_cap_mask
        details["protected_corner_mask"] = protected_corner_mask
        details["mask_policy_bubble_silhouette_applied_count"] = int(
            bubble_silhouette_applied_count
        )
        details["mask_policy_bubble_silhouette_fallback_count"] = int(
            bubble_silhouette_fallback_count
        )
        details["mask_policy_verified_bubble_dilate_reduction_applied_count"] = 0
        details["mask_policy_verified_bubble_dilate_reduced_pixel_count"] = 0
        if (
            bubble_silhouette_applied_count > 0
            and final_dilate_size > BUBBLE_VERIFIED_INTERIOR_FINAL_DILATE_SIZE
            and np.any(bubble_cap_mask)
        ):
            pre_expand_mask = details.get("final_mask_pre_expand")
            if (
                isinstance(pre_expand_mask, np.ndarray)
                and pre_expand_mask.shape[:2] == img.shape[:2]
            ):
                reduced_mask, _text_free_glyph_count = (
                    _dilate_ctd_final_mask_by_block_policy(
                        pre_expand_mask,
                        img.shape,
                        blk_list,
                        final_dilate_size=(
                            BUBBLE_VERIFIED_INTERIOR_FINAL_DILATE_SIZE
                        ),
                        text_free_dilate_size=int(
                            cfg.get("text_free_final_mask_dilate_size", 1) or 1
                        ),
                    )
                )
                current_mask = np.where(
                    np.asarray(details.get("final_mask")) > 0,
                    255,
                    0,
                ).astype(np.uint8)
                verified_cap = np.asarray(bubble_cap_mask) > 0
                reduced_inside_cap = (
                    (current_mask > 0)
                    & (np.asarray(reduced_mask) > 0)
                    & verified_cap
                )
                removed_by_narrower_dilation = int(
                    np.count_nonzero(
                        (current_mask > 0)
                        & verified_cap
                        & ~reduced_inside_cap
                    )
                )
                if removed_by_narrower_dilation:
                    current_mask[verified_cap] = np.where(
                        reduced_inside_cap[verified_cap],
                        255,
                        0,
                    ).astype(np.uint8)
                    details["final_mask_post_expand"] = current_mask.copy()
                    details["final_mask"] = current_mask
                    details["final_mask_pixel_count"] = int(
                        np.count_nonzero(current_mask)
                    )
                    details[
                        "mask_policy_verified_bubble_dilate_reduction_applied_count"
                    ] = int(bubble_silhouette_applied_count)
                    details[
                        "mask_policy_verified_bubble_dilate_reduced_pixel_count"
                    ] = removed_by_narrower_dilation
    if str(details.get("mask_refiner", "") or "") == "ctd":
        current_mask = normalize_edit_mask(details.get("final_mask"), img.shape)
        protected_mask = normalize_edit_mask(details.get("protect_mask"), img.shape)
        reintroduced_protected_pixels = int(
            np.count_nonzero((current_mask > 0) & (protected_mask > 0))
        )
        if reintroduced_protected_pixels:
            current_mask[protected_mask > 0] = 0
            details["final_mask"] = current_mask
            details["final_mask_pixel_count"] = int(
                np.count_nonzero(current_mask)
            )
            details["mask_policy_protected_pixel_removed_count"] = (
                reintroduced_protected_pixels
            )
            if (
                not np.any(current_mask)
                and int(details.get("ctd_or_mask_pixel_count", 0) or 0) > 0
            ):
                details["mask_candidate_source"] = MASK_CANDIDATE_SOURCE_NONE
                details["mask_decision"] = MASK_DECISION_REVIEW
                details["mask_reject_reason"] = "ctd_claim_fully_protected"
        details["final_mask_post_expand"] = current_mask.copy()
    details["source_glyph_extension"] = apply_spotting_source_glyph_extension(
        img,
        blk_list,
        details,
    )
    details["mask_policy_version"] = MASK_POLICY_VERSION
    details["final_mask_dilate_size"] = final_dilate_size
    annotate_block_mask_attribution(
        blk_list,
        details.get("final_mask"),
        img.shape,
        candidate_source=str(details.get("mask_candidate_source") or MASK_CANDIDATE_SOURCE_CTD_REFINED),
    )

    if return_details:
        return details
    return details["final_mask"]
