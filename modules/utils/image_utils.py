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
from modules.utils.inpainting_runtime import normalized_mask_refiner_settings
from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
    normalize_mask_inpaint_mode,
)
from modules.utils.mask_roi import (
    normalize_xyxy,
    resolve_block_ctd_roi,
    uses_glyph_only_mask_strategy,
)

MASK_POLICY_VERSION = "ctd_lama_mask_policy_v3"
MASK_DECISION_ACCEPTED = "accepted"
MASK_DECISION_REVIEW = "review"
MASK_CANDIDATE_SOURCE_CTD_REFINED = "ctd_refined"
MASK_CANDIDATE_SOURCE_CTD_OR = "ctd_raw_refined_final_or"
MASK_CANDIDATE_SOURCE_TEXT_FREE_GLYPH_THIN = "text_free_glyph_thin"
MASK_CANDIDATE_SOURCE_NONE = "none"
MASK_REJECT_LEGACY_WINDOW_ONLY_NO_CTD_MASK = "legacy_bbox_window_only_no_ctd_mask"
MASK_REJECT_RENDER_WITHOUT_ERASE_MASK = "render_without_erase_mask"


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


def _build_candidate_window_mask(
    image_shape: tuple[int, ...],
    block_list,
) -> tuple[np.ndarray, int]:
    window_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    bubble_window_count = 0
    for block in list(block_list or []):
        roi = resolve_block_ctd_roi(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 <= x1 or y2 <= y1:
            continue
        window_mask[y1:y2, x1:x2] = 255
        if str(getattr(block, "text_class", "") or "") == "text_bubble" and getattr(block, "bubble_xyxy", None) is not None:
            bubble_window_count += 1
    return window_mask, bubble_window_count


def _build_protected_region_mask(
    image_shape: tuple[int, ...],
    protected_blocks,
) -> tuple[np.ndarray, int]:
    """Build a hard exclusion mask for blocks routed away from auto-inpaint.

    ``split_inpaint_protected_ocr_blocks`` removes risky blocks before CTD is
    invoked.  This second, geometry-level exclusion is deliberate defence in
    depth: a neighbouring block's final dilation must not cross into a review
    or preserve region after the routing decision has been made.
    """
    protected_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    protected_count = 0
    for block in list(protected_blocks or []):
        action = str(getattr(block, "processing_action", "") or "").strip().lower()
        reason = str(getattr(block, "_inpaint_protected_reason", "") or "").strip()
        if action not in {"review", "preserve"} and not reason:
            continue

        # Bubble geometry is the potential destructive edit envelope for
        # dialogue.  text-free/UI blocks use their bounded OCR geometry.
        roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_shape)
        if roi is None:
            roi = resolve_block_ctd_roi(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = roi
        protected_mask[y1:y2, x1:x2] = 255
        protected_count += 1
    return protected_mask, protected_count


def _exclude_protected_regions(
    details: dict[str, Any],
    image_shape: tuple[int, ...],
    protected_blocks,
) -> None:
    """Remove auto-inpaint pixels from explicitly protected geometry.

    The masks in ``details`` are kept mutually consistent so downstream
    cleanup, checkpoint identity, diagnostics, and debug exports all observe
    the same safety boundary.
    """
    protected_mask, protected_count = _build_protected_region_mask(
        image_shape,
        protected_blocks,
    )
    details["mask_policy_protected_region_block_count"] = int(protected_count)
    details["mask_policy_protected_region_pixel_count"] = int(
        np.count_nonzero(protected_mask)
    )
    details["protected_region_mask"] = protected_mask
    details["mask_policy_protected_region_removed_pixel_count"] = 0
    if not np.any(protected_mask):
        return

    removed_final = 0
    for key in (
        "raw_mask",
        "refined_mask",
        "final_mask_pre_expand",
        "final_mask_post_expand",
        "final_mask",
    ):
        current = details.get(key)
        if current is None:
            continue
        current_mask = np.where(np.asarray(current) > 0, 255, 0).astype(np.uint8)
        if current_mask.shape[:2] != image_shape[:2]:
            continue
        if key == "final_mask":
            removed_final = int(np.count_nonzero(current_mask & (protected_mask > 0)))
        details[key] = np.where(
            (current_mask > 0) & (protected_mask <= 0),
            255,
            0,
        ).astype(np.uint8)

    details["final_mask_pixel_count"] = int(
        np.count_nonzero(details.get("final_mask"))
    )
    details["mask_policy_protected_region_removed_pixel_count"] = int(
        removed_final
    )


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


def _build_glyph_only_window_mask(
    image_shape: tuple[int, ...],
    block_list,
) -> tuple[np.ndarray, int, int]:
    window_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    text_free_count = 0
    structure_protect_count = 0
    for block in list(block_list or []):
        text_class = str(getattr(block, "text_class", "") or "")
        glyph_only = uses_glyph_only_mask_strategy(block)
        if text_class != "text_free" and not glyph_only:
            continue
        roi = resolve_block_ctd_roi(block, image_shape)
        if roi is None:
            continue
        x1, y1, x2, y2 = [int(v) for v in roi]
        if x2 <= x1 or y2 <= y1:
            continue
        window_mask[y1:y2, x1:x2] = 255
        if text_class == "text_free":
            text_free_count += 1
        if (
            str(getattr(block, "mask_strategy", "") or "")
            == "glyph_only_structure_protect"
        ):
            structure_protect_count += 1
    return window_mask, text_free_count, structure_protect_count


def _dilate_ctd_final_mask_by_block_policy(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    block_list,
    *,
    final_dilate_size: int,
    text_free_dilate_size: int,
) -> tuple[np.ndarray, int, int]:
    mask_arr = np.where(np.asarray(mask) > 0, 255, 0).astype(np.uint8)
    if int(final_dilate_size) <= 0 or mask_arr.size == 0 or not np.any(mask_arr):
        return mask_arr, 0, 0
    glyph_only_window, text_free_window_count, structure_protect_count = (
        _build_glyph_only_window_mask(image_shape, block_list)
    )
    if not np.any(glyph_only_window):
        return _dilate_final_mask(mask_arr, final_dilate_size), 0, 0

    glyph_only_mask = np.where(
        (mask_arr > 0) & (glyph_only_window > 0), 255, 0
    ).astype(np.uint8)
    other_mask = np.where(
        (mask_arr > 0) & (glyph_only_window <= 0), 255, 0
    ).astype(np.uint8)
    dilated_other = _dilate_final_mask(other_mask, final_dilate_size)
    dilated_glyph_only = _dilate_final_mask(
        glyph_only_mask,
        max(0, int(text_free_dilate_size)),
    )
    merged = np.where(
        (dilated_other > 0) | (dilated_glyph_only > 0), 255, 0
    ).astype(np.uint8)
    if not np.any(glyph_only_mask):
        return merged, 0, 0
    return merged, text_free_window_count, structure_protect_count


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
        source_box = _normalize_xyxy_for_shape(getattr(block, "xyxy", None), image_shape)
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
        "mask_policy_structure_protect_glyph_applied_count": 0,
        "mask_policy_removed_pixel_count": 0,
        "mask_policy_outside_bubble_removed_pixel_count": 0,
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


def generate_mask(
    img: np.ndarray,
    blk_list,
    default_padding: int = 5,
    settings: dict[str, Any] | None = None,
    return_details: bool = False,
    precomputed_mask_details: dict[str, Any] | None = None,
    protected_blocks=None,
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
        details["mask_policy_structure_protect_glyph_applied_count"] = 0
        details["mask_policy_removed_pixel_count"] = 0
        details["mask_policy_outside_bubble_removed_pixel_count"] = 0
        details["legacy_bbox_role"] = "window_only"
        details["legacy_bbox_direct_erase_disabled"] = True
        details["ctd_legacy_rectangle_rescue_disabled"] = True

    final_dilate_size = int(cfg.get("final_mask_dilate_size", 8) or 0)
    if final_dilate_size > 0:
        if str(details.get("mask_refiner", "") or "") == "ctd":
            (
                final_mask,
                text_free_glyph_count,
                structure_protect_glyph_count,
            ) = _dilate_ctd_final_mask_by_block_policy(
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
            if structure_protect_glyph_count:
                details[
                    "mask_policy_structure_protect_glyph_applied_count"
                ] = int(
                    details.get(
                        "mask_policy_structure_protect_glyph_applied_count",
                        0,
                    )
                    or 0
                ) + int(structure_protect_glyph_count)
        else:
            final_mask = _dilate_final_mask(details.get("final_mask"), final_dilate_size)
        details["final_mask_post_expand"] = final_mask.copy()
        details["final_mask"] = final_mask
        details["final_mask_pixel_count"] = int(np.count_nonzero(final_mask))
    if str(details.get("mask_refiner", "") or "") == "ctd":
        window_mask, bubble_window_count = _build_candidate_window_mask(img.shape, blk_list)
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
    _exclude_protected_regions(
        details,
        img.shape,
        protected_blocks,
    )
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
