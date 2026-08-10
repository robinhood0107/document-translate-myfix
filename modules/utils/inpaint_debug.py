from __future__ import annotations

import json
import os
from typing import Iterable

import imkit as imk
import numpy as np
from PIL import Image, ImageDraw

from modules.utils.debug_artifacts import (
    atomic_debug_image,
    atomic_debug_json,
)
from modules.utils.mask_roi import get_mask_roi_type
from modules.utils.inpaint_envelope import build_text_free_erase_envelope


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        channel = image[:, :, 0]
        return np.stack([channel] * 3, axis=-1).astype(np.uint8)
    return image.astype(np.uint8)


def has_debug_exports(export_settings: dict | None) -> bool:
    settings = export_settings or {}
    return any(
        bool(settings.get(key, False))
        for key in (
            "export_detector_overlay",
            "export_raw_mask",
            "export_mask_overlay",
            "export_cleanup_mask_delta",
            "export_debug_metadata",
        )
    )


def _normalize_mask(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(image_shape[:2], dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != image_shape[:2]:
        normalized = np.zeros(image_shape[:2], dtype=np.uint8)
        h = min(normalized.shape[0], arr.shape[0])
        w = min(normalized.shape[1], arr.shape[1])
        normalized[:h, :w] = arr[:h, :w]
        arr = normalized
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


def _build_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    base = ensure_three_channel(image).astype(np.float32)
    overlay = base.copy()
    mask_pixels = mask > 0
    if np.any(mask_pixels):
        tint = np.array(color, dtype=np.float32)
        overlay[mask_pixels] = base[mask_pixels] * (1.0 - alpha) + tint * alpha
    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def _collect_bubble_boxes(blocks: Iterable) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    bubbles: list[tuple[int, int, int, int]] = []
    for block in blocks or []:
        bubble = getattr(block, "bubble_xyxy", None)
        if bubble is None or len(bubble) < 4:
            continue
        box = tuple(int(float(v)) for v in bubble[:4])
        if box in seen:
            continue
        seen.add(box)
        bubbles.append(box)
    return bubbles


def build_detector_overlay(image: np.ndarray, blocks: Iterable) -> np.ndarray:
    canvas = Image.fromarray(ensure_three_channel(image))
    draw = ImageDraw.Draw(canvas)
    palette = {
        "bubble": (63, 135, 245),
        "text_bubble": (54, 197, 94),
        "text_free": (255, 64, 64),
        "ctd_roi": (60, 220, 255),
        "cleanup_roi": (255, 214, 10),
    }

    for x1, y1, x2, y2 in _collect_bubble_boxes(blocks):
        draw.rectangle([x1, y1, x2, y2], outline=palette["bubble"], width=2)

    for block in blocks or []:
        bbox = getattr(block, "xyxy", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        color = palette.get(getattr(block, "text_class", ""), (255, 64, 64))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        ctd_roi = getattr(block, "ctd_roi_xyxy", None) or getattr(block, "mask_roi_xyxy", None)
        if ctd_roi is not None and len(ctd_roi) >= 4:
            rx1, ry1, rx2, ry2 = [int(float(v)) for v in ctd_roi[:4]]
            draw.rectangle([rx1, ry1, rx2, ry2], outline=palette["ctd_roi"], width=1)

        cleanup_roi = getattr(block, "cleanup_roi_xyxy", None)
        if cleanup_roi is not None and len(cleanup_roi) >= 4:
            cx1, cy1, cx2, cy2 = [int(float(v)) for v in cleanup_roi[:4]]
            draw.rectangle([cx1, cy1, cx2, cy2], outline=palette["cleanup_roi"], width=1)

    return np.array(canvas, dtype=np.uint8)


def serialize_inpaint_block(block, index: int) -> dict:
    inpaint_boxes = []
    raw_inpaint_boxes = getattr(block, "inpaint_bboxes", None)
    if raw_inpaint_boxes is not None:
        for box in raw_inpaint_boxes:
            inpaint_boxes.append([int(float(v)) for v in box[:4]])
    image_shape = getattr(block, "_debug_image_shape", None)
    text_free_envelope = None
    if image_shape is not None:
        text_free_envelope = build_text_free_erase_envelope(block, image_shape)

    return {
        "index": int(index),
        "xyxy": [int(float(v)) for v in getattr(block, "xyxy", (0, 0, 0, 0))[:4]],
        "mask_anchor_xyxy": (
            [int(float(v)) for v in getattr(block, "_mask_anchor_xyxy", ())[:4]]
            if getattr(block, "_mask_anchor_xyxy", None) is not None
            else None
        ),
        "mask_anchor_source": str(
            getattr(block, "_mask_anchor_source", "") or ""
        ),
        "mask_anchor_relation": str(
            getattr(block, "_mask_anchor_relation", "") or ""
        ),
        "render_original_xyxy": (
            [int(float(v)) for v in getattr(block, "_render_original_xyxy", ())[:4]]
            if getattr(block, "_render_original_xyxy", None) is not None
            else None
        ),
        "render_area_xyxy": (
            [int(float(v)) for v in getattr(block, "_render_area_xyxy", ())[:4]]
            if getattr(block, "_render_area_xyxy", None) is not None
            else None
        ),
        "render_bubble_xyxy": (
            [int(float(v)) for v in getattr(block, "_render_bubble_xyxy", ())[:4]]
            if getattr(block, "_render_bubble_xyxy", None) is not None
            else None
        ),
        "render_area_source": str(
            getattr(block, "_render_area_source", "") or ""
        ),
        "translation_raw": str(
            getattr(block, "_render_translation_raw", getattr(block, "translation", "")) or ""
        ),
        "render_text": str(
            getattr(block, "_render_text", getattr(block, "translation", "")) or ""
        ),
        "render_html_applied": bool(getattr(block, "_render_html_applied", False)),
        "render_fallback_font_family": str(
            getattr(block, "_render_fallback_font_family", "") or ""
        ),
        "render_normalization_applied": bool(
            getattr(block, "_render_normalization_applied", False)
        ),
        "render_normalization_reasons": list(
            getattr(block, "_render_normalization_reasons", []) or []
        ),
        "bubble_xyxy": (
            [int(float(v)) for v in getattr(block, "bubble_xyxy", ())[:4]]
            if getattr(block, "bubble_xyxy", None) is not None
            else None
        ),
        "ctd_roi_xyxy": (
            [int(float(v)) for v in getattr(block, "ctd_roi_xyxy", ())[:4]]
            if getattr(block, "ctd_roi_xyxy", None) is not None
            else None
        ),
        "cleanup_roi_xyxy": (
            [int(float(v)) for v in getattr(block, "cleanup_roi_xyxy", ())[:4]]
            if getattr(block, "cleanup_roi_xyxy", None) is not None
            else None
        ),
        "mask_roi_xyxy": (
            [int(float(v)) for v in getattr(block, "mask_roi_xyxy", ())[:4]]
            if getattr(block, "mask_roi_xyxy", None) is not None
            else None
        ),
        "roi_type": get_mask_roi_type(block),
        "text_class": getattr(block, "text_class", "") or "",
        "semantic_role": str(
            getattr(block, "semantic_role", "") or ""
        ),
        "processing_action": str(
            getattr(block, "processing_action", "") or ""
        ),
        "processing_decision_source": str(
            getattr(block, "processing_decision_source", "") or ""
        ),
        "processing_decision_reasons": list(
            getattr(block, "processing_decision_reasons", []) or []
        ),
        "canonical_block_id": str(
            getattr(block, "canonical_block_id", "") or ""
        ),
        "duplicate_alias_block_ids": list(
            getattr(block, "duplicate_alias_block_ids", []) or []
        ),
        "duplicate_alias_count": int(
            getattr(block, "duplicate_alias_count", 0) or 0
        ),
        "compound_group_id": str(
            getattr(block, "compound_group_id", "") or ""
        ),
        "merge_split_diagnostics": dict(
            getattr(block, "merge_split_diagnostics", {}) or {}
        ),
        "mask_strategy": str(
            getattr(block, "mask_strategy", "") or ""
        ),
        "mask_strategy_reason": str(
            getattr(block, "mask_strategy_reason", "") or ""
        ),
        "mask_actual_bbox": (
            [
                int(float(value))
                for value in getattr(block, "mask_actual_bbox", ())[:4]
            ]
            if getattr(block, "mask_actual_bbox", None) is not None
            else None
        ),
        "mask_actual_pixel_count": int(
            getattr(block, "mask_actual_pixel_count", 0) or 0
        ),
        "inpaint_bboxes": inpaint_boxes,
        "text_free_erase_envelope_xyxy": (
            [int(v) for v in text_free_envelope]
            if text_free_envelope is not None
            else None
        ),
        "hard_box_applied": bool(getattr(block, "_hard_box_applied", False)),
        "hard_box_reason_codes": list(getattr(block, "_hard_box_reason_codes", []) or []),
        "hard_box_rescue_roi_xyxy": (
            [int(float(v)) for v in getattr(block, "_hard_box_rescue_roi_xyxy", ())[:4]]
            if getattr(block, "_hard_box_rescue_roi_xyxy", None) is not None
            else None
        ),
        "legacy_fill_ratio": float(getattr(block, "_legacy_fill_ratio", 0.0) or 0.0),
        "rescue_fill_ratio": float(getattr(block, "_rescue_fill_ratio", 0.0) or 0.0),
        "legacy_mask_pixel_count": int(getattr(block, "_legacy_mask_pixel_count", 0) or 0),
        "rescue_mask_pixel_count": int(getattr(block, "_rescue_mask_pixel_count", 0) or 0),
        "final_mask_pixel_count": int(getattr(block, "_final_mask_pixel_count", 0) or 0),
        "block_final_mask_pixel_count": int(getattr(block, "block_final_mask_pixel_count", 0) or 0),
        "block_mask_iou": float(getattr(block, "block_mask_iou", 0.0) or 0.0),
        "block_mask_span_coverage": float(getattr(block, "block_mask_span_coverage", 0.0) or 0.0),
        "block_mask_bbox": (
            [int(float(v)) for v in getattr(block, "block_mask_bbox", ())[:4]]
            if getattr(block, "block_mask_bbox", None) is not None
            else None
        ),
        "block_mask_source": str(getattr(block, "block_mask_source", "") or ""),
        "block_mask_decision": str(getattr(block, "block_mask_decision", "") or ""),
        "hard_box_metrics": dict(getattr(block, "_hard_box_metrics", {}) or {}),
        "erase_mode": str(getattr(block, "_erase_mode", "") or ""),
        "erase_edit_pixel_count": int(getattr(block, "_erase_edit_pixel_count", 0) or 0),
        "erase_protect_pixel_count": int(getattr(block, "_erase_protect_pixel_count", 0) or 0),
        "erase_skipped_reason": str(getattr(block, "_erase_skipped_reason", "") or ""),
        "mask_policy": str(getattr(block, "_mask_policy", "") or ""),
        "render_restore_applied": bool(getattr(block, "_render_restore_applied", False)),
        "ui_panel_mode": str(getattr(block, "ui_panel_mode", "") or ""),
        "ui_panel_preview_path": str(getattr(block, "ui_panel_preview_path", "") or ""),
        "mask_decision": str(getattr(block, "mask_decision", "") or ""),
        "mask_reject_reason": str(getattr(block, "mask_reject_reason", "") or ""),
        "bubble_panel_text_candidate": bool(getattr(block, "bubble_panel_text_candidate", False)),
        "bubble_panel_group_id": str(getattr(block, "bubble_panel_group_id", "") or ""),
        "bubble_panel_member_indices": list(getattr(block, "bubble_panel_member_indices", []) or []),
        "bubble_panel_mask_pixel_count": int(getattr(block, "bubble_panel_mask_pixel_count", 0) or 0),
        "bubble_panel_mask_source": str(getattr(block, "bubble_panel_mask_source", "") or ""),
        "bubble_panel_merge_decision": str(getattr(block, "bubble_panel_merge_decision", "") or ""),
        "bubble_merge_reocr_needed": bool(getattr(block, "bubble_merge_reocr_needed", False)),
    }


def build_inpaint_debug_metadata(
    *,
    image_path: str,
    run_type: str,
    detector_key: str,
    detector_engine: str,
    device: str,
    inpainter: str,
    hd_strategy: str,
    blocks: Iterable,
    raw_mask: np.ndarray | None,
    final_mask: np.ndarray | None = None,
    final_mask_pre_expand: np.ndarray | None = None,
    final_mask_post_expand: np.ndarray | None = None,
    residue_mask: np.ndarray | None = None,
    cleanup_delta: np.ndarray | None = None,
    cleanup_stats: dict | None = None,
    mask_refiner: str = "legacy_bbox",
    protect_mask_applied: bool = False,
    protect_mask: np.ndarray | None = None,
    refiner_backend: str = "legacy",
    refiner_device: str = "cpu",
    inpainter_backend: str = "unknown",
    legacy_base_mask: np.ndarray | None = None,
    hard_box_rescue_mask: np.ndarray | None = None,
    hard_box_applied_count: int | None = None,
    hard_box_reason_totals: dict | None = None,
    mask_quality_policy: str = "",
    mask_policy_bubble_clamp_applied_count: int = 0,
    mask_policy_bubble_silhouette_applied_count: int = 0,
    mask_policy_bubble_silhouette_fallback_count: int = 0,
    mask_policy_text_free_glyph_applied_count: int = 0,
    mask_policy_removed_pixel_count: int = 0,
    mask_policy_outside_bubble_removed_pixel_count: int = 0,
    ctd_legacy_rectangle_rescue_disabled: bool = False,
    text_free_image_glyph_rescue_count: int = 0,
    text_free_image_glyph_rescue_mask_pixel_count: int = 0,
    mask_policy_version: str = "",
    mask_candidate_source: str = "",
    mask_decision: str = "",
    mask_reject_reason: str = "",
    mask_score_outside_change: float = 0.0,
    mask_score_outline_damage: float = 0.0,
    mask_score_residue: float = 0.0,
    mask_score_color_delta: float = 0.0,
    ui_panel_mode: str = "",
    ui_panel_preview_path: str = "",
    inpaint_runtime_diagnostics: dict | None = None,
) -> dict:
    block_list = list(blocks or [])
    if final_mask is not None:
        for block in block_list:
            block._debug_image_shape = final_mask.shape
    cleanup_stats = cleanup_stats or {}
    raw_mask_pixels = int(np.count_nonzero(raw_mask)) if raw_mask is not None else 0
    final_mask_pixels = int(np.count_nonzero(final_mask)) if final_mask is not None else 0
    final_mask_pre_expand_pixels = int(np.count_nonzero(final_mask_pre_expand)) if final_mask_pre_expand is not None else 0
    final_mask_post_expand_pixels = int(np.count_nonzero(final_mask_post_expand)) if final_mask_post_expand is not None else 0
    residue_mask_pixels = int(np.count_nonzero(residue_mask)) if residue_mask is not None else 0
    cleanup_delta_pixels = int(np.count_nonzero(cleanup_delta)) if cleanup_delta is not None else 0
    protect_mask_pixels = int(np.count_nonzero(protect_mask)) if protect_mask is not None else 0
    legacy_base_mask_pixels = int(np.count_nonzero(legacy_base_mask)) if legacy_base_mask is not None else 0
    hard_box_rescue_mask_pixels = int(np.count_nonzero(hard_box_rescue_mask)) if hard_box_rescue_mask is not None else 0
    if hard_box_applied_count is None:
        hard_box_applied_count = sum(1 for block in block_list if bool(getattr(block, "_hard_box_applied", False)))
    if hard_box_reason_totals is None:
        reason_totals: dict[str, int] = {}
        for block in block_list:
            for code in list(getattr(block, "_hard_box_reason_codes", []) or []):
                reason_totals[code] = reason_totals.get(code, 0) + 1
        hard_box_reason_totals = reason_totals
    duplicate_bubble_inner_fill = cleanup_stats.get("duplicate_bubble_inner_fill") or {}
    return {
        "image_path": image_path,
        "run_type": run_type,
        "detector_key": detector_key,
        "detector_engine": detector_engine,
        "device": device,
        "inpainter": inpainter,
        "inpainter_backend": inpainter_backend,
        "hd_strategy": hd_strategy,
        "mask_refiner": mask_refiner,
        "refiner_backend": refiner_backend,
        "refiner_device": refiner_device,
        "protect_mask_applied": bool(protect_mask_applied),
        "protect_mask_pixel_count": protect_mask_pixels,
        "block_count": len(block_list),
        "raw_mask_pixel_count": raw_mask_pixels,
        "legacy_base_mask_pixel_count": legacy_base_mask_pixels,
        "hard_box_rescue_mask_pixel_count": hard_box_rescue_mask_pixels,
        "final_mask_pixel_count": final_mask_pixels,
        "final_mask_pre_expand_pixel_count": final_mask_pre_expand_pixels,
        "final_mask_post_expand_pixel_count": final_mask_post_expand_pixels,
        "residue_mask_pixel_count": residue_mask_pixels,
        "cleanup_delta_pixel_count": cleanup_delta_pixels,
        "hard_box_applied_count": int(hard_box_applied_count or 0),
        "hard_box_reason_totals": dict(hard_box_reason_totals or {}),
        "mask_quality_policy": str(mask_quality_policy or ""),
        "mask_policy_bubble_clamp_applied_count": int(mask_policy_bubble_clamp_applied_count or 0),
        "mask_policy_bubble_silhouette_applied_count": int(
            mask_policy_bubble_silhouette_applied_count or 0
        ),
        "mask_policy_bubble_silhouette_fallback_count": int(
            mask_policy_bubble_silhouette_fallback_count or 0
        ),
        "mask_policy_text_free_glyph_applied_count": int(mask_policy_text_free_glyph_applied_count or 0),
        "mask_policy_removed_pixel_count": int(mask_policy_removed_pixel_count or 0),
        "mask_policy_outside_bubble_removed_pixel_count": int(mask_policy_outside_bubble_removed_pixel_count or 0),
        "ctd_legacy_rectangle_rescue_disabled": bool(ctd_legacy_rectangle_rescue_disabled),
        "text_free_image_glyph_rescue_count": int(text_free_image_glyph_rescue_count or 0),
        "text_free_image_glyph_rescue_mask_pixel_count": int(text_free_image_glyph_rescue_mask_pixel_count or 0),
        "mask_policy_version": str(mask_policy_version or ""),
        "mask_candidate_source": str(mask_candidate_source or ""),
        "mask_decision": str(mask_decision or ""),
        "mask_reject_reason": str(mask_reject_reason or ""),
        "mask_score_outside_change": float(mask_score_outside_change or 0.0),
        "mask_score_outline_damage": float(mask_score_outline_damage or 0.0),
        "mask_score_residue": float(mask_score_residue or 0.0),
        "mask_score_color_delta": float(mask_score_color_delta or 0.0),
        "ui_panel_mode": str(ui_panel_mode or ""),
        "ui_panel_preview_path": str(ui_panel_preview_path or ""),
        "inpaint_runtime_diagnostics": dict(
            inpaint_runtime_diagnostics or {}
        ),
        "cleanup_applied": bool(cleanup_stats.get("applied", False)),
        "cleanup_component_count": int(cleanup_stats.get("component_count", 0) or 0),
        "cleanup_block_count": int(cleanup_stats.get("block_count", 0) or 0),
        "pass2_applied": bool(cleanup_stats.get("applied", False)),
        "pass2_component_count": int(cleanup_stats.get("component_count", 0) or 0),
        "pass2_candidate_count": int(cleanup_stats.get("pass2_candidate_count", 0) or 0),
        "pass2_bubble_candidate_count": int(cleanup_stats.get("pass2_bubble_candidate_count", 0) or 0),
        "pass2_bubble_kept_count": int(cleanup_stats.get("pass2_bubble_kept_count", 0) or 0),
        "pass2_text_free_candidate_count": int(cleanup_stats.get("pass2_text_free_candidate_count", 0) or 0),
        "pass2_text_free_kept_count": int(cleanup_stats.get("pass2_text_free_kept_count", 0) or 0),
        "pass2_residue_mask_pre_cap_pixel_count": int(cleanup_stats.get("residue_mask_pre_cap_pixel_count", 0) or 0),
        "pass2_residue_mask_cap_pixel_count": int(cleanup_stats.get("residue_mask_cap_pixel_count", 0) or 0),
        "pass2_residue_mask_cap_dilate_px": int(cleanup_stats.get("residue_mask_cap_dilate_px", 0) or 0),
        "pass2_backend": str(cleanup_stats.get("pass2_backend", "") or ""),
        "pass2_name": str(cleanup_stats.get("pass_name", "") or ""),
        "duplicate_bubble_inner_fill_applied": bool(duplicate_bubble_inner_fill.get("applied", False)),
        "duplicate_bubble_inner_fill_pixel_count": int(
            duplicate_bubble_inner_fill.get("duplicate_bubble_inner_fill_pixel_count", 0) or 0
        ),
        "duplicate_bubble_inner_fill_backend": str(
            duplicate_bubble_inner_fill.get("duplicate_bubble_inner_fill_backend", "") or ""
        ),
        "blocks": [serialize_inpaint_block(block, idx) for idx, block in enumerate(block_list)],
    }


def _write_image(base_dir: str, folder: str, archive_bname: str, filename: str, image: np.ndarray) -> str:
    target_dir = os.path.join(base_dir, folder, archive_bname)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, filename)
    imk.write_image(path, image)
    return path


def _write_json(base_dir: str, folder: str, archive_bname: str, filename: str, payload: dict) -> str:
    target_dir = os.path.join(base_dir, folder, archive_bname)
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path


def export_inpaint_debug_artifacts(
    *,
    export_root: str,
    archive_bname: str,
    page_base_name: str,
    image: np.ndarray,
    blocks: Iterable,
    export_settings: dict | None,
    raw_mask: np.ndarray | None = None,
    mask_overlay_mask: np.ndarray | None = None,
    cleanup_delta: np.ndarray | None = None,
    metadata: dict | None = None,
    page_output_dir: str = "",
) -> dict[str, str]:
    settings = export_settings or {}
    if not has_debug_exports(settings):
        return {}

    image_rgb = ensure_three_channel(image)
    normalized_raw_mask = _normalize_mask(raw_mask, image_rgb.shape)
    normalized_mask_overlay = _normalize_mask(mask_overlay_mask, image_rgb.shape)
    normalized_cleanup_delta = _normalize_mask(cleanup_delta, image_rgb.shape)
    written: dict[str, str] = {}

    def write_image(
        key: str,
        folder: str,
        legacy_name: str,
        cache_name: str,
        value: np.ndarray,
    ) -> None:
        if page_output_dir:
            written[key] = atomic_debug_image(
                page_output_dir,
                cache_name,
                value,
            )
        else:
            written[key] = _write_image(
                export_root,
                folder,
                archive_bname,
                legacy_name,
                value,
            )

    if settings.get("export_detector_overlay", False):
        write_image(
            "detector_overlay",
            "detector_overlays",
            f"{page_base_name}_detector_overlay.png",
            "detector-overlay.png",
            build_detector_overlay(image_rgb, blocks),
        )

    if settings.get("export_raw_mask", False):
        write_image(
            "raw_mask",
            "raw_masks",
            f"{page_base_name}_raw_mask.png",
            "inpaint-raw-mask.png",
            _mask_to_rgb(normalized_raw_mask),
        )

    if settings.get("export_mask_overlay", False):
        write_image(
            "mask_overlay",
            "mask_overlays",
            f"{page_base_name}_mask_overlay.png",
            "inpaint-mask-overlay.png",
            _build_mask_overlay(image_rgb, normalized_mask_overlay),
        )

    if settings.get("export_cleanup_mask_delta", False):
        write_image(
            "cleanup_delta",
            "cleanup_mask_delta",
            f"{page_base_name}_cleanup_delta.png",
            "inpaint-cleanup-delta.png",
            _mask_to_rgb(normalized_cleanup_delta),
        )

    if settings.get("export_debug_metadata", False):
        if page_output_dir:
            written["debug_metadata"] = atomic_debug_json(
                page_output_dir,
                "debug-metadata.json",
                metadata or {},
            )
        else:
            written["debug_metadata"] = _write_json(
                export_root,
                "debug_metadata",
                archive_bname,
                f"{page_base_name}_debug.json",
                metadata or {},
            )
    return written
