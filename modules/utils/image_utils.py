from __future__ import annotations

import base64
from typing import Any

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
    protect_mask = build_protect_mask(
        img,
        block_list,
        ProtectMaskSettings(
            keep_existing_lines=bool(cfg.get("keep_existing_lines", True)),
        ),
    )
    protected_mask = np.where(
        (ctd_final_mask > 0) & (np.asarray(protect_mask) <= 0),
        255,
        0,
    ).astype(np.uint8)
    fallback_used = bool(ctd_result.fallback_used)
    refiner_backend = str(ctd_result.backend or "ctd")
    final_mask = protected_mask
    legacy_fallback_details: dict[str, Any] | None = None

    if not np.any(final_mask) and np.any(ctd_final_mask):
        final_mask = ctd_final_mask.copy()
        fallback_used = True
        refiner_backend = f"{refiner_backend}+protect_fallback"

    if not np.any(final_mask) and block_list:
        legacy_fallback_details = _legacy_details(
            img,
            block_list,
            cfg,
            default_padding=default_padding,
        )
        final_mask = np.where(
            np.asarray(legacy_fallback_details.get("final_mask")) > 0,
            255,
            0,
        ).astype(np.uint8)
        fallback_used = True
        refiner_backend = f"{refiner_backend}+legacy_bbox_fallback"

    details = {
        "raw_mask": raw_mask,
        "refined_mask": refined_mask,
        "protect_mask": np.where(np.asarray(protect_mask) > 0, 255, 0).astype(np.uint8),
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
        "mask_inpaint_mode": str(cfg.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE) or DEFAULT_MASK_INPAINT_MODE),
    }
    if legacy_fallback_details:
        details["legacy_base_mask"] = legacy_fallback_details.get("legacy_base_mask")
        details["hard_box_rescue_mask"] = legacy_fallback_details.get("hard_box_rescue_mask")
        details["hard_box_applied_count"] = int(legacy_fallback_details.get("hard_box_applied_count", 0) or 0)
        details["hard_box_reason_totals"] = dict(legacy_fallback_details.get("hard_box_reason_totals", {}) or {})
        details["legacy_base_mask_pixel_count"] = int(legacy_fallback_details.get("legacy_base_mask_pixel_count", 0) or 0)
        details["hard_box_rescue_mask_pixel_count"] = int(legacy_fallback_details.get("hard_box_rescue_mask_pixel_count", 0) or 0)
    return details


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
        details["mask_refiner"] = "ctd"
        details["keep_existing_lines"] = bool(cfg.get("keep_existing_lines", True))
        details["refiner_backend"] = "ctd+legacy_bbox_exception_fallback"
        details["refiner_device"] = str(cfg.get("ctd_device", "cuda") or "cuda")
        details["fallback_used"] = True
        details["mask_inpaint_mode"] = cfg["mask_inpaint_mode"]
    if return_details:
        return details
    return details["final_mask"]
