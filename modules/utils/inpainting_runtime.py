from __future__ import annotations

from typing import Any

from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
    normalize_mask_inpaint_mode,
    uses_legacy_bbox_source_mode,
    uses_source_compat_mode,
    uses_source_parity_detector,
)

DEFAULT_CTD_SETTINGS = {
    "mask_refiner": "legacy_bbox",
    "mask_inpaint_mode": DEFAULT_MASK_INPAINT_MODE,
    "ctd_detect_size": 1280,
    "ctd_det_rearrange_max_batches": 4,
    "ctd_device": "cuda",
    "ctd_font_size_multiplier": 1.0,
    "ctd_font_size_max": -1,
    "ctd_font_size_min": -1,
    "ctd_mask_dilate_size": 2,
    "keep_existing_lines": False,
}

DEFAULT_INPAINTER_SETTINGS = {
    "AOT": {
        "backend": "torch",
        "device": "cuda",
        "inpaint_size": 2048,
        "precision": "fp32",
    },
    "lama_large_512px": {
        "backend": "torch",
        "device": "cuda",
        "inpaint_size": 1536,
        "precision": "bf16",
    },
    "lama_mpe": {
        "backend": "torch",
        "device": "cuda",
        "inpaint_size": 2048,
        "precision": "fp32",
    },
}

INPAINTER_DEPRECATION_MAP = {
    "LaMa": "lama_large_512px",
}


def normalize_inpainter_key(value: str | None) -> str:
    raw = str(value or "").strip()
    if not raw:
        return "AOT"
    return INPAINTER_DEPRECATION_MAP.get(raw, raw)


def inpainter_backend_for(key: str | None) -> str:
    normalized = normalize_inpainter_key(key)
    return str(DEFAULT_INPAINTER_SETTINGS.get(normalized, DEFAULT_INPAINTER_SETTINGS["AOT"])["backend"])


def inpainter_default_settings(key: str | None) -> dict[str, Any]:
    normalized = normalize_inpainter_key(key)
    defaults = DEFAULT_INPAINTER_SETTINGS.get(normalized, DEFAULT_INPAINTER_SETTINGS["AOT"])
    return dict(defaults)


def normalized_mask_refiner_settings(raw: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CTD_SETTINGS)
    if raw:
        merged.update(raw)
    mode = normalize_mask_inpaint_mode(merged.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE))
    merged["mask_inpaint_mode"] = mode
    merged["ctd_detect_size"] = int(merged.get("ctd_detect_size", 1280) or 1280)
    merged["ctd_det_rearrange_max_batches"] = int(merged.get("ctd_det_rearrange_max_batches", 4) or 4)
    merged["ctd_device"] = str(merged.get("ctd_device", "cuda") or "cuda")
    merged["ctd_font_size_multiplier"] = float(merged.get("ctd_font_size_multiplier", 1.0) or 1.0)
    merged["ctd_font_size_max"] = int(merged.get("ctd_font_size_max", -1) or -1)
    merged["ctd_font_size_min"] = int(merged.get("ctd_font_size_min", -1) or -1)
    merged["ctd_mask_dilate_size"] = int(merged.get("ctd_mask_dilate_size", 2) or 2)

    if uses_source_parity_detector(mode):
        merged["mask_refiner"] = "ctd"
    elif uses_legacy_bbox_source_mode(mode):
        merged["mask_refiner"] = "legacy_bbox"
    else:
        merged["mask_refiner"] = str(merged.get("mask_refiner", "legacy_bbox") or "legacy_bbox")

    merged["keep_existing_lines"] = False if uses_source_compat_mode(mode) else bool(merged.get("keep_existing_lines", False))
    return merged
