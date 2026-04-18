from __future__ import annotations

from typing import Any

DEFAULT_CTD_SETTINGS = {
    "mask_refiner": "ctd",
    "ctd_detect_size": 1280,
    "ctd_det_rearrange_max_batches": 4,
    "ctd_device": "cuda",
    "ctd_font_size_multiplier": 1.0,
    "ctd_font_size_max": -1,
    "ctd_font_size_min": -1,
    "ctd_mask_dilate_size": 2,
    "keep_existing_lines": True,
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
    merged["mask_refiner"] = str(merged.get("mask_refiner", "ctd") or "ctd")
    merged["ctd_detect_size"] = int(merged.get("ctd_detect_size", 1280) or 1280)
    merged["ctd_det_rearrange_max_batches"] = int(merged.get("ctd_det_rearrange_max_batches", 4) or 4)
    merged["ctd_device"] = str(merged.get("ctd_device", "cuda") or "cuda")
    merged["ctd_font_size_multiplier"] = float(merged.get("ctd_font_size_multiplier", 1.0) or 1.0)
    merged["ctd_font_size_max"] = int(merged.get("ctd_font_size_max", -1) or -1)
    merged["ctd_font_size_min"] = int(merged.get("ctd_font_size_min", -1) or -1)
    merged["ctd_mask_dilate_size"] = int(merged.get("ctd_mask_dilate_size", 2) or 2)
    merged["keep_existing_lines"] = bool(merged.get("keep_existing_lines", True))
    return merged
