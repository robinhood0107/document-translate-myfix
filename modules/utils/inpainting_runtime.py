from __future__ import annotations

from typing import Any

from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
    normalize_mask_inpaint_mode,
)

DEFAULT_MASK_RUNTIME_SETTINGS = {
    "mask_refiner": "legacy_bbox",
    "mask_inpaint_mode": DEFAULT_MASK_INPAINT_MODE,
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
    merged = dict(DEFAULT_MASK_RUNTIME_SETTINGS)
    if raw:
        merged.update(raw)
    merged["mask_inpaint_mode"] = normalize_mask_inpaint_mode(merged.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE))
    merged["mask_refiner"] = "legacy_bbox"
    merged["keep_existing_lines"] = False
    return merged
