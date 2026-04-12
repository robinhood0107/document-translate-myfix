from __future__ import annotations

import base64
from typing import Any

import imkit as imk
import numpy as np
from PySide6.QtGui import QColor

from modules.masking import build_legacy_bbox_mask_details
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


def _normalize_mask_refiner_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(settings or {})
    data["mask_refiner"] = "legacy_bbox"
    data["mask_inpaint_mode"] = normalize_mask_inpaint_mode(data.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE))
    data["keep_existing_lines"] = False
    return data


def generate_mask(
    img: np.ndarray,
    blk_list,
    default_padding: int = 5,
    settings: dict[str, Any] | None = None,
    return_details: bool = False,
    precomputed_mask_details: dict[str, Any] | None = None,
):
    del precomputed_mask_details

    cfg = _normalize_mask_refiner_settings(settings)
    details = build_legacy_bbox_mask_details(
        img,
        list(blk_list or []),
        cfg,
        default_padding=default_padding,
    )
    if return_details:
        return details
    return details["final_mask"]
