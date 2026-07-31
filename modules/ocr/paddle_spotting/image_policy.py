"""Full-page image policy for official PaddleOCR-VL Spotting."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


PADDLE_SPOTTING_OFFICIAL_IMAGE_MAX_PIXELS = 1_605_632
PADDLE_SPOTTING_LOW_RES_DOUBLE_THRESHOLD = 1500


def preprocess_spotting_image(
    image: np.ndarray,
    *,
    low_resolution_threshold: int = PADDLE_SPOTTING_LOW_RES_DOUBLE_THRESHOLD,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply the existing low-resolution policy and preserve aspect ratio."""

    height, width = image.shape[:2]
    doubled = width < low_resolution_threshold and height < (
        low_resolution_threshold
    )
    if doubled:
        request = cv2.resize(
            image,
            (width * 2, height * 2),
            interpolation=cv2.INTER_CUBIC,
        )
    else:
        request = image
    request_height, request_width = request.shape[:2]
    return request, {
        "profile": "official_spotting_v1",
        "original_width": int(width),
        "original_height": int(height),
        "request_width": int(request_width),
        "request_height": int(request_height),
        "low_resolution_doubled": bool(doubled),
        "aspect_ratio_preserved": True,
        "coordinate_space": "normalized_0_1000",
    }


__all__ = [
    "PADDLE_SPOTTING_LOW_RES_DOUBLE_THRESHOLD",
    "PADDLE_SPOTTING_OFFICIAL_IMAGE_MAX_PIXELS",
    "preprocess_spotting_image",
]
