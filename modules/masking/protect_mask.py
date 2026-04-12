from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np


@dataclass(slots=True)
class ProtectMaskSettings:
    keep_existing_lines: bool = True
    border_band_px: int = 3
    canny_low: int = 80
    canny_high: int = 180
    line_dilate_px: int = 1


def _normalize_box(box) -> tuple[int, int, int, int] | None:
    if box is None or len(box) < 4:
        return None
    x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _bubble_border_band(shape: tuple[int, int], bubble_box, width: int) -> np.ndarray:
    band = np.zeros(shape, dtype=np.uint8)
    norm = _normalize_box(bubble_box)
    if norm is None:
        return band
    x1, y1, x2, y2 = norm
    cv2.rectangle(band, (x1, y1), (x2, y2), 255, thickness=max(1, width))
    if width > 1:
        kernel = np.ones((width, width), dtype=np.uint8)
        band = cv2.dilate(band, kernel, iterations=1)
    return band


def build_protect_mask(
    image: np.ndarray,
    blocks: Iterable,
    settings: ProtectMaskSettings | None = None,
) -> np.ndarray:
    cfg = settings or ProtectMaskSettings()
    base = np.zeros(image.shape[:2], dtype=np.uint8)
    if not cfg.keep_existing_lines:
        return base

    # Protect only speech-bubble border bands. The previous global edge/dark-line
    # mask was preserving too much interior text structure, which made inpainting
    # look like it was not happening even when the mask was correct.
    protect = np.zeros_like(base)
    for block in blocks or []:
        if getattr(block, "text_class", "") != "text_bubble":
            continue
        bubble_box = getattr(block, "bubble_xyxy", None)
        if bubble_box is None:
            continue
        protect = cv2.bitwise_or(
            protect,
            _bubble_border_band(
                image.shape[:2],
                bubble_box,
                width=max(1, int(cfg.border_band_px)),
            ),
        )

    return np.where(protect > 0, 255, 0).astype(np.uint8)
