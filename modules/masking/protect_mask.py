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


def _line_protect_mask(image: np.ndarray, dilate_px: int, canny_low: int, canny_high: int) -> np.ndarray:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.astype(np.uint8)
    edges = cv2.Canny(gray, threshold1=canny_low, threshold2=canny_high)
    _, dark_lines = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)
    protect = cv2.bitwise_or(edges, dark_lines)
    if dilate_px > 0:
        kernel = np.ones((2 * dilate_px + 1, 2 * dilate_px + 1), dtype=np.uint8)
        protect = cv2.dilate(protect, kernel, iterations=1)
    return protect


def build_protect_mask(
    image: np.ndarray,
    blocks: Iterable,
    settings: ProtectMaskSettings | None = None,
) -> np.ndarray:
    cfg = settings or ProtectMaskSettings()
    base = np.zeros(image.shape[:2], dtype=np.uint8)
    if not cfg.keep_existing_lines:
        return base

    protect = _line_protect_mask(
        image,
        dilate_px=max(0, int(cfg.line_dilate_px)),
        canny_low=int(cfg.canny_low),
        canny_high=int(cfg.canny_high),
    )

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
