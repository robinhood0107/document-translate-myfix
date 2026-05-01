from __future__ import annotations

from typing import Iterable

import cv2
import numpy as np


def normalize_xyxy(box, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
    if box is None or len(box) < 4:
        return None
    h, w = image_shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _padding_for_text_free(
    width: int,
    height: int,
    *,
    residue_risk: bool,
    image_area_ratio: float,
) -> tuple[int, int]:
    if residue_risk:
        pad_x = max(6, min(26, int(round(width * 0.12))))
        pad_y = max(8, min(34, int(round(height * 0.18))))
    else:
        pad_x = max(4, min(18, int(round(width * 0.08))))
        pad_y = max(5, min(24, int(round(height * 0.12))))

    # Avoid turning already large free-text regions into visible rectangular edits.
    if image_area_ratio > 0.035:
        pad_x = max(2, int(round(pad_x * 0.55)))
        pad_y = max(3, int(round(pad_y * 0.55)))
    elif image_area_ratio > 0.020:
        pad_x = max(3, int(round(pad_x * 0.75)))
        pad_y = max(4, int(round(pad_y * 0.75)))

    aspect = height / max(1, width)
    if aspect >= 2.5:
        # Tall free text usually sits near panel art; keep side padding restrained.
        pad_x = min(pad_x, 12 if not residue_risk else 16)
        pad_y = max(pad_y, min(34 if residue_risk else 24, int(round(height * 0.14))))
    return pad_x, pad_y


def build_text_free_erase_envelope(
    block,
    image_shape: tuple[int, ...],
    *,
    residue_risk: bool = False,
) -> tuple[int, int, int, int] | None:
    if str(getattr(block, "text_class", "") or "") != "text_free":
        return None
    bbox = normalize_xyxy(getattr(block, "xyxy", None), image_shape)
    if bbox is None:
        return None

    x1, y1, x2, y2 = bbox
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    image_area = max(1, int(image_shape[0]) * int(image_shape[1]))
    area_ratio = (width * height) / float(image_area)
    pad_x, pad_y = _padding_for_text_free(
        width,
        height,
        residue_risk=residue_risk,
        image_area_ratio=area_ratio,
    )

    h, w = image_shape[:2]
    envelope = (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(w, x2 + pad_x),
        min(h, y2 + pad_y),
    )
    if envelope[2] <= envelope[0] or envelope[3] <= envelope[1]:
        return None
    return envelope


def mask_for_xyxy(image_shape: tuple[int, ...], xyxy: tuple[int, int, int, int] | None) -> np.ndarray:
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    if xyxy is None:
        return mask
    x1, y1, x2, y2 = xyxy
    mask[y1:y2, x1:x2] = 255
    return mask


def merge_close_components_within_envelope(
    mask: np.ndarray,
    envelope: tuple[int, int, int, int] | None,
    *,
    kernel_size: int = 5,
) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    bounded = np.where(mask > 0, 255, 0).astype(np.uint8)
    if envelope is not None:
        bounded = cv2.bitwise_and(bounded, mask_for_xyxy(bounded.shape, envelope))
    if not np.any(bounded):
        return bounded
    kernel_size = max(3, int(kernel_size) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    merged = cv2.morphologyEx(bounded, cv2.MORPH_CLOSE, kernel, iterations=1)
    merged = cv2.dilate(merged, np.ones((3, 3), np.uint8), iterations=1)
    if envelope is not None:
        merged = cv2.bitwise_and(merged, mask_for_xyxy(merged.shape, envelope))
    return np.where(merged > 0, 255, 0).astype(np.uint8)


def append_unique_box(boxes: Iterable, box: tuple[int, int, int, int] | None) -> list[list[int]]:
    output: list[list[int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw in boxes or []:
        try:
            norm = tuple(int(float(v)) for v in list(raw)[:4])
        except (TypeError, ValueError):
            continue
        if norm[2] <= norm[0] or norm[3] <= norm[1] or norm in seen:
            continue
        seen.add(norm)
        output.append(list(norm))
    if box is not None and box not in seen:
        output.append(list(box))
    return output
