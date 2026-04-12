from __future__ import annotations

from typing import Any

import cv2
import numpy as np


CARRIER_KIND_NONE = "none"
CARRIER_KIND_SPEECH_BUBBLE = "speech_bubble"
CARRIER_KIND_CAPTION_PLATE = "caption_plate"


def default_carrier_kind(text_class: str | None) -> str:
    return CARRIER_KIND_SPEECH_BUBBLE if (text_class or "") == "text_bubble" else CARRIER_KIND_NONE


def normalize_xyxy(box, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    if box is None or len(box) < 4:
        return None
    img_h, img_w = image_shape[:2]
    x1, y1, x2, y2 = [int(float(v)) for v in box[:4]]
    x1 = max(0, min(x1, img_w))
    x2 = max(0, min(x2, img_w))
    y1 = max(0, min(y1, img_h))
    y2 = max(0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def intersect_xyxy(
    lhs: tuple[int, int, int, int] | None,
    rhs: tuple[int, int, int, int] | None,
) -> tuple[int, int, int, int] | None:
    if lhs is None:
        return rhs
    if rhs is None:
        return lhs
    x1 = max(lhs[0], rhs[0])
    y1 = max(lhs[1], rhs[1])
    x2 = min(lhs[2], rhs[2])
    y2 = min(lhs[3], rhs[3])
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def expand_xyxy(
    box,
    image_shape: tuple[int, int] | tuple[int, int, int],
    *,
    pad_ratio: float,
    min_pad: int,
    max_pad: int,
) -> tuple[int, int, int, int] | None:
    norm = normalize_xyxy(box, image_shape)
    if norm is None:
        return None
    x1, y1, x2, y2 = norm
    img_h, img_w = image_shape[:2]
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad = min(max_pad, max(min_pad, int(round(max(width, height) * pad_ratio))))
    ex1 = max(0, x1 - pad)
    ey1 = max(0, y1 - pad)
    ex2 = min(img_w, x2 + pad)
    ey2 = min(img_h, y2 + pad)
    if ex2 <= ex1 or ey2 <= ey1:
        return None
    return ex1, ey1, ex2, ey2


def _empty_metrics() -> dict[str, float]:
    return {
        "white_ratio": 0.0,
        "chroma_ratio": 0.0,
        "mean_chroma": 0.0,
        "mean_lightness": 0.0,
        "lightness_std": 0.0,
        "dark_border_ratio": 0.0,
        "analysis_area": 0.0,
    }


def _ensure_three_channel(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        return np.stack([arr] * 3, axis=-1).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return np.repeat(arr, 3, axis=2).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return arr[:, :, :3].astype(np.uint8)
    raise ValueError("unsupported image shape")


def resolve_caption_plate_mask_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    explicit = normalize_xyxy(getattr(block, "carrier_mask_roi_xyxy", None), image_shape)
    if explicit is not None:
        return explicit
    if (getattr(block, "carrier_kind", "") or "") != CARRIER_KIND_CAPTION_PLATE:
        return None
    local_roi = expand_xyxy(
        getattr(block, "xyxy", None),
        image_shape,
        pad_ratio=0.08,
        min_pad=4,
        max_pad=12,
    )
    bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image_shape)
    return intersect_xyxy(local_roi, bubble_roi)


def resolve_effective_text_bubble_roi(block, image_shape: tuple[int, int] | tuple[int, int, int]) -> tuple[int, int, int, int] | None:
    if (getattr(block, "text_class", "") or "") != "text_bubble":
        return None
    return resolve_caption_plate_mask_roi(block, image_shape) or normalize_xyxy(
        getattr(block, "bubble_xyxy", None),
        image_shape,
    )


def classify_text_block_carrier(
    image: np.ndarray,
    block,
) -> tuple[str, dict[str, float], tuple[int, int, int, int] | None]:
    default_kind = default_carrier_kind(getattr(block, "text_class", "") or "")
    metrics = _empty_metrics()
    if default_kind != CARRIER_KIND_SPEECH_BUBBLE:
        return default_kind, metrics, None

    bubble_roi = normalize_xyxy(getattr(block, "bubble_xyxy", None), image.shape)
    text_roi = normalize_xyxy(getattr(block, "xyxy", None), image.shape)
    if bubble_roi is None or text_roi is None:
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None

    bx1, by1, bx2, by2 = bubble_roi
    bubble_w = bx2 - bx1
    bubble_h = by2 - by1
    if bubble_w <= 6 or bubble_h <= 6:
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None

    inner_roi = normalize_xyxy((bx1 + 3, by1 + 3, bx2 - 3, by2 - 3), image.shape)
    if inner_roi is None:
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None
    caption_roi = intersect_xyxy(
        expand_xyxy(text_roi, image.shape, pad_ratio=0.08, min_pad=4, max_pad=12),
        bubble_roi,
    )

    crop = _ensure_three_channel(image)[by1:by2, bx1:bx2]
    analysis_mask = np.zeros((bubble_h, bubble_w), dtype=bool)
    ix1, iy1, ix2, iy2 = inner_roi
    analysis_mask[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1] = True
    if caption_roi is not None:
        ex1, ey1, ex2, ey2 = caption_roi
        analysis_mask[ey1 - by1:ey2 - by1, ex1 - bx1:ex2 - bx1] = False

    analysis_area = int(np.count_nonzero(analysis_mask))
    bubble_area = int(max(1, bubble_w * bubble_h))
    metrics["analysis_area"] = float(analysis_area)
    if analysis_area < max(400, int(round(bubble_area * 0.10))):
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None

    border_mask = np.ones((bubble_h, bubble_w), dtype=bool)
    border_mask[iy1 - by1:iy2 - by1, ix1 - bx1:ix2 - bx1] = False

    lab = cv2.cvtColor(crop, cv2.COLOR_RGB2LAB)
    lightness = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32) - 128.0
    b = lab[:, :, 2].astype(np.float32) - 128.0
    chroma = np.sqrt(np.maximum(0.0, a * a + b * b))

    analysis_lightness = lightness[analysis_mask]
    analysis_chroma = chroma[analysis_mask]
    if analysis_lightness.size == 0 or analysis_chroma.size == 0:
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None

    white_mask = (lightness >= 205.0) & (chroma <= 12.0)
    chroma_mask = chroma >= 18.0
    dark_border_mask = (lightness <= 120.0) & (chroma <= 20.0)

    metrics.update(
        {
            "white_ratio": float(np.count_nonzero(white_mask & analysis_mask)) / float(max(1, analysis_area)),
            "chroma_ratio": float(np.count_nonzero(chroma_mask & analysis_mask)) / float(max(1, analysis_area)),
            "mean_chroma": float(np.mean(analysis_chroma)),
            "mean_lightness": float(np.mean(analysis_lightness)),
            "lightness_std": float(np.std(analysis_lightness)),
            "dark_border_ratio": float(np.count_nonzero(dark_border_mask & border_mask)) / float(max(1, np.count_nonzero(border_mask))),
        }
    )
    if any(not np.isfinite(value) for value in metrics.values()):
        return CARRIER_KIND_SPEECH_BUBBLE, _empty_metrics(), None

    is_colored_caption_plate = (
        metrics["dark_border_ratio"] <= 0.18
        and metrics["white_ratio"] <= 0.38
        and (
            metrics["mean_chroma"] >= 10.0
            or metrics["chroma_ratio"] >= 0.18
            or (
                metrics["white_ratio"] <= 0.30
                and 80.0 <= metrics["mean_lightness"] <= 195.0
                and metrics["lightness_std"] >= 8.0
            )
        )
    )
    is_dark_caption_plate = (
        metrics["white_ratio"] <= 0.28
        and metrics["mean_chroma"] >= 10.0
        and metrics["lightness_std"] >= 20.0
        and metrics["dark_border_ratio"] <= 0.32
    )
    is_caption_plate = is_colored_caption_plate or is_dark_caption_plate
    if not is_caption_plate:
        return CARRIER_KIND_SPEECH_BUBBLE, metrics, None

    return CARRIER_KIND_CAPTION_PLATE, metrics, caption_roi


def annotate_text_block_carriers(image: np.ndarray, blocks: list[Any]) -> None:
    for block in blocks or []:
        kind, metrics, roi = classify_text_block_carrier(image, block)
        block.carrier_kind = kind
        block.carrier_metrics = dict(metrics)
        block.carrier_mask_roi_xyxy = list(roi) if roi is not None else None
