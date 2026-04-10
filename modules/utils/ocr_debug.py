from __future__ import annotations

import json
import os

import imkit as imk
import numpy as np
from PIL import Image, ImageOps


OCR_STATUS_OK = "ok"
OCR_STATUS_EMPTY_INITIAL = "empty_initial"
OCR_STATUS_OK_AFTER_RETRY = "ok_after_retry"
OCR_STATUS_EMPTY_AFTER_RETRY = "empty_after_retry"
DEFAULT_RETRY_CROP_X_RATIO = 0.06
DEFAULT_RETRY_CROP_Y_RATIO = 0.10


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        channel = image[:, :, 0]
        return np.stack([channel] * 3, axis=-1).astype(np.uint8)
    return image.astype(np.uint8)


def _clip_bbox(x1: float, y1: float, x2: float, y2: float, image_shape: tuple[int, ...]) -> tuple[int, int, int, int]:
    img_h, img_w = image_shape[:2]
    x1_i = max(0, min(int(np.floor(x1)), img_w))
    y1_i = max(0, min(int(np.floor(y1)), img_h))
    x2_i = max(0, min(int(np.ceil(x2)), img_w))
    y2_i = max(0, min(int(np.ceil(y2)), img_h))
    return x1_i, y1_i, x2_i, y2_i


def expand_bbox(
    xyxy,
    image_shape: tuple[int, ...],
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    clamp_xyxy=None,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    x_pad = width * x_ratio
    y_pad = height * y_ratio
    expanded = _clip_bbox(x1 - x_pad, y1 - y_pad, x2 + x_pad, y2 + y_pad, image_shape)
    if clamp_xyxy is None:
        return expanded
    clamp_box = _clip_bbox(*[float(v) for v in clamp_xyxy], image_shape)
    ex1, ey1, ex2, ey2 = expanded
    cx1, cy1, cx2, cy2 = clamp_box
    return max(ex1, cx1), max(ey1, cy1), min(ex2, cx2), min(ey2, cy2)


def crop_bbox_image(
    image: np.ndarray,
    xyxy,
    auto_rotate_tall: bool = True,
) -> np.ndarray | None:
    x1, y1, x2, y2 = _clip_bbox(*[float(v) for v in xyxy], image.shape)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = image[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    crop = ensure_three_channel(crop)
    if auto_rotate_tall:
        h, w = crop.shape[:2]
        if h > 0 and w > 0 and (h / float(w)) >= 1.5:
            crop = np.rot90(crop)
    return crop


def crop_block_image(
    image: np.ndarray,
    xyxy,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    auto_rotate_tall: bool = True,
    clamp_xyxy=None,
) -> np.ndarray | None:
    bbox = expand_bbox(xyxy, image.shape, x_ratio=x_ratio, y_ratio=y_ratio, clamp_xyxy=clamp_xyxy)
    return crop_bbox_image(image, bbox, auto_rotate_tall=auto_rotate_tall)


def resolve_block_crop_bbox(
    block,
    image_shape: tuple[int, ...],
    *,
    x_ratio: float = 0.0,
    y_ratio: float = 0.0,
    bubble_as_clamp: bool = True,
    fallback_to_bubble: bool = True,
) -> tuple[tuple[int, int, int, int] | None, str]:
    text_bbox = getattr(block, "xyxy", None)
    bubble_bbox = getattr(block, "bubble_xyxy", None)
    clamp_bbox = bubble_bbox if bubble_as_clamp and bubble_bbox is not None else None

    if text_bbox is not None:
        bbox = expand_bbox(
            text_bbox,
            image_shape,
            x_ratio=x_ratio,
            y_ratio=y_ratio,
            clamp_xyxy=clamp_bbox,
        )
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            return bbox, "xyxy"

    if fallback_to_bubble and bubble_bbox is not None:
        bbox = expand_bbox(bubble_bbox, image_shape)
        x1, y1, x2, y2 = bbox
        if x2 > x1 and y2 > y1:
            return bbox, "bubble_fallback"

    return None, ""


def _otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    sum_total = np.dot(np.arange(256, dtype=np.float64), hist)
    sum_background = 0.0
    weight_background = 0.0
    max_variance = -1.0
    threshold = 0

    for idx in range(256):
        weight_background += hist[idx]
        if weight_background <= 0:
            continue

        weight_foreground = total - weight_background
        if weight_foreground <= 0:
            break

        sum_background += idx * hist[idx]
        mean_background = sum_background / weight_background
        mean_foreground = (sum_total - sum_background) / weight_foreground
        variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        if variance > max_variance:
            max_variance = variance
            threshold = idx

    return int(threshold)


def build_retry_crop_bbox(
    image_shape: tuple[int, ...],
    xyxy,
    *,
    clamp_xyxy=None,
    x_ratio: float = DEFAULT_RETRY_CROP_X_RATIO,
    y_ratio: float = DEFAULT_RETRY_CROP_Y_RATIO,
) -> tuple[int, int, int, int] | None:
    if xyxy is None:
        return None
    bbox = expand_bbox(image_shape=image_shape, xyxy=xyxy, x_ratio=x_ratio, y_ratio=y_ratio, clamp_xyxy=clamp_xyxy)
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return None
    return bbox


def build_retry_crop_from_bbox(image: np.ndarray, bbox) -> np.ndarray | None:
    if bbox is None:
        return None
    crop = crop_bbox_image(image, bbox, auto_rotate_tall=True)
    if crop is None:
        return None
    gray = imk.to_gray(crop)
    contrasted = np.array(ImageOps.autocontrast(Image.fromarray(gray)), dtype=np.uint8)
    threshold = _otsu_threshold(contrasted)
    binary = np.where(contrasted > threshold, 255, 0).astype(np.uint8)
    return ensure_three_channel(binary)


def build_retry_crop(
    image: np.ndarray,
    xyxy,
    *,
    clamp_xyxy=None,
    x_ratio: float = DEFAULT_RETRY_CROP_X_RATIO,
    y_ratio: float = DEFAULT_RETRY_CROP_Y_RATIO,
) -> np.ndarray | None:
    bbox = build_retry_crop_bbox(
        image.shape,
        xyxy,
        clamp_xyxy=clamp_xyxy,
        x_ratio=x_ratio,
        y_ratio=y_ratio,
    )
    return build_retry_crop_from_bbox(image, bbox)


def set_block_ocr_diagnostics(
    block,
    *,
    text: str,
    confidence: float,
    status: str,
    empty_reason: str,
    attempt_count: int,
    raw_text: str | None = None,
    sanitized_text: str | None = None,
) -> None:
    final_text = text or ""
    block.text = final_text
    block.ocr_confidence = float(confidence or 0.0)
    block.ocr_status = status
    block.ocr_empty_reason = empty_reason or ""
    block.ocr_attempt_count = int(attempt_count or 0)
    block.ocr_raw_text = final_text if raw_text is None else str(raw_text or "")
    block.ocr_sanitized_text = final_text if sanitized_text is None else str(sanitized_text or "")


def set_block_ocr_crop_diagnostics(
    block,
    *,
    effective_crop_xyxy=None,
    retry_crop_xyxy=None,
    crop_source: str | None = None,
) -> None:
    if effective_crop_xyxy is not None:
        block.ocr_effective_crop_xyxy = [int(float(v)) for v in effective_crop_xyxy]
    elif not hasattr(block, "ocr_effective_crop_xyxy"):
        block.ocr_effective_crop_xyxy = None

    if retry_crop_xyxy is not None:
        block.ocr_retry_crop_xyxy = [int(float(v)) for v in retry_crop_xyxy]
    elif not hasattr(block, "ocr_retry_crop_xyxy"):
        block.ocr_retry_crop_xyxy = None

    if crop_source is not None:
        block.ocr_crop_source = crop_source
    elif not hasattr(block, "ocr_crop_source"):
        block.ocr_crop_source = ""


def is_block_ocr_empty(block) -> bool:
    return getattr(block, "ocr_status", "") in {
        OCR_STATUS_EMPTY_INITIAL,
        OCR_STATUS_EMPTY_AFTER_RETRY,
    }


def build_ocr_debug_payload(
    page: str,
    ocr_engine: str,
    source_lang: str,
    blk_list,
) -> dict:
    payload = {
        "page": page,
        "ocr_engine": ocr_engine or "",
        "source_lang": source_lang or "",
        "blocks": [],
    }
    for idx, blk in enumerate(blk_list or []):
        x1, y1, x2, y2 = [int(float(v)) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
        payload["blocks"].append(
            {
                "index": idx,
                "bbox": [x1, y1, x2, y2],
                "text": getattr(blk, "text", "") or "",
                "raw_text": getattr(blk, "ocr_raw_text", getattr(blk, "text", "")) or "",
                "sanitized_text": getattr(blk, "ocr_sanitized_text", getattr(blk, "text", "")) or "",
                "confidence": float(getattr(blk, "ocr_confidence", 0.0) or 0.0),
                "status": getattr(blk, "ocr_status", "") or "",
                "empty_reason": getattr(blk, "ocr_empty_reason", "") or "",
                "attempt_count": int(getattr(blk, "ocr_attempt_count", 0) or 0),
                "effective_crop_xyxy": getattr(blk, "ocr_effective_crop_xyxy", None),
                "retry_crop_xyxy": getattr(blk, "ocr_retry_crop_xyxy", None),
                "crop_source": getattr(blk, "ocr_crop_source", "") or "",
            }
        )
    return payload


def export_ocr_debug_artifacts(
    output_dir: str,
    page_base_name: str,
    image: np.ndarray,
    blk_list,
    ocr_engine: str,
    source_lang: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    payload = build_ocr_debug_payload(page_base_name, ocr_engine, source_lang, blk_list)
    debug_path = os.path.join(output_dir, f"{page_base_name}_ocr_debug.json")
    with open(debug_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=4)

    for idx, blk in enumerate(blk_list or []):
        if getattr(blk, "ocr_status", "") != OCR_STATUS_EMPTY_AFTER_RETRY:
            continue
        retry_bbox = getattr(blk, "ocr_retry_crop_xyxy", None) or getattr(blk, "xyxy", (0, 0, 0, 0))
        retry_crop = build_retry_crop_from_bbox(image, retry_bbox)
        if retry_crop is None or retry_crop.size == 0:
            continue
        retry_path = os.path.join(output_dir, f"{page_base_name}_block_{idx}_retry.png")
        imk.write_image(retry_path, retry_crop)
