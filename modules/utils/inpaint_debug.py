from __future__ import annotations

import json
import os
from typing import Iterable

import imkit as imk
import numpy as np
from PIL import Image, ImageDraw


def ensure_three_channel(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1).astype(np.uint8)
    if image.ndim == 3 and image.shape[2] == 1:
        channel = image[:, :, 0]
        return np.stack([channel] * 3, axis=-1).astype(np.uint8)
    return image.astype(np.uint8)


def has_debug_exports(export_settings: dict | None) -> bool:
    settings = export_settings or {}
    return any(
        bool(settings.get(key, False))
        for key in (
            "export_detector_overlay",
            "export_raw_mask",
            "export_mask_overlay",
            "export_cleanup_mask_delta",
            "export_debug_metadata",
        )
    )


def _normalize_mask(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        return np.zeros(image_shape[:2], dtype=np.uint8)
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    if arr.shape[:2] != image_shape[:2]:
        normalized = np.zeros(image_shape[:2], dtype=np.uint8)
        h = min(normalized.shape[0], arr.shape[0])
        w = min(normalized.shape[1], arr.shape[1])
        normalized[:h, :w] = arr[:h, :w]
        arr = normalized
    return np.where(arr > 0, 255, 0).astype(np.uint8)


def _mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    normalized = np.where(mask > 0, 255, 0).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


def _build_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 0, 0),
    alpha: float = 0.35,
) -> np.ndarray:
    base = ensure_three_channel(image).astype(np.float32)
    overlay = base.copy()
    mask_pixels = mask > 0
    if np.any(mask_pixels):
        tint = np.array(color, dtype=np.float32)
        overlay[mask_pixels] = base[mask_pixels] * (1.0 - alpha) + tint * alpha
    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def _collect_bubble_boxes(blocks: Iterable) -> list[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    bubbles: list[tuple[int, int, int, int]] = []
    for block in blocks or []:
        bubble = getattr(block, "bubble_xyxy", None)
        if bubble is None or len(bubble) < 4:
            continue
        box = tuple(int(float(v)) for v in bubble[:4])
        if box in seen:
            continue
        seen.add(box)
        bubbles.append(box)
    return bubbles


def build_detector_overlay(image: np.ndarray, blocks: Iterable) -> np.ndarray:
    canvas = Image.fromarray(ensure_three_channel(image))
    draw = ImageDraw.Draw(canvas)
    palette = {
        "bubble": (255, 170, 0),
        "text_bubble": (54, 197, 94),
        "text_free": (63, 135, 245),
    }

    for x1, y1, x2, y2 in _collect_bubble_boxes(blocks):
        draw.rectangle([x1, y1, x2, y2], outline=palette["bubble"], width=2)

    for block in blocks or []:
        bbox = getattr(block, "xyxy", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [int(float(v)) for v in bbox[:4]]
        color = palette.get(getattr(block, "text_class", ""), (255, 64, 64))
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

    return np.array(canvas, dtype=np.uint8)


def serialize_inpaint_block(block, index: int) -> dict:
    inpaint_boxes = []
    raw_inpaint_boxes = getattr(block, "inpaint_bboxes", None)
    if raw_inpaint_boxes is not None:
        for box in raw_inpaint_boxes:
            inpaint_boxes.append([int(float(v)) for v in box[:4]])

    return {
        "index": int(index),
        "xyxy": [int(float(v)) for v in getattr(block, "xyxy", (0, 0, 0, 0))[:4]],
        "bubble_xyxy": (
            [int(float(v)) for v in getattr(block, "bubble_xyxy", ())[:4]]
            if getattr(block, "bubble_xyxy", None) is not None
            else None
        ),
        "text_class": getattr(block, "text_class", "") or "",
        "inpaint_bboxes": inpaint_boxes,
    }


def build_inpaint_debug_metadata(
    *,
    image_path: str,
    run_type: str,
    detector_key: str,
    detector_engine: str,
    device: str,
    inpainter: str,
    hd_strategy: str,
    blocks: Iterable,
    raw_mask: np.ndarray | None,
    cleanup_delta: np.ndarray | None,
    cleanup_stats: dict | None,
) -> dict:
    block_list = list(blocks or [])
    cleanup_stats = cleanup_stats or {}
    raw_mask_pixels = int(np.count_nonzero(raw_mask)) if raw_mask is not None else 0
    cleanup_delta_pixels = int(np.count_nonzero(cleanup_delta)) if cleanup_delta is not None else 0
    return {
        "image_path": image_path,
        "run_type": run_type,
        "detector_key": detector_key,
        "detector_engine": detector_engine,
        "device": device,
        "inpainter": inpainter,
        "hd_strategy": hd_strategy,
        "block_count": len(block_list),
        "raw_mask_pixel_count": raw_mask_pixels,
        "cleanup_delta_pixel_count": cleanup_delta_pixels,
        "cleanup_applied": bool(cleanup_stats.get("applied", False)),
        "cleanup_component_count": int(cleanup_stats.get("component_count", 0) or 0),
        "cleanup_block_count": int(cleanup_stats.get("block_count", 0) or 0),
        "blocks": [serialize_inpaint_block(block, idx) for idx, block in enumerate(block_list)],
    }


def _write_image(base_dir: str, folder: str, archive_bname: str, filename: str, image: np.ndarray) -> None:
    target_dir = os.path.join(base_dir, folder, archive_bname)
    os.makedirs(target_dir, exist_ok=True)
    imk.write_image(os.path.join(target_dir, filename), image)


def _write_json(base_dir: str, folder: str, archive_bname: str, filename: str, payload: dict) -> None:
    target_dir = os.path.join(base_dir, folder, archive_bname)
    os.makedirs(target_dir, exist_ok=True)
    with open(os.path.join(target_dir, filename), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def export_inpaint_debug_artifacts(
    *,
    export_root: str,
    archive_bname: str,
    page_base_name: str,
    image: np.ndarray,
    blocks: Iterable,
    export_settings: dict | None,
    raw_mask: np.ndarray | None,
    cleanup_delta: np.ndarray | None,
    metadata: dict | None,
) -> None:
    settings = export_settings or {}
    if not has_debug_exports(settings):
        return

    image_rgb = ensure_three_channel(image)
    normalized_raw_mask = _normalize_mask(raw_mask, image_rgb.shape)
    normalized_cleanup_delta = _normalize_mask(cleanup_delta, image_rgb.shape)

    if settings.get("export_detector_overlay", False):
        _write_image(
            export_root,
            "detector_overlays",
            archive_bname,
            f"{page_base_name}_detector_overlay.png",
            build_detector_overlay(image_rgb, blocks),
        )

    if settings.get("export_raw_mask", False):
        _write_image(
            export_root,
            "raw_masks",
            archive_bname,
            f"{page_base_name}_raw_mask.png",
            _mask_to_rgb(normalized_raw_mask),
        )

    if settings.get("export_mask_overlay", False):
        _write_image(
            export_root,
            "mask_overlays",
            archive_bname,
            f"{page_base_name}_mask_overlay.png",
            _build_mask_overlay(image_rgb, normalized_raw_mask),
        )

    if settings.get("export_cleanup_mask_delta", False):
        _write_image(
            export_root,
            "cleanup_mask_delta",
            archive_bname,
            f"{page_base_name}_cleanup_delta.png",
            _mask_to_rgb(normalized_cleanup_delta),
        )

    if settings.get("export_debug_metadata", False):
        _write_json(
            export_root,
            "debug_metadata",
            archive_bname,
            f"{page_base_name}_debug.json",
            metadata or {},
        )
