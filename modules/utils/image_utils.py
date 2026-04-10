from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
import imkit as imk
from PySide6.QtGui import QColor

from modules.detection.utils.content import get_inpaint_bboxes
from modules.masking import CTDRefiner, CTDRefinerSettings, ProtectMaskSettings, build_protect_mask
from modules.utils.textblock import TextBlock


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


def _populate_legacy_inpaint_boxes(img: np.ndarray, blk_list: list[TextBlock]) -> None:
    for blk in blk_list:
        if getattr(blk, "xyxy", None) is None:
            continue
        bboxes = get_inpaint_bboxes(
            blk.xyxy,
            img,
            bubble_bbox=getattr(blk, "bubble_xyxy", None),
        )
        blk.inpaint_bboxes = bboxes


def _generate_legacy_mask(img: np.ndarray, blk_list: list[TextBlock], default_padding: int = 5) -> np.ndarray:
    h, w, _ = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    long_edge = 2048

    for blk in blk_list:
        if getattr(blk, "xyxy", None) is None:
            continue

        bboxes = getattr(blk, "inpaint_bboxes", None)
        if bboxes is None:
            bboxes = get_inpaint_bboxes(
                blk.xyxy,
                img,
                bubble_bbox=getattr(blk, "bubble_xyxy", None),
            )
            blk.inpaint_bboxes = bboxes
        if bboxes is None or len(bboxes) == 0:
            continue

        xs = [x for x1, _, x2, _ in bboxes for x in (x1, x2)]
        ys = [y for _, y1, _, y2 in bboxes for y in (y1, y2)]
        min_x, max_x = int(min(xs)), int(max(xs))
        min_y, max_y = int(min(ys)), int(max(ys))
        roi_w, roi_h = max_x - min_x + 1, max_y - min_y + 1

        ds = max(1.0, max(roi_w, roi_h) / long_edge)
        mw, mh = int(roi_w / ds) + 2, int(roi_h / ds) + 2
        pad_offset = 1

        small = np.zeros((mh, mw), dtype=np.uint8)
        for x1, y1, x2, y2 in bboxes:
            x1i = int((x1 - min_x) / ds) + pad_offset
            y1i = int((y1 - min_y) / ds) + pad_offset
            x2i = int((x2 - min_x) / ds) + pad_offset
            y2i = int((y2 - min_y) / ds) + pad_offset
            small = imk.rectangle(small, (x1i, y1i), (x2i, y2i), 255, -1)

        kernel = imk.get_structuring_element(imk.MORPH_RECT, (15, 15))
        closed = imk.morphology_ex(small, imk.MORPH_CLOSE, kernel)
        contours, _ = imk.find_contours(closed)
        if not contours:
            continue

        polys = []
        for cnt in contours:
            pts = cnt.squeeze(1)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            pts_f = (pts.astype(np.float32) - pad_offset) * ds
            pts_f[:, 0] += min_x
            pts_f[:, 1] += min_y
            polys.append(pts_f.astype(np.int32))
        if not polys:
            continue

        block_mask = np.zeros((h, w), dtype=np.uint8)
        block_mask = imk.fill_poly(block_mask, polys, 255)

        kernel_size = max(default_padding, 5)
        widths = [max(1, x2 - x1) for x1, _, x2, _ in bboxes]
        heights = [max(1, y2 - y1) for _, y1, _, y2 in bboxes]
        median_span = max(float(np.median(widths)), float(np.median(heights)))
        dynamic_padding = int(round(median_span * 0.08))
        kernel_size = max(kernel_size, min(9, max(3, dynamic_padding)))

        if getattr(blk, "text_class", None) == "text_bubble" and getattr(blk, "bubble_xyxy", None) is not None:
            bx1, by1, bx2, by2 = blk.bubble_xyxy
            valid = [
                p
                for p in polys
                if (p[:, 0] >= bx1).all()
                and (p[:, 0] <= bx2).all()
                and (p[:, 1] >= by1).all()
                and (p[:, 1] <= by2).all()
            ]
            if valid:
                dists = []
                for poly in valid:
                    dists.extend([
                        poly[:, 0].min() - bx1,
                        bx2 - poly[:, 0].max(),
                        poly[:, 1].min() - by1,
                        by2 - poly[:, 1].max(),
                    ])
                min_dist = min(dists)
                if kernel_size >= min_dist:
                    kernel_size = max(1, int(min_dist * 0.8))

        dil_kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated = imk.dilate(block_mask, dil_kernel, iterations=4)
        mask = np.bitwise_or(mask, dilated)

    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _normalize_mask_refiner_settings(settings: dict[str, Any] | None) -> dict[str, Any]:
    data = dict(settings or {})
    data.setdefault("mask_refiner", "legacy_bbox")
    data.setdefault("keep_existing_lines", False)
    data.setdefault("ctd_detect_size", 1280)
    data.setdefault("ctd_det_rearrange_max_batches", 4)
    data.setdefault("ctd_device", "cuda")
    data.setdefault("ctd_font_size_multiplier", 1.0)
    data.setdefault("ctd_font_size_max", -1)
    data.setdefault("ctd_font_size_min", -1)
    data.setdefault("ctd_mask_dilate_size", 2)
    return data


def generate_mask(
    img: np.ndarray,
    blk_list: list[TextBlock],
    default_padding: int = 5,
    settings: dict[str, Any] | None = None,
    return_details: bool = False,
):
    cfg = _normalize_mask_refiner_settings(settings)
    _populate_legacy_inpaint_boxes(img, blk_list)

    raw_mask = None
    refined_mask = None
    protect_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    backend = "legacy"
    device = "cpu"
    fallback_used = False

    if cfg.get("mask_refiner") == "ctd":
        try:
            refiner = CTDRefiner(
                CTDRefinerSettings(
                    detect_size=int(cfg["ctd_detect_size"]),
                    det_rearrange_max_batches=int(cfg["ctd_det_rearrange_max_batches"]),
                    device=str(cfg["ctd_device"]),
                    font_size_multiplier=float(cfg["ctd_font_size_multiplier"]),
                    font_size_max=int(cfg["ctd_font_size_max"]),
                    font_size_min=int(cfg["ctd_font_size_min"]),
                    mask_dilate_size=int(cfg["ctd_mask_dilate_size"]),
                )
            )
            result = refiner.refine(img, blk_list)
            raw_mask = result.raw_mask
            refined_mask = result.refined_mask
            backend = result.backend
            device = result.device
            protect_mask = build_protect_mask(
                img,
                blk_list,
                ProtectMaskSettings(keep_existing_lines=bool(cfg.get("keep_existing_lines", False))),
            )
            final_mask = cv2.bitwise_and(result.final_mask, cv2.bitwise_not(protect_mask))
            if not np.any(final_mask) and np.any(result.final_mask):
                final_mask = result.final_mask.copy()
                fallback_used = True
            if not np.any(final_mask):
                final_mask = _generate_legacy_mask(img, blk_list, default_padding=default_padding)
                fallback_used = True
        except Exception:
            raw_mask = None
            refined_mask = None
            final_mask = _generate_legacy_mask(img, blk_list, default_padding=default_padding)
            fallback_used = True
    else:
        final_mask = _generate_legacy_mask(img, blk_list, default_padding=default_padding)

    final_mask = np.where(final_mask > 0, 255, 0).astype(np.uint8)
    if raw_mask is None:
        raw_mask = final_mask.copy()
    if refined_mask is None:
        refined_mask = final_mask.copy()

    if return_details:
        return {
            "raw_mask": np.where(raw_mask > 0, 255, 0).astype(np.uint8),
            "refined_mask": np.where(refined_mask > 0, 255, 0).astype(np.uint8),
            "protect_mask": np.where(protect_mask > 0, 255, 0).astype(np.uint8),
            "final_mask": final_mask,
            "mask_refiner": cfg.get("mask_refiner", "legacy_bbox"),
            "keep_existing_lines": bool(cfg.get("keep_existing_lines", False)),
            "refiner_backend": backend,
            "refiner_device": device,
            "fallback_used": fallback_used,
        }
    return final_mask
