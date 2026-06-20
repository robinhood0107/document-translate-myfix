from __future__ import annotations

import cv2
import numpy as np

from .geometry import calculate_iou, is_mostly_contained


def _normalize_box(box, image_shape: tuple[int, ...]) -> tuple[int, int, int, int] | None:
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


def _component_is_rule_like(w: int, h: int, area: int) -> bool:
    long_side = max(int(w), int(h))
    short_side = min(int(w), int(h))
    if long_side < 48 or short_side > 5:
        return False
    aspect = float(long_side) / float(max(1, short_side))
    fill_ratio = float(area) / float(max(1, int(w) * int(h)))
    return aspect >= 16.0 and fill_ratio >= 0.35


def _bubble_search_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=np.uint8)
    if h <= 0 or w <= 0:
        return mask
    pad = max(4, int(round(min(h, w) * 0.05)))
    if w <= pad * 2 or h <= pad * 2:
        return mask
    mask[pad:h - pad, pad:w - pad] = 255
    return mask


def _filter_dark_anchor(candidate_mask: np.ndarray, search_mask: np.ndarray) -> np.ndarray:
    if (
        candidate_mask.size == 0
        or search_mask.size == 0
        or candidate_mask.shape[:2] != search_mask.shape[:2]
        or not np.any(candidate_mask)
        or not np.any(search_mask)
    ):
        return np.zeros_like(candidate_mask, dtype=np.uint8)

    search = np.where(search_mask > 0, 255, 0).astype(np.uint8)
    candidate = np.where((candidate_mask > 0) & (search > 0), 255, 0).astype(np.uint8)
    if not np.any(candidate):
        return np.zeros_like(candidate, dtype=np.uint8)

    coords = cv2.findNonZero(search)
    if coords is None:
        return np.zeros_like(candidate, dtype=np.uint8)
    sx, sy, sw, sh = cv2.boundingRect(coords)
    sx2 = sx + sw
    sy2 = sy + sh
    search_area = max(1, int(sw) * int(sh))
    min_area = max(8, int(round(search_area * 0.00012)))
    max_bbox_ratio = 0.55
    edge_bbox_ratio = 0.10

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (candidate > 0).astype(np.uint8),
        8,
        cv2.CV_32S,
    )
    filtered = np.zeros_like(candidate, dtype=np.uint8)
    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if int(area) < min_area or w <= 0 or h <= 0:
            continue
        bbox_area = int(w * h)
        if bbox_area > int(round(search_area * max_bbox_ratio)):
            continue

        long_side = max(int(w), int(h))
        short_side = min(int(w), int(h))
        aspect = float(long_side) / float(max(1, short_side))
        fill_ratio = float(area) / float(max(1, bbox_area))
        if short_side <= 4 and long_side >= 24:
            continue
        if aspect >= 12.0:
            continue
        if aspect >= 6.0 and fill_ratio <= 0.22:
            continue
        if _component_is_rule_like(int(w), int(h), int(area)):
            continue

        touches_search_edge = x <= sx or y <= sy or (x + w) >= sx2 or (y + h) >= sy2
        if touches_search_edge and bbox_area > int(round(search_area * edge_bbox_ratio)):
            continue

        component = labels[y:y + h, x:x + w] == label_idx
        filtered[y:y + h, x:x + w][component] = 255

    return np.where(filtered > 0, 255, 0).astype(np.uint8)


def _bubble_glyph_mask(crop_rgb: np.ndarray) -> np.ndarray:
    if crop_rgb.size == 0:
        return np.zeros(crop_rgb.shape[:2], dtype=np.uint8)

    search_mask = _bubble_search_mask(crop_rgb.shape[:2])
    if not np.any(search_mask):
        return search_mask

    gray = cv2.cvtColor(crop_rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(crop_rgb.astype(np.uint8, copy=False), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    search = search_mask > 0
    values = gray[search]
    if values.size < 8:
        return np.zeros_like(search_mask)

    mean = float(np.mean(values))
    std = max(1.0, float(np.std(values)))
    dark_ceiling = int(round(min(125.0, max(35.0, mean - std * 0.30))))
    dark_candidate = np.where((gray <= dark_ceiling) & search, 255, 0).astype(np.uint8)
    dark_anchor = _filter_dark_anchor(dark_candidate, search_mask)
    if not np.any(dark_anchor):
        return np.zeros_like(search_mask)

    bright_floor = int(round(max(165.0, min(230.0, mean + std * 0.45))))
    outline_zone = cv2.dilate(
        dark_anchor,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)),
        iterations=1,
    )
    bright_outline = np.where(
        (gray >= bright_floor) & (sat <= 90) & (outline_zone > 0) & search,
        255,
        0,
    ).astype(np.uint8)
    glyph_mask = cv2.bitwise_or(dark_anchor, bright_outline)
    glyph_mask = cv2.dilate(
        glyph_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
        iterations=1,
    )
    return np.where((glyph_mask > 0) & search, 255, 0).astype(np.uint8)


def _overlaps_existing_text(candidate: tuple[int, int, int, int], text_boxes: list[tuple[int, int, int, int]]) -> bool:
    for text_box in text_boxes:
        if calculate_iou(list(candidate), list(text_box)) >= 0.20:
            return True
        if is_mostly_contained(list(text_box), list(candidate), 0.65):
            return True
        if is_mostly_contained(list(candidate), list(text_box), 0.65):
            return True
    return False


def _iter_boxes(boxes):
    if boxes is None:
        return []
    arr = np.asarray(boxes)
    if arr.size == 0:
        return []
    if arr.ndim == 1:
        return [arr.tolist()]
    return arr.tolist()


def detect_bubble_text_rescue_boxes(
    image_rgb: np.ndarray,
    bubble_boxes,
    text_boxes,
) -> list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]]:
    """Infer missed text boxes inside detected speech bubbles.

    This deliberately only searches inside bubble boxes that do not already have
    a matched text box. It is a conservative detector rescue, not a page-wide OCR
    or SFX detector.
    """

    if image_rgb.size == 0:
        return []

    normalized_bubbles = [
        box for box in (_normalize_box(box, image_rgb.shape) for box in _iter_boxes(bubble_boxes)) if box is not None
    ]
    normalized_text = [
        box for box in (_normalize_box(box, image_rgb.shape) for box in _iter_boxes(text_boxes)) if box is not None
    ]
    if not normalized_bubbles:
        return []

    rescued: list[tuple[tuple[int, int, int, int], tuple[int, int, int, int]]] = []
    for bubble in normalized_bubbles:
        if any(
            is_mostly_contained(list(bubble), list(text_box), 0.55)
            or calculate_iou(list(bubble), list(text_box)) >= 0.08
            for text_box in normalized_text
        ):
            continue

        x1, y1, x2, y2 = bubble
        crop = np.ascontiguousarray(image_rgb[y1:y2, x1:x2])
        glyph_mask = _bubble_glyph_mask(crop)
        if not np.any(glyph_mask):
            continue

        coords = cv2.findNonZero(glyph_mask)
        if coords is None:
            continue
        gx, gy, gw, gh = cv2.boundingRect(coords)
        if gw < 8 or gh < 8:
            continue

        bubble_area = max(1, (x2 - x1) * (y2 - y1))
        bbox_area = int(gw * gh)
        if bbox_area < max(24, int(round(bubble_area * 0.002))):
            continue
        if bbox_area > int(round(bubble_area * 0.65)):
            continue

        pad = max(2, int(round(min(gw, gh) * 0.06)))
        candidate = (
            max(x1, x1 + gx - pad),
            max(y1, y1 + gy - pad),
            min(x2, x1 + gx + gw + pad),
            min(y2, y1 + gy + gh + pad),
        )
        if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
            continue
        if _overlaps_existing_text(candidate, normalized_text):
            continue

        normalized_text.append(candidate)
        rescued.append((candidate, bubble))

    return rescued
