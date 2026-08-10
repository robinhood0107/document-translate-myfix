from __future__ import annotations

import cv2
import numpy as np


def extract_bubble_interior_cap_crop(
    image_crop: np.ndarray,
    seed_mask: np.ndarray,
    *,
    erode_px: int = 1,
    min_area_ratio: float = 0.05,
    max_area_ratio: float = 0.995,
    min_seed_coverage: float = 0.98,
    preserve_seed_after_erode: bool = True,
    erode_below_area_ratio: float | None = None,
    erode_shape: int = cv2.MORPH_ELLIPSE,
) -> np.ndarray | None:
    """Detect a speech-bubble interior that safely contains the text seed."""
    if image_crop is None or seed_mask is None or image_crop.size == 0:
        return None
    seed = np.asarray(seed_mask)
    if seed.ndim == 3:
        seed = seed[:, :, 0]
    if seed.shape[:2] != image_crop.shape[:2]:
        return None
    seed = np.where(seed > 0, 255, 0).astype(np.uint8)
    seed_pixel_count = int(np.count_nonzero(seed))
    if seed_pixel_count <= 0:
        return None

    try:
        from modules.source_parity_vendor.utils.textblock_mask import (
            extract_ballon_mask,
        )

        detected, _non_text_mask = extract_ballon_mask(image_crop, seed)
    except Exception:
        return None
    if detected is None or np.asarray(detected).size == 0:
        return None

    cap = np.asarray(detected)
    if cap.ndim == 3:
        cap = cap[:, :, 0]
    if cap.shape[:2] != seed.shape[:2]:
        cap = cv2.resize(
            cap,
            (seed.shape[1], seed.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
    cap = np.where(cap > 0, 255, 0).astype(np.uint8)
    area_ratio = float(np.count_nonzero(cap)) / float(max(1, cap.size))
    seed_coverage = float(np.count_nonzero((cap > 0) & (seed > 0))) / float(
        seed_pixel_count
    )
    if not (
        float(min_area_ratio) <= area_ratio <= float(max_area_ratio)
        and seed_coverage >= float(min_seed_coverage)
    ):
        return None

    px = max(0, int(erode_px))
    should_erode = px > 0 and (
        erode_below_area_ratio is None
        or area_ratio < float(erode_below_area_ratio)
    )
    if should_erode:
        kernel = cv2.getStructuringElement(
            int(erode_shape),
            (2 * px + 1, 2 * px + 1),
            (px, px),
        )
        cap = cv2.erode(cap, kernel, iterations=1)
        if preserve_seed_after_erode:
            cap = np.where((cap > 0) | (seed > 0), 255, 0).astype(np.uint8)
    return cap
