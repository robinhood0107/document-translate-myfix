from __future__ import annotations

import cv2
import numpy as np

from modules.source_parity_vendor.utils.textblock_mask import extract_ballon_mask

from .contracts import binary_mask


def _ballons_masks(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    image = np.asarray(image_bgr)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Ballons silhouette expects a BGR image")
    seed = binary_mask(seed_mask, image.shape[:2])
    if not np.any(seed):
        return None, None
    image_rgb = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2RGB)
    detected, non_text = extract_ballon_mask(image_rgb, seed)
    if detected is None or np.asarray(detected).size == 0:
        return None, None
    cap = binary_mask(np.asarray(detected), seed.shape)
    background = (
        None
        if non_text is None or np.asarray(non_text).size == 0
        else binary_mask(np.asarray(non_text), seed.shape)
    )
    return cap, background


def extract_ballons_native_interior(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
) -> np.ndarray | None:
    cap, _background = _ballons_masks(image_bgr, seed_mask)
    return cap


def extract_pr2_validated_interior(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
    *,
    min_seed_coverage: float = 0.98,
    min_area_ratio: float = 0.20,
    max_area_ratio: float = 0.995,
) -> np.ndarray | None:
    seed = binary_mask(seed_mask, np.asarray(image_bgr).shape[:2])
    seed_count = int(np.count_nonzero(seed))
    if seed_count <= 0:
        return None
    cap, _background = _ballons_masks(image_bgr, seed)
    if cap is None:
        return None
    area_ratio = float(np.count_nonzero(cap)) / float(max(1, cap.size))
    seed_coverage = float(np.count_nonzero((cap > 0) & (seed > 0))) / float(
        seed_count
    )
    if not (
        float(min_area_ratio) <= area_ratio <= float(max_area_ratio)
        and seed_coverage >= float(min_seed_coverage)
    ):
        return None
    cap = cv2.erode(cap, np.ones((3, 3), np.uint8), iterations=1)
    post_coverage = float(np.count_nonzero((cap > 0) & (seed > 0))) / float(
        seed_count
    )
    if post_coverage < float(min_seed_coverage) or not np.any(cap):
        return None
    return binary_mask(cap, seed.shape)


def ballons_native_clean_background(
    image_bgr: np.ndarray,
    seed_mask: np.ndarray,
) -> bool:
    _cap, background = _ballons_masks(image_bgr, seed_mask)
    if background is None or not np.any(background):
        return False
    pixels = np.asarray(image_bgr)[background > 0, :3]
    if pixels.size == 0:
        return False
    average = np.median(pixels, axis=0)
    std_rgb = np.std(pixels.astype(np.float32) - average, axis=0)
    threshold = 7.0 if float(np.std(std_rgb)) > 1.0 else 10.0
    return float(np.max(std_rgb)) < threshold
