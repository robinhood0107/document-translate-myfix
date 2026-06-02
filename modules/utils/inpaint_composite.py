from __future__ import annotations

import numpy as np


def normalize_edit_mask(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
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


def composite_with_edit_mask(
    original_image: np.ndarray | None,
    edited_image: np.ndarray | None,
    edit_mask: np.ndarray | None,
) -> np.ndarray | None:
    if original_image is None or edited_image is None:
        return edited_image
    original = np.asarray(original_image)
    edited = np.asarray(edited_image)
    if original.shape != edited.shape:
        return edited_image

    mask = normalize_edit_mask(edit_mask, original.shape)
    if edited.ndim == 3:
        mask = mask[:, :, None]
    return np.where(mask > 0, edited, original).astype(edited.dtype, copy=False)


def count_changed_outside_edit_mask(
    original_image: np.ndarray | None,
    edited_image: np.ndarray | None,
    edit_mask: np.ndarray | None,
    *,
    threshold: int = 0,
) -> int:
    if original_image is None or edited_image is None:
        return 0
    original = np.asarray(original_image)
    edited = np.asarray(edited_image)
    if original.shape != edited.shape:
        return 0
    mask = normalize_edit_mask(edit_mask, original.shape)
    if original.ndim == 3:
        changed = np.any(np.abs(edited.astype(np.int16) - original.astype(np.int16)) > int(threshold), axis=2)
    else:
        changed = np.abs(edited.astype(np.int16) - original.astype(np.int16)) > int(threshold)
    return int(np.count_nonzero(changed & (mask <= 0)))
