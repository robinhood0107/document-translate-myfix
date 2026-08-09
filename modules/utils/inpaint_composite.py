from __future__ import annotations

import numpy as np


# 블록 하나가 만드는 임시 배열의 목표 크기(바이트). 이미지 크기가 아니라 임시
# 배열 크기에만 상한을 두므로, 해상도가 올라가도 사용량이 늘지 않는다.
_CHUNK_WORKING_BYTES = 8 * 1024 * 1024


def _row_chunks(shape: tuple[int, ...], *, item_bytes: int):
    """행 블록 경계를 만든다. 블록 하나의 임시 배열이 목표 크기를 넘지 않게 한다."""

    height = int(shape[0])
    row_bytes = max(1, int(np.prod(shape[1:], dtype=np.int64)) * int(item_bytes))
    rows = max(1, _CHUNK_WORKING_BYTES // row_bytes)
    for start in range(0, height, rows):
        yield start, min(start + rows, height)


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
    # `np.where(arr > 0, 255, 0)` 는 파이썬 int 리터럴 때문에 전체 크기 int64
    # 배열을 만든다. 4K 페이지에서 그것만 66 MiB 다. 결과는 어차피 uint8 이다.
    binarized = np.zeros(arr.shape, dtype=np.uint8)
    binarized[arr > 0] = 255
    return binarized


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
    # 전체 이미지를 한 번에 int16 으로 올리면 4K 페이지에서 임시 배열만 190 MiB 가
    # 넘는다. 이 값은 진단용 카운트일 뿐이므로 행 블록으로 나눠 센다.
    limit = int(threshold)
    total = 0
    for start, stop in _row_chunks(original.shape, item_bytes=2):
        diff = np.abs(
            edited[start:stop].astype(np.int16) - original[start:stop].astype(np.int16)
        )
        changed = diff > limit
        if original.ndim == 3:
            changed = np.any(changed, axis=2)
        total += int(np.count_nonzero(changed & (mask[start:stop] <= 0)))
    return total
