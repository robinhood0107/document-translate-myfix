from __future__ import annotations

import tracemalloc
import unittest

import numpy as np

from modules.inpainting.base import blend_with_mask
from modules.utils.inpaint_composite import (
    composite_with_edit_mask,
    count_changed_outside_edit_mask,
)


# 실제로 죽었던 페이지 크기. uint8 로 23.7 MiB, float64 로 189.8 MiB 다.
PAGE_SHAPE = (2160, 3840, 3)
PAGE_BYTES = PAGE_SHAPE[0] * PAGE_SHAPE[1] * PAGE_SHAPE[2]


MIB = 1024 * 1024


def _peak_bytes(call) -> tuple[object, int]:
    """``call()`` 이 새로 잡은 메모리의 정점. 입력은 호출 밖에서 미리 만들 것."""

    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        value = call()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return value, peak


class BlendWithMaskTests(unittest.TestCase):
    def test_blend_matches_the_float_reference(self) -> None:
        rng = np.random.default_rng(20260807)
        result = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        image = rng.integers(0, 256, size=(37, 53, 3), dtype=np.uint8)
        mask = rng.integers(0, 256, size=(37, 53), dtype=np.uint8)

        alpha = mask[:, :, np.newaxis] / 255.0
        expected = np.rint(result * alpha + image * (1.0 - alpha)).astype(np.uint8)

        np.testing.assert_array_equal(blend_with_mask(result, image, mask), expected)

    def test_a_hard_mask_selects_each_source_exactly(self) -> None:
        result = np.full((8, 8, 3), 200, dtype=np.uint8)
        image = np.full((8, 8, 3), 50, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[:4] = 255

        blended = blend_with_mask(result, image, mask)

        np.testing.assert_array_equal(blended[:4], result[:4])
        np.testing.assert_array_equal(blended[4:], image[4:])

    def test_the_blend_returns_uint8_not_float64(self) -> None:
        # float64 를 돌려주면 하류 전체가 8배로 부풀어 오른다.
        blended = blend_with_mask(
            np.zeros((16, 16, 3), np.uint8),
            np.zeros((16, 16, 3), np.uint8),
            np.zeros((16, 16), np.uint8),
        )
        self.assertEqual(blended.dtype, np.uint8)

    def test_a_full_page_blend_stays_far_below_the_old_float64_peak(self) -> None:
        result = np.zeros(PAGE_SHAPE, dtype=np.uint8)
        image = np.zeros(PAGE_SHAPE, dtype=np.uint8)
        mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)

        blended, peak = _peak_bytes(lambda: blend_with_mask(result, image, mask))

        self.assertEqual(blended.shape, PAGE_SHAPE)
        # 실측: 예전 구현 569.5 MiB -> 현재 53.1 MiB. 출력 하나(23.7 MiB)에 고정
        # 크기 작업 버퍼(29.3 MiB)를 더한 값이다.
        self.assertLess(peak, PAGE_BYTES * 4)

    def test_working_memory_does_not_grow_with_image_height(self) -> None:
        # 임시 배열 크기가 이미지 크기와 무관해야 한다. 그래야 어떤 해상도에서도
        # 같은 메모리로 돈다.
        def working_bytes(height: int) -> int:
            shape = (height, 2048, 3)
            # 입력 할당이 측정에 섞이지 않도록 추적 밖에서 만든다.
            result = np.zeros(shape, np.uint8)
            image = np.zeros(shape, np.uint8)
            mask = np.zeros(shape[:2], np.uint8)
            _value, peak = _peak_bytes(lambda: blend_with_mask(result, image, mask))
            # 반환 버퍼는 이미지에 비례할 수밖에 없다. 작업 버퍼만 본다.
            return peak - height * 2048 * 3

        small = working_bytes(1024)
        large = working_bytes(8192)
        # 실측으로는 높이 1024 부터 16384 까지 정확히 29.3 MiB 로 평평하다.
        self.assertLess(large, small + 4 * MIB)


class CompositeMemoryTests(unittest.TestCase):
    def test_changed_pixel_count_matches_the_unchunked_reference(self) -> None:
        rng = np.random.default_rng(20260807)
        original = rng.integers(0, 256, size=(129, 71, 3), dtype=np.uint8)
        edited = original.copy()
        edited[10:20, 5:15] = 0
        edited[100:110, 30:40] = 255
        mask = np.zeros((129, 71), dtype=np.uint8)
        mask[10:20, 5:15] = 255

        changed = np.any(
            np.abs(edited.astype(np.int16) - original.astype(np.int16)) > 0,
            axis=2,
        )
        expected = int(np.count_nonzero(changed & (mask <= 0)))

        self.assertEqual(
            count_changed_outside_edit_mask(original, edited, mask),
            expected,
        )

    def test_a_full_page_count_stays_bounded(self) -> None:
        original = np.zeros(PAGE_SHAPE, dtype=np.uint8)
        edited = np.zeros(PAGE_SHAPE, dtype=np.uint8)
        mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)

        _value, peak = _peak_bytes(
            lambda: count_changed_outside_edit_mask(original, edited, mask)
        )

        # 실측: 예전 구현 150.3 MiB -> 현재 41.2 MiB.
        self.assertLess(peak, PAGE_BYTES * 2)

    def test_composite_still_selects_by_mask(self) -> None:
        original = np.full((6, 6, 3), 10, dtype=np.uint8)
        edited = np.full((6, 6, 3), 200, dtype=np.uint8)
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[:3] = 255

        composed = composite_with_edit_mask(original, edited, mask)

        np.testing.assert_array_equal(composed[:3], edited[:3])
        np.testing.assert_array_equal(composed[3:], original[3:])
