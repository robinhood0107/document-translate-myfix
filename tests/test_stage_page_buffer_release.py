from __future__ import annotations

import unittest

import numpy as np

from pipeline.stage_batched_processor import (
    StagePageContext,
    _approximate_buffer_bytes,
)


PAGE_SHAPE = (2160, 3840, 3)
PAGE_BYTES = PAGE_SHAPE[0] * PAGE_SHAPE[1] * PAGE_SHAPE[2]
MIB = 1024 * 1024


def _loaded_page() -> StagePageContext:
    """스테이지 배치 파이프라인이 렌더 직전에 들고 있는 상태 그대로."""

    ctx = StagePageContext(
        image_path="page.png",
        image_name="page.png",
        source_lang="Japanese",
        target_lang="Korean",
    )
    ctx.image = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    ctx.inpaint_input_img = np.zeros(PAGE_SHAPE, dtype=np.uint8)
    ctx.raw_mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)
    ctx.mask = np.zeros(PAGE_SHAPE[:2], dtype=np.uint8)
    ctx.mask[:100, :100] = 255
    ctx.mask_details = {"raw_mask": ctx.raw_mask, "final_mask": ctx.mask}
    ctx.patches = [{"bbox": (0, 0, 64, 64), "cleaned": np.zeros((64, 64, 3), np.uint8)}]
    return ctx


class PageBufferReleaseTests(unittest.TestCase):
    def test_release_frees_every_full_resolution_array(self) -> None:
        ctx = _loaded_page()

        ctx.release_page_buffers()

        self.assertIsNone(ctx.image)
        self.assertIsNone(ctx.inpaint_input_img)
        self.assertIsNone(ctx.raw_mask)
        self.assertIsNone(ctx.mask)
        self.assertEqual(ctx.patches, [])
        # mask_details 는 같은 마스크를 다시 참조한다. 비우지 않으면 위에서
        # 놓아준 배열이 그대로 살아남는다.
        self.assertEqual(ctx.mask_details, {})

    def test_release_reports_the_bytes_it_freed(self) -> None:
        ctx = _loaded_page()

        released = ctx.release_page_buffers()

        # 원본 + 인페인팅 결과(각 23.7 MiB)에 마스크 두 장(각 7.9 MiB)만 해도
        # 페이지당 60 MiB 를 넘는다. 이게 배치 내내 쌓이던 값이다.
        self.assertGreater(released, 60 * MIB)
        self.assertEqual(ctx.released_buffer_bytes, released)

    def test_mask_pixel_count_survives_the_release(self) -> None:
        # 스윕이 끝난 뒤의 집계 때문에 이미지를 붙들고 있을 이유는 없다.
        ctx = _loaded_page()
        expected = int(np.count_nonzero(ctx.mask))

        ctx.release_page_buffers()

        self.assertEqual(expected, 100 * 100)
        self.assertEqual(ctx.mask_pixel_count, expected)

    def test_releasing_twice_is_harmless(self) -> None:
        ctx = _loaded_page()

        first = ctx.release_page_buffers()
        second = ctx.release_page_buffers()

        self.assertGreater(first, 0)
        self.assertEqual(second, 0)
        self.assertEqual(ctx.released_buffer_bytes, first)

    def test_release_on_a_page_that_never_loaded_is_harmless(self) -> None:
        ctx = StagePageContext(
            image_path="page.png",
            image_name="page.png",
            source_lang="Japanese",
            target_lang="Korean",
        )

        self.assertEqual(ctx.release_page_buffers(), 0)
        self.assertEqual(ctx.mask_pixel_count, 0)


class BufferSizeEstimateTests(unittest.TestCase):
    def test_arrays_nested_in_containers_are_counted(self) -> None:
        array = np.zeros((100, 100), dtype=np.uint8)
        self.assertEqual(
            _approximate_buffer_bytes({"a": [array], "b": (array,)}),
            2 * array.nbytes,
        )

    def test_a_cycle_terminates(self) -> None:
        payload: dict[str, object] = {"array": np.zeros((10, 10), np.uint8)}
        payload["self"] = payload

        self.assertEqual(_approximate_buffer_bytes(payload), 100)

    def test_objects_holding_arrays_as_attributes_are_counted(self) -> None:
        class CheckpointHit:
            def __init__(self) -> None:
                self.cleaned_image = np.zeros((50, 50, 3), np.uint8)

        hit = CheckpointHit()
        self.assertEqual(_approximate_buffer_bytes(hit), hit.cleaned_image.nbytes)

    def test_plain_scalars_contribute_nothing(self) -> None:
        self.assertEqual(_approximate_buffer_bytes({"n": 5, "s": "x" * 1000}), 0)
