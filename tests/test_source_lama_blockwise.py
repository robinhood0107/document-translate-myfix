from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.inpainting.source_lama_blockwise import (
    SourceLaMaLarge,
    _clip_half_open_bbox,
)


@dataclass
class _Block:
    xyxy: list[int]


def _unloaded_inpainter() -> SourceLaMaLarge:
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.inpaint_by_block = True
    inpainter.check_need_inpaint = False
    inpainter.ensure_loaded = lambda: None
    return inpainter


def test_clip_half_open_bbox_clips_to_image_and_skips_empty() -> None:
    assert _clip_half_open_bbox([-3, 2, 4, 5], 10, 10) == [0, 2, 4, 5]
    assert _clip_half_open_bbox([8, 8, 12, 13], 10, 10) == [8, 8, 10, 10]
    assert _clip_half_open_bbox([-5, 2, -1, 5], 10, 10) is None
    assert _clip_half_open_bbox([5, 5, 5, 8], 10, 10) is None


def test_blockwise_inpaint_skips_out_of_bounds_bbox_without_touching_right_edge() -> None:
    inpainter = _unloaded_inpainter()
    inpainter.memory_safe_inpaint = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        AssertionError("out-of-image block should be skipped")
    )
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    result = inpainter.inpaint(image, mask, [_Block([-5, 2, -1, 5])])

    assert np.array_equal(result, image)


def test_blockwise_inpaint_uses_clipped_bbox_for_partial_negative_block() -> None:
    inpainter = _unloaded_inpainter()
    seen_shapes: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def _fake_inpaint(image_crop, mask_crop, textblock_list=None):
        seen_shapes.append((image_crop.shape, mask_crop.shape))
        return np.full_like(image_crop, 77)

    inpainter.memory_safe_inpaint = _fake_inpaint
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    result = inpainter.inpaint(image, mask, [_Block([-3, 2, 4, 5])])

    assert seen_shapes
    assert seen_shapes[0][0][0] > 0
    assert seen_shapes[0][0][1] > 0
    assert np.count_nonzero(result[:, :5]) > 0
    assert np.count_nonzero(result[:, -1]) == 0
