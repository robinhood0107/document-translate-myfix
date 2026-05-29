from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modules.inpainting.source_lama_blockwise import (
    SourceLaMaLarge,
    _clip_half_open_bbox,
    source_lama_blockwise_inpaint,
)
from modules.utils.textblock import TextBlock


@dataclass
class _Block:
    xyxy: list[int]


class _CallableInpainter:
    runtime_device = "cpu"
    precision = "fp32"
    inpaint_size = 64

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, image, mask, config):
        self.calls += 1
        output = image.copy()
        output[mask > 0] = 99
        return output


def _text_block(*, xyxy, text_class, bubble_xyxy=None) -> TextBlock:
    return TextBlock(
        text_bbox=np.asarray(xyxy, dtype=np.int32),
        bubble_bbox=np.asarray(bubble_xyxy, dtype=np.int32) if bubble_xyxy is not None else None,
        text_class=text_class,
    )


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


def test_source_lama_blockwise_routes_bubbles_without_calling_lama_fallback() -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 20:24] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(xyxy=[18, 18, 28, 28], bubble_xyxy=[8, 8, 40, 40], text_class="text_bubble")
    inpainter = _CallableInpainter()

    result = source_lama_blockwise_inpaint(image, mask, [block], inpainter, config=None)

    assert inpainter.calls == 0
    assert block._erase_mode == "bubble_flat_fill"
    assert np.all(result[0, 0] == image[0, 0])
    assert int(np.mean(result[20:24, 20:24])) < 180


def test_source_lama_blockwise_keeps_text_free_on_lama_path(monkeypatch) -> None:
    image = np.full((56, 56, 3), 128, dtype=np.uint8)
    image[14:18, 14:18] = 245
    mask = np.zeros((56, 56), dtype=np.uint8)
    mask[14:18, 14:18] = 255
    mask[34:38, 34:38] = 255
    bubble_block = _text_block(xyxy=[12, 12, 22, 22], bubble_xyxy=[6, 6, 30, 30], text_class="text_bubble")
    text_free_block = _text_block(xyxy=[32, 32, 42, 42], text_class="text_free")
    seen: dict[str, np.ndarray] = {}

    class _FakeSourceLaMa:
        def inpaint(self, source_image, source_mask, source_blocks, check_need_inpaint=True):
            seen["mask"] = source_mask.copy()
            seen["block_count"] = np.asarray([len(source_blocks)], dtype=np.int32)
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )

    result = source_lama_blockwise_inpaint(
        image,
        mask,
        [bubble_block, text_free_block],
        _CallableInpainter(),
        config=None,
    )

    assert int(seen["block_count"][0]) == 1
    assert np.count_nonzero(seen["mask"][14:18, 14:18]) == 0
    assert np.count_nonzero(seen["mask"][34:38, 34:38]) == 16
    assert np.all(result[34:38, 34:38] == 77)
    assert bubble_block._erase_mode == "bubble_flat_fill"
    assert int(np.mean(result[14:18, 14:18])) < 180
