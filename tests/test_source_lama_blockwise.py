from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import SimpleNamespace

import numpy as np

from modules.inpainting.source_lama_blockwise import (
    SourceLaMaLarge,
    SourceLaMaKey,
    _INPAINTER_CACHE,
    _apply_protected_corner_guard,
    _clip_half_open_bbox,
    _resolve_source_blocks,
    release_source_lama_cache,
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

    def _fake_inpaint(
        image_crop,
        mask_crop,
        textblock_list=None,
        *,
        diagnostic_context=None,
    ):
        seen_shapes.append((image_crop.shape, mask_crop.shape))
        assert textblock_list is None
        assert diagnostic_context == {"phase": "block", "block_index": 7}
        return np.full_like(image_crop, 77)

    inpainter.memory_safe_inpaint = _fake_inpaint
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.full((10, 10), 255, dtype=np.uint8)

    result = inpainter.inpaint(
        image,
        mask,
        [_Block([-3, 2, 4, 5])],
        diagnostic_block_indices=[7],
    )

    assert seen_shapes
    assert seen_shapes[0][0][0] > 0
    assert seen_shapes[0][0][1] > 0
    assert np.count_nonzero(result[:, :5]) > 0
    assert np.count_nonzero(result[:, -1]) == 0


def test_source_block_indices_fail_closed_when_explicit_mapping_is_short() -> None:
    blocks = [_Block([1, 1, 4, 4]), _Block([5, 5, 8, 8])]

    _resolved, explicit_indices = _resolve_source_blocks(blocks, [7])
    _resolved, implicit_indices = _resolve_source_blocks(blocks)

    assert explicit_indices == [7, None]
    assert implicit_indices == [0, 1]


def test_release_source_lama_cache_drops_only_cached_model_references() -> None:
    release_source_lama_cache()
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.device = "cuda"
    inpainter.precision = "bf16"
    inpainter.inpaint_size = 1536
    native_model = object()
    inpainter.model = native_model
    key = SourceLaMaKey("cuda", "bf16", 1536)
    _INPAINTER_CACHE[key] = inpainter

    report = release_source_lama_cache()

    assert report == {
        "cache_entry_count": 1,
        "loaded_model_count": 1,
        "gpu_loaded_model_count": 1,
        "expected_process_reclaim_mb": 0.0,
        "untracked_gpu_resource_count": 1,
        "gpu_release_expected": True,
    }
    assert _INPAINTER_CACHE == {}
    assert inpainter.model is None


def test_memory_safe_inpaint_records_block_runtime_diagnostics(monkeypatch) -> None:
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.device = "cuda"
    inpainter.precision = "bf16"
    inpainter.inpaint_size = 64
    inpainter.run_diagnostics = []
    inpainter.ensure_loaded = lambda: None
    inpainter._inpaint = lambda image, _mask, _blocks=None: image.copy()
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.inspect_learned_inpainter_runtime",
        lambda *_args, **_kwargs: {
            "actual_device": "cuda",
            "actual_precision": "bf16",
            "cpu_fallback_used": False,
        },
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: False,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)

    result = inpainter.memory_safe_inpaint(
        image,
        mask,
        diagnostic_context={"phase": "block", "block_index": 3},
    )

    assert np.array_equal(result, image)
    assert len(inpainter.run_diagnostics) == 1
    diagnostics = inpainter.run_diagnostics[0]
    assert diagnostics["phase"] == "block"
    assert diagnostics["block_index"] == 3
    assert diagnostics["elapsed_seconds"] >= 0.0
    assert diagnostics["status"] == "completed"
    assert diagnostics["cuda_memory_diagnostics_unavailable"] is True


def test_memory_safe_inpaint_records_available_cuda_memory_diagnostics(
    monkeypatch,
) -> None:
    inpainter = object.__new__(SourceLaMaLarge)
    inpainter.device = "cuda:1"
    inpainter.precision = "bf16"
    inpainter.inpaint_size = 64
    inpainter.run_diagnostics = []
    inpainter.ensure_loaded = lambda: None
    inpainter._inpaint = lambda image, _mask, _blocks=None: image.copy()
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.inspect_learned_inpainter_runtime",
        lambda *_args, **_kwargs: {
            "actual_device": "cuda:1",
            "actual_precision": "bf16",
            "cpu_fallback_used": False,
        },
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.device",
        lambda value: f"device:{value}",
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.memory_allocated",
        lambda device: 64 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.memory_reserved",
        lambda device: 96 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.max_memory_allocated",
        lambda device: 128 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.max_memory_reserved",
        lambda device: 192 * 1024 * 1024,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    mask = np.full((8, 8), 255, dtype=np.uint8)

    result = inpainter.memory_safe_inpaint(
        image,
        mask,
        diagnostic_context={"phase": "block", "block_index": 3},
    )

    assert np.array_equal(result, image)
    diagnostics = inpainter.run_diagnostics[0]
    assert diagnostics["cuda_memory_diagnostics_available"] is True
    assert "cuda_memory_diagnostics_unavailable" not in diagnostics
    assert diagnostics["cuda_memory_allocated_mb"] == 64.0
    assert diagnostics["cuda_memory_reserved_mb"] == 96.0
    assert diagnostics["page_peak_vram_allocated_mb"] == 128.0
    assert diagnostics["page_peak_vram_reserved_mb"] == 192.0


def test_blockwise_diagnostic_context_never_reenters_with_crop_block(
    monkeypatch,
) -> None:
    inpainter = _unloaded_inpainter()
    inpainter.device = "cuda"
    inpainter.precision = "bf16"
    inpainter.inpaint_size = 64
    inpainter.run_diagnostics = []
    seen_textblock_lists: list[object] = []

    def _fake_inpaint(image_crop, _mask_crop, textblock_list=None):
        seen_textblock_lists.append(textblock_list)
        return np.full_like(image_crop, 77)

    inpainter._inpaint = _fake_inpaint
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.inspect_learned_inpainter_runtime",
        lambda *_args, **_kwargs: {
            "actual_device": "cuda",
            "actual_precision": "bf16",
            "cpu_fallback_used": False,
        },
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: False,
    )
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[8:16, 8:16] = 255

    result = inpainter.inpaint(image, mask, [_Block([8, 8, 16, 16])])

    assert seen_textblock_lists == [None]
    assert np.all(result[mask > 0] == 77)
    assert np.array_equal(result[mask <= 0], image[mask <= 0])


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
    assert (
        sha256(np.ascontiguousarray(result).tobytes()).hexdigest()
        == "6fa8dacc9c1c6683973f57ab97f412a2fe0bc1c286bd143b5abafc995a2d96f1"
    )


def test_missing_bubble_stat_index_is_not_misattributed_to_first_block(
    monkeypatch,
) -> None:
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:16, 12:16] = 255
    block = _text_block(
        xyxy=[10, 10, 18, 18],
        bubble_xyxy=[4, 4, 28, 28],
        text_class="text_bubble",
    )

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.erase_text_bubble_regions",
        lambda *_args, **_kwargs: SimpleNamespace(
            image=image.copy(),
            edit_mask=mask.copy(),
            fallback_mask=np.zeros_like(mask),
            stats={
                "blocks": [
                    {
                        "index": None,
                        "mode": "bubble_flat_fill",
                        "elapsed_seconds": 0.01,
                    }
                ]
            },
        ),
    )

    _result, diagnostics = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_diagnostics=True,
    )

    assert diagnostics[0]["phase"] == "bubble_erase"
    assert diagnostics[0]["block_index"] is None


def test_source_lama_blockwise_returns_expanded_bubble_edit_mask() -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 24:27] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(xyxy=[18, 18, 30, 28], bubble_xyxy=[8, 8, 40, 40], text_class="text_bubble")

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert np.count_nonzero(edit_mask) > np.count_nonzero(mask)
    assert np.count_nonzero(edit_mask[20:24, 24:27]) > 0
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0


def test_source_lama_blockwise_restores_protected_bubble_corners() -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 24:27] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    protected = np.zeros((48, 48), dtype=np.uint8)
    protected[20:24, 24:27] = 255
    block = _text_block(
        xyxy=[18, 18, 30, 28],
        bubble_xyxy=[8, 8, 40, 40],
        text_class="text_bubble",
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
        protected_corner_mask=protected,
    )

    assert np.count_nonzero(edit_mask[protected > 0]) == 0
    assert np.array_equal(result[protected > 0], image[protected > 0])
    assert np.count_nonzero(edit_mask[mask > 0]) > 0


def test_empty_protected_corner_guard_preserves_the_existing_result_and_mask() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    result = np.full((8, 8, 3), 77, dtype=np.uint8)

    guarded, edit_mask = _apply_protected_corner_guard(
        original,
        result,
        None,
        None,
    )

    assert guarded is result
    assert edit_mask is None


def test_protected_corner_guard_with_unknown_edit_mask_restores_only_protection() -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    result = np.full((8, 8, 3), 77, dtype=np.uint8)
    protected = np.zeros((8, 8), dtype=np.uint8)
    protected[2:4, 3:5] = 255

    guarded, edit_mask = _apply_protected_corner_guard(
        original,
        result,
        None,
        protected,
    )

    assert edit_mask is None
    assert np.array_equal(guarded[protected > 0], original[protected > 0])
    assert np.all(guarded[protected <= 0] == 77)


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
        def inpaint(
            self,
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=True,
            *,
            diagnostic_block_indices=None,
        ):
            seen["mask"] = source_mask.copy()
            seen["block_count"] = np.asarray([len(source_blocks)], dtype=np.int32)
            seen["block_indices"] = np.asarray(
                diagnostic_block_indices,
                dtype=np.int32,
            )
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
        (block for block in [bubble_block, text_free_block]),
        _CallableInpainter(),
        config=None,
    )

    assert int(seen["block_count"][0]) == 1
    assert seen["block_indices"].tolist() == [1]
    assert np.count_nonzero(seen["mask"][14:18, 14:18]) == 0
    assert np.count_nonzero(seen["mask"][34:38, 34:38]) == 16
    assert np.all(result[34:38, 34:38] == 77)
    assert bubble_block._erase_mode == "bubble_flat_fill"
    assert int(np.mean(result[14:18, 14:18])) < 180


def test_source_lama_blockwise_routes_line_art_bubbles_to_lama_fallback(monkeypatch) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    image[46:49, 8:88] = 20
    image[12:84, 68:71] = 30
    image[30:52, 32:38] = 245
    image[30:52, 48:54] = 245
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:52, 32:38] = 255
    mask[30:52, 48:54] = 255
    block = _text_block(xyxy=[26, 24, 60, 58], bubble_xyxy=[8, 8, 88, 88], text_class="text_bubble")
    seen: dict[str, np.ndarray] = {}

    class _FakeSourceLaMa:
        def inpaint(
            self,
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=True,
            *,
            diagnostic_block_indices=None,
        ):
            seen["mask"] = source_mask.copy()
            seen["block_count"] = np.asarray([len(source_blocks)], dtype=np.int32)
            seen["block_indices"] = np.asarray(
                diagnostic_block_indices,
                dtype=np.int32,
            )
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert block._erase_mode == "bubble_lama_fallback"
    assert int(seen["block_count"][0]) == 1
    assert seen["block_indices"].tolist() == [0]
    assert np.count_nonzero(seen["mask"]) > 0
    assert np.count_nonzero(seen["mask"][46:49, 8:24]) == 0
    assert np.count_nonzero(seen["mask"][46:49, 60:88]) == 0
    assert np.all(result[46:49, 8:24] == image[46:49, 8:24])
    assert np.all(result[46:49, 60:88] == image[46:49, 60:88])
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0


def test_source_lama_blockwise_preserves_bubble_outline_while_clearing_interior(monkeypatch) -> None:
    image = np.full((120, 140, 3), 180, dtype=np.uint8)
    yy, xx = np.ogrid[:120, :140]
    oval = (((xx - 70) / 48.0) ** 2 + ((yy - 60) / 36.0) ** 2) <= 1.0
    inner = (((xx - 70) / 44.0) ** 2 + ((yy - 60) / 32.0) ** 2) <= 1.0
    outline = oval & ~inner
    image[oval] = 245
    image[outline] = 20
    image[44:78, 54:62] = 12
    image[44:52, 54:86] = 12
    image[58:66, 54:82] = 248
    mask = np.zeros((120, 140), dtype=np.uint8)
    mask[40:84, 50:90] = 255
    block = _text_block(xyxy=[48, 36, 94, 88], bubble_xyxy=[20, 20, 120, 100], text_class="text_bubble")
    seen: dict[str, np.ndarray] = {}

    class _FakeSourceLaMa:
        def inpaint(
            self,
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=True,
            *,
            diagnostic_block_indices=None,
        ):
            seen["mask"] = source_mask.copy()
            seen["block_count"] = np.asarray([len(source_blocks)], dtype=np.int32)
            seen["block_indices"] = np.asarray(
                diagnostic_block_indices,
                dtype=np.int32,
            )
            output = source_image.copy()
            output[source_mask > 0] = 245
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert block._erase_mode == "bubble_lama_fallback"
    assert int(seen["block_count"][0]) == 1
    assert seen["block_indices"].tolist() == [0]
    assert np.count_nonzero(seen["mask"][outline]) == 0
    assert np.array_equal(result[outline], image[outline])
    assert np.mean(result[44:78, 54:86]) > np.mean(image[44:78, 54:86])
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0
