from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from modules.inpainting.source_lama_blockwise import (
    SourceLaMaLarge,
    SourceLaMaKey,
    _INPAINTER_CACHE,
    _apply_protected_corner_guard,
    _clip_half_open_bbox,
    _resolve_source_blocks,
    _split_bubble_source_mask,
    release_source_lama_cache,
    source_lama_blockwise_inpaint,
    source_lama_blockwise_inpaint_result,
)
from modules.inpainting.runtime_contract import InpaintingCudaOOMError
from modules.utils.image_utils import annotate_block_mask_attribution, generate_mask
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


def test_canonical_result_preserves_legacy_pixels_and_returns_sparse_evidence(
    monkeypatch,
) -> None:
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[12:16, 12:16] = 255
    canonical_block = _text_block(
        xyxy=[8, 8, 20, 20],
        text_class="text_free",
    )
    legacy_block = deepcopy(canonical_block)

    class _FakeSourceLaMa:
        def __init__(self) -> None:
            self.run_diagnostics: list[dict] = []

        def inpaint(
            self,
            source_image,
            source_mask,
            _blocks,
            *,
            check_need_inpaint,
            diagnostic_block_indices,
        ):
            output = source_image.copy()
            output[source_mask > 0] = 99
            self.run_diagnostics.append(
                {
                    "phase": "block",
                    "status": "completed",
                    "is_inference": True,
                    "block_index": diagnostic_block_indices[0],
                }
            )
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )

    canonical = source_lama_blockwise_inpaint_result(
        image,
        mask,
        [canonical_block],
        _CallableInpainter(),
        config=None,
    )
    legacy_image, legacy_mask, legacy_diagnostics = source_lama_blockwise_inpaint(
        image,
        mask,
        [legacy_block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
        return_diagnostics=True,
    )

    np.testing.assert_array_equal(canonical.image, legacy_image)
    np.testing.assert_array_equal(canonical.edit_mask, legacy_mask)
    assert canonical.diagnostics == legacy_diagnostics
    assert len(canonical.evidence) == 1
    item = canonical.evidence[0]
    assert item.block_index == 0
    assert item.source_owned is not None
    assert item.source_owned.pixel_count == 16
    assert item.source_owned.mask.nbytes < mask.nbytes
    assert item.ownership_protect is not None
    assert item.ownership_protect.pixel_count == 16


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


def test_source_lama_blockwise_routes_bubbles_without_calling_lama_fallback(
    monkeypatch,
) -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 20:24] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(xyxy=[18, 18, 28, 28], bubble_xyxy=[8, 8, 40, 40], text_class="text_bubble")
    inpainter = _CallableInpainter()
    interior_cap = np.zeros((32, 32), dtype=np.uint8)
    interior_cap[1:31, 1:31] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

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


def test_source_lama_blockwise_returns_expanded_bubble_edit_mask(
    monkeypatch,
) -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 24:27] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(xyxy=[18, 18, 30, 28], bubble_xyxy=[8, 8, 40, 40], text_class="text_bubble")
    interior_cap = np.zeros((32, 32), dtype=np.uint8)
    interior_cap[2:30, 2:30] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

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


def test_source_lama_blockwise_restores_protected_bubble_corners(
    monkeypatch,
) -> None:
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
    interior_cap = np.zeros((32, 32), dtype=np.uint8)
    interior_cap[2:30, 2:30] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
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
            seen["check_need_inpaint"] = bool(check_need_inpaint)
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
    interior_cap = np.zeros((24, 24), dtype=np.uint8)
    interior_cap[1:23, 1:23] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result = source_lama_blockwise_inpaint(
        image,
        mask,
        (block for block in [bubble_block, text_free_block]),
        _CallableInpainter(),
        config=None,
        check_need_inpaint=False,
    )

    assert int(seen["block_count"][0]) == 1
    assert seen["block_indices"].tolist() == [1]
    assert np.count_nonzero(seen["mask"][14:18, 14:18]) == 0
    assert np.count_nonzero(seen["mask"][34:38, 34:38]) == 16
    assert bool(seen["check_need_inpaint"]) is False
    assert np.all(result[34:38, 34:38] == 77)
    assert bubble_block._erase_mode == "bubble_flat_fill"
    assert int(np.mean(result[14:18, 14:18])) < 180


def test_source_lama_blockwise_delegates_fully_priority_owned_bubble_to_lama(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:32, 28:32] = 255
    image[mask > 0] = 245
    bubble = _text_block(
        xyxy=[24, 24, 36, 36],
        bubble_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[24, 24, 36, 36],
        text_class="text_free",
    )
    seen_masks: list[np.ndarray] = []
    seen_block_counts: list[int] = []
    seen_block_indices: list[list[int | None]] = []

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
            seen_masks.append(source_mask.copy())
            seen_block_counts.append(len(source_blocks))
            seen_block_indices.append(list(diagnostic_block_indices or []))
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [bubble, text_free],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert len(seen_masks) == 1
    np.testing.assert_array_equal(seen_masks[0], mask)
    assert seen_block_counts == [1]
    assert seen_block_indices == [[1]]
    assert np.all(result[mask > 0] == 77)
    assert np.count_nonzero(edit_mask[mask > 0]) == 16
    assert bubble._erase_mode == "bubble_lama_fallback"
    assert bubble._erase_skipped_reason == "lama_priority_owned"


def test_priority_overlap_keeps_anchor_external_bubble_source_on_local_route(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[14:18, 14:18] = 255
    mask[34:38, 34:38] = 255
    mask[66:70, 66:70] = 255
    image[mask > 0] = 245
    first_bubble = _text_block(
        xyxy=[12, 12, 20, 20],
        bubble_xyxy=[4, 4, 44, 44],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[12, 12, 20, 20],
        text_class="text_free",
    )
    second_bubble = _text_block(
        xyxy=[64, 64, 72, 72],
        bubble_xyxy=[52, 52, 92, 92],
        text_class="text_bubble",
    )
    seen_masks: list[np.ndarray] = []
    seen_block_indices: list[list[int | None]] = []

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
            seen_masks.append(source_mask.copy())
            seen_block_indices.append(list(diagnostic_block_indices or []))
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    interior_cap = np.zeros((40, 40), dtype=np.uint8)
    interior_cap[2:38, 2:38] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [first_bubble, text_free, second_bubble],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert len(seen_masks) == 1
    assert seen_block_indices == [[1]]
    assert np.count_nonzero(seen_masks[0][14:18, 14:18]) == 16
    assert np.count_nonzero(seen_masks[0][34:38, 34:38]) == 0
    assert np.all(result[14:18, 14:18] == 77)
    assert np.count_nonzero(edit_mask[34:38, 34:38]) == 16
    assert np.count_nonzero(edit_mask[66:70, 66:70]) == 16
    assert int(np.mean(result[34:38, 34:38])) < 180
    assert int(np.mean(result[66:70, 66:70])) < 180
    assert first_bubble._erase_skipped_reason != "lama_priority_owned"


def test_protected_bubble_candidate_with_no_final_edit_fails_closed(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    bubble_source = np.zeros((96, 96), dtype=np.uint8)
    bubble_source[20:36, 20:44] = 255
    text_free_source = np.zeros((96, 96), dtype=np.uint8)
    text_free_source[28:48, 48:68] = 255
    mask[(bubble_source > 0) | (text_free_source > 0)] = 255
    image[mask > 0] = 167
    image[35:39, 56:60] = 245
    bubble = _text_block(
        xyxy=[16, 16, 72, 54],
        bubble_xyxy=[8, 8, 88, 72],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[44, 24, 72, 52],
        text_class="text_free",
    )
    seen_masks: list[np.ndarray] = []
    seen_block_indices: list[list[int | None]] = []

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
            seen_masks.append(source_mask.copy())
            seen_block_indices.append(list(diagnostic_block_indices or []))
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    interior_cap = np.zeros((64, 80), dtype=np.uint8)
    interior_cap[2:62, 2:78] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [bubble, text_free],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert len(seen_masks) == 1
    assert seen_block_indices == [[1]]
    assert np.count_nonzero(seen_masks[0]) == 400
    assert np.count_nonzero(seen_masks[0][bubble_source > 0]) == 0
    assert np.count_nonzero(edit_mask) == 400
    assert np.count_nonzero(edit_mask[bubble_source > 0]) == 0
    assert np.all(result[bubble_source > 0] == 167)
    assert np.all(result[text_free_source > 0] == 77)
    assert bubble._erase_mode == "bubble_skipped"
    assert bubble._erase_skipped_reason == (
        "bubble_protected_source_seed_unavailable"
    )
    assert bubble._erase_edit_pixel_count == 0


def test_text_free_priority_protects_only_nearby_halo_from_bubble_fill(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[28:32, 24:28] = 255
    mask[28:32, 32:36] = 255
    image[mask > 0] = 245
    halo = np.zeros((64, 64), dtype=np.uint8)
    halo[26:34, 30:38] = 255
    halo[28:32, 32:36] = 0
    image[halo > 0] = 245
    bubble_orphan = np.zeros((64, 64), dtype=np.uint8)
    bubble_orphan[20:24, 36:40] = 255
    image[bubble_orphan > 0] = 245
    bubble = _text_block(
        xyxy=[18, 18, 46, 42],
        bubble_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[30, 18, 48, 44],
        text_class="text_free",
    )

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
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    interior_cap = np.zeros((48, 48), dtype=np.uint8)
    interior_cap[2:46, 2:46] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [bubble, text_free],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(edit_mask[halo > 0]) == 0
    assert np.count_nonzero(changed & (halo > 0)) == 0
    assert np.all(result[28:32, 32:36] == 77)
    assert np.count_nonzero(edit_mask[bubble_orphan > 0]) == 16
    assert np.count_nonzero(changed & (bubble_orphan > 0)) == 16
    assert np.count_nonzero(edit_mask[28:32, 24:28]) == 16


def test_source_lama_blockwise_routes_line_art_bubbles_to_lama_fallback(monkeypatch) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    image[46:49, 8:88] = 20
    image[12:84, 68:71] = 30
    image[30:52, 32:38] = 245
    image[30:34, 32:46] = 245
    image[30:52, 48:54] = 245
    image[30:34, 48:62] = 245
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:52, 32:38] = 255
    mask[30:34, 32:46] = 255
    mask[30:52, 48:54] = 255
    mask[30:34, 48:62] = 255
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
            seen["check_need_inpaint"] = bool(check_need_inpaint)
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
    interior_cap = np.zeros((80, 80), dtype=np.uint8)
    interior_cap[6:74, 6:74] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
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
    assert bool(seen["check_need_inpaint"]) is False
    assert np.count_nonzero(seen["mask"]) > 0
    assert np.count_nonzero(seen["mask"][46:49, 8:24]) == 0
    assert np.count_nonzero(seen["mask"][46:49, 60:88]) == 0
    assert np.all(result[46:49, 8:24] == image[46:49, 8:24])
    assert np.all(result[46:49, 60:88] == image[46:49, 60:88])
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0


def test_source_lama_blockwise_exposes_capless_dense_hard_box_as_required_skip(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[20:36, 20:44] = 255
    image[mask > 0] = 167
    block = _text_block(
        xyxy=[20, 20, 44, 36],
        bubble_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
    )
    inpainter_calls: list[int] = []

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
            inpainter_calls.append(int(np.count_nonzero(source_mask)))
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert inpainter_calls == []
    assert block._erase_mode == "bubble_skipped"
    assert block._erase_skipped_reason == (
        "bubble_interior_cap_source_seed_unavailable"
    )
    assert np.count_nonzero(edit_mask) == 0
    np.testing.assert_array_equal(result, image)


def test_source_lama_blockwise_preserves_attributed_capless_dense_seed(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[34:50, 30:54] = 255
    image[mask > 0] = 167
    block = _text_block(
        xyxy=[30, 34, 54, 50],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
    annotate_block_mask_attribution([block], mask, image.shape)
    block = deepcopy(block)
    seen: list[np.ndarray] = []

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
            seen.append(source_mask.copy())
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert block.mask_actual_pixel_count == np.count_nonzero(mask)
    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], mask)
    np.testing.assert_array_equal(edit_mask, mask)
    assert np.all(result[mask > 0] == 77)


def test_generate_mask_attribution_reaches_capless_source_lama(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    ctd_mask = np.zeros((96, 96), dtype=np.uint8)
    ctd_mask[34:50, 30:54] = 255
    image[ctd_mask > 0] = 167
    block = _text_block(
        xyxy=[30, 34, 54, 50],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )

    class _FakeCTDRefiner:
        def __init__(self, _settings) -> None:
            pass

        def refine(self, _image, _blocks):
            return SimpleNamespace(
                raw_mask=ctd_mask.copy(),
                refined_mask=ctd_mask.copy(),
                final_mask=ctd_mask.copy(),
                backend="torch",
                device="cuda",
                fallback_used=False,
            )

    monkeypatch.setattr(
        "modules.utils.image_utils.CTDRefiner",
        _FakeCTDRefiner,
    )
    monkeypatch.setattr(
        "modules.utils.image_utils.build_protect_mask",
        lambda *_args, **_kwargs: np.zeros(image.shape[:2], dtype=np.uint8),
    )
    details = generate_mask(
        image,
        [block],
        settings={
            "mask_refiner": "ctd",
            "keep_existing_lines": False,
            "final_mask_dilate_size": 0,
        },
        return_details=True,
    )
    generated_mask = details["final_mask"]
    assert block.mask_actual_pixel_count == np.count_nonzero(generated_mask)
    assert block.block_mask_source == "ctd_raw_refined_final_or"
    block = deepcopy(block)
    seen: list[np.ndarray] = []

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
            seen.append(source_mask.copy())
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        generated_mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], generated_mask)
    np.testing.assert_array_equal(edit_mask, generated_mask)
    assert np.all(result[generated_mask > 0] == 77)


def test_attributed_fallback_clips_partial_source_to_semantic_prior(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    for y in range(16, 82, 6):
        for x in range(14, 82, 6):
            image[y:y + 3, x:x + 3] = 85
    source = np.zeros((96, 96), dtype=np.uint8)
    inside_prior = np.zeros_like(source)
    inside_prior[36:48, 34:50] = 255
    outside_prior = np.zeros_like(source)
    outside_prior[70:75, 70:75] = 255
    source[(inside_prior > 0) | (outside_prior > 0)] = 255
    image[inside_prior > 0] = 245
    image[outside_prior > 0] = 245
    block = _text_block(
        xyxy=[30, 34, 54, 50],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
    annotate_block_mask_attribution([block], source, image.shape)
    assert block.mask_actual_pixel_count == np.count_nonzero(source)

    class _FakeSourceLaMa:
        def __init__(self, seen) -> None:
            self.seen = seen

        def inpaint(
            self,
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=True,
            *,
            diagnostic_block_indices=None,
        ):
            self.seen.append(source_mask.copy())
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    interior_cap = np.zeros((80, 80), dtype=np.uint8)
    interior_cap[2:78, 2:78] = 255
    for cap in (None, interior_cap):
        seen: list[np.ndarray] = []
        monkeypatch.setattr(
            "modules.inpainting.source_lama_blockwise.get_source_lama_large",
            lambda **_kwargs: _FakeSourceLaMa(seen),
        )
        monkeypatch.setattr(
            "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
            lambda *_args, _cap=cap, **_kwargs: _cap,
        )
        result, edit_mask = source_lama_blockwise_inpaint(
            image,
            source,
            [deepcopy(block)],
            _CallableInpainter(),
            config=None,
            return_edit_mask=True,
        )

        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], inside_prior)
        np.testing.assert_array_equal(edit_mask, inside_prior)
        assert np.all(result[inside_prior > 0] == 77)
        np.testing.assert_array_equal(
            result[outside_prior > 0],
            image[outside_prior > 0],
        )


def test_source_lama_blockwise_preserves_attributed_dense_seed_in_texture(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    texture = np.zeros((96, 96), dtype=np.uint8)
    for y in range(16, 82, 6):
        for x in range(14, 82, 6):
            texture[y:y + 3, x:x + 3] = 255
    image[texture > 0] = 85
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[36:60, 36:60] = 255
    image[mask > 0] = 245
    texture[mask > 0] = 0
    block = _text_block(
        xyxy=[30, 30, 66, 66],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
    annotate_block_mask_attribution([block], mask, image.shape)
    block = deepcopy(block)
    seen: list[np.ndarray] = []

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
            seen.append(source_mask.copy())
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    interior_cap = np.zeros((80, 80), dtype=np.uint8)
    interior_cap[2:78, 2:78] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert len(seen) == 1
    np.testing.assert_array_equal(seen[0], mask)
    np.testing.assert_array_equal(edit_mask, mask)
    assert np.count_nonzero(edit_mask[texture > 0]) == 0
    np.testing.assert_array_equal(result[texture > 0], image[texture > 0])


def test_attributed_dense_seed_requires_anchor_overlap_for_trust(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[20:36, 20:44] = 255
    image[mask > 0] = 167
    block = _text_block(
        xyxy=[70, 70, 90, 90],
        bubble_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
    )
    annotate_block_mask_attribution([block], mask, image.shape)
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert block.mask_actual_pixel_count > 0
    assert block._erase_mode == "bubble_skipped"
    assert block._erase_skipped_reason == (
        "bubble_interior_cap_source_seed_unavailable"
    )
    assert np.count_nonzero(edit_mask) == 0
    np.testing.assert_array_equal(result, image)


def test_source_lama_blockwise_capless_fallback_excludes_structure(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    structural_bar = np.zeros((96, 96), dtype=np.uint8)
    structural_bar[46:53, 24:72] = 255
    image[structural_bar > 0] = 20
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[38:42, 44:50] = 255
    image[mask > 0] = 245
    block = _text_block(
        xyxy=[20, 28, 76, 62],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
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
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
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
    assert block._erase_skipped_reason == "bubble_interior_cap_unavailable"
    np.testing.assert_array_equal(seen["mask"], mask)
    np.testing.assert_array_equal(edit_mask, mask)
    assert np.all(result[mask > 0] == 77)
    np.testing.assert_array_equal(
        result[structural_bar > 0],
        image[structural_bar > 0],
    )


def test_source_lama_blockwise_capless_fallback_excludes_attached_structure(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    structural_bar = np.zeros((96, 96), dtype=np.uint8)
    structural_bar[46:53, 24:72] = 255
    image[structural_bar > 0] = 20
    attached_glyph = np.zeros((96, 96), dtype=np.uint8)
    attached_glyph[40:46, 44:50] = 255
    image[attached_glyph > 0] = 245
    isolated_glyph = np.zeros((96, 96), dtype=np.uint8)
    isolated_glyph[30:36, 60:66] = 255
    image[isolated_glyph > 0] = 245
    mask = np.where(
        (structural_bar > 0)
        | (attached_glyph > 0)
        | (isolated_glyph > 0),
        255,
        0,
    ).astype(np.uint8)
    block = _text_block(
        xyxy=[20, 28, 76, 62],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
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
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: None,
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
    assert block._erase_skipped_reason == (
        "bubble_interior_cap_source_seed_partially_suppressed"
    )
    assert np.count_nonzero(seen["mask"][structural_bar > 0]) == 0
    assert np.count_nonzero(seen["mask"][attached_glyph > 0]) == 0
    assert np.count_nonzero(seen["mask"][isolated_glyph > 0]) == 36
    np.testing.assert_array_equal(edit_mask, seen["mask"])
    np.testing.assert_array_equal(
        result[structural_bar > 0],
        image[structural_bar > 0],
    )
    np.testing.assert_array_equal(
        result[attached_glyph > 0],
        image[attached_glyph > 0],
    )
    assert np.all(result[isolated_glyph > 0] == 77)


def test_source_lama_blockwise_uses_source_only_fallback_without_text_prior(
    monkeypatch,
) -> None:
    image = np.full((112, 112, 3), 150, dtype=np.uint8)
    visible_line = np.zeros((112, 112), dtype=np.uint8)
    visible_line[55:58, 44:69] = 255
    image[visible_line > 0] = 20
    mask = np.zeros((112, 112), dtype=np.uint8)
    mask[50:62, 53:59] = 255
    image[mask > 0] = 245
    visible_line[mask > 0] = 0
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=[8, 8, 104, 104],
        text_class="text_bubble",
    )
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
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    interior_cap = np.zeros((96, 96), dtype=np.uint8)
    interior_cap[2:94, 2:94] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
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
    assert block._erase_skipped_reason == (
        "text_prior_unavailable_structure_ambiguous"
    )
    np.testing.assert_array_equal(seen["mask"], mask)
    assert np.all(result[mask > 0] == 77)
    np.testing.assert_array_equal(
        result[visible_line > 0],
        image[visible_line > 0],
    )
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
    mask[44:78, 54:62] = 255
    mask[44:52, 54:86] = 255
    mask[58:66, 54:82] = 255
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
    interior_cap = np.where(
        inner[20:100, 20:120],
        255,
        0,
    ).astype(np.uint8)
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert block._erase_mode != "bubble_lama_fallback"
    assert seen == {}
    assert np.array_equal(result[outline], image[outline])
    assert np.mean(result[44:78, 54:86]) > np.mean(image[44:78, 54:86])
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0


def test_source_lama_blockwise_routes_missing_bubble_roi_to_conservative_lama(
    monkeypatch,
) -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 20:24] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(
        xyxy=[18, 18, 28, 28],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    seen: dict[str, object] = {}
    source_lama = _unloaded_inpainter()

    def fake_memory_safe_inpaint(
        source_image,
        source_mask,
        textblock_list=None,
        *,
        diagnostic_context=None,
    ):
        assert textblock_list is None
        assert diagnostic_context == {"phase": "block", "block_index": 0}
        output = source_image.copy()
        output[source_mask > 0] = 77
        return output

    source_lama.memory_safe_inpaint = fake_memory_safe_inpaint
    real_inpaint = SourceLaMaLarge.inpaint.__get__(source_lama, SourceLaMaLarge)

    def recording_inpaint(
        source_image,
        source_mask,
        source_blocks,
        check_need_inpaint=True,
        *,
        diagnostic_block_indices=None,
    ):
        seen["mask"] = source_mask.copy()
        seen["block_count"] = len(source_blocks)
        seen["block_indices"] = list(diagnostic_block_indices or [])
        seen["check_need_inpaint"] = bool(check_need_inpaint)
        return real_inpaint(
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=check_need_inpaint,
            diagnostic_block_indices=diagnostic_block_indices,
        )

    source_lama.inpaint = recording_inpaint

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: source_lama,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        _CallableInpainter(),
        config=None,
        check_need_inpaint=True,
        return_edit_mask=True,
    )

    assert seen["block_count"] == 1
    assert seen["block_indices"] == [0]
    assert seen["check_need_inpaint"] is False
    assert np.count_nonzero(seen["mask"]) == 16
    assert np.all(result[20:24, 20:24] == 77)
    assert np.array_equal(edit_mask, mask)
    assert block._erase_mode == "bubble_lama_fallback"
    assert block._erase_edit_pixel_count == 16
    assert block._erase_skipped_reason == "missing_bubble_roi"


def test_source_lama_blockwise_routes_missing_all_geometry_to_generic_inpainter() -> None:
    image = np.full((48, 48, 3), 128, dtype=np.uint8)
    image[20:24, 20:24] = 245
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    inpainter = _CallableInpainter()

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        inpainter,
        config=None,
        return_edit_mask=True,
    )

    assert inpainter.calls == 1
    assert np.all(result[20:24, 20:24] == 99)
    assert np.array_equal(edit_mask, mask)
    assert block._erase_mode == "bubble_lama_fallback"
    assert block._erase_edit_pixel_count == 0
    assert block._erase_skipped_reason == "missing_bubble_roi"


def test_source_lama_blockwise_routes_unowned_missing_geometry_separately(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[10:14, 10:14] = 245
    image[42:46, 42:46] = 245
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[10:14, 10:14] = 255
    mask[42:46, 42:46] = 255
    missing_geometry = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[8, 8, 16, 16],
        bubble_xyxy=None,
        text_class="text_free",
    )
    seen: dict[str, object] = {}
    source_lama = _unloaded_inpainter()

    def fake_memory_safe_inpaint(
        source_image,
        source_mask,
        textblock_list=None,
        *,
        diagnostic_context=None,
    ):
        assert textblock_list is None
        assert diagnostic_context == {"phase": "block", "block_index": 1}
        output = source_image.copy()
        output[source_mask > 0] = 77
        return output

    source_lama.memory_safe_inpaint = fake_memory_safe_inpaint
    real_inpaint = SourceLaMaLarge.inpaint.__get__(source_lama, SourceLaMaLarge)

    def recording_inpaint(
        source_image,
        source_mask,
        source_blocks,
        check_need_inpaint=True,
        *,
        diagnostic_block_indices=None,
    ):
        seen["mask"] = source_mask.copy()
        seen["block_count"] = len(source_blocks)
        seen["block_indices"] = list(diagnostic_block_indices or [])
        return real_inpaint(
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=check_need_inpaint,
            diagnostic_block_indices=diagnostic_block_indices,
        )

    source_lama.inpaint = recording_inpaint
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: source_lama,
    )
    generic_inpainter = _CallableInpainter()

    result, edit_mask, diagnostics = source_lama_blockwise_inpaint(
        image,
        mask,
        [missing_geometry, text_free],
        generic_inpainter,
        config=None,
        check_need_inpaint=False,
        return_edit_mask=True,
        return_diagnostics=True,
    )

    assert seen["block_count"] == 1
    assert seen["block_indices"] == [1]
    assert np.count_nonzero(seen["mask"]) == 16
    assert generic_inpainter.calls == 1
    assert np.all(result[10:14, 10:14] == 77)
    assert np.all(result[42:46, 42:46] == 99)
    assert np.array_equal(edit_mask, mask)
    changed = np.any(result != image, axis=2)
    assert np.array_equal(changed, mask > 0)
    assert missing_geometry._erase_mode == "bubble_lama_fallback"
    assert missing_geometry._erase_edit_pixel_count == 0
    assert missing_geometry._erase_skipped_reason == "missing_bubble_roi"
    inference_diagnostics = [
        item for item in diagnostics if bool(item.get("is_inference", True))
    ]
    assert len(inference_diagnostics) == 1
    assert inference_diagnostics[0]["phase"] == "generic"
    assert inference_diagnostics[0]["status"] == "completed"
    assert inference_diagnostics[0]["mask_pixel_count"] == 16


def test_unowned_lama_uses_owned_cleanup_as_adjacent_context(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    owned = np.zeros((64, 64), dtype=np.uint8)
    owned[20:24, 20:24] = 255
    unowned = np.zeros((64, 64), dtype=np.uint8)
    unowned[20:24, 24:28] = 255
    source = np.where(
        (owned > 0) | (unowned > 0),
        255,
        0,
    ).astype(np.uint8)
    image[source > 0] = 245
    bubble = _text_block(
        xyxy=[44, 44, 52, 52],
        bubble_xyxy=[40, 40, 56, 56],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[20, 20, 24, 24],
        bubble_xyxy=None,
        text_class="text_free",
    )

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
            output = source_image.copy()
            output[source_mask > 0] = 77
            return output

    class _ContextCheckingInpainter(_CallableInpainter):
        def __call__(self, input_image, input_mask, config):
            self.calls += 1
            assert np.all(input_image[owned > 0] == 77)
            output = input_image.copy()
            output[input_mask > 0] = 99
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: _FakeSourceLaMa(),
    )
    inpainter = _ContextCheckingInpainter()

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        source,
        [bubble, text_free],
        inpainter,
        config=None,
        return_edit_mask=True,
    )

    assert inpainter.calls == 1
    assert np.all(result[owned > 0] == 77)
    assert np.all(result[unowned > 0] == 99)
    np.testing.assert_array_equal(edit_mask, source)


def test_generic_unowned_diagnostic_satisfies_cuda_export_gate(
    monkeypatch,
) -> None:
    class _CudaInpainter:
        name = "lama_large_512px"
        runtime_device = "cuda:1"
        device = "cuda:1"
        precision = "bf16"
        inpaint_size = 1536

        def __init__(self) -> None:
            self.calls = 0
            self.run_diagnostics = [{"phase": "stale", "is_inference": True}]
            parameter = SimpleNamespace(
                device="cuda:1",
                dtype="torch.bfloat16",
            )
            self.model = SimpleNamespace(
                parameters=lambda: iter((parameter,)),
                buffers=lambda: iter(()),
            )

        def __call__(self, source_image, source_mask, config):
            self.calls += 1
            output = source_image.copy()
            output[source_mask > 0] = 99
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.memory_allocated",
        lambda _device: 64 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.memory_reserved",
        lambda _device: 96 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.max_memory_allocated",
        lambda _device: 128 * 1024 * 1024,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.max_memory_reserved",
        lambda _device: 192 * 1024 * 1024,
    )
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[11:15, 13:18] = 255
    image[mask > 0] = 245
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    inpainter = _CudaInpainter()

    result, edit_mask, diagnostics = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        inpainter,
        config=None,
        return_edit_mask=True,
        return_diagnostics=True,
    )

    assert inpainter.calls == 1
    assert np.array_equal(edit_mask, mask)
    assert np.count_nonzero(np.any(result != image, axis=2)) == 20
    assert len(diagnostics) == 1
    diagnostic = diagnostics[0]
    assert diagnostic["phase"] == "generic"
    assert diagnostic["inpainter_key"] == "lama_large_512px"
    assert diagnostic["actual_device"] == "cuda:1"
    assert diagnostic["actual_precision"] == "bf16"
    assert diagnostic["device_verified_from_model"] is True
    assert diagnostic["model_parameter_device"] == "cuda:1"
    assert diagnostic["mask_bbox"] == [13, 11, 18, 15]
    assert diagnostic["mask_pixel_count"] == 20
    assert diagnostic["cuda_memory_allocated_mb"] == 64.0
    assert diagnostic["cuda_memory_reserved_mb"] == 96.0
    assert diagnostic["page_peak_vram_allocated_mb"] == 128.0
    assert diagnostic["page_peak_vram_reserved_mb"] == 192.0
    assert diagnostic["cuda_memory_diagnostics_available"] is True

    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "export_inpaint_debug.py"
    )
    spec = importlib.util.spec_from_file_location(
        "export_inpaint_debug_for_generic_gate_test",
        script_path,
    )
    assert spec is not None and spec.loader is not None
    export_module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = export_module
    spec.loader.exec_module(export_module)
    summary = {
        "inpainter": "lama_large_512px",
        "use_gpu": True,
        "hd_strategy": "Original",
        "inpainter_runtime": diagnostic,
        "cpu_fallback_count": 0,
        "non_cuda_refiner_count": 0,
        "peak_vram_unavailable_count": 0,
        "peak_vram_reset_failure_count": 0,
        "cuda_memory_diagnostics_unavailable_count": 0,
        "zero_block_count": 0,
        "empty_final_mask_count": 0,
        "image_count": 1,
        "success_count": 1,
        "runtime_inference_call_count": 1,
    }
    record = {
        "image": "neutral-page",
        "changed_outside_final_mask_pixel_count_exact": 0,
    }
    assert export_module._required_gate_failures(
        summary,
        {"private": [record]},
        require_cuda_lama=True,
        require_rounded_bubble_gate=False,
        required_image_count=1,
    ) == []


def test_generic_unowned_cuda_oom_retries_once_with_bounded_roi(
    monkeypatch,
) -> None:
    class _RetryingCudaInpainter:
        name = "lama_large_512px"
        runtime_device = "cuda:0"
        device = "cuda:0"
        precision = "bf16"

        def __init__(self) -> None:
            self.calls: list[tuple[tuple[int, ...], np.ndarray]] = []
            parameter = SimpleNamespace(
                device="cuda:0",
                dtype="torch.bfloat16",
            )
            self.model = SimpleNamespace(
                parameters=lambda: iter((parameter,)),
                buffers=lambda: iter(()),
            )

        def __call__(self, source_image, source_mask, _config):
            self.calls.append((source_image.shape, source_mask.copy()))
            if len(self.calls) == 1:
                raise RuntimeError("CUDA out of memory")
            output = source_image.copy()
            output[source_mask > 0] = 99
            return output

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.empty_cache",
        lambda: None,
    )
    for attribute in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            f"modules.inpainting.source_lama_blockwise.torch.cuda.{attribute}",
            lambda _device: 0,
        )
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[42:46, 44:50] = 255
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    inpainter = _RetryingCudaInpainter()

    result, edit_mask, diagnostics = source_lama_blockwise_inpaint(
        image,
        mask,
        [block],
        inpainter,
        config=None,
        return_edit_mask=True,
        return_diagnostics=True,
    )

    assert len(inpainter.calls) == 2
    assert inpainter.calls[1][0][:2] != image.shape[:2]
    assert np.array_equal(edit_mask, mask)
    assert np.all(result[mask > 0] == 99)
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "completed_after_roi_retry"
    assert diagnostics[0]["oom_retry_count"] == 1
    assert diagnostics[0]["oom_retry_roi"] is not None


def test_generic_unowned_cuda_oom_retry_fails_with_typed_error(
    monkeypatch,
) -> None:
    class _FailingCudaInpainter:
        name = "lama_large_512px"
        runtime_device = "cuda:0"
        device = "cuda:0"
        precision = "bf16"

        def __init__(self) -> None:
            self.calls = 0
            parameter = SimpleNamespace(
                device="cuda:0",
                dtype="torch.bfloat16",
            )
            self.model = SimpleNamespace(
                parameters=lambda: iter((parameter,)),
                buffers=lambda: iter(()),
            )

        def __call__(self, _source_image, _source_mask, _config):
            self.calls += 1
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.empty_cache",
        lambda: None,
    )
    for attribute in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            f"modules.inpainting.source_lama_blockwise.torch.cuda.{attribute}",
            lambda _device: 0,
        )
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[42:46, 44:50] = 255
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    inpainter = _FailingCudaInpainter()

    with np.testing.assert_raises(InpaintingCudaOOMError) as captured:
        source_lama_blockwise_inpaint(
            image,
            mask,
            [block],
            inpainter,
            config=None,
            return_diagnostics=True,
        )

    assert inpainter.calls == 2
    assert captured.exception.diagnostics["status"] == (
        "failed_after_roi_retry"
    )
    assert captured.exception.diagnostics["oom_retry_count"] == 1


def test_generic_unowned_cuda_oom_retry_preserves_non_oom_failure_diagnostics(
    monkeypatch,
) -> None:
    class _MixedFailureCudaInpainter:
        name = "lama_large_512px"
        runtime_device = "cuda:0"
        device = "cuda:0"
        precision = "bf16"

        def __init__(self) -> None:
            self.calls = 0
            parameter = SimpleNamespace(
                device="cuda:0",
                dtype="torch.bfloat16",
            )
            self.model = SimpleNamespace(
                parameters=lambda: iter((parameter,)),
                buffers=lambda: iter(()),
            )

        def __call__(self, _source_image, _source_mask, _config):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA out of memory")
            raise ValueError("invalid retry input")

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.is_available",
        lambda: True,
    )
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.torch.cuda.empty_cache",
        lambda: None,
    )
    for attribute in (
        "memory_allocated",
        "memory_reserved",
        "max_memory_allocated",
        "max_memory_reserved",
    ):
        monkeypatch.setattr(
            f"modules.inpainting.source_lama_blockwise.torch.cuda.{attribute}",
            lambda _device: 0,
        )
    image = np.full((96, 96, 3), 128, dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[42:46, 44:50] = 255
    block = _text_block(
        xyxy=[0, 0, 0, 0],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    inpainter = _MixedFailureCudaInpainter()

    with np.testing.assert_raises(InpaintingCudaOOMError) as captured:
        source_lama_blockwise_inpaint(
            image,
            mask,
            [block],
            inpainter,
            config=None,
            return_diagnostics=True,
        )

    assert inpainter.calls == 2
    assert isinstance(captured.exception.__cause__, ValueError)
    diagnostics = captured.exception.diagnostics
    assert diagnostics["status"] == "failed_during_roi_retry"
    assert diagnostics["oom_retry_count"] == 1
    assert diagnostics["retry_error"] == "ValueError"


def test_source_lama_blockwise_routes_mixed_missing_bubble_roi_to_conservative_lama(
    monkeypatch,
) -> None:
    image = np.full((64, 64, 3), 128, dtype=np.uint8)
    image[16:20, 16:20] = 245
    image[20:24, 20:24] = 245
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[16:20, 16:20] = 255
    mask[20:24, 20:24] = 255
    valid_bubble = _text_block(
        xyxy=[14, 14, 22, 22],
        bubble_xyxy=[6, 6, 30, 30],
        text_class="text_bubble",
    )
    missing_roi = _text_block(
        xyxy=[20, 20, 28, 28],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    seen: dict[str, object] = {}
    source_lama = _unloaded_inpainter()

    def fake_memory_safe_inpaint(
        source_image,
        source_mask,
        textblock_list=None,
        *,
        diagnostic_context=None,
    ):
        assert textblock_list is None
        assert diagnostic_context == {"phase": "block", "block_index": 1}
        output = source_image.copy()
        output[source_mask > 0] = 77
        return output

    source_lama.memory_safe_inpaint = fake_memory_safe_inpaint
    real_inpaint = SourceLaMaLarge.inpaint.__get__(source_lama, SourceLaMaLarge)

    def recording_inpaint(
        source_image,
        source_mask,
        source_blocks,
        check_need_inpaint=True,
        *,
        diagnostic_block_indices=None,
    ):
        seen["mask"] = source_mask.copy()
        seen["block_count"] = len(source_blocks)
        seen["block_indices"] = list(diagnostic_block_indices or [])
        seen["check_need_inpaint"] = bool(check_need_inpaint)
        return real_inpaint(
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=check_need_inpaint,
            diagnostic_block_indices=diagnostic_block_indices,
        )

    source_lama.inpaint = recording_inpaint

    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: source_lama,
    )
    interior_cap = np.zeros((24, 24), dtype=np.uint8)
    interior_cap[1:23, 1:23] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result = source_lama_blockwise_inpaint(
        image,
        mask,
        [valid_bubble, missing_roi],
        _CallableInpainter(),
        config=None,
    )

    assert seen["block_count"] == 1
    assert seen["block_indices"] == [1]
    assert seen["check_need_inpaint"] is False
    assert np.count_nonzero(seen["mask"][16:20, 16:20]) == 0
    assert np.count_nonzero(seen["mask"][20:24, 20:24]) == 16
    assert np.all(result[20:24, 20:24] == 77)
    assert missing_roi._erase_mode == "bubble_lama_fallback"
    assert missing_roi._erase_edit_pixel_count == 16
    assert missing_roi._erase_skipped_reason == "missing_bubble_roi"


def test_missing_bubble_result_is_protected_from_neighbor_fallback(
    monkeypatch,
) -> None:
    image = np.full((96, 96, 3), 150, dtype=np.uint8)
    image[46:49, 8:88] = 20
    image[30:52, 32:38] = 245
    image[30:34, 32:46] = 245
    image[30:52, 50:56] = 245
    image[30:34, 50:64] = 245
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[30:52, 32:38] = 255
    mask[30:34, 32:46] = 255
    mask[30:52, 50:56] = 255
    mask[30:34, 50:64] = 255
    valid_bubble = _text_block(
        xyxy=[26, 24, 64, 58],
        bubble_xyxy=[8, 8, 88, 88],
        text_class="text_bubble",
    )
    missing_roi = _text_block(
        xyxy=[48, 28, 66, 54],
        bubble_xyxy=None,
        text_class="text_bubble",
    )

    class _FakeSourceLaMa:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []
            self.run_diagnostics: list[dict] = []

        def inpaint(
            self,
            source_image,
            source_mask,
            source_blocks,
            check_need_inpaint=True,
            *,
            diagnostic_block_indices=None,
        ):
            call_index = len(self.calls)
            self.calls.append(
                {
                    "mask": source_mask.copy(),
                    "block_count": len(source_blocks),
                    "check_need_inpaint": bool(check_need_inpaint),
                    "block_indices": list(diagnostic_block_indices or []),
                }
            )
            output = source_image.copy()
            output[source_mask > 0] = 77 if call_index == 0 else 66
            return output

    source_lama = _FakeSourceLaMa()
    monkeypatch.setattr(
        "modules.inpainting.source_lama_blockwise.get_source_lama_large",
        lambda **_kwargs: source_lama,
    )
    interior_cap = np.zeros((80, 80), dtype=np.uint8)
    interior_cap[6:74, 6:74] = 255
    monkeypatch.setattr(
        "modules.utils.bubble_erase.extract_bubble_interior_cap_crop",
        lambda *_args, **_kwargs: interior_cap,
    )

    result, edit_mask = source_lama_blockwise_inpaint(
        image,
        mask,
        [valid_bubble, missing_roi],
        _CallableInpainter(),
        config=None,
        return_edit_mask=True,
    )

    assert valid_bubble._erase_mode == "bubble_lama_fallback"
    assert len(source_lama.calls) == 2
    missing_call, fallback_call = source_lama.calls
    assert missing_call["block_indices"] == [1]
    assert missing_call["check_need_inpaint"] is False
    assert fallback_call["block_indices"] == [0]
    assert fallback_call["check_need_inpaint"] is False
    assert np.count_nonzero(missing_call["mask"][30:52, 50:56]) > 0
    assert np.count_nonzero(fallback_call["mask"][30:52, 50:56]) == 0
    assert np.all(result[30:52, 50:56] == 77)
    assert np.all(result[30:52, 32:38] == 66)
    changed = np.any(result != image, axis=2)
    assert np.count_nonzero(changed & (edit_mask <= 0)) == 0


def test_split_bubble_mask_keeps_connected_valid_glyph_out_of_missing_roi_route() -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[14:18, 14:18] = 255
    mask[18:22, 18:22] = 255
    valid_bubble = _text_block(
        xyxy=[12, 12, 20, 20],
        bubble_xyxy=[8, 8, 30, 30],
        text_class="text_bubble",
    )
    missing_roi = _text_block(
        xyxy=[18, 18, 24, 24],
        bubble_xyxy=None,
        text_class="text_bubble",
    )

    (
        bubble_mask,
        bubble_blocks,
        lama_blocks,
        priority_mask,
        missing_mask,
        missing_blocks,
        bubble_protected_mask,
    ) = _split_bubble_source_mask(
        mask,
        [valid_bubble, missing_roi],
        (48, 48, 3),
    )

    assert bubble_blocks == [valid_bubble]
    assert lama_blocks == []
    assert missing_blocks == [missing_roi]
    assert np.count_nonzero(bubble_mask[14:18, 14:18]) == 16
    assert np.count_nonzero(bubble_mask[18:22, 18:22]) == 0
    assert np.count_nonzero(priority_mask[14:18, 14:18]) == 0
    assert np.count_nonzero(priority_mask[18:22, 18:22]) == 16
    assert np.array_equal(priority_mask, missing_mask)
    assert np.array_equal(bubble_protected_mask, priority_mask)


def test_split_missing_bubble_does_not_take_text_free_priority_pixels() -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[18:22, 18:22] = 255
    missing_roi = _text_block(
        xyxy=[16, 16, 24, 24],
        bubble_xyxy=None,
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[16, 16, 24, 24],
        text_class="text_free",
    )

    (
        bubble_mask,
        bubble_blocks,
        lama_blocks,
        priority_mask,
        missing_mask,
        missing_blocks,
        bubble_protected_mask,
    ) = _split_bubble_source_mask(
        mask,
        [missing_roi, text_free],
        (48, 48, 3),
    )

    assert bubble_blocks == []
    assert lama_blocks == [text_free]
    assert missing_blocks == [missing_roi]
    assert np.count_nonzero(bubble_mask) == 0
    np.testing.assert_array_equal(priority_mask, mask)
    assert np.count_nonzero(missing_mask) == 0
    assert np.count_nonzero(bubble_protected_mask[16:24, 16:24]) > 16
    assert missing_roi._erase_edit_pixel_count == 0


def test_split_text_free_priority_uses_clipped_two_pixel_bubble_protection() -> None:
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[12:16, 12:16] = 255
    mask[24:28, 24:28] = 255
    bubble = _text_block(
        xyxy=[10, 10, 32, 32],
        bubble_xyxy=[8, 8, 40, 40],
        text_class="text_bubble",
    )
    text_free = _text_block(
        xyxy=[20, 20, 36, 36],
        text_class="text_free",
    )

    (
        bubble_mask,
        _bubble_blocks,
        _lama_blocks,
        priority_mask,
        _missing_mask,
        _missing_blocks,
        bubble_protected_mask,
    ) = _split_bubble_source_mask(
        mask,
        [bubble, text_free],
        (48, 48, 3),
    )

    assert np.count_nonzero(priority_mask) == 16
    assert np.count_nonzero(bubble_protected_mask) > 16
    assert np.count_nonzero(bubble_protected_mask[:20]) == 0
    assert np.count_nonzero(bubble_protected_mask[:, :20]) == 0
    assert bubble_protected_mask[24, 22] == 255
    assert bubble_protected_mask[24, 29] == 255
    assert bubble_protected_mask[24, 21] == 0
    assert bubble_protected_mask[24, 30] == 0
    assert bubble_protected_mask[34, 34] == 0
    assert np.count_nonzero(bubble_mask[12:16, 12:16]) == 16
    assert np.count_nonzero(bubble_mask[24:28, 24:28]) == 0

    edge_mask = np.zeros((48, 48), dtype=np.uint8)
    edge_mask[20:24, 20:24] = 255
    edge_text_free = _text_block(
        xyxy=[20, 16, 36, 32],
        text_class="text_free",
    )
    (
        _edge_bubble_mask,
        _edge_bubble_blocks,
        _edge_lama_blocks,
        _edge_priority_mask,
        _edge_missing_mask,
        _edge_missing_blocks,
        edge_protected_mask,
    ) = _split_bubble_source_mask(
        edge_mask,
        [bubble, edge_text_free],
        (48, 48, 3),
    )
    assert edge_protected_mask[21, 19] == 0
    assert edge_protected_mask[21, 25] == 255
    assert edge_protected_mask[21, 26] == 0
