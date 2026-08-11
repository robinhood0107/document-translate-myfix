from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import imkit as imk
from modules.detection.utils.content import (
    _process_stats_vectorized,
    detect_content_in_bbox,
)
from modules.utils.bubble_erase import (
    ERASE_MODE_BUBBLE_LAMA_FALLBACK,
    fill_bubble_edit_mask,
)
from modules.utils.inpaint_composite import count_changed_outside_edit_mask
from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint


def test_small_cjk_detector_includes_four_pixel_crop_edge_component() -> None:
    stats = np.zeros((2, 5), dtype=np.int32)
    stats[1, imk.CC_STAT_LEFT] = 0
    stats[1, imk.CC_STAT_TOP] = 0
    stats[1, imk.CC_STAT_WIDTH] = 2
    stats[1, imk.CC_STAT_HEIGHT] = 2
    stats[1, imk.CC_STAT_AREA] = 4

    result = _process_stats_vectorized(
        stats,
        (16, 16),
        min_area=4,
        margin=0,
        inclusive_min_area=True,
    )

    np.testing.assert_array_equal(result, np.asarray([[0, 0, 2, 2]]))


def test_small_cjk_detector_public_flag_preserves_strict_default() -> None:
    image = np.full((16, 16, 3), 100, dtype=np.uint8)
    image[0:2, 0:2] = 0
    image[10:14, 10:14] = 255

    default_boxes = detect_content_in_bbox(image, min_area=4, margin=0)
    strict_boxes = detect_content_in_bbox(
        image,
        min_area=4,
        margin=0,
        inclusive_min_area=False,
    )
    inclusive_boxes = detect_content_in_bbox(
        image,
        min_area=4,
        margin=0,
        inclusive_min_area=True,
    )

    np.testing.assert_array_equal(default_boxes, strict_boxes)
    assert [0, 0, 3, 3] not in strict_boxes.tolist()
    assert [0, 0, 3, 3] in inclusive_boxes.tolist()


def test_residue_cleanup_processes_all_blocks_past_the_legacy_page_cap(
    monkeypatch,
) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    blocks = [
        SimpleNamespace(
            xyxy=[0, 0, 64, 64],
            bubble_xyxy=[0, 0, 64, 64],
            cleanup_roi_xyxy=[0, 0, 64, 64],
            text_class="text_bubble",
        )
        for _ in range(5)
    ]
    boxes = [
        (x, y, x + 2, y + 2)
        for y in range(0, 20, 3)
        for x in range(0, 15, 3)
    ][:30]
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: list(boxes),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((64, 64), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _edit_mask, **_kwargs: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        (block for block in blocks),
        None,
        None,
    )

    assert stats["component_count"] == 150
    assert stats["residue_pass_truncated_block_count"] == 0
    assert stats["residue_pass_cap_dropped_candidate_count"] == 0
    assert stats["pass2_backend"] == "test_fill"


def test_residue_cleanup_reports_the_per_block_component_cap(monkeypatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.full((64, 64), 255, dtype=np.uint8)
    block = SimpleNamespace(
        xyxy=[0, 0, 64, 64],
        bubble_xyxy=[0, 0, 64, 64],
        cleanup_roi_xyxy=[0, 0, 64, 64],
        text_class="text_bubble",
    )
    boxes = [
        (x, y, x + 2, y + 2)
        for y in range(0, 24, 3)
        for x in range(0, 15, 3)
    ][:40]
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: iter(boxes),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((64, 64), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _edit_mask, **_kwargs: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        (item for item in [block]),
        None,
        None,
    )

    assert stats["component_count"] == 35
    assert stats["residue_pass_truncated_block_count"] == 1
    assert stats["residue_pass_cap_dropped_candidate_count"] == 5


def test_per_block_cap_does_not_stop_the_next_block(monkeypatch) -> None:
    image = np.zeros((96, 96, 3), dtype=np.uint8)
    mask = np.full((96, 96), 255, dtype=np.uint8)
    blocks = [
        SimpleNamespace(
            xyxy=[0, 0, 48, 96],
            bubble_xyxy=[0, 0, 48, 96],
            cleanup_roi_xyxy=[0, 0, 48, 96],
            text_class="text_bubble",
        ),
        SimpleNamespace(
            xyxy=[48, 0, 96, 96],
            bubble_xyxy=[48, 0, 96, 96],
            cleanup_roi_xyxy=[48, 0, 96, 96],
            text_class="text_bubble",
        ),
    ]
    calls = 0
    first_boxes = [
        (x, y, x + 2, y + 2)
        for y in range(0, 32, 4)
        for x in range(0, 20, 4)
    ][:40]
    second_boxes = [(x, 60, x + 2, 62) for x in range(0, 40, 4)][:10]

    def detect(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        return first_boxes if calls == 1 else second_boxes

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        detect,
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((96, 48), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _mask, **_kwargs: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        blocks,
        None,
        None,
    )

    assert stats["component_count"] == 45
    assert stats["residue_pass_cap_dropped_candidate_count"] == 5
    assert stats["block_count"] == 2


def test_residue_cleanup_keeps_small_crop_edge_cjk_contract(monkeypatch) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[4:8, 4:8] = 255
    block = SimpleNamespace(
        xyxy=[4, 4, 8, 8],
        bubble_xyxy=[4, 4, 28, 28],
        cleanup_roi_xyxy=[4, 4, 28, 28],
        text_class="text_bubble",
    )
    seen_detection_kwargs: list[dict] = []

    def detect(_crop, **kwargs):
        seen_detection_kwargs.append(dict(kwargs))
        return [(0, 0, 2, 2)]

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        detect,
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((24, 24), 255, dtype=np.uint8),
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        lambda source, _edit_mask, **_kwargs: (source.copy(), "test_fill"),
    )

    _cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [block],
        None,
        None,
    )

    assert seen_detection_kwargs == [
        {
            "min_area": 4,
            "margin": 0,
            "inclusive_min_area": True,
        }
    ]
    assert stats["component_count"] == 1


def test_residue_cleanup_fills_each_block_with_its_bubble_roi(monkeypatch) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    mask[40:44, 40:44] = 255
    blocks = [
        SimpleNamespace(
            xyxy=[8, 8, 12, 12],
            bubble_xyxy=[4, 4, 28, 28],
            cleanup_roi_xyxy=[4, 4, 28, 28],
            text_class="text_bubble",
        ),
        SimpleNamespace(
            xyxy=[40, 40, 44, 44],
            bubble_xyxy=[36, 36, 64, 64],
            cleanup_roi_xyxy=[36, 36, 64, 64],
            text_class="text_bubble",
        ),
    ]
    seen_rois: list[tuple[int, int, int, int] | None] = []
    fill_values = iter((80, 160))

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(4, 4, 8, 8)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((24, 24), 255, dtype=np.uint8),
    )

    def fill(
        source,
        _edit_mask,
        *,
        bubble_roi=None,
        background_exclude_mask=None,
    ):
        assert background_exclude_mask is not None
        seen_rois.append(bubble_roi)
        filled = source.copy()
        filled[_edit_mask > 0] = next(fill_values)
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        blocks,
        None,
        None,
    )

    assert seen_rois == [(0, 0, 24, 24), (0, 0, 28, 28)]
    assert np.all(cleaned[8:12, 8:12] == 80)
    assert np.all(cleaned[40:44, 40:44] == 160)
    assert stats["block_count"] == 2
    assert stats["pass2_backend_distribution"] == {"test_fill": 2}
    assert stats["pass2_applied_block_count"] == 2
    assert stats["pass2_fallback_block_count"] == 0


def test_residue_cleanup_same_roi_uses_prior_fill_as_context(monkeypatch) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[16:20, 16:20] = 255
    mask[28:32, 28:32] = 255
    blocks = [
        SimpleNamespace(
            xyxy=[16, 16, 20, 20],
            bubble_xyxy=[8, 8, 40, 40],
            cleanup_roi_xyxy=[8, 8, 40, 40],
            text_class="text_bubble",
            tag=0,
        ),
        SimpleNamespace(
            xyxy=[28, 28, 32, 32],
            bubble_xyxy=[8, 8, 40, 40],
            cleanup_roi_xyxy=[8, 8, 40, 40],
            text_class="text_bubble",
            tag=1,
        ),
    ]
    seen_sources: list[np.ndarray] = []

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(8, 8, 12, 12), (20, 20, 24, 24)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )

    def prior(_image, block, _roi, **_kwargs):
        result = np.zeros((32, 32), dtype=np.uint8)
        y1 = 8 if block.tag == 0 else 20
        result[y1 : y1 + 4, y1 : y1 + 4] = 255
        return result

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        prior,
    )

    def fill(source, edit_mask, **_kwargs):
        seen_sources.append(source.copy())
        filled = source.copy()
        filled[edit_mask > 0] = 80 if len(seen_sources) == 1 else 160
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        blocks,
        None,
        None,
    )

    assert len(seen_sources) == 2
    assert np.all(seen_sources[1][8:12, 8:12] == 80)
    assert np.all(cleaned[16:20, 16:20] == 80)
    assert np.all(cleaned[28:32, 28:32] == 160)
    assert stats["pass2_applied_block_count"] == 2


def test_fill_bubble_edit_mask_forwards_the_actual_roi() -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    edit_mask = np.zeros((64, 64), dtype=np.uint8)
    edit_mask[16:48, 16:48] = 255

    filled, backend = fill_bubble_edit_mask(
        image,
        edit_mask,
        bubble_roi=(16, 16, 48, 48),
    )

    assert backend == ERASE_MODE_BUBBLE_LAMA_FALLBACK
    np.testing.assert_array_equal(filled, image)


def test_fill_bubble_edit_mask_forwards_background_exclusion() -> None:
    image = np.full((32, 32, 3), 150, dtype=np.uint8)
    edit_mask = np.zeros((32, 32), dtype=np.uint8)
    edit_mask[8:24, 8:24] = 255

    filled, backend = fill_bubble_edit_mask(
        image,
        edit_mask,
        bubble_roi=(0, 0, 32, 32),
    )
    excluded_filled, excluded_backend = fill_bubble_edit_mask(
        image,
        edit_mask,
        bubble_roi=(0, 0, 32, 32),
        background_exclude_mask=np.full((32, 32), 255, dtype=np.uint8),
    )

    assert backend != ERASE_MODE_BUBBLE_LAMA_FALLBACK
    assert excluded_backend == ERASE_MODE_BUBBLE_LAMA_FALLBACK
    np.testing.assert_array_equal(excluded_filled, image)


def test_residue_cleanup_reports_partial_fallback_and_preserves_protected(
    monkeypatch,
) -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:12, 8:12] = 255
    mask[40:44, 40:44] = 255
    protected = np.zeros_like(mask)
    protected[40, 40] = 255
    blocks = [
        SimpleNamespace(
            xyxy=[8, 8, 12, 12],
            bubble_xyxy=[4, 4, 28, 28],
            cleanup_roi_xyxy=[4, 4, 28, 28],
            text_class="text_bubble",
        ),
        SimpleNamespace(
            xyxy=[40, 40, 44, 44],
            bubble_xyxy=[36, 36, 64, 64],
            cleanup_roi_xyxy=[36, 36, 64, 64],
            text_class="text_bubble",
        ),
    ]
    calls = 0

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(4, 4, 8, 8)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda _image, _block, roi, **_kwargs: np.full(
            (roi[3] - roi[1], roi[2] - roi[0]),
            255,
            dtype=np.uint8,
        ),
    )

    def fill(source, edit_mask, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return source.copy(), ERASE_MODE_BUBBLE_LAMA_FALLBACK
        filled = source.copy()
        filled[edit_mask > 0] = 170
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        blocks,
        None,
        None,
        protected_corner_mask=protected,
    )

    assert calls == 2
    assert stats["pass2_backend"] == "mixed"
    assert stats["pass2_backend_distribution"] == {
        ERASE_MODE_BUBBLE_LAMA_FALLBACK: 1,
        "test_fill": 1,
    }
    assert stats["pass2_applied_block_count"] == 1
    assert stats["pass2_fallback_block_count"] == 1
    assert stats["pass2_applied_pixel_count"] == int(
        np.count_nonzero(stats["residue_mask"])
    )
    assert stats["residue_mask_pre_cap_pixel_count"] >= stats[
        "residue_mask_cap_pixel_count"
    ]
    assert stats["residue_mask_cap_pixel_count"] > stats[
        "pass2_applied_pixel_count"
    ]
    assert int(np.count_nonzero(stats["residue_mask"][8:12, 8:12])) == 0
    assert np.all(cleaned[8:12, 8:12] == 0)
    assert np.all(cleaned[41:44, 41:44] == 170)
    assert np.all(cleaned[protected > 0] == image[protected > 0])
    assert int(np.count_nonzero(final_mask[protected > 0])) == 0
    assert count_changed_outside_edit_mask(
        image,
        cleaned,
        stats["residue_mask"],
    ) == 0


def test_residue_cleanup_reports_pre_cap_pixels_and_excludes_protected_samples(
    monkeypatch,
) -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[16, 16] = 255
    protected = np.zeros_like(mask)
    protected[14, 14] = 255
    block = SimpleNamespace(
        xyxy=[15, 15, 18, 18],
        bubble_xyxy=[8, 8, 24, 24],
        cleanup_roi_xyxy=[8, 8, 24, 24],
        text_class="text_bubble",
    )
    seen_exclusion: list[np.ndarray] = []

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(5, 5, 12, 12)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((16, 16), 255, dtype=np.uint8),
    )

    def fill(source, edit_mask, *, background_exclude_mask=None, **_kwargs):
        assert background_exclude_mask is not None
        seen_exclusion.append(background_exclude_mask.copy())
        filled = source.copy()
        filled[edit_mask > 0] = 170
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [block],
        None,
        None,
        protected_corner_mask=protected,
    )

    assert stats["residue_mask_pre_cap_pixel_count"] > stats[
        "residue_mask_cap_pixel_count"
    ]
    assert stats["residue_mask_cap_pixel_count"] == int(
        np.count_nonzero(stats["residue_mask"])
    )
    assert len(seen_exclusion) == 1
    assert seen_exclusion[0][14 - 8, 14 - 8] == 255
    assert seen_exclusion[0][16 - 8, 16 - 8] == 255
    assert int(np.count_nonzero(final_mask[protected > 0])) == 0
    np.testing.assert_array_equal(cleaned[protected > 0], image[protected > 0])


def test_residue_cleanup_does_not_reopen_pr2_structure_guard(monkeypatch) -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    image[30:32, 20:36] = 20
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[27:30, 27:30] = 255
    block = SimpleNamespace(
        xyxy=[26, 26, 31, 31],
        bubble_xyxy=[8, 8, 56, 56],
        cleanup_roi_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
        _erase_mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        _erase_skipped_reason="",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("PR2 structure-guarded block must not enter cleanup")

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        fail_if_called,
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fail_if_called,
    )

    for reason in (
        "line_art_intrusion",
        "bubble_interior_cap_unavailable",
        "lama_priority_owned",
        "missing_bubble_roi",
        "line_art_source_seed_unavailable",
        "microtexture_source_seed_partially_suppressed",
    ):
        block._erase_skipped_reason = reason
        cleaned, final_mask, stats = refine_bubble_residue_inpaint(
            image,
            mask,
            [block],
            None,
            None,
        )

        assert stats["residue_pass_structure_guard_block_count"] == 1
        assert stats["applied"] is False
        np.testing.assert_array_equal(cleaned, image)
        np.testing.assert_array_equal(final_mask, mask)


def test_residue_cleanup_preserves_partial_text_free_priority(monkeypatch) -> None:
    image = np.full((48, 48, 3), 150, dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    bubble_source = np.zeros_like(mask)
    bubble_source[14:18, 14:18] = 255
    text_free_source = np.zeros_like(mask)
    text_free_source[28:32, 28:32] = 255
    mask[(bubble_source > 0) | (text_free_source > 0)] = 255
    image[mask > 0] = 245
    bubble = SimpleNamespace(
        xyxy=[12, 12, 20, 20],
        bubble_xyxy=[8, 8, 40, 40],
        cleanup_roi_xyxy=[8, 8, 40, 40],
        text_class="text_bubble",
        _erase_mode="bubble_flat_fill",
        _erase_skipped_reason="",
    )
    text_free = SimpleNamespace(
        xyxy=[26, 26, 34, 34],
        text_class="text_free",
    )

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(6, 6, 10, 10), (20, 20, 24, 24)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((32, 32), 255, dtype=np.uint8),
    )

    def fill(source, edit_mask, **_kwargs):
        filled = source.copy()
        filled[edit_mask > 0] = 80
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [bubble, text_free],
        None,
        None,
    )

    assert stats["pass2_text_free_candidate_count"] == 0
    assert np.count_nonzero(stats["residue_mask"][bubble_source > 0]) == 16
    assert np.count_nonzero(stats["residue_mask"][text_free_source > 0]) == 0
    assert np.all(cleaned[bubble_source > 0] == 80)
    assert np.all(cleaned[text_free_source > 0] == 245)


def test_residue_cleanup_guarded_neighbor_cannot_reown_structure(monkeypatch) -> None:
    image = np.full((64, 64, 3), 150, dtype=np.uint8)
    line = np.zeros((64, 64), dtype=np.uint8)
    line[30:32, 24:40] = 255
    image[line > 0] = 20
    mask = np.zeros((64, 64), dtype=np.uint8)
    guarded_source = np.zeros_like(mask)
    guarded_source[27:29, 27:37] = 255
    normal_source = np.zeros_like(mask)
    normal_source[25:28, 42:45] = 255
    mask[(guarded_source > 0) | (normal_source > 0)] = 255
    guarded = SimpleNamespace(
        xyxy=[24, 24, 40, 35],
        bubble_xyxy=[8, 8, 56, 56],
        cleanup_roi_xyxy=[8, 8, 56, 56],
        text_class="text_bubble",
        _erase_mode=ERASE_MODE_BUBBLE_LAMA_FALLBACK,
        _erase_skipped_reason="line_art_intrusion",
    )
    normal = SimpleNamespace(
        xyxy=[40, 23, 47, 31],
        bubble_xyxy=[10, 8, 58, 56],
        cleanup_roi_xyxy=[10, 8, 58, 56],
        text_class="text_bubble",
        _erase_mode="bubble_flat_fill",
        _erase_skipped_reason="",
    )

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(14, 22, 30, 24), (32, 17, 35, 20)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((48, 48), 255, dtype=np.uint8),
    )

    def fill(source, edit_mask, **_kwargs):
        filled = source.copy()
        filled[edit_mask > 0] = 80
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [guarded, normal],
        None,
        None,
    )

    assert stats["residue_pass_structure_guard_block_count"] == 1
    assert np.count_nonzero(stats["residue_mask"][line > 0]) == 0
    np.testing.assert_array_equal(cleaned[line > 0], image[line > 0])
    assert np.count_nonzero(stats["residue_mask"][normal_source > 0]) == 0
    np.testing.assert_array_equal(cleaned[normal_source > 0], image[normal_source > 0])


def test_residue_cleanup_overlapping_blocks_use_first_success_once(
    monkeypatch,
) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = SimpleNamespace(
        xyxy=[20, 20, 24, 24],
        bubble_xyxy=[8, 8, 40, 40],
        cleanup_roi_xyxy=[8, 8, 40, 40],
        text_class="text_bubble",
    )
    calls = 0

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(12, 12, 16, 16)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((32, 32), 255, dtype=np.uint8),
    )

    def fill(source, edit_mask, **_kwargs):
        nonlocal calls
        calls += 1
        filled = source.copy()
        filled[edit_mask > 0] = 90 + calls
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [block, block],
        None,
        None,
    )

    assert calls == 1
    assert np.all(cleaned[20:24, 20:24] == 91)
    assert stats["pass2_applied_block_count"] == 1


def test_residue_cleanup_overlapping_fallback_can_be_retried(monkeypatch) -> None:
    image = np.zeros((48, 48, 3), dtype=np.uint8)
    mask = np.zeros((48, 48), dtype=np.uint8)
    mask[20:24, 20:24] = 255
    block = SimpleNamespace(
        xyxy=[20, 20, 24, 24],
        bubble_xyxy=[8, 8, 40, 40],
        cleanup_roi_xyxy=[8, 8, 40, 40],
        text_class="text_bubble",
    )
    calls = 0

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.detect_content_in_bbox",
        lambda *_args, **_kwargs: [(12, 12, 16, 16)],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup._build_bubble_faint_boxes",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.build_text_prior_mask",
        lambda *_args, **_kwargs: np.full((32, 32), 255, dtype=np.uint8),
    )

    def fill(source, edit_mask, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return source.copy(), ERASE_MODE_BUBBLE_LAMA_FALLBACK
        filled = source.copy()
        filled[edit_mask > 0] = 90
        return filled, "test_fill"

    monkeypatch.setattr(
        "modules.utils.inpaint_cleanup.fill_bubble_edit_mask",
        fill,
    )

    cleaned, _final_mask, stats = refine_bubble_residue_inpaint(
        image,
        mask,
        [block, block],
        None,
        None,
    )

    assert calls == 2
    assert np.all(cleaned[20:24, 20:24] == 90)
    assert stats["pass2_fallback_block_count"] == 1
    assert stats["pass2_applied_block_count"] == 1
