from __future__ import annotations

import hashlib

import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.incremental_plan import (
    INCREMENTAL_INPAINT_MODES,
    IncrementalInpaintPlan,
    PageIncrementalInpaintPlan,
    composite_incremental_result,
    execute_incremental_inpaint,
    union_incremental_plans,
)


def _mask(shape: tuple[int, int], ys: slice, xs: slice) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    result[ys, xs] = 255
    return result


def _plan(
    mode: str,
    *,
    generation: np.ndarray,
    commit: np.ndarray,
    existing: np.ndarray,
) -> IncrementalInpaintPlan:
    return IncrementalInpaintPlan(
        mode=mode,  # type: ignore[arg-type]
        generation_mask=generation,
        commit_mask=commit,
        existing_source_edit=existing,
    )


def _execute(
    source: np.ndarray,
    page,
    generated: np.ndarray | None,
):
    calls: list[np.ndarray] = []

    def callback(actual_source: np.ndarray, mask: np.ndarray) -> np.ndarray:
        calls.append(mask.copy())
        assert generated is not None
        return generated

    result, receipt = execute_incremental_inpaint(
        source,
        page,
        callback,
        provider="synthetic-lama",
    )
    return result, receipt, calls


@pytest.mark.parametrize("mode", sorted(INCREMENTAL_INPAINT_MODES))
def test_incremental_plan_accepts_every_v34_mode(mode: str) -> None:
    shape = (12, 18)
    existing = _mask(shape, slice(2, 5), slice(2, 6))
    commit = (
        _mask(shape, slice(3, 5), slice(4, 7))
        if mode in {"narrow_replacement", "conditional_segmenter"}
        else _mask(shape, slice(7, 10), slice(10, 14))
    )
    generation = commit.copy()
    if mode in {"narrow_replacement", "conditional_segmenter"}:
        generation = np.bitwise_or(generation, existing)

    plan = _plan(
        mode,
        generation=generation,
        commit=commit,
        existing=existing,
    )

    assert plan.mode == mode
    assert plan.shape == shape
    assert plan.generation_mask.dtype == np.uint8
    assert plan.generation_mask.flags.c_contiguous
    assert not plan.generation_mask.flags.writeable
    assert plan.lama_request_count == 1


def test_incremental_plan_rejects_unknown_mode_shape_and_nonbinary_mask() -> None:
    shape = (8, 10)
    empty = np.zeros(shape, dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown incremental inpaint mode"):
        _plan("wide_guess", generation=empty, commit=empty, existing=empty)
    with pytest.raises(ValueError, match="commit_mask shape mismatch"):
        _plan(
            "additive",
            generation=empty,
            commit=np.zeros((7, 10), dtype=np.uint8),
            existing=empty,
        )
    invalid = empty.copy()
    invalid[3, 4] = 127
    with pytest.raises(ValueError, match="only 0 and 255"):
        _plan("additive", generation=invalid, commit=empty, existing=empty)


def test_commit_mask_must_be_a_subset_of_generation_mask() -> None:
    shape = (10, 14)
    generation = _mask(shape, slice(2, 5), slice(2, 5))
    commit = generation.copy()
    commit[7, 9] = 255

    with pytest.raises(ValueError, match="subset of generation_mask"):
        _plan(
            "narrow_replacement",
            generation=generation,
            commit=commit,
            existing=np.zeros(shape, dtype=np.uint8),
        )


def test_additive_excludes_existing_while_context_commits_only_addition() -> None:
    shape = (12, 16)
    existing = _mask(shape, slice(3, 7), slice(3, 8))
    context = existing.copy()
    context[8:10, 10:13] = 255
    addition = _mask(shape, slice(8, 10), slice(10, 13))

    with pytest.raises(ValueError, match="additive generation_mask"):
        _plan(
            "additive",
            generation=context,
            commit=addition,
            existing=existing,
        )
    context_plan = _plan(
        "context_additive",
        generation=context,
        commit=addition,
        existing=existing,
    )
    assert np.count_nonzero(context_plan.generation_mask & existing) > 0
    assert np.count_nonzero(context_plan.commit_mask & existing) == 0

    invalid_commit = addition.copy()
    invalid_commit[4, 4] = 255
    with pytest.raises(ValueError, match="commit_mask must contain additions only"):
        _plan(
            "context_additive",
            generation=context,
            commit=invalid_commit,
            existing=existing,
        )


def test_page_union_returns_at_most_one_lama_request_mask() -> None:
    shape = (18, 24)
    first_commit = _mask(shape, slice(3, 6), slice(3, 7))
    second_commit = _mask(shape, slice(11, 14), slice(15, 20))
    first = _plan(
        "additive",
        generation=first_commit,
        commit=first_commit,
        existing=np.zeros(shape, dtype=np.uint8),
    )
    second_generation = second_commit.copy()
    second_generation[9:11, 13:22] = 255
    second = _plan(
        "context_additive",
        generation=second_generation,
        commit=second_commit,
        existing=np.zeros(shape, dtype=np.uint8),
    )

    page = union_incremental_plans((first, second))

    assert page.lama_request_count == 1
    assert len(page.lama_request_masks()) == 1
    assert np.array_equal(
        page.generation_mask > 0,
        (first.generation_mask > 0) | (second.generation_mask > 0),
    )
    assert np.array_equal(
        page.commit_mask > 0,
        (first.commit_mask > 0) | (second.commit_mask > 0),
    )


def test_page_plan_rejects_a_forged_union_mask() -> None:
    shape = (8, 12)
    commit = _mask(shape, slice(2, 4), slice(3, 6))
    plan = _plan(
        "additive",
        generation=commit,
        commit=commit,
        existing=np.zeros(shape, dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="must equal its per-plan union"):
        PageIncrementalInpaintPlan(
            plans=(plan,),
            generation_mask=np.zeros(shape, dtype=np.uint8),
            commit_mask=np.zeros(shape, dtype=np.uint8),
            existing_source_edit=np.zeros(shape, dtype=np.uint8),
            replacement_source_edit=np.zeros(shape, dtype=np.uint8),
        )


def test_empty_page_plan_has_no_request_and_is_byte_identical() -> None:
    shape = (10, 14)
    page = union_incremental_plans((), shape=shape)
    original = np.arange(shape[0] * shape[1] * 3, dtype=np.uint8).reshape(
        shape[0], shape[1], 3
    )
    baseline = np.flip(original, axis=1).copy()
    baseline_mask = _mask(shape, slice(2, 5), slice(3, 7))
    before_sha = hashlib.sha256(baseline.tobytes()).hexdigest()

    generated, receipt, calls = _execute(original, page, None)
    candidate, final_mask = composite_incremental_result(
        original,
        baseline,
        generated,
        baseline_mask,
        page,
        receipt,
    )

    assert page.lama_request_count == 0
    assert calls == []
    assert page.lama_request_masks() == ()
    assert hashlib.sha256(candidate.tobytes()).hexdigest() == before_sha
    assert np.array_equal(candidate, baseline)
    assert np.array_equal(final_mask, baseline_mask)


@pytest.mark.parametrize(
    "mode", ["narrow_replacement", "conditional_segmenter"]
)
def test_empty_replacement_cannot_restore_existing_pr6_edits(mode: str) -> None:
    shape = (10, 14)
    empty = np.zeros(shape, dtype=np.uint8)
    existing = _mask(shape, slice(2, 6), slice(3, 8))

    with pytest.raises(
        ValueError,
        match="empty replacement commit must not restore existing source edits",
    ):
        _plan(
            mode,
            generation=empty,
            commit=empty,
            existing=existing,
        )


def test_context_additive_generation_changes_only_exact_commit_addition() -> None:
    shape = (12, 16)
    original = np.full((*shape, 3), 20, dtype=np.uint8)
    baseline = original.copy()
    existing = _mask(shape, slice(2, 5), slice(2, 6))
    baseline[existing > 0] = 80
    addition = _mask(shape, slice(8, 10), slice(10, 14))
    generation = np.bitwise_or(existing, addition)
    page = union_incremental_plans(
        (
            _plan(
                "context_additive",
                generation=generation,
                commit=addition,
                existing=existing,
            ),
        )
    )
    expected_generated = np.full_like(original, 230)
    generated, receipt, calls = _execute(original, page, expected_generated)

    candidate, final_mask = composite_incremental_result(
        original,
        baseline,
        generated,
        existing,
        page,
        receipt,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0], page.generation_mask)
    assert np.all(candidate[addition > 0] == 230)
    assert np.array_equal(candidate[addition == 0], baseline[addition == 0])
    assert np.array_equal(final_mask > 0, (existing > 0) | (addition > 0))


@pytest.mark.parametrize(
    "mode", ["narrow_replacement", "conditional_segmenter"]
)
def test_replacement_modes_restore_source_edit_then_commit_narrow_result(
    mode: str,
) -> None:
    shape = (14, 20)
    original = np.full((*shape, 3), 25, dtype=np.uint8)
    baseline = original.copy()
    source_edit = _mask(shape, slice(3, 9), slice(3, 11))
    unrelated = _mask(shape, slice(10, 13), slice(14, 18))
    baseline_mask = np.bitwise_or(source_edit, unrelated)
    baseline[source_edit > 0] = 90
    baseline[unrelated > 0] = 120
    narrow = _mask(shape, slice(5, 8), slice(6, 10))
    generation = narrow.copy()
    generation[4:9, 5:11] = 255
    page = union_incremental_plans(
        (
            _plan(
                mode,
                generation=generation,
                commit=narrow,
                existing=source_edit,
            ),
        )
    )
    expected_generated = np.full_like(original, 210)
    generated, receipt, calls = _execute(original, page, expected_generated)

    candidate, final_mask = composite_incremental_result(
        original,
        baseline,
        generated,
        baseline_mask,
        page,
        receipt,
    )

    assert len(calls) == 1
    restored = (source_edit > 0) & (narrow <= 0)
    assert np.array_equal(candidate[restored], original[restored])
    assert np.all(candidate[narrow > 0] == 210)
    assert np.array_equal(candidate[unrelated > 0], baseline[unrelated > 0])
    assert np.count_nonzero(final_mask[restored]) == 0
    assert np.all(final_mask[narrow > 0] == 255)
    assert np.all(final_mask[unrelated > 0] == 255)


def test_page_union_restores_only_replacement_plan_source_edits() -> None:
    shape = (12, 18)
    additive_existing = _mask(shape, slice(2, 4), slice(2, 5))
    replacement_existing = _mask(shape, slice(6, 10), slice(4, 9))
    addition = _mask(shape, slice(2, 4), slice(12, 15))
    replacement = _mask(shape, slice(7, 9), slice(6, 8))
    page = union_incremental_plans(
        (
            _plan(
                "additive",
                generation=addition,
                commit=addition,
                existing=additive_existing,
            ),
            _plan(
                "narrow_replacement",
                generation=replacement,
                commit=replacement,
                existing=replacement_existing,
            ),
        )
    )

    assert np.array_equal(
        page.existing_source_edit > 0,
        (additive_existing > 0) | (replacement_existing > 0),
    )
    assert np.array_equal(page.replacement_source_edit, replacement_existing)


def test_replacement_source_edit_must_belong_to_baseline_mask() -> None:
    shape = (8, 12)
    source_edit = _mask(shape, slice(2, 5), slice(3, 7))
    narrow = _mask(shape, slice(3, 4), slice(4, 6))
    page = union_incremental_plans(
        (
            _plan(
                "narrow_replacement",
                generation=narrow,
                commit=narrow,
                existing=source_edit,
            ),
        )
    )
    image = np.zeros((*shape, 3), dtype=np.uint8)

    generated, receipt, _calls = _execute(image, page, image)
    with pytest.raises(ValueError, match="subset of baseline_mask"):
        composite_incremental_result(
            image,
            image,
            generated,
            np.zeros(shape, dtype=np.uint8),
            page,
            receipt,
        )


def test_replacement_rejects_an_unrelated_existing_component() -> None:
    shape = (14, 22)
    targeted = _mask(shape, slice(3, 7), slice(3, 8))
    unrelated = _mask(shape, slice(9, 12), slice(16, 20))
    existing = np.bitwise_or(targeted, unrelated)
    commit = _mask(shape, slice(4, 6), slice(4, 7))

    with pytest.raises(
        ValueError,
        match="replacement source component must contact commit_mask",
    ):
        _plan(
            "narrow_replacement",
            generation=targeted,
            commit=commit,
            existing=existing,
        )


def test_replacement_component_checks_scan_only_component_bounding_boxes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2160, 3840)
    existing = np.zeros(shape, dtype=np.uint8)
    commit = np.zeros(shape, dtype=np.uint8)
    for index in range(48):
        y = 40 + (index // 12) * 480
        x = 40 + (index % 12) * 300
        existing[y : y + 5, x : x + 7] = 255
        commit[y + 1 : y + 4, x + 1 : x + 6] = 255

    compared_shapes: list[tuple[int, int]] = []
    original = __import__(
        "benchmarking.inpaint_detector_bakeoff.incremental_plan",
        fromlist=["cv2"],
    ).cv2.connectedComponentsWithStats

    class TrackingLabels(np.ndarray):
        def __new__(cls, value: np.ndarray) -> "TrackingLabels":
            return np.asarray(value).view(cls)

        def __eq__(self, other: object) -> np.ndarray:  # type: ignore[override]
            compared_shapes.append(tuple(self.shape))
            return np.asarray(self).__eq__(other)

    def tracked_components(*args: object, **kwargs: object):
        count, labels, stats, centroids = original(*args, **kwargs)
        return count, TrackingLabels(labels), stats, centroids

    monkeypatch.setattr(
        "benchmarking.inpaint_detector_bakeoff.incremental_plan.cv2."
        "connectedComponentsWithStats",
        tracked_components,
    )

    _plan(
        "narrow_replacement",
        generation=np.bitwise_or(existing, commit),
        commit=commit,
        existing=existing,
    )

    assert len(compared_shapes) == 48
    assert max(height * width for height, width in compared_shapes) <= 35


def test_receipt_rejects_source_mask_and_generated_tampering() -> None:
    shape = (10, 16)
    commit = _mask(shape, slice(3, 6), slice(5, 9))
    page = union_incremental_plans(
        (
            _plan(
                "additive",
                generation=commit,
                commit=commit,
                existing=np.zeros(shape, dtype=np.uint8),
            ),
        )
    )
    source = np.full((*shape, 3), 25, dtype=np.uint8)
    baseline = source.copy()
    expected = np.full_like(source, 200)
    generated, receipt, calls = _execute(source, page, expected)
    assert len(calls) == 1

    changed_source = source.copy()
    changed_source[0, 0] = 26
    with pytest.raises(ValueError, match="source SHA differs"):
        composite_incremental_result(
            changed_source,
            baseline,
            generated,
            np.zeros(shape, dtype=np.uint8),
            page,
            receipt,
        )

    assert generated is not None
    changed_generated = generated.copy()
    changed_generated[0, 0] = 199
    with pytest.raises(ValueError, match="generated-image SHA differs"):
        composite_incremental_result(
            source,
            baseline,
            changed_generated,
            np.zeros(shape, dtype=np.uint8),
            page,
            receipt,
        )
