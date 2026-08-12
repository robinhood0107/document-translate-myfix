from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .contracts import CombinationClosureRecord, FactorizedRunRecord, binary_mask
from .stage1 import PageMasks


FillCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


def restrict_candidate_to_final_mask(
    source: np.ndarray,
    candidate: np.ndarray,
    original_final_mask: np.ndarray,
    restricted_final_mask: np.ndarray,
) -> np.ndarray:
    """Restore immutable source pixels removed by a final protection pass."""

    original = np.ascontiguousarray(np.asarray(source)[:, :, :3])
    result = np.ascontiguousarray(np.asarray(candidate)[:, :, :3]).copy()
    if result.shape != original.shape:
        raise ValueError("candidate shape mismatch")
    shape = original.shape[:2]
    old = binary_mask(original_final_mask, shape)
    new = binary_mask(restricted_final_mask, shape)
    if np.any((new > 0) & (old <= 0)):
        raise ValueError("restricted final mask cannot add pixels")
    restore = (old > 0) & (new <= 0)
    result[restore] = original[restore]
    return np.ascontiguousarray(result)


def _local_fill_domain(
    shape: tuple[int, int],
    edit: np.ndarray,
    interior: np.ndarray | None,
) -> np.ndarray:
    if interior is not None and np.any(interior):
        return binary_mask(interior, shape)
    points = cv2.findNonZero((edit > 0).astype(np.uint8))
    if points is None:
        return np.zeros(shape, dtype=np.uint8)
    x, y, width, height = cv2.boundingRect(points)
    pad = max(12, int(round(max(width, height) * 0.5)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(shape[1], x + width + pad)
    y2 = min(shape[0], y + height + pad)
    domain = np.zeros(shape, dtype=np.uint8)
    domain[y1:y2, x1:x2] = 255
    return domain


def _fill_samples(
    source: np.ndarray,
    edit: np.ndarray,
    domain: np.ndarray,
    exclude: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    allowed = (domain > 0) & (edit == 0) & (exclude == 0)
    ys, xs = np.where(allowed)
    return ys, xs, np.asarray(source)[ys, xs, :3]


def fill_factorized_mask(
    source: np.ndarray,
    edit_mask: np.ndarray,
    *,
    backend: str,
    interior_mask: np.ndarray | None = None,
    background_exclude_mask: np.ndarray | None = None,
    background_sample_edit_mask: np.ndarray | None = None,
    lama_fill: FillCallable | None = None,
    minimum_samples: int = 32,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate one fill backend while preserving immutable exact-mask composite."""

    original = np.ascontiguousarray(np.asarray(source)[:, :, :3])
    shape = original.shape[:2]
    edit = binary_mask(edit_mask, shape)
    if not np.any(edit):
        return original.copy(), {
            "backend": backend,
            "applied": False,
            "reason": "empty_edit_mask",
            "edit_pixel_count": 0,
            "sample_pixel_count": 0,
        }
    interior = binary_mask(interior_mask, shape) if interior_mask is not None else None
    exclude = (
        binary_mask(background_exclude_mask, shape)
        if background_exclude_mask is not None
        else np.zeros(shape, dtype=np.uint8)
    )
    domain = _local_fill_domain(shape, edit, interior)
    sample_edit = (
        binary_mask(background_sample_edit_mask, shape)
        if background_sample_edit_mask is not None
        else edit
    )
    if np.any((sample_edit > 0) & (edit == 0)):
        raise ValueError("background sample exclusion must be inside the edit mask")
    ys, xs, samples = _fill_samples(original, sample_edit, domain, exclude)
    sample_count = int(samples.shape[0])
    normalized = str(backend).strip().lower()
    generated = original.copy()
    diagnostics: dict[str, object] = {
        "backend": normalized,
        "applied": False,
        "reason": "",
        "edit_pixel_count": int(np.count_nonzero(edit)),
        "sample_pixel_count": sample_count,
        "sample_exclusion_pixel_count": int(np.count_nonzero(sample_edit)),
    }

    if normalized in {"current_lama", "ballons_lama"}:
        if lama_fill is None:
            raise ValueError(f"{normalized} requires a LaMa fill callback")
        generated = np.asarray(lama_fill(original.copy(), edit.copy()))[:, :, :3]
        if generated.shape != original.shape:
            raise ValueError("LaMa fill callback returned an invalid image shape")
    elif sample_count < int(minimum_samples):
        diagnostics["reason"] = "insufficient_roi_background_samples"
        return original.copy(), diagnostics
    elif normalized == "robust_flat_median":
        color = np.median(samples.astype(np.float32), axis=0)
        generated[edit > 0] = np.clip(np.rint(color), 0, 255).astype(np.uint8)
    elif normalized == "planar_gradient":
        if sample_count < max(64, int(minimum_samples)):
            diagnostics["reason"] = "insufficient_gradient_samples"
            return original.copy(), diagnostics
        if int(np.ptp(xs)) < 4 or int(np.ptp(ys)) < 4:
            diagnostics["reason"] = "insufficient_gradient_spread"
            return original.copy(), diagnostics
        if sample_count > 4096:
            indices = np.linspace(0, sample_count - 1, 4096, dtype=np.int64)
            fit_xs = xs[indices]
            fit_ys = ys[indices]
            fit_samples = samples[indices]
        else:
            fit_xs, fit_ys, fit_samples = xs, ys, samples
        design = np.column_stack(
            (
                np.ones(fit_xs.shape[0], dtype=np.float64),
                fit_xs.astype(np.float64),
                fit_ys.astype(np.float64),
            )
        )
        target_y, target_x = np.where(edit > 0)
        target_design = np.column_stack(
            (
                np.ones(target_x.shape[0], dtype=np.float64),
                target_x.astype(np.float64),
                target_y.astype(np.float64),
            )
        )
        for channel in range(3):
            coefficients, *_rest = np.linalg.lstsq(
                design,
                fit_samples[:, channel].astype(np.float64),
                rcond=None,
            )
            values = target_design @ coefficients
            generated[target_y, target_x, channel] = np.clip(
                np.rint(values), 0, 255
            ).astype(np.uint8)
    elif normalized == "telea":
        neutralized = original.copy()
        neutral_color = np.median(samples.astype(np.float32), axis=0)
        neutralized[(domain > 0) & (exclude > 0)] = np.clip(
            np.rint(neutral_color), 0, 255
        ).astype(np.uint8)
        generated = cv2.inpaint(neutralized, edit, 3.0, cv2.INPAINT_TELEA)
    else:
        raise KeyError(f"unknown fill backend: {backend}")

    candidate = original.copy()
    candidate[edit > 0] = generated[edit > 0]
    diagnostics["applied"] = True
    return np.ascontiguousarray(candidate), diagnostics


def reconstruction_error(
    candidate: np.ndarray,
    known_background: np.ndarray,
    edit_mask: np.ndarray,
) -> float | None:
    left = np.asarray(candidate)[:, :, :3].astype(np.float32)
    right = np.asarray(known_background)[:, :, :3].astype(np.float32)
    if left.shape != right.shape:
        raise ValueError("known-background image shape mismatch")
    edit = binary_mask(edit_mask, left.shape[:2]) > 0
    if not np.any(edit):
        return None
    delta = left[edit] - right[edit]
    return float(np.mean(delta * delta))


def build_factorized_matrix(
    axes: Mapping[str, Sequence[str]],
    controls: Mapping[str, str],
) -> list[dict[str, str]]:
    """Return every logical Cartesian selection in deterministic role order."""

    ordered_axes = tuple(controls)
    if set(axes) != set(ordered_axes):
        raise ValueError("matrix axes and controls must have the same role names")
    for role in ordered_axes:
        if controls[role] not in axes[role]:
            raise ValueError(f"control is missing from matrix axis: {role}")
    values = []
    for role in ordered_axes:
        choices = tuple(dict.fromkeys(str(value) for value in axes[role]))
        if not choices:
            raise ValueError(f"matrix axis is empty: {role}")
        values.append(choices)
    return [dict(zip(ordered_axes, selection)) for selection in product(*values)]


INVALID_COMBINATION_REASONS = frozenset(
    {
        "broad_expansion_requires_broad_router",
        "bubble_fill_requires_silhouette",
        "roi_trigger_requires_roi_detector",
        "runtime_detector_limit_exceeded",
        "oracle_product_candidate",
        "stage1_fill_backend_forbidden",
        "broad_expansion_requires_source_seed",
        "provider_asset_or_parity_missing",
    }
)


def validate_factorized_selection(
    selection: Mapping[str, str],
    *,
    stage: str,
    family_metadata: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    oracle_only_ids: Iterable[str] = (),
) -> tuple[str | None, str]:
    """Return a stable exclusion reason and closure class for one logical run."""

    expansion = str(selection.get("expansion") or "").lower()
    router = str(selection.get("router") or "").upper()
    silhouette = str(selection.get("silhouette") or "").lower()
    fill = str(selection.get("fill") or "").lower()
    detector = str(selection.get("detector") or "").lower()
    if expansion in {"bubble_interior", "validated_interior"} and router in {
        "R0",
        "CONTROL_R0",
    }:
        return "broad_expansion_requires_broad_router", "invalid_with_reason"
    if fill in {"robust_flat_median", "planar_gradient", "telea", "conditional_hybrid"} and (
        not silhouette or "empty" in silhouette
    ):
        return "bubble_fill_requires_silhouette", "invalid_with_reason"
    if "roi_trigger" in selection and selection["roi_trigger"] != "none" and "roi" not in detector:
        return "roi_trigger_requires_roi_detector", "invalid_with_reason"
    detector_count = int(selection.get("runtime_detector_count", "1") or "1")
    if detector_count > 2:
        return "runtime_detector_limit_exceeded", "invalid_with_reason"
    oracle = set(oracle_only_ids)
    if stage == "product" and any(value in oracle for value in selection.values()):
        return "oracle_product_candidate", "invalid_with_reason"
    if stage == "stage1" and fill != "mask_only":
        return "stage1_fill_backend_forbidden", "invalid_with_reason"
    if family_metadata:
        for role, value in selection.items():
            metadata = family_metadata.get(role, {}).get(value, {})
            if metadata.get("asset_status") in {"missing", "blocked"} or metadata.get(
                "parity_status"
            ) in {"missing", "failed"}:
                return "provider_asset_or_parity_missing", "blocked_asset"
    return None, "executed"


def build_combination_closure_ledger(
    selections: Iterable[Mapping[str, str]],
    *,
    stage: str,
    family_metadata: Mapping[str, Mapping[str, Mapping[str, object]]] | None = None,
    oracle_only_ids: Iterable[str] = (),
) -> list[CombinationClosureRecord]:
    records: list[CombinationClosureRecord] = []
    for selection in selections:
        normalized = {str(role): str(value) for role, value in selection.items()}
        logical_id = "__".join(normalized[role] for role in normalized)
        reason, state = validate_factorized_selection(
            normalized,
            stage=stage,
            family_metadata=family_metadata,
            oracle_only_ids=oracle_only_ids,
        )
        records.append(
            CombinationClosureRecord(
                logical_id=logical_id,
                selection=normalized,
                closure_state=state,
                reason=reason or "",
            )
        )
    return records


def assert_complete_closure_ledger(
    logical_selections: Iterable[Mapping[str, str]],
    ledger: Iterable[CombinationClosureRecord],
) -> None:
    expected = {
        "__".join(str(selection[role]) for role in selection)
        for selection in logical_selections
    }
    rows = list(ledger)
    actual = {row.logical_id for row in rows}
    if len(actual) != len(rows):
        raise ValueError("combination closure ledger contains duplicate logical ids")
    missing = sorted(expected.difference(actual))
    extra = sorted(actual.difference(expected))
    if missing or extra:
        raise ValueError(f"combination closure ledger mismatch: missing={missing}, extra={extra}")


def _hard_gate_passes(metrics: Mapping[str, object]) -> bool:
    if (
        metrics.get("target_extent_independent") is False
        or metrics.get("target_inventory_independent") is False
    ):
        return False
    zero_metrics = (
        "protected_structure_overlap",
        "protected_structure_changed",
        "ambiguous_structure_overlap",
        "ambiguous_structure_changed",
        "outside_final_changed",
        "broad_route_false_positive",
        "no_edit_false_edit",
        "required_skip_count",
        "preserve_edit_overlap",
        "missed_target_instance_count",
        "page_residue_worsened_count",
    )
    if any(int(metrics.get(name, 0) or 0) != 0 for name in zero_metrics):
        return False
    coverage = metrics.get("aggregate_target_coverage")
    minimum = metrics.get("minimum_target_instance_coverage")
    seed_recall = metrics.get("target_instance_seed_recall")
    aggregate_residue = metrics.get("aggregate_residue_score")
    baseline_residue = metrics.get("baseline_aggregate_residue_score")
    residue_gate_applicable = bool(metrics.get("residue_gate_applicable", True))
    residue_improved = not residue_gate_applicable or (
        aggregate_residue is not None
        and baseline_residue is not None
        and float(aggregate_residue) < float(baseline_residue)
    )
    return (
        (coverage is None or float(coverage) >= 0.98)
        and (minimum is None or float(minimum) >= 0.98)
        and (seed_recall is None or float(seed_recall) >= 1.0)
        and residue_improved
    )


def select_pareto_records(
    records: Iterable[FactorizedRunRecord],
) -> list[FactorizedRunRecord]:
    """Mark safe non-oracle records Pareto/dominated on residue, reconstruction, time."""

    rows = list(records)
    eligible = [
        row for row in rows if not row.oracle_only and _hard_gate_passes(row.metrics)
    ]

    def axes(row: FactorizedRunRecord) -> tuple[float, float, float, float]:
        metrics = row.metrics
        def metric(name: str, default: float) -> float:
            value = metrics.get(name)
            return default if value is None else float(value)

        return (
            -metric("aggregate_target_coverage", 0.0),
            metric("aggregate_residue_score", float("inf")),
            metric("reconstruction_mse", float("inf")),
            metric("runtime_seconds", float("inf")),
        )

    pareto_ids: set[str] = set()
    for candidate in eligible:
        current = axes(candidate)
        dominated = False
        for other in eligible:
            if other.run_id == candidate.run_id:
                continue
            comparison = axes(other)
            if all(left <= right for left, right in zip(comparison, current)) and any(
                left < right for left, right in zip(comparison, current)
            ):
                dominated = True
                break
        if not dominated:
            pareto_ids.add(candidate.run_id)

    output: list[FactorizedRunRecord] = []
    for row in rows:
        if row.oracle_only:
            output.append(replace(row, status="family_complete"))
        elif (
            row.metrics.get("target_extent_independent") is False
            or row.metrics.get("target_inventory_independent") is False
        ):
            output.append(
                replace(
                    row,
                    status="information_limited",
                    closure_reason=(
                        row.closure_reason
                        or (
                            "target_extent_not_independent"
                            if row.metrics.get("target_extent_independent") is False
                            else "target_inventory_not_independent"
                        )
                    ),
                )
            )
        elif row.run_id in pareto_ids:
            output.append(replace(row, status="pareto"))
        elif row in eligible:
            output.append(replace(row, status="dominated"))
        else:
            output.append(
                replace(
                    row,
                    status="dominated",
                    closure_reason=row.closure_reason or "hard_gate_failed",
                )
            )
    return output


def composite_positive_result(
    baseline: np.ndarray,
    generated: np.ndarray,
    positive_edit: np.ndarray,
    baseline_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Composite only detector-positive pixels onto an immutable baseline."""

    left = np.asarray(baseline)
    right = np.asarray(generated)
    if left.shape != right.shape:
        raise ValueError("positive composite image shape mismatch")
    edit = binary_mask(positive_edit, left.shape[:2])
    existing = binary_mask(baseline_mask, left.shape[:2])
    candidate = left.copy()
    candidate[edit > 0] = right[edit > 0]
    final_mask = np.where((existing > 0) | (edit > 0), 255, 0).astype(np.uint8)
    return np.ascontiguousarray(candidate), np.ascontiguousarray(final_mask)


def composite_replacement_result(
    original: np.ndarray,
    baseline: np.ndarray,
    generated: np.ndarray,
    replacement_edit: np.ndarray,
    baseline_mask: np.ndarray,
    existing_source_edit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace source-owned edits while retaining non-source baseline edits."""

    source = np.asarray(original)
    prior = np.asarray(baseline)
    replacement = np.asarray(generated)
    if source.shape != prior.shape or source.shape != replacement.shape:
        raise ValueError("replacement composite image shape mismatch")
    shape = source.shape[:2]
    edit = binary_mask(replacement_edit, shape)
    prior_mask = binary_mask(baseline_mask, shape)
    source_edit = binary_mask(existing_source_edit, shape)
    safe_prior = np.where(
        (prior_mask > 0) & (source_edit == 0),
        255,
        0,
    ).astype(np.uint8)
    candidate = source.copy()
    candidate[safe_prior > 0] = prior[safe_prior > 0]
    candidate[edit > 0] = replacement[edit > 0]
    final_mask = cv2.bitwise_or(safe_prior, edit)
    return np.ascontiguousarray(candidate), np.ascontiguousarray(final_mask)


def changed_mask(source: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    left = np.asarray(source)
    right = np.asarray(candidate)
    if left.shape != right.shape:
        raise ValueError("stage2 image shape mismatch")
    changed = np.any(left[:, :, :3] != right[:, :, :3], axis=2)
    return np.where(changed, 255, 0).astype(np.uint8)


def _component_coverages(target: np.ndarray, mask: np.ndarray) -> list[float]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (target > 0).astype(np.uint8),
        connectivity=8,
    )
    values: list[float] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area > 0:
            x = int(stats[index, cv2.CC_STAT_LEFT])
            y = int(stats[index, cv2.CC_STAT_TOP])
            width = int(stats[index, cv2.CC_STAT_WIDTH])
            height = int(stats[index, cv2.CC_STAT_HEIGHT])
            component = labels[y : y + height, x : x + width] == index
            local_mask = mask[y : y + height, x : x + width] > 0
            values.append(
                float(np.count_nonzero(component & local_mask)) / float(area)
            )
    return values


def _gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(image)[:, :, :3], cv2.COLOR_BGR2GRAY)


def residue_score(
    source: np.ndarray,
    candidate: np.ndarray,
    target: np.ndarray,
) -> tuple[float | None, float, int]:
    normalized = binary_mask(target, source.shape[:2])
    source_gray = _gray(source)
    candidate_gray = _gray(candidate)
    source_background = cv2.GaussianBlur(source_gray, (15, 15), 0)
    candidate_background = cv2.GaussianBlur(candidate_gray, (15, 15), 0)
    source_contrast = np.abs(
        source_gray.astype(np.int16) - source_background.astype(np.int16)
    )
    candidate_contrast = np.abs(
        candidate_gray.astype(np.int16) - candidate_background.astype(np.int16)
    )
    residue_source = (normalized > 0) & (source_contrast >= 8)
    source_count = int(np.count_nonzero(residue_source))
    if source_count <= 0:
        return None, 0.0, 0
    ratios = np.minimum(
        candidate_contrast[residue_source].astype(np.float32)
        / np.maximum(source_contrast[residue_source].astype(np.float32), 1.0),
        1.0,
    )
    score = float(np.mean(ratios))
    return score, float(np.sum(ratios)), source_count


def score_stage2_page(
    source: np.ndarray,
    candidate: np.ndarray,
    detector_mask: np.ndarray,
    masks: PageMasks,
    *,
    baseline: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray]:
    detector = binary_mask(detector_mask, source.shape[:2])
    changed = changed_mask(source, candidate)
    target_pixels = int(np.count_nonzero(masks.target))
    target_coverages = _component_coverages(masks.target, detector)
    score, score_sum, score_count = residue_score(
        source,
        candidate,
        masks.target,
    )
    baseline_score = None
    if baseline is not None:
        baseline_score, _baseline_sum, _baseline_count = residue_score(
            source,
            baseline,
            masks.target,
        )
    record = {
        "detector_mask_pixel_count": int(np.count_nonzero(detector)),
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_outside_detector_mask_pixel_count": int(
            np.count_nonzero((changed > 0) & (detector == 0))
        ),
        "target_pixel_count": target_pixels,
        "target_detector_covered_pixel_count": int(
            np.count_nonzero((masks.target > 0) & (detector > 0))
        ),
        "target_detector_coverage": (
            float(np.count_nonzero((masks.target > 0) & (detector > 0)))
            / float(target_pixels)
            if target_pixels
            else None
        ),
        "target_component_coverages": target_coverages,
        "minimum_target_component_coverage": (
            min(target_coverages) if target_coverages else None
        ),
        "protected_changed_pixel_count": int(
            np.count_nonzero((masks.protected > 0) & (changed > 0))
        ),
        "ambiguous_changed_pixel_count": int(
            np.count_nonzero((masks.ambiguous > 0) & (changed > 0))
        ),
        "residue_score": score,
        "residue_score_sum": score_sum,
        "residue_source_contrast_pixel_count": score_count,
        "baseline_residue_score": baseline_score,
        "residue_score_delta_from_baseline": (
            float(score) - float(baseline_score)
            if score is not None and baseline_score is not None
            else None
        ),
    }
    return record, changed
