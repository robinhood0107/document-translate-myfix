from __future__ import annotations

from dataclasses import replace
from itertools import product
import math
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .contracts import CombinationClosureRecord, FactorizedRunRecord, binary_mask
from .stage1 import PageMasks


FillCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


RELATIVE_PRODUCT_SAFETY_ZERO_FIELDS = (
    "protected_structure_overlap",
    "protected_structure_changed",
    "ambiguous_structure_overlap",
    "ambiguous_structure_changed",
    "preserve_edit_overlap",
    "ownership_leak_pixel_count",
    "corner_edit_overlap_pixel_count",
    "outside_final_changed",
    "broad_route_false_positive",
    "no_edit_false_edit",
    "required_skip_count",
    "cpu_fallback_count",
)


def _relative_page_map(
    rows: Sequence[Mapping[str, object]],
    *,
    label: str,
) -> dict[str, Mapping[str, object]]:
    mapped: dict[str, Mapping[str, object]] = {}
    for row in rows:
        page_id = str(row.get("page_id") or "").strip()
        if not page_id or page_id in mapped:
            raise ValueError(f"{label} page rows require unique page ids")
        mapped[page_id] = row
    if not mapped:
        raise ValueError(f"{label} page rows must not be empty")
    return mapped


def _relative_instance_coverages(
    pages: Mapping[str, Mapping[str, object]],
) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    for page_id, row in pages.items():
        scores = row.get("target_instance_edit_scores")
        if not isinstance(scores, list) or any(
            not isinstance(score, Mapping) for score in scores
        ):
            raise ValueError("relative product page lacks target instance scores")
        for score in scores:
            instance_id = str(score.get("instance_id") or "").strip()
            coverage = score.get("coverage")
            key = (page_id, instance_id)
            if (
                not instance_id
                or key in values
                or not _finite_number(coverage)
                or not 0.0 <= float(coverage) <= 1.0
            ):
                raise ValueError("relative product instance score is invalid")
            values[key] = float(coverage)
    return values


def evaluate_relative_product_gate(
    *,
    baseline_metrics: Mapping[str, object],
    candidate_metrics: Mapping[str, object],
    baseline_pages: Sequence[Mapping[str, object]],
    candidate_pages: Sequence[Mapping[str, object]],
    candidate_kind: str,
    residue_tolerance: float = 1e-9,
    instance_delta: float = 0.01,
) -> dict[str, object]:
    """Compare one product candidate with a stronger sealed baseline.

    The existing strict 98%/100% research gate remains unchanged.  This gate
    admits a best-relative product only when every destructive-safety metric is
    zero and the candidate improves at least one quality axis without losing a
    baseline-good target instance.
    """

    kind = str(candidate_kind).strip().lower()
    if kind not in {"balanced", "fill_only"}:
        raise ValueError("candidate_kind must be balanced or fill_only")
    if (
        not _finite_number(residue_tolerance)
        or float(residue_tolerance) < 0.0
        or not _finite_number(instance_delta)
        or not 0.0 < float(instance_delta) <= 1.0
    ):
        raise ValueError("relative gate tolerances are invalid")

    failures: list[str] = []
    for field in RELATIVE_PRODUCT_SAFETY_ZERO_FIELDS:
        value = candidate_metrics.get(field)
        if not _finite_number(value) or float(value) != 0.0:
            failures.append(f"safety_nonzero:{field}")
    if candidate_metrics.get("runtime_telemetry_complete") is not True:
        failures.append("runtime_telemetry_incomplete")
    maximum_calls = candidate_metrics.get("maximum_positive_lama_inference_per_page")
    if not _finite_number(maximum_calls) or float(maximum_calls) not in {0.0, 1.0}:
        failures.append("positive_lama_page_call_limit")
    inference_count = candidate_metrics.get("positive_lama_inference_count")
    if (
        not _finite_number(inference_count)
        or float(inference_count) < 0.0
        or not float(inference_count).is_integer()
    ):
        failures.append("positive_lama_inference_count_invalid")
    elif int(inference_count) > 0:
        provider = str(candidate_metrics.get("lama_runtime_provider") or "").lower()
        precision = str(candidate_metrics.get("lama_runtime_precision") or "").lower()
        if not provider.startswith("cuda") or precision != "bf16":
            failures.append("positive_lama_not_cuda")

    baseline_by_page = _relative_page_map(baseline_pages, label="baseline")
    candidate_by_page = _relative_page_map(candidate_pages, label="candidate")
    if set(baseline_by_page) != set(candidate_by_page):
        raise ValueError("relative product page inventories differ")
    baseline_instances = _relative_instance_coverages(baseline_by_page)
    candidate_instances = _relative_instance_coverages(candidate_by_page)
    if set(baseline_instances) != set(candidate_instances):
        raise ValueError("relative product target instance inventories differ")

    baseline_coverage = baseline_metrics.get("aggregate_target_coverage")
    candidate_coverage = candidate_metrics.get("aggregate_target_coverage")
    if not _finite_number(baseline_coverage) or not _finite_number(candidate_coverage):
        raise ValueError("relative product aggregate coverage is unavailable")
    baseline_coverage_value = float(baseline_coverage)
    candidate_coverage_value = float(candidate_coverage)
    if candidate_coverage_value + 1e-12 < baseline_coverage_value:
        failures.append("aggregate_target_coverage_regressed")

    newly_missed = 0
    regressed_from_98 = 0
    improved = 0
    regressed = 0
    baseline_98 = candidate_98 = 0
    delta = float(instance_delta)
    for key, baseline_value in baseline_instances.items():
        candidate_value = candidate_instances[key]
        baseline_98 += int(baseline_value >= 0.98)
        candidate_98 += int(candidate_value >= 0.98)
        newly_missed += int(baseline_value > 0.0 and candidate_value <= 0.0)
        regressed_from_98 += int(baseline_value >= 0.98 and candidate_value < 0.98)
        improved += int(candidate_value - baseline_value >= delta)
        regressed += int(baseline_value - candidate_value >= delta)
    if newly_missed:
        failures.append("newly_missed_required_instance")
    if regressed_from_98:
        failures.append("baseline_98_instance_regressed")
    if regressed and improved <= regressed:
        failures.append("instance_improvement_not_greater_than_regression")

    page_residue_worsened = 0
    baseline_residue_sum = candidate_residue_sum = 0.0
    residue_page_count = 0
    tolerance = float(residue_tolerance)
    for page_id in sorted(baseline_by_page):
        baseline_residue = baseline_by_page[page_id].get("residue_score")
        candidate_residue = candidate_by_page[page_id].get("residue_score")
        if baseline_residue is None and candidate_residue is None:
            continue
        if not _finite_number(baseline_residue) or not _finite_number(candidate_residue):
            failures.append(f"page_residue_unavailable:{page_id}")
            continue
        baseline_value = float(baseline_residue)
        candidate_value = float(candidate_residue)
        baseline_residue_sum += baseline_value
        candidate_residue_sum += candidate_value
        residue_page_count += 1
        page_residue_worsened += int(candidate_value > baseline_value + tolerance)
    if page_residue_worsened:
        failures.append("required_page_residue_worsened")

    baseline_residue = baseline_metrics.get("aggregate_residue_score")
    candidate_residue = candidate_metrics.get("aggregate_residue_score")
    residue_comparable = _finite_number(baseline_residue) and _finite_number(
        candidate_residue
    )
    residue_improved = bool(
        residue_comparable
        and float(candidate_residue) < float(baseline_residue) - tolerance
    )
    residue_nonworse = bool(
        residue_comparable
        and float(candidate_residue) <= float(baseline_residue) + tolerance
    )
    if residue_page_count and not residue_nonworse:
        failures.append("aggregate_residue_regressed")

    mask_identity_preserved = (
        str(baseline_metrics.get("output_mask_set_sha256") or "")
        == str(candidate_metrics.get("output_mask_set_sha256") or "")
        and bool(str(baseline_metrics.get("output_mask_set_sha256") or ""))
    )
    coverage_improved = candidate_coverage_value > baseline_coverage_value + 1e-12
    instance_98_improved = candidate_98 > baseline_98
    if kind == "fill_only":
        if not mask_identity_preserved:
            failures.append("fill_only_edit_mask_changed")
        if not residue_improved:
            failures.append("fill_only_residue_not_improved")
    elif not (coverage_improved or instance_98_improved or residue_improved):
        failures.append("no_strict_relative_improvement")

    strict_seed_eligible = (
        _finite_number(candidate_metrics.get("target_instance_seed_recall"))
        and float(candidate_metrics["target_instance_seed_recall"]) >= 1.0
        and _finite_number(candidate_metrics.get("missed_target_instance_count"))
        and float(candidate_metrics["missed_target_instance_count"]) == 0.0
    )
    return {
        "candidate_kind": kind,
        "relative_product_pass": not failures,
        "strict_seed_eligible": bool(strict_seed_eligible),
        "gate_failures": sorted(set(failures)),
        "baseline_aggregate_target_coverage": baseline_coverage_value,
        "candidate_aggregate_target_coverage": candidate_coverage_value,
        "baseline_98_instance_count": baseline_98,
        "candidate_98_instance_count": candidate_98,
        "newly_missed_required_instance_count": newly_missed,
        "regressed_from_98_instance_count": regressed_from_98,
        "improved_instance_count": improved,
        "regressed_instance_count": regressed,
        "page_residue_worsened_count": page_residue_worsened,
        "baseline_aggregate_residue_score": (
            float(baseline_residue) if _finite_number(baseline_residue) else None
        ),
        "candidate_aggregate_residue_score": (
            float(candidate_residue) if _finite_number(candidate_residue) else None
        ),
        "residue_improved": residue_improved,
        "edit_mask_identity_preserved": mask_identity_preserved,
    }


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
    if fill in {
        "robust_flat_median",
        "planar_gradient",
        "telea",
        "conditional_hybrid",
        "conditional_refill_existing",
    } and (
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
    if expansion in {
        "content_component",
        "bubble_interior",
        "validated_interior",
        "lab_dilate1",
        "lab_dilate2",
        "lab_dilate3",
        "lab_dilate4",
    } and family_metadata:
        detector_metadata = family_metadata.get("detector", {}).get(detector, {})
        if detector_metadata.get("source_seed_available") is False:
            return "broad_expansion_requires_source_seed", "invalid_with_reason"
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


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _hard_gate_passes(metrics: Mapping[str, object]) -> bool:
    """Fail closed unless a run carries the complete v3 safety contract.

    Older result rows often omitted a metric and were consequently interpreted
    as zero (or treated missing target coverage as acceptable).  That makes an
    incomplete artifact eligible for Pareto selection.  The gate deliberately
    distinguishes a source-only mask run from a product/fill run, but every
    required field must be explicitly present and finite in either mode.
    """

    if any(
        metrics.get(name) is not True
        for name in (
            "target_extent_independent",
            "target_inventory_independent",
            "target_review_complete",
        )
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
        "ownership_leak_pixel_count",
        "corner_edit_overlap_pixel_count",
        "missed_target_instance_count",
        "page_residue_worsened_count",
    )
    if any(
        name not in metrics
        or not _finite_number(metrics.get(name))
        or int(metrics[name]) != 0
        for name in zero_metrics
    ):
        return False
    for name in ("page_count", "required_target_instance_count", "target_pixel_count"):
        if name not in metrics or not _finite_number(metrics.get(name)):
            return False
        if int(metrics[name]) < 0:
            return False
    if int(metrics["page_count"]) < 1:
        return False
    coverage = metrics.get("aggregate_target_coverage")
    minimum = metrics.get("minimum_target_instance_coverage")
    seed_recall = metrics.get("target_instance_seed_recall")
    required_instances = int(metrics["required_target_instance_count"])
    target_pixels = int(metrics["target_pixel_count"])
    if required_instances > 0:
        if not all(_finite_number(value) for value in (coverage, minimum, seed_recall)):
            return False
        role_recall = metrics.get("target_instance_seed_recall_by_semantic_role")
        if not isinstance(role_recall, Mapping) or not role_recall:
            return False
        if any(
            not str(role).strip()
            or not _finite_number(value)
            or float(value) < 1.0
            for role, value in role_recall.items()
        ):
            return False
    elif any(value is not None for value in (minimum, seed_recall)):
        return False
    if target_pixels > 0:
        if not _finite_number(coverage):
            return False
    elif coverage is not None:
        return False
    aggregate_residue = metrics.get("aggregate_residue_score")
    baseline_residue = metrics.get("baseline_aggregate_residue_score")
    residue_gate_applicable = metrics.get("residue_gate_applicable")
    if not isinstance(residue_gate_applicable, bool):
        return False
    residue_improved = not residue_gate_applicable or (
        _finite_number(aggregate_residue)
        and _finite_number(baseline_residue)
        and float(aggregate_residue) < float(baseline_residue)
    )
    reconstruction_gate_applicable = metrics.get("reconstruction_gate_applicable")
    if not isinstance(reconstruction_gate_applicable, bool):
        return False
    if reconstruction_gate_applicable:
        reconstruction = metrics.get("reconstruction_mse")
        narrow_control = metrics.get("narrow_control_reconstruction_mse")
        if not _finite_number(reconstruction) or not _finite_number(narrow_control):
            return False
        if float(reconstruction) > float(narrow_control) + 1e-12:
            return False

    runtime_telemetry_complete = metrics.get("runtime_telemetry_complete")
    if runtime_telemetry_complete is not True:
        return False
    runtime_seconds = metrics.get("runtime_seconds")
    inference_count = metrics.get("positive_lama_inference_count")
    maximum_page_calls = metrics.get("maximum_positive_lama_inference_per_page")
    cpu_fallback_count = metrics.get("cpu_fallback_count")
    if not all(
        _finite_number(value)
        for value in (
            runtime_seconds,
            inference_count,
            maximum_page_calls,
            cpu_fallback_count,
        )
    ):
        return False
    if (
        float(runtime_seconds) < 0.0
        or int(inference_count) < 0
        or int(maximum_page_calls) not in {0, 1}
        or int(cpu_fallback_count) != 0
    ):
        return False
    if int(inference_count) > 0:
        provider = str(metrics.get("lama_runtime_provider") or "").lower()
        precision = str(metrics.get("lama_runtime_precision") or "").lower()
        if not provider.startswith("cuda") or not precision:
            return False
        for name in (
            "positive_lama_runtime_p95_seconds",
            "peak_vram_allocated_mib",
            "peak_vram_reserved_mib",
        ):
            if not _finite_number(metrics.get(name)) or float(metrics[name]) < 0.0:
                return False
    return (
        (target_pixels == 0 or float(coverage) >= 0.98)
        and (required_instances == 0 or float(minimum) >= 0.98)
        and (required_instances == 0 or float(seed_recall) >= 1.0)
        and residue_improved
    )


def attach_reconstruction_control(
    records: Iterable[FactorizedRunRecord],
    control_run_id: str | None,
) -> list[FactorizedRunRecord]:
    """Bind synthetic reconstruction candidates to one executed narrow control."""

    rows = list(records)
    applicable = [
        row
        for row in rows
        if row.metrics.get("reconstruction_gate_applicable") is True
    ]
    if not applicable:
        return rows
    if not str(control_run_id or "").strip():
        return rows
    controls = [row for row in rows if row.run_id == str(control_run_id)]
    if len(controls) != 1:
        raise ValueError("reconstruction control run must be present exactly once")
    control_mse = controls[0].metrics.get("reconstruction_mse")
    if not _finite_number(control_mse):
        raise ValueError("reconstruction control run lacks a finite MSE")
    output: list[FactorizedRunRecord] = []
    for row in rows:
        if row.metrics.get("reconstruction_gate_applicable") is not True:
            output.append(row)
            continue
        metrics = dict(row.metrics)
        metrics["narrow_control_reconstruction_mse"] = float(control_mse)
        output.append(replace(row, metrics=metrics))
    return output


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
            or row.metrics.get("target_review_complete") is False
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
                            else (
                                "target_inventory_not_independent"
                                if row.metrics.get("target_inventory_independent") is False
                                else "target_review_incomplete"
                            )
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
