#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    CombinationClosureRecord,
    FactorizedRunRecord,
    binary_mask,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    PageMasks,
    RegionMasks,
    broad_route_false_positive_pixels,
    decide_bubble_route,
    expand_detector_claim,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    attach_reconstruction_control,
    assert_complete_closure_ledger,
    build_combination_closure_ledger,
    build_factorized_matrix,
    composite_positive_result,
    fill_factorized_mask,
    reconstruction_error,
    score_stage2_page,
    select_pareto_records,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


def _route_fill_backend(
    fill_id: str,
    route_decision: str,
    bubble_route_class: str | None = None,
) -> str:
    normalized = str(fill_id).strip().lower()
    if normalized == "narrow_lama_broad_flat":
        return "robust_flat_median" if route_decision == "broad" else "current_lama"
    if normalized == "conditional_hybrid":
        if route_decision != "broad":
            return "current_lama"
        if bubble_route_class == "clean_flat":
            return "robust_flat_median"
        if bubble_route_class == "clean_gradient":
            return "planar_gradient"
        return "current_lama"
    return normalized


def _authoritative_region_overlap_mask(
    regions: Sequence[RegionMasks],
    shape: tuple[int, int],
) -> np.ndarray:
    seen = np.zeros(shape, np.uint8)
    overlap = np.zeros(shape, np.uint8)
    for region in regions:
        owned = binary_mask(region.ownership, shape)
        overlap[(seen > 0) & (owned > 0)] = 255
        seen[owned > 0] = 255
    return overlap


def _fill_conditional_hybrid_regions(
    source: np.ndarray,
    edit_mask: np.ndarray,
    masks: PageMasks,
    *,
    route_decision: str,
    background_exclude_mask: np.ndarray,
    lama_fill,
    narrow_claim: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    """Fill each authoritative v4 region from one immutable page source."""

    original = np.ascontiguousarray(np.asarray(source)[:, :, :3])
    shape = original.shape[:2]
    edit = binary_mask(edit_mask, shape)
    candidate = original.copy()
    assigned = np.zeros(shape, np.uint8)
    fallback = np.zeros(shape, np.uint8)
    diagnostics: list[dict[str, object]] = []
    region_edits = [
        cv2.bitwise_and(edit, binary_mask(region.ownership, shape))
        for region in masks.regions
    ]
    authoritative_overlap = _authoritative_region_overlap_mask(
        masks.regions, shape
    )
    overlap_conflict = cv2.bitwise_and(edit, authoritative_overlap)
    if narrow_claim is not None:
        narrow = binary_mask(narrow_claim, shape)
        if np.any((overlap_conflict > 0) & (narrow == 0)):
            raise AssertionError(
                "authoritative overlap fallback escaped the narrow detector claim"
            )
    conflict_region_ids = sorted(
        region.region_id
        for region, region_edit in zip(masks.regions, region_edits)
        if np.any((region_edit > 0) & (overlap_conflict > 0))
    )
    fallback[overlap_conflict > 0] = 255
    assigned[overlap_conflict > 0] = 255

    for region, claimed_region_edit in zip(masks.regions, region_edits):
        region_edit = cv2.bitwise_and(
            claimed_region_edit,
            cv2.bitwise_not(overlap_conflict),
        )
        conflict_pixels = int(
            np.count_nonzero(
                (claimed_region_edit > 0) & (overlap_conflict > 0)
            )
        )
        if not np.any(claimed_region_edit):
            continue
        if not np.any(region_edit):
            diagnostics.append(
                {
                    "region_id": region.region_id,
                    "route_class": region.bubble_route_class,
                    "backend": "current_lama",
                    "edit_pixel_count": 0,
                    "overlap_conflict_pixel_count": conflict_pixels,
                    "deferred_page_fallback": True,
                    "fallback_reason": "authoritative_region_overlap",
                }
            )
            continue
        if np.any((assigned > 0) & (region_edit > 0)):
            raise AssertionError(
                "conditional hybrid conflict subtraction left overlapping regions"
            )
        assigned[region_edit > 0] = 255
        backend = _route_fill_backend(
            "conditional_hybrid",
            route_decision,
            region.bubble_route_class,
        )
        if backend in {"current_lama", "ballons_lama"}:
            fallback[region_edit > 0] = 255
            diagnostics.append(
                {
                    "region_id": region.region_id,
                    "route_class": region.bubble_route_class,
                    "backend": "current_lama",
                    "edit_pixel_count": int(np.count_nonzero(region_edit)),
                    "overlap_conflict_pixel_count": conflict_pixels,
                    "deferred_page_fallback": True,
                    "fallback_reason": "region_requires_lama",
                }
            )
            continue
        generated, detail = fill_factorized_mask(
            original,
            region_edit,
            backend=backend,
            interior_mask=region.bubble_interior,
            background_exclude_mask=background_exclude_mask,
        )
        candidate[region_edit > 0] = generated[region_edit > 0]
        diagnostics.append(
            {
                "region_id": region.region_id,
                "route_class": region.bubble_route_class,
                "overlap_conflict_pixel_count": conflict_pixels,
                **detail,
            }
        )

    unassigned = cv2.bitwise_and(edit, cv2.bitwise_not(assigned))
    fallback[unassigned > 0] = 255
    if np.any(fallback):
        generated, detail = fill_factorized_mask(
            original,
            fallback,
            backend="current_lama",
            background_exclude_mask=background_exclude_mask,
            lama_fill=lama_fill,
        )
        candidate[fallback > 0] = generated[fallback > 0]
        fallback_reasons: list[str] = []
        if np.any(overlap_conflict):
            fallback_reasons.append("authoritative_region_overlap")
        if np.any(unassigned):
            fallback_reasons.append("unassigned_edit")
        if any(
            row.get("deferred_page_fallback") is True
            and row.get("fallback_reason") != "authoritative_region_overlap"
            for row in diagnostics
        ):
            fallback_reasons.append("region_requires_lama")
        diagnostics.append(
            {
                "region_id": "__page_fallback__",
                "route_class": "fallback",
                "route_decision": "narrow",
                "fallback_scope": "narrow_page_level",
                "fallback_reasons": fallback_reasons,
                "overlap_conflict_pixel_count": int(
                    np.count_nonzero(overlap_conflict)
                ),
                "overlap_conflict_region_ids": conflict_region_ids,
                **detail,
            }
        )
    if np.any(candidate[edit == 0] != original[edit == 0]):
        raise AssertionError("conditional hybrid changed immutable outside pixels")
    return np.ascontiguousarray(candidate), {
        "backend": "conditional_hybrid",
        "applied": bool(np.any(edit)),
        "edit_pixel_count": int(np.count_nonzero(edit)),
        "region_fills": diagnostics,
        "positive_lama_inference_count": int(np.any(fallback)),
        "overlap_conflict_pixel_count": int(np.count_nonzero(overlap_conflict)),
        "overlap_conflict_region_ids": conflict_region_ids,
        "overlap_conflict_fallback_used": bool(np.any(overlap_conflict)),
        "authoritative_region_overlap_pixel_count": int(
            np.count_nonzero(authoritative_overlap)
        ),
        "authoritative_overlap_edit_pixel_count": int(
            np.count_nonzero(overlap_conflict)
        ),
        "authoritative_overlap_narrow_verified": narrow_claim is not None,
    }


FAMILY = "inpaint-factorized-v3"
CATEGORY = "40-inpaint-mask-render"


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    return _sha256_uncached(path)


def _sha256_uncached(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _logical_inventory_sha256(
    closure_ledger: list[CombinationClosureRecord],
) -> str:
    inventory = sorted(
        (
            {
                "logical_id": row.logical_id,
                "selection": dict(row.selection),
            }
            for row in closure_ledger
        ),
        key=lambda row: row["logical_id"],
    )
    encoded = json.dumps(
        inventory, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _pixel_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def aggregate_factorized_page_statistics(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Rebuild every quality/safety aggregate from canonical per-page facts."""

    if not rows:
        raise ValueError("factorized aggregation requires at least one page row")
    page_ids: list[str] = []
    facts: list[Mapping[str, object]] = []
    for row in rows:
        page_id = str(row.get("page_id") or "")
        fact = row.get("canonical_statistics")
        if not page_id or page_id in page_ids:
            raise ValueError("factorized page statistics require unique page IDs")
        if not isinstance(fact, Mapping) or fact.get("schema_version") != (
            "inpaint-factorized-page-statistics-v1"
        ):
            raise ValueError("factorized page row lacks canonical sufficient statistics")
        if str(fact.get("page_id") or "") != page_id:
            raise ValueError("factorized canonical page identity differs from row")
        if str(row.get("canonical_statistics_sha256") or "") != _canonical_sha256(
            fact
        ):
            raise ValueError("factorized canonical page statistics SHA differs")
        page_ids.append(page_id)
        facts.append(fact)

    def nonnegative_integer(fact: Mapping[str, object], field: str) -> int:
        value = fact.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"factorized page statistics require {field}")
        return value

    def optional_nonnegative_integer(
        fact: Mapping[str, object], field: str
    ) -> int:
        if field not in fact:
            return 0
        return nonnegative_integer(fact, field)

    def optional_finite(fact: Mapping[str, object], field: str) -> float | None:
        value = fact.get(field)
        if value is None:
            return None
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
        ):
            raise ValueError(f"factorized page statistics contain invalid {field}")
        return float(value)

    seed_scores: list[Mapping[str, object]] = []
    edit_scores: list[Mapping[str, object]] = []
    role_counts: dict[str, list[int]] = {}
    for fact in facts:
        current_seed = fact.get("target_instance_seed_scores")
        current_edit = fact.get("target_instance_edit_scores")
        if not isinstance(current_seed, list) or not isinstance(current_edit, list):
            raise ValueError("factorized page statistics lack target instance scores")
        if any(not isinstance(value, Mapping) for value in (*current_seed, *current_edit)):
            raise ValueError("factorized target instance scores must be objects")
        seed_by_id = {str(value.get("instance_id") or ""): value for value in current_seed}
        edit_by_id = {str(value.get("instance_id") or ""): value for value in current_edit}
        if (
            not all(seed_by_id)
            or len(seed_by_id) != len(current_seed)
            or not all(edit_by_id)
            or len(edit_by_id) != len(current_edit)
            or set(seed_by_id) != set(edit_by_id)
        ):
            raise ValueError("factorized target instance score inventory differs")
        for instance_id in sorted(seed_by_id):
            seed = seed_by_id[instance_id]
            edit = edit_by_id[instance_id]
            if not isinstance(seed.get("seeded"), bool):
                raise ValueError("factorized seed score lacks a boolean decision")
            role = str(seed.get("semantic_role") or "").strip()
            coverage = optional_finite(edit, "coverage")
            if not role or coverage is None or not 0.0 <= coverage <= 1.0:
                raise ValueError("factorized instance score is incomplete")
            counts = role_counts.setdefault(role, [0, 0])
            counts[0] += int(bool(seed["seeded"]))
            counts[1] += 1
            seed_scores.append(seed)
            edit_scores.append(edit)

    target_pixels = sum(nonnegative_integer(fact, "target_pixel_count") for fact in facts)
    target_edit = sum(
        nonnegative_integer(fact, "target_edit_pixel_count") for fact in facts
    )
    if target_edit > target_pixels:
        raise ValueError("factorized target edit pixels exceed target inventory")
    residue_sum = 0.0
    baseline_sum = 0.0
    residue_count = 0
    baseline_count = 0
    page_residue_worsened = 0
    reconstruction_values: list[float] = []
    call_durations: list[float] = []
    providers: set[str] = set()
    precisions: set[str] = set()
    for fact in facts:
        contrast = nonnegative_integer(fact, "residue_source_contrast_pixel_count")
        residue = optional_finite(fact, "residue_score")
        baseline = optional_finite(fact, "baseline_residue_score")
        if contrast and residue is not None:
            residue_sum += residue * contrast
            residue_count += contrast
        if contrast and baseline is not None:
            baseline_sum += baseline * contrast
            baseline_count += contrast
        if residue is not None and baseline is not None and residue > baseline + 1e-12:
            page_residue_worsened += 1
        reconstruction = optional_finite(fact, "reconstruction_mse")
        if reconstruction is not None:
            reconstruction_values.append(reconstruction)
        durations = fact.get("positive_lama_call_durations_seconds")
        if not isinstance(durations, list) or any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not np.isfinite(float(value))
            or float(value) < 0.0
            for value in durations
        ):
            raise ValueError("factorized page statistics contain invalid LaMa timings")
        if len(durations) != nonnegative_integer(
            fact, "positive_lama_inference_count"
        ):
            raise ValueError("factorized LaMa timings differ from inference count")
        call_durations.extend(float(value) for value in durations)
        if durations:
            provider = str(fact.get("lama_runtime_provider") or "").strip().lower()
            precision = str(fact.get("lama_runtime_precision") or "").strip().lower()
            if not provider or not precision:
                raise ValueError("factorized LaMa page lacks provider/precision")
            providers.add(provider)
            precisions.add(precision)
    if len(providers) > 1 or len(precisions) > 1:
        raise ValueError("factorized run mixed LaMa providers or precisions")

    inference_counts = [
        nonnegative_integer(fact, "positive_lama_inference_count") for fact in facts
    ]
    if any(value > 1 for value in inference_counts):
        raise ValueError("factorized page exceeded one positive LaMa inference")
    for fact in facts:
        overlap_edit = optional_nonnegative_integer(
            fact, "conditional_hybrid_overlap_conflict_pixel_count"
        )
        if overlap_edit and fact.get("authoritative_overlap_narrow_verified") is not True:
            raise ValueError(
                "factorized authoritative overlap fallback is not narrow-verified"
            )
    runtime_seconds = sum(
        optional_finite(fact, "runtime_seconds") or 0.0 for fact in facts
    )
    if runtime_seconds < 0.0:
        raise ValueError("factorized runtime must not be negative")

    def sum_pixels(field: str) -> int:
        return sum(nonnegative_integer(fact, field) for fact in facts)

    def optional_max(field: str) -> float | None:
        values = [optional_finite(fact, field) for fact in facts]
        present = [value for value in values if value is not None]
        return max(present) if present else None

    output_inventory = [
        {
            "page_id": str(fact["page_id"]),
            "final_mask_pixel_sha256": str(fact.get("final_mask_pixel_sha256") or ""),
        }
        for fact in facts
    ]
    candidate_inventory = [
        {
            "page_id": str(fact["page_id"]),
            "candidate_pixel_sha256": str(fact.get("candidate_pixel_sha256") or ""),
        }
        for fact in facts
    ]
    if any(
        len(str(row[key])) != 64
        for row, key in (
            *((row, "final_mask_pixel_sha256") for row in output_inventory),
            *((row, "candidate_pixel_sha256") for row in candidate_inventory),
        )
    ):
        raise ValueError("factorized page statistics lack output pixel identities")
    return {
        "page_count": len(facts),
        "target_extent_independent": all(
            fact.get("target_extent_independent") is True for fact in facts
        ),
        "target_inventory_independent": all(
            fact.get("target_inventory_independent") is True for fact in facts
        ),
        "target_review_complete": all(
            fact.get("target_review_complete") is True for fact in facts
        ),
        "target_mask_provenance": sorted(
            {str(fact.get("target_mask_provenance") or "") for fact in facts}
        ),
        "target_instance_seed_recall": (
            float(sum(bool(value["seeded"]) for value in seed_scores))
            / float(len(seed_scores))
            if seed_scores
            else None
        ),
        "target_instance_seed_recall_by_semantic_role": {
            role: float(hits) / float(total)
            for role, (hits, total) in sorted(role_counts.items())
        },
        "required_target_instance_count": len(seed_scores),
        "target_pixel_count": target_pixels,
        "missed_target_instance_count": sum(
            not bool(value["seeded"]) for value in seed_scores
        ),
        "aggregate_target_coverage": (
            float(target_edit) / float(target_pixels) if target_pixels else None
        ),
        "minimum_target_instance_coverage": (
            min(float(value["coverage"]) for value in edit_scores)
            if edit_scores
            else None
        ),
        "protected_structure_overlap": sum_pixels(
            "protected_structure_overlap_pixel_count"
        ),
        "protected_structure_changed": sum_pixels(
            "protected_structure_changed_pixel_count"
        ),
        "preserve_edit_overlap": sum_pixels("preserve_edit_overlap_pixel_count"),
        "ambiguous_structure_overlap": sum_pixels(
            "ambiguous_structure_overlap_pixel_count"
        ),
        "ambiguous_structure_changed": sum_pixels(
            "ambiguous_structure_changed_pixel_count"
        ),
        "outside_final_changed": sum_pixels("outside_final_changed_pixel_count"),
        "broad_route_false_positive": sum_pixels(
            "broad_route_false_positive_pixel_count"
        ),
        "conditional_hybrid_overlap_conflict_pixel_count": sum(
            optional_nonnegative_integer(
                fact, "conditional_hybrid_overlap_conflict_pixel_count"
            )
            for fact in facts
        ),
        "conditional_hybrid_overlap_fallback_page_count": sum(
            int(
                optional_nonnegative_integer(
                    fact, "conditional_hybrid_overlap_conflict_pixel_count"
                )
                > 0
            )
            for fact in facts
        ),
        "authoritative_region_overlap_pixel_count": sum(
            optional_nonnegative_integer(
                fact, "authoritative_region_overlap_pixel_count"
            )
            for fact in facts
        ),
        "no_edit_false_edit": sum(
            nonnegative_integer(fact, "edit_pixel_count")
            for fact in facts
            if fact.get("no_edit") is True
        ),
        "required_skip_count": sum(
            int(fact.get("required_skip") is True) for fact in facts
        ),
        "page_residue_worsened_count": page_residue_worsened,
        "aggregate_residue_score": (
            residue_sum / float(residue_count) if residue_count else None
        ),
        "baseline_aggregate_residue_score": (
            baseline_sum / float(baseline_count) if baseline_count else None
        ),
        "reconstruction_mse": (
            float(np.mean(reconstruction_values)) if reconstruction_values else None
        ),
        "runtime_seconds": runtime_seconds,
        "positive_lama_inference_count": sum(inference_counts),
        "maximum_positive_lama_inference_per_page": max(inference_counts),
        "residue_gate_applicable": any(
            fact.get("residue_gate_applicable") is True for fact in facts
        ),
        "reconstruction_gate_applicable": bool(reconstruction_values),
        "runtime_telemetry_complete": all(
            fact.get("runtime_telemetry_complete") is True for fact in facts
        ),
        "positive_lama_runtime_p95_seconds": (
            float(np.percentile(np.asarray(call_durations, np.float64), 95.0))
            if call_durations
            else None
        ),
        "peak_vram_allocated_mib": optional_max("peak_vram_allocated_mib"),
        "peak_vram_reserved_mib": optional_max("peak_vram_reserved_mib"),
        "cpu_fallback_count": sum_pixels("cpu_fallback_count"),
        "lama_runtime_provider": next(iter(providers), ""),
        "lama_runtime_precision": next(iter(precisions), ""),
        "output_mask_set_sha256": _canonical_sha256(
            sorted(output_inventory, key=lambda value: value["page_id"])
        ),
        "candidate_image_set_sha256": _canonical_sha256(
            sorted(candidate_inventory, key=lambda value: value["page_id"])
        ),
    }


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _read_written_output(path: Path, *, role: str) -> np.ndarray:
    mode = cv2.IMREAD_COLOR if role == "candidate_image" else cv2.IMREAD_GRAYSCALE
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if value is None or value.size == 0:
        raise ValueError(f"factorized output artifact is unreadable: {path}")
    if role != "candidate_image" and not set(np.unique(value)).issubset({0, 255}):
        raise ValueError(f"factorized output mask is not strict binary: {path}")
    return np.ascontiguousarray(value)


def _seal_factorized_output_inventory(
    output_root: Path,
    records: Sequence[FactorizedRunRecord],
    page_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, object], frozenset[str]]:
    """Bind written finalist-capable bytes to their canonical page identities."""

    artifacts: list[dict[str, object]] = []
    complete_run_ids: list[str] = []
    for record in records:
        rows = page_rows.get(record.run_id, ())
        expected: list[tuple[str, str, Path, str]] = []
        for row in rows:
            page_id = str(row.get("page_id") or "")
            canonical = row.get("canonical_statistics")
            if not page_id or not isinstance(canonical, Mapping):
                raise ValueError("factorized output inventory lacks canonical page facts")
            run_root = output_root / "runs" / record.run_id
            expected.extend(
                (
                    (
                        page_id,
                        "detector_seed_mask",
                        run_root / "detector_seed_masks" / f"{page_id}.png",
                        str(canonical.get("detector_seed_mask_pixel_sha256") or ""),
                    ),
                    (
                        page_id,
                        "edit_mask",
                        run_root / "edit_masks" / f"{page_id}.png",
                        str(canonical.get("output_edit_mask_pixel_sha256") or ""),
                    ),
                    (
                        page_id,
                        "final_mask",
                        run_root / "final_masks" / f"{page_id}.png",
                        str(canonical.get("final_mask_pixel_sha256") or ""),
                    ),
                    (
                        page_id,
                        "candidate_image",
                        run_root / "candidate_images" / f"{page_id}.png",
                        str(canonical.get("candidate_pixel_sha256") or ""),
                    ),
                )
            )
        if not expected or any(not path.is_file() for _page, _role, path, _sha in expected):
            continue
        run_artifacts: list[dict[str, object]] = []
        expected_shapes: dict[str, tuple[int, int]] = {}
        for page_id, role, path, expected_pixel_sha in expected:
            decoded = _read_written_output(path, role=role)
            pixel_sha = _pixel_sha256(decoded)
            if pixel_sha != expected_pixel_sha:
                raise ValueError(
                    "factorized written output pixel SHA differs from canonical page facts"
                )
            shape = decoded.shape[:2]
            previous_shape = expected_shapes.setdefault(page_id, shape)
            if previous_shape != shape:
                raise ValueError("factorized written output shapes differ within a page")
            relative = path.resolve().relative_to(output_root.resolve()).as_posix()
            artifact: dict[str, object] = {
                "run_id": record.run_id,
                "page_id": page_id,
                "role": role,
                "relative_path": relative,
                "artifact_sha256": _sha256(path),
                "pixel_sha256": pixel_sha,
                "shape": list(decoded.shape),
                "dtype": str(decoded.dtype),
            }
            if role != "candidate_image":
                artifact["foreground_pixel_count"] = int(np.count_nonzero(decoded))
            run_artifacts.append(artifact)
        artifacts.extend(run_artifacts)
        complete_run_ids.append(record.run_id)

    artifacts.sort(
        key=lambda row: (
            str(row["run_id"]),
            str(row["page_id"]),
            str(row["role"]),
        )
    )
    complete_run_ids.sort()
    canonical = {
        "records": artifacts,
        "complete_run_ids": complete_run_ids,
    }
    inventory = {
        "schema_version": "inpaint-factorized-output-artifact-inventory-v1",
        **canonical,
        "inventory_sha256": _canonical_sha256(canonical),
    }
    path = output_root / "factorized-output-artifact-inventory.json"
    _write_json(path, inventory)
    return (
        {
            "relative_path": path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(path),
            "inventory_sha256": inventory["inventory_sha256"],
            "artifact_count": len(artifacts),
            "complete_run_ids": complete_run_ids,
        },
        frozenset(complete_run_ids),
    )


_RUNTIME_PAGE_FIELDS = (
    "runtime_seconds",
    "positive_lama_inference_count",
    "positive_lama_call_durations_seconds",
    "runtime_telemetry_complete",
    "cpu_fallback_count",
    "lama_runtime_provider",
    "lama_runtime_precision",
    "peak_vram_allocated_mib",
    "peak_vram_reserved_mib",
)

_RUNTIME_AGGREGATE_FIELDS = (
    "runtime_seconds",
    "positive_lama_inference_count",
    "maximum_positive_lama_inference_per_page",
    "runtime_telemetry_complete",
    "positive_lama_runtime_p95_seconds",
    "peak_vram_allocated_mib",
    "peak_vram_reserved_mib",
    "cpu_fallback_count",
    "lama_runtime_provider",
    "lama_runtime_precision",
)


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _resolve_lama_model_path() -> Path:
    from modules.utils.download import ModelDownloader, ModelID

    return Path(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX))


def _runtime_identity(
    *,
    device: str,
    precision: str,
    inpaint_size: int,
    lama_model_path: Path | None = None,
) -> dict[str, object]:
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT
    )
    tracked_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"], cwd=ROOT
    )
    from modules.utils.download import ModelDownloader, ModelID

    model_path = lama_model_path or _resolve_lama_model_path()
    model_spec = ModelDownloader.registry[ModelID.LAMA_LARGE_512PX]
    expected_model_sha = str(model_spec.sha256[0] or "")
    identity: dict[str, object] = {
        "code_commit": _git_head(),
        "tracked_worktree_clean": not bool(tracked_status.strip()),
        "tracked_worktree_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "requested_device": str(device),
        "requested_precision": str(precision),
        "inpaint_size": int(inpaint_size),
        "vram_measurement_scope": "page_reset_then_run_max",
        "lama_model_asset_id": "lama_large_512px",
        "lama_model_registry_sha256": expected_model_sha,
        "lama_model_present": model_path.is_file(),
        "lama_model_sha256": _sha256(model_path) if model_path.is_file() else "",
    }
    try:
        import torch

        identity.update(
            {
                "torch_version": str(torch.__version__),
                "torch_cuda_version": str(torch.version.cuda or ""),
                "cudnn_version": (
                    int(torch.backends.cudnn.version())
                    if torch.backends.cudnn.version() is not None
                    else None
                ),
                "cuda_available": bool(torch.cuda.is_available()),
                "gpu_name": (
                    str(torch.cuda.get_device_name(torch.cuda.current_device()))
                    if str(device).lower().startswith("cuda")
                    and torch.cuda.is_available()
                    else ""
                ),
            }
        )
    except (ImportError, RuntimeError):
        identity.update(
            {
                "torch_version": "",
                "torch_cuda_version": "",
                "cudnn_version": None,
                "cuda_available": False,
                "gpu_name": "",
            }
        )
    return identity


def _seal_runtime_source_inventory(
    output_root: Path,
    *,
    runtime_identity: Mapping[str, object],
    lama_model_path: Path,
) -> dict[str, object]:
    """Persist the exact runner/diff bytes and model pre/post identity used."""

    source_root = output_root / "runtime-source"
    source_root.mkdir(parents=True, exist_ok=True)
    runner_path = Path(__file__).resolve()
    runner_bytes = runner_path.read_bytes()
    runner_snapshot = source_root / "benchmark_inpaint_factorized_v3.py"
    runner_snapshot.write_bytes(runner_bytes)
    runner_relative = runner_path.relative_to(ROOT).as_posix()
    patch_bytes = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--"], cwd=ROOT
    )
    patch_path = source_root / "tracked-diff.patch"
    patch_path.write_bytes(patch_bytes)
    runner_patch_bytes = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD", "--", runner_relative], cwd=ROOT
    )
    runner_patch_path = source_root / "runner-diff.patch"
    runner_patch_path.write_bytes(runner_patch_bytes)

    from modules.utils.download import ModelDownloader, ModelID

    model_path = lama_model_path
    actual_post_sha = _sha256_uncached(model_path) if model_path.is_file() else ""
    expected_sha = str(
        ModelDownloader.registry[ModelID.LAMA_LARGE_512PX].sha256[0] or ""
    )
    actual_pre_sha = str(runtime_identity.get("lama_model_sha256") or "")
    records = [
        {
            "role": "runner_source_snapshot",
            "source_path": runner_relative,
            "relative_path": runner_snapshot.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(runner_snapshot),
            "source_bytes_sha256": hashlib.sha256(runner_bytes).hexdigest(),
        },
        {
            "role": "tracked_diff_patch",
            "relative_path": patch_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(patch_path),
            "byte_count": len(patch_bytes),
        },
        {
            "role": "runner_diff_patch",
            "source_path": runner_relative,
            "relative_path": runner_patch_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(runner_patch_path),
            "byte_count": len(runner_patch_bytes),
        },
    ]
    canonical = {
        "code_commit": str(runtime_identity["code_commit"]),
        "tracked_worktree_clean": len(patch_bytes) == 0,
        "records": records,
        "lama_model": {
            "asset_id": "lama_large_512px",
            "registry_expected_sha256": expected_sha,
            "actual_pre_sha256": actual_pre_sha,
            "actual_post_sha256": actual_post_sha,
            "present_pre": bool(runtime_identity.get("lama_model_present")),
            "present_post": model_path.is_file(),
        },
    }
    inventory = {
        "schema_version": "inpaint-factorized-runtime-source-inventory-v1",
        **canonical,
        "inventory_sha256": _canonical_sha256(canonical),
    }
    path = output_root / "factorized-runtime-source-inventory.json"
    _write_json(path, inventory)
    return {
        "role": "runtime_source_inventory",
        "relative_path": path.relative_to(output_root).as_posix(),
        "artifact_sha256": _sha256(path),
        "inventory_sha256": inventory["inventory_sha256"],
    }


def _seal_factorized_runtime_evidence(
    output_root: Path,
    records: Sequence[FactorizedRunRecord],
    page_rows: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    runtime_identity: Mapping[str, object],
    runtime_source_inventory: Mapping[str, object],
    total_inference_count: int,
) -> tuple[dict[str, object], frozenset[str]]:
    """Seal measured per-call/page telemetry independently from result JSON."""

    run_records: list[dict[str, object]] = []
    complete_run_ids: list[str] = []
    for record in records:
        rows = page_rows.get(record.run_id, ())
        pages: list[dict[str, object]] = []
        complete = bool(rows)
        for row in rows:
            canonical = row.get("canonical_statistics")
            runtime = row.get("_runtime_evidence")
            if not isinstance(canonical, Mapping) or not isinstance(runtime, Mapping):
                complete = False
                continue
            page_id = str(row.get("page_id") or "")
            if not page_id or runtime.get("page_id") != page_id:
                complete = False
                continue
            summary = {field: canonical.get(field) for field in _RUNTIME_PAGE_FIELDS}
            if runtime.get("summary") != summary:
                raise ValueError("factorized runtime evidence differs from canonical page telemetry")
            events = runtime.get("inference_events")
            if not isinstance(events, list):
                complete = False
                continue
            pages.append(
                {
                    "page_id": page_id,
                    "summary": summary,
                    "inference_events": events,
                }
            )
        aggregate = {
            field: record.metrics.get(field) for field in _RUNTIME_AGGREGATE_FIELDS
        }
        run_records.append(
            {
                "run_id": record.run_id,
                "runtime_identity": dict(runtime_identity),
                "pages": pages,
                "aggregate": aggregate,
            }
        )
        if complete and len(pages) == len(rows):
            complete_run_ids.append(record.run_id)

    run_records.sort(key=lambda row: str(row["run_id"]))
    complete_run_ids.sort()
    canonical = {
        "runtime_identity": dict(runtime_identity),
        "runtime_source_inventory": dict(runtime_source_inventory),
        "runs": run_records,
        "complete_run_ids": complete_run_ids,
        "positive_lama_inference_count": int(total_inference_count),
    }
    ledger = {
        "schema_version": "inpaint-factorized-runtime-evidence-v1",
        **canonical,
        "ledger_sha256": _canonical_sha256(canonical),
    }
    path = output_root / "factorized-runtime-evidence.json"
    _write_json(path, ledger)
    for rows in page_rows.values():
        for row in rows:
            if isinstance(row, dict):
                row.pop("_runtime_evidence", None)
    return (
        {
            "role": "runtime_evidence",
            "relative_path": path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(path),
            "ledger_sha256": ledger["ledger_sha256"],
            "runtime_source_inventory_sha256": runtime_source_inventory[
                "inventory_sha256"
            ],
            "complete_run_ids": complete_run_ids,
            "positive_lama_inference_count": int(total_inference_count),
        },
        frozenset(complete_run_ids),
    )


def _downgrade_unsealed_factorized_finalists(
    records: Sequence[FactorizedRunRecord],
    complete_output_run_ids: frozenset[str],
    complete_runtime_run_ids: frozenset[str],
) -> list[FactorizedRunRecord]:
    downgraded: list[FactorizedRunRecord] = []
    for row in records:
        if row.status not in {"pareto", "family_complete"}:
            downgraded.append(row)
        elif row.run_id not in complete_output_run_ids:
            downgraded.append(
                replace(
                    row,
                    status="information_limited",
                    closure_reason="output_artifact_inventory_missing",
                )
            )
        elif row.run_id not in complete_runtime_run_ids:
            downgraded.append(
                replace(
                    row,
                    status="information_limited",
                    closure_reason="runtime_evidence_ledger_missing",
                )
            )
        else:
            downgraded.append(row)
    return downgraded


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _read_image(
    path: str | Path,
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    key = str(path)
    if cache is not None and key in cache:
        return cache[key]
    image = cv2.imdecode(np.fromfile(key, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    if cache is not None:
        cache[key] = image
    return image


def _read_mask(
    path: object,
    shape: tuple[int, int],
    cache: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    value = _path_value(path)
    if value is None:
        return np.zeros(shape, dtype=np.uint8)
    if cache is not None and value in cache:
        mask = cache[value]
        if mask.shape != shape:
            raise ValueError(f"mask shape mismatch: {mask.shape} != {shape}")
        return mask
    mask = cv2.imdecode(np.fromfile(value, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(value)
    normalized = binary_mask(mask, shape)
    if cache is not None:
        cache[value] = normalized
    return normalized


def _sparse_mask_indices(
    path: str,
    shape: tuple[int, int],
    cache: dict[str, np.ndarray],
) -> np.ndarray:
    cached = cache.get(path)
    if cached is not None:
        return cached
    mask = _read_mask(path, shape, None)
    indices = np.flatnonzero(mask.reshape(-1)).astype(np.int64, copy=False)
    if indices.size == 0:
        raise ValueError(f"target instance mask is empty: {path}")
    cache[path] = indices
    return indices


def _page_artifact(
    family: dict[str, object],
    page_id: str,
    key: str,
    fallback_key: str | None = None,
) -> object:
    pages = family.get("pages")
    if not isinstance(pages, dict):
        raise ValueError("matrix family must contain a pages object")
    page = pages.get(page_id)
    if not isinstance(page, dict):
        raise ValueError(f"matrix family is missing page {page_id}")
    if key not in page and fallback_key is not None:
        key = fallback_key
    if key not in page:
        raise ValueError(f"matrix page {page_id} is missing artifact {key}")
    return page[key]


def _entries(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries: dict[str, dict[str, object]] = {}
    for entry in payload.get("pages", []):
        if not isinstance(entry, dict):
            continue
        normalized = dict(entry)
        for field in (
            "existing_source_edit_mask",
            "baseline",
            "baseline_mask",
            "known_background",
        ):
            value = _path_value(normalized.get(field))
            if value is None:
                continue
            artifact = Path(value)
            if not artifact.is_absolute():
                artifact = path.parent / artifact
            normalized[field] = str(artifact.resolve())
        entries[str(entry.get("page_id") or "")] = normalized
    return entries


def _with_candidate_ownership(
    masks: PageMasks,
    ownership: np.ndarray,
    broad_ownership: np.ndarray,
    interior: np.ndarray,
) -> PageMasks:
    return PageMasks(
        masks.target,
        masks.protected,
        masks.ambiguous,
        ownership,
        masks.claim_seed,
        masks.existing_edit,
        masks.target_instances,
        interior,
        masks.corner,
        broad_ownership,
        masks.preserve,
        masks.regions,
    )


def _annotation_masks(
    page,
    entry: dict[str, object],
    shape: tuple[int, int],
    cache: dict[str, np.ndarray],
    *,
    sparse_evidence: bool = False,
) -> PageMasks:
    target = _read_mask(page.target_text_mask, shape, cache)
    protected = _read_mask(page.protected_structure_mask, shape, cache)
    ambiguous = _read_mask(page.ambiguous_structure_mask, shape, cache)
    ownership = (
        _read_mask(page.ownership_mask, shape, cache)
        if page.ownership_mask
        else np.full(shape, 255, np.uint8)
    )
    claim_seed = (
        _read_mask(page.claim_seed_mask, shape, cache)
        if page.claim_seed_mask
        else np.full(shape, 255, np.uint8)
    )
    existing_path = _path_value(
        entry.get("existing_source_edit_mask", entry.get("baseline_mask"))
    )
    existing = _read_mask(existing_path, shape, cache)
    interior = _read_mask(page.bubble_interior_mask, shape, cache)
    corner = _read_mask(page.corner_protect_mask, shape, cache)
    preserve = _read_mask(page.preserve_mask, shape, cache)
    instances = () if sparse_evidence else tuple(
        (record.instance_id, _read_mask(record.mask_path, shape, cache))
        for record in page.target_instances
        if record.priority == "required"
    )
    # Region and target-instance artifacts are full-page PNGs in the sealed
    # source manifest.  Keeping hundreds of those arrays resident turns E1
    # evaluation into multi-gigabyte work.  The full helper contract remains
    # the default; the matrix runner opts into sparse streaming explicitly.
    regions = () if sparse_evidence else tuple(
        RegionMasks(
            region.region_id,
            region.bubble_route_class,
            _read_mask(region.bubble_interior_mask, shape, cache),
            _read_mask(region.ownership_mask, shape, cache),
            _read_mask(region.protected_structure_mask, shape, cache),
            _read_mask(region.ambiguous_structure_mask, shape, cache),
            _read_mask(region.corner_protect_mask, shape, cache),
        )
        for region in page.regions
    )
    return PageMasks(
        target,
        protected,
        ambiguous,
        ownership,
        claim_seed,
        existing,
        instances,
        interior,
        corner,
        ownership,
        preserve,
        regions,
    )


def _component_coverages(target: np.ndarray, mask: np.ndarray) -> list[float]:
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
        (target > 0).astype(np.uint8), 8
    )
    values: list[float] = []
    for index in range(1, count):
        x, y, width, height, area = (int(value) for value in stats[index])
        if area <= 0:
            continue
        component = labels[y:y + height, x:x + width] == index
        local = mask[y:y + height, x:x + width] > 0
        values.append(float(np.count_nonzero(component & local)) / float(area))
    return values


def _score_cached_mask_only(
    page_id: str,
    source: np.ndarray,
    baseline: np.ndarray,
    final_mask: np.ndarray,
    masks: PageMasks,
    cache: dict[str, dict[str, object]],
) -> tuple[dict[str, object], np.ndarray]:
    cached = cache.get(page_id)
    if cached is None:
        invariant, _changed = score_stage2_page(
            source,
            baseline,
            np.zeros(source.shape[:2], np.uint8),
            masks,
            baseline=baseline,
        )
        cache[page_id] = invariant
    else:
        invariant = cached
    # A mask-only run never changes pixels.  Do not retain one full-page zero
    # array per corpus page merely to report that invariant across candidates.
    changed = np.zeros(source.shape[:2], np.uint8)
    detector = binary_mask(final_mask, source.shape[:2])
    metrics = dict(invariant)
    covered = int(np.count_nonzero((masks.target > 0) & (detector > 0)))
    target_pixels = int(np.count_nonzero(masks.target))
    coverages = _component_coverages(masks.target, detector)
    metrics.update(
        {
            "detector_mask_pixel_count": int(np.count_nonzero(detector)),
            "changed_outside_detector_mask_pixel_count": int(
                np.count_nonzero((changed > 0) & (detector == 0))
            ),
            "target_detector_covered_pixel_count": covered,
            "target_detector_coverage": (
                float(covered) / float(target_pixels) if target_pixels else None
            ),
            "target_component_coverages": coverages,
            "minimum_target_component_coverage": (
                min(coverages) if coverages else None
            ),
        }
    )
    return metrics, changed


class _LamaPool:
    def __init__(self, *, device: str, precision: str, inpaint_size: int) -> None:
        self.device = device
        self.precision = precision
        self.inpaint_size = int(inpaint_size)
        self._model: Any | None = None
        self.call_count = 0
        self.call_durations: list[float] = []

    def begin_page_runtime_scope(self) -> None:
        if not str(self.device).lower().startswith("cuda"):
            return
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA runtime scope requested without CUDA availability")
        torch.cuda.reset_peak_memory_stats()

    def fill(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        if self._model is None:
            from modules.inpainting.source_lama_blockwise import SourceLaMaLarge

            self._model = SourceLaMaLarge(
                device=self.device,
                precision=self.precision,
                inpaint_size=self.inpaint_size,
            )
            self._model.ensure_loaded()
        generated_rgb = self._model.memory_safe_inpaint(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            mask,
        )
        self.call_count += 1
        self.call_durations.append(time.perf_counter() - started)
        return cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)

    def runtime_events_since(
        self, call_index: int, *, backend: str
    ) -> list[dict[str, object]]:
        start = int(call_index)
        durations = self.call_durations[start:]
        diagnostics = (
            list(getattr(self._model, "run_diagnostics", []) or [])[-len(durations) :]
            if self._model is not None and durations
            else []
        )
        events: list[dict[str, object]] = []
        for offset, duration in enumerate(durations):
            diagnostic = (
                diagnostics[offset]
                if offset < len(diagnostics) and isinstance(diagnostics[offset], dict)
                else {}
            )
            events.append(
                {
                    "call_index": start + offset + 1,
                    "backend": str(backend),
                    "provider": str(diagnostic.get("actual_device") or ""),
                    "precision": str(diagnostic.get("actual_precision") or ""),
                    "cpu_fallback": bool(
                        diagnostic.get("cpu_fallback_used", False)
                    ),
                    "duration_seconds": float(duration),
                }
            )
        return events

    def runtime_metrics_since(self, call_index: int) -> dict[str, object]:
        events = self.runtime_events_since(call_index, backend="current_lama")
        durations = [float(row["duration_seconds"]) for row in events]
        providers = {str(row["provider"]) for row in events if row["provider"]}
        precisions = {str(row["precision"]) for row in events if row["precision"]}
        telemetry_complete = not events or (
            len(providers) == 1 and len(precisions) == 1
        )
        peak_allocated = peak_reserved = None
        if durations and str(self.device).lower().startswith("cuda"):
            import torch

            peak_allocated = float(torch.cuda.max_memory_allocated()) / (1024.0**2)
            peak_reserved = float(torch.cuda.max_memory_reserved()) / (1024.0**2)
        return {
            "runtime_telemetry_complete": telemetry_complete,
            "positive_lama_runtime_p95_seconds": (
                float(np.percentile(np.asarray(durations, np.float64), 95.0))
                if durations
                else None
            ),
            "peak_vram_allocated_mib": peak_allocated,
            "peak_vram_reserved_mib": peak_reserved,
            "cpu_fallback_count": sum(bool(row["cpu_fallback"]) for row in events),
            "lama_runtime_provider": next(iter(providers), ""),
            "lama_runtime_precision": next(iter(precisions), ""),
        }


def _run_combination(
    combination: dict[str, str],
    *,
    matrix: dict[str, object],
    pages,
    entries: dict[str, dict[str, object]],
    output_root: Path,
    lama_pool: _LamaPool,
    instance_index_cache: dict[str, np.ndarray],
    mask_only_score_cache: dict[str, dict[str, object]],
) -> tuple[FactorizedRunRecord, list[dict[str, object]]]:
    families = matrix.get("families")
    if not isinstance(families, dict):
        raise ValueError("matrix spec must contain families")
    detector = families["detector"][combination["detector"]]
    ownership_family = families["ownership"][combination["ownership"]]
    silhouette = families["silhouette"][combination["silhouette"]]
    router = families["router"][combination["router"]]
    expansion_id = combination["expansion"]
    fill_id = combination["fill"]
    if not all(
        isinstance(value, dict)
        for value in (detector, ownership_family, silhouette, router)
    ):
        raise ValueError("matrix family entries must be objects")
    run_id = "__".join(combination[role] for role in combination)
    run_root = output_root / "runs" / run_id
    started = time.perf_counter()
    rows: list[dict[str, object]] = []
    total_target = 0
    total_target_edit = 0
    seed_total = 0
    seed_hits = 0
    instance_coverages: list[float] = []
    role_seed_counts: dict[str, list[int]] = {}
    residue_sum = 0.0
    residue_count = 0
    baseline_sum = 0.0
    baseline_count = 0
    reconstruction_values: list[float] = []
    protected_changed = 0
    protected_overlap = 0
    preserve_overlap = 0
    ambiguous_changed = 0
    ambiguous_overlap = 0
    outside_changed = 0
    no_edit_false = 0
    broad_false = 0
    required_skips = 0
    page_residue_worsened = 0
    calls_before = lama_pool.call_count

    for page in pages:
        page_started = time.perf_counter()
        begin_runtime_scope = getattr(lama_pool, "begin_page_runtime_scope", None)
        if callable(begin_runtime_scope):
            begin_runtime_scope()
        page_calls_before = lama_pool.call_count
        # E1 contains ~400 MiB for each full corpus mask.  A process-wide
        # artifact cache held several detector/ownership/silhouette families
        # plus annotations simultaneously and grew past 8 GiB.  Keep only one
        # page's decoded images and masks alive; the OS file cache still makes
        # repeated combinations inexpensive without retaining NumPy arrays.
        image_cache: dict[str, np.ndarray] = {}
        mask_cache: dict[str, np.ndarray] = {}
        entry = entries[page.page_id]
        if fill_id == "mask_only":
            height = int(entry.get("height") or 0)
            width = int(entry.get("width") or 0)
            if height <= 0 or width <= 0:
                source = _read_image(page.source_image, image_cache)
                shape = source.shape[:2]
            else:
                shape = (height, width)
                source = np.zeros((height, width, 3), np.uint8)
        else:
            source = _read_image(page.source_image, image_cache)
            shape = source.shape[:2]
        annotation = _annotation_masks(
            page,
            entry,
            shape,
            mask_cache,
            sparse_evidence=fill_id != "conditional_hybrid",
        )
        ownership = _read_mask(
            _page_artifact(ownership_family, page.page_id, "mask"), shape, mask_cache
        )
        broad_ownership = _read_mask(
            _page_artifact(
                ownership_family, page.page_id, "broad_mask", "mask"
            ),
            shape,
            mask_cache,
        )
        interior = _read_mask(
            _page_artifact(silhouette, page.page_id, "interior"), shape, mask_cache
        )
        masks = _with_candidate_ownership(
            annotation, ownership, broad_ownership, interior
        )
        raw = _read_mask(
            _page_artifact(detector, page.page_id, "raw"), shape, mask_cache
        )
        refined = _read_mask(
            _page_artifact(detector, page.page_id, "refined"), shape, mask_cache
        )
        dilated = _read_mask(
            _page_artifact(detector, page.page_id, "dilated"), shape, mask_cache
        )
        seed_variant = str(detector.get("seed_variant") or "raw")
        detector_seed = {
            "raw": raw,
            "refined": refined,
            "dilated": dilated,
        }[seed_variant]
        route_seed = cv2.bitwise_and(detector_seed, ownership)
        content = None
        if expansion_id == "content_component":
            content = _read_mask(
                _page_artifact(ownership_family, page.page_id, "content_components"),
                shape,
                mask_cache,
            )
        broad = expand_detector_claim(
            expansion_id,
            seed=route_seed,
            raw=raw,
            refined=refined,
            dilated=dilated,
            content_components=content,
            bubble_interior=interior,
        )
        router_pages = router.get("pages", {})
        evidence = router_pages.get(page.page_id, {}) if isinstance(router_pages, dict) else {}
        if not isinstance(evidence, dict):
            raise ValueError("router page evidence must be an object")
        route_masks = {
            key: (
                _read_mask(evidence.get(path_key), shape, mask_cache)
                if evidence.get(path_key)
                else None
            )
            for key, path_key in (
                ("ballons_clean_mask", "ballons_clean_mask"),
                ("pr2_clean_mask", "pr2_clean_mask"),
                ("segmentation_valid_mask", "segmentation_valid_mask"),
                ("unsafe_signal_mask", "unsafe_signal_mask"),
            )
        }
        authoritative_overlap = _authoritative_region_overlap_mask(
            masks.regions, shape
        )
        if fill_id == "conditional_hybrid" and np.any(authoritative_overlap):
            existing_unsafe = route_masks.get("unsafe_signal_mask")
            route_masks["unsafe_signal_mask"] = (
                authoritative_overlap
                if existing_unsafe is None
                else cv2.bitwise_or(existing_unsafe, authoritative_overlap)
            )
        exclude = cv2.bitwise_or(masks.protected, masks.ambiguous)
        if masks.corner is not None:
            exclude = cv2.bitwise_or(exclude, masks.corner)
        if masks.preserve is not None:
            exclude = cv2.bitwise_or(exclude, masks.preserve)
        background_samples = int(
            np.count_nonzero(
                (interior > 0)
                & (route_seed == 0)
                & (exclude == 0)
                & (
                    (masks.broad_ownership > 0)
                    if masks.broad_ownership is not None
                    else (masks.ownership > 0)
                )
            )
        )
        decision = decide_bubble_route(
            str(router.get("algorithm") or combination["router"]),
            narrow_claim=detector_seed,
            broad_claim=broad,
            seed=route_seed,
            masks=masks,
            ballons_clean=bool(evidence.get("ballons_clean", False)),
            pr2_clean=bool(evidence.get("pr2_clean", False)),
            segmentation_valid=bool(evidence.get("segmentation_valid", False)),
            texture=bool(evidence.get("texture", False)),
            microtexture=bool(evidence.get("microtexture", False)),
            line_art=bool(evidence.get("line_art", False)),
            ambiguous=bool(evidence.get("ambiguous", False)),
            background_sample_count=background_samples,
            minimum_background_samples=int(router.get("minimum_background_samples", 32)),
            **route_masks,
        )
        clean_annotation = page.bubble_route_class in {"clean_flat", "clean_gradient"}
        broad_only = cv2.bitwise_and(
            decision.edit_mask,
            cv2.bitwise_not(detector_seed),
        )
        if page.regions:
            source_clean = np.zeros(shape, np.uint8)
            for region in masks.regions:
                if region.bubble_route_class in {"clean_flat", "clean_gradient"}:
                    source_clean[region.bubble_interior > 0] = 255
            broad_route_false_pixels = int(
                np.count_nonzero((broad_only > 0) & (source_clean == 0))
            )
            broad_route_false = broad_route_false_pixels > 0
        else:
            broad_route_false_pixels = int(np.count_nonzero(broad_only)) if (
                decision.decision == "broad" and not clean_annotation
            ) else 0
            broad_route_false = broad_route_false_pixels > 0
        broad_false += broad_route_false_pixels
        if decision.decision == "skip" and not page.no_edit:
            required_skips += 1

        seed_scores = []
        edit_scores = []
        flat_seed = detector_seed.reshape(-1)
        flat_edit = decision.edit_mask.reshape(-1)
        for instance_record in page.target_instances:
            if instance_record.priority != "required":
                continue
            instance_id = instance_record.instance_id
            indices = _sparse_mask_indices(
                instance_record.mask_path, shape, instance_index_cache
            )
            pixels = int(indices.size)
            seed_covered = int(np.count_nonzero(flat_seed[indices]))
            edit_covered = int(np.count_nonzero(flat_edit[indices]))
            seed_scores.append(
                {
                    "instance_id": instance_id,
                    "semantic_role": str(instance_record.semantic_role),
                    "seeded": seed_covered > 0,
                }
            )
            coverage = float(edit_covered) / float(pixels) if pixels else 0.0
            edit_scores.append({"instance_id": instance_id, "coverage": coverage})
            seed_total += 1
            seed_hits += int(seed_covered > 0)
            role_counts = role_seed_counts.setdefault(
                str(instance_record.semantic_role), [0, 0]
            )
            role_counts[0] += int(seed_covered > 0)
            role_counts[1] += 1
            instance_coverages.append(coverage)

        baseline_path = _path_value(entry.get("baseline"))
        baseline_mask_path = _path_value(entry.get("baseline_mask"))
        baseline = (
            _read_image(baseline_path, image_cache) if baseline_path else source.copy()
        )
        baseline_mask = (
            _read_mask(baseline_mask_path, shape, mask_cache)
            if baseline_mask_path
            else np.zeros(shape, dtype=np.uint8)
        )
        if fill_id == "mask_only":
            candidate = baseline.copy()
            final_mask = cv2.bitwise_or(baseline_mask, decision.edit_mask)
            fill_diagnostics = {"backend": fill_id, "applied": False}
        else:
            if fill_id == "conditional_hybrid" and masks.regions:
                generated, fill_diagnostics = _fill_conditional_hybrid_regions(
                    source,
                    decision.edit_mask,
                    masks,
                    route_decision=decision.decision,
                    background_exclude_mask=exclude,
                    lama_fill=lama_pool.fill,
                    narrow_claim=detector_seed,
                )
            else:
                selected_fill = _route_fill_backend(
                    fill_id,
                    decision.decision,
                    page.bubble_route_class,
                )
                callback = (
                    lama_pool.fill
                    if selected_fill in {"current_lama", "ballons_lama"}
                    else None
                )
                generated, fill_diagnostics = fill_factorized_mask(
                    source,
                    decision.edit_mask,
                    backend=selected_fill,
                    interior_mask=interior,
                    background_exclude_mask=exclude,
                    background_sample_edit_mask=(
                        route_seed if decision.decision == "broad" else None
                    ),
                    lama_fill=callback,
                )
                if selected_fill != fill_id:
                    fill_diagnostics["requested_backend"] = fill_id
            candidate, final_mask = composite_positive_result(
                baseline,
                generated,
                decision.edit_mask,
                baseline_mask,
            )
        if fill_id == "mask_only":
            metrics, changed = _score_cached_mask_only(
                page.page_id,
                source,
                baseline,
                final_mask,
                masks,
                mask_only_score_cache,
            )
        else:
            metrics, changed = score_stage2_page(
                source,
                candidate,
                final_mask,
                masks,
                baseline=baseline,
            )
        target_pixels = int(np.count_nonzero(masks.target))
        target_edit = int(
            np.count_nonzero((masks.target > 0) & (decision.edit_mask > 0))
        )
        total_target += target_pixels
        total_target_edit += target_edit
        protected_changed += int(metrics["protected_changed_pixel_count"])
        protected_overlap += int(
            np.count_nonzero((decision.edit_mask > 0) & (masks.protected > 0))
        )
        if masks.preserve is not None:
            preserve_overlap += int(
                np.count_nonzero((decision.edit_mask > 0) & (masks.preserve > 0))
            )
        ambiguous_changed += int(metrics["ambiguous_changed_pixel_count"])
        ambiguous_overlap += int(
            np.count_nonzero((decision.edit_mask > 0) & (masks.ambiguous > 0))
        )
        outside_changed += int(metrics["changed_outside_detector_mask_pixel_count"])
        no_edit_false += int(page.no_edit) * int(np.count_nonzero(decision.edit_mask))
        residue_value = metrics.get("residue_score")
        residue_pixels = int(metrics.get("residue_source_contrast_pixel_count") or 0)
        if residue_value is not None and residue_pixels:
            residue_sum += float(residue_value) * residue_pixels
            residue_count += residue_pixels
        baseline_value = metrics.get("baseline_residue_score")
        if baseline_value is not None and residue_pixels:
            baseline_sum += float(baseline_value) * residue_pixels
            baseline_count += residue_pixels
        if (
            residue_value is not None
            and baseline_value is not None
            and float(residue_value) > float(baseline_value) + 1e-12
        ):
            page_residue_worsened += 1
        known_background = _path_value(entry.get("known_background"))
        reconstruction = None
        if known_background and fill_id != "mask_only":
            reconstruction = reconstruction_error(
                candidate,
                _read_image(known_background, image_cache),
                decision.edit_mask,
            )
            if reconstruction is not None:
                reconstruction_values.append(reconstruction)
        page_runtime = lama_pool.runtime_metrics_since(page_calls_before)
        page_durations = [
            float(value)
            for value in lama_pool.call_durations[page_calls_before:]
        ]
        page_call_count = lama_pool.call_count - page_calls_before
        runtime_backend = (
            "ballons_lama"
            if page_call_count and fill_id == "ballons_lama"
            else "current_lama"
            if page_call_count
            else str(fill_diagnostics.get("backend") or fill_id)
        )
        runtime_event_method = getattr(lama_pool, "runtime_events_since", None)
        if callable(runtime_event_method):
            runtime_events = runtime_event_method(
                page_calls_before, backend=runtime_backend
            )
        else:
            runtime_events = [
                {
                    "call_index": page_calls_before + index + 1,
                    "backend": runtime_backend,
                    "provider": str(page_runtime["lama_runtime_provider"]),
                    "precision": str(page_runtime["lama_runtime_precision"]),
                    "cpu_fallback": bool(
                        int(page_runtime["cpu_fallback_count"]) > index
                    ),
                    "duration_seconds": duration,
                }
                for index, duration in enumerate(page_durations)
            ]
        if len(runtime_events) != page_call_count:
            raise ValueError("factorized runtime event count differs from LaMa calls")
        if [float(row["duration_seconds"]) for row in runtime_events] != page_durations:
            raise ValueError("factorized runtime event durations differ from measured calls")
        if sum(bool(row["cpu_fallback"]) for row in runtime_events) != int(
            page_runtime["cpu_fallback_count"]
        ):
            raise ValueError("factorized runtime fallback events differ from telemetry")
        canonical_statistics = {
            "schema_version": "inpaint-factorized-page-statistics-v1",
            "page_id": page.page_id,
            "target_extent_independent": page.target_extent_independent,
            "target_inventory_independent": page.target_inventory_independent,
            "target_review_complete": page.target_review_complete,
            "target_mask_provenance": page.target_mask_provenance,
            "no_edit": bool(page.no_edit),
            "required_skip": bool(
                decision.decision == "skip" and not page.no_edit
            ),
            "target_pixel_count": target_pixels,
            "target_edit_pixel_count": target_edit,
            "target_instance_seed_scores": seed_scores,
            "target_instance_edit_scores": edit_scores,
            "edit_pixel_count": int(np.count_nonzero(decision.edit_mask)),
            "protected_structure_overlap_pixel_count": int(
                np.count_nonzero(
                    (decision.edit_mask > 0) & (masks.protected > 0)
                )
            ),
            "protected_structure_changed_pixel_count": int(
                metrics["protected_changed_pixel_count"]
            ),
            "preserve_edit_overlap_pixel_count": int(
                np.count_nonzero(
                    (decision.edit_mask > 0) & (masks.preserve > 0)
                )
                if masks.preserve is not None
                else 0
            ),
            "ambiguous_structure_overlap_pixel_count": int(
                np.count_nonzero(
                    (decision.edit_mask > 0) & (masks.ambiguous > 0)
                )
            ),
            "ambiguous_structure_changed_pixel_count": int(
                metrics["ambiguous_changed_pixel_count"]
            ),
            "outside_final_changed_pixel_count": int(
                metrics["changed_outside_detector_mask_pixel_count"]
            ),
            "broad_route_false_positive_pixel_count": broad_route_false_pixels,
            "conditional_hybrid_overlap_conflict_pixel_count": int(
                fill_diagnostics.get("authoritative_overlap_edit_pixel_count", 0)
            ),
            "authoritative_region_overlap_pixel_count": int(
                fill_diagnostics.get(
                    "authoritative_region_overlap_pixel_count", 0
                )
            ),
            "authoritative_overlap_narrow_verified": bool(
                fill_diagnostics.get("authoritative_overlap_narrow_verified", False)
            ),
            "residue_score": residue_value,
            "baseline_residue_score": baseline_value,
            "residue_source_contrast_pixel_count": residue_pixels,
            "reconstruction_mse": reconstruction,
            "residue_gate_applicable": fill_id != "mask_only",
            "runtime_seconds": time.perf_counter() - page_started,
            "positive_lama_inference_count": (
                lama_pool.call_count - page_calls_before
            ),
            "positive_lama_call_durations_seconds": page_durations,
            "runtime_telemetry_complete": page_runtime[
                "runtime_telemetry_complete"
            ],
            "cpu_fallback_count": page_runtime["cpu_fallback_count"],
            "lama_runtime_provider": page_runtime["lama_runtime_provider"],
            "lama_runtime_precision": page_runtime["lama_runtime_precision"],
            "peak_vram_allocated_mib": page_runtime["peak_vram_allocated_mib"],
            "peak_vram_reserved_mib": page_runtime["peak_vram_reserved_mib"],
            "detector_seed_mask_pixel_sha256": _pixel_sha256(detector_seed),
            "output_edit_mask_pixel_sha256": _pixel_sha256(decision.edit_mask),
            "final_mask_pixel_sha256": _pixel_sha256(final_mask),
            "candidate_pixel_sha256": _pixel_sha256(candidate),
        }
        row = {
            "page_id": page.page_id,
            "route_decision": decision.decision,
            "route_reasons": [
                *decision.reasons,
                *(
                    ("authoritative_region_overlap_narrow_fallback",)
                    if int(
                        fill_diagnostics.get(
                            "authoritative_overlap_edit_pixel_count", 0
                        )
                    )
                    else ()
                ),
            ],
            "broad_route_false_positive": broad_route_false,
            "broad_route_false_positive_pixel_count": broad_route_false_pixels,
            "target_instance_seed_scores": seed_scores,
            "target_instance_edit_scores": edit_scores,
            "edit_pixel_count": int(np.count_nonzero(decision.edit_mask)),
            "fill": fill_diagnostics,
            "reconstruction_mse": reconstruction,
            "canonical_statistics": canonical_statistics,
            "canonical_statistics_sha256": _canonical_sha256(
                canonical_statistics
            ),
            "_runtime_evidence": {
                "page_id": page.page_id,
                "summary": {
                    field: canonical_statistics.get(field)
                    for field in _RUNTIME_PAGE_FIELDS
                },
                "inference_events": runtime_events,
            },
            **metrics,
        }
        rows.append(row)
        retain_artifacts = bool(matrix.get("retain_page_artifacts", False))
        required_only = bool(
            matrix.get("retain_required_page_artifacts_only", False)
        )
        if retain_artifacts and (not required_only or not page.no_edit):
            _write_image(run_root / "detector_seed_masks" / f"{page.page_id}.png", detector_seed)
            _write_image(run_root / "edit_masks" / f"{page.page_id}.png", decision.edit_mask)
            _write_image(run_root / "final_masks" / f"{page.page_id}.png", final_mask)
            _write_image(run_root / "candidate_images" / f"{page.page_id}.png", candidate)
            _write_image(run_root / "changed_masks" / f"{page.page_id}.png", changed)

    metrics = aggregate_factorized_page_statistics(rows)
    oracle_only_ids = set(matrix.get("oracle_only", []))
    oracle_only = any(value in oracle_only_ids for value in combination.values())
    record = FactorizedRunRecord(
        run_id=run_id,
        detector_id=combination["detector"],
        ownership_id=combination["ownership"],
        silhouette_id=combination["silhouette"],
        router_id=combination["router"],
        expansion_id=combination["expansion"],
        fill_id=fill_id,
        oracle_only=oracle_only,
        status="active",
        metrics=metrics,
        selection=dict(combination),
    )
    return record, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the v3 role-factorized control/single/pairwise matrix. "
            "Detector masks and route evidence must be source-only artifacts."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--limit", type=int)
    return parser


def _declared_combinations(
    matrix: dict[str, object],
    axes: dict[str, list[str]],
    controls: dict[str, str],
) -> list[dict[str, str]]:
    combinations = (
        build_factorized_matrix(axes, controls)
        if bool(matrix.get("factorized", True))
        else [dict(controls)]
    )
    explicit = matrix.get("explicit_combinations", [])
    if not isinstance(explicit, list):
        raise ValueError("explicit_combinations must be a list")
    seen = {tuple(sorted(record.items())) for record in combinations}
    for raw in explicit:
        if not isinstance(raw, dict):
            raise ValueError("explicit combination must be an object")
        record = dict(controls)
        for role, value in raw.items():
            if role not in controls or not isinstance(value, str) or not value:
                raise ValueError(f"invalid explicit combination field: {role}")
            if value not in axes[role]:
                raise ValueError(
                    f"explicit combination value is outside axis: {role}={value}"
                )
            record[role] = value
        key = tuple(sorted(record.items()))
        if key not in seen:
            seen.add(key)
            combinations.append(record)
    return combinations


def _logical_id(selection: dict[str, str]) -> str:
    return "__".join(selection[role] for role in selection)


def _content_value(value: object) -> object:
    """Replace artifact paths with bytes SHA while retaining algorithm settings."""

    if isinstance(value, dict):
        return {
            str(key): _content_value(nested)
            for key, nested in sorted(value.items())
            if key not in {"provider", "candidate_id", "status"}
        }
    if isinstance(value, list):
        return [_content_value(nested) for nested in value]
    if isinstance(value, str):
        path = Path(value)
        if path.is_file():
            return {"artifact_sha256": _sha256(path)}
        return value
    return value


def _combination_content_sha256(
    combination: dict[str, str],
    *,
    matrix: dict[str, object],
    manifest_sha256: str,
) -> str:
    families = matrix.get("families")
    if not isinstance(families, dict):
        raise ValueError("matrix spec must contain families")
    payload: dict[str, object] = {
        "manifest_sha256": manifest_sha256,
        "expansion": combination["expansion"],
        "fill": combination["fill"],
    }
    for role in ("detector", "ownership", "silhouette", "router"):
        role_families = families.get(role)
        if not isinstance(role_families, dict):
            raise ValueError(f"matrix families lacks role: {role}")
        selected = role_families.get(combination[role])
        if not isinstance(selected, dict):
            raise ValueError(f"matrix lacks selected family: {role}/{combination[role]}")
        payload[role] = _content_value(selected)
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_closure_ledger(
    combinations: list[dict[str, str]],
    *,
    matrix: dict[str, object],
    manifest_sha256: str,
) -> tuple[list[CombinationClosureRecord], list[dict[str, str]]]:
    families = matrix.get("families")
    family_metadata = families if isinstance(families, dict) else None
    oracle_only = matrix.get("oracle_only", [])
    initial: list[CombinationClosureRecord] = []
    executable: list[dict[str, str]] = []
    content_owner: dict[str, str] = {}
    for combination in combinations:
        stage = (
            "oracle"
            if bool(matrix.get("oracle_experiment", False))
            else ("stage1" if combination["fill"] == "mask_only" else "product")
        )
        row = build_combination_closure_ledger(
            [combination],
            stage=stage,
            family_metadata=family_metadata,
            oracle_only_ids=oracle_only if isinstance(oracle_only, list) else (),
        )[0]
        if row.closure_state != "executed":
            initial.append(row)
            continue
        content_sha = _combination_content_sha256(
            combination,
            matrix=matrix,
            manifest_sha256=manifest_sha256,
        )
        owner = content_owner.get(content_sha)
        if owner is not None:
            initial.append(
                CombinationClosureRecord(
                    logical_id=row.logical_id,
                    selection=row.selection,
                    closure_state="reused_by_sha",
                    content_sha256=content_sha,
                    reused_from=owner,
                )
            )
            continue
        content_owner[content_sha] = row.logical_id
        initial.append(
            CombinationClosureRecord(
                logical_id=row.logical_id,
                selection=row.selection,
                closure_state="executed",
                content_sha256=content_sha,
            )
        )
        executable.append(combination)
    assert_complete_closure_ledger(combinations, initial)
    return initial, executable


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = args.manifest.resolve()
    validate_source_only_manifest_v4(manifest_path)
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("schema_version") != "inpaint-factorized-matrix-v3":
        raise ValueError("unsupported factorized matrix schema")
    axes = matrix.get("axes")
    controls = matrix.get("controls")
    if not isinstance(axes, dict) or not isinstance(controls, dict):
        raise ValueError("factorized matrix requires axes and controls")
    all_combinations = _declared_combinations(matrix, axes, controls)
    manifest_sha256 = _sha256(manifest_path)
    closure_ledger, physical_combinations = _prepare_closure_ledger(
        all_combinations,
        matrix=matrix,
        manifest_sha256=manifest_sha256,
    )
    if args.start_index < 1:
        raise ValueError("--start-index must be at least 1")
    indexed_combinations = list(enumerate(physical_combinations, start=1))[
        args.start_index - 1:
    ]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be at least 1")
        indexed_combinations = indexed_combinations[:args.limit]
    combinations_to_run = [combination for _index, combination in indexed_combinations]
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest_path)
        entries = _entries(manifest_path)
        lama_pool = _LamaPool(
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        lama_model_path = _resolve_lama_model_path()
        runtime_identity = _runtime_identity(
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
            lama_model_path=lama_model_path,
        )
        records: list[FactorizedRunRecord] = []
        page_rows: dict[str, list[dict[str, object]]] = {}
        instance_index_cache: dict[str, np.ndarray] = {}
        mask_only_score_cache: dict[str, dict[str, object]] = {}
        for combination_index, combination in enumerate(combinations_to_run, start=1):
            print(
                f"factorized-v3 {combination_index}/{len(combinations_to_run)} "
                f"{combination}",
                flush=True,
            )
            record, rows = _run_combination(
                combination,
                matrix=matrix,
                pages=pages,
                entries=entries,
                output_root=output_root,
                lama_pool=lama_pool,
                instance_index_cache=instance_index_cache,
                mask_only_score_cache=mask_only_score_cache,
            )
            records.append(record)
            page_rows[record.run_id] = rows
            gc.collect()
        records = attach_reconstruction_control(
            records,
            str(matrix.get("reconstruction_control_run_id") or "") or None,
        )
        ranked = select_pareto_records(records)
        output_inventory, complete_output_run_ids = _seal_factorized_output_inventory(
            output_root,
            ranked,
            page_rows,
        )
        runtime_source_inventory = _seal_runtime_source_inventory(
            output_root,
            runtime_identity=runtime_identity,
            lama_model_path=lama_model_path,
        )
        runtime_evidence, complete_runtime_run_ids = (
            _seal_factorized_runtime_evidence(
                output_root,
                ranked,
                page_rows,
                runtime_identity=runtime_identity,
                runtime_source_inventory=runtime_source_inventory,
                total_inference_count=lama_pool.call_count,
            )
        )
        ranked = _downgrade_unsealed_factorized_finalists(
            ranked,
            complete_output_run_ids,
            complete_runtime_run_ids,
        )
        ranked_by_id = {record.run_id: record for record in ranked}
        closure_records: list[dict[str, object]] = []
        for closure_row in closure_ledger:
            serialized = closure_row.as_record()
            executed = ranked_by_id.get(closure_row.logical_id)
            if executed is not None:
                serialized["runtime_diagnostics"] = {
                    "conditional_hybrid_overlap_conflict_pixel_count": int(
                        executed.metrics.get(
                            "conditional_hybrid_overlap_conflict_pixel_count", 0
                        )
                    ),
                    "conditional_hybrid_overlap_fallback_page_count": int(
                        executed.metrics.get(
                            "conditional_hybrid_overlap_fallback_page_count", 0
                        )
                    ),
                }
            closure_records.append(serialized)
        result = {
            "schema_version": "inpaint-factorized-results-v3",
            "manifest_sha256": manifest_sha256,
            "matrix_sha256": _sha256(matrix_path),
            "logical_inventory_sha256": _logical_inventory_sha256(
                closure_ledger
            ),
            "logical_combination_count": len(all_combinations),
            "physical_combination_count": len(physical_combinations),
            "combination_count": len(ranked),
            "closure_ledger": closure_records,
            "positive_lama_inference_count": lama_pool.call_count,
            "reconstruction_control_run_id": str(
                matrix.get("reconstruction_control_run_id") or ""
            ),
            "runs": [record.as_record() for record in ranked],
            "pages": page_rows,
            "output_artifact_inventory": output_inventory,
            "runtime_source_inventory": runtime_source_inventory,
            "runtime_evidence_ledger": runtime_evidence,
        }
        _write_json(output_root / "factorized-results.json", result)
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": result["manifest_sha256"],
                    "matrix_sha256": result["matrix_sha256"],
                    "combination_count": result["combination_count"],
                    "positive_lama_inference_count": result[
                        "positive_lama_inference_count"
                    ],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(managed.run_root)
        else:
            print(output_root)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(
                error,
                metadata={"manifest": manifest_path.name, "matrix": matrix_path.name},
            )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
