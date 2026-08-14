#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    binary_mask,
    mask_sha256,
)
from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (  # noqa: E402
    _validate_stage1_output_artifacts,
)
from benchmarking.inpaint_detector_bakeoff.proposal_refinement import (  # noqa: E402
    RegionAdmissionEvidence,
    refine_detector_proposal,
)
from benchmarking.inpaint_detector_bakeoff.semantic import (  # noqa: E402
    TRANSLATE,
    product_semantic_decision,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    _resolve_manifest_artifact,
    load_page_masks,
    load_stage1_manifest,
    validate_source_only_manifest_v4,
)
from scripts.build_inpaint_product_policy_overlay_v33 import (  # noqa: E402
    validate_policy_overlay,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-proposal-refinement-v33"
CATEGORY = "40-inpaint-mask-render"
SCHEMA_VERSION = "inpaint-proposal-refinement-results-v33"


@dataclass(frozen=True, slots=True)
class CandidateSpec:
    candidate_id: str
    provider_mode: str
    expansion_mode: str
    admission_policy: str


CANDIDATES = (
    CandidateSpec("g1_f_raw_core", "finetune", "raw_core", "g1"),
    CandidateSpec("g1_f_connected_halo", "finetune", "connected_halo", "g1"),
    CandidateSpec("g1_t_raw_core", "tiled", "raw_core", "g1"),
    CandidateSpec("g1_t_connected_halo", "tiled", "connected_halo", "g1"),
    CandidateSpec("g1_or_raw_core", "or", "raw_core", "g1"),
    CandidateSpec("g1_or_connected_halo", "or", "connected_halo", "g1"),
    CandidateSpec("g2_or_raw_core", "or", "raw_core", "g2"),
    CandidateSpec("g2_or_connected_halo", "or", "connected_halo", "g2"),
    CandidateSpec("g3_or_raw_core", "or", "raw_core", "g3"),
    CandidateSpec("g3_or_connected_halo", "or", "connected_halo", "g3"),
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _read_image(path: str | Path, flags: int) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(Path(path), dtype=np.uint8), flags)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    return value


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    value = _read_image(path, cv2.IMREAD_GRAYSCALE)
    if value.shape != shape:
        raise ValueError(f"v3.3 mask shape mismatch: {value.shape} != {shape}")
    return binary_mask(value, shape)


def _write_mask(path: Path, mask: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"v3.3 output mask must be fresh: {path}")
    normalized = binary_mask(mask)
    encoded, buffer = cv2.imencode(".png", normalized)
    if not encoded:
        raise RuntimeError("failed to encode v3.3 output mask")
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_bytes(buffer.tobytes())
    temporary.replace(path)
    decoded = _read_mask(path, normalized.shape)
    if not np.array_equal(decoded, normalized):
        raise RuntimeError("v3.3 output mask changed during encoding")
    return {
        "file_sha256": _sha256(path),
        "pixel_sha256": mask_sha256(decoded),
        "pixel_count": int(np.count_nonzero(decoded)),
    }


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _stage1_input(
    run_root: Path,
    *,
    expected_variant: str,
    expected_source_manifest_sha256: str,
    page_ids: tuple[str, ...],
) -> dict[str, object]:
    result_path = run_root / "stage1-results.json"
    payload = _read_json(result_path)
    _validate_stage1_output_artifacts(payload, result_path)
    if payload.get("manifest_sha256") != expected_source_manifest_sha256:
        raise ValueError("v3.3 detector run uses a different source manifest")
    if str(payload.get("variant") or "") != expected_variant:
        raise ValueError("v3.3 detector run variant differs")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping) or summary.get("page_count") != len(page_ids):
        raise ValueError("v3.3 detector run page count differs")
    variant_dir = "dilated" if expected_variant == "dilated" else "raw"
    mask_root = run_root / "native_masks" / variant_dir
    files = {path.stem: path.resolve() for path in mask_root.glob("*.png")}
    if set(files) != set(page_ids):
        raise ValueError("v3.3 detector mask inventory differs from E1")
    return {
        "run_root": str(run_root.resolve()),
        "result_path": str(result_path.resolve()),
        "result_sha256": _sha256(result_path),
        "candidate": str(payload.get("candidate") or ""),
        "variant": expected_variant,
        "mask_root": str(mask_root.resolve()),
        "masks": files,
    }


def _required_instance_masks(page, shape: tuple[int, int]):
    required = []
    optional = []
    for record in page.target_instances:
        mask = _read_mask(record.mask_path, shape)
        row = (record.instance_id, mask)
        if record.priority == "required":
            required.append(row)
        elif record.priority == "optional":
            optional.append(row)
    return tuple(required), tuple(optional)


def _candidate_page_score(
    *,
    page,
    proposal: np.ndarray,
    safe_addition: np.ndarray,
    final_mask: np.ndarray,
    baseline_mask: np.ndarray,
    masks,
    required_instances,
    optional_instances,
    hard_protection: np.ndarray,
    component_records,
) -> dict[str, object]:
    required_scores = []
    baseline_98 = final_98 = newly_missed = 0
    for instance_id, target in required_instances:
        pixels = int(np.count_nonzero(target))
        seeded_pixels = int(np.count_nonzero((target > 0) & (proposal > 0)))
        baseline_pixels = int(
            np.count_nonzero((target > 0) & (baseline_mask > 0))
        )
        final_pixels = int(np.count_nonzero((target > 0) & (final_mask > 0)))
        baseline_coverage = float(baseline_pixels) / pixels if pixels else 0.0
        coverage = float(final_pixels) / pixels if pixels else 0.0
        baseline_98 += int(baseline_coverage >= 0.98)
        final_98 += int(coverage >= 0.98)
        newly_missed += int(baseline_coverage > 0.0 and coverage <= 0.0)
        required_scores.append(
            {
                "instance_id": instance_id,
                "seeded": seeded_pixels > 0,
                "seeded_pixel_count": seeded_pixels,
                "baseline_coverage": baseline_coverage,
                "coverage": coverage,
            }
        )
    optional_union = np.zeros(final_mask.shape, dtype=np.uint8)
    for _instance_id, target in optional_instances:
        optional_union[target > 0] = 255
    reason_counts = Counter(record.reason for record in component_records)
    accepted_components = sum(record.accepted for record in component_records)
    target_pixels = int(np.count_nonzero(masks.target))
    baseline_target = int(
        np.count_nonzero((masks.target > 0) & (baseline_mask > 0))
    )
    final_target = int(
        np.count_nonzero((masks.target > 0) & (final_mask > 0))
    )
    action_translate_available = any(
        record.semantic_action == TRANSLATE for record in component_records
    )
    return {
        "page_id": page.page_id,
        "expected_edit": page.expected_edit,
        "proposal_pixel_count": int(np.count_nonzero(proposal)),
        "safe_addition_pixel_count": int(np.count_nonzero(safe_addition)),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)),
        "target_pixel_count": target_pixels,
        "baseline_target_covered_pixel_count": baseline_target,
        "target_covered_pixel_count": final_target,
        "baseline_target_coverage": (
            float(baseline_target) / target_pixels if target_pixels else None
        ),
        "target_coverage": float(final_target) / target_pixels if target_pixels else None,
        "target_instance_scores": required_scores,
        "baseline_98_instance_count": baseline_98,
        "coverage_98_instance_count": final_98,
        "newly_missed_required_instance_count": newly_missed,
        "protected_addition_overlap": int(
            np.count_nonzero((safe_addition > 0) & (masks.protected > 0))
        ),
        "ambiguous_addition_overlap": int(
            np.count_nonzero((safe_addition > 0) & (masks.ambiguous > 0))
        ),
        "corner_addition_overlap": int(
            np.count_nonzero((safe_addition > 0) & (masks.corner > 0))
            if masks.corner is not None
            else 0
        ),
        "hard_protection_addition_overlap": int(
            np.count_nonzero((safe_addition > 0) & (hard_protection > 0))
        ),
        "ownership_leak_pixel_count": int(
            np.count_nonzero((safe_addition > 0) & (masks.ownership == 0))
        ),
        "proposal_escape_pixel_count": int(
            np.count_nonzero((safe_addition > 0) & (proposal == 0))
        ),
        "existing_pr6_overlap_pixel_count": int(
            np.count_nonzero((safe_addition > 0) & (baseline_mask > 0))
        ),
        "optional_neutral_addition_pixel_count": int(
            np.count_nonzero((safe_addition > 0) & (optional_union > 0))
        ),
        "zero_translate_action_addition_pixel_count": (
            int(np.count_nonzero(safe_addition))
            if not action_translate_available
            else 0
        ),
        "component_count": len(component_records),
        "accepted_component_count": accepted_components,
        "component_reason_counts": dict(sorted(reason_counts.items())),
        "output_safe_addition_pixel_sha256": mask_sha256(safe_addition),
        "output_final_mask_pixel_sha256": mask_sha256(final_mask),
    }


def _aggregate_candidate(
    spec: CandidateSpec,
    pages: list[dict[str, object]],
) -> dict[str, object]:
    required_scores = [
        score
        for page in pages
        for score in page["target_instance_scores"]  # type: ignore[index]
    ]
    target_pixels = sum(int(page["target_pixel_count"]) for page in pages)
    baseline_target = sum(
        int(page["baseline_target_covered_pixel_count"]) for page in pages
    )
    final_target = sum(int(page["target_covered_pixel_count"]) for page in pages)
    zero_fields = (
        "protected_addition_overlap",
        "ambiguous_addition_overlap",
        "corner_addition_overlap",
        "hard_protection_addition_overlap",
        "ownership_leak_pixel_count",
        "proposal_escape_pixel_count",
        "existing_pr6_overlap_pixel_count",
        "zero_translate_action_addition_pixel_count",
    )
    safety = {field: sum(int(page[field]) for page in pages) for field in zero_fields}
    safety_failures = [field for field, value in safety.items() if value]
    seeded = sum(bool(score["seeded"]) for score in required_scores)
    required_count = len(required_scores)
    baseline_98 = sum(
        float(score["baseline_coverage"]) >= 0.98 for score in required_scores
    )
    final_98 = sum(float(score["coverage"]) >= 0.98 for score in required_scores)
    newly_missed = sum(
        float(score["baseline_coverage"]) > 0.0
        and float(score["coverage"]) <= 0.0
        for score in required_scores
    )
    aggregate = {
        "candidate_id": spec.candidate_id,
        "provider_mode": spec.provider_mode,
        "expansion_mode": spec.expansion_mode,
        "admission_policy": spec.admission_policy,
        "page_count": len(pages),
        "required_target_instance_count": required_count,
        "seeded_target_instance_count": seeded,
        "missed_target_instance_count": required_count - seeded,
        "target_instance_seed_recall": (
            float(seeded) / required_count if required_count else None
        ),
        "strict_seed_eligible": seeded == required_count,
        "seed_admitted": required_count > 0,
        "target_pixel_count": target_pixels,
        "baseline_target_covered_pixel_count": baseline_target,
        "target_covered_pixel_count": final_target,
        "baseline_aggregate_target_coverage": (
            float(baseline_target) / target_pixels if target_pixels else None
        ),
        "aggregate_target_coverage": (
            float(final_target) / target_pixels if target_pixels else None
        ),
        "baseline_98_instance_count": baseline_98,
        "coverage_98_instance_count": final_98,
        "newly_missed_required_instance_count": newly_missed,
        "safe_addition_pixel_count": sum(
            int(page["safe_addition_pixel_count"]) for page in pages
        ),
        "optional_neutral_addition_pixel_count": sum(
            int(page["optional_neutral_addition_pixel_count"]) for page in pages
        ),
        **safety,
        "incremental_safety_pass": not safety_failures and newly_missed == 0,
        "incremental_safety_failures": sorted(
            safety_failures
            + (["newly_missed_required_instance"] if newly_missed else [])
        ),
        "relative_product_pass": None,
        "relative_product_reason": "stage2_not_run",
        "output_mask_set_sha256": _canonical_sha256(
            sorted(
                (
                    {
                        "page_id": str(page["page_id"]),
                        "safe_addition": str(
                            page["output_safe_addition_pixel_sha256"]
                        ),
                        "final_mask": str(page["output_final_mask_pixel_sha256"]),
                    }
                    for page in pages
                ),
                key=lambda row: row["page_id"],
            )
        ),
    }
    return aggregate


def _shortlist(candidates: list[dict[str, object]]) -> list[str]:
    safe = [
        row
        for row in candidates
        if row.get("incremental_safety_pass") is True
        and float(row.get("aggregate_target_coverage") or 0.0)
        >= float(row.get("baseline_aggregate_target_coverage") or 0.0)
    ]
    safe.sort(
        key=lambda row: (
            -int(row["coverage_98_instance_count"]),
            -float(row["aggregate_target_coverage"]),
            int(row["safe_addition_pixel_count"]),
            str(row["candidate_id"]),
        )
    )
    return [str(row["candidate_id"]) for row in safe[:2]]


def run_refinement(
    *,
    source_manifest_path: Path,
    relative_manifest_path: Path,
    policy_overlay_path: Path,
    finetune_raw_run: Path,
    finetune_native3_run: Path,
    tiled_raw_run: Path,
    tiled_native3_run: Path,
    c23_result_path: Path,
    output_root: Path,
) -> dict[str, object]:
    source_binding = validate_source_only_manifest_v4(source_manifest_path)
    relative_binding = validate_source_only_manifest_v4(relative_manifest_path)
    relative_seal = _read_json(
        relative_manifest_path.with_suffix(relative_manifest_path.suffix + ".seal.json")
    )
    if relative_seal.get("source_manifest_sha256") != source_binding["manifest_sha256"]:
        raise ValueError("v3.3 relative baseline is not bound to E1 r16")
    overlay = validate_policy_overlay(
        policy_overlay_path,
        manifest_path=source_manifest_path,
    )
    pages = load_stage1_manifest(relative_manifest_path)
    page_ids = tuple(page.page_id for page in pages)
    if tuple(sorted(page_ids)) != tuple(source_binding["page_ids"]):
        raise ValueError("v3.3 relative and source page inventories differ")
    raw_relative = _read_json(relative_manifest_path)
    entries = {
        str(row.get("page_id") or ""): row
        for row in raw_relative.get("pages", [])
        if isinstance(row, dict)
    }
    if set(entries) != set(page_ids):
        raise ValueError("v3.3 relative manifest raw page inventory differs")

    inputs = {
        "finetune_raw": _stage1_input(
            finetune_raw_run,
            expected_variant="raw",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "finetune_native3": _stage1_input(
            finetune_native3_run,
            expected_variant="dilated",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "tiled_raw": _stage1_input(
            tiled_raw_run,
            expected_variant="raw",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
        "tiled_native3": _stage1_input(
            tiled_native3_run,
            expected_variant="dilated",
            expected_source_manifest_sha256=str(source_binding["manifest_sha256"]),
            page_ids=page_ids,
        ),
    }
    c23 = _read_json(c23_result_path)
    if c23.get("manifest_sha256") != source_binding["manifest_sha256"]:
        raise ValueError("v3.3 C23 control uses a different E1 manifest")
    c23_summary = c23.get("summary")
    if not isinstance(c23_summary, Mapping):
        raise ValueError("v3.3 C23 control lacks summary")

    output_root.mkdir(parents=True, exist_ok=True)
    if any(output_root.iterdir()):
        raise FileExistsError("v3.3 proposal refinement output must be fresh")
    component_tmp = output_root / ".component-decisions.jsonl.partial"
    candidate_pages: dict[str, list[dict[str, object]]] = {
        spec.candidate_id: [] for spec in CANDIDATES
    }
    output_artifacts: list[dict[str, object]] = []
    baseline_inventory: list[dict[str, str]] = []
    with component_tmp.open("w", encoding="utf-8", newline="\n") as component_stream:
        for page in pages:
            entry = entries[page.page_id]
            source = _read_image(page.source_image, cv2.IMREAD_COLOR)
            shape = source.shape[:2]
            baseline_path = _resolve_manifest_artifact(
                relative_manifest_path,
                entry.get("baseline_mask"),
            )
            if baseline_path is None:
                raise ValueError(f"v3.3 baseline mask is missing: {page.page_id}")
            masks = load_page_masks(
                page,
                shape,
                existing_edit_path=str(baseline_path),
                strict_binary=True,
            )
            baseline = binary_mask(masks.existing_edit, shape)
            baseline_inventory.append(
                {
                    "page_id": page.page_id,
                    "file_sha256": _sha256(baseline_path),
                    "pixel_sha256": mask_sha256(baseline),
                }
            )
            required_instances, optional_instances = _required_instance_masks(
                page, shape
            )
            raw_regions = entry.get("regions")
            if not isinstance(raw_regions, list) or len(raw_regions) != len(
                masks.regions
            ):
                raise ValueError("v3.3 region semantic inventory differs")
            regions = []
            for raw_region, region_masks in zip(raw_regions, masks.regions):
                if not isinstance(raw_region, Mapping) or str(
                    raw_region.get("region_id") or ""
                ) != region_masks.region_id:
                    raise ValueError("v3.3 region semantic order differs")
                regions.append(
                    RegionAdmissionEvidence(
                        region_masks.region_id,
                        region_masks.ownership,
                        product_semantic_decision(raw_region),
                    )
                )
            detector_masks = {
                key: _read_mask(value["masks"][page.page_id], shape)  # type: ignore[index]
                for key, value in inputs.items()
            }
            for spec in CANDIDATES:
                refined = refine_detector_proposal(
                    finetune_raw=detector_masks["finetune_raw"],
                    finetune_native3=detector_masks["finetune_native3"],
                    tiled_raw=detector_masks["tiled_raw"],
                    tiled_native3=detector_masks["tiled_native3"],
                    provider_mode=spec.provider_mode,
                    expansion_mode=spec.expansion_mode,
                    admission_policy=spec.admission_policy,
                    regions=tuple(regions),
                    pr6_existing_edit=baseline,
                    source_raw_owned=masks.claim_seed,
                    structure_protect=masks.protected,
                    ambiguous_protect=masks.ambiguous,
                    corner_protect=(
                        masks.corner
                        if masks.corner is not None
                        else np.zeros(shape, dtype=np.uint8)
                    ),
                )
                final_mask = cv2.bitwise_or(baseline, refined.safe_addition)
                page_score = _candidate_page_score(
                    page=page,
                    proposal=refined.proposal,
                    safe_addition=refined.safe_addition,
                    final_mask=final_mask,
                    baseline_mask=baseline,
                    masks=masks,
                    required_instances=required_instances,
                    optional_instances=optional_instances,
                    hard_protection=refined.hard_protection,
                    component_records=refined.component_records,
                )
                candidate_pages[spec.candidate_id].append(page_score)
                mask_record_path = (
                    output_root
                    / "runs"
                    / spec.candidate_id
                    / "safe_additions"
                    / f"{page.page_id}.png"
                )
                mask_record = _write_mask(
                    mask_record_path,
                    refined.safe_addition,
                )
                output_artifacts.append(
                    {
                        "candidate_id": spec.candidate_id,
                        "page_id": page.page_id,
                        "role": "safe_addition",
                        "relative_path": mask_record_path.relative_to(
                            output_root
                        ).as_posix(),
                        **mask_record,
                    }
                )
                for record in refined.component_records:
                    component_stream.write(
                        json.dumps(
                            {
                                "candidate_id": spec.candidate_id,
                                "page_id": page.page_id,
                                **asdict(record),
                            },
                            ensure_ascii=False,
                            sort_keys=True,
                        )
                        + "\n"
                    )
    component_path = output_root / "component-decisions.jsonl"
    component_tmp.replace(component_path)
    candidates = [
        _aggregate_candidate(spec, candidate_pages[spec.candidate_id])
        for spec in CANDIDATES
    ]
    shortlist = _shortlist(candidates)
    output_artifacts.sort(
        key=lambda row: (str(row["candidate_id"]), str(row["page_id"]))
    )
    inventory_payload = {
        "schema_version": "inpaint-proposal-refinement-output-inventory-v33",
        "source_manifest_sha256": source_binding["manifest_sha256"],
        "relative_manifest_sha256": relative_binding["manifest_sha256"],
        "policy_overlay_sha256": overlay["overlay_sha256"],
        "candidate_ids": [spec.candidate_id for spec in CANDIDATES],
        "page_ids": sorted(page_ids),
        "artifacts": output_artifacts,
    }
    inventory_payload["inventory_sha256"] = _canonical_sha256(inventory_payload)
    inventory_path = output_root / "output-artifact-inventory.json"
    _atomic_json(inventory_path, inventory_payload)
    result = {
        "schema_version": SCHEMA_VERSION,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_worktree_clean": not bool(
            subprocess.check_output(
                ["git", "status", "--short", "--untracked-files=no"],
                cwd=ROOT,
                text=True,
            ).strip()
        ),
        "evaluator_sha256": _sha256(Path(__file__).resolve()),
        "source_manifest": {
            "manifest_sha256": source_binding["manifest_sha256"],
            "page_inventory_sha256": source_binding["page_inventory_sha256"],
            "page_count": source_binding["page_count"],
        },
        "relative_manifest": {
            "manifest_sha256": relative_binding["manifest_sha256"],
            "seal_sha256": relative_binding["seal_sha256"],
        },
        "policy_overlay": {
            "artifact_sha256": _sha256(policy_overlay_path),
            "overlay_sha256": overlay["overlay_sha256"],
            "policy_id": overlay["policy_id"],
        },
        "input_detector_runs": {
            key: {
                field: value[field]
                for field in (
                    "run_root",
                    "result_path",
                    "result_sha256",
                    "candidate",
                    "variant",
                    "mask_root",
                )
            }
            for key, value in inputs.items()
        },
        "g0_c23_control": {
            "artifact_sha256": _sha256(c23_result_path),
            "summary": dict(c23_summary),
        },
        "baseline": {
            "mask_set_sha256": _canonical_sha256(
                sorted(baseline_inventory, key=lambda row: row["page_id"])
            ),
            "pages": baseline_inventory,
        },
        "candidates": candidates,
        "pages": candidate_pages,
        "shortlist": shortlist,
        "shortlist_limit": 2,
        "component_decisions": {
            "relative_path": component_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(component_path),
        },
        "output_inventory": {
            "relative_path": inventory_path.relative_to(output_root).as_posix(),
            "artifact_sha256": _sha256(inventory_path),
            "inventory_sha256": inventory_payload["inventory_sha256"],
            "artifact_count": len(output_artifacts),
        },
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate conservative detector-proposal additions on PR6."
    )
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--relative-manifest", type=Path, required=True)
    parser.add_argument("--policy-overlay", type=Path, required=True)
    parser.add_argument("--finetune-raw-run", type=Path, required=True)
    parser.add_argument("--finetune-native3-run", type=Path, required=True)
    parser.add_argument("--tiled-raw-run", type=Path, required=True)
    parser.add_argument("--tiled-native3-run", type=Path, required=True)
    parser.add_argument("--c23-result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    try:
        result = run_refinement(
            source_manifest_path=args.source_manifest.resolve(),
            relative_manifest_path=args.relative_manifest.resolve(),
            policy_overlay_path=args.policy_overlay.resolve(),
            finetune_raw_run=args.finetune_raw_run.resolve(),
            finetune_native3_run=args.finetune_native3_run.resolve(),
            tiled_raw_run=args.tiled_raw_run.resolve(),
            tiled_native3_run=args.tiled_native3_run.resolve(),
            c23_result_path=args.c23_result.resolve(),
            output_root=output_root,
        )
        result_path = output_root / "proposal-refinement-results.json"
        _atomic_json(result_path, result)
        if managed is not None:
            managed.complete(
                metadata={
                    "source_manifest_sha256": result["source_manifest"][
                        "manifest_sha256"
                    ],
                    "policy_overlay_sha256": result["policy_overlay"][
                        "overlay_sha256"
                    ],
                    "shortlist": result["shortlist"],
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(managed.run_root)
        else:
            print(result_path)
        return 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
