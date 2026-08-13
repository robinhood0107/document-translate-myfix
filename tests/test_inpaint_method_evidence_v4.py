from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (
    _validate_runtime_evidence_ledger,
    accounted_evidence_from_artifact,
    blocked_asset_evidence,
    merge_method_evidence,
    registry_evidence_adapter_gaps,
    scope_manifest_binding,
)
from benchmarking.inpaint_detector_bakeoff.method_closure import (
    build_method_family_closure,
    MethodVariantEvidence,
    MethodVariantRequirement,
    requirements_from_registry,
)
from benchmarking.inpaint_detector_bakeoff.contracts import mask_sha256
from benchmarking.inpaint_detector_bakeoff.stage1 import (
    source_manifest_page_inventory_sha256,
    summarize as summarize_stage1_pages,
)
from scripts.build_inpaint_method_closure_v4 import build_closure
from scripts.update_inpaint_method_evidence_v4 import update_evidence
from scripts.benchmark_inpaint_factorized_v3 import (
    aggregate_factorized_page_statistics,
    _declared_combinations,
    _prepare_closure_ledger,
    _seal_runtime_source_inventory,
    _sha256,
)
from scripts.benchmark_inpaint_detector_fusions_v4 import (
    aggregate_fusion_page_statistics,
)
from scripts.benchmark_inpaint_semantic_policies_v4 import (
    aggregate_semantic_page_statistics,
)
from modules.utils.download import ModelDownloader, ModelID


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _source_only_page(tmp_path: Path, page_id: str) -> dict[str, object]:
    shape = (8, 10)
    source = np.full((*shape, 3), 200, np.uint8)
    target = np.zeros(shape, np.uint8)
    target[2:4, 3:5] = 255
    zero = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    values = {
        "path": source,
        "target_text_mask": target,
        "preserve_mask": zero,
        "protected_structure_mask": zero,
        "ambiguous_structure_mask": zero,
        "ownership_mask": full,
        "claim_seed_mask": full,
        "bubble_interior_mask": full,
        "corner_protect_mask": zero,
        "existing_source_edit_mask": zero,
    }
    artifacts: dict[str, Path] = {}
    for field, value in values.items():
        artifact = tmp_path / f"{page_id}-{field}.png"
        assert cv2.imwrite(str(artifact), value)
        artifacts[field] = artifact
    region = {
        "region_id": "region",
        "bubble_route_class": "clean_flat",
        "bubble_interior_mask": str(artifacts["ownership_mask"]),
        "ownership_mask": str(artifacts["ownership_mask"]),
        "protected_structure_mask": str(artifacts["protected_structure_mask"]),
        "ambiguous_structure_mask": str(artifacts["ambiguous_structure_mask"]),
        "corner_protect_mask": str(artifacts["corner_protect_mask"]),
    }
    hashes = {
        field: hashlib.sha256(artifact.read_bytes()).hexdigest()
        for field, artifact in artifacts.items()
    }
    hashes.update(
        {
            "target_instances": {"target": hashes["target_text_mask"]},
            "regions": {
                "region": {
                    "bubble_interior_mask": hashes["ownership_mask"],
                    "ownership_mask": hashes["ownership_mask"],
                    "protected_structure_mask": hashes[
                        "protected_structure_mask"
                    ],
                    "ambiguous_structure_mask": hashes[
                        "ambiguous_structure_mask"
                    ],
                    "corner_protect_mask": hashes["corner_protect_mask"],
                }
            },
        }
    )
    return {
        "page_id": page_id,
        **{field: str(artifact) for field, artifact in artifacts.items()},
        "width": shape[1],
        "height": shape[0],
        "source_sha256": hashes["path"],
        "artifact_sha256": hashes,
        "target_instances": [{
            "instance_id": "target",
            "mask_path": str(artifacts["target_text_mask"]),
            "region_id": "region",
            "semantic_role": "dialogue_bubble",
            "processing_action": "translate_inpaint",
            "priority": "required",
        }],
        "regions": [region],
        "expected_edit": "required",
        "annotation_basis": "source_only_v4",
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "target_mask_provenance": "source_only_v4",
    }


def _seal_scope_manifest(path: Path) -> None:
    _write_json(
        path.with_suffix(path.suffix + ".seal.json"),
        {
            "schema_version": "inpaint-factorized-manifest-seal-v4",
            "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "candidate_generated": False,
            "candidate_seen": False,
            "annotation_frozen_before_candidate": True,
        },
    )


def _scope_manifest(tmp_path: Path, *, corpus: str = "e1") -> Path:
    path = _write_json(
        tmp_path / f"{corpus}-manifest.json",
        {
            "schema_version": "inpaint-factorized-source-manifest-v4",
            "corpus_id": corpus,
            "split_role": "development_source_only",
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "target_extent_independent": True,
            "target_inventory_independent": True,
            "target_review_complete": True,
            "pages": [_source_only_page(tmp_path, "p")],
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["page_count"] = 1
    payload["page_ids"] = ["p"]
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        payload["pages"]
    )
    _write_json(path, payload)
    _seal_scope_manifest(path)
    return path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(*, passing: bool = False) -> dict[str, object]:
    return summarize_stage1_pages([_stage1_page(passing=passing)])


def _stage1_page(*, passing: bool = False) -> dict[str, object]:
    coverage = 1.0 if passing else 0.9
    seeded = passing
    return {
        "page_id": "p",
        "target_pixel_count": 10,
        "target_edit_pixel_count": int(coverage * 10),
        "component_coverages": [coverage],
        "target_instance_seed_scores": [{"seeded": seeded}],
        "target_instance_edit_scores": [{"coverage": coverage}],
        "raw_claim_protected_overlap": 0,
        "raw_claim_ambiguous_overlap": 0,
        "raw_claim_outside_ownership_pixel_count": 0,
        "protected_edit_overlap": 0,
        "ambiguous_edit_overlap": 0,
        "ownership_leak_pixel_count": 0,
        "preserve_edit_overlap": 0,
        "false_edit_pixel_count": 0,
    }


def _passing_factorized_metrics() -> dict[str, object]:
    return {
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "protected_structure_overlap": 0,
        "protected_structure_changed": 0,
        "ambiguous_structure_overlap": 0,
        "ambiguous_structure_changed": 0,
        "outside_final_changed": 0,
        "broad_route_false_positive": 0,
        "no_edit_false_edit": 0,
        "required_skip_count": 0,
        "preserve_edit_overlap": 0,
        "missed_target_instance_count": 0,
        "page_residue_worsened_count": 0,
        "page_count": 1,
        "required_target_instance_count": 0,
        "target_pixel_count": 0,
        "aggregate_target_coverage": None,
        "minimum_target_instance_coverage": None,
        "target_instance_seed_recall": None,
        "residue_gate_applicable": False,
        "aggregate_residue_score": None,
        "baseline_aggregate_residue_score": None,
        "reconstruction_gate_applicable": False,
        "runtime_telemetry_complete": True,
        "runtime_seconds": 0.0,
        "positive_lama_inference_count": 0,
        "maximum_positive_lama_inference_per_page": 0,
        "cpu_fallback_count": 0,
    }


def _factorized_artifact(
    tmp_path: Path,
    manifest: Path,
    *,
    detector: str = "current_ctd_raw",
    status: str = "dominated",
    oracle_only: bool = False,
) -> Path:
    selection = {
        "detector": detector,
        "ownership": "control_text_prior",
        "silhouette": "pr2_validated",
        "router": "control_r0",
        "expansion": "raw",
        "fill": "current_lama",
    }
    matrix = {
        "schema_version": "inpaint-factorized-matrix-v3",
        "manifest": str(manifest.resolve()),
        "factorized": True,
        "controls": selection,
        "axes": {role: [value] for role, value in selection.items()},
        "families": {
            "detector": {detector: {}},
            "ownership": {"control_text_prior": {}},
            "silhouette": {"pr2_validated": {}},
            "router": {"control_r0": {}},
        },
        "oracle_only": [],
        "explicit_combinations": [],
    }
    matrix_path = _write_json(tmp_path / "matrix.json", matrix)
    serialized_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    combinations = _declared_combinations(
        serialized_matrix,
        serialized_matrix["axes"],
        serialized_matrix["controls"],
    )
    ledger, _physical = _prepare_closure_ledger(
        combinations, matrix=serialized_matrix, manifest_sha256=_sha(manifest)
    )
    closure = [row.as_record() for row in ledger]
    run_id = str(closure[0]["logical_id"])
    canonical = {
        "schema_version": "inpaint-factorized-page-statistics-v1",
        "page_id": "p",
        "target_extent_independent": status != "information_limited",
        "target_inventory_independent": True,
        "target_review_complete": True,
        "target_mask_provenance": "source_only_v4",
        "no_edit": False,
        "required_skip": status == "dominated",
        "target_pixel_count": 0,
        "target_edit_pixel_count": 0,
        "target_instance_seed_scores": [],
        "target_instance_edit_scores": [],
        "edit_pixel_count": 0,
        "protected_structure_overlap_pixel_count": 0,
        "protected_structure_changed_pixel_count": 0,
        "preserve_edit_overlap_pixel_count": 0,
        "ambiguous_structure_overlap_pixel_count": 0,
        "ambiguous_structure_changed_pixel_count": 0,
        "outside_final_changed_pixel_count": 0,
        "broad_route_false_positive_pixel_count": 0,
        "residue_score": None,
        "baseline_residue_score": None,
        "residue_source_contrast_pixel_count": 0,
        "reconstruction_mse": None,
        "residue_gate_applicable": False,
        "runtime_seconds": 0.0,
        "positive_lama_inference_count": 0,
        "positive_lama_call_durations_seconds": [],
        "runtime_telemetry_complete": True,
        "cpu_fallback_count": 0,
        "lama_runtime_provider": "",
        "lama_runtime_precision": "",
        "peak_vram_allocated_mib": None,
        "peak_vram_reserved_mib": None,
        "output_edit_mask_pixel_sha256": SHA_A,
        "final_mask_pixel_sha256": SHA_B,
        "candidate_pixel_sha256": SHA_C,
    }
    page_row = {
        "page_id": "p",
        "canonical_statistics": canonical,
        "canonical_statistics_sha256": hashlib.sha256(
            json.dumps(
                canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }
    metrics = aggregate_factorized_page_statistics([page_row])
    logical_inventory_sha256 = hashlib.sha256(
        json.dumps(
            [
                {
                    "logical_id": closure[0]["logical_id"],
                    "selection": closure[0]["selection"],
                }
            ],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return _write_json(
        tmp_path / "factorized.json",
        {
            "schema_version": "inpaint-factorized-results-v3",
            "manifest_sha256": _sha(manifest),
            "matrix_sha256": _sha(matrix_path),
            "logical_inventory_sha256": logical_inventory_sha256,
            "logical_combination_count": 1,
            "physical_combination_count": 1,
            "combination_count": 1,
            "positive_lama_inference_count": 0,
            "closure_ledger": closure,
            "runs": [
                {
                    "run_id": run_id,
                    "detector_id": detector,
                    "ownership_id": "control_text_prior",
                    "silhouette_id": "pr2_validated",
                    "router_id": "control_r0",
                    "expansion_id": "raw",
                    "fill_id": "current_lama",
                    "oracle_only": oracle_only,
                    "status": status,
                    "metrics": metrics,
                    "closure_reason": "hard_gate_failed" if status == "dominated" else "",
                    "selection": selection,
                }
            ],
            "reconstruction_control_run_id": "",
            "pages": {run_id: [page_row]},
        },
    )


def _attach_factorized_output_inventory(artifact: Path) -> Path:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    matrix_path = artifact.parent / "matrix.json"
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    matrix["controls"]["fill"] = "mask_only"
    matrix["axes"]["fill"] = ["mask_only"]
    _write_json(matrix_path, matrix)
    manifest_path = artifact.parent / "e1-manifest.json"
    combinations = _declared_combinations(
        matrix,
        matrix["axes"],
        matrix["controls"],
    )
    closure, _physical = _prepare_closure_ledger(
        combinations,
        matrix=matrix,
        manifest_sha256=_sha(manifest_path),
    )
    run = payload["runs"][0]
    old_run_id = run["run_id"]
    selection = dict(closure[0].selection)
    run_id = closure[0].logical_id
    run.update(
        {
            "run_id": run_id,
            "fill_id": "mask_only",
            "selection": selection,
        }
    )
    payload["pages"] = {run_id: payload["pages"].pop(old_run_id)}
    payload["matrix_sha256"] = _sha(matrix_path)
    payload["closure_ledger"] = [closure[0].as_record()]
    payload["logical_inventory_sha256"] = hashlib.sha256(
        json.dumps(
            [{"logical_id": run_id, "selection": selection}],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    records: list[dict[str, object]] = []
    complete: list[str] = []
    for run in payload["runs"]:
        run_id = run["run_id"]
        for page in payload["pages"][run_id]:
            page_id = page["page_id"]
            canonical = page["canonical_statistics"]
            edit = np.zeros((8, 10), np.uint8)
            edit[2:4, 3:5] = 255
            final = edit.copy()
            candidate = np.full((8, 10, 3), 200, np.uint8)
            for role, value, field in (
                (
                    "detector_seed_mask",
                    edit,
                    "detector_seed_mask_pixel_sha256",
                ),
                ("edit_mask", edit, "output_edit_mask_pixel_sha256"),
                ("final_mask", final, "final_mask_pixel_sha256"),
                ("candidate_image", candidate, "candidate_pixel_sha256"),
            ):
                path = artifact.parent / f"factorized-{role}.png"
                assert cv2.imwrite(str(path), value)
                decoded = cv2.imread(
                    str(path),
                    cv2.IMREAD_COLOR if role == "candidate_image" else cv2.IMREAD_GRAYSCALE,
                )
                assert decoded is not None
                pixel_sha = hashlib.sha256(
                    np.ascontiguousarray(decoded).tobytes()
                ).hexdigest()
                canonical[field] = pixel_sha
                record = {
                    "run_id": run_id,
                    "page_id": page_id,
                    "role": role,
                    "relative_path": path.name,
                    "artifact_sha256": _sha(path),
                    "pixel_sha256": pixel_sha,
                    "shape": list(decoded.shape),
                    "dtype": str(decoded.dtype),
                }
                if role != "candidate_image":
                    record["foreground_pixel_count"] = int(np.count_nonzero(decoded))
                records.append(record)
            canonical.update(
                {
                    "required_skip": False,
                    "target_pixel_count": 4,
                    "target_edit_pixel_count": 4,
                    "target_instance_seed_scores": [
                        {
                            "instance_id": "target",
                            "semantic_role": "dialogue_bubble",
                            "seeded": True,
                        }
                    ],
                    "target_instance_edit_scores": [
                        {"instance_id": "target", "coverage": 1.0}
                    ],
                    "edit_pixel_count": 4,
                    "conditional_hybrid_overlap_conflict_pixel_count": 0,
                    "authoritative_region_overlap_pixel_count": 0,
                    "authoritative_overlap_narrow_verified": False,
                    "residue_gate_applicable": False,
                }
            )
            page["canonical_statistics_sha256"] = hashlib.sha256(
                json.dumps(
                    canonical,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        run["metrics"] = aggregate_factorized_page_statistics(
            payload["pages"][run_id]
        )
        run["status"] = "pareto"
        run["closure_reason"] = ""
        complete.append(run_id)
    records.sort(key=lambda row: (row["run_id"], row["page_id"], row["role"]))
    canonical_inventory = {"records": records, "complete_run_ids": complete}
    inventory = {
        "schema_version": "inpaint-factorized-output-artifact-inventory-v1",
        **canonical_inventory,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                canonical_inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    inventory_path = _write_json(
        artifact.parent / "factorized-output-artifact-inventory.json", inventory
    )
    payload["output_artifact_inventory"] = {
        "relative_path": inventory_path.name,
        "artifact_sha256": _sha(inventory_path),
        "inventory_sha256": inventory["inventory_sha256"],
        "artifact_count": len(records),
        "complete_run_ids": complete,
    }
    registry_model_sha = str(
        ModelDownloader.registry[ModelID.LAMA_LARGE_512PX].sha256[0]
    )
    runtime_identity = {
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "tracked_worktree_clean": False,
        "tracked_worktree_diff_sha256": "d" * 64,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "opencv_version": cv2.__version__,
        "requested_device": "cpu",
        "requested_precision": "uint8_binary",
        "inpaint_size": 0,
        "vram_measurement_scope": "page_reset_then_run_max",
        "lama_model_asset_id": "lama_large_512px",
        "lama_model_registry_sha256": registry_model_sha,
        "lama_model_present": False,
        "lama_model_sha256": "",
        "torch_version": "",
        "torch_cuda_version": "",
        "cudnn_version": None,
        "cuda_available": False,
        "gpu_name": "",
    }
    runner_snapshot = artifact.parent / "runtime-runner.py"
    runner_source = ROOT / "scripts" / "benchmark_inpaint_factorized_v3.py"
    runner_snapshot.write_bytes(runner_source.read_bytes())
    runner_patch_path = artifact.parent / "runtime-runner.patch"
    runner_patch_path.write_bytes(
        subprocess.check_output(
            [
                "git",
                "diff",
                "--binary",
                "HEAD",
                "--",
                "scripts/benchmark_inpaint_factorized_v3.py",
            ],
            cwd=ROOT,
        )
    )
    patch_path = artifact.parent / "runtime.patch"
    patch_path.write_bytes(runner_patch_path.read_bytes())
    tracked_worktree_clean = not patch_path.read_bytes()
    runtime_identity["tracked_worktree_clean"] = tracked_worktree_clean
    source_records = [
        {
            "role": "runner_source_snapshot",
            "source_path": "scripts/benchmark_inpaint_factorized_v3.py",
            "relative_path": runner_snapshot.name,
            "artifact_sha256": _sha(runner_snapshot),
            "source_bytes_sha256": _sha(runner_snapshot),
        },
        {
            "role": "tracked_diff_patch",
            "relative_path": patch_path.name,
            "artifact_sha256": _sha(patch_path),
            "byte_count": len(patch_path.read_bytes()),
        },
        {
            "role": "runner_diff_patch",
            "source_path": "scripts/benchmark_inpaint_factorized_v3.py",
            "relative_path": runner_patch_path.name,
            "artifact_sha256": _sha(runner_patch_path),
            "byte_count": len(runner_patch_path.read_bytes()),
        },
    ]
    runtime_identity["tracked_worktree_diff_sha256"] = _sha(patch_path)
    source_canonical = {
        "code_commit": runtime_identity["code_commit"],
        "tracked_worktree_clean": tracked_worktree_clean,
        "records": source_records,
        "lama_model": {
            "asset_id": "lama_large_512px",
            "registry_expected_sha256": registry_model_sha,
            "actual_pre_sha256": "",
            "actual_post_sha256": "",
            "present_pre": False,
            "present_post": False,
        },
    }
    source_inventory = {
        "schema_version": "inpaint-factorized-runtime-source-inventory-v1",
        **source_canonical,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                source_canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    source_inventory_path = _write_json(
        artifact.parent / "factorized-runtime-source-inventory.json",
        source_inventory,
    )
    source_binding = {
        "role": "runtime_source_inventory",
        "relative_path": source_inventory_path.name,
        "artifact_sha256": _sha(source_inventory_path),
        "inventory_sha256": source_inventory["inventory_sha256"],
    }
    runtime_runs = []
    for run in payload["runs"]:
        pages = []
        for page in payload["pages"][run["run_id"]]:
            canonical = page["canonical_statistics"]
            pages.append(
                {
                    "page_id": page["page_id"],
                    "summary": {
                        field: canonical.get(field)
                        for field in (
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
                    },
                    "inference_events": [],
                }
            )
        runtime_runs.append(
            {
                "run_id": run["run_id"],
                "runtime_identity": runtime_identity,
                "pages": pages,
                "aggregate": {
                    field: run["metrics"].get(field)
                    for field in (
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
                },
            }
        )
    runtime_canonical = {
        "runtime_identity": runtime_identity,
        "runtime_source_inventory": source_binding,
        "runs": runtime_runs,
        "complete_run_ids": complete,
        "positive_lama_inference_count": 0,
    }
    runtime_ledger = {
        "schema_version": "inpaint-factorized-runtime-evidence-v1",
        **runtime_canonical,
        "ledger_sha256": hashlib.sha256(
            json.dumps(
                runtime_canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    runtime_path = _write_json(
        artifact.parent / "factorized-runtime-evidence.json", runtime_ledger
    )
    payload["runtime_evidence_ledger"] = {
        "role": "runtime_evidence",
        "relative_path": runtime_path.name,
        "artifact_sha256": _sha(runtime_path),
        "ledger_sha256": runtime_ledger["ledger_sha256"],
        "runtime_source_inventory_sha256": source_inventory[
            "inventory_sha256"
        ],
        "complete_run_ids": complete,
        "positive_lama_inference_count": 0,
    }
    payload["runtime_source_inventory"] = source_binding
    return _write_json(artifact, payload)


def _reseal_test_runtime_ledger(
    artifact: Path,
    payload: dict[str, object],
    ledger: dict[str, object],
) -> None:
    binding = payload["runtime_evidence_ledger"]
    assert isinstance(binding, dict)
    canonical = {
        "runtime_identity": ledger["runtime_identity"],
        "runtime_source_inventory": ledger["runtime_source_inventory"],
        "runs": ledger["runs"],
        "complete_run_ids": ledger["complete_run_ids"],
        "positive_lama_inference_count": ledger[
            "positive_lama_inference_count"
        ],
    }
    ledger["ledger_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    ledger_path = artifact.parent / str(binding["relative_path"])
    _write_json(ledger_path, ledger)
    binding["artifact_sha256"] = _sha(ledger_path)
    binding["ledger_sha256"] = ledger["ledger_sha256"]
    binding["complete_run_ids"] = ledger["complete_run_ids"]
    binding["positive_lama_inference_count"] = ledger[
        "positive_lama_inference_count"
    ]
    _write_json(artifact, payload)


def _fusion_artifact(tmp_path: Path, manifest: Path) -> Path:
    page = {
        "page_id": "p",
        "target_pixel_count": 10,
        "target_edit_pixel_count": 9,
        "target_instance_scores": [
            {"instance_id": "i", "seeded": False, "coverage": 0.9}
        ],
        "protected_edit_overlap": 0,
        "ambiguous_edit_overlap": 0,
        "preserve_edit_overlap": 0,
        "ownership_leak_pixel_count": 0,
        "false_edit_pixel_count": 0,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "target_mask_provenance": "source_only_v4",
        "output_claim_mask_pixel_sha256": SHA_A,
        "output_edit_mask_pixel_sha256": SHA_A,
    }
    metrics = aggregate_fusion_page_statistics([page])
    output_sha = str(metrics["output_mask_set_sha256"])
    spec = {
        "schema_version": "inpaint-detector-fusion-spec-v4",
        "candidates": {"manga109_text": {"templates": {"raw": "unused"}}},
    }
    spec_path = _write_json(tmp_path / "fusion-spec.json", spec)
    selection = {
        "run_id": "manga109_text",
        "fusion": "single",
        "primary": "manga109_text",
        "secondary": "",
    }
    inventory = [{"logical_id": "manga109_text", "selection": selection}]
    logical_sha = hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return _write_json(
        tmp_path / "fusion.json",
        {
            "schema_version": "inpaint-detector-fusion-results-v4",
            "manifest_sha256": _sha(manifest),
            "spec_sha256": _sha(spec_path),
            "logical_inventory_sha256": logical_sha,
            "logical_combination_count": 1,
            "physical_output_count": 1,
            "unaccounted_combination_count": 0,
            "page_ids": ["p"],
            "closure_ledger": [
                {
                    "logical_id": "manga109_text",
                    "selection": selection,
                    "closure_state": "executed",
                    "reason": "",
                    "content_sha256": output_sha,
                    "reused_from": "",
                }
            ],
            "runs": [
                {
                    "run_id": "manga109_text",
                    "fusion": "single",
                    "primary": "manga109_text",
                    "secondary": "",
                    "trigger": "",
                    "oracle_only": False,
                    "status": "dominated",
                    "closure_reason": "hard_gate_failed",
                    "metrics": metrics,
                }
            ],
            "pages": {"manga109_text": [page]},
        },
    )


def _attach_fusion_output_inventory(artifact: Path) -> Path:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    run_id = run["run_id"]
    page = payload["pages"][run_id][0]
    edit = np.zeros((8, 10), np.uint8)
    edit[2:4, 3:5] = 255
    decoded_by_role: dict[str, np.ndarray] = {}
    records: list[dict[str, object]] = []
    for role in ("claim_mask", "edit_mask"):
        path = artifact.parent / f"fusion-{role}.png"
        assert cv2.imwrite(str(path), edit)
        decoded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert decoded is not None
        decoded_by_role[role] = decoded
        pixel_sha = hashlib.sha256(
            np.ascontiguousarray(decoded).tobytes()
        ).hexdigest()
        records.append(
            {
                "run_id": run_id,
                "page_id": page["page_id"],
                "role": role,
                "relative_path": path.name,
                "artifact_sha256": _sha(path),
                "pixel_sha256": pixel_sha,
                "shape": list(decoded.shape),
                "dtype": str(decoded.dtype),
                "foreground_pixel_count": 4,
            }
        )
    pixel_sha = records[0]["pixel_sha256"]
    page.update(
        {
            "target_pixel_count": 4,
            "target_edit_pixel_count": 4,
            "target_instance_scores": [
                {"instance_id": "target", "seeded": True, "coverage": 1.0}
            ],
            "edit_pixel_count": 4,
            "output_claim_mask_pixel_sha256": pixel_sha,
            "output_edit_mask_pixel_sha256": pixel_sha,
        }
    )
    metrics = aggregate_fusion_page_statistics([page])
    run["metrics"] = metrics
    run["status"] = "family_complete"
    run["closure_reason"] = ""
    payload["closure_ledger"][0]["content_sha256"] = metrics[
        "output_mask_set_sha256"
    ]
    records.sort(key=lambda row: (row["run_id"], row["page_id"], row["role"]))
    canonical_inventory = {"records": records, "complete_run_ids": [run_id]}
    inventory = {
        "schema_version": "inpaint-fusion-output-artifact-inventory-v1",
        **canonical_inventory,
        "inventory_sha256": hashlib.sha256(
            json.dumps(
                canonical_inventory,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    inventory_path = _write_json(
        artifact.parent / "fusion-output-artifact-inventory.json", inventory
    )
    payload["output_artifact_inventory"] = {
        "relative_path": inventory_path.name,
        "artifact_sha256": _sha(inventory_path),
        "inventory_sha256": inventory["inventory_sha256"],
        "artifact_count": len(records),
        "complete_run_ids": [run_id],
    }
    return _write_json(artifact, payload)


def _stage1_artifact(
    tmp_path: Path,
    manifest: Path,
    *,
    candidate: str = "ctd-synthetic-low-contrast-finetune-v4",
    variant: str = "raw",
) -> Path:
    page = _stage1_page()
    artifact_path = tmp_path / f"stage1-{candidate}-{variant}.json"
    masks = {
        "raw": np.zeros((8, 10), np.uint8),
        "refined": np.zeros((8, 10), np.uint8),
        "dilated": np.pad(
            np.full((2, 2), 255, np.uint8), ((3, 3), (4, 4))
        ),
        "positive": np.zeros((8, 10), np.uint8),
    }
    records: list[dict[str, object]] = []
    identities: dict[str, str] = {}
    for output_variant in ("raw", "refined", "dilated"):
        mask = masks[output_variant]
        path = tmp_path / f"{candidate}-{variant}-{output_variant}-p.png"
        assert cv2.imwrite(str(path), mask)
        binary_sha = mask_sha256(mask)
        record = {
            "page_id": "p",
            "role": "native_detector_mask",
            "variant": output_variant,
            "relative_path": path.name,
            "artifact_sha256": _sha(path),
            "binary_mask_sha256": binary_sha,
            "pixel_count": int(cv2.countNonZero(mask)),
        }
        records.append(record)
        identities[output_variant] = hashlib.sha256(
            json.dumps(
                [{
                    "page_id": "p",
                    "binary_mask_sha256": binary_sha,
                    "pixel_count": record["pixel_count"],
                }],
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    positive_path = tmp_path / f"{candidate}-{variant}-positive-p.png"
    assert cv2.imwrite(str(positive_path), masks["positive"])
    positive_binary_sha = mask_sha256(masks["positive"])
    positive_record = {
        "page_id": "p",
        "role": "positive_edit_mask",
        "variant": variant,
        "relative_path": positive_path.name,
        "artifact_sha256": _sha(positive_path),
        "binary_mask_sha256": positive_binary_sha,
        "pixel_count": 0,
    }
    records.append(positive_record)
    positive_identity = hashlib.sha256(
        json.dumps(
            [{
                "page_id": "p",
                "binary_mask_sha256": positive_binary_sha,
                "pixel_count": 0,
            }],
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    records.sort(key=lambda row: (
        str(row["role"]), str(row["variant"]), str(row["page_id"])
    ))
    inventory = {
        "schema_version": "inpaint-detector-output-artifact-inventory-v1",
        "records": records,
    }
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            records,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    inventory_path = _write_json(
        tmp_path / f"stage1-{candidate}-{variant}-output-inventory.json",
        inventory,
    )
    return _write_json(
        artifact_path,
        {
            "schema_version": "inpaint-detector-bakeoff-stage1-v1",
            "manifest_sha256": _sha(manifest),
            "candidate": candidate,
            "variant": variant,
            "model": {"sha256": SHA_A},
            "role_candidate": {
                "candidate_id": candidate,
                "provider": "provider",
                "role": "seed",
                "variant": "native-bundle-v2",
                "code_commit": "deadbeef",
                "model_sha256": SHA_B,
                "runtime_provider": "cpu",
                "preprocessing_contract_sha256": SHA_C,
            },
            "variant_output_identity": {
                "raw": {
                    "output_mask_set_sha256": identities["raw"],
                    "page_count": 1,
                    "provenance": "native_finetuned_ctd_text_seg",
                    "independent_output": True,
                },
                "refined": {
                    "output_mask_set_sha256": identities["refined"],
                    "page_count": 1,
                    "provenance": "exact_identity_reuse",
                    "independent_output": False,
                    "source_variant": "raw",
                    "source_output_mask_set_sha256": identities["raw"],
                },
                "dilated": {
                    "output_mask_set_sha256": identities["dilated"],
                    "page_count": 1,
                    "provenance": "elliptical_native3_from_raw",
                    "independent_output": True,
                    "source_variant": "raw",
                },
            },
            "positive_edit_output_identity": {
                "output_mask_set_sha256": positive_identity,
                "page_count": 1,
            },
            "output_artifact_inventory": {
                "relative_path": inventory_path.name,
                "artifact_sha256": _sha(inventory_path),
                "inventory_sha256": inventory["inventory_sha256"],
                "artifact_count": len(records),
            },
            "summary": _summary(),
            "pages": [page],
        },
    )


def _registry(tmp_path: Path, family: str = "current-ctd", variants: tuple[str, ...] = ("raw",)) -> Path:
    role = "seed" if family != "semantic-policy" else "semantic"
    return _write_json(
        tmp_path / "registry.json",
        {
            "schema_version": "inpaint-method-family-registry-v4",
            "families": [
                {
                    "family_id": family,
                    "role": role,
                    "evaluation_scopes": ["e1"],
                    "variants": list(variants),
                }
            ],
        },
    )


def _requirements(family: str = "current-ctd", variants: tuple[str, ...] = ("raw",)) -> tuple[MethodVariantRequirement, ...]:
    return tuple(MethodVariantRequirement(family, "seed", variant, "e1") for variant in variants)


def test_artifact_disposition_is_derived_and_content_is_bound(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    rows = accounted_evidence_from_artifact(
        _requirements(),
        artifact_path=_factorized_artifact(tmp_path, manifest),
        scope_manifest_path=manifest,
        family_id="current-ctd",
        variant_ids=frozenset({"raw"}),
        evaluation_scope="e1",
        upstream_contract_path=tmp_path / "matrix.json",
    )

    assert rows[0]["disposition"] == "dominated"
    assert rows[0]["reason"] == "hard_gate_failed"
    assert len(str(rows[0]["content_sha256"])) == 64
    assert rows[0]["content_identity_kind"] == "exact_output"


def test_artifact_rejects_unregistered_or_unproved_variant(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    with pytest.raises(ValueError, match="not registered"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"native3"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )
    with pytest.raises(ValueError, match="not declared"):
        accounted_evidence_from_artifact(
            _requirements(variants=("raw", "refined")), artifact_path=artifact,
            scope_manifest_path=manifest, family_id="current-ctd",
            variant_ids=frozenset({"refined"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_factorized_artifact_rejects_truncated_ledger_pages_and_counts(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["closure_ledger"] = []
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="logical count"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )
    payload = json.loads(_factorized_artifact(tmp_path, manifest).read_text(encoding="utf-8"))
    payload["pages"] = {}
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="page results"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_oracle_run_cannot_be_upgraded_to_pareto(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(
        tmp_path, manifest, status="pareto", oracle_only=True
    )
    with pytest.raises(ValueError, match="oracle-only"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_minimal_factorized_metrics_cannot_claim_pareto(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest, status="pareto")
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["runs"][0]["metrics"] = {"page_count": 1}
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_factorized_page_statistics_cannot_be_rebound_to_declared_metrics(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run_id = payload["runs"][0]["run_id"]
    row = payload["pages"][run_id][0]
    row["canonical_statistics"]["required_skip"] = False
    row["canonical_statistics_sha256"] = hashlib.sha256(
        json.dumps(
            row["canonical_statistics"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_artifact_without_exact_upstream_matrix_is_rejected(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    with pytest.raises(ValueError, match="exact upstream matrix/spec"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
        )


def test_factorized_artifact_rejects_invented_matrix_sha(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["matrix_sha256"] = SHA_A
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="matrix SHA differs"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_pareto_requires_sealed_output_artifact_inventory(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest, status="pareto")
    with pytest.raises(ValueError, match="complete output artifact inventory"):
        accounted_evidence_from_artifact(
            _requirements(variants=("raw", "refined")),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_finalist_reopens_sealed_candidate_and_mask_bytes(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    rows = accounted_evidence_from_artifact(
        _requirements(),
        artifact_path=artifact,
        scope_manifest_path=manifest,
        family_id="current-ctd",
        variant_ids=frozenset({"raw"}),
        evaluation_scope="e1",
        upstream_contract_path=tmp_path / "matrix.json",
    )
    assert rows[0]["disposition"] == "pareto"

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    binding = payload["output_artifact_inventory"]
    inventory = json.loads(
        (artifact.parent / binding["relative_path"]).read_text(encoding="utf-8")
    )
    candidate = next(
        row for row in inventory["records"] if row["role"] == "candidate_image"
    )
    (artifact.parent / candidate["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output artifact file SHA differs"):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_finalist_requires_sealed_runtime_evidence(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload.pop("runtime_evidence_ledger")
    _write_json(artifact, payload)

    with pytest.raises(ValueError, match="complete runtime evidence ledger"):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_runtime_ledger_rejects_coordinated_result_telemetry_tamper(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    page = payload["pages"][run["run_id"]][0]
    canonical = page["canonical_statistics"]
    canonical["runtime_seconds"] = 9.0
    page["canonical_statistics_sha256"] = hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    run["metrics"] = aggregate_factorized_page_statistics([page])
    _write_json(artifact, payload)

    with pytest.raises(ValueError, match="runtime page summary differs"):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_runtime_ledger_rejects_omitted_run_after_reseal(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    binding = payload["runtime_evidence_ledger"]
    ledger_path = artifact.parent / binding["relative_path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["runs"] = []
    ledger["complete_run_ids"] = []
    _reseal_test_runtime_ledger(artifact, payload, ledger)

    with pytest.raises(
        ValueError, match="runtime evidence must cover every result run exactly"
    ):
        _validate_runtime_evidence_ledger(
            payload,
            artifact,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({payload["runs"][0]["run_id"]}),
        )


def test_factorized_runtime_source_rejects_coordinated_clean_identity_reseal(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    source_binding = payload["runtime_source_inventory"]
    source_path = artifact.parent / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    tampered_clean = not bool(source["tracked_worktree_clean"])
    source["tracked_worktree_clean"] = tampered_clean
    source_canonical = {
        "code_commit": source["code_commit"],
        "tracked_worktree_clean": source["tracked_worktree_clean"],
        "records": source["records"],
        "lama_model": source["lama_model"],
    }
    source["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            source_canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(source_path, source)
    source_binding["artifact_sha256"] = _sha(source_path)
    source_binding["inventory_sha256"] = source["inventory_sha256"]

    runtime_binding = payload["runtime_evidence_ledger"]
    runtime_path = artifact.parent / runtime_binding["relative_path"]
    ledger = json.loads(runtime_path.read_text(encoding="utf-8"))
    ledger["runtime_identity"]["tracked_worktree_clean"] = tampered_clean
    ledger["runtime_source_inventory"] = source_binding
    for run in ledger["runs"]:
        run["runtime_identity"] = ledger["runtime_identity"]
    runtime_binding["runtime_source_inventory_sha256"] = source[
        "inventory_sha256"
    ]
    _reseal_test_runtime_ledger(artifact, payload, ledger)

    with pytest.raises(
        ValueError, match="tracked worktree identity differs from patch bytes"
    ):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_factorized_runtime_source_rejects_coordinated_runner_reseal(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    source_binding = payload["runtime_source_inventory"]
    source_path = artifact.parent / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    runner_record = next(
        row for row in source["records"] if row["role"] == "runner_source_snapshot"
    )
    runner_patch_record = next(
        row for row in source["records"] if row["role"] == "runner_diff_patch"
    )
    runner_path = artifact.parent / runner_record["relative_path"]
    runner_path.write_bytes(runner_path.read_bytes() + b"\n# coordinated tamper\n")
    runner_record["artifact_sha256"] = _sha(runner_path)
    runner_record["source_bytes_sha256"] = _sha(runner_path)
    baseline = subprocess.check_output(
        [
            "git",
            "show",
            f"{source['code_commit']}:scripts/benchmark_inpaint_factorized_v3.py",
        ],
        cwd=ROOT,
    )
    baseline_path = tmp_path / "baseline-runner.py"
    tampered_path = tmp_path / "tampered-runner.py"
    baseline_path.write_bytes(baseline)
    tampered_path.write_bytes(runner_path.read_bytes())
    tampered_patch = subprocess.run(
        [
            "git",
            "diff",
            "--no-index",
            "--binary",
            "--src-prefix=a/scripts/",
            "--dst-prefix=b/scripts/",
            str(baseline_path),
            str(tampered_path),
        ],
        cwd=tmp_path,
        stdout=subprocess.PIPE,
        check=False,
    ).stdout.replace(
        str(baseline_path).replace("\\", "/").encode(),
        b"benchmark_inpaint_factorized_v3.py",
    ).replace(
        str(tampered_path).replace("\\", "/").encode(),
        b"benchmark_inpaint_factorized_v3.py",
    )
    runner_patch_path = artifact.parent / runner_patch_record["relative_path"]
    runner_patch_path.write_bytes(tampered_patch)
    runner_patch_record["artifact_sha256"] = _sha(runner_patch_path)
    runner_patch_record["byte_count"] = len(tampered_patch)
    source_canonical = {
        "code_commit": source["code_commit"],
        "tracked_worktree_clean": source["tracked_worktree_clean"],
        "records": source["records"],
        "lama_model": source["lama_model"],
    }
    source["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            source_canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(source_path, source)
    source_binding["artifact_sha256"] = _sha(source_path)
    source_binding["inventory_sha256"] = source["inventory_sha256"]

    runtime_binding = payload["runtime_evidence_ledger"]
    runtime_path = artifact.parent / runtime_binding["relative_path"]
    ledger = json.loads(runtime_path.read_text(encoding="utf-8"))
    ledger["runtime_source_inventory"] = source_binding
    runtime_binding["runtime_source_inventory_sha256"] = source[
        "inventory_sha256"
    ]
    _reseal_test_runtime_ledger(artifact, payload, ledger)

    with pytest.raises(
        ValueError, match="runner patch differs from tracked diff patch"
    ):
        _validate_runtime_evidence_ledger(
            payload,
            artifact,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({payload["runs"][0]["run_id"]}),
        )


def test_factorized_runtime_source_rejects_non_sha_post_model_identity(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    source_binding = payload["runtime_source_inventory"]
    source_path = artifact.parent / source_binding["relative_path"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["lama_model"]["present_post"] = True
    source["lama_model"]["actual_post_sha256"] = "not-a-sha"
    source_canonical = {
        "code_commit": source["code_commit"],
        "tracked_worktree_clean": source["tracked_worktree_clean"],
        "records": source["records"],
        "lama_model": source["lama_model"],
    }
    source["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            source_canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _write_json(source_path, source)
    source_binding["artifact_sha256"] = _sha(source_path)
    source_binding["inventory_sha256"] = source["inventory_sha256"]
    runtime_binding = payload["runtime_evidence_ledger"]
    runtime_path = artifact.parent / runtime_binding["relative_path"]
    ledger = json.loads(runtime_path.read_text(encoding="utf-8"))
    ledger["runtime_source_inventory"] = source_binding
    runtime_binding["runtime_source_inventory_sha256"] = source[
        "inventory_sha256"
    ]
    _reseal_test_runtime_ledger(artifact, payload, ledger)

    with pytest.raises(ValueError, match="invalid asset SHA"):
        _validate_runtime_evidence_ledger(
            payload,
            artifact,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({payload["runs"][0]["run_id"]}),
        )


def test_runtime_source_inventory_rehashes_model_post_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "lama.ckpt"
    model_path.write_bytes(b"pre")
    primary_path_calls = 0

    def unexpected_primary_path(_cls: object, _model: object) -> str:
        nonlocal primary_path_calls
        primary_path_calls += 1
        return str(model_path)

    monkeypatch.setattr(
        ModelDownloader,
        "primary_path",
        classmethod(unexpected_primary_path),
    )
    _sha256.cache_clear()
    pre_sha = _sha256(model_path)
    model_path.write_bytes(b"post")
    output = tmp_path / "runtime-output"
    output.mkdir()
    binding = _seal_runtime_source_inventory(
        output,
        runtime_identity={
            "code_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "lama_model_present": True,
            "lama_model_sha256": pre_sha,
        },
        lama_model_path=model_path,
    )
    source = json.loads(
        (output / binding["relative_path"]).read_text(encoding="utf-8")
    )
    assert source["lama_model"]["actual_pre_sha256"] == pre_sha
    assert source["lama_model"]["actual_post_sha256"] == hashlib.sha256(
        b"post"
    ).hexdigest()
    assert source["lama_model"]["actual_post_sha256"] != pre_sha
    assert primary_path_calls == 0


def test_fusion_finalist_reopens_sealed_edit_mask_bytes(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_fusion_output_inventory(
        _fusion_artifact(tmp_path, manifest)
    )
    requirements = (
        MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),
    )
    rows = accounted_evidence_from_artifact(
        requirements,
        artifact_path=artifact,
        scope_manifest_path=manifest,
        family_id="manga109-text",
        variant_ids=frozenset({"raw"}),
        evaluation_scope="e1",
        upstream_contract_path=tmp_path / "fusion-spec.json",
        expected_fusion_candidate_ids=frozenset({"manga109_text"}),
    )
    assert rows[0]["disposition"] == "family_complete"

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    binding = payload["output_artifact_inventory"]
    inventory = json.loads(
        (artifact.parent / binding["relative_path"]).read_text(encoding="utf-8")
    )
    edit = inventory["records"][0]
    (artifact.parent / edit["relative_path"]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="output artifact file SHA differs"):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="manga109-text",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset({"manga109_text"}),
        )


def test_factorized_finalist_rejects_self_consistent_tampered_coverage_facts(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_factorized_output_inventory(
        _factorized_artifact(tmp_path, manifest, status="pareto")
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    page = payload["pages"][run["run_id"]][0]
    page["canonical_statistics"]["target_pixel_count"] = 5
    page["canonical_statistics"]["target_edit_pixel_count"] = 5
    page["canonical_statistics_sha256"] = hashlib.sha256(
        json.dumps(
            page["canonical_statistics"],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    run["metrics"] = aggregate_factorized_page_statistics([page])
    _write_json(artifact, payload)

    with pytest.raises(
        ValueError, match="target_pixel_count differs from sealed source artifacts"
    ):
        accounted_evidence_from_artifact(
            _requirements(),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="current-ctd",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "matrix.json",
        )


def test_fusion_finalist_rejects_self_consistent_tampered_coverage_facts(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _attach_fusion_output_inventory(
        _fusion_artifact(tmp_path, manifest)
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run = payload["runs"][0]
    page = payload["pages"][run["run_id"]][0]
    page["target_pixel_count"] = 5
    page["target_edit_pixel_count"] = 5
    run["metrics"] = aggregate_fusion_page_statistics([page])
    payload["closure_ledger"][0]["content_sha256"] = run["metrics"][
        "output_mask_set_sha256"
    ]
    _write_json(artifact, payload)
    requirements = (
        MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),
    )

    with pytest.raises(
        ValueError, match="target_pixel_count differs from sealed source artifacts"
    ):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="manga109-text",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset({"manga109_text"}),
        )


def test_artifact_must_cover_every_scope_page_not_a_fake_one_page_subset(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"].append(_source_only_page(tmp_path, "q"))
    payload["page_count"] = 2
    payload["page_ids"] = ["p", "q"]
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        payload["pages"]
    )
    _write_json(manifest, payload)
    _seal_scope_manifest(manifest)
    artifact = _factorized_artifact(tmp_path, manifest)
    with pytest.raises(ValueError, match="page identity|page IDs differ"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_artifact_page_identity_must_exactly_match_scope(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    run_id = next(iter(payload["pages"]))
    payload["pages"][run_id][0]["page_id"] = "other"
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="page identity|page IDs differ|sufficient statistics"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_fusion_uses_exact_output_identity_and_validates_closure(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _fusion_artifact(tmp_path, manifest)
    requirements = (MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),)
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id="manga109-text", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
        , upstream_contract_path=tmp_path / "fusion-spec.json",
        expected_fusion_candidate_ids=frozenset({"manga109_text"}),
    )
    assert rows[0]["content_identity_kind"] == "exact_output"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["runs"][0]["metrics"]["output_mask_set_sha256"] = SHA_A
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="manga109-text", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset({"manga109_text"}),
        )


def test_fusion_page_statistics_reject_tampered_aggregate(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _fusion_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["pages"]["manga109_text"][0]["target_edit_pixel_count"] = 10
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        accounted_evidence_from_artifact(
            (MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="manga109-text",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset({"manga109_text"}),
        )


def test_fusion_generic_variants_require_canonical_provider_pair_inventory(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _fusion_artifact(tmp_path, manifest)
    requirements = (
        MethodVariantRequirement("detector-fusion", "seed", "single", "e1"),
    )
    with pytest.raises(ValueError, match="canonical registered provider/variant"):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="detector-fusion",
            variant_ids=frozenset({"single"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset(
                {"manga109_text", "current_ctd_raw"}
            ),
        )


@pytest.mark.parametrize(
    ("field", "value", "error"),
    (
        ("primary", "tampered_detector", "selection differs"),
        ("status", "family_complete", "declared status differs"),
        ("closure_reason", "", "closure reason differs"),
    ),
)
def test_fusion_rejects_tampered_selection_status_or_reason(
    tmp_path: Path,
    field: str,
    value: str,
    error: str,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _fusion_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["runs"][0][field] = value
    _write_json(artifact, payload)
    requirements = (
        MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),
    )
    with pytest.raises(ValueError, match=error):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="manga109-text",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
            upstream_contract_path=tmp_path / "fusion-spec.json",
            expected_fusion_candidate_ids=frozenset({"manga109_text"}),
        )


@pytest.mark.parametrize(
    ("artifact_variant", "registered_variant"),
    (("raw", "raw"), ("refined", "refined"), ("dilated", "native3")),
)
def test_stage1_finetune_variants_require_full_schema(
    tmp_path: Path, artifact_variant: str, registered_variant: str
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _stage1_artifact(tmp_path, manifest, variant=artifact_variant)
    requirements = _requirements("ctd-synthetic-finetune", (registered_variant,))
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id="ctd-synthetic-finetune", variant_ids=frozenset({registered_variant}),
        evaluation_scope="e1"
    )
    assert rows[0]["disposition"] == "dominated"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    del payload["pages"]
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="pages"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="ctd-synthetic-finetune", variant_ids=frozenset({registered_variant}),
            evaluation_scope="e1"
        )


def test_stage1_rejects_summary_tampered_away_from_page_aggregation(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _stage1_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["summary"] = _summary(passing=True)
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="canonical page aggregation"):
        accounted_evidence_from_artifact(
            _requirements("ctd-synthetic-finetune", ("raw",)),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="ctd-synthetic-finetune",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
        )


def test_stage1_rejects_written_output_mask_byte_tampering(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _stage1_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    binding = payload["output_artifact_inventory"]
    inventory = json.loads(
        (artifact.parent / binding["relative_path"]).read_text(encoding="utf-8")
    )
    native = next(
        row
        for row in inventory["records"]
        if row["role"] == "native_detector_mask" and row["variant"] == "raw"
    )
    (artifact.parent / native["relative_path"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="output mask file SHA differs"):
        accounted_evidence_from_artifact(
            _requirements("ctd-synthetic-finetune", ("raw",)),
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="ctd-synthetic-finetune",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
        )


def test_synthetic_refined_is_counted_only_as_exact_raw_identity_reuse(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _stage1_artifact(tmp_path, manifest)
    requirements = _requirements(
        "ctd-synthetic-finetune", ("raw", "refined")
    )
    rows = accounted_evidence_from_artifact(
        requirements,
        artifact_path=artifact,
        scope_manifest_path=manifest,
        family_id="ctd-synthetic-finetune",
        variant_ids=frozenset({"raw", "refined"}),
        evaluation_scope="e1",
    )
    by_variant = {str(row["variant_id"]): row for row in rows}
    assert by_variant["raw"]["closure_state"] == "executed"
    assert by_variant["refined"]["closure_state"] == "reused_by_sha"
    assert by_variant["refined"]["content_sha256"] == by_variant["raw"][
        "content_sha256"
    ]
    assert by_variant["refined"]["reused_from"].endswith("/raw/e1")

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["variant_output_identity"]["refined"][
        "output_mask_set_sha256"
    ] = SHA_C
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="exact raw identity reuse"):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="ctd-synthetic-finetune",
            variant_ids=frozenset({"refined"}),
            evaluation_scope="e1",
        )


def test_source_protection_requires_full_page_and_summary_contract(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / "protection.json",
        {
            "schema_version": "inpaint-source-protection-reapply-v3",
            "manifest_sha256": _sha(manifest),
            "candidate_id": "c19-accepted-seed-final-protect",
            "summary": _summary(passing=True),
            "pages": [_stage1_page(passing=True)],
        },
    )
    requirements = (
        MethodVariantRequirement("exact-protection", "protection", "C19", "e1"),
    )
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id="exact-protection", variant_ids=frozenset({"C19"}),
        evaluation_scope="e1"
    )
    assert rows[0]["disposition"] == "family_complete"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["summary"]["page_count"] = 2
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="canonical page aggregation"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="exact-protection", variant_ids=frozenset({"C19"}),
            evaluation_scope="e1"
        )


def test_semantic_disposition_is_artifact_derived_and_blocked_needs_probe(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    policy_ids = (
        "current_default",
        "detector_explicit_role",
        "ocr_semantic_hint",
        "explicit_role_consensus",
        "human_oracle",
    )
    logical_sha = hashlib.sha256(
        json.dumps(sorted(policy_ids), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()

    def policy(policy_id: str) -> dict[str, object]:
        is_current = policy_id == "current_default"
        decision = {
            "instance_id": "i",
            "truth_role": "dialogue_bubble",
            "truth_action": "translate_inpaint",
            "priority": "required",
            "predicted_role": (
                "ambiguous" if is_current else "dialogue_bubble"
            ),
            "predicted_action": (
                "review" if is_current else "translate_inpaint"
            ),
            "available": True,
            "reason": "",
        }
        metrics = aggregate_semantic_page_statistics(
            [{"page_id": "p", "decisions": [decision]}]
        )
        return {
            "policy_id": policy_id,
            "oracle_only": policy_id == "human_oracle",
            "status": "dominated" if is_current else "family_complete",
            "closure_reason": "semantic_hard_gate_failed" if is_current else "",
            "metrics": metrics,
            "page": {"page_id": "p", "decisions": [decision]},
        }

    artifact = _write_json(
        tmp_path / "semantic.json",
        {
            "schema_version": "inpaint-semantic-policy-results-v4",
            "manifest_sha256": _sha(manifest),
            "policy_count": len(policy_ids),
            "unaccounted_policy_count": 0,
            "page_ids": ["p"],
            "logical_inventory_sha256": logical_sha,
            "policies": [
                {key: value for key, value in policy(policy_id).items() if key != "page"}
                for policy_id in policy_ids
            ],
            "pages": {
                policy_id: [policy(policy_id)["page"]]
                for policy_id in policy_ids
            },
        },
    )
    requirements = (MethodVariantRequirement("semantic-policy", "semantic", "current_default", "e1"),)
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id="semantic-policy", variant_ids=frozenset({"current_default"}), evaluation_scope="e1"
    )
    assert rows[0]["disposition"] == "dominated"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["policies"][0]["status"] = "blocked_asset"
    payload["policies"][0]["metrics"]["unavailable_instance_count"] = 1
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="aggregate metrics differ"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="semantic-policy", variant_ids=frozenset({"current_default"}), evaluation_scope="e1"
        )


def test_minimal_semantic_metrics_cannot_claim_pareto(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / "semantic-malicious.json",
        {
            "schema_version": "inpaint-semantic-policy-results-v4",
            "manifest_sha256": _sha(manifest),
            "policy_count": 1,
            "unaccounted_policy_count": 0,
            "page_ids": ["p"],
            "logical_inventory_sha256": hashlib.sha256(
                json.dumps(
                    sorted(
                        (
                            "current_default",
                            "detector_explicit_role",
                            "ocr_semantic_hint",
                            "explicit_role_consensus",
                            "human_oracle",
                        )
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest(),
            "policies": [{
                "policy_id": "current_default",
                "oracle_only": False,
                "status": "pareto",
                "closure_reason": "",
                "metrics": {"instance_count": 1},
            }],
        },
    )
    requirements = (
        MethodVariantRequirement(
            "semantic-policy", "semantic", "current_default", "e1"
        ),
    )
    with pytest.raises(
        ValueError,
        match="full policy inventory|semantic metrics require",
    ):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="semantic-policy",
            variant_ids=frozenset({"current_default"}), evaluation_scope="e1"
        )


def test_blocked_asset_requires_hashed_matching_probe(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    requirements = (MethodVariantRequirement("sickzil", "seed", "raw", "e1"),)
    probe = _write_json(
        tmp_path / "probe.json",
        {
            "schema_version": "inpaint-blocked-asset-probe-v1",
            "family_id": "sickzil",
            "variant_ids": ["raw"],
            "evaluation_scope": "e1",
            "scope_manifest_sha256": _sha(manifest),
            "status": "blocked_asset",
            "target": "seed-detector",
            "provider": "tensorflow-v1",
            "asset_id": "official-sickzil-checkpoint",
            "reason_code": "official_asset_unavailable",
            "checks": [{
                "kind": "filesystem",
                "target": "managed/sickzil/checkpoint",
                "found": False,
                "status": "unavailable",
                "evidence": "checkpoint path absent",
            }],
        },
    )
    rows = blocked_asset_evidence(
        requirements, scope_manifest_path=manifest, blocker_probe_path=probe,
        family_id="sickzil", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
    )
    assert rows[0]["blocker_probe_sha256"] == _sha(probe)
    payload = json.loads(probe.read_text(encoding="utf-8"))
    payload["scope_manifest_sha256"] = SHA_A
    _write_json(probe, payload)
    with pytest.raises(ValueError, match="scope manifest mismatch"):
        blocked_asset_evidence(
            requirements, scope_manifest_path=manifest, blocker_probe_path=probe,
            family_id="sickzil", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
        )


@pytest.mark.parametrize(
    "check",
    (
        {},
        {
            "kind": "filesystem",
            "target": "managed/sickzil/checkpoint",
            "found": True,
            "status": "success",
            "evidence": "asset exists",
        },
        {
            "kind": "unknown_probe",
            "target": "managed/sickzil/checkpoint",
            "found": False,
            "evidence": "missing",
        },
    ),
)
def test_blocked_asset_rejects_empty_successful_or_unsupported_checks(
    tmp_path: Path,
    check: dict[str, object],
) -> None:
    manifest = _scope_manifest(tmp_path)
    requirements = (MethodVariantRequirement("sickzil", "seed", "raw", "e1"),)
    probe = _write_json(
        tmp_path / "bad-probe.json",
        {
            "schema_version": "inpaint-blocked-asset-probe-v1",
            "family_id": "sickzil",
            "variant_ids": ["raw"],
            "evaluation_scope": "e1",
            "scope_manifest_sha256": _sha(manifest),
            "status": "blocked_asset",
            "target": "seed-detector",
            "provider": "tensorflow-v1",
            "asset_id": "official-sickzil-checkpoint",
            "reason_code": "official_asset_unavailable",
            "checks": [check],
        },
    )
    with pytest.raises(ValueError, match="check|unsupported"):
        blocked_asset_evidence(
            requirements,
            scope_manifest_path=manifest,
            blocker_probe_path=probe,
            family_id="sickzil",
            variant_ids=frozenset({"raw"}),
            evaluation_scope="e1",
        )


def test_merge_rejects_duplicate_and_scope_rebinding_even_on_replace() -> None:
    row = {"family_id": "x", "role": "seed", "variant_id": "raw",
           "evaluation_scope": "e1", "scope_manifest_sha256": SHA_A}
    with pytest.raises(ValueError, match="duplicate"):
        merge_method_evidence((row, row), ())
    changed = {**row, "scope_manifest_sha256": SHA_B}
    with pytest.raises(ValueError, match="cannot rebind"):
        merge_method_evidence((row,), (changed,), allow_replace=True)
    assert merge_method_evidence((row,), (row,)) == (row,)


def test_update_binds_scope_once_and_rejects_mixed_manifest_revision(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    evidence = tmp_path / "evidence.json"
    payload = update_evidence(
        registry_path=registry, evidence_path=evidence, artifact_path=artifact,
        scope_manifest_path=manifest, family_id="current-ctd",
        variant_ids=frozenset({"raw"}), evaluation_scope="e1"
        , upstream_contract_path=tmp_path / "matrix.json"
    )
    _write_json(evidence, payload)
    assert payload["scope_manifests"]["e1"]["sha256"] == _sha(manifest)
    changed_manifest = _scope_manifest(tmp_path, corpus="changed")
    changed_artifact = _factorized_artifact(tmp_path, changed_manifest)
    with pytest.raises(ValueError, match="rebinding is forbidden"):
        update_evidence(
            registry_path=registry, evidence_path=evidence, artifact_path=changed_artifact,
            scope_manifest_path=changed_manifest, family_id="current-ctd",
            variant_ids=frozenset({"raw"}), evaluation_scope="e1", allow_replace=True
            , upstream_contract_path=tmp_path / "matrix.json"
        )


def test_build_closure_records_input_hashes_and_scope_binding(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    evidence = tmp_path / "evidence.json"
    _write_json(
        evidence,
        update_evidence(
            registry_path=registry, evidence_path=evidence, artifact_path=artifact,
            scope_manifest_path=manifest, family_id="current-ctd",
            variant_ids=frozenset({"raw"}), evaluation_scope="e1"
            , upstream_contract_path=tmp_path / "matrix.json"
        ),
    )
    result = build_closure(registry, evidence)
    assert result["registry_sha256"] == _sha(registry)
    assert result["evidence_sha256"] == _sha(evidence)
    assert result["scope_manifests"]["e1"] == scope_manifest_binding(manifest)


def test_family_status_and_counts_do_not_turn_active_or_blocked_into_success() -> None:
    requirements = (
        MethodVariantRequirement("family", "seed", "active", "e1"),
        MethodVariantRequirement("family", "seed", "blocked", "e1"),
    )
    evidence = (
        MethodVariantEvidence(
            "family", "seed", "active", "e1", "executed", "active",
            artifact_sha256=SHA_A, scope_manifest_sha256=SHA_C,
            content_sha256=SHA_B, content_identity_kind="artifact_record",
        ),
        MethodVariantEvidence(
            "family", "seed", "blocked", "e1", "blocked_asset", "blocked_asset",
            reason="asset_missing", scope_manifest_sha256=SHA_C,
            blocker_probe_sha256=SHA_A,
        ),
    )
    result = build_method_family_closure(
        requirements, evidence,
        scope_manifests={"e1": {"sha256": SHA_C,
                                "schema_version": "inpaint-factorized-source-manifest-v4",
                                "corpus_id": "e1", "split_role": "development"}},
    )
    family = result["families"][0]
    assert family["status"] == "active"
    assert family["family_complete"] is False
    assert family["active_variant_count"] == 1
    assert family["blocked_variant_count"] == 1
    assert result["all_families_complete"] is False


def test_blocked_only_family_is_accounted_but_not_complete() -> None:
    requirement = MethodVariantRequirement("family", "seed", "blocked", "e1")
    evidence = MethodVariantEvidence(
        "family", "seed", "blocked", "e1", "blocked_asset", "blocked_asset",
        reason="asset_missing", scope_manifest_sha256=SHA_C,
        blocker_probe_sha256=SHA_A,
    )
    result = build_method_family_closure(
        (requirement,), (evidence,),
        scope_manifests={"e1": {"sha256": SHA_C,
                                "schema_version": "inpaint-factorized-source-manifest-v4",
                                "corpus_id": "e1", "split_role": "development"}},
    )
    assert result["all_requirements_accounted"] is True
    assert result["all_families_complete"] is False
    assert result["families"][0]["status"] == "blocked_asset"
    assert result["families"][0]["family_complete"] is False


@pytest.mark.parametrize("tamper", ("source_sha", "content_sha", "metrics"))
def test_generic_role_result_cannot_self_attest_unverified_upstream(
    tmp_path: Path,
    tamper: str,
) -> None:
    manifest = _scope_manifest(tmp_path)
    source = _write_json(
        tmp_path / "source.json",
        {
            "schema_version": "invented-source-result-v1",
            "manifest_sha256": _sha(manifest),
            "pages": [{"page_id": "p"}],
            "results": [{"result_id": "source-result"}],
        },
    )
    artifact = _write_json(
        tmp_path / "role-self-attested.json",
        {
            "schema_version": "inpaint-method-role-results-v4",
            "manifest_sha256": _sha(manifest),
            "source_artifact_sha256": _sha(source),
            "source_artifact_schema_version": "invented-source-result-v1",
            "record_count": 1,
            "page_count": 1,
            "unaccounted_record_count": 0,
            "pages": [{"page_id": "p"}],
            "records": [
                {
                    "family_id": "ownership",
                    "variant_id": "rtdetr_pixel",
                    "source_result_id": "source-result",
                    "content_sha256": SHA_B,
                    "page_ids": ["p"],
                    "oracle_only": False,
                    "status": "family_complete",
                    "closure_reason": "",
                    "metrics": _summary(passing=True),
                }
            ],
        },
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    if tamper == "source_sha":
        payload["source_artifact_sha256"] = SHA_A
    elif tamper == "content_sha":
        payload["records"][0]["content_sha256"] = SHA_A
    else:
        payload["records"][0]["metrics"] = _summary(passing=True)
    _write_json(artifact, payload)
    requirements = (
        MethodVariantRequirement("ownership", "ownership", "rtdetr_pixel", "e1"),
    )
    with pytest.raises(ValueError, match="unsupported evidence artifact schema"):
        accounted_evidence_from_artifact(
            requirements,
            artifact_path=artifact,
            scope_manifest_path=manifest,
            family_id="ownership",
            variant_ids=frozenset({"rtdetr_pixel"}),
            evaluation_scope="e1",
        )


def test_registry_reports_unimplemented_generic_role_adapters_as_gaps() -> None:
    registry = json.loads(
        (ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "method_registry_v4.json").read_text(encoding="utf-8")
    )
    gaps = registry_evidence_adapter_gaps(requirements_from_registry(registry))
    assert {
        (row["family_id"], row["variant_id"])
        for row in gaps
    } == {
        ("ownership", "rtdetr_pixel"),
        ("ownership", "c13_reconciliation"),
        ("exact-protection", "pr4_exact"),
        ("exact-composite", "immutable_original_exact_mask"),
    }


def test_scope_manifest_requires_canonical_identity(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"sealed": True})
    with pytest.raises(ValueError, match="source-only"):
        scope_manifest_binding(path)


def test_scope_manifest_rejects_candidate_derived_annotations(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0]["target_inventory_independent"] = False
    _write_json(manifest, payload)
    _seal_scope_manifest(manifest)
    with pytest.raises(ValueError, match="lacks target_inventory_independent"):
        scope_manifest_binding(manifest)


@pytest.mark.parametrize("pages", ([], [{"page_id": "p"}, {"page_id": "p"}]))
def test_scope_manifest_rejects_empty_or_duplicate_page_inventory(
    tmp_path: Path, pages: list[dict[str, object]]
) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if pages:
        valid = _source_only_page(tmp_path, "p")
        pages = [valid, dict(valid)]
    payload["pages"] = pages
    _write_json(manifest, payload)
    _seal_scope_manifest(manifest)
    with pytest.raises(ValueError, match="non-empty|duplicate"):
        scope_manifest_binding(manifest)


@pytest.mark.parametrize(
    "field",
    ("path", "target_text_mask", "protected_structure_mask"),
)
def test_scope_manifest_rejects_source_target_or_protect_byte_tampering(
    tmp_path: Path,
    field: str,
) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    Path(payload["pages"][0][field]).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        scope_manifest_binding(manifest)
