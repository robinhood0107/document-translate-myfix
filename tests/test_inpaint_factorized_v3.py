from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pytest

import scripts.benchmark_inpaint_factorized_v3 as factorized_runner

from benchmarking.inpaint_detector_bakeoff.contracts import (
    binary_mask,
    CandidateMaskResult,
    DetectorBox,
    FactorizedRunRecord,
    RegionEvaluationSpec,
    RoleCandidateSpec,
    Stage1Page,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (
    PageMasks,
    RegionMasks,
    broad_route_false_positive_pixels,
    decide_bubble_route,
    detector_roi_trigger_mask,
    expand_detector_claim,
    fuse_detector_claims,
    load_page_masks,
    load_stage1_manifest,
    read_detector_cache,
    run_stage1,
    score_page,
    write_detector_cache,
    _read_image as read_stage1_image,
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (
    attach_reconstruction_control,
    assert_complete_closure_ledger,
    build_combination_closure_ledger,
    build_factorized_matrix,
    evaluate_relative_product_gate,
    fill_factorized_mask,
    reconstruction_error,
    select_pareto_records,
)
from benchmarking.inpaint_detector_bakeoff.silhouette import (
    ballons_native_clean_background,
    extract_ballons_native_interior,
    extract_pr2_validated_interior,
)
from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (
    _validate_runtime_evidence_ledger,
    validate_evidence_artifact,
)
from benchmarking.inpaint_detector_bakeoff.paired_target import (
    paired_old_text_proposal,
    source_extent_variants,
)
from scripts.benchmark_inpaint_factorized_v3 import (
    _annotation_masks,
    _fill_conditional_hybrid_regions,
    _prepare_closure_ledger,
    _declared_combinations,
    _route_fill_backend,
    _with_candidate_ownership,
    main as factorized_main,
)
from scripts.build_inpaint_factorized_manifest_v3 import build_manifest
from scripts.build_inpaint_factorized_manifest_v4 import build_manifest as build_manifest_v4
from scripts.attach_inpaint_relative_baseline_v32 import (
    main as attach_relative_baseline_main,
)
from scripts.seal_inpaint_product_baseline_v32 import seal_product_baseline
from scripts.build_inpaint_product_eval_manifest_v32 import (
    _canonical_manifest_sha256,
    build_product_eval_manifest,
)
from scripts.build_inpaint_relative_matrix_v32 import build_relative_matrix
from scripts.build_inpaint_minimal_closure_matrix_v32 import (
    build_minimal_closure_matrix,
)
from scripts.adjudicate_inpaint_balanced_preflight_v32 import (
    adjudicate_balanced_preflight,
)
from scripts.adjudicate_inpaint_fill_preflight_v32 import (
    adjudicate_fill_preflight,
)
from scripts.adjudicate_inpaint_v32_final import adjudicate_final_selection
from scripts.adjudicate_inpaint_relative_v32 import adjudicate_relative_product
from scripts.build_inpaint_development_source_index_v4 import build_source_index
from scripts.build_inpaint_independent_target_review_v4 import (
    build_independent_target_review,
)
from scripts.build_inpaint_source_proposals_v4 import propose_semantic_contract
from scripts.apply_inpaint_source_review_v4 import apply_source_review
from scripts.record_inpaint_source_review_v4 import record_source_review
from scripts.record_inpaint_independent_target_review_v4 import (
    record_independent_target_review,
)
from scripts.apply_inpaint_independent_target_review_v4 import (
    _empty_region,
    _read_mask as read_independent_review_mask,
    apply_independent_target_review,
    seal_independent_manifest,
)
from scripts.build_inpaint_independent_review_decisions_v4 import (
    build_review_decisions,
    extend_source_location_review_overrides,
)
from scripts.build_inpaint_factorized_matrix_v3 import build_matrix
from scripts.build_inpaint_fill_synthetic_v3 import build_synthetic_manifest
from scripts.build_inpaint_fill_oracle_matrix_v3 import build_fill_matrix
from scripts.build_inpaint_v3_contact_sheet import build_contact_sheet
from scripts.build_inpaint_v32_three_case_sheet import build_three_case_sheet
from scripts.export_inpaint_silhouette_router_v3 import export_candidates
from scripts.merge_inpaint_factorized_v3 import merge_results
from scripts.export_inpaint_silhouette_consensus_v4 import consensus_masks
from scripts.benchmark_inpaint_detector_fusions_v4 import (
    _logical_runs as detector_fusion_runs,
    run_fusion_matrix,
    select_seed_admission_run_ids,
)
from scripts.benchmark_inpaint_semantic_policies_v4 import score_semantic_policies
from scripts.audit_inpaint_detector_ceiling_v4 import audit_detector_ceiling
from scripts.build_inpaint_method_closure_v4 import build_closure
from scripts.build_inpaint_generalization_synthetic_v4 import (
    _shift_mask as shift_generalization_mask,
    build_synthetic_manifest as build_generalization_synthetic_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


def _relative_metrics(
    *,
    coverage: float,
    residue: float,
    mask_sha: str,
) -> dict[str, object]:
    return {
        "aggregate_target_coverage": coverage,
        "aggregate_residue_score": residue,
        "protected_structure_overlap": 0,
        "protected_structure_changed": 0,
        "ambiguous_structure_overlap": 0,
        "ambiguous_structure_changed": 0,
        "preserve_edit_overlap": 0,
        "ownership_leak_pixel_count": 0,
        "corner_edit_overlap_pixel_count": 0,
        "outside_final_changed": 0,
        "broad_route_false_positive": 0,
        "no_edit_false_edit": 0,
        "required_skip_count": 0,
        "cpu_fallback_count": 0,
        "runtime_telemetry_complete": True,
        "maximum_positive_lama_inference_per_page": 0,
        "positive_lama_inference_count": 0,
        "lama_runtime_provider": "",
        "lama_runtime_precision": "",
        "target_instance_seed_recall": 0.99,
        "missed_target_instance_count": 1,
        "output_mask_set_sha256": mask_sha,
    }


def _relative_pages(
    *,
    first: float,
    second: float,
    residue: float,
) -> list[dict[str, object]]:
    return [
        {
            "page_id": "page-001",
            "residue_score": residue,
            "target_instance_edit_scores": [
                {"instance_id": "first", "coverage": first},
                {"instance_id": "second", "coverage": second},
            ],
        }
    ]


def test_relative_product_gate_admits_safe_best_effort_improvement() -> None:
    result = evaluate_relative_product_gate(
        baseline_metrics=_relative_metrics(
            coverage=0.90, residue=0.50, mask_sha="a" * 64
        ),
        candidate_metrics=_relative_metrics(
            coverage=0.91, residue=0.40, mask_sha="b" * 64
        ),
        baseline_pages=_relative_pages(first=0.99, second=0.50, residue=0.50),
        candidate_pages=_relative_pages(first=0.99, second=0.52, residue=0.40),
        candidate_kind="balanced",
    )

    assert result["relative_product_pass"] is True
    assert result["strict_seed_eligible"] is False
    assert result["candidate_98_instance_count"] == 1
    assert result["gate_failures"] == []


def test_relative_product_gate_keeps_destructive_safety_absolute() -> None:
    candidate = _relative_metrics(
        coverage=0.91, residue=0.40, mask_sha="b" * 64
    )
    candidate["preserve_edit_overlap"] = 1

    result = evaluate_relative_product_gate(
        baseline_metrics=_relative_metrics(
            coverage=0.90, residue=0.50, mask_sha="a" * 64
        ),
        candidate_metrics=candidate,
        baseline_pages=_relative_pages(first=0.99, second=0.50, residue=0.50),
        candidate_pages=_relative_pages(first=0.99, second=0.52, residue=0.40),
        candidate_kind="balanced",
    )

    assert result["relative_product_pass"] is False
    assert "safety_nonzero:preserve_edit_overlap" in result["gate_failures"]


def test_relative_product_gate_rejects_fractional_counts_and_non_bf16_lama() -> None:
    baseline = _relative_metrics(coverage=0.90, residue=0.50, mask_sha="a" * 64)
    pages_before = _relative_pages(first=0.99, second=0.50, residue=0.50)
    pages_after = _relative_pages(first=0.99, second=0.52, residue=0.40)
    fractional = _relative_metrics(
        coverage=0.91,
        residue=0.40,
        mask_sha="b" * 64,
    )
    fractional["preserve_edit_overlap"] = 0.5
    fractional["maximum_positive_lama_inference_per_page"] = 0.5
    fractional["positive_lama_inference_count"] = 0.5

    rejected = evaluate_relative_product_gate(
        baseline_metrics=baseline,
        candidate_metrics=fractional,
        baseline_pages=pages_before,
        candidate_pages=pages_after,
        candidate_kind="balanced",
    )

    assert "safety_nonzero:preserve_edit_overlap" in rejected["gate_failures"]
    assert "positive_lama_page_call_limit" in rejected["gate_failures"]
    assert "positive_lama_inference_count_invalid" in rejected["gate_failures"]

    wrong_precision = dict(fractional)
    wrong_precision["preserve_edit_overlap"] = 0
    wrong_precision["maximum_positive_lama_inference_per_page"] = 1
    wrong_precision["positive_lama_inference_count"] = 1
    wrong_precision["lama_runtime_provider"] = "cuda:0"
    wrong_precision["lama_runtime_precision"] = "fp32"
    rejected = evaluate_relative_product_gate(
        baseline_metrics=baseline,
        candidate_metrics=wrong_precision,
        baseline_pages=pages_before,
        candidate_pages=pages_after,
        candidate_kind="balanced",
    )

    assert "positive_lama_not_cuda" in rejected["gate_failures"]


def test_relative_product_gate_requires_fill_only_mask_identity_and_lower_residue() -> None:
    baseline = _relative_metrics(coverage=0.90, residue=0.50, mask_sha="a" * 64)
    candidate = _relative_metrics(coverage=0.90, residue=0.40, mask_sha="a" * 64)
    pages_before = _relative_pages(first=0.99, second=0.50, residue=0.50)
    pages_after = _relative_pages(first=0.99, second=0.50, residue=0.40)

    accepted = evaluate_relative_product_gate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        baseline_pages=pages_before,
        candidate_pages=pages_after,
        candidate_kind="fill_only",
    )
    assert accepted["relative_product_pass"] is True

    candidate["output_mask_set_sha256"] = "b" * 64
    rejected = evaluate_relative_product_gate(
        baseline_metrics=baseline,
        candidate_metrics=candidate,
        baseline_pages=pages_before,
        candidate_pages=pages_after,
        candidate_kind="fill_only",
    )
    assert rejected["relative_product_pass"] is False
    assert "fill_only_edit_mask_changed" in rejected["gate_failures"]


def test_balanced_relative_adjudication_requires_admitted_preflight(
    tmp_path: Path,
) -> None:
    manifest_sha = "c" * 64

    def _result(path: Path, run_id: str, *, improved: bool) -> Path:
        path.write_text(
            json.dumps(
                {
                    "schema_version": "inpaint-factorized-results-v3",
                    "manifest_sha256": manifest_sha,
                    "runs": [
                        {
                            "run_id": run_id,
                            "selection": {
                                "detector": "best-fusion",
                                "router": "R0",
                                "fill": "conditional_refill_existing",
                            },
                            "metrics": _relative_metrics(
                                coverage=0.91 if improved else 0.90,
                                residue=0.40 if improved else 0.50,
                                mask_sha=("b" if improved else "a") * 64,
                            ),
                        }
                    ],
                    "pages": {
                        run_id: _relative_pages(
                            first=0.99,
                            second=0.52 if improved else 0.50,
                            residue=0.40 if improved else 0.50,
                        )
                    },
                }
            ),
            encoding="utf-8",
        )
        return path

    baseline = _result(tmp_path / "baseline.json", "b0", improved=False)
    candidate = _result(tmp_path / "candidate.json", "b2", improved=True)
    with pytest.raises(ValueError, match="requires a sealed preflight"):
        adjudicate_relative_product(
            baseline_path=baseline,
            baseline_run_id="b0",
            candidate_path=candidate,
            candidate_run_id="b2",
            candidate_kind="balanced",
        )

    preflight = tmp_path / "preflight.json"
    preflight.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-balanced-preflight-adjudication-v32",
                "manifest_sha256": manifest_sha,
                "semantic_provider": "ocr_provenance_verifier",
                "seed_admitted": True,
                "balanced_candidate_admitted": True,
            }
        ),
        encoding="utf-8",
    )
    result = adjudicate_relative_product(
        baseline_path=baseline,
        baseline_run_id="b0",
        candidate_path=candidate,
        candidate_run_id="b2",
        candidate_kind="balanced",
        balanced_preflight_path=preflight,
    )

    assert result["relative_product_pass"] is True
    assert result["seed_admitted"] is True
    assert result["provenance"]["semantic_provider"] == "ocr_provenance_verifier"
    assert result["balanced_preflight_sha256"] == hashlib.sha256(
        preflight.read_bytes()
    ).hexdigest()


def _write_image(path: Path, image: np.ndarray) -> str:
    assert cv2.imwrite(str(path), image)
    return str(path)


def _bind_independent_review_fixture(
    semantic_path: Path,
    ledger_path: Path,
    decisions_path: Path,
) -> None:
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic.update(
        {
            "schema_version": "inpaint-factorized-source-decisions-v4",
            "candidate_seen": False,
            "review_complete": True,
        }
    )
    for page in semantic["pages"]:
        page.update(
            {
                "candidate_seen": False,
                "reviewed_source_only": True,
                "review_complete": True,
            }
        )
    semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["semantic_manifest"] = str(semantic_path.resolve())
    ledger["semantic_manifest_sha256"] = hashlib.sha256(
        semantic_path.read_bytes()
    ).hexdigest()
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    decisions["review_ledger"] = str(ledger_path.resolve())
    decisions["review_ledger_sha256"] = hashlib.sha256(
        ledger_path.read_bytes()
    ).hexdigest()
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")


def test_independent_review_applier_rejects_candidate_seen_semantics_or_wrong_ledger(
    tmp_path: Path,
) -> None:
    semantic = tmp_path / "semantic.json"
    ledger = tmp_path / "ledger.json"
    decisions = tmp_path / "decisions.json"
    semantic.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-decisions-v4",
                "candidate_seen": True,
                "review_complete": True,
                "pages": [
                    {
                        "page_id": "p",
                        "candidate_seen": False,
                        "reviewed_source_only": True,
                        "review_complete": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-ledger-v4",
                "candidate_seen": False,
                "semantic_manifest": str(semantic.resolve()),
                "semantic_manifest_sha256": hashlib.sha256(
                    semantic.read_bytes()
                ).hexdigest(),
                "rows": [],
                "full_page_inventory_pending": [],
            }
        ),
        encoding="utf-8",
    )
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-decisions-v4",
                "candidate_seen": False,
                "review_complete": True,
                "review_ledger": str(ledger.resolve()),
                "review_ledger_sha256": hashlib.sha256(
                    ledger.read_bytes()
                ).hexdigest(),
                "decisions": [],
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="semantic review input is not source-only"):
        apply_independent_target_review(
            semantic, ledger, decisions, tmp_path / "candidate-seen-output"
        )

    payload = json.loads(semantic.read_text(encoding="utf-8"))
    payload["candidate_seen"] = False
    semantic.write_text(json.dumps(payload), encoding="utf-8")
    ledger_payload = json.loads(ledger.read_text(encoding="utf-8"))
    ledger_payload["semantic_manifest_sha256"] = hashlib.sha256(
        semantic.read_bytes()
    ).hexdigest()
    ledger.write_text(json.dumps(ledger_payload), encoding="utf-8")
    decision_payload = json.loads(decisions.read_text(encoding="utf-8"))
    decision_payload["review_ledger"] = str(tmp_path / "different-ledger.json")
    decision_payload["review_ledger_sha256"] = hashlib.sha256(
        ledger.read_bytes()
    ).hexdigest()
    decisions.write_text(json.dumps(decision_payload), encoding="utf-8")

    with pytest.raises(ValueError, match="bind a different review ledger"):
        apply_independent_target_review(
            semantic, ledger, decisions, tmp_path / "wrong-ledger-output"
        )


def _write_strict_source_manifest(
    path: Path,
    payload: dict[str, object],
) -> Path:
    pages = payload.get("pages")
    assert isinstance(pages, list) and pages
    for page_index, page in enumerate(pages):
        assert isinstance(page, dict)
        page_id = str(page.get("page_id") or f"p{page_index}")
        source_path = page.get("path")
        if not isinstance(source_path, str) or not source_path:
            source_path = _write_image(
                path.parent / f"{page_id}-source.png",
                np.full((32, 48, 3), 200, np.uint8),
            )
            page["path"] = source_path
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        assert source is not None
        shape = source.shape[:2]
        zero = np.zeros(shape, np.uint8)
        full = np.full(shape, 255, np.uint8)
        zero_path = _write_image(path.parent / f"{page_id}-zero.png", zero)
        full_path = _write_image(path.parent / f"{page_id}-full.png", full)
        raw_instances = page.get("target_instances", [])
        assert isinstance(raw_instances, list)
        required_union = np.zeros(shape, np.uint8)
        preserve_union = np.zeros(shape, np.uint8)
        ambiguous_union = np.zeros(shape, np.uint8)
        raw_regions = page.get("regions")
        if not isinstance(raw_regions, list) or not raw_regions:
            raw_regions = [{"region_id": "region"}]
            page["regions"] = raw_regions
        region_ids = [str(region.get("region_id") or "") for region in raw_regions]
        for instance_index, instance in enumerate(raw_instances):
            assert isinstance(instance, dict)
            instance.setdefault("region_id", region_ids[0])
            instance.setdefault("semantic_role", "dialogue_bubble")
            priority = str(instance.get("priority") or "required")
            instance["priority"] = priority
            instance.setdefault(
                "processing_action",
                "translate_inpaint"
                if priority == "required"
                else ("preserve" if priority == "optional" else "review"),
            )
            mask_path = instance.get("mask_path")
            mask = (
                cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                if isinstance(mask_path, str) and mask_path
                else None
            )
            if mask is None:
                mask = np.zeros(shape, np.uint8)
                y = 2 + instance_index * 4
                mask[y:y + 2, 3:7] = 255
                mask_path = _write_image(
                    path.parent / f"{page_id}-instance-{instance_index}.png",
                    mask,
                )
                instance["mask_path"] = mask_path
            destination = (
                required_union
                if priority == "required"
                else preserve_union
                if priority == "optional"
                else ambiguous_union
            )
            destination[mask > 0] = 255
        target_path = (
            _write_image(path.parent / f"{page_id}-target.png", required_union)
            if np.any(required_union)
            else None
        )
        preserve_path = _write_image(
            path.parent / f"{page_id}-preserve.png", preserve_union
        )
        ambiguous_path = _write_image(
            path.parent / f"{page_id}-ambiguous.png", ambiguous_union
        )
        page.update(
            {
                "target_text_mask": target_path,
                "preserve_mask": preserve_path,
                "protected_structure_mask": zero_path,
                "ambiguous_structure_mask": ambiguous_path,
                "ownership_mask": full_path,
                "claim_seed_mask": full_path,
                "bubble_interior_mask": full_path,
                "corner_protect_mask": zero_path,
                "existing_source_edit_mask": str(
                    page.get("existing_source_edit_mask") or zero_path
                ),
                "expected_edit": "required" if np.any(required_union) else "none",
                "width": shape[1],
                "height": shape[0],
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
                "annotation_basis": "source_only_v4",
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
                "target_mask_provenance": str(
                    page.get("target_mask_provenance") or "source_only_v4"
                ),
            }
        )
        for region in raw_regions:
            assert isinstance(region, dict)
            region.update(
                {
                    "bubble_route_class": str(
                        region.get("bubble_route_class") or "clean_flat"
                    ),
                    "bubble_interior_mask": full_path,
                    "ownership_mask": full_path,
                    "protected_structure_mask": zero_path,
                    "ambiguous_structure_mask": ambiguous_path,
                    "corner_protect_mask": zero_path,
                }
            )
        page["source_sha256"] = hashlib.sha256(
            Path(str(source_path)).read_bytes()
        ).hexdigest()
    payload.update(
        {
            "schema_version": "inpaint-factorized-source-manifest-v4",
            "corpus_id": str(payload.get("corpus_id") or "fixture"),
            "split_role": "development_source_only",
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "target_extent_independent": True,
            "target_inventory_independent": True,
            "target_review_complete": True,
        }
    )
    page_ids = sorted(str(page["page_id"]) for page in pages)
    payload["page_count"] = len(page_ids)
    payload["page_ids"] = page_ids
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    for page in pages:
        page["artifact_sha256"] = manifest_page_artifact_sha256(path, page)
        page["source_sha256"] = page["artifact_sha256"]["path"]
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        pages
    )
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    seal = {
        "schema_version": "inpaint-factorized-manifest-seal-v4",
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    path.with_suffix(path.suffix + ".seal.json").write_text(
        json.dumps(seal, sort_keys=True), encoding="utf-8"
    )
    return path


def test_v4_runners_reject_tampered_source_artifact_before_outputs(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    payload: dict[str, object] = {
        "pages": [
            {
                "page_id": "p",
                "regions": [
                    {"region_id": "r", "proposal": {"text_class": "text_bubble"}}
                ],
                "target_instances": [
                    {
                        "instance_id": "i",
                        "region_id": "r",
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                        "priority": "required",
                    }
                ],
            }
        ]
    }
    _write_strict_source_manifest(manifest, payload)
    page = payload["pages"][0]  # type: ignore[index]
    target_path = Path(str(page["target_text_mask"]))  # type: ignore[index]
    source = cv2.imread(str(page["path"]), cv2.IMREAD_COLOR)  # type: ignore[index]
    assert source is not None
    assert cv2.imwrite(str(target_path), np.zeros(source.shape[:2], np.uint8))

    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        score_semantic_policies(manifest)
    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        run_fusion_matrix(manifest, tmp_path / "missing-spec.json")
    with pytest.raises(ValueError, match="artifact SHA inventory differs"):
        factorized_main(
            [
                "--manifest",
                str(manifest),
                "--matrix",
                str(tmp_path / "missing-matrix.json"),
                "--output-dir",
                str(tmp_path / "must-not-exist"),
            ]
        )
    assert not (tmp_path / "must-not-exist").exists()


def test_relative_baseline_attachment_preserves_frozen_annotations(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    payload: dict[str, object] = {
        "pages": [
            {
                "page_id": "p",
                "regions": [
                    {"region_id": "r", "proposal": {"text_class": "text_bubble"}}
                ],
                "target_instances": [
                    {
                        "instance_id": "i",
                        "region_id": "r",
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                        "priority": "required",
                    }
                ],
            }
        ]
    }
    _write_strict_source_manifest(source_manifest, payload)
    source_page = payload["pages"][0]  # type: ignore[index]
    source_path = Path(str(source_page["path"]))  # type: ignore[index]
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    assert source_image is not None
    corpus_dir = tmp_path / "corpus"
    (corpus_dir / "cleaned_images").mkdir(parents=True)
    (corpus_dir / "final_masks").mkdir(parents=True)
    baseline_image = corpus_dir / "cleaned_images" / "p_cleaned.png"
    baseline_mask = corpus_dir / "final_masks" / "p_final_mask.png"
    assert cv2.imwrite(str(baseline_image), source_image)
    edit = np.zeros(source_image.shape[:2], np.uint8)
    edit[2:4, 2:4] = 255
    assert cv2.imwrite(str(baseline_mask), edit)
    metrics_pages = tmp_path / "pages.jsonl"
    metrics_pages.write_text(
        json.dumps(
            {
                "page_id": "p",
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "cleaned_sha256": hashlib.sha256(baseline_image.read_bytes()).hexdigest(),
                "final_mask_sha256": hashlib.sha256(baseline_mask.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_manifest = tmp_path / "baseline-manifest.json"
    baseline_manifest.write_text(
        json.dumps(
            seal_product_baseline(
                source_manifest_path=source_manifest,
                corpus_artifact_dir=corpus_dir,
                metrics_pages_path=metrics_pages,
                product_commit="a" * 40,
            ),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    output = tmp_path / "relative-manifest.json"
    assert attach_relative_baseline_main(
        [
            "--source-manifest",
            str(source_manifest),
            "--baseline-manifest",
            str(baseline_manifest),
            "--output",
            str(output),
        ]
    ) == 0
    attached = json.loads(output.read_text(encoding="utf-8"))
    attached_page = attached["pages"][0]
    assert attached_page["target_instances"] == source_page["target_instances"]
    assert attached_page["baseline"] == str(baseline_image.resolve())
    assert attached_page["baseline_mask"] == str(baseline_mask.resolve())
    assert attached_page["existing_source_edit_mask"] == str(
        baseline_mask.resolve()
    )
    matrix = build_relative_matrix(output)
    assert [
        row["fill"] for row in matrix["explicit_combinations"]
    ] == ["mask_only", "conditional_refill_existing"]
    assert matrix["families"]["detector"]["pr6_baseline_edit"]["pages"]["p"][
        "raw"
    ] == str(baseline_mask.resolve())


def test_mask_only_preflight_scores_the_real_source_image(
    tmp_path: Path,
) -> None:
    shape = (24, 32)
    source = np.full((*shape, 3), 180, np.uint8)
    baseline = source.copy()
    baseline[4:8, 5:10] = 230
    target = np.zeros(shape, np.uint8)
    target[4:8, 5:10] = 255
    protected = np.zeros(shape, np.uint8)
    protected[14:18, 20:26] = 255
    full = np.full(shape, 255, np.uint8)
    zero = np.zeros(shape, np.uint8)
    source_path = _write_image(tmp_path / "source.png", source)
    baseline_path = _write_image(tmp_path / "baseline.png", baseline)
    target_path = _write_image(tmp_path / "target.png", target)
    protected_path = _write_image(tmp_path / "protected.png", protected)
    full_path = _write_image(tmp_path / "full.png", full)
    zero_path = _write_image(tmp_path / "zero.png", zero)
    manifest_path = tmp_path / "manifest.json"
    payload: dict[str, object] = {
        "pages": [
            {
                "page_id": "p",
                "path": source_path,
                "baseline": baseline_path,
                "baseline_mask": target_path,
                "existing_source_edit_mask": target_path,
                "regions": [{"region_id": "r"}],
                "target_instances": [
                    {"instance_id": "i", "region_id": "r", "priority": "required"}
                ],
            }
        ]
    }
    _write_strict_source_manifest(manifest_path, payload)
    page = payload["pages"][0]  # type: ignore[index]
    page["baseline"] = baseline_path  # type: ignore[index]
    page["baseline_mask"] = target_path  # type: ignore[index]
    page["existing_source_edit_mask"] = target_path  # type: ignore[index]
    page["protected_structure_mask"] = protected_path  # type: ignore[index]
    page["ownership_mask"] = full_path  # type: ignore[index]
    page["claim_seed_mask"] = target_path  # type: ignore[index]
    page["corner_protect_mask"] = zero_path  # type: ignore[index]
    page["artifact_sha256"] = manifest_page_artifact_sha256(  # type: ignore[index]
        manifest_path, page  # type: ignore[arg-type]
    )
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        payload["pages"]  # type: ignore[arg-type]
    )
    manifest_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest_path.with_suffix(manifest_path.suffix + ".seal.json").write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-manifest-seal-v4",
                "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                "candidate_generated": False,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    matrix = build_relative_matrix(manifest_path)
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")
    output = tmp_path / "output"

    assert factorized_main(
        [
            "--manifest",
            str(manifest_path),
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--limit",
            "1",
        ]
    ) == 0
    result = json.loads((output / "factorized-results.json").read_text("utf-8"))
    run_id = result["runs"][0]["run_id"]
    statistics = result["pages"][run_id][0]["canonical_statistics"]
    assert statistics["protected_structure_changed_pixel_count"] == 0
    assert result["pages"][run_id][0]["changed_pixel_count"] == int(
        np.count_nonzero(target)
    )


def test_product_eval_manifest_is_a_lossless_source_only_view(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    payload: dict[str, object] = {
        "pages": [
            {
                "page_id": "required-page",
                "regions": [{"region_id": "r"}],
                "target_instances": [
                    {
                        "instance_id": "i",
                        "region_id": "r",
                        "priority": "required",
                    }
                ],
            },
            {
                "page_id": "no-edit-page",
                "regions": [{"region_id": "r"}],
                "target_instances": [],
            },
        ]
    }
    _write_strict_source_manifest(source_manifest, payload)
    output = tmp_path / "product-eval.json"
    product, provenance = build_product_eval_manifest(
        source_manifest_path=source_manifest,
        output_path=output,
        source_lock_git_sha="a" * 40,
    )
    assert product["manifest_sha256"] == _canonical_manifest_sha256(product)
    pages = {page["page_id"]: page for page in product["pages"]}
    source_pages = {page["page_id"]: page for page in payload["pages"]}
    required = pages["required-page"]
    assert required["target_text_mask"]["path"] == str(
        Path(str(source_pages["required-page"]["target_text_mask"])).resolve()
    )
    assert required["protected_structure_mask"]["path"] == str(
        Path(
            str(source_pages["required-page"]["protected_structure_mask"])
        ).resolve()
    )
    no_edit_target = Path(pages["no-edit-page"]["target_text_mask"]["path"])
    assert np.count_nonzero(cv2.imread(str(no_edit_target), cv2.IMREAD_GRAYSCALE)) == 0
    assert provenance["annotation_transform"] == "none"
    assert provenance["required_page_count"] == 1
    assert provenance["no_edit_page_count"] == 1


def test_product_eval_manifest_rejects_reused_zero_mask_directory(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source-manifest.json"
    payload: dict[str, object] = {
        "pages": [
            {
                "page_id": "p",
                "regions": [{"region_id": "r"}],
                "target_instances": [],
            }
        ]
    }
    _write_strict_source_manifest(source_manifest, payload)
    output = tmp_path / "product-eval.json"
    zero_dir = tmp_path / "product-eval-zero-targets"
    zero_dir.mkdir()
    with pytest.raises(FileExistsError, match="zero-mask directory must be fresh"):
        build_product_eval_manifest(
            source_manifest_path=source_manifest,
            output_path=output,
            source_lock_git_sha="a" * 40,
        )


def test_balanced_preflight_rejects_unavailable_semantic_provider(
    tmp_path: Path,
) -> None:
    manifest_sha = "a" * 64
    fusion = tmp_path / "fusion.json"
    fusion.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-fusion-results-v4",
                "manifest_sha256": manifest_sha,
                "runs": [
                    {
                        "run_id": "best",
                        "seed_admitted": True,
                        "metrics": {
                            "missed_target_instance_count": 6,
                            "target_instance_seed_recall": 653 / 659,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-semantic-policy-results-v4",
                "manifest_sha256": manifest_sha,
                "policies": [
                    {
                        "policy_id": "ocr_provenance_verifier",
                        "status": "blocked_asset",
                        "oracle_only": False,
                        "metrics": {
                            "preserve_destructive_count": 0,
                            "ambiguous_destructive_count": 0,
                            "no_edit_false_translate_page_count": 0,
                            "unavailable_instance_count": 809,
                            "required_translate_recall": 0.0,
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    result = adjudicate_balanced_preflight(
        fusion_path=fusion,
        fusion_run_id="best",
        semantic_path=semantic,
        semantic_policy_id="ocr_provenance_verifier",
    )
    assert result["strict_seed_eligible"] is False
    assert result["seed_admitted"] is True
    assert result["balanced_candidate_admitted"] is False
    assert "semantic_provider_unavailable" in result["gate_failures"]
    assert (
        "semantic_gate_nonzero:unavailable_instance_count"
        in result["gate_failures"]
    )
    semantic_payload = json.loads(semantic.read_text(encoding="utf-8"))
    policy = semantic_payload["policies"][0]
    policy["status"] = "dominated"
    policy["metrics"]["unavailable_instance_count"] = 0
    policy["metrics"]["required_translate_recall"] = 0.99
    semantic.write_text(json.dumps(semantic_payload), encoding="utf-8")
    admitted = adjudicate_balanced_preflight(
        fusion_path=fusion,
        fusion_run_id="best",
        semantic_path=semantic,
        semantic_policy_id="ocr_provenance_verifier",
    )
    assert admitted["strict_seed_eligible"] is False
    assert admitted["balanced_candidate_admitted"] is True


def test_balanced_preflight_cli_imports_from_a_direct_script_process() -> None:
    completed = subprocess.run(
        [
            str(ROOT / ".venv-win" / "Scripts" / "python.exe"),
            "-B",
            str(ROOT / "scripts" / "adjudicate_inpaint_balanced_preflight_v32.py"),
            "--help",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Fail closed before spending CUDA" in completed.stdout


def test_v32_three_case_sheet_omits_rejected_balanced_column(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    assert cv2.imwrite(str(image), np.full((20, 30, 3), 180, np.uint8))
    assert cv2.imwrite(str(mask), np.zeros((20, 30), np.uint8))
    rows = [
        {
            "case_id": case_id,
            "source": str(image),
            "control": str(image),
            "fill_only": str(image),
            "edit_mask": str(mask),
            "protect_mask": str(mask),
            "crop_xyxy": [0, 0, 30, 20],
        }
        for case_id in ("japan-i_102", "japan-p_015", "japan-096")
    ]
    sheet = build_three_case_sheet(
        {
            "schema_version": "inpaint-v32-three-case-contact-sheet-v1",
            "balanced_available": False,
            "cell_width": 80,
            "cell_height": 60,
            "rows": rows,
        }
    )
    assert sheet.size == (320, 304)


def test_v32_three_case_sheet_omits_all_rejected_candidate_columns(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    mask = tmp_path / "mask.png"
    assert cv2.imwrite(str(image), np.full((20, 30, 3), 180, np.uint8))
    assert cv2.imwrite(str(mask), np.zeros((20, 30), np.uint8))
    sheet = build_three_case_sheet(
        {
            "schema_version": "inpaint-v32-three-case-contact-sheet-v1",
            "fill_only_available": False,
            "balanced_available": False,
            "cell_width": 80,
            "cell_height": 60,
            "rows": [
                {
                    "case_id": case_id,
                    "source": str(image),
                    "control": str(image),
                    "edit_mask": str(mask),
                    "protect_mask": str(mask),
                    "crop_xyxy": [0, 0, 30, 20],
                }
                for case_id in ("japan-i_102", "japan-p_015", "japan-096")
            ],
        }
    )
    assert sheet.size == (240, 304)


def test_fill_preflight_rejects_unsafe_pr6_mask_before_cuda(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "page_count": 1,
                "source_annotation_manifest_sha256": "a" * 64,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    metrics = _relative_metrics(coverage=0.9, residue=0.2, mask_sha="b" * 64)
    metrics.update(
        {
            "page_count": 1,
            "protected_structure_changed": 7,
            "no_edit_false_edit": 9,
        }
    )
    factorized = tmp_path / "factorized.json"
    factorized.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-results-v3",
                "manifest_sha256": hashlib.sha256(
                    source_manifest.read_bytes()
                ).hexdigest(),
                "runs": [{"run_id": "b0", "metrics": metrics}],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "manifest_corpora": {
                    "e1": {"parent_manifest_sha256": "a" * 64}
                },
                "success_count": 1,
                "failure_count": 0,
                "cpu_fallback_count": 0,
                "required_skipped_block_count": 0,
                "protected_structure_changed_pixel_count_exact": 7,
                "ambiguous_structure_changed_pixel_count_exact": 0,
                "changed_outside_final_mask_pixel_count_exact": 0,
                "unexpected_none_edit_count": 1,
                "inpainter_runtime": {
                    "actual_device": "cuda",
                    "actual_precision": "bf16",
                    "cpu_fallback_used": False,
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = adjudicate_fill_preflight(
        factorized_path=factorized,
        run_id="b0",
        source_manifest_path=source_manifest,
        product_summary_path=summary,
    )

    assert result["fill_candidate_admitted"] is False
    assert result["cuda_stage2_authorized"] is False
    assert "safety_nonzero:protected_structure_changed" in result["gate_failures"]
    assert "safety_nonzero:no_edit_false_edit" in result["gate_failures"]
    assert (
        "product_summary_nonzero:protected_structure_changed_pixel_count_exact"
        in result["gate_failures"]
    )


def test_v32_final_selection_keeps_pr6_when_both_candidates_fail(
    tmp_path: Path,
) -> None:
    balanced = tmp_path / "balanced.json"
    balanced.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-balanced-preflight-adjudication-v32",
                "manifest_sha256": "a" * 64,
                "balanced_candidate_admitted": False,
                "gate_failures": ["semantic_provider_unavailable"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    fill = tmp_path / "fill.json"
    fill.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-fill-preflight-adjudication-v32",
                "source_annotation_manifest_sha256": "a" * 64,
                "fill_candidate_admitted": False,
                "gate_failures": ["safety_nonzero:protected_structure_changed"],
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    result = adjudicate_final_selection(
        balanced_preflight_path=balanced,
        fill_preflight_path=fill,
    )

    assert result["selected_candidate"] == "current_pr6"
    assert result["relative_product_pass"] is False
    assert result["product_pr_rebase_authorized"] is False
    assert result["a5_authorized"] is False
    assert result["a5_state"] == "unavailable"


def test_minimal_closure_matrix_covers_every_non_oracle_role_variant() -> None:
    controls = {
        "detector": "d0",
        "expansion": "e0",
        "fill": "mask_only",
        "ownership": "o0",
        "router": "r0",
        "silhouette": "s0",
    }
    source = {
        "schema_version": "inpaint-factorized-matrix-v3",
        "axes": {
            "detector": ["d0", "oracle", "d1"],
            "expansion": ["e0", "e1"],
            "fill": ["mask_only"],
            "ownership": ["o0", "o1"],
            "router": ["r0", "r1"],
            "silhouette": ["s0", "s1"],
        },
        "controls": controls,
        "oracle_only": ["oracle"],
        "explicit_combinations": [
            {
                "detector": "d1",
                "expansion": "e1",
                "fill": "mask_only",
                "ownership": "o1",
                "router": "r1",
                "silhouette": "s1",
            }
        ],
    }

    result = build_minimal_closure_matrix(source)

    assert result["closure_reduction"]["coverage_complete"] is True
    assert result["closure_reduction"]["selected_combination_count"] == 2
    rows = result["explicit_combinations"]
    for role in ("detector", "expansion", "ownership", "router", "silhouette"):
        expected = set(source["axes"][role]) - {"oracle"}
        assert {row[role] for row in rows} == expected


def test_semantic_policy_matrix_scores_defaults_and_blocks_missing_evidence(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "pages": [
                    {
                        "page_id": "p",
                        "regions": [
                            {"region_id": "dialogue", "proposal": {"text_class": "text_bubble"}},
                            {"region_id": "sfx", "proposal": {"text_class": "text_free"}},
                        ],
                        "target_instances": [
                            {
                                "instance_id": "required",
                                "region_id": "dialogue",
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            },
                            {
                                "instance_id": "preserve",
                                "region_id": "sfx",
                                "semantic_role": "sfx",
                                "processing_action": "preserve",
                                "priority": "optional",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _write_strict_source_manifest(
        manifest, json.loads(manifest.read_text(encoding="utf-8"))
    )
    result = score_semantic_policies(manifest)
    policies = {row["policy_id"]: row for row in result["policies"]}

    assert result["unaccounted_policy_count"] == 0
    assert result["page_ids"] == ["p"]
    assert set(result["pages"]) == {
        "current_default",
        "detector_explicit_role",
        "ocr_semantic_hint",
        "ocr_provenance_verifier",
        "explicit_role_consensus",
        "human_oracle",
    }
    assert len(result["logical_inventory_sha256"]) == 64
    assert policies["current_default"]["status"] == "dominated"
    assert policies["current_default"]["metrics"]["required_translate_recall"] == 1.0
    assert policies["current_default"]["metrics"]["preserve_destructive_count"] == 1
    assert policies["detector_explicit_role"]["status"] == "blocked_asset"
    assert policies["ocr_semantic_hint"]["status"] == "blocked_asset"
    assert policies["ocr_provenance_verifier"]["status"] == "blocked_asset"
    assert policies["explicit_role_consensus"]["status"] == "blocked_asset"
    assert policies["human_oracle"]["status"] == "family_complete"
    assert policies["human_oracle"]["oracle_only"] is True


def test_ocr_provenance_verifier_requires_provider_evidence_and_abstains_without_text(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "pages": [
                    {
                        "page_id": "p",
                        "regions": [
                            {
                                "region_id": "translate",
                                "proposal": {
                                    "text_class": "text_bubble",
                                    "ocr_evidence_available": True,
                                    "ocr_text": "example",
                                    "ocr_script": "Latin",
                                    "ocr_confidence": 0.75,
                                },
                            },
                            {
                                "region_id": "preserve",
                                "proposal": {
                                    "text_class": "sfx",
                                    "ocr_evidence_available": True,
                                },
                            },
                            {
                                "region_id": "abstain",
                                "proposal": {
                                    "text_class": "text_free",
                                    "ocr_evidence_available": True,
                                    "ocr_text": "",
                                    "ocr_script": "Latin",
                                    "ocr_confidence": 0.9,
                                },
                            },
                        ],
                        "target_instances": [
                            {
                                "instance_id": "required",
                                "region_id": "translate",
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            },
                            {
                                "instance_id": "preserve",
                                "region_id": "preserve",
                                "semantic_role": "sfx",
                                "processing_action": "preserve",
                                "priority": "optional",
                            },
                            {
                                "instance_id": "abstain",
                                "region_id": "abstain",
                                "semantic_role": "dialogue_free",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_strict_source_manifest(
        manifest, json.loads(manifest.read_text(encoding="utf-8"))
    )

    result = score_semantic_policies(manifest)
    rows = result["pages"]["ocr_provenance_verifier"][0]["decisions"]
    by_id = {row["instance_id"]: row for row in rows}
    assert by_id["required"]["predicted_action"] == "translate_inpaint"
    assert by_id["preserve"]["predicted_action"] == "preserve"
    assert by_id["abstain"]["predicted_action"] == "review"
    assert by_id["abstain"]["available"] is True


def test_semantic_policy_counts_false_translate_actions_on_no_edit_pages(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "pages": [
                    {
                        "page_id": "p",
                        "expected_edit": "none",
                        "regions": [
                            {
                                "region_id": "false-translate",
                                "proposal": {"text_class": "text_bubble"},
                            }
                        ],
                        "target_instances": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    _write_strict_source_manifest(
        manifest, json.loads(manifest.read_text(encoding="utf-8"))
    )

    policies = {
        row["policy_id"]: row for row in score_semantic_policies(manifest)["policies"]
    }
    assert policies["current_default"]["metrics"][
        "no_edit_false_translate_page_count"
    ] == 1
    assert policies["current_default"]["status"] == "dominated"
    assert policies["human_oracle"]["metrics"][
        "no_edit_false_translate_page_count"
    ] == 0


def test_semantic_policy_blocks_partially_missing_provider_evidence(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "pages": [
                    {
                        "page_id": "p",
                        "regions": [
                            {
                                "region_id": "present",
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                            },
                            {"region_id": "missing"},
                        ],
                        "target_instances": [
                            {
                                "instance_id": "present",
                                "region_id": "present",
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            },
                            {
                                "instance_id": "missing",
                                "region_id": "missing",
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _write_strict_source_manifest(
        manifest, json.loads(manifest.read_text(encoding="utf-8"))
    )
    policies = {
        row["policy_id"]: row for row in score_semantic_policies(manifest)["policies"]
    }
    explicit = policies["detector_explicit_role"]
    assert explicit["metrics"]["unavailable_instance_count"] == 1
    assert explicit["status"] == "blocked_asset"
    assert explicit["closure_reason"] == "semantic_evidence_missing"


def test_detector_ceiling_reports_instances_missed_by_every_provider(
    tmp_path: Path,
) -> None:
    source = np.full((24, 32, 3), 180, np.uint8)
    target_a = np.zeros(source.shape[:2], np.uint8)
    target_a[4:8, 4:8] = 255
    target_b = np.zeros(source.shape[:2], np.uint8)
    target_b[14:18, 22:26] = 255
    target_union = cv2.bitwise_or(target_a, target_b)
    claim_a = target_a.copy()
    empty = np.zeros(source.shape[:2], np.uint8)
    source_path = _write_image(tmp_path / "source.png", source)
    target_a_path = _write_image(tmp_path / "target-a.png", target_a)
    target_b_path = _write_image(tmp_path / "target-b.png", target_b)
    target_union_path = _write_image(tmp_path / "target-union.png", target_union)
    claim_a_path = _write_image(tmp_path / "claim-a.png", claim_a)
    empty_path = _write_image(tmp_path / "empty.png", empty)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "target_inventory_independent": True,
                "target_review_complete": True,
                "pages": [
                    {
                        "page_id": "p",
                        "path": source_path,
                        "source_sha256": hashlib.sha256(Path(source_path).read_bytes()).hexdigest(),
                        "width": 32,
                        "height": 24,
                        "expected_edit": "required",
                        "target_inventory_independent": True,
                        "target_review_complete": True,
                        "target_mask_provenance": "human_source_review",
                        "target_instances": [
                            {
                                "instance_id": "a",
                                "region_id": "r",
                                "mask_path": target_a_path,
                                "semantic_role": "dialogue_free",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                                "source_reviewed": True,
                            },
                            {
                                "instance_id": "b",
                                "region_id": "r",
                                "mask_path": target_b_path,
                                "semantic_role": "dialogue_free",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                                "source_reviewed": True,
                            },
                        ],
                        "regions": [
                            {
                                "region_id": "r",
                                "bubble_route_class": "ambiguous",
                                "bubble_interior_mask": empty_path,
                                "ownership_mask": empty_path,
                                "protected_structure_mask": empty_path,
                                "ambiguous_structure_mask": empty_path,
                                "corner_protect_mask": empty_path,
                            }
                        ],
                        "target_text_mask": target_union_path,
                        "protected_structure_mask": empty_path,
                        "ambiguous_structure_mask": empty_path,
                        "ownership_mask": empty_path,
                        "claim_seed_mask": empty_path,
                        "bubble_interior_mask": empty_path,
                        "corner_protect_mask": empty_path,
                        "existing_source_edit_mask": empty_path,
                        "preserve_mask": empty_path,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec = tmp_path / "spec.json"
    spec.write_text(
        json.dumps(
            {
                "candidates": {
                    "a": {"templates": {"raw": claim_a_path}},
                    "empty": {"templates": {"raw": empty_path}},
                }
            }
        ),
        encoding="utf-8",
    )

    result = audit_detector_ceiling(manifest, spec)

    assert result["candidate_count"] == 2
    assert result["required_instance_count"] == 2
    assert result["all_candidate_union_seeded_instance_count"] == 1
    assert result["all_candidate_union_missed_instance_count"] == 1
    assert result["missing_by_page"] == {"p": 1}
    assert result["missing_instances"][0]["instance_id"] == "b"


def test_method_family_closure_keeps_partial_family_active(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-registry-v4",
                "families": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "evaluation_scopes": ["e1"],
                        "variants": ["raw", "refined", "dilated"],
                    },
                    {
                        "family_id": "exact-protection",
                        "role": "protection",
                        "evaluation_scopes": ["e1"],
                        "variants": ["structure", "ownership"],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-evidence-v4",
                "scope_manifests": {
                    "e1": {
                        "sha256": "1" * 64,
                        "schema_version": "inpaint-factorized-source-manifest-v4",
                        "corpus_id": "test-e1",
                        "split_role": "development_source_only",
                    }
                },
                "evidence": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "raw",
                        "evaluation_scope": "e1",
                        "closure_state": "executed",
                        "disposition": "dominated",
                        "artifact_sha256": "a" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "content_sha256": "c" * 64,
                        "content_identity_kind": "exact_output",
                    },
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "refined",
                        "evaluation_scope": "e1",
                        "closure_state": "reused_by_sha",
                        "disposition": "dominated",
                        "artifact_sha256": "a" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "content_sha256": "c" * 64,
                        "content_identity_kind": "exact_output",
                        "reused_from": "detector-a/seed/raw/e1",
                    },
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "dilated",
                        "evaluation_scope": "e1",
                        "closure_state": "invalid_with_reason",
                        "disposition": "dominated",
                        "reason": "combination_incompatible",
                        "artifact_sha256": "e" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "invalid_parent_record_id": "dilated-combination",
                        "invalid_gate_facts_sha256": "f" * 64,
                    },
                    {
                        "family_id": "exact-protection",
                        "role": "protection",
                        "variant_id": "structure",
                        "evaluation_scope": "e1",
                        "closure_state": "executed",
                        "disposition": "pareto",
                        "artifact_sha256": "b" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "content_sha256": "d" * 64,
                        "content_identity_kind": "artifact_record",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_closure(registry, evidence)
    families = {row["family_id"]: row for row in result["families"]}

    assert result["unaccounted_variant_count"] == 1
    assert result["all_families_complete"] is False
    assert families["detector-a"]["family_complete"] is True
    assert families["detector-a"]["status"] == "family_complete"
    assert families["exact-protection"]["status"] == "active"
    assert families["exact-protection"]["missing_variants"] == ["ownership"]
    assert families["exact-protection"]["missing_requirements"] == [
        {"variant_id": "ownership", "evaluation_scope": "e1"}
    ]


def test_method_family_closure_rejects_unproven_sha_reuse(tmp_path: Path) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-registry-v4",
                "families": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "evaluation_scopes": ["e1"],
                        "variants": ["raw"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-evidence-v4",
                "scope_manifests": {
                    "e1": {
                        "sha256": "1" * 64,
                        "schema_version": "inpaint-factorized-source-manifest-v4",
                        "corpus_id": "test-e1",
                        "split_role": "development_source_only",
                    }
                },
                "evidence": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "raw",
                        "evaluation_scope": "e1",
                        "closure_state": "reused_by_sha",
                        "disposition": "dominated",
                        "scope_manifest_sha256": "1" * 64,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="artifact SHA"):
        build_closure(registry, evidence)

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["evidence"][0]["artifact_sha256"] = "a" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content SHA"):
        build_closure(registry, evidence)

    payload["evidence"][0]["content_sha256"] = "c" * 64
    payload["evidence"][0]["content_identity_kind"] = "exact_output"
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="source evidence key"):
        build_closure(registry, evidence)

    payload["evidence"][0]["artifact_sha256"] = "A" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="lowercase"):
        build_closure(registry, evidence)


def test_method_family_closure_requires_reused_sha_to_match_executed_source(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-registry-v4",
                "families": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "evaluation_scopes": ["e1"],
                        "variants": ["raw", "refined"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-method-family-evidence-v4",
                "scope_manifests": {
                    "e1": {
                        "sha256": "1" * 64,
                        "schema_version": "inpaint-factorized-source-manifest-v4",
                        "corpus_id": "test-e1",
                        "split_role": "development_source_only",
                    }
                },
                "evidence": [
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "raw",
                        "evaluation_scope": "e1",
                        "closure_state": "executed",
                        "disposition": "dominated",
                        "artifact_sha256": "a" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "content_sha256": "c" * 64,
                        "content_identity_kind": "exact_output",
                    },
                    {
                        "family_id": "detector-a",
                        "role": "seed",
                        "variant_id": "refined",
                        "evaluation_scope": "e1",
                        "closure_state": "reused_by_sha",
                        "disposition": "dominated",
                        "artifact_sha256": "b" * 64,
                        "scope_manifest_sha256": "1" * 64,
                        "content_sha256": "d" * 64,
                        "content_identity_kind": "exact_output",
                        "reused_from": "detector-a/seed/raw/e1",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="content SHA differs"):
        build_closure(registry, evidence)

    payload = json.loads(evidence.read_text(encoding="utf-8"))
    payload["evidence"][1]["content_sha256"] = "c" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    result = build_closure(registry, evidence)
    assert result["all_families_complete"] is True

    payload["evidence"][1]["scope_manifest_sha256"] = "2" * 64
    evidence.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical binding"):
        build_closure(registry, evidence)


def test_v4_method_registry_covers_every_role_and_required_variant() -> None:
    registry = json.loads(
        (
            ROOT
            / "benchmarking"
            / "inpaint_detector_bakeoff"
            / "method_registry_v4.json"
        ).read_text(encoding="utf-8")
    )
    families = {row["family_id"]: row for row in registry["families"]}
    roles = {row["role"] for row in registry["families"]}

    assert roles == {
        "seed",
        "semantic",
        "ownership",
        "silhouette",
        "router",
        "expansion",
        "protection",
        "fill",
        "composite",
    }
    assert set(families["roi-trigger"]["variants"]) == {
        "none",
        "always",
        "seed_missing",
        "raw_refined_disagreement",
        "source_seed_unavailable",
        "union",
    }
    assert set(families["exact-protection"]["variants"]) == {
        "pr4_exact",
        "C14",
        "C15",
        "C17",
        "C18",
        "C19",
        "C21",
        "C22",
        "C23",
    }
    assert families["exact-protection-historical"]["evaluation_scopes"] == [
        "historical-a1"
    ]
    assert "pr4_exact" not in families["exact-protection-historical"]["variants"]
    assert set(families["fill-backend"]["variants"]) == {
        "current_lama",
        "ballons_lama",
        "robust_flat_median",
        "planar_gradient",
        "telea",
        "conditional_hybrid",
        "conditional_refill_existing",
    }


def test_generalization_synthetic_v4_covers_required_failure_families(
    tmp_path: Path,
) -> None:
    payload = build_generalization_synthetic_manifest(tmp_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    pages = load_stage1_manifest(manifest)
    by_id = {page.page_id: page for page in pages}

    assert len(pages) == 18
    assert "synthetic-small-cjk-bright" in by_id
    assert "synthetic-vertical-outline" in by_id
    assert "synthetic-crop-edge" in by_id
    assert "synthetic-paper-noise" in by_id
    assert "synthetic-halftone" in by_id
    assert "synthetic-hatching" in by_id
    assert "synthetic-line-art" in by_id
    assert "synthetic-partial-detection" in by_id
    assert "synthetic-complete-miss" in by_id
    assert "synthetic-silhouette-under" in by_id
    assert "synthetic-silhouette-over" in by_id
    assert "synthetic-silhouette-empty" in by_id
    assert "synthetic-ownership-conflict" in by_id
    assert by_id["synthetic-unowned-meaningful"].target_instances[0].semantic_role == "dialogue_free"
    assert by_id["synthetic-preserve-sfx"].no_edit is True
    assert by_id["synthetic-preserve-sfx"].target_instances[0].processing_action == "preserve"


def test_generalization_shadow_shift_clips_without_page_wrap() -> None:
    mask = np.zeros((16, 20), np.uint8)
    mask[-3:, -3:] = 255

    shifted = shift_generalization_mask(mask, 3, 3)

    assert np.count_nonzero(shifted) == 0
    assert np.count_nonzero(shifted[:4, :4]) == 0


def test_manifest_rejects_instance_owned_only_by_a_different_region(
    tmp_path: Path,
) -> None:
    shape = (24, 32)
    source = np.full((*shape, 3), 180, np.uint8)
    target = np.zeros(shape, np.uint8)
    target[6:10, 18:22] = 255
    left = np.zeros(shape, np.uint8)
    left[:, :16] = 255
    right = np.zeros(shape, np.uint8)
    right[:, 16:] = 255
    zero = np.zeros(shape, np.uint8)
    paths = {
        "source": _write_image(tmp_path / "source.png", source),
        "target": _write_image(tmp_path / "target.png", target),
        "left": _write_image(tmp_path / "left.png", left),
        "right": _write_image(tmp_path / "right.png", right),
        "zero": _write_image(tmp_path / "zero.png", zero),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "pages": [
                    {
                        "page_id": "p",
                        "path": paths["source"],
                        "expected_edit": "required",
                        "target_text_mask": paths["target"],
                        "preserve_mask": paths["zero"],
                        "protected_structure_mask": paths["zero"],
                        "ambiguous_structure_mask": paths["zero"],
                        "ownership_mask": paths["right"],
                        "claim_seed_mask": paths["target"],
                        "bubble_interior_mask": paths["right"],
                        "corner_protect_mask": paths["zero"],
                        "existing_source_edit_mask": paths["zero"],
                        "target_instances": [
                            {
                                "instance_id": "misowned",
                                "region_id": "left",
                                "mask_path": paths["target"],
                                "semantic_role": "dialogue_bubble",
                                "processing_action": "translate_inpaint",
                                "priority": "required",
                            }
                        ],
                        "regions": [
                            {
                                "region_id": "left",
                                "bubble_route_class": "ambiguous",
                                "bubble_interior_mask": paths["left"],
                                "ownership_mask": paths["left"],
                                "protected_structure_mask": paths["zero"],
                                "ambiguous_structure_mask": paths["zero"],
                                "corner_protect_mask": paths["zero"],
                            },
                            {
                                "region_id": "right",
                                "bubble_route_class": "ambiguous",
                                "bubble_interior_mask": paths["right"],
                                "ownership_mask": paths["right"],
                                "protected_structure_mask": paths["zero"],
                                "ambiguous_structure_mask": paths["zero"],
                                "corner_protect_mask": paths["zero"],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    page = load_stage1_manifest(manifest)[0]
    with pytest.raises(ValueError, match="referenced region ownership"):
        load_page_masks(page, shape)


def test_paired_target_proposal_extracts_removed_source_strokes_only() -> None:
    source = np.full((80, 96, 3), 235, np.uint8)
    paired = source.copy()
    cv2.putText(source, "A", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    cv2.putText(paired, "B", (58, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (20, 20, 20), 3)
    source_text = np.zeros(source.shape[:2], np.uint8)
    cv2.putText(source_text, "A", (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)
    paired_text = np.zeros(source.shape[:2], np.uint8)
    cv2.putText(paired_text, "B", (58, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 255, 3)

    proposal = paired_old_text_proposal(source, paired)

    assert np.count_nonzero(proposal.extent_mask & source_text) > 0
    assert np.count_nonzero(proposal.extent_mask & paired_text) == 0
    assert proposal.instance_masks


def test_paired_target_proposal_rejects_global_compression_noise() -> None:
    source = np.full((64, 72, 3), 190, np.uint8)
    noise = np.indices(source.shape[:2]).sum(axis=0) % 5
    paired = np.clip(source.astype(np.int16) + noise[..., None] - 2, 0, 255).astype(
        np.uint8
    )

    proposal = paired_old_text_proposal(source, paired)

    assert np.count_nonzero(proposal.extent_mask) == 0
    assert proposal.instance_masks == ()


def test_source_extent_variants_expand_only_source_local_seed_support() -> None:
    source = np.full((88, 120, 3), 235, np.uint8)
    text = np.zeros(source.shape[:2], np.uint8)
    cv2.putText(text, "A", (18, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.6, 255, 4, cv2.LINE_AA)
    source[text > 0] = 20
    line = np.zeros(source.shape[:2], np.uint8)
    cv2.line(line, (86, 12), (86, 76), 255, 3, cv2.LINE_AA)
    source[line > 0] = 25
    seed = np.zeros(source.shape[:2], np.uint8)
    seed[40:52, 30:38] = text[40:52, 30:38]

    variants = source_extent_variants(source, seed)

    assert set(variants) == {
        "strict",
        "balanced",
        "edge_supported",
        "location_dilate1",
        "location_dilate2",
    }
    assert all(np.count_nonzero(value & text) >= np.count_nonzero(seed) for value in variants.values())
    assert all(np.count_nonzero(value & line) == 0 for value in variants.values())


def test_source_extent_variants_fail_closed_for_empty_or_wrong_shape() -> None:
    source = np.full((20, 24, 3), 180, np.uint8)
    empty = source_extent_variants(source, np.zeros((20, 24), np.uint8))
    assert all(np.count_nonzero(value) == 0 for value in empty.values())
    with pytest.raises(ValueError, match="shape mismatch"):
        source_extent_variants(source, np.zeros((19, 24), np.uint8))


def test_independent_target_review_keeps_unpaired_inventory_pending(
    tmp_path: Path,
) -> None:
    shape = (48, 64)
    source = np.full((*shape, 3), 235, np.uint8)
    cv2.putText(source, "A", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (20, 20, 20), 2)
    source_a = _write_image(tmp_path / "source-a.png", source)
    source_b = _write_image(tmp_path / "source-b.png", source)
    location = np.zeros(shape, np.uint8)
    location[18:31, 14:28] = 255
    location_path = _write_image(tmp_path / "location.png", location)
    ownership = np.zeros(shape, np.uint8)
    ownership[8:42, 6:40] = 255
    ownership_path = _write_image(tmp_path / "ownership.png", ownership)
    target = np.zeros(shape, np.uint8)
    target[18:31, 14:28] = 255
    target_path = _write_image(tmp_path / "semantic-target.png", target)

    source_index = tmp_path / "source-index.json"
    source_index.write_text(
        json.dumps(
            {
                "pages": [
                    {"page_id": "paired", "path": source_a},
                    {"page_id": "unpaired", "path": source_b},
                ]
            }
        ),
        encoding="utf-8",
    )
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_id": "paired",
                        "regions": [
                            {"region_id": "r1", "ownership_mask": ownership_path}
                        ],
                        "target_instances": [
                            {
                                "instance_id": "i1",
                                "region_id": "r1",
                                "priority": "required",
                                "semantic_role": "dialogue_bubble",
                                "mask_path": target_path,
                            }
                        ],
                    },
                    {
                        "page_id": "unpaired",
                        "regions": [],
                        "target_instances": [
                            {
                                "instance_id": "i2",
                                "region_id": "region-page-review",
                                "priority": "required",
                                "semantic_role": "dialogue_free",
                                "mask_path": target_path,
                            }
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    paired = tmp_path / "paired.json"
    paired.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "pages": [
                    {"page_id": "paired", "target_text_mask": location_path},
                    {"page_id": "unpaired", "target_text_mask": None},
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = build_independent_target_review(
        source_index,
        semantic,
        paired,
        tmp_path / "review",
    )

    assert payload["review_complete"] is False
    assert payload["source_index"] == str(source_index.resolve())
    assert payload["source_index_sha256"] == hashlib.sha256(
        source_index.read_bytes()
    ).hexdigest()
    assert payload["target_inventory_independent"] is False
    assert payload["review_row_count"] == 2
    assert payload["rows"][0]["semantic_role_proposal"] == "dialogue_bubble"
    assert payload["rows"][0]["selected_extent"] is None
    assert payload["rows"][1]["semantic_role_proposal"] == "dialogue_free"
    assert payload["rows"][1]["inventory_source"] == "source_manifest_location_aid_only"
    assert payload["rows"][1]["source_priority_proposal"] == "required"
    assert payload["full_page_inventory_pending_count"] == 1
    assert payload["full_page_inventory_pending"][0]["page_id"] == "unpaired"


def test_independent_target_review_recorder_requires_every_row_and_page(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "review_complete": False,
                "rows": [{"review_id": "review-0000"}],
                "full_page_inventory_pending": [{"page_id": "p2"}],
            }
        ),
        encoding="utf-8",
    )
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "decisions": [],
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="review decisions differ"):
        record_independent_target_review(
            ledger_path, incomplete, tmp_path / "not-written.json"
        )

    complete = tmp_path / "complete.json"
    complete.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "decisions": [
                    {
                        "review_id": "review-0000",
                        "extent": "balanced",
                        "semantic": "required",
                    }
                ],
                "full_page_inventory": [
                    {"page_id": "p2", "status": "complete_no_required_text"}
                ],
            }
        ),
        encoding="utf-8",
    )
    payload = record_independent_target_review(
        ledger_path, complete, tmp_path / "recorded.json"
    )
    assert payload["review_complete"] is True
    assert len(payload["decisions"]) == 1
    assert len(payload["full_page_inventory"]) == 1


def test_independent_target_review_recorder_rejects_edit_extent_for_ambiguous(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "review_complete": False,
                "rows": [{"review_id": "review-0000"}],
                "full_page_inventory_pending": [],
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "decisions": [
                    {
                        "review_id": "review-0000",
                        "extent": "strict",
                        "semantic": "ambiguous",
                    }
                ],
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-edit semantic"):
        record_independent_target_review(
            ledger, decisions, tmp_path / "not-written.json"
        )


def test_independent_target_review_applier_seals_only_selected_safe_extent(
    tmp_path: Path,
) -> None:
    shape = (32, 40)
    source = _write_image(tmp_path / "source.png", np.full((*shape, 3), 230, np.uint8))
    empty = _write_image(tmp_path / "empty.png", np.zeros(shape, np.uint8))
    ownership_mask = np.zeros(shape, np.uint8)
    ownership_mask[4:28, 4:36] = 255
    ownership = _write_image(tmp_path / "ownership.png", ownership_mask)
    protect_mask = np.zeros(shape, np.uint8)
    protect_mask[10:12, 6:34] = 255
    protect = _write_image(tmp_path / "protect.png", protect_mask)
    extent_mask = np.zeros(shape, np.uint8)
    extent_mask[8:16, 12:22] = 255
    extent = _write_image(tmp_path / "extent.png", extent_mask)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "corpus_id": "example",
                "pages": [
                    {
                        "page_id": "p1",
                        "target_text_mask": None,
                        "preserve_mask": empty,
                        "protected_structure_mask": protect,
                        "ambiguous_structure_mask": empty,
                        "ownership_mask": ownership,
                        "expected_edit": "none",
                        "target_instances": [],
                        "regions": [
                            {
                                "region_id": "r1",
                                "bubble_route_class": "line_art",
                                "bubble_interior_mask": empty,
                                "ownership_mask": ownership,
                                "protected_structure_mask": protect,
                                "ambiguous_structure_mask": empty,
                                "corner_protect_mask": empty,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source_index = tmp_path / "source-index.json"
    source_index.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_id": "p1",
                        "path": source,
                        "height": shape[0],
                        "width": shape[1],
                        "source_sha256": hashlib.sha256(
                            Path(source).read_bytes()
                        ).hexdigest(),
                        "set_id": "example",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-ledger-v4",
                "candidate_seen": False,
                "review_complete": False,
                "source_index": str(source_index.resolve()),
                "source_index_sha256": hashlib.sha256(
                    source_index.read_bytes()
                ).hexdigest(),
                "rows": [
                    {
                        "review_id": "review-0000",
                        "page_id": "p1",
                        "region_id": "r1",
                        "semantic_role_proposal": "dialogue_bubble",
                        "location_seed": extent,
                        "extent_variants": {"location_dilate1": extent},
                    }
                ],
                "full_page_inventory_pending": [],
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-decisions-v4",
                "candidate_seen": False,
                "review_complete": True,
                "decisions": [
                    {
                        "review_id": "review-0000",
                        "extent": "location_dilate1",
                        "semantic": "required",
                    }
                ],
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )

    _bind_independent_review_fixture(semantic, ledger, decisions)
    payload = apply_independent_target_review(
        semantic, ledger, decisions, tmp_path / "output"
    )

    page = payload["pages"][0]
    target = cv2.imread(page["target_text_mask"], cv2.IMREAD_GRAYSCALE)
    assert page["annotation_basis"] == "source_only_v4"
    assert payload["target_inventory_independent"] is True
    assert payload["target_extent_independent"] is True
    normalized_protect = cv2.imread(
        page["protected_structure_mask"], cv2.IMREAD_GRAYSCALE
    )
    assert np.array_equal(target, extent_mask)
    assert np.count_nonzero(normalized_protect & target) == 0


def test_independent_target_review_uses_source_only_manual_row_extent(
    tmp_path: Path,
) -> None:
    shape = (32, 40)
    source = _write_image(tmp_path / "source.png", np.full((*shape, 3), 230, np.uint8))
    empty = _write_image(tmp_path / "empty.png", np.zeros(shape, np.uint8))
    ownership_mask = np.zeros(shape, np.uint8)
    ownership_mask[3:29, 3:37] = 255
    ownership = _write_image(tmp_path / "ownership.png", ownership_mask)
    protect_mask = np.zeros(shape, np.uint8)
    protect_mask[14:16, 5:35] = 255
    protect = _write_image(tmp_path / "protect.png", protect_mask)
    proposal_mask = np.zeros(shape, np.uint8)
    proposal_mask[8:10, 12:14] = 255
    proposal = _write_image(tmp_path / "proposal.png", proposal_mask)
    manual_mask = np.zeros(shape, np.uint8)
    manual_mask[7:20, 10:24] = 255
    manual = _write_image(tmp_path / "manual.png", manual_mask)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "corpus_id": "example",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": source,
                        "height": shape[0],
                        "width": shape[1],
                        "target_text_mask": None,
                        "preserve_mask": empty,
                        "protected_structure_mask": protect,
                        "ambiguous_structure_mask": empty,
                        "ownership_mask": ownership,
                        "expected_edit": "none",
                        "target_instances": [],
                        "regions": [
                            {
                                "region_id": "r1",
                                "bubble_route_class": "line_art",
                                "bubble_interior_mask": empty,
                                "ownership_mask": ownership,
                                "protected_structure_mask": protect,
                                "ambiguous_structure_mask": empty,
                                "corner_protect_mask": empty,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-ledger-v4",
                "candidate_seen": False,
                "review_complete": False,
                "rows": [
                    {
                        "review_id": "review-0000",
                        "page_id": "p1",
                        "region_id": "r1",
                        "semantic_role_proposal": "dialogue_bubble",
                        "location_seed": proposal,
                        "extent_variants": {"balanced": proposal},
                    }
                ],
                "full_page_inventory_pending": [],
            }
        ),
        encoding="utf-8",
    )
    raw_decisions = tmp_path / "raw-decisions.json"
    raw_decisions.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "decisions": [
                    {
                        "review_id": "review-0000",
                        "extent": "manual",
                        "semantic": "required",
                        "manual_extent_path": manual,
                    }
                ],
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    record_independent_target_review(ledger, raw_decisions, decisions)

    _bind_independent_review_fixture(semantic, ledger, decisions)
    payload = apply_independent_target_review(
        semantic, ledger, decisions, tmp_path / "output"
    )

    target = cv2.imread(payload["pages"][0]["target_text_mask"], cv2.IMREAD_GRAYSCALE)
    expected = manual_mask.copy()
    expected[ownership_mask == 0] = 0
    assert np.array_equal(target, expected)
    assert np.count_nonzero(target) > np.count_nonzero(proposal_mask)


def test_independent_target_review_applier_replaces_unpaired_page_with_instances(
    tmp_path: Path,
) -> None:
    shape = (36, 48)
    source_path = tmp_path / "source.png"
    source = np.full((*shape, 3), 230, np.uint8)
    source[7:13, 8:16] = 20
    source[21:27, 28:37] = 20
    _write_image(source_path, source)
    empty = _write_image(tmp_path / "empty.png", np.zeros(shape, np.uint8))
    ownership_mask = np.zeros(shape, np.uint8)
    ownership_mask[3:32, 4:44] = 255
    ownership = _write_image(tmp_path / "ownership.png", ownership_mask)
    first_mask = np.zeros(shape, np.uint8)
    first_mask[7:13, 8:16] = 255
    second_mask = np.zeros(shape, np.uint8)
    second_mask[21:27, 28:37] = 255
    first = _write_image(tmp_path / "first.png", first_mask)
    second = _write_image(tmp_path / "second.png", second_mask)
    old_target = _write_image(tmp_path / "old-target.png", first_mask | second_mask)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-manifest-v4",
                "corpus_id": "example",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": str(source_path),
                        "height": shape[0],
                        "width": shape[1],
                        "target_text_mask": old_target,
                        "preserve_mask": empty,
                        "protected_structure_mask": empty,
                        "ambiguous_structure_mask": empty,
                        "ownership_mask": ownership,
                        "expected_edit": "required",
                        "target_instances": [
                            {
                                "instance_id": "old-circular",
                                "region_id": "r1",
                                "mask_path": old_target,
                                "priority": "required",
                            }
                        ],
                        "regions": [
                            {
                                "region_id": "r1",
                                "bubble_route_class": "clean_flat",
                                "bubble_interior_mask": ownership,
                                "ownership_mask": ownership,
                                "protected_structure_mask": empty,
                                "ambiguous_structure_mask": empty,
                                "corner_protect_mask": empty,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    inventory = tmp_path / "manual-inventory.json"
    inventory.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-manual-inventory-v4",
                "candidate_seen": False,
                "source_reviewed": True,
                "page_id": "p1",
                "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                "instances": [
                    {
                        "instance_id": "line-1",
                        "region_id": "r1",
                        "mask_path": first,
                        "semantic_role": "dialogue_bubble",
                        "priority": "required",
                    },
                    {
                        "instance_id": "line-2",
                        "region_id": "r1",
                        "mask_path": second,
                        "semantic_role": "dialogue_bubble",
                        "priority": "required",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-ledger-v4",
                "candidate_seen": False,
                "rows": [],
                "full_page_inventory_pending": [{"page_id": "p1"}],
            }
        ),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-target-review-decisions-v4",
                "candidate_seen": False,
                "review_complete": True,
                "decisions": [],
                "full_page_inventory": [
                    {
                        "page_id": "p1",
                        "status": "complete_with_manual_inventory",
                        "manual_inventory_path": str(inventory),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    _bind_independent_review_fixture(semantic, ledger, decisions)
    payload = apply_independent_target_review(
        semantic, ledger, decisions, tmp_path / "output"
    )

    page = payload["pages"][0]
    assert [value["instance_id"] for value in page["target_instances"]] == [
        "manual-line-1",
        "manual-line-2",
    ]
    assert payload["instance_counts"] == {
        "required": 2,
        "preserve": 0,
        "ambiguous": 0,
    }
    target = cv2.imread(page["target_text_mask"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(target, first_mask | second_mask)
    assert all(value["instance_id"] != "old-circular" for value in page["target_instances"])


def test_independent_review_decision_builder_requires_unowned_rows_and_all_sheets(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps(
            {
                "candidate_seen": False,
                "rows": [
                    {
                        "review_id": "review-0000",
                        "inventory_source": "paired_location_with_human_semantic_region",
                    },
                    {
                        "review_id": "review-0001",
                        "inventory_source": "paired_location_outside_known_ownership",
                    },
                ],
                "review_sheets": ["sheet-1.jpg"],
                "full_page_inventory_pending": [],
            }
        ),
        encoding="utf-8",
    )
    overrides = tmp_path / "overrides.json"
    overrides.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-independent-review-overrides-v4",
                "candidate_seen": False,
                "reviewed_sheets": [1],
                "row_overrides": {},
                "full_page_inventory": [],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unowned review row"):
        build_review_decisions(ledger, overrides, tmp_path / "not-written.json")

    value = json.loads(overrides.read_text(encoding="utf-8"))
    value["row_overrides"]["review-0001"] = {
        "extent": "reject",
        "semantic": "not_text",
    }
    overrides.write_text(json.dumps(value), encoding="utf-8")
    payload = build_review_decisions(ledger, overrides, tmp_path / "decisions.json")
    assert payload["decisions"] == [
        {"review_id": "review-0000", "extent": "balanced", "semantic": "required"},
        {"review_id": "review-0001", "extent": "reject", "semantic": "not_text"},
    ]


def test_binary_mask_does_not_allocate_int64_where_temporary(monkeypatch) -> None:
    source = np.array([[0, 1], [255, -1]], dtype=np.int16)

    def forbidden_where(*_args, **_kwargs):
        raise AssertionError("binary mask must not promote 2K masks through np.where")

    monkeypatch.setattr(np, "where", forbidden_where)

    result = binary_mask(source)

    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert result.tolist() == [[0, 255], [255, 0]]


def test_detector_roi_triggers_select_source_only_ownership_components() -> None:
    shape = (24, 36)
    ownership = np.zeros(shape, np.uint8)
    ownership[3:11, 3:13] = 255
    ownership[13:21, 22:33] = 255
    raw = np.zeros(shape, np.uint8)
    raw[5:7, 5:8] = 255
    refined = raw.copy()
    refined[7:9, 8:10] = 255
    source_seed = np.zeros(shape, np.uint8)
    source_seed[4:6, 4:6] = 255

    missing = detector_roi_trigger_mask(
        "seed-missing",
        ownership=ownership,
        primary_raw=raw,
        primary_refined=refined,
        source_seed=source_seed,
    )
    disagreement = detector_roi_trigger_mask(
        "raw-refined-disagreement",
        ownership=ownership,
        primary_raw=raw,
        primary_refined=refined,
        source_seed=source_seed,
    )
    unavailable = detector_roi_trigger_mask(
        "source-seed-unavailable",
        ownership=ownership,
        primary_raw=raw,
        primary_refined=refined,
        source_seed=source_seed,
    )

    assert np.count_nonzero(missing[13:21, 22:33]) == 88
    assert np.count_nonzero(missing[:12]) == 0
    assert np.count_nonzero(disagreement[3:11, 3:13]) == 80
    assert np.count_nonzero(disagreement[13:21, 22:33]) == 0
    assert np.array_equal(unavailable, missing)


def test_detector_fusion_never_turns_ownership_geometry_into_claim() -> None:
    shape = (20, 28)
    ownership = np.zeros(shape, np.uint8)
    ownership[2:18, 2:26] = 255
    primary = np.zeros(shape, np.uint8)
    primary[5:8, 5:8] = 255
    secondary = np.zeros(shape, np.uint8)
    secondary[12:15, 18:22] = 255
    secondary[0:2, 0:2] = 255
    trigger = np.zeros(shape, np.uint8)
    trigger[10:18, 15:26] = 255

    union = fuse_detector_claims(
        "or", primary, secondary, ownership=ownership
    )
    intersection = fuse_detector_claims(
        "and", primary, secondary, ownership=ownership
    )
    recovery = fuse_detector_claims(
        "gated-recovery",
        primary,
        secondary,
        ownership=ownership,
        trigger_mask=trigger,
    )

    assert np.count_nonzero(union) == 21
    assert np.count_nonzero(intersection) == 0
    assert np.count_nonzero(recovery) == 21
    assert np.count_nonzero(recovery[trigger > 0]) == 12
    assert np.count_nonzero(recovery[(trigger > 0) & (secondary == 0)]) == 0
    assert np.count_nonzero(union[ownership == 0]) == 0


def test_detector_fusion_matrix_covers_singles_pairs_and_roi_triggers() -> None:
    runs = detector_fusion_runs(
        ("primary", "secondary", "roi"),
        frozenset({"roi"}),
    )
    run_ids = {row["run_id"] for row in runs}

    assert len(runs) == 19
    assert {"primary", "secondary", "roi"}.issubset(run_ids)
    assert "primary__or__secondary" in run_ids
    assert "primary__and__roi" in run_ids
    assert "primary__gated_always__roi" in run_ids
    assert "primary__gated_seed_missing__roi" in run_ids
    assert "secondary__gated_union__roi" in run_ids


def test_detector_fusion_best_effort_admission_keeps_recall_and_safety_views() -> None:
    def row(run_id: str, missed: int, coverage: float, false_edit: int) -> dict[str, object]:
        return {
            "run_id": run_id,
            "seed_eligible": missed == 0,
            "metrics": {
                "missed_target_instance_count": missed,
                "aggregate_target_coverage": coverage,
                "minimum_target_instance_coverage": 0.0,
                "false_edit_pixel_count": false_edit,
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
            },
        }

    selected = select_seed_admission_run_ids(
        [
            row("coverage", 7, 0.95, 1700),
            row("safety", 7, 0.91, 1200),
            row("worse-recall", 8, 0.99, 10),
        ]
    )

    assert selected == frozenset({"coverage", "safety"})


def test_detector_fusion_strict_seed_candidates_take_precedence() -> None:
    rows = [
        {
            "run_id": "strict",
            "seed_eligible": True,
            "metrics": {
                "missed_target_instance_count": 0,
                "aggregate_target_coverage": 0.9,
                "minimum_target_instance_coverage": 0.8,
                "false_edit_pixel_count": 100,
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
            },
        },
        {
            "run_id": "best-effort",
            "seed_eligible": False,
            "metrics": {
                "missed_target_instance_count": 1,
                "aggregate_target_coverage": 1.0,
                "minimum_target_instance_coverage": 0.0,
                "false_edit_pixel_count": 0,
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
            },
        },
    ]

    assert select_seed_admission_run_ids(rows) == frozenset({"strict"})


def test_detector_fusion_respects_manifest_existing_source_edit(tmp_path: Path) -> None:
    shape = (18, 24)
    source = np.full((*shape, 3), 180, np.uint8)
    target = np.zeros(shape, np.uint8)
    target[5:9, 6:12] = 255
    ownership = np.full(shape, 255, np.uint8)
    zero = np.zeros(shape, np.uint8)
    existing = target.copy()
    source_path = _write_image(tmp_path / "source.png", source)
    target_path = _write_image(tmp_path / "target.png", target)
    ownership_path = _write_image(tmp_path / "ownership.png", ownership)
    zero_path = _write_image(tmp_path / "zero.png", zero)
    existing_path = _write_image(tmp_path / "existing.png", existing)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-bakeoff-manifest-v3",
                "pages": [
                {
                    "page_id": "p1",
                    "path": source_path,
                    "target_mask_provenance": "synthetic_ground_truth",
                        "target_extent_independent": True,
                        "target_inventory_independent": True,
                        "target_review_complete": True,
                    "target_text_mask": target_path,
                        "target_instances": [
                            {"instance_id": "i1", "mask_path": target_path}
                        ],
                        "bubble_route_class": "clean_flat",
                        "bubble_interior_mask": ownership_path,
                        "protected_structure_mask": zero_path,
                        "ambiguous_structure_mask": zero_path,
                        "ownership_mask": ownership_path,
                        "corner_protect_mask": zero_path,
                        "existing_source_edit_mask": existing_path,
                        "expected_edit": "required",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-fusion-spec-v4",
                "candidates": {
                    "detector": {
                        "templates": {"raw": target_path, "refined": target_path}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    _write_strict_source_manifest(
        manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    output_root = tmp_path / "fusion-output"
    result = run_fusion_matrix(
        manifest_path,
        spec_path,
        output_root=output_root,
    )

    metrics = result["runs"][0]["metrics"]
    assert result["page_ids"] == ["p1"]
    assert result["pages"]["detector"][0]["page_id"] == "p1"
    assert len(result["logical_inventory_sha256"]) == 64
    assert result["output_artifact_inventory"]["artifact_count"] == 2
    assert (
        output_root
        / result["output_artifact_inventory"]["relative_path"]
    ).is_file()
    assert metrics["target_instance_seed_recall"] == 1.0
    assert metrics["aggregate_target_coverage"] == 0.0
    assert metrics["output_mask_set_sha256"] != hashlib.sha256().hexdigest()


def test_detector_fusion_marks_detector_derived_targets_information_limited(
    tmp_path: Path,
) -> None:
    shape = (18, 24)
    source = np.full((*shape, 3), 180, np.uint8)
    target = np.zeros(shape, np.uint8)
    target[5:9, 6:12] = 255
    ownership = np.full(shape, 255, np.uint8)
    zero = np.zeros(shape, np.uint8)
    source_path = _write_image(tmp_path / "source.png", source)
    target_path = _write_image(tmp_path / "target.png", target)
    ownership_path = _write_image(tmp_path / "ownership.png", ownership)
    zero_path = _write_image(tmp_path / "zero.png", zero)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-bakeoff-manifest-v3",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": source_path,
                        "target_mask_provenance": "current_ctd_raw_components",
                        "target_extent_independent": False,
                        "target_text_mask": target_path,
                        "target_instances": [
                            {"instance_id": "i1", "mask_path": target_path}
                        ],
                        "bubble_route_class": "clean_flat",
                        "bubble_interior_mask": ownership_path,
                        "protected_structure_mask": zero_path,
                        "ambiguous_structure_mask": zero_path,
                        "ownership_mask": ownership_path,
                        "corner_protect_mask": zero_path,
                        "existing_source_edit_mask": zero_path,
                        "expected_edit": "required",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-fusion-spec-v4",
                "candidates": {
                    "detector": {
                        "templates": {"raw": target_path, "refined": target_path}
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="strict source-only manifest v4"):
        run_fusion_matrix(manifest_path, spec_path)


def test_stage1_reads_unicode_source_and_mask_paths(tmp_path: Path) -> None:
    unicode_dir = tmp_path / "日本語_경로"
    unicode_dir.mkdir()
    source = np.full((9, 11, 3), 177, np.uint8)
    mask = np.zeros((9, 11), np.uint8)
    mask[2:5, 3:7] = 255
    source_path = unicode_dir / "원본_日本.png"
    mask_path = unicode_dir / "마스크_日本.png"
    assert cv2.imencode(".png", source)[1].tofile(source_path) is None
    assert cv2.imencode(".png", mask)[1].tofile(mask_path) is None
    page = Stage1Page(
        page_id="unicode",
        source_image=str(source_path),
        target_text_mask=str(mask_path),
        protected_structure_mask=None,
        ambiguous_structure_mask=None,
    )

    loaded = load_page_masks(page, source.shape[:2])

    assert np.array_equal(read_stage1_image(str(source_path)), source)
    assert np.array_equal(loaded.target, mask)


def _page_masks(shape: tuple[int, int] = (48, 64)) -> PageMasks:
    zeros = np.zeros(shape, dtype=np.uint8)
    return PageMasks(
        target=zeros.copy(),
        protected=zeros.copy(),
        ambiguous=zeros.copy(),
        ownership=np.full(shape, 255, np.uint8),
        claim_seed=np.full(shape, 255, np.uint8),
        existing_edit=zeros.copy(),
        bubble_interior=np.full(shape, 255, np.uint8),
        corner=zeros.copy(),
    )


def test_manifest_v3_requires_instance_route_and_evidence_fields(tmp_path: Path) -> None:
    source = np.full((24, 32, 3), 200, dtype=np.uint8)
    target = np.zeros((24, 32), dtype=np.uint8)
    target[8:12, 10:14] = 255
    zeros = np.zeros_like(target)
    full = np.full_like(target, 255)
    source_path = _write_image(tmp_path / "source.png", source)
    target_path = _write_image(tmp_path / "target.png", target)
    zeros_path = _write_image(tmp_path / "zeros.png", zeros)
    full_path = _write_image(tmp_path / "full.png", full)
    manifest = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v3",
        "pages": [
            {
                "page_id": "p1",
                "path": source_path,
                "target_mask_provenance": "synthetic_ground_truth",
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
                "target_text_mask": target_path,
                "target_instances": [
                    {"instance_id": "glyph-1", "mask_path": target_path}
                ],
                "bubble_route_class": "clean_flat",
                "bubble_interior_mask": full_path,
                "protected_structure_mask": zeros_path,
                "ambiguous_structure_mask": zeros_path,
                "ownership_mask": full_path,
                "corner_protect_mask": zeros_path,
                "expected_edit": "required",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    pages = load_stage1_manifest(path)
    masks = load_page_masks(pages[0], source.shape[:2])

    assert pages[0].bubble_route_class == "clean_flat"
    assert [value[0] for value in masks.target_instances] == ["glyph-1"]
    assert np.array_equal(masks.target_instances[0][1], target)


def test_development_source_index_binds_proposal_only_pair_by_sha(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    paired_dir = tmp_path / "paired"
    source_dir.mkdir()
    paired_dir.mkdir()
    source = np.full((12, 16, 3), 180, np.uint8)
    paired = source.copy()
    paired[3:7, 4:8] = 240
    _write_image(source_dir / "001.png", source)
    _write_image(paired_dir / "001.jpg", paired)

    payload = build_source_index([f"elven::{source_dir}::{paired_dir}"])

    page = payload["pages"][0]
    assert payload["candidate_images_generated"] is False
    assert payload["inpainting_invoked"] is False
    assert page["page_id"] == "elven-001"
    assert page["paired_reference"]["proposal_only"] is True
    assert page["paired_reference"]["source_sha256"] == page["source_sha256"]
    assert len(page["paired_reference"]["reference_sha256"]) == 64


def test_development_source_index_rejects_incomplete_pair_set(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    paired_dir = tmp_path / "paired"
    source_dir.mkdir()
    paired_dir.mkdir()
    _write_image(source_dir / "001.png", np.zeros((8, 8, 3), np.uint8))

    with pytest.raises(ValueError, match="paired set mismatch"):
        build_source_index([f"elven::{source_dir}::{paired_dir}"])


def test_source_proposal_semantics_are_candidate_blind_and_fail_closed() -> None:
    assert propose_semantic_contract(
        "text_bubble", paired_change_contact=False
    ) == ("dialogue_bubble", "translate_inpaint", "required")
    assert propose_semantic_contract(
        "text_free", paired_change_contact=True
    ) == ("dialogue_free", "translate_inpaint", "required")
    assert propose_semantic_contract(
        "text_free", paired_change_contact=False
    ) == ("ambiguous", "review", "ambiguous")


def test_source_review_rebuilds_required_preserve_and_ambiguous_masks(
    tmp_path: Path,
) -> None:
    shape = (16, 20)
    zeros = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    first = zeros.copy()
    first[2:5, 2:5] = 255
    second = zeros.copy()
    second[8:11, 10:13] = 255
    paths = {
        "zeros": _write_image(tmp_path / "zeros.png", zeros),
        "full": _write_image(tmp_path / "full.png", full),
        "first": _write_image(tmp_path / "first.png", first),
        "second": _write_image(tmp_path / "second.png", second),
    }
    proposals = {
        "schema_version": "inpaint-factorized-source-decisions-v4",
        "corpus_id": "fixture",
        "candidate_seen": False,
        "pages": [{
            "page_id": "p1",
            "target_text_mask": paths["first"],
            "preserve_mask": paths["zeros"],
            "protected_structure_mask": paths["zeros"],
            "ambiguous_structure_mask": paths["second"],
            "ownership_mask": paths["full"],
            "claim_seed_mask": paths["full"],
            "bubble_interior_mask": paths["full"],
            "corner_protect_mask": paths["zeros"],
            "expected_edit": "required",
            "target_instances": [
                {"instance_id": "i1", "region_id": "r1", "mask_path": paths["first"], "semantic_role": "dialogue_bubble", "processing_action": "translate_inpaint", "priority": "required"},
                {"instance_id": "i2", "region_id": "r1", "mask_path": paths["second"], "semantic_role": "ambiguous", "processing_action": "review", "priority": "ambiguous"},
            ],
            "regions": [{
                "region_id": "r1",
                "bubble_route_class": "ambiguous",
                "bubble_interior_mask": paths["full"],
                "ownership_mask": paths["full"],
                "protected_structure_mask": paths["zeros"],
                "ambiguous_structure_mask": paths["second"],
                "corner_protect_mask": paths["zeros"],
            }],
        }],
    }
    ledger = {
        "candidate_seen": False,
        "rows": [{"review_id": "review-1", "page_id": "p1", "instance_id": "i2"}],
    }
    review = {
        "schema_version": "inpaint-source-review-decisions-v4",
        "candidate_seen": False,
        "decisions": [{"review_id": "review-1", "decision": "preserve", "semantic_role": "sfx"}],
    }
    proposal_path = tmp_path / "proposals.json"
    ledger_path = tmp_path / "ledger.json"
    review_path = tmp_path / "review.json"
    proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    review_path.write_text(json.dumps(review), encoding="utf-8")

    result = apply_source_review(
        proposal_path, ledger_path, review_path, tmp_path / "reviewed"
    )

    page = result["pages"][0]
    assert result["candidate_seen"] is False
    assert result["review_complete"] is True
    assert page["target_instances"][1]["priority"] == "optional"
    assert np.array_equal(cv2.imread(page["target_text_mask"], 0), first)
    assert np.array_equal(cv2.imread(page["preserve_mask"], 0), second)
    assert not np.any(cv2.imread(page["ambiguous_structure_mask"], 0))


def test_source_review_record_is_candidate_blind_and_complete(tmp_path: Path) -> None:
    ledger = {
        "candidate_seen": False,
        "rows": [
            {"review_id": "review-0000", "semantic_role_proposal": "ambiguous"},
            {"review_id": "review-0001", "semantic_role_proposal": "dialogue_bubble"},
            {"review_id": "review-0002", "semantic_role_proposal": "ambiguous"},
        ],
    }
    ledger_path = tmp_path / "ledger.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")

    payload = record_source_review(
        ledger_path,
        tmp_path / "decisions.json",
        preserve_ids={"review-0000"},
        ui_ids={"review-0002"},
    )

    assert payload["candidate_seen"] is False
    assert payload["review_complete"] is True
    assert [(row["decision"], row["semantic_role"]) for row in payload["decisions"]] == [
        ("preserve", "sfx"),
        ("required", "dialogue_bubble"),
        ("required", "ui_or_sign"),
    ]


def test_source_review_records_empty_no_edit_page_as_fail_closed_region(
    tmp_path: Path,
) -> None:
    shape = (8, 10)
    zeros = _write_image(tmp_path / "zeros.png", np.zeros(shape, np.uint8))
    proposals = {
        "schema_version": "inpaint-factorized-source-decisions-v4",
        "corpus_id": "fixture",
        "candidate_seen": False,
        "pages": [{
            "page_id": "empty",
            "target_text_mask": zeros,
            "preserve_mask": zeros,
            "protected_structure_mask": zeros,
            "ambiguous_structure_mask": zeros,
            "ownership_mask": zeros,
            "claim_seed_mask": zeros,
            "bubble_interior_mask": zeros,
            "corner_protect_mask": zeros,
            "expected_edit": "none",
            "target_instances": [],
            "regions": [],
        }],
    }
    ledger = {"candidate_seen": False, "rows": []}
    decisions = {
        "schema_version": "inpaint-source-review-decisions-v4",
        "candidate_seen": False,
        "decisions": [],
    }
    proposal_path = tmp_path / "proposals.json"
    ledger_path = tmp_path / "ledger.json"
    review_path = tmp_path / "review.json"
    proposal_path.write_text(json.dumps(proposals), encoding="utf-8")
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    review_path.write_text(json.dumps(decisions), encoding="utf-8")

    payload = apply_source_review(
        proposal_path, ledger_path, review_path, tmp_path / "reviewed"
    )

    page = payload["pages"][0]
    assert page["expected_edit"] == "none"
    assert page["target_text_mask"] is None
    assert page["regions"][0]["region_id"] == "region-page-empty"
    assert page["regions"][0]["proposal"]["empty_no_edit_page"] is True


def test_manifest_v3_rejects_target_instance_union_mismatch(tmp_path: Path) -> None:
    shape = (16, 20)
    source_path = _write_image(tmp_path / "source.png", np.full((*shape, 3), 180, np.uint8))
    target = np.zeros(shape, np.uint8)
    target[4:8, 4:8] = 255
    instance = np.zeros(shape, np.uint8)
    instance[4:7, 4:8] = 255
    zeros = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    paths = {
        "target": _write_image(tmp_path / "target.png", target),
        "instance": _write_image(tmp_path / "instance.png", instance),
        "zeros": _write_image(tmp_path / "zeros.png", zeros),
        "full": _write_image(tmp_path / "full.png", full),
    }
    manifest = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v3",
        "pages": [
            {
                "page_id": "p1",
                "path": source_path,
                "target_text_mask": paths["target"],
                "target_instances": [
                    {"instance_id": "glyph", "mask_path": paths["instance"]}
                ],
                "bubble_route_class": "clean_flat",
                "bubble_interior_mask": paths["full"],
                "protected_structure_mask": paths["zeros"],
                "ambiguous_structure_mask": paths["zeros"],
                "ownership_mask": paths["full"],
                "expected_edit": "required",
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    page = load_stage1_manifest(path)[0]

    with pytest.raises(ValueError, match="union of target_instances"):
        load_page_masks(page, shape)


def test_manifest_v4_loads_region_semantics_and_proposal_only_reference(
    tmp_path: Path,
) -> None:
    shape = (24, 32)
    source = np.full((*shape, 3), 180, np.uint8)
    required = np.zeros(shape, np.uint8)
    required[6:10, 6:10] = 255
    optional = np.zeros(shape, np.uint8)
    optional[14:18, 20:24] = 255
    ambiguous = np.zeros(shape, np.uint8)
    ambiguous[2:4, 24:28] = 255
    zeros = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    paths = {
        "source": _write_image(tmp_path / "source.png", source),
        "required": _write_image(tmp_path / "required.png", required),
        "optional": _write_image(tmp_path / "optional.png", optional),
        "ambiguous": _write_image(tmp_path / "ambiguous.png", ambiguous),
        "zeros": _write_image(tmp_path / "zeros.png", zeros),
        "full": _write_image(tmp_path / "full.png", full),
    }
    source_sha = hashlib.sha256(Path(paths["source"]).read_bytes()).hexdigest()
    reference_path = tmp_path / "reference.png"
    reference_path.write_bytes(Path(paths["source"]).read_bytes())
    manifest = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v4",
        "pages": [
            {
                "page_id": "p1",
                "path": paths["source"],
                "target_text_mask": paths["required"],
                "preserve_mask": paths["optional"],
                "target_instances": [
                    {
                        "instance_id": "dialogue",
                        "region_id": "bubble",
                        "mask_path": paths["required"],
                        "semantic_role": "dialogue_bubble",
                        "processing_action": "translate_inpaint",
                        "priority": "required",
                    },
                    {
                        "instance_id": "sfx",
                        "region_id": "bubble",
                        "mask_path": paths["optional"],
                        "semantic_role": "sfx",
                        "processing_action": "preserve",
                        "priority": "optional",
                    },
                    {
                        "instance_id": "review",
                        "region_id": "bubble",
                        "mask_path": paths["ambiguous"],
                        "semantic_role": "ambiguous",
                        "processing_action": "review",
                        "priority": "ambiguous",
                    },
                ],
                "regions": [
                    {
                        "region_id": "bubble",
                        "bubble_route_class": "clean_flat",
                        "bubble_interior_mask": paths["full"],
                        "ownership_mask": paths["full"],
                        "protected_structure_mask": paths["zeros"],
                        "ambiguous_structure_mask": paths["ambiguous"],
                        "corner_protect_mask": paths["zeros"],
                    }
                ],
                "protected_structure_mask": paths["zeros"],
                "ambiguous_structure_mask": paths["ambiguous"],
                "ownership_mask": paths["full"],
                "claim_seed_mask": paths["full"],
                "bubble_interior_mask": paths["full"],
                "corner_protect_mask": paths["zeros"],
                "existing_source_edit_mask": paths["zeros"],
                "expected_edit": "required",
                "paired_reference": {
                    "path": str(reference_path),
                    "source_sha256": source_sha,
                    "reference_sha256": source_sha,
                    "proposal_only": True,
                },
            }
        ],
    }
    path = tmp_path / "manifest-v4.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    page = load_stage1_manifest(path)[0]
    masks = load_page_masks(page, shape)

    assert [instance.priority for instance in page.target_instances] == [
        "required",
        "optional",
        "ambiguous",
    ]
    assert page.paired_reference is not None
    assert page.paired_reference.proposal_only is True
    assert len(masks.regions) == 1
    assert np.array_equal(masks.target, required)
    assert np.array_equal(masks.preserve, optional)

    claim = cv2.bitwise_or(required, optional)
    result = CandidateMaskResult("mixed", claim, claim, claim)
    row, edit = score_page(page, result, masks, variant="raw")

    assert np.count_nonzero(edit & optional) == 0
    assert row["preserve_edit_overlap"] == 0
    assert np.count_nonzero(edit & required) == np.count_nonzero(required)


def test_factorized_source_manifest_v4_uses_the_strict_region_contract(
    tmp_path: Path,
) -> None:
    shape = (12, 16)
    source = np.full((*shape, 3), 180, np.uint8)
    target = np.zeros(shape, np.uint8)
    target[4:8, 5:9] = 255
    full = np.full(shape, 255, np.uint8)
    zero = np.zeros(shape, np.uint8)
    paths = {
        "source": _write_image(tmp_path / "source.png", source),
        "target": _write_image(tmp_path / "target.png", target),
        "full": _write_image(tmp_path / "full.png", full),
        "zero": _write_image(tmp_path / "zero.png", zero),
    }
    payload = {
        "schema_version": "inpaint-factorized-source-manifest-v4",
        "pages": [{
            "page_id": "p",
            "path": paths["source"],
            "target_text_mask": paths["target"],
            "preserve_mask": paths["zero"],
            "target_instances": [{
                "instance_id": "glyph",
                "region_id": "unowned",
                "mask_path": paths["target"],
                "semantic_role": "dialogue_free",
                "processing_action": "translate_inpaint",
                "priority": "required",
            }],
            "regions": [{
                "region_id": "unowned",
                "bubble_route_class": "ambiguous",
                "bubble_interior_mask": paths["zero"],
                "ownership_mask": paths["zero"],
                "protected_structure_mask": paths["zero"],
                "ambiguous_structure_mask": paths["zero"],
                "corner_protect_mask": paths["zero"],
            }],
            "protected_structure_mask": paths["zero"],
            "ambiguous_structure_mask": paths["zero"],
            "ownership_mask": paths["zero"],
            "claim_seed_mask": paths["target"],
            "bubble_interior_mask": paths["zero"],
            "corner_protect_mask": paths["zero"],
            "existing_source_edit_mask": paths["zero"],
            "expected_edit": "required",
        }],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    page = load_stage1_manifest(path)[0]

    assert page.target_instances[0].region_id == "unowned"
    assert page.regions[0].region_id == "unowned"


def test_independent_unowned_region_ownership_is_exact_target_union(
    tmp_path: Path,
) -> None:
    shape = (10, 12)
    page_dir = tmp_path / "page"
    exact = np.zeros(shape, np.uint8)
    exact[3:6, 4:8] = 255

    region = _empty_region(
        page_dir,
        shape,
        "region-unowned-review",
        ownership=exact,
    )
    ownership = cv2.imread(region["ownership_mask"], cv2.IMREAD_GRAYSCALE)
    interior = cv2.imread(region["bubble_interior_mask"], cv2.IMREAD_GRAYSCALE)

    assert np.array_equal(ownership, exact)
    assert np.count_nonzero(interior) == 0


def test_independent_review_missing_optional_mask_is_exact_zero() -> None:
    assert np.array_equal(
        read_independent_review_mask(None, (7, 9)),
        np.zeros((7, 9), np.uint8),
    )
    with pytest.raises(FileNotFoundError):
        read_independent_review_mask(None)


def test_independent_manifest_seal_binds_manifest_and_review_decisions(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    decisions = tmp_path / "decisions.json"
    manifest.write_text(
        json.dumps({
            "schema_version": "inpaint-factorized-source-manifest-v4",
            "corpus_id": "example",
            "instance_counts": {"required": 1, "preserve": 2, "ambiguous": 0},
            "annotation_frozen_before_candidate": True,
            "candidate_seen": False,
            "target_extent_independent": True,
            "target_inventory_independent": True,
            "target_review_complete": True,
        }),
        encoding="utf-8",
    )
    decisions.write_text(json.dumps({
        "schema_version": "inpaint-independent-target-review-decisions-v4",
        "candidate_seen": False,
        "review_complete": True,
    }), encoding="utf-8")

    seal = seal_independent_manifest(manifest, decisions)

    assert seal["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert seal["review_decisions_sha256"] == hashlib.sha256(
        decisions.read_bytes()
    ).hexdigest()
    assert seal["candidate_seen"] is False
    assert json.loads(
        manifest.with_suffix(".json.seal.json").read_text(encoding="utf-8")
    ) == seal


def test_independent_manifest_seal_rejects_candidate_seen_input(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.json"
    decisions = tmp_path / "decisions.json"
    manifest.write_text(json.dumps({
        "schema_version": "inpaint-factorized-source-manifest-v4",
        "annotation_frozen_before_candidate": True,
        "candidate_seen": True,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
    }), encoding="utf-8")
    decisions.write_text(json.dumps({
        "schema_version": "inpaint-independent-target-review-decisions-v4",
        "candidate_seen": False,
        "review_complete": True,
    }), encoding="utf-8")

    with pytest.raises(ValueError, match="not source-only review complete"):
        seal_independent_manifest(manifest, decisions)


def test_record_review_accepts_complete_inventory_from_reviewed_rows(
    tmp_path: Path,
) -> None:
    ledger = {
        "candidate_seen": False,
        "review_complete": False,
        "rows": [],
        "full_page_inventory_pending": [{"page_id": "p"}],
    }
    decisions = {
        "candidate_seen": False,
        "decisions": [],
        "full_page_inventory": [{
            "page_id": "p",
            "status": "complete_with_reviewed_rows",
        }],
    }
    ledger_path = tmp_path / "ledger.json"
    decisions_path = tmp_path / "decisions.json"
    output_path = tmp_path / "output.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    decisions_path.write_text(json.dumps(decisions), encoding="utf-8")

    result = record_independent_target_review(
        ledger_path,
        decisions_path,
        output_path,
    )

    assert result["full_page_inventory"] == [{
        "page_id": "p",
        "status": "complete_with_reviewed_rows",
        "manual_inventory_path": "",
    }]


def test_review_decisions_use_reviewed_source_priority_as_semantic_default(
    tmp_path: Path,
) -> None:
    ledger = {
        "candidate_seen": False,
        "rows": [
            {
                "review_id": "review-0000",
                "inventory_source": "source_manifest_location_aid_only",
                "source_priority_proposal": "required",
            },
            {
                "review_id": "review-0001",
                "inventory_source": "source_manifest_location_aid_only",
                "source_priority_proposal": "optional",
            },
        ],
        "review_sheets": ["sheet.jpg"],
        "full_page_inventory_pending": [],
    }
    overrides = {
        "schema_version": "inpaint-independent-review-overrides-v4",
        "candidate_seen": False,
        "reviewed_sheets": [1],
        "rows_per_sheet": 8,
        "row_overrides": {},
        "full_page_inventory": [],
    }
    ledger_path = tmp_path / "ledger.json"
    overrides_path = tmp_path / "overrides.json"
    output_path = tmp_path / "decisions.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    overrides_path.write_text(json.dumps(overrides), encoding="utf-8")

    result = build_review_decisions(ledger_path, overrides_path, output_path)

    assert result["decisions"] == [
        {"review_id": "review-0000", "extent": "balanced", "semantic": "required"},
        {"review_id": "review-0001", "extent": "balanced", "semantic": "preserve"},
    ]


def test_extend_source_location_review_overrides_preserves_prior_work(
    tmp_path: Path,
) -> None:
    ledger = {
        "candidate_seen": False,
        "rows": [
            {
                "review_id": "review-0000",
                "inventory_source": "paired_location_with_human_semantic_region",
            },
            {
                "review_id": "review-0001",
                "inventory_source": "source_manifest_location_aid_only",
                "source_priority_proposal": "required",
            },
        ],
        "review_sheets": ["one.jpg", "two.jpg"],
        "full_page_inventory_pending": [{"page_id": "p"}],
    }
    base = {
        "schema_version": "inpaint-independent-review-overrides-v4",
        "candidate_seen": False,
        "reviewed_sheets": [1],
        "rows_per_sheet": 1,
        "known_default_extent": "balanced",
        "row_overrides": {
            "review-0000": {"extent": "strict", "semantic": "required"}
        },
        "full_page_inventory": [
            {"page_id": "p", "status": "complete_no_required_text"}
        ],
    }
    ledger_path = tmp_path / "ledger.json"
    base_path = tmp_path / "base.json"
    output_path = tmp_path / "effective.json"
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    base_path.write_text(json.dumps(base), encoding="utf-8")

    result = extend_source_location_review_overrides(
        ledger_path,
        base_path,
        output_path,
        reviewed_sheet_through=2,
        source_location_default_extent="strict",
        reject_review_ids=("review-0001",),
    )

    assert result["reviewed_sheets"] == [1, 2]
    assert result["source_location_default_extent"] == "strict"
    assert result["row_overrides"]["review-0000"] == {
        "extent": "strict",
        "semantic": "required",
    }
    assert result["row_overrides"]["review-0001"] == {
        "extent": "reject",
        "semantic": "not_text",
    }
    assert result["full_page_inventory"] == [
        {"page_id": "p", "status": "complete_with_reviewed_rows"}
    ]


def test_independent_ambiguous_review_becomes_hard_preserve_mask(
    tmp_path: Path,
) -> None:
    shape = (16, 20)
    source = _write_image(tmp_path / "source.png", np.full((*shape, 3), 180, np.uint8))
    empty = _write_image(tmp_path / "empty.png", np.zeros(shape, np.uint8))
    ownership = _write_image(tmp_path / "ownership.png", np.full(shape, 255, np.uint8))
    location_mask = np.zeros(shape, np.uint8)
    location_mask[5:9, 7:12] = 255
    location = _write_image(tmp_path / "location.png", location_mask)
    semantic = tmp_path / "semantic.json"
    semantic.write_text(json.dumps({
        "schema_version": "inpaint-factorized-source-manifest-v4",
        "corpus_id": "example",
        "pages": [{
            "page_id": "p",
            "path": source,
            "height": shape[0],
            "width": shape[1],
            "target_text_mask": None,
            "preserve_mask": empty,
            "protected_structure_mask": empty,
            "ambiguous_structure_mask": empty,
            "ownership_mask": ownership,
            "expected_edit": "none",
            "target_instances": [],
            "regions": [{
                "region_id": "r",
                "bubble_route_class": "ambiguous",
                "bubble_interior_mask": empty,
                "ownership_mask": ownership,
                "protected_structure_mask": empty,
                "ambiguous_structure_mask": empty,
                "corner_protect_mask": empty,
            }],
        }],
    }), encoding="utf-8")
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps({
        "schema_version": "inpaint-independent-target-review-ledger-v4",
        "candidate_seen": False,
        "rows": [{
            "review_id": "review-0000",
            "page_id": "p",
            "region_id": "r",
            "location_seed": location,
            "extent_variants": {},
        }],
        "full_page_inventory_pending": [],
    }), encoding="utf-8")
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({
        "schema_version": "inpaint-independent-target-review-decisions-v4",
        "candidate_seen": False,
        "review_complete": True,
        "decisions": [{
            "review_id": "review-0000",
            "extent": "reject",
            "semantic": "ambiguous",
        }],
        "full_page_inventory": [],
    }), encoding="utf-8")

    _bind_independent_review_fixture(semantic, ledger, decisions)
    payload = apply_independent_target_review(
        semantic, ledger, decisions, tmp_path / "output"
    )

    page = payload["pages"][0]
    ambiguous = cv2.imread(page["ambiguous_structure_mask"], cv2.IMREAD_GRAYSCALE)
    assert np.array_equal(ambiguous, location_mask)
    assert page["target_instances"][0]["priority"] == "ambiguous"


def test_manifest_v4_rejects_optional_text_as_edit_target(tmp_path: Path) -> None:
    shape = (12, 16)
    mask = np.zeros(shape, np.uint8)
    mask[4:8, 4:8] = 255
    full = np.full(shape, 255, np.uint8)
    source = np.full((*shape, 3), 180, np.uint8)
    paths = {
        "source": _write_image(tmp_path / "source.png", source),
        "mask": _write_image(tmp_path / "mask.png", mask),
        "full": _write_image(tmp_path / "full.png", full),
        "zero": _write_image(tmp_path / "zero.png", np.zeros(shape, np.uint8)),
    }
    payload = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v4",
        "pages": [{
            "page_id": "p",
            "path": paths["source"],
            "target_text_mask": None,
            "preserve_mask": paths["mask"],
            "target_instances": [{
                "instance_id": "sfx",
                "region_id": "r",
                "mask_path": paths["mask"],
                "semantic_role": "sfx",
                "processing_action": "translate_inpaint",
                "priority": "optional",
            }],
            "regions": [{
                "region_id": "r",
                "bubble_route_class": "ambiguous",
                "bubble_interior_mask": paths["full"],
                "ownership_mask": paths["full"],
                "protected_structure_mask": paths["zero"],
                "ambiguous_structure_mask": paths["zero"],
                "corner_protect_mask": paths["zero"],
            }],
            "expected_edit": "none",
        }],
    }
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="optional target instance must use preserve"):
        load_stage1_manifest(path)


def test_manifest_v4_builder_rejects_candidate_seen_decisions(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    assert cv2.imwrite(str(source), np.full((8, 8, 3), 180, np.uint8))
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"pages": [{"page_id": "p", "path": str(source)}]}),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-decisions-v4",
                "candidate_seen": True,
                "pages": [{"page_id": "p"}],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="before viewing candidates"):
        build_manifest_v4(source_manifest, decisions)


def test_manifest_v4_builder_rejects_non_proposal_paired_reference(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.png"
    reference = tmp_path / "reference.png"
    assert cv2.imwrite(str(source), np.full((8, 8, 3), 180, np.uint8))
    assert cv2.imwrite(str(reference), np.full((8, 8, 3), 200, np.uint8))
    zero = tmp_path / "zero.png"
    full = tmp_path / "full.png"
    assert cv2.imwrite(str(zero), np.zeros((8, 8), np.uint8))
    assert cv2.imwrite(str(full), np.full((8, 8), 255, np.uint8))
    source_manifest = tmp_path / "source.json"
    source_manifest.write_text(
        json.dumps({"pages": [{"page_id": "p", "path": str(source)}]}),
        encoding="utf-8",
    )
    decisions = tmp_path / "decisions.json"
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-decisions-v4",
                "candidate_seen": False,
                "pages": [{
                    "page_id": "p",
                    "target_text_mask": None,
                    "preserve_mask": str(zero),
                    "target_instances": [],
                    "regions": [{
                        "region_id": "r",
                        "bubble_route_class": "ambiguous",
                        "bubble_interior_mask": str(full),
                        "ownership_mask": str(full),
                        "protected_structure_mask": str(zero),
                        "ambiguous_structure_mask": str(zero),
                        "corner_protect_mask": str(zero),
                    }],
                    "protected_structure_mask": str(zero),
                    "ambiguous_structure_mask": str(zero),
                    "ownership_mask": str(full),
                    "bubble_interior_mask": str(full),
                    "corner_protect_mask": str(zero),
                    "expected_edit": "none",
                    "paired_reference": {
                        "path": str(reference),
                        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
                        "proposal_only": False,
                    },
                }],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="proposal_only"):
        build_manifest_v4(source_manifest, decisions)


def test_mixed_page_scores_broad_edit_only_outside_clean_regions() -> None:
    shape = (24, 48)
    zeros = np.zeros(shape, np.uint8)
    left = np.zeros(shape, np.uint8)
    left[:, :24] = 255
    right = np.zeros(shape, np.uint8)
    right[:, 24:] = 255
    masks = PageMasks(
        target=zeros.copy(),
        protected=zeros.copy(),
        ambiguous=zeros.copy(),
        ownership=np.full(shape, 255, np.uint8),
        claim_seed=np.full(shape, 255, np.uint8),
        existing_edit=zeros.copy(),
        regions=(
            RegionMasks("clean", "clean_flat", left, left, zeros, zeros, zeros),
            RegionMasks("texture", "texture", right, right, zeros, zeros, zeros),
        ),
    )
    broad = np.zeros(shape, np.uint8)
    broad[4:20, 4:20] = 255
    broad[4:20, 28:44] = 255

    assert broad_route_false_positive_pixels(broad & left, masks) == 0
    assert broad_route_false_positive_pixels(broad, masks) == 256
    sparse_masks = replace(masks, regions=())
    assert (
        broad_route_false_positive_pixels(
            broad, sparse_masks, clean_region_mask=left
        )
        == 256
    )


def test_factorized_runner_uses_canonical_broad_route_scorer() -> None:
    source = (
        ROOT / "scripts" / "benchmark_inpaint_factorized_v3.py"
    ).read_text(encoding="utf-8")

    assert "clean_region_mask=streamed_clean_ownership" in source
    assert "source_clean[region.bubble_interior > 0]" not in source


def test_runner_cache_path_preserves_v4_region_and_preserve_masks(
    tmp_path: Path,
) -> None:
    shape = (16, 24)
    source = np.full((*shape, 3), 180, np.uint8)
    zeros = np.zeros(shape, np.uint8)
    preserve = np.zeros(shape, np.uint8)
    preserve[3:7, 4:8] = 255
    ownership = np.full(shape, 255, np.uint8)
    interior = np.zeros(shape, np.uint8)
    interior[1:15, 1:23] = 255
    source_path = _write_image(tmp_path / "source.png", source)
    zeros_path = _write_image(tmp_path / "zeros.png", zeros)
    preserve_path = _write_image(tmp_path / "preserve.png", preserve)
    ownership_path = _write_image(tmp_path / "ownership.png", ownership)
    interior_path = _write_image(tmp_path / "interior.png", interior)
    page = Stage1Page(
        page_id="page",
        source_image=source_path,
        target_text_mask=zeros_path,
        protected_structure_mask=zeros_path,
        ambiguous_structure_mask=zeros_path,
        ownership_mask=ownership_path,
        claim_seed_mask=ownership_path,
        bubble_interior_mask=interior_path,
        corner_protect_mask=zeros_path,
        preserve_mask=preserve_path,
        regions=(
            RegionEvaluationSpec(
                "region",
                "clean_flat",
                interior_path,
                ownership_path,
                zeros_path,
                zeros_path,
                zeros_path,
            ),
        ),
    )

    masks = _annotation_masks(
        page,
        {"existing_source_edit_mask": zeros_path},
        shape,
        {},
    )
    replaced = _with_candidate_ownership(
        masks,
        ownership,
        ownership,
        interior,
    )

    assert np.array_equal(masks.preserve, preserve)
    assert masks.regions[0].region_id == "region"
    assert np.array_equal(replaced.preserve, preserve)
    assert replaced.regions == masks.regions


def test_instance_seed_recall_is_separate_from_final_edit_coverage() -> None:
    shape = (20, 28)
    first = np.zeros(shape, np.uint8)
    second = np.zeros(shape, np.uint8)
    first[4:8, 4:8] = 255
    second[12:16, 18:22] = 255
    target = cv2.bitwise_or(first, second)
    claim = np.zeros(shape, np.uint8)
    claim[4, 4] = 255
    claim[12, 18] = 255
    protected = np.zeros(shape, np.uint8)
    protected[12, 18] = 255
    masks = PageMasks(
        target,
        protected,
        np.zeros(shape, np.uint8),
        np.full(shape, 255, np.uint8),
        np.full(shape, 255, np.uint8),
        np.zeros(shape, np.uint8),
        (("first", first), ("second", second)),
    )
    result = CandidateMaskResult("candidate", claim, claim, claim)
    page = Stage1Page("page", "source", None, None, None)

    record, _edit = score_page(page, result, masks, variant="raw")

    assert record["target_instance_seed_recall"] == 1.0
    assert record["missed_target_instance_ids"] == []
    assert record["minimum_target_instance_edit_coverage"] == 0.0


def test_role_candidate_cache_roundtrip_and_provenance_mismatch(tmp_path: Path) -> None:
    spec = RoleCandidateSpec(
        candidate_id="ctd-raw",
        provider="ballons-ctd",
        role="seed",
        variant="raw",
        code_commit="abc123",
        model_sha256="11" * 32,
        runtime_provider="cuda",
        preprocessing_contract_sha256="22" * 32,
    )
    raw = np.zeros((12, 16), np.uint8)
    raw[3:8, 4:9] = 255
    stage = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = CandidateMaskResult(
        "ctd-raw",
        raw,
        raw,
        raw,
        boxes=(DetectorBox((4, 3, 9, 8), "text", 0.9, "reference"),),
        stage_tensors={"preprocess": stage},
        runtime={"seconds": 1.25, "device": "cuda"},
    )
    source_sha = "33" * 32

    entry = write_detector_cache(tmp_path, spec, source_sha, result)
    loaded = read_detector_cache(tmp_path, spec, source_sha)

    assert loaded is not None
    assert np.array_equal(loaded.raw_mask, raw)
    assert np.array_equal(loaded.stage_tensors["preprocess"], stage)
    assert loaded.boxes == result.boxes
    assert loaded.runtime["seconds"] == 1.25
    assert loaded.runtime["device"] == "cuda"
    assert loaded.runtime["cache_hit"] is True
    metadata_path = entry / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["cache_payload"]["model_sha256"] = "44" * 32
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="provenance mismatch"):
        read_detector_cache(tmp_path, spec, source_sha)


def test_cached_page_inference_requires_composite_input_sha(tmp_path: Path) -> None:
    spec = RoleCandidateSpec(
        candidate_id="ctd-ownership-roi",
        provider="ballons-ctd",
        role="seed",
        variant="raw",
        code_commit="abc123",
        model_sha256="11" * 32,
        runtime_provider="cuda",
        preprocessing_contract_sha256="22" * 32,
    )

    with pytest.raises(ValueError, match="composite cache input SHA"):
        run_stage1(
            (),
            lambda _image: None,
            variant="raw",
            candidate_spec=spec,
            cache_root=tmp_path,
            page_infer=lambda _page, _image: None,
        )


def test_expansion_requires_seed_and_component_contact() -> None:
    shape = (32, 40)
    raw = np.zeros(shape, np.uint8)
    raw[8:12, 8:12] = 255
    components = np.zeros(shape, np.uint8)
    components[6:15, 6:15] = 255
    components[20:28, 28:36] = 255
    empty = np.zeros(shape, np.uint8)

    no_seed = expand_detector_claim(
        "content_component",
        seed=empty,
        raw=raw,
        refined=raw,
        dilated=raw,
        content_components=components,
    )
    selected = expand_detector_claim(
        "content_component",
        seed=raw,
        raw=raw,
        refined=raw,
        dilated=raw,
        content_components=components,
    )

    assert np.count_nonzero(no_seed) == 0
    assert np.count_nonzero(selected[6:15, 6:15]) == 81
    assert np.count_nonzero(selected[20:28, 28:36]) == 0


@pytest.mark.parametrize("radius", [1, 2, 3, 4])
def test_lab_dilation_expands_only_from_detector_seed(radius: int) -> None:
    shape = (24, 24)
    seed = np.zeros(shape, np.uint8)
    seed[12, 12] = 255

    expanded = expand_detector_claim(
        f"lab_dilate{radius}",
        seed=seed,
        raw=seed,
        refined=seed,
        dilated=seed,
    )

    assert np.count_nonzero(expanded) == (radius * 2 + 1) ** 2
    assert expanded[12 - radius, 12 - radius] == 255
    assert expanded[11 - radius, 12] == 0


def test_broad_route_requires_every_clean_bubble_condition() -> None:
    masks = _page_masks()
    seed = np.zeros(masks.target.shape, np.uint8)
    seed[20:22, 28:30] = 255
    broad = np.zeros_like(seed)
    broad[14:30, 20:40] = 255

    accepted = decide_bubble_route(
        "R3",
        narrow_claim=seed,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        ballons_clean=True,
        pr2_clean=True,
        segmentation_valid=True,
        background_sample_count=128,
    )
    texture = decide_bubble_route(
        "R3",
        narrow_claim=seed,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        ballons_clean=True,
        pr2_clean=True,
        texture=True,
        background_sample_count=128,
    )

    assert accepted.decision == "broad"
    assert np.array_equal(accepted.edit_mask, broad)
    assert texture.decision == "narrow"
    assert np.array_equal(texture.edit_mask, seed)
    assert "texture_present" in texture.reasons


def test_broad_route_fails_to_narrow_on_protection_or_sparse_samples() -> None:
    masks = _page_masks()
    seed = np.zeros(masks.target.shape, np.uint8)
    seed[20:22, 28:30] = 255
    broad = np.zeros_like(seed)
    broad[14:30, 20:40] = 255
    masks.protected[16:18, 22:24] = 255

    decision = decide_bubble_route(
        "R2",
        narrow_claim=seed,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        pr2_clean=True,
        background_sample_count=2,
    )

    assert decision.decision == "narrow"
    assert "broad_overlaps_structure_protect" in decision.reasons
    assert "insufficient_roi_background_samples" in decision.reasons
    assert np.count_nonzero(decision.edit_mask & masks.protected) == 0


def test_router_applies_broad_mask_only_inside_clean_block_patch() -> None:
    masks = _page_masks((40, 80))
    seed = np.zeros((40, 80), np.uint8)
    seed[18:22, 12:16] = 255
    seed[18:22, 58:62] = 255
    broad = np.zeros_like(seed)
    broad[8:32, 4:36] = 255
    broad[8:32, 44:76] = 255
    clean = np.zeros_like(seed)
    clean[8:32, 4:36] = 255
    unsafe = np.zeros_like(seed)
    unsafe[8:32, 44:76] = 255

    decision = decide_bubble_route(
        "R1",
        narrow_claim=seed,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        background_sample_count=128,
        ballons_clean_mask=clean,
        unsafe_signal_mask=unsafe,
    )

    assert decision.decision == "broad"
    assert np.count_nonzero(decision.edit_mask[8:32, 4:36]) > 0
    assert np.count_nonzero(decision.edit_mask[8:32, 44:76]) == 16


def test_router_uses_pixel_ownership_for_narrow_and_region_ownership_for_broad() -> None:
    shape = (40, 60)
    seed = np.zeros(shape, np.uint8)
    seed[18:22, 28:32] = 255
    false_claim = np.zeros(shape, np.uint8)
    false_claim[4:8, 4:8] = 255
    narrow = cv2.bitwise_or(seed, false_claim)
    broad = np.zeros(shape, np.uint8)
    broad[8:32, 16:44] = 255
    masks = PageMasks(
        target=seed.copy(),
        protected=np.zeros(shape, np.uint8),
        ambiguous=np.zeros(shape, np.uint8),
        ownership=seed.copy(),
        claim_seed=seed.copy(),
        existing_edit=np.zeros(shape, np.uint8),
        bubble_interior=broad.copy(),
        corner=np.zeros(shape, np.uint8),
        broad_ownership=broad.copy(),
    )

    decision = decide_bubble_route(
        "R1",
        narrow_claim=narrow,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        ballons_clean=True,
        background_sample_count=128,
    )

    assert decision.decision == "broad"
    assert np.array_equal(decision.edit_mask, broad)
    assert np.count_nonzero(decision.edit_mask[4:8, 4:8]) == 0


def test_router_reopens_only_detector_seed_from_runtime_baseline_mask() -> None:
    shape = (40, 60)
    seed = np.zeros(shape, np.uint8)
    seed[18:22, 28:32] = 255
    broad = np.zeros(shape, np.uint8)
    broad[8:32, 16:44] = 255
    existing = broad.copy()
    masks = PageMasks(
        target=seed.copy(),
        protected=np.zeros(shape, np.uint8),
        ambiguous=np.zeros(shape, np.uint8),
        ownership=seed.copy(),
        claim_seed=seed.copy(),
        existing_edit=existing,
        bubble_interior=broad.copy(),
        corner=np.zeros(shape, np.uint8),
        broad_ownership=broad.copy(),
    )

    decision = decide_bubble_route(
        "R1",
        narrow_claim=seed,
        broad_claim=broad,
        seed=seed,
        masks=masks,
        ballons_clean=True,
        background_sample_count=128,
    )

    assert decision.decision == "broad"
    assert np.array_equal(decision.edit_mask, seed)


@pytest.mark.parametrize(
    "backend",
    ["robust_flat_median", "planar_gradient", "telea"],
)
def test_non_lama_fill_backends_change_only_exact_mask(backend: str) -> None:
    y, x = np.mgrid[0:64, 0:80]
    source = np.stack((120 + x // 8, 130 + y // 8, 140 + (x + y) // 16), axis=-1).astype(np.uint8)
    edit = np.zeros(source.shape[:2], np.uint8)
    edit[28:36, 36:44] = 255
    source[edit > 0] = 20
    interior = np.zeros_like(edit)
    interior[8:56, 12:68] = 255

    candidate, diagnostics = fill_factorized_mask(
        source,
        edit,
        backend=backend,
        interior_mask=interior,
    )

    assert diagnostics["applied"] is True
    assert np.array_equal(candidate[edit == 0], source[edit == 0])
    assert np.any(candidate[edit > 0] != source[edit > 0])


def test_page_union_uses_one_lama_call_and_exact_composite() -> None:
    source = np.full((48, 64, 3), 180, np.uint8)
    edit = np.zeros(source.shape[:2], np.uint8)
    edit[8:12, 8:12] = 255
    edit[30:35, 45:50] = 255
    calls: list[int] = []

    def fake_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        calls.append(int(np.count_nonzero(mask)))
        generated = image.copy()
        generated[mask > 0] = 77
        generated[0, 0] = 1
        return generated

    candidate, diagnostics = fill_factorized_mask(
        source,
        edit,
        backend="current_lama",
        lama_fill=fake_lama,
    )

    assert calls == [41]
    assert diagnostics["applied"] is True
    assert np.all(candidate[edit > 0] == 77)
    assert np.array_equal(candidate[edit == 0], source[edit == 0])


def test_broad_fill_samples_clean_interior_but_excludes_detector_seed() -> None:
    source = np.full((48, 64, 3), 230, np.uint8)
    interior = np.zeros(source.shape[:2], np.uint8)
    interior[8:40, 10:54] = 255
    seed = np.zeros_like(interior)
    seed[20:28, 28:36] = 255
    source[seed > 0] = 20

    candidate, diagnostics = fill_factorized_mask(
        source,
        interior,
        backend="robust_flat_median",
        interior_mask=interior,
        background_sample_edit_mask=seed,
    )

    assert diagnostics["applied"] is True
    assert diagnostics["sample_pixel_count"] == int(
        np.count_nonzero((interior > 0) & (seed == 0))
    )
    assert np.all(candidate[interior > 0] == 230)
    assert np.array_equal(candidate[interior == 0], source[interior == 0])


def test_broad_fill_rejects_sample_exclusion_outside_edit() -> None:
    source = np.full((16, 20, 3), 180, np.uint8)
    edit = np.zeros(source.shape[:2], np.uint8)
    edit[4:12, 6:14] = 255
    invalid = edit.copy()
    invalid[0, 0] = 255

    with pytest.raises(ValueError, match="inside the edit mask"):
        fill_factorized_mask(
            source,
            edit,
            backend="robust_flat_median",
            background_sample_edit_mask=invalid,
        )


def test_route_hybrid_uses_lama_only_for_narrow_and_flat_for_broad() -> None:
    assert _route_fill_backend("narrow_lama_broad_flat", "narrow") == "current_lama"
    assert (
        _route_fill_backend("narrow_lama_broad_flat", "broad")
        == "robust_flat_median"
    )
    assert _route_fill_backend("telea", "broad") == "telea"
    assert (
        _route_fill_backend("conditional_hybrid", "broad", "clean_flat")
        == "robust_flat_median"
    )
    assert (
        _route_fill_backend("conditional_hybrid", "broad", "clean_gradient")
        == "planar_gradient"
    )
    assert (
        _route_fill_backend("conditional_hybrid", "broad", "texture")
        == "current_lama"
    )
    assert (
        _route_fill_backend("conditional_hybrid", "narrow", "clean_flat")
        == "current_lama"
    )
    assert (
        _route_fill_backend(
            "conditional_refill_existing", "narrow", "clean_flat"
        )
        == "robust_flat_median"
    )
    assert (
        _route_fill_backend(
            "conditional_refill_existing", "narrow", "clean_gradient"
        )
        == "planar_gradient"
    )
    assert (
        _route_fill_backend("conditional_refill_existing", "narrow", "line_art")
        == "current_lama"
    )


def test_oracle_background_reconstruction_scores_fill_independently() -> None:
    truth = np.full((40, 48, 3), 230, np.uint8)
    source = truth.copy()
    source[16:24, 20:28] = 20
    edit = np.zeros(source.shape[:2], np.uint8)
    edit[16:24, 20:28] = 255

    candidate, _diagnostics = fill_factorized_mask(
        source,
        edit,
        backend="robust_flat_median",
    )

    assert reconstruction_error(candidate, truth, edit) == 0.0


def test_matrix_generates_every_logical_cartesian_combination() -> None:
    axes = {
        "detector": ("d0", "d1", "d2"),
        "router": ("r0", "r1"),
        "fill": ("f0", "f1"),
    }
    controls = {"detector": "d0", "router": "r0", "fill": "f0"}

    records = build_factorized_matrix(axes, controls)

    assert len(records) == 12
    assert controls in records
    assert any(
        row["detector"] != "d0" and row["router"] != "r0" and row["fill"] != "f0"
        for row in records
    )


def test_closure_ledger_accounts_for_executed_invalid_and_blocked() -> None:
    selections = [
        {
            "detector": "full",
            "ownership": "o",
            "silhouette": "s",
            "router": "R1",
            "expansion": "raw",
            "fill": "mask_only",
        },
        {
            "detector": "full",
            "ownership": "o",
            "silhouette": "s",
            "router": "R0",
            "expansion": "bubble_interior",
            "fill": "mask_only",
        },
        {
            "detector": "blocked",
            "ownership": "o",
            "silhouette": "s",
            "router": "R1",
            "expansion": "raw",
            "fill": "mask_only",
        },
    ]
    ledger = build_combination_closure_ledger(
        selections,
        stage="stage1",
        family_metadata={
            "detector": {
                "blocked": {"asset_status": "blocked"},
            }
        },
    )

    assert_complete_closure_ledger(selections, ledger)
    assert [record.closure_state for record in ledger] == [
        "executed",
        "invalid_with_reason",
        "blocked_asset",
    ]
    assert ledger[1].reason == "broad_expansion_requires_broad_router"
    assert ledger[2].reason == "provider_asset_or_parity_missing"


@pytest.mark.parametrize(
    ("selection", "stage", "reason"),
    (
        (
            {
                "detector": "full",
                "ownership": "o",
                "silhouette": "empty",
                "router": "R1",
                "expansion": "raw",
                "fill": "telea",
            },
            "product",
            "bubble_fill_requires_silhouette",
        ),
        (
            {
                "detector": "full",
                "ownership": "o",
                "silhouette": "s",
                "router": "R1",
                "roi_trigger": "seed_missing",
                "expansion": "raw",
                "fill": "mask_only",
            },
            "stage1",
            "roi_trigger_requires_roi_detector",
        ),
        (
            {
                "detector": "full",
                "ownership": "o",
                "silhouette": "s",
                "router": "R1",
                "runtime_detector_count": "3",
                "expansion": "raw",
                "fill": "mask_only",
            },
            "stage1",
            "runtime_detector_limit_exceeded",
        ),
        (
            {
                "detector": "oracle",
                "ownership": "o",
                "silhouette": "s",
                "router": "R1",
                "expansion": "raw",
                "fill": "current_lama",
            },
            "product",
            "oracle_product_candidate",
        ),
        (
            {
                "detector": "full",
                "ownership": "o",
                "silhouette": "s",
                "router": "R1",
                "expansion": "raw",
                "fill": "telea",
            },
            "stage1",
            "stage1_fill_backend_forbidden",
        ),
    ),
)
def test_every_static_invalid_combination_reason_is_represented(
    selection: dict[str, str],
    stage: str,
    reason: str,
) -> None:
    ledger = build_combination_closure_ledger(
        [selection],
        stage=stage,
        oracle_only_ids=("oracle",),
    )

    assert ledger[0].closure_state == "invalid_with_reason"
    assert ledger[0].reason == reason


def test_broad_expansion_without_source_seed_is_invalid() -> None:
    selection = {
        "detector": "empty_detector",
        "ownership": "o",
        "silhouette": "s",
        "router": "R4",
        "expansion": "content_component",
        "fill": "mask_only",
    }

    ledger = build_combination_closure_ledger(
        [selection],
        stage="stage1",
        family_metadata={
            "detector": {"empty_detector": {"source_seed_available": False}}
        },
    )

    assert ledger[0].closure_state == "invalid_with_reason"
    assert ledger[0].reason == "broad_expansion_requires_source_seed"


def test_closure_ledger_rejects_unaccounted_combination() -> None:
    selections = [
        {"detector": "d0", "router": "R0"},
        {"detector": "d1", "router": "R0"},
    ]
    ledger = build_combination_closure_ledger(
        selections[:1],
        stage="stage1",
    )

    with pytest.raises(ValueError, match="closure ledger mismatch"):
        assert_complete_closure_ledger(selections, ledger)


def test_physical_matrix_reuses_content_identical_logical_families(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "mask.png"
    artifact.write_bytes(b"same-artifact")
    combinations = [
        {
            "detector": detector,
            "ownership": "o",
            "silhouette": "s",
            "router": "R0",
            "expansion": "raw",
            "fill": "mask_only",
        }
        for detector in ("d0", "d1")
    ]
    detector = {
        "seed_variant": "raw",
        "pages": {"p": {"raw": str(artifact), "refined": str(artifact), "dilated": str(artifact)}},
    }
    matrix = {
        "families": {
            "detector": {"d0": detector, "d1": detector},
            "ownership": {"o": {"pages": {"p": {"mask": str(artifact)}}}},
            "silhouette": {"s": {"pages": {"p": {"interior": str(artifact)}}}},
            "router": {"R0": {"algorithm": "R0", "pages": {"p": {}}}},
        },
        "oracle_only": [],
    }

    ledger, physical = _prepare_closure_ledger(
        combinations,
        matrix=matrix,
        manifest_sha256="11" * 32,
    )

    assert len(physical) == 1
    assert [row.closure_state for row in ledger] == ["executed", "reused_by_sha"]
    assert ledger[1].reused_from == ledger[0].logical_id
    assert ledger[1].content_sha256 == ledger[0].content_sha256


def test_explicit_matrix_can_add_one_compatible_multi_role_run() -> None:
    axes = {
        "detector": ["d0", "d1"],
        "ownership": ["o0", "o1"],
        "silhouette": ["s0", "s1"],
        "router": ["r0", "r1"],
        "expansion": ["raw", "bubble_interior"],
        "fill": ["mask_only"],
    }
    controls = {
        "detector": "d0",
        "ownership": "o0",
        "silhouette": "s0",
        "router": "r0",
        "expansion": "raw",
        "fill": "mask_only",
    }
    matrix = {
        "factorized": False,
        "explicit_combinations": [
            {
                "detector": "d1",
                "ownership": "o1",
                "silhouette": "s1",
                "router": "r1",
                "expansion": "bubble_interior",
            }
        ],
    }

    runs = _declared_combinations(matrix, axes, controls)

    assert len(runs) == 2
    assert runs[1]["detector"] == "d1"
    assert runs[1]["expansion"] == "bubble_interior"


def _complete_hard_gate_metrics(**updates: object) -> dict[str, object]:
    metrics: dict[str, object] = {
        "page_count": 1,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "required_target_instance_count": 1,
        "target_pixel_count": 10,
        "aggregate_target_coverage": 1.0,
        "minimum_target_instance_coverage": 1.0,
        "target_instance_seed_recall": 1.0,
        "target_instance_seed_recall_by_semantic_role": {"dialogue_bubble": 1.0},
        "aggregate_residue_score": 0.1,
        "baseline_aggregate_residue_score": 0.2,
        "residue_gate_applicable": True,
        "reconstruction_gate_applicable": False,
        "reconstruction_mse": 1.0,
        "runtime_seconds": 1.0,
        "runtime_telemetry_complete": True,
        "positive_lama_inference_count": 0,
        "maximum_positive_lama_inference_per_page": 0,
        "cpu_fallback_count": 0,
        "positive_lama_runtime_p95_seconds": None,
        "peak_vram_allocated_mib": None,
        "peak_vram_reserved_mib": None,
        "lama_runtime_provider": "",
        "lama_runtime_precision": "",
        "protected_structure_overlap": 0,
        "protected_structure_changed": 0,
        "ambiguous_structure_overlap": 0,
        "ambiguous_structure_changed": 0,
        "outside_final_changed": 0,
        "broad_route_false_positive": 0,
        "no_edit_false_edit": 0,
        "required_skip_count": 0,
        "preserve_edit_overlap": 0,
        "ownership_leak_pixel_count": 0,
        "corner_edit_overlap_pixel_count": 0,
        "missed_target_instance_count": 0,
        "page_residue_worsened_count": 0,
    }
    metrics.update(updates)
    return metrics


def test_pareto_selection_excludes_oracle_and_hard_gate_failure() -> None:
    safe = _complete_hard_gate_metrics()
    records = [
        FactorizedRunRecord("safe", "d", "o", "s", "r", "e", "f", False, "active", safe),
        FactorizedRunRecord("oracle", "d", "o", "s", "r", "e", "f", True, "active", {**safe, "aggregate_residue_score": 0.0}),
        FactorizedRunRecord("unsafe", "d", "o", "s", "r", "e", "f", False, "active", {**safe, "protected_structure_overlap": 1}),
    ]

    selected = {record.run_id: record for record in select_pareto_records(records)}

    assert selected["safe"].status == "pareto"
    assert selected["oracle"].status == "family_complete"
    assert selected["unsafe"].status == "dominated"
    assert selected["unsafe"].closure_reason == "hard_gate_failed"


def test_pareto_selection_marks_detector_derived_targets_information_limited() -> None:
    metrics = _complete_hard_gate_metrics(
        target_extent_independent=False,
        target_mask_provenance=["current_ctd_raw_components"],
        residue_gate_applicable=False,
    )
    record = FactorizedRunRecord(
        "circular", "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "information_limited"
    assert selected[0].closure_reason == "target_extent_not_independent"


def test_pareto_selection_requires_independent_target_inventory() -> None:
    metrics = _complete_hard_gate_metrics(
        target_inventory_independent=False,
        target_mask_provenance=["paired_source_extent"],
        residue_gate_applicable=False,
    )
    record = FactorizedRunRecord(
        "inventory-circular", "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "information_limited"
    assert selected[0].closure_reason == "target_inventory_not_independent"


def test_pareto_selection_requires_completed_source_review() -> None:
    metrics = _complete_hard_gate_metrics(
        target_review_complete=False,
        residue_gate_applicable=False,
    )
    record = FactorizedRunRecord(
        "unreviewed", "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "information_limited"
    assert selected[0].closure_reason == "target_review_incomplete"


def test_pareto_gate_rejects_missing_residue_and_ambiguous_changes() -> None:
    safe = _complete_hard_gate_metrics()
    records = [
        FactorizedRunRecord(
            "safe", "d", "o", "s", "r", "e", "f", False, "active", safe
        ),
        FactorizedRunRecord(
            "missing-residue",
            "d",
            "o",
            "s",
            "r",
            "e",
            "f",
            False,
            "active",
            {**safe, "aggregate_residue_score": None},
        ),
        FactorizedRunRecord(
            "ambiguous-change",
            "d",
            "o",
            "s",
            "r",
            "e",
            "f",
            False,
            "active",
            {**safe, "ambiguous_structure_changed": 1},
        ),
    ]

    selected = {record.run_id: record for record in select_pareto_records(records)}

    assert selected["safe"].status == "pareto"
    assert selected["missing-residue"].status == "dominated"
    assert selected["missing-residue"].closure_reason == "hard_gate_failed"
    assert selected["ambiguous-change"].status == "dominated"
    assert selected["ambiguous-change"].closure_reason == "hard_gate_failed"


@pytest.mark.parametrize(
    "missing",
    (
        "protected_structure_overlap",
        "aggregate_target_coverage",
        "target_instance_seed_recall",
        "runtime_telemetry_complete",
    ),
)
def test_pareto_gate_fails_closed_when_required_metric_is_missing(
    missing: str,
) -> None:
    metrics = _complete_hard_gate_metrics()
    metrics.pop(missing)
    record = FactorizedRunRecord(
        missing, "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "dominated"
    assert selected[0].closure_reason == "hard_gate_failed"


def test_reconstruction_gate_requires_narrow_control_and_rejects_worse_fill() -> None:
    control = FactorizedRunRecord(
        "control",
        "d",
        "o",
        "s",
        "r",
        "raw",
        "current_lama",
        False,
        "active",
        _complete_hard_gate_metrics(
            reconstruction_gate_applicable=True,
            reconstruction_mse=2.0,
        ),
    )
    worse = FactorizedRunRecord(
        "worse",
        "d",
        "o",
        "s",
        "r",
        "broad",
        "telea",
        False,
        "active",
        _complete_hard_gate_metrics(
            reconstruction_gate_applicable=True,
            reconstruction_mse=3.0,
        ),
    )

    selected = {
        row.run_id: row
        for row in select_pareto_records(
            attach_reconstruction_control([control, worse], "control")
        )
    }

    assert selected["control"].status == "pareto"
    assert selected["worse"].status == "dominated"
    assert selected["worse"].closure_reason == "hard_gate_failed"


def test_4k_component_expansion_has_no_python_quadratic_loop() -> None:
    shape = (4096, 4096)
    components = np.zeros(shape, np.uint8)
    for y in range(32, 4064, 128):
        for x in range(32, 4064, 128):
            components[y : y + 3, x : x + 3] = 255
    seed = np.zeros(shape, np.uint8)
    seed[32:35, 32:35] = 255
    started = time.perf_counter()

    selected = expand_detector_claim(
        "content_component",
        seed=seed,
        raw=seed,
        refined=seed,
        dilated=seed,
        content_components=components,
    )

    elapsed = time.perf_counter() - started
    assert np.count_nonzero(selected) == 9
    assert elapsed < 5.0


def test_factorized_runner_executes_declared_control_matrix(tmp_path: Path) -> None:
    shape = (48, 64)
    truth = np.full((*shape, 3), 230, np.uint8)
    source = truth.copy()
    source[20:28, 28:36] = 20
    target = np.zeros(shape, np.uint8)
    target[20:28, 28:36] = 255
    zeros = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    source_path = _write_image(tmp_path / "source.png", source)
    truth_path = _write_image(tmp_path / "truth.png", truth)
    target_path = _write_image(tmp_path / "target.png", target)
    zeros_path = _write_image(tmp_path / "zeros.png", zeros)
    full_path = _write_image(tmp_path / "full.png", full)
    manifest = {
        "schema_version": "inpaint-detector-bakeoff-manifest-v3",
        "pages": [
            {
                "page_id": "p1",
                "path": source_path,
                "target_text_mask": target_path,
                "target_instances": [
                    {"instance_id": "glyph", "mask_path": target_path}
                ],
                "bubble_route_class": "clean_flat",
                "bubble_interior_mask": full_path,
                "protected_structure_mask": zeros_path,
                "ambiguous_structure_mask": zeros_path,
                "ownership_mask": full_path,
                "corner_protect_mask": zeros_path,
                "expected_edit": "required",
                "baseline": source_path,
                "baseline_mask": zeros_path,
                "known_background": truth_path,
                "target_mask_provenance": "synthetic_ground_truth",
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
            }
        ],
    }
    matrix = {
        "schema_version": "inpaint-factorized-matrix-v3",
        "axes": {
            "detector": ["detector"],
            "ownership": ["ownership"],
            "silhouette": ["silhouette"],
            "router": ["R0"],
            "expansion": ["raw"],
            "fill": ["robust_flat_median"],
        },
        "controls": {
            "detector": "detector",
            "ownership": "ownership",
            "silhouette": "silhouette",
            "router": "R0",
            "expansion": "raw",
            "fill": "robust_flat_median",
        },
        "families": {
            "detector": {
                "detector": {
                    "seed_variant": "raw",
                    "pages": {
                        "p1": {
                            "raw": target_path,
                            "refined": target_path,
                            "dilated": target_path,
                        }
                    },
                }
            },
            "ownership": {
                "ownership": {
                    "pages": {
                        "p1": {
                            "mask": full_path,
                            "content_components": target_path,
                        }
                    }
                }
            },
            "silhouette": {
                "silhouette": {"pages": {"p1": {"interior": full_path}}}
            },
            "router": {"R0": {"algorithm": "R0", "pages": {"p1": {}}}},
        },
        "retain_page_artifacts": True,
        "oracle_only": [],
        "reconstruction_control_run_id": (
            "detector__ownership__silhouette__R0__raw__robust_flat_median"
        ),
    }
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    output = tmp_path / "output"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")
    _write_strict_source_manifest(manifest_path, manifest)

    exit_code = factorized_main(
        [
            "--manifest",
            str(manifest_path),
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(output),
            "--device",
            "cpu",
        ]
    )

    result = json.loads((output / "factorized-results.json").read_text(encoding="utf-8"))
    assert exit_code == 0
    assert result["combination_count"] == 1
    assert len(result["logical_inventory_sha256"]) == 64
    assert result["runs"][0]["status"] == "pareto"
    assert result["runs"][0]["metrics"]["aggregate_target_coverage"] == 1.0
    assert result["runs"][0]["metrics"][
        "target_instance_seed_recall_by_semantic_role"
    ] == {"dialogue_bubble": 1.0}
    assert result["runs"][0]["metrics"]["reconstruction_mse"] == 0.0
    assert result["runs"][0]["metrics"]["outside_final_changed"] == 0
    run_id = result["runs"][0]["run_id"]
    assert result["pages"][run_id][0]["canonical_statistics"][
        "schema_version"
    ] == "inpaint-factorized-page-statistics-v1"
    assert len(result["runs"][0]["metrics"]["output_mask_set_sha256"]) == 64
    assert result["output_artifact_inventory"]["artifact_count"] == 5
    inventory = json.loads(
        (output / "factorized-output-artifact-inventory.json").read_text(
            encoding="utf-8"
        )
    )
    assert inventory["schema_version"] == (
        "inpaint-factorized-output-artifact-inventory-v2"
    )
    assert {record["role"] for record in inventory["records"]} == {
        "detector_seed_mask",
        "effective_ownership_mask",
        "edit_mask",
        "final_mask",
        "candidate_image",
    }
    assert result["output_artifact_inventory"]["complete_run_ids"] == [run_id]
    runtime_binding = result["runtime_evidence_ledger"]
    assert runtime_binding["role"] == "runtime_evidence"
    assert runtime_binding["complete_run_ids"] == [run_id]
    runtime_ledger = json.loads(
        (output / runtime_binding["relative_path"]).read_text(encoding="utf-8")
    )
    assert runtime_ledger["runs"][0]["pages"][0]["inference_events"] == []
    assert runtime_ledger["runs"][0]["aggregate"][
        "positive_lama_inference_count"
    ] == 0


def test_manifest_builder_uses_only_source_manifest_and_frozen_decisions(
    tmp_path: Path,
) -> None:
    source_manifest = tmp_path / "source.json"
    decisions = tmp_path / "decisions.json"
    source_manifest.write_text(
        json.dumps(
            {
                "corpus_id": "a1",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": "source.png",
                        "baseline": "baseline.png",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    decisions.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-source-decisions-v3",
                "corpus_id": "a1",
                "pages": [
                    {
                        "page_id": "p1",
                        "target_text_mask": "target.png",
                        "target_instances": [
                            {"instance_id": "glyph", "mask_path": "target.png"}
                        ],
                        "bubble_route_class": "ambiguous",
                        "bubble_interior_mask": None,
                        "protected_structure_mask": "protected.png",
                        "ambiguous_structure_mask": "ambiguous.png",
                        "ownership_mask": "ownership.png",
                        "corner_protect_mask": "corner.png",
                        "expected_edit": "required",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = build_manifest(source_manifest, decisions)

    assert result["schema_version"] == "inpaint-detector-bakeoff-manifest-v3"
    assert result["annotation_frozen_before_candidate"] is True
    assert result["pages"][0]["annotation_basis"] == "source_only_v3"
    assert result["pages"][0]["baseline"] == "baseline.png"


def test_manifest_builder_attaches_sealed_baseline_by_matching_source(
    tmp_path: Path,
) -> None:
    source_path = _write_image(
        tmp_path / "source.png", np.full((8, 10, 3), 150, np.uint8)
    )
    baseline_path = _write_image(
        tmp_path / "baseline.png", np.full((8, 10, 3), 180, np.uint8)
    )
    baseline_mask = _write_image(
        tmp_path / "baseline-mask.png", np.zeros((8, 10), np.uint8)
    )
    source_manifest = tmp_path / "source.json"
    decisions = tmp_path / "decisions.json"
    baseline_manifest = tmp_path / "baseline.json"
    source_manifest.write_text(
        json.dumps({"pages": [{"page_id": "p1", "path": source_path}]}),
        encoding="utf-8",
    )
    decisions.write_text(
        json.dumps({
            "schema_version": "inpaint-factorized-source-decisions-v3",
            "pages": [{
                "page_id": "p1",
                "target_text_mask": None,
                "target_instances": [],
                "bubble_route_class": "ambiguous",
                "bubble_interior_mask": None,
                "protected_structure_mask": None,
                "ambiguous_structure_mask": None,
                "ownership_mask": None,
                "corner_protect_mask": None,
                "expected_edit": "none",
            }],
        }),
        encoding="utf-8",
    )
    baseline_manifest.write_text(
        json.dumps({"pages": [{
            "page_id": "p1",
            "path": source_path,
            "baseline": {"path": baseline_path},
            "baseline_mask": {"path": baseline_mask},
        }]}),
        encoding="utf-8",
    )

    result = build_manifest(source_manifest, decisions, baseline_manifest)

    assert result["pages"][0]["baseline"] == baseline_path
    assert result["pages"][0]["baseline_mask"] == baseline_mask
    assert result["pages"][0]["existing_source_edit_mask"] == baseline_mask
    assert result["baseline_manifest_sha256"]


def test_factorized_matrix_builder_resolves_control_and_family_artifacts(
    tmp_path: Path,
) -> None:
    shape = (16, 20)
    source = _write_image(tmp_path / "source.png", np.full((*shape, 3), 150, np.uint8))
    paths: dict[str, str] = {}
    for name in (
        "target",
        "protected",
        "ambiguous",
        "ownership",
        "seed",
        "interior",
        "corner",
        "existing",
        "candidate",
        "candidate_owner",
    ):
        mask = np.zeros(shape, np.uint8)
        if name == "target":
            mask[4:7, 5:8] = 255
        paths[name] = _write_image(tmp_path / f"{name}.png", mask)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-bakeoff-manifest-v3",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": source,
                        "target_text_mask": paths["target"],
                        "target_instances": [
                            {"instance_id": "i1", "mask_path": paths["target"]}
                        ],
                        "protected_structure_mask": paths["protected"],
                        "ambiguous_structure_mask": paths["ambiguous"],
                        "ownership_mask": paths["ownership"],
                        "claim_seed_mask": paths["seed"],
                        "bubble_interior_mask": paths["interior"],
                        "corner_protect_mask": paths["corner"],
                        "existing_source_edit_mask": paths["existing"],
                        "bubble_route_class": "clean_flat",
                        "expected_edit": "required",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-family-registry-v3",
                "families": {
                    "detector": {
                        "candidate_detector": {
                            "provider": "test",
                            "seed_variant": "raw",
                            "templates": {"raw": paths["candidate"]},
                        }
                    },
                    "ownership": {
                        "candidate_ownership": {
                            "provider": "test",
                            "templates": {"mask": paths["candidate_owner"]},
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix(manifest, registry)

    detector_page = matrix["families"]["detector"]["candidate_detector"]["pages"]["p1"]
    ownership_page = matrix["families"]["ownership"]["candidate_ownership"]["pages"]["p1"]
    assert matrix["controls"]["detector"] == "control_source_edit"
    assert detector_page["raw"] == str(Path(paths["candidate"]).resolve())
    assert detector_page["refined"] == detector_page["raw"]
    assert ownership_page["content_components"] == str(
        Path(paths["candidate_owner"]).resolve()
    )
    assert matrix["controls"]["silhouette"] == "control_empty_silhouette"
    assert "control_dual_ownership" in matrix["families"]["ownership"]
    assert "annotation_interior_oracle" in matrix["oracle_only"]


def test_matrix_builder_expands_only_declared_compatible_sets(tmp_path: Path) -> None:
    shape = (8, 10)
    image_path = _write_image(tmp_path / "source.png", np.full((*shape, 3), 150, np.uint8))
    zero_path = _write_image(tmp_path / "zero.png", np.zeros(shape, np.uint8))
    target = np.zeros(shape, np.uint8)
    target[2:4, 3:5] = 255
    target_path = _write_image(tmp_path / "target.png", target)
    full_path = _write_image(tmp_path / "full.png", np.full(shape, 255, np.uint8))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-bakeoff-manifest-v3",
                "pages": [{
                    "page_id": "p1",
                    "path": image_path,
                    "target_text_mask": target_path,
                    "target_instances": [{"instance_id": "i1", "mask_path": target_path}],
                    "protected_structure_mask": zero_path,
                    "ambiguous_structure_mask": zero_path,
                    "ownership_mask": full_path,
                    "claim_seed_mask": full_path,
                    "bubble_interior_mask": full_path,
                    "corner_protect_mask": zero_path,
                    "existing_source_edit_mask": zero_path,
                    "bubble_route_class": "clean_flat",
                    "expected_edit": "required",
                }],
            }
        ),
        encoding="utf-8",
    )
    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-family-registry-v3",
                "settings": {"factorized": False},
                "axes": {
                    "expansion": ["raw", "bubble_interior"],
                    "fill": ["mask_only"],
                },
                "families": {
                    "detector": {"d1": {"templates": {"raw": target_path}}},
                    "ownership": {"o1": {"templates": {"mask": full_path}}},
                    "silhouette": {"s1": {"templates": {"interior": full_path}}},
                    "router": {"r1": {"algorithm": "R1"}, "r2": {"algorithm": "R2"}},
                },
                "compatible_sets": [{
                    "base": {"detector": "d1", "ownership": "o1", "silhouette": "s1"},
                    "vary": {
                        "router": ["r1", "r2"],
                        "expansion": ["raw", "bubble_interior"],
                    },
                }],
            }
        ),
        encoding="utf-8",
    )

    matrix = build_matrix(manifest, registry)

    assert matrix["factorized"] is False
    assert len(matrix["explicit_combinations"]) == 4
    assert all(row["detector"] == "d1" for row in matrix["explicit_combinations"])


def test_ballons_and_pr2_silhouettes_preserve_seed_in_clean_bubble() -> None:
    image = np.full((80, 96, 3), 75, np.uint8)
    cv2.ellipse(image, (48, 40), (34, 26), 0, 0, 360, (20, 20, 20), 3)
    cv2.ellipse(image, (48, 40), (31, 23), 0, 0, 360, (245, 245, 245), -1)
    seed = np.zeros(image.shape[:2], np.uint8)
    seed[32:48, 43:53] = 255

    native = extract_ballons_native_interior(image, seed)
    validated = extract_pr2_validated_interior(image, seed)

    assert native is not None
    assert validated is not None
    assert np.count_nonzero((seed > 0) & (native == 0)) == 0
    assert np.count_nonzero((seed > 0) & (validated == 0)) <= 3
    assert ballons_native_clean_background(image, seed) is True


def test_silhouette_export_uses_only_seed_contacting_blocks(tmp_path: Path) -> None:
    image = np.full((80, 96, 3), 75, np.uint8)
    cv2.ellipse(image, (48, 40), (34, 26), 0, 0, 360, (20, 20, 20), 3)
    cv2.ellipse(image, (48, 40), (31, 23), 0, 0, 360, (245, 245, 245), -1)
    seed = np.zeros(image.shape[:2], np.uint8)
    seed[32:48, 43:53] = 255
    zero = np.zeros(image.shape[:2], np.uint8)
    full = np.full(image.shape[:2], 255, np.uint8)
    source_path = _write_image(tmp_path / "source.png", image)
    seed_path = _write_image(tmp_path / "p1_seed.png", seed)
    zero_path = _write_image(tmp_path / "zero.png", zero)
    full_path = _write_image(tmp_path / "full.png", full)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "inpaint-detector-bakeoff-manifest-v3",
                "pages": [
                    {
                        "page_id": "p1",
                        "path": source_path,
                        "target_text_mask": seed_path,
                        "target_instances": [
                            {"instance_id": "i1", "mask_path": seed_path}
                        ],
                        "bubble_route_class": "clean_flat",
                        "bubble_interior_mask": full_path,
                        "protected_structure_mask": zero_path,
                        "ambiguous_structure_mask": zero_path,
                        "ownership_mask": full_path,
                        "corner_protect_mask": zero_path,
                        "expected_edit": "required",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "p1_blocks.json").write_text(
        json.dumps(
            {
                "blocks": [
                    {
                        "bubble_xyxy": [10, 10, 86, 70],
                        "erase_mode": "bubble_flat_fill",
                        "erase_skipped_reason": "",
                    },
                    {
                        "bubble_xyxy": [0, 0, 8, 8],
                        "erase_mode": "bubble_flat_fill",
                        "erase_skipped_reason": "",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "p1_ctbd.json").write_text(
        json.dumps(
            {
                "boxes": [
                    {"label": "bubble", "xyxy": [10, 10, 86, 70]},
                    {"label": "bubble", "xyxy": [0, 0, 8, 8]},
                ]
            }
        ),
        encoding="utf-8",
    )

    result = export_candidates(
        manifest_path,
        block_metadata_template=str(tmp_path / "{page_id}_blocks.json"),
        seed_template=str(tmp_path / "{page_id}_seed.png"),
        ctbd_metadata_template=str(tmp_path / "{page_id}_ctbd.json"),
        output_root=tmp_path / "out",
    )

    assert result["index"][0]["selected_block_count"] == 1
    assert result["pages"]["p1"]["ballons_clean"] is True
    assert result["pages"]["p1"]["pr2_clean"] is True
    ctbd = cv2.imread(str(tmp_path / "out" / "ctbd_bubble" / "p1.png"), 0)
    assert np.count_nonzero(ctbd[0:8, 0:8]) == 0
    assert np.count_nonzero(ctbd[10:70, 10:86]) > 0


def test_contact_sheet_requires_exact_four_review_roles(tmp_path: Path) -> None:
    image = np.full((40, 60, 3), 180, np.uint8)
    image_path = _write_image(tmp_path / "image.png", image)
    mask = np.zeros(image.shape[:2], np.uint8)
    mask[10:20, 20:30] = 255
    mask_path = _write_image(tmp_path / "mask.png", mask)
    spec = {
        "cell_width": 120,
        "cell_height": 80,
        "rows": [
            {
                "kind": kind,
                "label": kind,
                "source": image_path,
                "control": image_path,
                "candidate_1": image_path,
                "candidate_2": image_path,
                "edit_mask": mask_path,
                "protect_mask": mask_path,
                "crop_xyxy": [0, 0, 60, 40],
            }
            for kind in ("small_text", "clean_bubble", "halftone", "line_adjacent")
        ],
    }

    sheet = build_contact_sheet(spec)

    assert sheet.width == 600
    assert sheet.height == 34 + 4 * 108


def test_isolated_factorized_results_merge_and_recompute_pareto(
    tmp_path: Path,
) -> None:
    def write_result(name: str, run_id: str, runtime: float) -> Path:
        metrics = _complete_hard_gate_metrics(
            runtime_seconds=runtime,
            residue_gate_applicable=False,
        )
        path = tmp_path / name / "factorized-results.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "schema_version": "inpaint-factorized-results-v3",
                    "manifest_sha256": "manifest",
                    "matrix_sha256": "matrix",
                    "positive_lama_inference_count": 0,
                    "runs": [{
                        "run_id": run_id,
                        "detector_id": run_id,
                        "ownership_id": "o",
                        "silhouette_id": "s",
                        "router_id": "r",
                        "expansion_id": "raw",
                        "fill_id": "mask_only",
                        "oracle_only": False,
                        "status": "pareto",
                        "metrics": metrics,
                        "closure_reason": "",
                    }],
                    "pages": {run_id: [{"page_id": "p1"}]},
                }
            ),
            encoding="utf-8",
        )
        return path

    merged = merge_results(
        [write_result("one", "fast", 1.0), write_result("two", "slow", 2.0)]
    )

    statuses = {row["run_id"]: row["status"] for row in merged["runs"]}
    assert statuses == {"fast": "pareto", "slow": "dominated"}
    assert set(merged["pages"]) == {"fast", "slow"}


def test_isolated_factorized_results_require_every_executed_closure_row(
    tmp_path: Path,
) -> None:
    ledger = [
        {
            "logical_id": run_id,
            "selection": {"detector": run_id},
            "closure_state": "executed",
            "reason": "",
            "content_sha256": run_id,
            "reused_from": "",
        }
        for run_id in ("one", "two")
    ]

    def write_result(name: str, run_id: str) -> Path:
        path = tmp_path / name / "factorized-results.json"
        path.parent.mkdir()
        metrics = _complete_hard_gate_metrics(residue_gate_applicable=False)
        path.write_text(
            json.dumps(
                {
                    "schema_version": "inpaint-factorized-results-v3",
                    "manifest_sha256": "manifest",
                    "matrix_sha256": "matrix",
                    "logical_combination_count": 2,
                    "physical_combination_count": 2,
                    "closure_ledger": ledger,
                    "positive_lama_inference_count": 0,
                    "runs": [
                        {
                            "run_id": run_id,
                            "detector_id": run_id,
                            "ownership_id": "o",
                            "silhouette_id": "s",
                            "router_id": "r",
                            "expansion_id": "raw",
                            "fill_id": "mask_only",
                            "oracle_only": False,
                            "status": "active",
                            "metrics": metrics,
                            "closure_reason": "",
                        }
                    ],
                    "pages": {run_id: [{"page_id": "p1"}]},
                }
            ),
            encoding="utf-8",
        )
        return path

    one = write_result("one", "one")
    two = write_result("two", "two")

    with pytest.raises(ValueError, match="missing=\\['two'\\]"):
        merge_results([one])

    merged = merge_results([one, two])
    assert merged["logical_combination_count"] == 2
    assert merged["physical_combination_count"] == 2
    assert merged["unaccounted_combination_count"] == 0
    assert merged["closure_ledger"] == ledger
    assert len(merged["logical_inventory_sha256"]) == 64


def test_isolated_factorized_results_merge_local_closure_runtime_diagnostics(
    tmp_path: Path,
) -> None:
    ledger = [
        {
            "logical_id": run_id,
            "selection": {"detector": run_id},
            "closure_state": "executed",
            "reason": "",
            "content_sha256": run_id,
            "reused_from": "",
        }
        for run_id in ("one", "two")
    ]

    def write_result(name: str, run_id: str, conflict_count: int) -> Path:
        local_ledger = [dict(row) for row in ledger]
        local_ledger[[row["logical_id"] for row in ledger].index(run_id)][
            "runtime_diagnostics"
        ] = {"conditional_hybrid_overlap_conflict_pixel_count": conflict_count}
        path = tmp_path / name / "factorized-results.json"
        path.parent.mkdir()
        path.write_text(
            json.dumps(
                {
                    "schema_version": "inpaint-factorized-results-v3",
                    "manifest_sha256": "manifest",
                    "matrix_sha256": "matrix",
                    "logical_combination_count": 2,
                    "physical_combination_count": 2,
                    "closure_ledger": local_ledger,
                    "positive_lama_inference_count": 0,
                    "runs": [
                        {
                            "run_id": run_id,
                            "detector_id": run_id,
                            "ownership_id": "o",
                            "silhouette_id": "s",
                            "router_id": "r",
                            "expansion_id": "raw",
                            "fill_id": "mask_only",
                            "oracle_only": False,
                            "status": "active",
                            "metrics": _complete_hard_gate_metrics(
                                residue_gate_applicable=False
                            ),
                            "closure_reason": "",
                        }
                    ],
                    "pages": {run_id: [{"page_id": "p1"}]},
                }
            ),
            encoding="utf-8",
        )
        return path

    merged = merge_results(
        [write_result("one", "one", 3), write_result("two", "two", 5)]
    )

    diagnostics = {
        row["logical_id"]: row.get("runtime_diagnostics")
        for row in merged["closure_ledger"]
    }
    assert diagnostics == {
        "one": {"conditional_hybrid_overlap_conflict_pixel_count": 3},
        "two": {"conditional_hybrid_overlap_conflict_pixel_count": 5},
    }


def test_silhouette_consensus_builds_union_intersection_and_n_of_four() -> None:
    masks = {name: np.zeros((6, 8), np.uint8) for name in (
        "ballons", "pr2", "ctbd", "manga109"
    )}
    masks["ballons"][1:4, 1:4] = 255
    masks["pr2"][2:5, 2:5] = 255
    masks["ctbd"][2:4, 3:6] = 255
    masks["manga109"][3:5, 2:6] = 255

    products = consensus_masks(masks)

    expected_union = (masks["ballons"] > 0) | (masks["pr2"] > 0)
    expected_intersection = (masks["ballons"] > 0) & (masks["pr2"] > 0)
    stack = np.stack([mask > 0 for mask in masks.values()], axis=0)
    assert np.array_equal(products["ballons_pr2_union"] > 0, expected_union)
    assert np.array_equal(
        products["ballons_pr2_intersection"] > 0, expected_intersection
    )
    assert np.array_equal(
        products["two_of_four_consensus"] > 0,
        np.count_nonzero(stack, axis=0) >= 2,
    )
    assert np.array_equal(
        products["three_of_four_consensus"] > 0,
        np.count_nonzero(stack, axis=0) >= 3,
    )


def test_known_background_synthetic_manifest_covers_all_fill_routes(
    tmp_path: Path,
) -> None:
    manifest = build_synthetic_manifest(tmp_path)

    assert manifest["schema_version"] == "inpaint-factorized-source-manifest-v4"
    assert manifest["split_role"] == "synthetic_known_ground_truth"
    assert manifest["candidate_seen"] is False
    assert len(manifest["pages"]) == 10
    assert {page["bubble_route_class"] for page in manifest["pages"]} == {
        "clean_flat",
        "clean_gradient",
        "texture",
        "line_art",
        "ambiguous",
    }
    for page in manifest["pages"]:
        assert page["source_sha256"] == page["artifact_sha256"]["path"]
        assert len(page["regions"]) == 1
        source = cv2.imread(page["path"], cv2.IMREAD_COLOR)
        truth = cv2.imread(page["known_background"], cv2.IMREAD_COLOR)
        target = cv2.imread(page["target_text_mask"], cv2.IMREAD_GRAYSCALE)
        protected = cv2.imread(
            page["protected_structure_mask"], cv2.IMREAD_GRAYSCALE
        )
        assert np.count_nonzero(target) > 0
        assert np.count_nonzero((target > 0) & (protected > 0)) == 0
        assert np.count_nonzero(np.any(source != truth, axis=2) & (target > 0)) > 0


def test_fill_oracle_matrix_keeps_unsafe_routes_narrow(tmp_path: Path) -> None:
    manifest = build_synthetic_manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    matrix = build_fill_matrix(manifest_path)

    assert matrix["oracle_experiment"] is True
    assert len(matrix["explicit_combinations"]) == 13
    assert {
        row["fill"] for row in matrix["explicit_combinations"]
    } == {
        "robust_flat_median",
        "planar_gradient",
        "telea",
        "current_lama",
        "ballons_lama",
        "conditional_refill_existing",
        "conditional_hybrid",
    }
    pages = matrix["families"]["router"]["clean_route_oracle"]["pages"]
    clean = pages["synthetic-clean_flat"]
    unsafe = pages["synthetic-halftone"]
    assert clean["ballons_clean_mask"].endswith("interior.png")
    assert unsafe["ballons_clean_mask"].endswith("zero.png")

    ledger, physical = _prepare_closure_ledger(
        matrix["explicit_combinations"],
        matrix=matrix,
        manifest_sha256="f" * 64,
    )
    assert len(physical) == 13
    assert {row.closure_state for row in ledger} == {"executed"}


def test_conditional_hybrid_uses_authoritative_mixed_region_routes() -> None:
    height, width = 80, 120
    truth = np.zeros((height, width, 3), np.uint8)
    truth[:, :60] = (210, 210, 210)
    yy, xx = np.indices((height, 60))
    gradient = np.clip(120 + xx + yy // 4, 0, 255).astype(np.uint8)
    truth[:, 60:, 0] = gradient
    truth[:, 60:, 1] = np.clip(gradient + 10, 0, 255)
    truth[:, 60:, 2] = np.clip(gradient + 20, 0, 255)
    edit = np.zeros((height, width), np.uint8)
    edit[30:40, 20:34] = 255
    edit[32:43, 80:96] = 255
    source = truth.copy()
    source[edit > 0] = 15
    zero = np.zeros((height, width), np.uint8)
    left = np.zeros_like(zero)
    left[:, :60] = 255
    right = np.zeros_like(zero)
    right[:, 60:] = 255
    masks = PageMasks(
        target=edit,
        protected=zero,
        ambiguous=zero,
        ownership=np.full_like(zero, 255),
        claim_seed=np.full_like(zero, 255),
        existing_edit=zero,
        bubble_interior=np.full_like(zero, 255),
        corner=zero,
        broad_ownership=np.full_like(zero, 255),
        preserve=zero,
        regions=(
            RegionMasks("flat", "clean_flat", left, left, zero, zero, zero),
            RegionMasks(
                "gradient", "clean_gradient", right, right, zero, zero, zero
            ),
        ),
    )

    def unexpected_lama(_image: np.ndarray, _mask: np.ndarray) -> np.ndarray:
        raise AssertionError("clean mixed regions must not use the LaMa fallback")

    candidate, diagnostics = _fill_conditional_hybrid_regions(
        source,
        edit,
        masks,
        route_decision="broad",
        background_exclude_mask=zero,
        lama_fill=unexpected_lama,
    )

    assert diagnostics["positive_lama_inference_count"] == 0
    assert [row["backend"] for row in diagnostics["region_fills"]] == [
        "robust_flat_median",
        "planar_gradient",
    ]
    assert all(row["applied"] is True for row in diagnostics["region_fills"])
    assert np.array_equal(candidate[edit == 0], source[edit == 0])
    assert np.max(np.abs(candidate[edit > 0].astype(int) - truth[edit > 0])) <= 1

    refill, refill_diagnostics = _fill_conditional_hybrid_regions(
        source,
        edit,
        masks,
        route_decision="narrow",
        background_exclude_mask=zero,
        lama_fill=unexpected_lama,
        fill_policy="conditional_refill_existing",
    )
    assert refill_diagnostics["backend"] == "conditional_refill_existing"
    assert refill_diagnostics["positive_lama_inference_count"] == 0
    assert [row["backend"] for row in refill_diagnostics["region_fills"]] == [
        "robust_flat_median",
        "planar_gradient",
    ]
    assert np.array_equal(refill[edit == 0], source[edit == 0])
    assert np.max(np.abs(refill[edit > 0].astype(int) - truth[edit > 0])) <= 1


def test_conditional_hybrid_broad_region_samples_exclude_only_narrow_seed() -> None:
    shape = (48, 64)
    source = np.full((*shape, 3), 230, np.uint8)
    interior = np.zeros(shape, np.uint8)
    interior[8:40, 10:54] = 255
    seed = np.zeros(shape, np.uint8)
    seed[20:28, 28:36] = 255
    source[seed > 0] = 20
    zero = np.zeros(shape, np.uint8)
    masks = PageMasks(
        target=seed,
        protected=zero,
        ambiguous=zero,
        ownership=seed,
        claim_seed=seed,
        existing_edit=zero,
        bubble_interior=interior,
        corner=zero,
        broad_ownership=interior,
        preserve=zero,
        regions=(
            RegionMasks("flat", "clean_flat", interior, seed, zero, zero, zero),
        ),
    )

    candidate, diagnostics = _fill_conditional_hybrid_regions(
        source,
        interior,
        masks,
        route_decision="broad",
        background_exclude_mask=zero,
        lama_fill=lambda _image, _mask: (_ for _ in ()).throw(
            AssertionError("clean broad region must not use LaMa")
        ),
        narrow_claim=seed,
    )

    detail = diagnostics["region_fills"][0]
    assert detail["applied"] is True
    assert diagnostics["positive_lama_inference_count"] == 0
    assert detail["edit_pixel_count"] == int(np.count_nonzero(interior))
    assert detail["sample_exclusion_pixel_count"] == int(np.count_nonzero(seed))
    assert np.all(candidate[interior > 0] == 230)


def test_conditional_hybrid_routes_authoritative_overlap_to_one_narrow_lama_call() -> None:
    shape = (48, 64)
    source = np.full((*shape, 3), 220, np.uint8)
    edit = np.zeros(shape, np.uint8)
    edit[20:28, 28:36] = 255
    source[edit > 0] = 20
    zero = np.zeros(shape, np.uint8)
    first = np.zeros(shape, np.uint8)
    first[8:40, 8:40] = 255
    second = np.zeros(shape, np.uint8)
    second[8:40, 24:56] = 255
    ownership = cv2.bitwise_or(first, second)
    masks = PageMasks(
        target=edit,
        protected=zero,
        ambiguous=zero,
        ownership=ownership,
        claim_seed=edit,
        existing_edit=zero,
        bubble_interior=ownership,
        corner=zero,
        broad_ownership=ownership,
        preserve=zero,
        regions=(
            RegionMasks("left", "clean_flat", first, first, zero, zero, zero),
            RegionMasks("right", "clean_gradient", second, second, zero, zero, zero),
        ),
    )
    calls: list[np.ndarray] = []

    def fake_lama(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        calls.append(mask.copy())
        candidate = image.copy()
        candidate[mask > 0] = 220
        return candidate

    candidate, diagnostics = _fill_conditional_hybrid_regions(
        source,
        edit,
        masks,
        route_decision="broad",
        background_exclude_mask=zero,
        lama_fill=fake_lama,
        narrow_claim=edit,
    )

    assert len(calls) == 1
    assert np.array_equal(calls[0], edit)
    assert diagnostics["positive_lama_inference_count"] == 1
    assert diagnostics["authoritative_region_overlap_pixel_count"] > 0
    assert diagnostics["authoritative_overlap_edit_pixel_count"] == int(
        np.count_nonzero(edit)
    )
    assert diagnostics["authoritative_overlap_narrow_verified"] is True
    fallback = diagnostics["region_fills"][-1]
    assert fallback["fallback_scope"] == "narrow_page_level"
    assert fallback["fallback_reasons"] == ["authoritative_region_overlap"]
    assert np.array_equal(candidate[edit == 0], source[edit == 0])


def test_conditional_hybrid_rejects_broad_overlap_outside_narrow_claim() -> None:
    shape = (16, 20)
    source = np.full((*shape, 3), 200, np.uint8)
    edit = np.zeros(shape, np.uint8)
    edit[4:12, 6:14] = 255
    narrow = np.zeros(shape, np.uint8)
    narrow[6:10, 8:12] = 255
    zero = np.zeros(shape, np.uint8)
    full = np.full(shape, 255, np.uint8)
    masks = PageMasks(
        target=narrow,
        protected=zero,
        ambiguous=zero,
        ownership=full,
        claim_seed=narrow,
        existing_edit=zero,
        bubble_interior=full,
        corner=zero,
        broad_ownership=full,
        preserve=zero,
        regions=(
            RegionMasks("a", "clean_flat", full, full, zero, zero, zero),
            RegionMasks("b", "clean_flat", full, full, zero, zero, zero),
        ),
    )

    with pytest.raises(AssertionError, match="escaped the narrow detector claim"):
        _fill_conditional_hybrid_regions(
            source,
            edit,
            masks,
            route_decision="broad",
            background_exclude_mask=zero,
            lama_fill=lambda image, _mask: image,
            narrow_claim=narrow,
        )


def test_synthetic_ownership_conflict_completes_with_one_narrow_page_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_root = tmp_path / "synthetic"
    payload = build_generalization_synthetic_manifest(synthetic_root)
    conflict = next(
        page
        for page in payload["pages"]
        if page["page_id"] == "synthetic-ownership-conflict"
    )
    payload["pages"] = [conflict]
    payload["page_count"] = 1
    payload["page_ids"] = [conflict["page_id"]]
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        payload["pages"]
    )
    manifest = synthetic_root / "synthetic-inpaint-generalization-v4.json"
    manifest.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest.with_suffix(manifest.suffix + ".seal.json").write_text(
        json.dumps(
            {
                "schema_version": "inpaint-factorized-manifest-seal-v4",
                "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "candidate_generated": False,
                "candidate_seen": False,
                "annotation_frozen_before_candidate": True,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    page_id = str(conflict["page_id"])
    target = str(conflict["target_text_mask"])
    ownership = str(conflict["ownership_mask"])
    interior = str(conflict["bubble_interior_mask"])
    zero = str(conflict["protected_structure_mask"])
    matrix = {
        "schema_version": "inpaint-factorized-matrix-v3",
        "manifest": str(manifest),
        "axes": {
            "detector": ["detector"],
            "ownership": ["ownership"],
            "silhouette": ["silhouette"],
            "router": ["R3"],
            "expansion": ["bubble_interior"],
            "fill": ["conditional_hybrid"],
        },
        "controls": {
            "detector": "detector",
            "ownership": "ownership",
            "silhouette": "silhouette",
            "router": "R3",
            "expansion": "bubble_interior",
            "fill": "conditional_hybrid",
        },
        "families": {
            "detector": {
                "detector": {
                    "seed_variant": "raw",
                    "pages": {
                        page_id: {
                            "raw": target,
                            "refined": target,
                            "dilated": target,
                        }
                    },
                }
            },
            "ownership": {
                "ownership": {
                    "pages": {
                        page_id: {
                            "mask": ownership,
                            "broad_mask": ownership,
                            "content_components": target,
                        }
                    }
                }
            },
            "silhouette": {
                "silhouette": {"pages": {page_id: {"interior": interior}}}
            },
            "router": {
                "R3": {
                    "algorithm": "R3",
                    "minimum_background_samples": 1,
                    "pages": {
                        page_id: {
                            "ballons_clean": True,
                            "pr2_clean": True,
                            "ballons_clean_mask": interior,
                            "pr2_clean_mask": interior,
                            "unsafe_signal_mask": zero,
                        }
                    },
                }
            },
        },
        "retain_page_artifacts": True,
        "oracle_only": [],
    }
    matrix_path = tmp_path / "matrix.json"
    matrix_path.write_text(json.dumps(matrix, sort_keys=True), encoding="utf-8")
    output = tmp_path / "output"

    class FakeLamaPool:
        last: "FakeLamaPool | None" = None

        def __init__(self, **_kwargs: object) -> None:
            self.call_count = 0
            self.call_durations: list[float] = []
            FakeLamaPool.last = self

        def fill(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
            self.call_count += 1
            self.call_durations.append(0.01)
            candidate = image.copy()
            candidate[mask > 0] = 245
            return candidate

        def runtime_metrics_since(self, call_index: int) -> dict[str, object]:
            called = self.call_count - call_index
            return {
                "runtime_telemetry_complete": True,
                "positive_lama_runtime_p95_seconds": 0.01 if called else None,
                "peak_vram_allocated_mib": 1.0 if called else None,
                "peak_vram_reserved_mib": 1.0 if called else None,
                "cpu_fallback_count": 0,
                "lama_runtime_provider": "cuda:0" if called else "",
                "lama_runtime_precision": "bf16" if called else "",
            }

    monkeypatch.setattr(factorized_runner, "_LamaPool", FakeLamaPool)
    original_runtime_identity = factorized_runner._runtime_identity
    monkeypatch.setattr(
        factorized_runner,
        "_runtime_identity",
        lambda **_kwargs: {
            **original_runtime_identity(
                device="cpu",
                precision="bf16",
                inpaint_size=2048,
                lama_model_path=_kwargs["lama_model_path"],
            ),
            "requested_device": "cuda",
            "torch_version": "test",
            "torch_cuda_version": "test",
            "cudnn_version": 1,
            "cuda_available": True,
            "gpu_name": "test-gpu",
        },
    )

    assert factorized_main(
        [
            "--manifest",
            str(manifest),
            "--matrix",
            str(matrix_path),
            "--output-dir",
            str(output),
            "--device",
            "cuda",
        ]
    ) == 0
    result = json.loads(
        (output / "factorized-results.json").read_text(encoding="utf-8")
    )
    run = result["runs"][0]
    row = result["pages"][run["run_id"]][0]
    diagnostics = row["fill"]
    assert FakeLamaPool.last is not None
    assert FakeLamaPool.last.call_count == 1
    assert result["closure_ledger"][0]["closure_state"] == "executed"
    assert diagnostics["authoritative_overlap_narrow_verified"] is True
    assert diagnostics["authoritative_overlap_edit_pixel_count"] > 0
    assert diagnostics["region_fills"][-1]["fallback_reasons"] == [
        "authoritative_region_overlap"
    ]
    edit_path = output / "runs" / run["run_id"] / "edit_masks" / f"{page_id}.png"
    assert np.array_equal(
        cv2.imread(str(edit_path), cv2.IMREAD_GRAYSCALE),
        cv2.imread(target, cv2.IMREAD_GRAYSCALE),
    )
    assert row["canonical_statistics"]["outside_final_changed_pixel_count"] == 0
    runtime_binding = result["runtime_evidence_ledger"]
    runtime_ledger = json.loads(
        (output / runtime_binding["relative_path"]).read_text(encoding="utf-8")
    )
    event = runtime_ledger["runs"][0]["pages"][0]["inference_events"][0]
    assert event == {
        "backend": "current_lama",
        "call_index": 1,
        "cpu_fallback": False,
        "duration_seconds": 0.01,
        "precision": "bf16",
        "provider": "cuda:0",
    }
    result_path = output / "factorized-results.json"
    _validate_runtime_evidence_ledger(
        result,
        result_path,
        schema="inpaint-factorized-results-v3",
        finalists=frozenset({run["run_id"]}),
    )
    validate_evidence_artifact(result)

    runtime_path = output / runtime_binding["relative_path"]

    def reseal_runtime_ledger() -> None:
        runtime_canonical = {
            "runtime_identity": runtime_ledger["runtime_identity"],
            "runtime_source_inventory": runtime_ledger[
                "runtime_source_inventory"
            ],
            "runs": runtime_ledger["runs"],
            "complete_run_ids": runtime_ledger["complete_run_ids"],
            "positive_lama_inference_count": runtime_ledger[
                "positive_lama_inference_count"
            ],
        }
        runtime_ledger["ledger_sha256"] = hashlib.sha256(
            json.dumps(
                runtime_canonical,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        runtime_path.write_text(
            json.dumps(runtime_ledger, sort_keys=True), encoding="utf-8"
        )
        runtime_binding["ledger_sha256"] = runtime_ledger["ledger_sha256"]
        runtime_binding["complete_run_ids"] = runtime_ledger[
            "complete_run_ids"
        ]
        runtime_binding["positive_lama_inference_count"] = runtime_ledger[
            "positive_lama_inference_count"
        ]
        runtime_binding["artifact_sha256"] = hashlib.sha256(
            runtime_path.read_bytes()
        ).hexdigest()

    single_result = json.loads(json.dumps(result))
    single_ledger = json.loads(json.dumps(runtime_ledger))
    second_run = json.loads(json.dumps(run))
    second_run_id = f"{run['run_id']}__second"
    second_run["run_id"] = second_run_id
    result["runs"].append(second_run)
    result["pages"][second_run_id] = json.loads(
        json.dumps(result["pages"][run["run_id"]])
    )
    second_runtime_run = json.loads(json.dumps(runtime_ledger["runs"][0]))
    second_runtime_run["run_id"] = second_run_id
    second_runtime_run["pages"][0]["inference_events"][0]["call_index"] = 2
    runtime_ledger["runs"].append(second_runtime_run)
    runtime_ledger["complete_run_ids"].append(second_run_id)
    runtime_ledger["positive_lama_inference_count"] = 2
    result["positive_lama_inference_count"] = 2
    reseal_runtime_ledger()
    _validate_runtime_evidence_ledger(
        result,
        result_path,
        schema="inpaint-factorized-results-v3",
        finalists=frozenset({run["run_id"]}),
    )
    runtime_ledger["runs"][0]["pages"][0]["inference_events"][0][
        "call_index"
    ] = 2
    runtime_ledger["runs"][1]["pages"][0]["inference_events"][0][
        "call_index"
    ] = 1
    reseal_runtime_ledger()
    with pytest.raises(ValueError, match="global call inventory order differs"):
        _validate_runtime_evidence_ledger(
            result,
            result_path,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({run["run_id"]}),
        )

    result.clear()
    result.update(single_result)
    runtime_ledger = single_ledger
    runtime_binding = result["runtime_evidence_ledger"]
    runtime_path = output / runtime_binding["relative_path"]
    event = runtime_ledger["runs"][0]["pages"][0]["inference_events"][0]
    event["call_index"] = 2
    reseal_runtime_ledger()
    with pytest.raises(ValueError, match="global call inventory"):
        _validate_runtime_evidence_ledger(
            result,
            result_path,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({run["run_id"]}),
        )

    event["call_index"] = 1
    event["backend"] = "ballons_lama"
    reseal_runtime_ledger()
    with pytest.raises(ValueError, match="backend differs from run selection"):
        _validate_runtime_evidence_ledger(
            result,
            result_path,
            schema="inpaint-factorized-results-v3",
            finalists=frozenset({run["run_id"]}),
        )
