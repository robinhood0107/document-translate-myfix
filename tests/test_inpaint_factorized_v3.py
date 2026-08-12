from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

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
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (
    assert_complete_closure_ledger,
    build_combination_closure_ledger,
    build_factorized_matrix,
    fill_factorized_mask,
    reconstruction_error,
    select_pareto_records,
)
from benchmarking.inpaint_detector_bakeoff.silhouette import (
    ballons_native_clean_background,
    extract_ballons_native_interior,
    extract_pr2_validated_interior,
)
from scripts.benchmark_inpaint_factorized_v3 import (
    _annotation_masks,
    _prepare_closure_ledger,
    _declared_combinations,
    _route_fill_backend,
    _with_candidate_ownership,
    main as factorized_main,
)
from scripts.build_inpaint_factorized_manifest_v3 import build_manifest
from scripts.build_inpaint_factorized_manifest_v4 import build_manifest as build_manifest_v4
from scripts.build_inpaint_development_source_index_v4 import build_source_index
from scripts.build_inpaint_source_proposals_v4 import propose_semantic_contract
from scripts.apply_inpaint_source_review_v4 import apply_source_review
from scripts.record_inpaint_source_review_v4 import record_source_review
from scripts.build_inpaint_factorized_matrix_v3 import build_matrix
from scripts.build_inpaint_fill_synthetic_v3 import build_synthetic_manifest
from scripts.build_inpaint_fill_oracle_matrix_v3 import build_fill_matrix
from scripts.build_inpaint_v3_contact_sheet import build_contact_sheet
from scripts.export_inpaint_silhouette_router_v3 import export_candidates
from scripts.merge_inpaint_factorized_v3 import merge_results
from scripts.export_inpaint_silhouette_consensus_v4 import consensus_masks
from scripts.benchmark_inpaint_detector_fusions_v4 import (
    _logical_runs as detector_fusion_runs,
    run_fusion_matrix,
)


def _write_image(path: Path, image: np.ndarray) -> str:
    assert cv2.imwrite(str(path), image)
    return str(path)


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

    assert len(runs) == 17
    assert {"primary", "secondary", "roi"}.issubset(run_ids)
    assert "primary__or__secondary" in run_ids
    assert "primary__and__roi" in run_ids
    assert "primary__gated_seed_missing__roi" in run_ids
    assert "secondary__gated_union__roi" in run_ids


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

    result = run_fusion_matrix(manifest_path, spec_path)

    metrics = result["runs"][0]["metrics"]
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

    result = run_fusion_matrix(manifest_path, spec_path)

    run = result["runs"][0]
    assert run["status"] == "information_limited"
    assert run["closure_reason"] == "target_extent_not_independent"
    assert run["metrics"]["target_extent_independent"] is False
    assert run["metrics"]["target_mask_provenance"] == [
        "current_ctd_raw_components"
    ]


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
                "bubble_interior_mask": paths["full"],
                "corner_protect_mask": paths["zeros"],
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


def test_pareto_selection_excludes_oracle_and_hard_gate_failure() -> None:
    safe = {
        "aggregate_target_coverage": 1.0,
        "minimum_target_instance_coverage": 1.0,
        "target_instance_seed_recall": 1.0,
        "aggregate_residue_score": 0.1,
        "baseline_aggregate_residue_score": 0.2,
        "reconstruction_mse": 1.0,
        "runtime_seconds": 1.0,
    }
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
    metrics = {
        "target_extent_independent": False,
        "target_mask_provenance": ["current_ctd_raw_components"],
        "aggregate_target_coverage": 1.0,
        "minimum_target_instance_coverage": 1.0,
        "target_instance_seed_recall": 1.0,
        "residue_gate_applicable": False,
    }
    record = FactorizedRunRecord(
        "circular", "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "information_limited"
    assert selected[0].closure_reason == "target_extent_not_independent"


def test_pareto_selection_requires_independent_target_inventory() -> None:
    metrics = {
        "target_extent_independent": True,
        "target_inventory_independent": False,
        "target_mask_provenance": ["paired_source_extent"],
        "aggregate_target_coverage": 1.0,
        "minimum_target_instance_coverage": 1.0,
        "target_instance_seed_recall": 1.0,
        "residue_gate_applicable": False,
    }
    record = FactorizedRunRecord(
        "inventory-circular", "d", "o", "s", "r", "e", "f", False, "active", metrics
    )

    selected = select_pareto_records([record])

    assert selected[0].status == "information_limited"
    assert selected[0].closure_reason == "target_inventory_not_independent"


def test_pareto_gate_rejects_missing_residue_and_ambiguous_changes() -> None:
    safe = {
        "aggregate_target_coverage": 1.0,
        "minimum_target_instance_coverage": 1.0,
        "target_instance_seed_recall": 1.0,
        "aggregate_residue_score": 0.1,
        "baseline_aggregate_residue_score": 0.2,
        "residue_gate_applicable": True,
    }
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
    }
    manifest_path = tmp_path / "manifest.json"
    matrix_path = tmp_path / "matrix.json"
    output = tmp_path / "output"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    matrix_path.write_text(json.dumps(matrix), encoding="utf-8")

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
    assert result["runs"][0]["status"] == "pareto"
    assert result["runs"][0]["metrics"]["aggregate_target_coverage"] == 1.0
    assert result["runs"][0]["metrics"]["reconstruction_mse"] == 0.0
    assert result["runs"][0]["metrics"]["outside_final_changed"] == 0


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
        metrics = {
            "aggregate_target_coverage": 1.0,
            "minimum_target_instance_coverage": 1.0,
            "target_instance_seed_recall": 1.0,
            "protected_structure_overlap": 0,
            "protected_structure_changed": 0,
            "ambiguous_structure_overlap": 0,
            "ambiguous_structure_changed": 0,
            "outside_final_changed": 0,
            "broad_route_false_positive": 0,
            "no_edit_false_edit": 0,
            "required_skip_count": 0,
            "missed_target_instance_count": 0,
            "page_residue_worsened_count": 0,
            "runtime_seconds": runtime,
            "residue_gate_applicable": False,
        }
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
        metrics = {
            "aggregate_target_coverage": 1.0,
            "minimum_target_instance_coverage": 1.0,
            "target_instance_seed_recall": 1.0,
            "protected_structure_overlap": 0,
            "protected_structure_changed": 0,
            "ambiguous_structure_overlap": 0,
            "ambiguous_structure_changed": 0,
            "outside_final_changed": 0,
            "broad_route_false_positive": 0,
            "no_edit_false_edit": 0,
            "required_skip_count": 0,
            "missed_target_instance_count": 0,
            "page_residue_worsened_count": 0,
            "runtime_seconds": 1.0,
            "residue_gate_applicable": False,
        }
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

    assert len(manifest["pages"]) == 10
    assert {page["bubble_route_class"] for page in manifest["pages"]} == {
        "clean_flat",
        "clean_gradient",
        "texture",
        "line_art",
        "ambiguous",
    }
    for page in manifest["pages"]:
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
    assert len(matrix["explicit_combinations"]) == 11
    assert {
        row["fill"] for row in matrix["explicit_combinations"]
    } == {
        "robust_flat_median",
        "planar_gradient",
        "telea",
        "current_lama",
        "ballons_lama",
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
    assert len(physical) == 11
    assert {row.closure_state for row in ledger} == {"executed"}
