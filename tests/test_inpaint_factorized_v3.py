from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.contracts import (
    CandidateMaskResult,
    DetectorBox,
    FactorizedRunRecord,
    RoleCandidateSpec,
    Stage1Page,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (
    PageMasks,
    decide_bubble_route,
    expand_detector_claim,
    load_page_masks,
    load_stage1_manifest,
    read_detector_cache,
    score_page,
    write_detector_cache,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (
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
    _declared_combinations,
    _route_fill_backend,
    main as factorized_main,
)
from scripts.build_inpaint_factorized_manifest_v3 import build_manifest
from scripts.build_inpaint_factorized_matrix_v3 import build_matrix
from scripts.build_inpaint_fill_synthetic_v3 import build_synthetic_manifest
from scripts.build_inpaint_v3_contact_sheet import build_contact_sheet
from scripts.export_inpaint_silhouette_router_v3 import export_candidates
from scripts.merge_inpaint_factorized_v3 import merge_results


def _write_image(path: Path, image: np.ndarray) -> str:
    assert cv2.imwrite(str(path), image)
    return str(path)


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


def test_route_hybrid_uses_lama_only_for_narrow_and_flat_for_broad() -> None:
    assert _route_fill_backend("narrow_lama_broad_flat", "narrow") == "current_lama"
    assert (
        _route_fill_backend("narrow_lama_broad_flat", "broad")
        == "robust_flat_median"
    )
    assert _route_fill_backend("telea", "broad") == "telea"


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


def test_matrix_has_control_single_and_pairwise_but_no_triple_product() -> None:
    axes = {
        "detector": ("d0", "d1", "d2"),
        "router": ("r0", "r1"),
        "fill": ("f0", "f1"),
    }
    controls = {"detector": "d0", "router": "r0", "fill": "f0"}

    records = build_factorized_matrix(axes, controls)

    assert len(records) == 10
    assert controls in records
    assert not any(
        row["detector"] != "d0" and row["router"] != "r0" and row["fill"] != "f0"
        for row in records
    )


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


def test_known_background_synthetic_manifest_covers_all_fill_routes(
    tmp_path: Path,
) -> None:
    manifest = build_synthetic_manifest(tmp_path)

    assert [page["bubble_route_class"] for page in manifest["pages"]] == [
        "clean_flat",
        "clean_gradient",
        "texture",
        "line_art",
    ]
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
