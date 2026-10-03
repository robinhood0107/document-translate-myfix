from __future__ import annotations

from pathlib import Path
from types import MappingProxyType, SimpleNamespace
import json

import cv2
import numpy as np
import pytest

import scripts.benchmark_inpaint_glyph_refinement_v34 as v34_runner
from benchmarking.inpaint_detector_bakeoff.contracts import TargetInstance, mask_sha256
from benchmarking.inpaint_detector_bakeoff.glyph_refinement import (
    GlyphRefinementResult,
    extract_roi_local_glyph_masks,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import PageMasks
from scripts.benchmark_inpaint_glyph_refinement_v34 import (
    ARTIFACT_ROLES,
    CANDIDATES,
    DETECTOR_BUNDLE_EXPECTATIONS,
    MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE,
    MAX_SHARED_ARTIFACT_WRITES_PER_PAGE,
    SourceRegionEvidence,
    SourceRoutingEvidence,
    SourceRoutingOverlayRegionRecord,
    SeedlessOcrRegionRecord,
    build_candidate_plan,
    bind_detector_inference_bundle,
    build_source_routing_evidence,
    connected_existing_source_edit,
    connected_existing_source_edit_by_owner,
    extract_page_seedless_results,
    load_source_region_evidence,
    mask_only_final_mask,
    merge_glyph_refinement_results,
    score_candidate_page,
    shortlist_candidates,
    validate_output_inventory,
    write_normalized_page_artifacts,
    _canonical_sha256,
    _sha256,
    _unsigned_overlay_sha256,
    _source_semantic_decision,
    validate_seedless_ocr_overlay,
)


def _mask(shape: tuple[int, int], *points: tuple[int, int]) -> np.ndarray:
    value = np.zeros(shape, dtype=np.uint8)
    for y, x in points:
        value[y, x] = 255
    return value


def _write_mask(path: Path, mask: np.ndarray) -> None:
    encoded, buffer = cv2.imencode(".png", mask)
    assert encoded
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buffer.tobytes())


def _glyph(
    *,
    shape: tuple[int, int] = (8, 8),
    seed: np.ndarray | None = None,
    core: np.ndarray | None = None,
    effect: np.ndarray | None = None,
    protect: np.ndarray | None = None,
    provenance: dict[str, object] | None = None,
) -> GlyphRefinementResult:
    zero = np.zeros(shape, dtype=np.uint8)
    actual_seed = zero if seed is None else seed
    actual_core = zero if core is None else core
    actual_effect = zero if effect is None else effect
    actual_protect = zero if protect is None else protect
    refined = cv2.bitwise_or(actual_core, actual_effect)
    masks = {
        "owned_detector_seed": actual_seed.copy(),
        "effect_support": actual_effect.copy(),
        "candidate_mask": refined.copy(),
        "glyph_core": actual_core.copy(),
        "glyph_effect": actual_effect.copy(),
        "refined_mask": refined.copy(),
        "rejected_mask": zero.copy(),
        "hard_protect": actual_protect.copy(),
    }
    for value in masks.values():
        value.setflags(write=False)
    hashes = {
        "owned_detector_seed_sha256": mask_sha256(masks["owned_detector_seed"]),
        "effect_support_sha256": mask_sha256(masks["effect_support"]),
        "candidate_mask_sha256": mask_sha256(masks["candidate_mask"]),
        "glyph_core_sha256": mask_sha256(masks["glyph_core"]),
        "glyph_effect_sha256": mask_sha256(masks["glyph_effect"]),
        "refined_mask_sha256": mask_sha256(masks["refined_mask"]),
        "rejected_mask_sha256": mask_sha256(masks["rejected_mask"]),
        "hard_protect_sha256": mask_sha256(masks["hard_protect"]),
    }
    return GlyphRefinementResult(
        **masks,
        component_records=(),
        owner_records=(),
        provenance=MappingProxyType(
            {"schema_version": "test", **(provenance or {}), **hashes}
        ),
    )


def _source_region(
    region_id: str,
    ownership: np.ndarray,
    *,
    protected: np.ndarray | None = None,
    ambiguous: np.ndarray | None = None,
    corner: np.ndarray | None = None,
    available: bool = True,
) -> SourceRegionEvidence:
    zero = np.zeros(ownership.shape, dtype=np.uint8)
    return SourceRegionEvidence(
        region_id=region_id,
        ownership=ownership,
        protected=zero if protected is None else protected,
        ambiguous=zero if ambiguous is None else ambiguous,
        corner=zero if corner is None else corner,
        available=available,
        failure_reason="" if available else "source_protected_missing_or_invalid",
        artifacts=(),
    )


def _routing(
    raw_regions: tuple[dict[str, object], ...],
    source_regions: tuple[SourceRegionEvidence, ...],
    shape: tuple[int, int],
) -> SourceRoutingEvidence:
    sealed = tuple(
        SourceRoutingOverlayRegionRecord(
            page_id="example-page",
            source_sha256="1" * 64,
            region_id=str(raw["region_id"]),
            source_region_record_sha256=_canonical_sha256(raw),
            artifacts=MappingProxyType({}),
            **_source_semantic_decision(raw),
        )
        for raw in raw_regions
    )
    return build_source_routing_evidence(
        raw_regions, source_regions, sealed, shape=shape
    )


def _plan(spec, *, glyph: GlyphRefinementResult, baseline_mask: np.ndarray):
    ownership = np.full(glyph.refined_mask.shape, 255, dtype=np.uint8)
    return build_candidate_plan(
        spec,
        glyph=glyph,
        baseline_mask=baseline_mask,
        authoritative_owner_masks=(("owner-0", ownership),),
        all_ownership=ownership,
    )


def test_source_routing_uses_source_masks_and_fails_closed_on_overlap() -> None:
    shape = (6, 8)
    left = np.zeros(shape, dtype=np.uint8)
    left[1:5, 1:5] = 255
    right = np.zeros(shape, dtype=np.uint8)
    right[1:5, 4:7] = 255
    protected = _mask(shape, (2, 2))
    ambiguous = _mask(shape, (3, 2))
    corner = _mask(shape, (4, 2))
    raw = (
        {"region_id": "translate", "proposal": {"text_class": "text_free"}},
        {"region_id": "preserve", "proposal": {"text_class": "sfx"}},
    )
    routed = _routing(
        raw,
        (
            _source_region(
                "translate",
                left,
                protected=protected,
                ambiguous=ambiguous,
                corner=corner,
            ),
            _source_region("preserve", right),
        ),
        shape,
    )

    assert routed.authoritative_translate_ownership[1, 1] == 255
    assert routed.authoritative_translate_ownership[2, 2] == 0
    assert routed.ownership_conflict[2, 4] == 255
    assert routed.preserve_or_abstain[2, 6] == 255
    assert routed.hard_protect[2, 2] == 255
    assert routed.hard_protect[3, 2] == 255
    assert routed.hard_protect[4, 2] == 255
    assert routed.hard_protect[2, 4] == 255


def test_touching_translate_owners_cannot_share_one_detector_seed() -> None:
    shape = (80, 80)
    image = np.full((*shape, 3), 255, dtype=np.uint8)
    image[30:45, 25:38] = 0
    left = np.zeros(shape, dtype=np.uint8)
    left[20:60, 10:32] = 255
    right = np.zeros(shape, dtype=np.uint8)
    right[20:60, 32:60] = 255
    raw = (
        {"region_id": "left", "proposal": {"text_class": "text_free"}},
        {"region_id": "right", "proposal": {"text_class": "text_free"}},
    )
    routed = _routing(
        raw, (_source_region("left", left), _source_region("right", right)), shape
    )
    assert np.all(routed.hard_protect[20:60, 31:33] == 255)
    seed = _mask(shape, (35, 30))
    result = extract_roi_local_glyph_masks(
        image,
        detector_seed=seed,
        ocr_ownership=routed.authoritative_translate_ownership,
        hard_protect=routed.hard_protect,
        detector_provider="neutral-detector",
        ownership_provider="sealed-source-owners",
    )
    assert np.count_nonzero(result.refined_mask[:, 32:60]) == 0
    assert np.count_nonzero(result.refined_mask) == 0


def test_empty_source_routing_and_zero_glyph_keep_baseline_byte_identical() -> None:
    shape = (7, 7)
    routed = _routing((), (), shape)
    assert np.count_nonzero(routed.hard_protect) == 0
    assert routed.authoritative_translate_owner_masks == ()
    baseline = _mask(shape, (3, 3))
    glyph = _glyph(shape=shape)
    spec = next(row for row in CANDIDATES if row.candidate_id == "b2_narrow_replacement")
    plan = build_candidate_plan(spec, glyph=glyph, baseline_mask=baseline)
    assert np.array_equal(mask_only_final_mask(baseline, plan), baseline)


def test_detector_bundle_binds_new_page_set_without_old_e1_output_sha() -> None:
    expected = DETECTOR_BUNDLE_EXPECTATIONS["finetune"]
    role = {
        "candidate_id": expected.candidate_id,
        "model_sha256": expected.role_model_sha256,
        "preprocessing_contract_sha256": expected.preprocessing_contract_sha256,
    }
    identity = {
        "raw": {"output_mask_set_sha256": "a" * 64},
        "dilated": {"output_mask_set_sha256": "b" * 64},
    }
    raw = {
        "role_candidate": role,
        "variant_output_identity": identity,
        "model": {"sha256": expected.model_asset_sha256},
        "candidate": expected.candidate_id,
        "result_sha256": "c" * 64,
    }
    dilated = {**raw, "result_sha256": "d" * 64}
    binding = bind_detector_inference_bundle(raw, dilated, expected)
    assert binding["raw_output_mask_set_sha256"] == "a" * 64
    assert binding["dilated_output_mask_set_sha256"] == "b" * 64


def test_missing_source_sibling_abstains_instead_of_using_eval_annotation(
    tmp_path: Path,
) -> None:
    shape = (5, 5)
    ownership = np.full(shape, 255, dtype=np.uint8)
    ownership_path = tmp_path / "source" / "region" / "ownership.png"
    _write_mask(ownership_path, ownership)
    raw = {
        "region_id": "r0",
        "ownership_mask": str(ownership_path),
        "protected_structure_mask": str(tmp_path / "eval-protected.png"),
        "proposal": {"text_class": "text_free"},
    }

    sealed = SourceRoutingOverlayRegionRecord(
        page_id="example-page",
        source_sha256="1" * 64,
        region_id="r0",
        source_region_record_sha256=_canonical_sha256(raw),
        artifacts=MappingProxyType({}),
        **_source_semantic_decision(raw),
    )
    with pytest.raises(ValueError, match="artifact role is missing"):
        load_source_region_evidence(raw, sealed, shape=shape)


def test_connected_existing_source_edit_selects_only_adjacent_components() -> None:
    shape = (9, 12)
    existing = np.zeros(shape, dtype=np.uint8)
    existing[2:5, 2:5] = 255
    existing[2:5, 8:11] = 255
    anchor = _mask(shape, (3, 5))

    selected = connected_existing_source_edit(existing, anchor)

    assert np.all(selected[2:5, 2:5] == 255)
    assert np.count_nonzero(selected[:, 8:11]) == 0


def test_candidate_plans_separate_generation_commit_and_replacement() -> None:
    shape = (8, 8)
    baseline = np.zeros(shape, dtype=np.uint8)
    baseline[2:5, 2:4] = 255
    seed = _mask(shape, (3, 3), (3, 4), (3, 5))
    core = _mask(shape, (3, 4), (3, 5))
    effect = _mask(shape, (2, 5), (4, 5))
    glyph = _glyph(shape=shape, seed=seed, core=core, effect=effect)
    by_id = {spec.candidate_id: spec for spec in CANDIDATES}

    control = _plan(by_id["b0_pr6"], glyph=glyph, baseline_mask=baseline)
    context = _plan(
        by_id["b1_context_additive"], glyph=glyph, baseline_mask=baseline
    )
    narrow = _plan(
        by_id["b2_narrow_replacement"], glyph=glyph, baseline_mask=baseline
    )
    segmented = _plan(
        by_id["b3_conditional_segmenter"], glyph=glyph, baseline_mask=baseline
    )

    assert control.lama_request_count == 0
    assert np.count_nonzero(context.commit_mask & baseline) == 0
    assert np.any((context.generation_mask > 0) & (baseline > 0))
    assert np.array_equal(narrow.commit_mask, core)
    assert np.count_nonzero(narrow.replacement_source_edit) > 0
    assert np.array_equal(segmented.commit_mask, cv2.bitwise_or(core, effect))
    assert np.count_nonzero(segmented.replacement_source_edit) > 0
    assert narrow.lama_request_count == segmented.lama_request_count == 1


def test_candidate_plan_never_expands_without_owned_seed() -> None:
    shape = (5, 5)
    glyph = _glyph(shape=shape, core=_mask(shape, (2, 2)))
    spec = next(row for row in CANDIDATES if row.candidate_id == "b2_narrow_replacement")

    with pytest.raises(AssertionError, match="expanded without an owned source seed"):
        _plan(spec, glyph=glyph, baseline_mask=np.zeros(shape, np.uint8))


@pytest.mark.parametrize(
    "candidate_id",
    [
        "b1_context_additive",
        "b2_narrow_replacement",
        "b3_conditional_segmenter",
    ],
)
def test_rejected_seed_touching_baseline_is_strict_noop(candidate_id: str) -> None:
    shape = (7, 7)
    baseline = np.zeros(shape, dtype=np.uint8)
    baseline[2:5, 2:5] = 255
    rejected_seed = _mask(shape, (3, 3))
    glyph = _glyph(shape=shape, seed=rejected_seed)
    spec = next(row for row in CANDIDATES if row.candidate_id == candidate_id)

    plan = _plan(spec, glyph=glyph, baseline_mask=baseline)
    final = mask_only_final_mask(baseline, plan)

    assert plan.lama_request_count == 0
    assert np.count_nonzero(plan.generation_mask) == 0
    assert np.count_nonzero(plan.commit_mask) == 0
    assert np.count_nonzero(plan.replacement_source_edit) == 0
    assert np.array_equal(final, baseline)


def test_seedless_result_becomes_a_distinct_source_positive_seed() -> None:
    shape = (6, 6)
    zero = np.zeros(shape, dtype=np.uint8)
    core = _mask(shape, (2, 2), (2, 3))
    seedless = _glyph(
        shape=shape,
        core=core,
        provenance={
            "seed_mode": "authoritative_ocr_seedless",
            "status": "completed",
        },
    )

    merged = merge_glyph_refinement_results(
        (seedless,), shape=shape, hard_protect=zero
    )

    assert np.array_equal(merged.refined_mask, core)
    assert np.array_equal(merged.owned_detector_seed, core)
    assert merged.provenance["no_seed_no_expansion"] is True
    for field in (
        "owned_detector_seed",
        "effect_support",
        "candidate_mask",
        "glyph_core",
        "glyph_effect",
        "refined_mask",
        "rejected_mask",
        "hard_protect",
    ):
        assert np.asarray(getattr(merged, field)).flags.writeable is False


def test_candidate_plan_rehashes_glyph_evidence_and_rejects_tamper() -> None:
    shape = (6, 6)
    seed = _mask(shape, (2, 2))
    glyph = _glyph(shape=shape, seed=seed, core=seed)
    glyph.glyph_core.setflags(write=True)
    glyph.glyph_core[2, 3] = 255
    glyph.glyph_core.setflags(write=False)
    spec = next(row for row in CANDIDATES if row.candidate_id == "b2_narrow_replacement")

    with pytest.raises(ValueError, match="glyph_core SHA differs"):
        build_candidate_plan(spec, glyph=glyph, baseline_mask=np.zeros(shape, np.uint8))


def test_seedless_page_runner_requires_overlay_record() -> None:
    shape = (8, 8)
    source = np.zeros((*shape, 3), dtype=np.uint8)
    ownership = np.zeros(shape, dtype=np.uint8)
    ownership[1:7, 1:7] = 255
    region = _source_region("r0", ownership)
    routing = _routing(
        ({"region_id": "r0", "proposal": {"text_class": "text_free"}},),
        (region,),
        shape,
    )

    results, status, positive = extract_page_seedless_results(
        source,
        page_id="example-page",
        raw_regions=(
            {"region_id": "r0", "proposal": {"text_class": "text_free"}},
        ),
        source_regions=(region,),
        routing=routing,
        evidence_records={},
    )

    assert results == ()
    assert status["information_limited"] is True
    assert status["status_counts"] == {"information_limited": 1}
    assert np.count_nonzero(positive) == 0


def test_seedless_page_runner_uses_only_matching_authoritative_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (8, 8)
    source = np.zeros((*shape, 3), dtype=np.uint8)
    ownership = np.zeros(shape, dtype=np.uint8)
    ownership[1:7, 1:7] = 255
    region = _source_region("r0", ownership)
    raw = {"region_id": "r0", "proposal": {"text_class": "text_free"}}
    routing = _routing((raw,), (region,), shape)
    core = _mask(shape, (3, 3), (3, 4))

    def fake_extract(source_image, *, ocr_ownership, hard_protect, evidence):
        return _glyph(
            shape=shape,
            core=core,
            protect=hard_protect,
            provenance={
                "seed_mode": "authoritative_ocr_seedless",
                "status": "completed",
                "reason": "test_source_cues",
            },
        )

    monkeypatch.setattr(v34_runner, "extract_seedless_roi_glyph_masks", fake_extract)
    record = SeedlessOcrRegionRecord(
        page_id="example-page",
        source_sha256="1" * 64,
        region_id="r0",
        owner_region_id="r0",
        owner_count=1,
        provider="example_ocr_provider",
        ocr_text="example",
        ocr_script="example_script",
        ocr_confidence=0.9,
        confidence_kind="detector_block_confidence",
        owner_binding_kind="canonical_block_region_id",
        canonical_block_id="block-0",
        source_record_sha256="2" * 64,
        source_provenance_sha256="3" * 64,
        processing_action="translate_inpaint",
        route_class="clean_translucent",
    )

    results, status, positive = extract_page_seedless_results(
        source,
        page_id="example-page",
        raw_regions=(raw,),
        source_regions=(region,),
        routing=routing,
        evidence_records={("example-page", "r0"): record},
    )

    assert len(results) == 1
    assert status["status_counts"] == {"information_limited": 1}
    assert status["regions"][0]["reason"] == "independent_spatial_text_cue_missing"
    assert status["information_limited"] is True
    assert status["finalist_eligible"] is False
    assert np.array_equal(positive, core)


def test_evaluation_masks_score_but_do_not_change_candidate_plan(tmp_path: Path) -> None:
    shape = (6, 6)
    target = _mask(shape, (2, 2))
    target_path = tmp_path / "target.png"
    _write_mask(target_path, target)
    page = SimpleNamespace(
        page_id="example-page",
        expected_edit="required",
        no_edit=False,
        target_instances=(
            TargetInstance("instance-0", str(target_path), priority="required"),
        ),
    )
    zero = np.zeros(shape, dtype=np.uint8)
    ownership = np.full(shape, 255, dtype=np.uint8)
    routing = SourceRoutingEvidence(
        authoritative_translate_ownership=ownership,
        authoritative_translate_owner_masks=(("r0", ownership),),
        all_ownership=ownership,
        preserve_or_abstain=zero,
        ownership_conflict=zero,
        source_region_protect=zero,
        source_region_ambiguous=zero,
        source_corner_protect=zero,
        hard_protect=zero,
        semantic_actions=(),
        artifact_inventory=(),
    )
    glyph = _glyph(shape=shape, seed=target, core=target)
    spec = next(row for row in CANDIDATES if row.candidate_id == "b3_conditional_segmenter")
    plan = _plan(spec, glyph=glyph, baseline_mask=zero)
    before = plan.commit_mask.copy()
    evaluation = PageMasks(
        target=target,
        protected=target.copy(),
        ambiguous=zero,
        ownership=ownership,
        claim_seed=ownership,
        existing_edit=zero,
        target_instances=(("instance-0", target),),
        preserve=zero,
    )

    score = score_candidate_page(
        page=page,
        evaluation_masks=evaluation,
        source_routing=routing,
        raw_detector_seed=target,
        source_positive_seed=target,
        glyph=glyph,
        baseline_mask=zero,
        plan=plan,
    )

    assert np.array_equal(plan.commit_mask, before)
    assert score["evaluation_safety"]["protected_delta_overlap"] == 1
    assert score["source_safety"]["source_hard_protect_commit_overlap"] == 0


def test_output_inventory_detects_mask_tampering(tmp_path: Path) -> None:
    shape = (4, 4)
    zero = np.zeros(shape, dtype=np.uint8)
    seed = _mask(shape, (1, 1))
    glyph = _glyph(shape=shape, seed=seed, core=seed)
    plans = {
        spec.candidate_id: _plan(
            spec,
            glyph=glyph,
            baseline_mask=zero,
        )
        for spec in CANDIDATES
    }
    artifacts, bindings, counts = write_normalized_page_artifacts(
        output_root=tmp_path,
        page_id="example-page",
        detector_seed=seed,
        source_claim_seed=zero,
        seedless_source_seed=zero,
        hard_protect=zero,
        seeded_glyph=glyph,
        conditional_glyph=glyph,
        candidate_plans=plans,
    )
    unsigned = {
        "schema_version": "test",
        "candidate_ids": [spec.candidate_id for spec in CANDIDATES],
        "page_ids": ["example-page"],
        "artifact_roles": list(ARTIFACT_ROLES),
        "mask_bindings": bindings,
        "write_counts": counts,
        "per_page_write_counts": [{"page_id": "example-page", **counts}],
        "artifacts": artifacts,
    }
    inventory = {**unsigned, "inventory_sha256": _canonical_sha256(unsigned)}
    validate_output_inventory(inventory, output_root=tmp_path)

    artifact = tmp_path / str(artifacts[0]["relative_path"])
    _write_mask(artifact, _mask((4, 4), (2, 2)))
    with pytest.raises(ValueError, match="artifact SHA differs"):
        validate_output_inventory(inventory, output_root=tmp_path)


def test_4k_normalized_inventory_bounds_shared_and_candidate_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2160, 3840)
    zero = np.zeros(shape, dtype=np.uint8)
    seed = _mask(shape, (1080, 1920))
    core = _mask(shape, (1080, 1920))
    effect = _mask(shape, (1080, 1921))
    seeded = _glyph(shape=shape, seed=seed, core=core, effect=effect)
    conditional = _glyph(shape=shape, seed=seed, core=core, effect=effect)
    plans = {
        spec.candidate_id: _plan(
            spec,
            glyph=(conditional if spec.candidate_id == "b3_conditional_segmenter" else seeded),
            baseline_mask=zero,
        )
        for spec in CANDIDATES
    }
    writes: list[Path] = []

    def fake_write(path: Path, mask: np.ndarray) -> dict[str, object]:
        writes.append(path)
        return {
            "file_sha256": "1" * 64,
            "pixel_sha256": mask_sha256(mask),
            "pixel_count": int(np.count_nonzero(mask)),
            "size_bytes": 1,
        }

    monkeypatch.setattr(v34_runner, "_write_mask", fake_write)
    artifacts, bindings, counts = write_normalized_page_artifacts(
        output_root=tmp_path,
        page_id="four-k-page",
        detector_seed=seed,
        source_claim_seed=zero,
        seedless_source_seed=zero,
        hard_protect=zero,
        seeded_glyph=seeded,
        conditional_glyph=conditional,
        candidate_plans=plans,
    )

    assert len(bindings) == len(CANDIDATES) * len(ARTIFACT_ROLES)
    assert len(writes) == len(artifacts) == counts["artifact_write_count"]
    assert counts["shared_artifact_write_count"] <= (
        MAX_SHARED_ARTIFACT_WRITES_PER_PAGE
    )
    assert counts["candidate_artifact_write_count"] <= (
        MAX_CANDIDATE_ARTIFACT_WRITES_PER_PAGE
    )
    assert not any(
        row["scope_kind"] == "candidate" and row["scope_id"] == "b0_pr6"
        for row in artifacts
    )
    seeded_core_ids = {
        row["artifact_id"]
        for row in bindings
        if row["role"] == "glyph_core"
        and row["candidate_id"] in {"b1_context_additive", "b3_conditional_segmenter"}
    }
    assert len(seeded_core_ids) == 1


def test_shortlist_is_safe_nonregressing_and_limited_to_two() -> None:
    rows = [
        {
            "candidate_id": "b0_pr6",
            "mask_only_safety_pass": True,
            "target_coverage_nonregression": True,
            "coverage_98_instance_count": 10,
            "aggregate_target_coverage": 0.9,
            "commit_pixel_count": 0,
        },
        {
            "candidate_id": "unsafe",
            "mask_only_safety_pass": False,
            "target_coverage_nonregression": True,
            "coverage_98_instance_count": 100,
            "aggregate_target_coverage": 1.0,
            "commit_pixel_count": 1,
        },
        *(
            {
                "candidate_id": f"safe-{index}",
                "mask_only_safety_pass": True,
                "target_coverage_nonregression": True,
                "coverage_98_instance_count": 20 - index,
                "aggregate_target_coverage": 0.95,
                "commit_pixel_count": index + 1,
            }
            for index in range(3)
        ),
    ]

    assert shortlist_candidates(rows) == ["safe-0", "safe-1"]


def test_mask_only_replacement_final_mask_restores_only_selected_source_edit() -> None:
    shape = (7, 7)
    baseline = np.zeros(shape, dtype=np.uint8)
    baseline[1:3, 1:3] = 255
    baseline[4:6, 4:6] = 255
    seed = _mask(shape, (2, 3))
    core = _mask(shape, (2, 2), (2, 3))
    glyph = _glyph(shape=shape, seed=seed, core=core)
    spec = next(row for row in CANDIDATES if row.candidate_id == "b2_narrow_replacement")
    plan = _plan(spec, glyph=glyph, baseline_mask=baseline)

    final = mask_only_final_mask(baseline, plan)

    assert np.all(final[4:6, 4:6] == 255)
    assert final[1, 1] == 0
    assert final[2, 3] == 255


def test_exact_owner_component_selection_uses_local_bounding_boxes_on_4k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shape = (2160, 3840)
    baseline = np.zeros(shape, dtype=np.uint8)
    anchor = np.zeros(shape, dtype=np.uint8)
    for index in range(48):
        y = 40 + (index // 12) * 480
        x = 40 + (index % 12) * 300
        baseline[y : y + 5, x : x + 7] = 255
        anchor[y + 2, x + 3] = 255
    owner = np.full(shape, 255, dtype=np.uint8)
    compared_shapes: list[tuple[int, int]] = []
    original = cv2.connectedComponentsWithStats

    class TrackingLabels(np.ndarray):
        def __new__(cls, value: np.ndarray) -> "TrackingLabels":
            return np.asarray(value).view(cls)

        def __eq__(self, other: object) -> np.ndarray:  # type: ignore[override]
            compared_shapes.append(tuple(self.shape))
            return np.asarray(self).__eq__(other)

    def tracked_components(*args: object, **kwargs: object):
        count, labels, stats, centroids = original(*args, **kwargs)
        return count, TrackingLabels(labels), stats, centroids

    monkeypatch.setattr(v34_runner.cv2, "connectedComponentsWithStats", tracked_components)
    selected = connected_existing_source_edit_by_owner(
        baseline,
        anchor,
        authoritative_owner_masks=(("owner-0", owner),),
        all_ownership=owner,
    )
    assert np.array_equal(selected, baseline)
    assert compared_shapes
    assert max(height * width for height, width in compared_shapes) <= 7 * 9


def _seedless_overlay_fixture() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    source_sha = "1" * 64
    manifest_sha = "2" * 64
    inventory_sha = "3" * 64
    source = {
        "pages": [
            {
                "page_id": "example-page",
                "source_sha256": source_sha,
                "regions": [{"region_id": "region-0"}],
            }
        ]
    }
    binding = {
        "manifest_sha256": manifest_sha,
        "page_inventory_sha256": inventory_sha,
    }
    overlay: dict[str, object] = {
        "schema_version": "inpaint-seedless-ocr-evidence-overlay-v34",
        "source_manifest_sha256": manifest_sha,
        "source_page_inventory_sha256": inventory_sha,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
        "pages": [
            {
                "page_id": "example-page",
                "source_sha256": source_sha,
                "regions": [
                    {
                        "region_id": "region-0",
                        "owner_region_id": "region-0",
                        "owner_count": 1,
                        "provider": "example_ocr_provider",
                        "ocr_text": "example",
                        "ocr_script": "example_script",
                        "ocr_confidence": 0.9,
                        "processing_action": "translate_inpaint",
                        "route_class": "clean_translucent",
                        "authoritative_ocr": True,
                        "candidate_seen": False,
                        "annotation_frozen_before_candidate": True,
                    }
                ],
            }
        ],
    }
    overlay["overlay_sha256"] = _unsigned_overlay_sha256(overlay)
    return source, binding, overlay


def _write_seedless_overlay(path: Path, overlay: dict[str, object]) -> None:
    path.write_text(json.dumps(overlay), encoding="utf-8")
    seal = {
        "schema_version": "inpaint-seedless-ocr-evidence-seal-v34",
        "overlay_file_sha256": _sha256(path),
        "candidate_generated": False,
        "candidate_seen": False,
        "annotation_frozen_before_candidate": True,
    }
    path.with_suffix(path.suffix + ".seal.json").write_text(
        json.dumps(seal), encoding="utf-8"
    )


def test_seedless_overlay_missing_is_information_limited() -> None:
    source, binding, _overlay = _seedless_overlay_fixture()

    result = validate_seedless_ocr_overlay(
        None,
        source_binding=binding,
        source_manifest_payload=source,
    )

    assert result["available"] is False
    assert result["status"] == "overlay_missing"
    assert result["records"] == {}


def test_legacy_seedless_overlay_without_evidence_chain_fails_closed(tmp_path: Path) -> None:
    source, binding, overlay = _seedless_overlay_fixture()
    path = tmp_path / "overlay.json"
    _write_seedless_overlay(path, overlay)

    with pytest.raises(ValueError, match="overlay fields differ"):
        validate_seedless_ocr_overlay(
            path,
            source_binding=binding,
            source_manifest_payload=source,
        )


@pytest.mark.parametrize("mutation", ["tamper", "manifest_mismatch"])
def test_seedless_overlay_tamper_or_manifest_mismatch_fails_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    source, binding, overlay = _seedless_overlay_fixture()
    path = tmp_path / "overlay.json"
    if mutation == "tamper":
        _write_seedless_overlay(path, overlay)
        overlay["pages"][0]["regions"][0]["ocr_text"] = "changed"  # type: ignore[index]
        path.write_text(json.dumps(overlay), encoding="utf-8")
    else:
        overlay["source_manifest_sha256"] = "4" * 64
        overlay["overlay_sha256"] = _unsigned_overlay_sha256(overlay)
        _write_seedless_overlay(path, overlay)

    with pytest.raises(ValueError):
        validate_seedless_ocr_overlay(
            path,
            source_binding=binding,
            source_manifest_payload=source,
        )
