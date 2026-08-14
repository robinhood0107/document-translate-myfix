from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from benchmarking.inpaint_detector_bakeoff.proposal_refinement import (
    RegionAdmissionEvidence,
    refine_detector_proposal,
)
from benchmarking.inpaint_detector_bakeoff.contracts import mask_sha256
from benchmarking.inpaint_detector_bakeoff.semantic import (
    PRESERVE,
    REVIEW,
    TRANSLATE,
    SemanticDecision,
    product_semantic_decision,
)
import scripts.build_inpaint_product_policy_overlay_v33 as policy_overlay
from scripts.benchmark_inpaint_proposal_refinement_v33 import (
    _canonical_sha256,
    _stage1_input,
)
from scripts.benchmark_inpaint_proposal_stage2_v33 import (
    _runtime_action_masks,
    _validated_addition_paths,
)


def _mask(shape: tuple[int, int], *boxes: tuple[int, int, int, int]) -> np.ndarray:
    result = np.zeros(shape, dtype=np.uint8)
    for x1, y1, x2, y2 in boxes:
        result[y1:y2, x1:x2] = 255
    return result


def _region(
    shape: tuple[int, int],
    region_id: str,
    box: tuple[int, int, int, int],
    action: str = TRANSLATE,
) -> RegionAdmissionEvidence:
    return RegionAdmissionEvidence(
        region_id,
        _mask(shape, box),
        SemanticDecision("dialogue_free", action),
    )


def _run(
    *,
    fine_raw: np.ndarray,
    fine_native3: np.ndarray | None = None,
    tiled_raw: np.ndarray | None = None,
    tiled_native3: np.ndarray | None = None,
    provider: str = "or",
    expansion: str = "connected_halo",
    policy: str = "g1",
    regions: tuple[RegionAdmissionEvidence, ...] | None = None,
    baseline: np.ndarray | None = None,
    source_seed: np.ndarray | None = None,
    structure: np.ndarray | None = None,
    ambiguous: np.ndarray | None = None,
    corner: np.ndarray | None = None,
):
    shape = fine_raw.shape
    zero = np.zeros(shape, dtype=np.uint8)
    return refine_detector_proposal(
        finetune_raw=fine_raw,
        finetune_native3=fine_native3 if fine_native3 is not None else fine_raw,
        tiled_raw=tiled_raw if tiled_raw is not None else zero,
        tiled_native3=tiled_native3 if tiled_native3 is not None else (
            tiled_raw if tiled_raw is not None else zero
        ),
        provider_mode=provider,
        expansion_mode=expansion,
        admission_policy=policy,
        regions=regions or (_region(shape, "r0", (0, 0, shape[1], shape[0])),),
        pr6_existing_edit=baseline if baseline is not None else zero,
        source_raw_owned=source_seed if source_seed is not None else zero,
        structure_protect=structure if structure is not None else zero,
        ambiguous_protect=ambiguous if ambiguous is not None else zero,
        corner_protect=corner if corner is not None else zero,
    )


def test_product_semantic_action_is_explicit_and_fail_closed() -> None:
    assert product_semantic_decision(
        {"proposal": {"text_class": "text_free"}}
    ).action == TRANSLATE
    assert product_semantic_decision(
        {"proposal": {"text_class": "sfx"}}
    ).action == PRESERVE
    assert product_semantic_decision(
        {"proposal": {"text_class": "unknown"}}
    ).action == REVIEW
    explicit = product_semantic_decision(
        {
            "proposal": {
                "text_class": "unknown-but-explicit",
                "processing_action": TRANSLATE,
            }
        }
    )
    assert explicit.action == TRANSLATE
    invalid = product_semantic_decision(
        {"proposal": {"processing_action": "unknown-action"}}
    )
    assert invalid.action == REVIEW
    assert invalid.available is False


def test_g1_keeps_only_component_touching_pr6_and_its_connected_halo() -> None:
    shape = (48, 64)
    raw = _mask(shape, (10, 12, 14, 18), (40, 12, 44, 18))
    native3 = _mask(shape, (7, 9, 17, 21), (37, 9, 47, 21))
    baseline = _mask(shape, (10, 16, 14, 20))

    result = _run(
        fine_raw=raw,
        fine_native3=native3,
        provider="finetune",
        policy="g1",
        baseline=baseline,
    )

    assert np.count_nonzero(result.safe_addition[9:21, 7:17]) > 0
    assert np.count_nonzero(result.safe_addition[9:21, 37:47]) == 0
    assert {row.reason for row in result.component_records} == {
        "accepted",
        "policy_no_existing_context",
    }
    assert np.count_nonzero(result.connected_halo) > 0


def test_whole_component_is_rejected_on_one_protected_pixel() -> None:
    shape = (40, 40)
    raw = _mask(shape, (10, 10, 18, 18))
    native3 = _mask(shape, (7, 7, 21, 21))
    baseline = _mask(shape, (10, 16, 18, 20))
    structure = _mask(shape, (20, 20, 21, 21))

    result = _run(
        fine_raw=raw,
        fine_native3=native3,
        provider="finetune",
        policy="g1",
        baseline=baseline,
        structure=structure,
    )

    assert np.count_nonzero(result.safe_addition) == 0
    assert result.component_records[0].reason == "exact_structure_contact"


def test_g2_requires_both_raw_providers_for_disjoint_component() -> None:
    shape = (40, 56)
    fine = _mask(shape, (10, 10, 16, 18), (34, 10, 40, 18))
    tiled = _mask(shape, (11, 11, 17, 19))
    proposal = _mask(shape, (8, 8, 19, 21), (32, 8, 42, 21))

    result = _run(
        fine_raw=fine,
        fine_native3=proposal,
        tiled_raw=tiled,
        tiled_native3=_mask(shape, (9, 9, 19, 21)),
        policy="g2",
    )

    assert np.count_nonzero(result.safe_addition[8:21, 8:19]) > 0
    assert np.count_nonzero(result.safe_addition[8:21, 32:42]) == 0
    assert any(
        row.reason == "policy_no_dual_support" for row in result.component_records
    )


def test_g3_admits_single_provider_only_with_source_raw_owned_seed() -> None:
    shape = (36, 48)
    fine = _mask(shape, (24, 10, 31, 18))
    source_seed = _mask(shape, (26, 12, 28, 16))

    rejected = _run(fine_raw=fine, policy="g3")
    accepted = _run(fine_raw=fine, policy="g3", source_seed=source_seed)

    assert np.count_nonzero(rejected.safe_addition) == 0
    assert np.count_nonzero(accepted.safe_addition) == np.count_nonzero(fine)
    assert accepted.component_records[0].touches_source_raw_owned is True


def test_overlapping_authoritative_regions_fail_closed() -> None:
    shape = (36, 48)
    fine = _mask(shape, (18, 10, 25, 18))
    regions = (
        _region(shape, "left", (0, 0, 24, 36)),
        _region(shape, "right", (20, 0, 48, 36)),
    )
    baseline = _mask(shape, (18, 16, 25, 20))

    result = _run(
        fine_raw=fine,
        provider="finetune",
        policy="g1",
        regions=regions,
        baseline=baseline,
    )

    assert np.count_nonzero(result.safe_addition) == 0
    assert result.component_records[0].reason == "ownership_conflict"


def test_preserve_and_abstain_regions_never_create_additions() -> None:
    shape = (32, 64)
    fine = _mask(shape, (8, 8, 16, 16), (44, 8, 52, 16))
    regions = (
        _region(shape, "preserve", (0, 0, 30, 32), PRESERVE),
        _region(shape, "review", (34, 0, 64, 32), REVIEW),
    )
    baseline = fine.copy()

    result = _run(
        fine_raw=fine,
        provider="finetune",
        policy="g1",
        regions=regions,
        baseline=baseline,
    )

    assert np.count_nonzero(result.safe_addition) == 0
    assert {row.reason for row in result.component_records} == {
        "semantic_preserve",
        "semantic_abstain",
    }


def test_stage2_rebuilds_preserve_and_abstain_from_region_actions() -> None:
    shape = (20, 30)
    masks = SimpleNamespace(
        regions=(
            SimpleNamespace(
                region_id="translated",
                ownership=_mask(shape, (0, 0, 10, 20)),
            ),
            SimpleNamespace(
                region_id="sound-effect",
                ownership=_mask(shape, (10, 0, 20, 20)),
            ),
            SimpleNamespace(
                region_id="unknown",
                ownership=_mask(shape, (20, 0, 30, 20)),
            ),
        )
    )
    entry = {
        "regions": [
            {
                "region_id": "translated",
                "proposal": {
                    "text_class": "sfx",
                    "processing_action": TRANSLATE,
                },
            },
            {
                "region_id": "sound-effect",
                "proposal": {"text_class": "onomatopoeia"},
            },
            {
                "region_id": "unknown",
                "proposal": {"text_class": "unknown"},
            },
        ]
    }

    preserve, abstain = _runtime_action_masks(entry, masks, shape)

    assert np.count_nonzero(preserve[:, :10]) == 0
    assert np.count_nonzero(preserve[:, 10:20]) == 200
    assert np.count_nonzero(preserve[:, 20:]) == 0
    assert np.count_nonzero(abstain[:, :20]) == 0
    assert np.count_nonzero(abstain[:, 20:]) == 200


def test_refinement_stays_linear_enough_for_many_4k_components() -> None:
    shape = (4096, 4096)
    raw = np.zeros(shape, dtype=np.uint8)
    for y in range(32, 4064, 128):
        for x in range(32, 4064, 128):
            raw[y : y + 3, x : x + 3] = 255
    baseline = raw.copy()
    started = time.perf_counter()

    result = _run(
        fine_raw=raw,
        provider="finetune",
        expansion="raw_core",
        policy="g1",
        baseline=baseline,
    )

    assert len(result.component_records) == 1024
    assert np.count_nonzero(result.safe_addition) == 0
    assert time.perf_counter() - started < 5.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_stage1_fixture(
    root: Path,
    *,
    manifest_sha256: str,
    selected_variant: str,
) -> Path:
    page_id = "neutral-page"
    records: list[dict[str, object]] = []
    identities: dict[str, dict[str, object]] = {}
    mask = np.zeros((12, 16), dtype=np.uint8)
    mask[3:7, 5:9] = 255
    for variant in ("raw", "refined", "dilated"):
        path = root / "native_masks" / variant / f"{page_id}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        assert cv2.imwrite(str(path), mask)
        identity = {
            "page_id": page_id,
            "binary_mask_sha256": mask_sha256(mask),
            "pixel_count": int(np.count_nonzero(mask)),
        }
        identities[variant] = {
            "output_mask_set_sha256": _canonical_sha256([identity]),
            "page_count": 1,
        }
        records.append(
            {
                "page_id": page_id,
                "role": "native_detector_mask",
                "variant": variant,
                "relative_path": path.relative_to(root).as_posix(),
                "artifact_sha256": _sha256(path),
                **identity,
            }
        )
    positive_path = root / "positive_edit_masks" / f"{page_id}_positive_edit.png"
    positive_path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(positive_path), mask)
    positive_identity = {
        "page_id": page_id,
        "binary_mask_sha256": mask_sha256(mask),
        "pixel_count": int(np.count_nonzero(mask)),
    }
    records.append(
        {
            "page_id": page_id,
            "role": "positive_edit_mask",
            "variant": selected_variant,
            "relative_path": positive_path.relative_to(root).as_posix(),
            "artifact_sha256": _sha256(positive_path),
            **positive_identity,
        }
    )
    records.sort(key=lambda row: (row["role"], row["variant"], row["page_id"]))
    inventory = {
        "schema_version": "inpaint-detector-output-artifact-inventory-v1",
        "records": records,
        "inventory_sha256": _canonical_sha256(records),
    }
    inventory_path = root / "output-artifact-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    result = {
        "schema_version": "inpaint-detector-bakeoff-stage1-v1",
        "candidate": "neutral-detector",
        "variant": selected_variant,
        "manifest_sha256": manifest_sha256,
        "summary": {"page_count": 1},
        "pages": [{"page_id": page_id}],
        "variant_output_identity": identities,
        "positive_edit_output_identity": {
            "output_mask_set_sha256": _canonical_sha256([positive_identity]),
            "page_count": 1,
        },
        "output_artifact_inventory": {
            "relative_path": inventory_path.name,
            "artifact_sha256": _sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "artifact_count": len(records),
        },
    }
    (root / "stage1-results.json").write_text(
        json.dumps(result), encoding="utf-8"
    )
    return root


def test_stage1_input_reopens_detector_output_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    manifest_sha = "a" * 64
    run_root = _write_stage1_fixture(
        tmp_path / "run",
        manifest_sha256=manifest_sha,
        selected_variant="raw",
    )
    accepted = _stage1_input(
        run_root,
        expected_variant="raw",
        expected_source_manifest_sha256=manifest_sha,
        page_ids=("neutral-page",),
    )
    assert accepted["variant"] == "raw"

    path = run_root / "native_masks" / "raw" / "neutral-page.png"
    changed = np.full((12, 16), 255, dtype=np.uint8)
    assert cv2.imwrite(str(path), changed)
    with pytest.raises(ValueError, match="mask file SHA differs"):
        _stage1_input(
            run_root,
            expected_variant="raw",
            expected_source_manifest_sha256=manifest_sha,
            page_ids=("neutral-page",),
        )


def test_policy_overlay_is_source_only_and_optional_neutral(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "pages": [
                    {
                        "page_id": "neutral-page",
                        "target_instances": [
                            {
                                "instance_id": "required-text",
                                "priority": "required",
                                "processing_action": "translate_inpaint",
                                "semantic_role": "dialogue_bubble",
                            },
                            {
                                "instance_id": "optional-mark",
                                "priority": "optional",
                                "processing_action": "preserve",
                                "semantic_role": "decorative",
                            },
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    binding = {
        "manifest_sha256": "b" * 64,
        "page_inventory_sha256": "c" * 64,
        "page_count": 1,
    }
    monkeypatch.setattr(
        policy_overlay,
        "validate_source_only_manifest_v4",
        lambda _path: binding,
    )
    payload = policy_overlay.build_policy_overlay(manifest)
    assert payload["instance_counts"] == {
        "required_translate": 1,
        "optional_neutral": 1,
        "hard_ambiguous": 0,
    }
    assert payload["candidate_seen"] is False

    overlay_path = tmp_path / "overlay.json"
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    assert policy_overlay.validate_policy_overlay(
        overlay_path, manifest_path=manifest
    ) == payload
    required_row = next(
        row
        for row in payload["instances"]
        if row["instance_id"] == "required-text"
    )
    required_row["evaluation_class"] = "optional_neutral"
    overlay_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="differs from sealed source policy"):
        policy_overlay.validate_policy_overlay(
            overlay_path, manifest_path=manifest
        )


def test_stage2_reopens_selected_safe_addition_and_rejects_tamper(
    tmp_path: Path,
) -> None:
    page_id = "neutral-page"
    candidate_id = "g1_or_raw_core"
    root = tmp_path / "mask-only"
    mask_path = root / "runs" / candidate_id / "safe_additions" / f"{page_id}.png"
    mask_path.parent.mkdir(parents=True)
    mask = _mask((20, 24), (6, 5, 12, 11))
    assert cv2.imwrite(str(mask_path), mask)
    record = {
        "candidate_id": candidate_id,
        "page_id": page_id,
        "role": "safe_addition",
        "relative_path": mask_path.relative_to(root).as_posix(),
        "file_sha256": _sha256(mask_path),
        "pixel_sha256": mask_sha256(mask),
        "pixel_count": int(np.count_nonzero(mask)),
    }
    inventory_body = {
        "schema_version": "inpaint-proposal-refinement-output-inventory-v33",
        "source_manifest_sha256": "a" * 64,
        "relative_manifest_sha256": "b" * 64,
        "policy_overlay_sha256": "c" * 64,
        "candidate_ids": [candidate_id],
        "page_ids": [page_id],
        "artifacts": [record],
    }
    inventory = {
        **inventory_body,
        "inventory_sha256": _canonical_sha256(inventory_body),
    }
    inventory_path = root / "output-artifact-inventory.json"
    inventory_path.write_text(json.dumps(inventory), encoding="utf-8")
    result = {
        "schema_version": "inpaint-proposal-refinement-results-v33",
        "pages": {
            candidate_id: [
                {
                    "page_id": page_id,
                    "output_safe_addition_pixel_sha256": mask_sha256(mask),
                }
            ]
        },
        "output_inventory": {
            "relative_path": inventory_path.name,
            "artifact_sha256": _sha256(inventory_path),
            "inventory_sha256": inventory["inventory_sha256"],
            "artifact_count": 1,
        },
    }
    result_path = root / "proposal-refinement-results.json"
    result_path.write_text(json.dumps(result), encoding="utf-8")

    paths = _validated_addition_paths(
        mask_only_result_path=result_path,
        mask_only=result,
        candidate_id=candidate_id,
        page_ids={page_id},
    )
    assert paths == {page_id: mask_path.resolve()}

    assert cv2.imwrite(str(mask_path), np.full(mask.shape, 255, dtype=np.uint8))
    with pytest.raises(ValueError, match="safe addition file SHA differs"):
        _validated_addition_paths(
            mask_only_result_path=result_path,
            mask_only=result,
            candidate_id=candidate_id,
            page_ids={page_id},
        )
