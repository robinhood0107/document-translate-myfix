from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (
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
from scripts.build_inpaint_method_closure_v4 import build_closure
from scripts.update_inpaint_method_evidence_v4 import update_evidence


ROOT = Path(__file__).resolve().parents[1]
SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _write_json(path: Path, payload: object) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def _scope_manifest(tmp_path: Path, *, corpus: str = "e1") -> Path:
    return _write_json(
        tmp_path / f"{corpus}-manifest.json",
        {
            "schema_version": "inpaint-factorized-source-manifest-v4",
            "corpus_id": corpus,
            "split_role": "development_source_only",
            "annotation_frozen_before_candidate": True,
            "pages": [{
                "page_id": "p",
                "target_extent_independent": True,
                "target_inventory_independent": True,
                "target_review_complete": True,
            }],
        },
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _summary(*, passing: bool = False) -> dict[str, object]:
    coverage = 1.0 if passing else 0.9
    return {
        "page_count": 1,
        "target_instance_count": 1,
        "aggregate_target_coverage": coverage,
        "minimum_component_coverage": coverage,
        "minimum_target_instance_edit_coverage": coverage,
        "target_instance_seed_recall": 1.0 if passing else 0.0,
        "protected_edit_overlap": 0,
        "ambiguous_edit_overlap": 0,
        "ownership_leak_pixel_count": 0,
        "preserve_edit_overlap": 0,
        "false_edit_pixel_count": 0,
        "missed_target_instance_count": 0 if passing else 1,
    }


def _factorized_artifact(
    tmp_path: Path,
    manifest: Path,
    *,
    detector: str = "current_ctd_raw",
    status: str = "dominated",
    oracle_only: bool = False,
) -> Path:
    run_id = "run"
    return _write_json(
        tmp_path / "factorized.json",
        {
            "schema_version": "inpaint-factorized-results-v3",
            "manifest_sha256": _sha(manifest),
            "logical_combination_count": 1,
            "physical_combination_count": 1,
            "combination_count": 1,
            "closure_ledger": [
                {
                    "logical_id": run_id,
                    "selection": {"detector": detector},
                    "closure_state": "executed",
                    "reason": "",
                    "content_sha256": SHA_C,
                    "reused_from": "",
                }
            ],
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
                    "metrics": {"page_count": 1},
                    "closure_reason": "hard_gate_failed" if status == "dominated" else "",
                }
            ],
            "pages": {run_id: [{"page_id": "p"}]},
        },
    )


def _fusion_artifact(tmp_path: Path, manifest: Path) -> Path:
    output_sha = "d" * 64
    return _write_json(
        tmp_path / "fusion.json",
        {
            "schema_version": "inpaint-detector-fusion-results-v4",
            "manifest_sha256": _sha(manifest),
            "logical_combination_count": 1,
            "physical_output_count": 1,
            "unaccounted_combination_count": 0,
            "page_ids": ["p"],
            "closure_ledger": [
                {
                    "logical_id": "fusion-run",
                    "selection": {"fusion": "single"},
                    "closure_state": "executed",
                    "reason": "",
                    "content_sha256": output_sha,
                    "reused_from": "",
                }
            ],
            "runs": [
                {
                    "run_id": "fusion-run",
                    "fusion": "single",
                    "primary": "manga109_text",
                    "secondary": "",
                    "trigger": "",
                    "oracle_only": False,
                    "status": "dominated",
                    "closure_reason": "hard_gate_failed",
                    "metrics": {"output_mask_set_sha256": output_sha},
                }
            ],
        },
    )


def _stage1_artifact(
    tmp_path: Path,
    manifest: Path,
    *,
    candidate: str = "ctd-synthetic-low-contrast-finetune-v4",
    variant: str = "raw",
) -> Path:
    return _write_json(
        tmp_path / f"stage1-{candidate}-{variant}.json",
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
            "summary": _summary(),
            "pages": [{"page_id": "p"}],
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
    )

    assert rows[0]["disposition"] == "dominated"
    assert rows[0]["reason"] == "hard_gate_failed"
    assert len(str(rows[0]["content_sha256"])) == 64
    assert rows[0]["content_identity_kind"] == "artifact_record"


def test_artifact_rejects_unregistered_or_unproved_variant(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    with pytest.raises(ValueError, match="not registered"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"native3"}), evaluation_scope="e1"
        )
    with pytest.raises(ValueError, match="not declared"):
        accounted_evidence_from_artifact(
            _requirements(variants=("raw", "refined")), artifact_path=artifact,
            scope_manifest_path=manifest, family_id="current-ctd",
            variant_ids=frozenset({"refined"}), evaluation_scope="e1"
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
        )
    payload = json.loads(_factorized_artifact(tmp_path, manifest).read_text(encoding="utf-8"))
    payload["pages"] = {}
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="page results"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
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
        )


def test_minimal_factorized_metrics_cannot_claim_pareto(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest, status="pareto")
    with pytest.raises(ValueError, match="not proved by fail-closed metrics"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
        )


def test_artifact_must_cover_every_scope_page_not_a_fake_one_page_subset(
    tmp_path: Path,
) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"].append(
        {
            "page_id": "q",
            "target_extent_independent": True,
            "target_inventory_independent": True,
            "target_review_complete": True,
        }
    )
    _write_json(manifest, payload)
    artifact = _factorized_artifact(tmp_path, manifest)
    with pytest.raises(ValueError, match="page IDs differ"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
        )


def test_artifact_page_identity_must_exactly_match_scope(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _factorized_artifact(tmp_path, manifest)
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["pages"]["run"][0]["page_id"] = "other"
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="page IDs differ"):
        accounted_evidence_from_artifact(
            _requirements(), artifact_path=artifact, scope_manifest_path=manifest,
            family_id="current-ctd", variant_ids=frozenset({"raw"}),
            evaluation_scope="e1"
        )


def test_fusion_uses_exact_output_identity_and_validates_closure(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _fusion_artifact(tmp_path, manifest)
    requirements = (MethodVariantRequirement("manga109-text", "seed", "raw", "e1"),)
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id="manga109-text", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
    )
    assert rows[0]["content_identity_kind"] == "exact_output"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["runs"][0]["metrics"]["output_mask_set_sha256"] = SHA_A
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="output SHA differs"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="manga109-text", variant_ids=frozenset({"raw"}), evaluation_scope="e1"
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


def test_source_protection_requires_full_page_and_summary_contract(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / "protection.json",
        {
            "schema_version": "inpaint-source-protection-reapply-v3",
            "manifest_sha256": _sha(manifest),
            "candidate_id": "c19-accepted-seed-final-protect",
            "summary": _summary(passing=True),
            "pages": [{"page_id": "p"}],
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
    with pytest.raises(ValueError, match="page count"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="exact-protection", variant_ids=frozenset({"C19"}),
            evaluation_scope="e1"
        )


def test_semantic_disposition_is_artifact_derived_and_blocked_needs_probe(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / "semantic.json",
        {
            "schema_version": "inpaint-semantic-policy-results-v4",
            "manifest_sha256": _sha(manifest),
            "policy_count": 1,
            "unaccounted_policy_count": 0,
            "page_ids": ["p"],
            "policies": [{"policy_id": "current_default", "oracle_only": False,
                          "status": "dominated", "closure_reason": "semantic_hard_gate_failed",
                          "metrics": {
                              "instance_count": 1,
                              "role_exact_accuracy": 0.0,
                              "action_exact_accuracy": 0.0,
                              "required_instance_count": 1,
                              "required_translate_recall": 0.0,
                              "preserve_instance_count": 0,
                              "preserve_destructive_count": 0,
                              "ambiguous_instance_count": 0,
                              "ambiguous_destructive_count": 0,
                              "unavailable_instance_count": 0,
                          }}],
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
    with pytest.raises(ValueError, match="non-executed|hashed asset probe"):
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
    with pytest.raises(ValueError, match="semantic metrics require"):
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
            "asset_id": "official-sickzil-checkpoint",
            "reason_code": "official_asset_unavailable",
            "checks": [{"kind": "filesystem", "found": False}],
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


@pytest.mark.parametrize(
    ("family_id", "role", "variant_id"),
    (
        ("ownership", "ownership", "rtdetr_pixel"),
        ("ownership", "ownership", "c13_reconciliation"),
        ("exact-protection", "protection", "pr4_exact"),
        ("exact-composite", "composite", "immutable_original_exact_mask"),
    ),
)
def test_generic_role_result_closes_only_full_source_bound_evidence(
    tmp_path: Path, family_id: str, role: str, variant_id: str
) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / f"role-{variant_id}.json",
        {
            "schema_version": "inpaint-method-role-results-v4",
            "manifest_sha256": _sha(manifest),
            "source_artifact_sha256": SHA_A,
            "source_artifact_schema_version": "source-result-v1",
            "record_count": 1,
            "page_count": 1,
            "unaccounted_record_count": 0,
            "pages": [{"page_id": "p"}],
            "records": [
                {
                    "family_id": family_id,
                    "variant_id": variant_id,
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
    requirements = (MethodVariantRequirement(family_id, role, variant_id, "e1"),)
    rows = accounted_evidence_from_artifact(
        requirements, artifact_path=artifact, scope_manifest_path=manifest,
        family_id=family_id, variant_ids=frozenset({variant_id}), evaluation_scope="e1"
    )
    assert rows[0]["disposition"] == "family_complete"
    assert rows[0]["content_identity_kind"] == "artifact_record"
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["records"][0]["page_ids"] = []
    _write_json(artifact, payload)
    with pytest.raises(ValueError, match="full page inventory"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id=family_id, variant_ids=frozenset({variant_id}), evaluation_scope="e1"
        )


def test_generic_role_result_rejects_status_not_proved_by_metrics(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    artifact = _write_json(
        tmp_path / "role-invalid.json",
        {
            "schema_version": "inpaint-method-role-results-v4",
            "manifest_sha256": _sha(manifest),
            "source_artifact_sha256": SHA_A,
            "source_artifact_schema_version": "source-result-v1",
            "record_count": 1,
            "page_count": 1,
            "unaccounted_record_count": 0,
            "pages": [{"page_id": "p"}],
            "records": [{"family_id": "ownership", "variant_id": "rtdetr_pixel",
                         "source_result_id": "source-result", "content_sha256": SHA_B,
                         "page_ids": ["p"], "oracle_only": False,
                         "status": "pareto", "closure_reason": "",
                         "metrics": _summary()}],
        },
    )
    requirements = (MethodVariantRequirement("ownership", "ownership", "rtdetr_pixel", "e1"),)
    with pytest.raises(ValueError, match="status differs|oracle-only"):
        accounted_evidence_from_artifact(
            requirements, artifact_path=artifact, scope_manifest_path=manifest,
            family_id="ownership", variant_ids=frozenset({"rtdetr_pixel"}), evaluation_scope="e1"
        )


def test_registry_has_an_honest_adapter_for_every_registered_requirement() -> None:
    registry = json.loads(
        (ROOT / "benchmarking" / "inpaint_detector_bakeoff" / "method_registry_v4.json").read_text(encoding="utf-8")
    )
    gaps = registry_evidence_adapter_gaps(requirements_from_registry(registry))
    assert gaps == ()


def test_scope_manifest_requires_canonical_identity(tmp_path: Path) -> None:
    path = _write_json(tmp_path / "bad.json", {"sealed": True})
    with pytest.raises(ValueError, match="source-only"):
        scope_manifest_binding(path)


def test_scope_manifest_rejects_candidate_derived_annotations(tmp_path: Path) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"][0]["target_inventory_independent"] = False
    _write_json(manifest, payload)
    with pytest.raises(ValueError, match="candidate-derived"):
        scope_manifest_binding(manifest)


@pytest.mark.parametrize("pages", ([], [{"page_id": "p"}, {"page_id": "p"}]))
def test_scope_manifest_rejects_empty_or_duplicate_page_inventory(
    tmp_path: Path, pages: list[dict[str, object]]
) -> None:
    manifest = _scope_manifest(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["pages"] = pages
    _write_json(manifest, payload)
    with pytest.raises(ValueError, match="non-empty|duplicate"):
        scope_manifest_binding(manifest)
