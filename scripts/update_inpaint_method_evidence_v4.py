#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (  # noqa: E402
    blocked_asset_evidence,
    evidence_rows_from_artifact,
    invalid_with_reason_evidence,
    merge_method_evidence,
    merge_scope_manifest_binding,
    scope_manifest_binding,
)
from benchmarking.inpaint_detector_bakeoff.method_closure import (  # noqa: E402
    evidence_from_records,
    requirements_from_registry,
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _registered_fusion_candidate_ids(
    registry: dict[str, object],
) -> frozenset[str] | None:
    families = registry.get("families")
    if not isinstance(families, list):
        raise ValueError("method family registry requires families")
    matches = [
        row
        for row in families
        if isinstance(row, dict) and row.get("family_id") == "detector-fusion"
    ]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("method family registry duplicates detector-fusion")
    values = matches[0].get("candidate_ids")
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(
            "detector-fusion registry requires canonical candidate_ids"
        )
    if len(values) != len(set(values)):
        raise ValueError("detector-fusion registry contains duplicate candidate_ids")
    return frozenset(values)


def update_evidence(
    *,
    registry_path: Path,
    evidence_path: Path,
    artifact_path: Path | None,
    scope_manifest_path: Path,
    family_id: str,
    variant_ids: frozenset[str],
    evaluation_scope: str,
    upstream_contract_path: Path | None = None,
    blocker_probe_path: Path | None = None,
    invalid_reason: str | None = None,
    invalid_parent_record_kind: str | None = None,
    invalid_parent_record_id: str | None = None,
    allow_replace: bool = False,
) -> dict[str, object]:
    registry = _read_json(registry_path)
    if registry.get("schema_version") != "inpaint-method-family-registry-v4":
        raise ValueError("unsupported method family registry schema")
    existing_payload = _read_json(evidence_path) if evidence_path.exists() else {}
    if evidence_path.exists() and existing_payload.get("schema_version") != (
        "inpaint-method-family-evidence-v4"
    ):
        raise ValueError("unsupported existing method family evidence schema")
    existing = existing_payload.get("evidence", [])
    if not isinstance(existing, list) or any(not isinstance(row, dict) for row in existing):
        raise ValueError("method evidence must contain an evidence object list")
    requirements = requirements_from_registry(registry)
    existing_bindings = existing_payload.get("scope_manifests", {})
    if not isinstance(existing_bindings, dict):
        raise ValueError("method evidence must contain scope_manifests bindings")
    existing_rows = evidence_from_records(existing)
    for row in existing_rows:
        binding = existing_bindings.get(row.evaluation_scope)
        if not isinstance(binding, dict) or str(binding.get("sha256") or "") != (
            row.scope_manifest_sha256
        ):
            raise ValueError(
                "existing method evidence differs from its canonical scope binding: "
                + row.evidence_key
            )
    bindings = merge_scope_manifest_binding(
        existing_bindings,
        evaluation_scope=evaluation_scope,
        binding=scope_manifest_binding(scope_manifest_path),
    )
    if blocker_probe_path is not None:
        if artifact_path is not None:
            raise ValueError("blocked asset evidence cannot also declare an artifact")
        if invalid_reason is not None:
            raise ValueError("blocked asset and invalid evidence are mutually exclusive")
        updates = blocked_asset_evidence(
            requirements,
            scope_manifest_path=scope_manifest_path,
            blocker_probe_path=blocker_probe_path,
            family_id=family_id,
            variant_ids=variant_ids,
            evaluation_scope=evaluation_scope,
        )
    elif invalid_reason is not None:
        if artifact_path is None:
            raise ValueError("invalid evidence requires a parent artifact")
        if not invalid_parent_record_kind or not invalid_parent_record_id:
            raise ValueError("invalid evidence requires parent record kind and id")
        updates = invalid_with_reason_evidence(
            requirements,
            scope_manifest_path=scope_manifest_path,
            parent_artifact_path=artifact_path,
            parent_record_kind=invalid_parent_record_kind,
            parent_record_id=invalid_parent_record_id,
            reason_code=invalid_reason,
            family_id=family_id,
            variant_ids=variant_ids,
            evaluation_scope=evaluation_scope,
        )
    else:
        if artifact_path is None:
            raise ValueError("executed evidence requires an artifact")
        updates = evidence_rows_from_artifact(
            requirements,
            artifact_path=artifact_path,
            scope_manifest_path=scope_manifest_path,
            family_id=family_id,
            variant_ids=variant_ids,
            evaluation_scope=evaluation_scope,
            upstream_contract_path=upstream_contract_path,
            expected_fusion_candidate_ids=_registered_fusion_candidate_ids(
                registry
            ),
        )
    if not updates:
        raise ValueError("no registered method variants match the requested artifact scope")
    return {
        "schema_version": "inpaint-method-family-evidence-v4",
        "scope_manifests": bindings,
        "evidence": list(
            merge_method_evidence(existing, updates, allow_replace=allow_replace)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Bind an executed private result artifact to registered method variants."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument(
        "--scope-manifest",
        type=Path,
        required=True,
        help="Exact sealed manifest whose byte SHA the artifact declares.",
    )
    parser.add_argument("--family", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="Explicit registered variant proved by this artifact; repeat as needed.",
    )
    parser.add_argument("--scope", required=True)
    parser.add_argument(
        "--upstream-contract",
        type=Path,
        help=(
            "Exact matrix/spec file whose byte SHA and full logical inventory "
            "the factorized/fusion artifact declares."
        ),
    )
    parser.add_argument(
        "--blocked-asset-probe",
        type=Path,
        help=(
            "Hashed probe JSON that records concrete checks for an unavailable asset."
        ),
    )
    parser.add_argument(
        "--invalid-reason",
        choices=(
            "upstream_seed_not_admitted",
            "upstream_semantic_gate_failed",
            "upstream_product_mask_gate_failed",
            "oracle_only_not_product",
            "combination_incompatible",
        ),
        help="Stable reason code for a stage-gated invalid requirement.",
    )
    parser.add_argument(
        "--invalid-parent-kind",
        choices=("run", "policy", "combination"),
        help="Record collection in the parent artifact used by invalid evidence.",
    )
    parser.add_argument(
        "--invalid-parent-id",
        help="Exact parent run/policy/combination id whose gate facts are bound.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Explicitly replace an existing family/role/variant/scope proof.",
    )
    args = parser.parse_args(argv)
    evidence_path = args.evidence.resolve()
    payload = update_evidence(
        registry_path=args.registry.resolve(),
        evidence_path=evidence_path,
        artifact_path=(args.artifact.resolve() if args.artifact else None),
        scope_manifest_path=args.scope_manifest.resolve(),
        family_id=str(args.family),
        variant_ids=frozenset(str(value) for value in args.variant),
        evaluation_scope=str(args.scope),
        upstream_contract_path=(
            args.upstream_contract.resolve() if args.upstream_contract else None
        ),
        blocker_probe_path=(
            args.blocked_asset_probe.resolve() if args.blocked_asset_probe else None
        ),
        invalid_reason=(str(args.invalid_reason) if args.invalid_reason else None),
        invalid_parent_record_kind=(
            str(args.invalid_parent_kind) if args.invalid_parent_kind else None
        ),
        invalid_parent_record_id=(
            str(args.invalid_parent_id) if args.invalid_parent_id else None
        ),
        allow_replace=bool(args.replace),
    )
    temporary = evidence_path.with_name(f".{evidence_path.name}.partial")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(evidence_path)
    print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
