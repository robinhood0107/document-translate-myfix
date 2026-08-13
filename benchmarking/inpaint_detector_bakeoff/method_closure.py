from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import COMBINATION_CLOSURE_STATES, ROLE_NAMES, ROLE_STATES


_CONTENT_IDENTITY_KINDS = frozenset({"exact_output", "artifact_record"})


def _is_sha256(value: str) -> bool:
    return (
        len(value) == 64
        and value == value.lower()
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class MethodVariantRequirement:
    family_id: str
    role: str
    variant_id: str
    evaluation_scope: str

    def __post_init__(self) -> None:
        if (
            not self.family_id.strip()
            or not self.variant_id.strip()
            or not self.evaluation_scope.strip()
        ):
            raise ValueError("method requirement contains an empty id")
        if self.role not in ROLE_NAMES:
            raise ValueError(f"unknown method role: {self.role}")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.family_id, self.role, self.variant_id, self.evaluation_scope


@dataclass(frozen=True, slots=True)
class MethodVariantEvidence:
    family_id: str
    role: str
    variant_id: str
    evaluation_scope: str
    closure_state: str
    disposition: str
    reason: str = ""
    artifact_sha256: str = ""
    scope_manifest_sha256: str = ""
    content_sha256: str = ""
    content_identity_kind: str = ""
    reused_from: str = ""
    blocker_probe_sha256: str = ""

    def __post_init__(self) -> None:
        MethodVariantRequirement(
            self.family_id, self.role, self.variant_id, self.evaluation_scope
        )
        if self.closure_state not in COMBINATION_CLOSURE_STATES:
            raise ValueError(f"unknown method closure state: {self.closure_state}")
        if self.disposition not in ROLE_STATES:
            raise ValueError(f"unknown method disposition: {self.disposition}")
        if self.closure_state in {"invalid_with_reason", "blocked_asset"} and not self.reason:
            raise ValueError(f"{self.closure_state} requires a reason")
        if not _is_sha256(self.scope_manifest_sha256):
            raise ValueError(
                "scope manifest SHA must be a lowercase 64-character hexadecimal digest"
            )
        if self.artifact_sha256 and not _is_sha256(self.artifact_sha256):
            raise ValueError(
                "artifact SHA must be a lowercase 64-character hexadecimal digest"
            )
        if self.content_sha256 and not _is_sha256(self.content_sha256):
            raise ValueError(
                "content SHA must be a lowercase 64-character hexadecimal digest"
            )
        if self.blocker_probe_sha256 and not _is_sha256(self.blocker_probe_sha256):
            raise ValueError(
                "blocker probe SHA must be a lowercase 64-character hexadecimal digest"
            )

        if self.closure_state in {"executed", "reused_by_sha"}:
            if not self.artifact_sha256:
                raise ValueError(f"{self.closure_state} requires an artifact SHA")
            if not self.content_sha256:
                raise ValueError(f"{self.closure_state} requires a content SHA")
            if self.content_identity_kind not in _CONTENT_IDENTITY_KINDS:
                raise ValueError(
                    f"{self.closure_state} requires a supported content identity kind"
                )
            if self.blocker_probe_sha256:
                raise ValueError("executed evidence cannot carry a blocker probe")
        elif self.content_sha256 or self.content_identity_kind:
            raise ValueError("non-executed evidence cannot claim output content identity")

        if self.closure_state == "reused_by_sha":
            if not self.reused_from:
                raise ValueError("reused_by_sha requires a source evidence key")
            if self.content_identity_kind != "exact_output":
                raise ValueError("reused_by_sha requires exact output identity")
        elif self.reused_from:
            raise ValueError("only reused_by_sha evidence may declare reused_from")

        if self.closure_state == "blocked_asset":
            if self.disposition != "blocked_asset":
                raise ValueError("blocked_asset closure requires blocked_asset disposition")
            if not self.blocker_probe_sha256:
                raise ValueError("blocked_asset evidence requires a hashed blocker probe")
            if self.artifact_sha256:
                raise ValueError("blocked_asset evidence cannot claim an execution artifact")
        elif self.disposition == "blocked_asset":
            raise ValueError("blocked_asset disposition requires blocked_asset closure")
        if self.disposition == "pareto" and self.closure_state not in {
            "executed",
            "reused_by_sha",
        }:
            raise ValueError("pareto disposition requires executed output evidence")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.family_id, self.role, self.variant_id, self.evaluation_scope

    @property
    def evidence_key(self) -> str:
        return "/".join(self.key)


def _validate_scope_manifests(
    scope_manifests: Mapping[str, Mapping[str, object]],
    required_scopes: Iterable[str],
) -> dict[str, str]:
    if not isinstance(scope_manifests, Mapping):
        raise ValueError("method evidence requires canonical scope_manifests bindings")
    bindings: dict[str, str] = {}
    for scope, raw in scope_manifests.items():
        normalized_scope = str(scope)
        if not normalized_scope or not isinstance(raw, Mapping):
            raise ValueError("scope manifest binding contains an invalid scope")
        digest = str(raw.get("sha256") or "")
        if not _is_sha256(digest):
            raise ValueError(f"scope manifest binding has an invalid SHA: {normalized_scope}")
        if not str(raw.get("schema_version") or "").startswith("inpaint-"):
            raise ValueError(
                f"scope manifest binding has an unsupported schema: {normalized_scope}"
            )
        if not str(raw.get("corpus_id") or "").strip() or not str(
            raw.get("split_role") or ""
        ).strip():
            raise ValueError(
                f"scope manifest binding lacks corpus identity: {normalized_scope}"
            )
        bindings[normalized_scope] = digest
    missing = sorted(
        {str(scope) for scope in required_scopes} - set(bindings)
    )
    if missing:
        raise ValueError(f"scope manifest binding is missing: {missing[0]}")
    return bindings


def build_method_family_closure(
    requirements: Iterable[MethodVariantRequirement],
    evidence: Iterable[MethodVariantEvidence],
    *,
    scope_manifests: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    required = tuple(requirements)
    observed = tuple(evidence)
    required_by_key = {row.key: row for row in required}
    observed_by_key = {row.key: row for row in observed}
    if len(required_by_key) != len(required):
        raise ValueError("duplicate method requirement")
    if len(observed_by_key) != len(observed):
        raise ValueError("duplicate method evidence")
    scope_bindings = _validate_scope_manifests(
        scope_manifests, {row.evaluation_scope for row in observed}
    )
    unexpected = sorted(set(observed_by_key) - set(required_by_key))
    if unexpected:
        raise ValueError(f"method evidence is not registered: {unexpected[0]}")
    for row in observed:
        canonical = scope_bindings.get(row.evaluation_scope)
        if canonical != row.scope_manifest_sha256:
            raise ValueError(
                "method evidence scope manifest differs from its canonical binding: "
                + row.evidence_key
            )

    observed_by_evidence_key = {row.evidence_key: row for row in observed}
    for row in observed:
        if row.closure_state != "reused_by_sha":
            continue
        source = observed_by_evidence_key.get(row.reused_from)
        if source is None:
            raise ValueError(f"reused evidence source is not registered: {row.reused_from}")
        if source.closure_state != "executed":
            raise ValueError("reused evidence source must be an executed variant")
        if source.evaluation_scope != row.evaluation_scope:
            raise ValueError("reused evidence scope differs from its source variant")
        if source.scope_manifest_sha256 != row.scope_manifest_sha256:
            raise ValueError("reused evidence scope manifest differs from its source variant")
        if source.content_identity_kind != "exact_output":
            raise ValueError("reused evidence source lacks exact output identity")
        if source.content_sha256 != row.content_sha256:
            raise ValueError("reused evidence content SHA differs from its source variant")
        if source.disposition != row.disposition:
            raise ValueError("reused evidence disposition differs from its source variant")

    families: dict[tuple[str, str], list[MethodVariantRequirement]] = {}
    for row in required:
        families.setdefault((row.family_id, row.role), []).append(row)

    records: list[dict[str, object]] = []
    unaccounted_total = 0
    active_total = 0
    blocked_total = 0
    information_limited_total = 0
    executed_total = 0
    reused_total = 0
    for (family_id, role), variants in sorted(families.items()):
        variant_records: list[dict[str, object]] = []
        missing: list[str] = []
        disposition_counts = {state: 0 for state in sorted(ROLE_STATES)}
        closure_counts = {state: 0 for state in sorted(COMBINATION_CLOSURE_STATES)}
        for requirement in sorted(
            variants, key=lambda value: (value.evaluation_scope, value.variant_id)
        ):
            row = observed_by_key.get(requirement.key)
            if row is None:
                missing.append(requirement.variant_id)
                active_total += 1
                variant_records.append(
                    {
                        "variant_id": requirement.variant_id,
                        "evaluation_scope": requirement.evaluation_scope,
                        "closure_state": "unaccounted",
                        "disposition": "active",
                        "reason": "missing_evidence",
                        "artifact_sha256": "",
                        "content_sha256": "",
                        "content_identity_kind": "",
                    }
                )
                continue
            disposition_counts[row.disposition] += 1
            closure_counts[row.closure_state] += 1
            active_total += int(row.disposition == "active")
            blocked_total += int(row.disposition == "blocked_asset")
            information_limited_total += int(
                row.disposition == "information_limited"
            )
            executed_total += int(row.closure_state == "executed")
            reused_total += int(row.closure_state == "reused_by_sha")
            variant_records.append(
                {
                    "variant_id": row.variant_id,
                    "evaluation_scope": row.evaluation_scope,
                    "closure_state": row.closure_state,
                    "disposition": row.disposition,
                    "reason": row.reason,
                    "artifact_sha256": row.artifact_sha256,
                    "scope_manifest_sha256": row.scope_manifest_sha256,
                    "content_sha256": row.content_sha256,
                    "content_identity_kind": row.content_identity_kind,
                    "reused_from": row.reused_from,
                    "blocker_probe_sha256": row.blocker_probe_sha256,
                }
            )
        unaccounted_total += len(missing)
        family_active_count = disposition_counts["active"] + len(missing)
        family_complete = (
            not missing
            and family_active_count == 0
            and disposition_counts["blocked_asset"] == 0
            and disposition_counts["information_limited"] == 0
        )
        if family_active_count:
            family_status = "active"
        elif disposition_counts["pareto"]:
            family_status = "pareto"
        elif disposition_counts["information_limited"]:
            family_status = "information_limited"
        elif disposition_counts["blocked_asset"]:
            family_status = "blocked_asset"
        else:
            family_status = "family_complete"
        records.append(
            {
                "family_id": family_id,
                "role": role,
                "status": family_status,
                "family_complete": family_complete,
                "required_variant_count": len(variants),
                "accounted_variant_count": len(variants) - len(missing),
                "active_variant_count": family_active_count,
                "blocked_variant_count": disposition_counts["blocked_asset"],
                "disposition_counts": disposition_counts,
                "closure_state_counts": closure_counts,
                "missing_variants": sorted(set(missing)),
                "missing_requirements": [
                    {
                        "variant_id": requirement.variant_id,
                        "evaluation_scope": requirement.evaluation_scope,
                    }
                    for requirement in sorted(
                        variants,
                        key=lambda value: (value.evaluation_scope, value.variant_id),
                    )
                    if requirement.key not in observed_by_key
                ],
                "variants": variant_records,
            }
        )

    all_accounted = unaccounted_total == 0
    return {
        "schema_version": "inpaint-method-family-closure-results-v4",
        "family_count": len(records),
        "required_variant_count": len(required),
        "accounted_variant_count": len(required) - unaccounted_total,
        "unaccounted_variant_count": unaccounted_total,
        "active_variant_count": active_total,
        "blocked_variant_count": blocked_total,
        "information_limited_variant_count": information_limited_total,
        "executed_variant_count": executed_total,
        "reused_variant_count": reused_total,
        "all_requirements_accounted": all_accounted,
        "all_families_complete": all(
            bool(record["family_complete"]) for record in records
        ),
        "families": records,
    }


def requirements_from_registry(
    registry: Mapping[str, object],
) -> tuple[MethodVariantRequirement, ...]:
    families = registry.get("families")
    if not isinstance(families, list):
        raise ValueError("method registry must contain a families list")
    output: list[MethodVariantRequirement] = []
    for family in families:
        if not isinstance(family, Mapping):
            raise ValueError("method family must be an object")
        family_id = str(family.get("family_id") or "")
        role = str(family.get("role") or "")
        variants = family.get("variants")
        scopes = family.get("evaluation_scopes")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"method family {family_id} must list variants")
        if not isinstance(scopes, list) or not scopes:
            raise ValueError(f"method family {family_id} must list evaluation scopes")
        for scope in scopes:
            for variant in variants:
                output.append(
                    MethodVariantRequirement(family_id, role, str(variant), str(scope))
                )
    return tuple(output)


def evidence_from_records(
    records: Iterable[Mapping[str, object]],
) -> tuple[MethodVariantEvidence, ...]:
    return tuple(
        MethodVariantEvidence(
            family_id=str(row.get("family_id") or ""),
            role=str(row.get("role") or ""),
            variant_id=str(row.get("variant_id") or ""),
            evaluation_scope=str(row.get("evaluation_scope") or ""),
            closure_state=str(row.get("closure_state") or ""),
            disposition=str(row.get("disposition") or ""),
            reason=str(row.get("reason") or ""),
            artifact_sha256=str(row.get("artifact_sha256") or ""),
            scope_manifest_sha256=str(row.get("scope_manifest_sha256") or ""),
            content_sha256=str(row.get("content_sha256") or ""),
            content_identity_kind=str(row.get("content_identity_kind") or ""),
            reused_from=str(row.get("reused_from") or ""),
            blocker_probe_sha256=str(row.get("blocker_probe_sha256") or ""),
        )
        for row in records
    )
