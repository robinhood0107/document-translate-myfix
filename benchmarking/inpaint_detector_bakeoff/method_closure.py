from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from .contracts import COMBINATION_CLOSURE_STATES, ROLE_NAMES, ROLE_STATES


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
    reused_from: str = ""

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
        if self.closure_state in {"executed", "reused_by_sha"} and not self.artifact_sha256:
            raise ValueError(f"{self.closure_state} requires an artifact SHA")
        if self.artifact_sha256 and (
            len(self.artifact_sha256) != 64
            or self.artifact_sha256 != self.artifact_sha256.lower()
            or any(character not in "0123456789abcdef" for character in self.artifact_sha256.lower())
        ):
            raise ValueError("artifact SHA must be a lowercase 64-character hexadecimal digest")
        if self.closure_state == "reused_by_sha" and not self.reused_from:
            raise ValueError("reused_by_sha requires a source evidence key")

    @property
    def key(self) -> tuple[str, str, str, str]:
        return self.family_id, self.role, self.variant_id, self.evaluation_scope

    @property
    def evidence_key(self) -> str:
        return "/".join(self.key)


def build_method_family_closure(
    requirements: Iterable[MethodVariantRequirement],
    evidence: Iterable[MethodVariantEvidence],
) -> dict[str, object]:
    required = tuple(requirements)
    observed = tuple(evidence)
    required_by_key = {row.key: row for row in required}
    observed_by_key = {row.key: row for row in observed}
    if len(required_by_key) != len(required):
        raise ValueError("duplicate method requirement")
    if len(observed_by_key) != len(observed):
        raise ValueError("duplicate method evidence")
    unexpected = sorted(set(observed_by_key) - set(required_by_key))
    if unexpected:
        raise ValueError(f"method evidence is not registered: {unexpected[0]}")
    observed_by_evidence_key = {row.evidence_key: row for row in observed}
    for row in observed:
        if row.closure_state != "reused_by_sha":
            continue
        source = observed_by_evidence_key.get(row.reused_from)
        if source is None:
            raise ValueError(f"reused evidence source is not registered: {row.reused_from}")
        if source.closure_state != "executed":
            raise ValueError("reused evidence source must be an executed variant")
        if source.artifact_sha256 != row.artifact_sha256:
            raise ValueError("reused evidence SHA differs from its source variant")

    families: dict[tuple[str, str], list[MethodVariantRequirement]] = {}
    for row in required:
        families.setdefault((row.family_id, row.role), []).append(row)

    records: list[dict[str, object]] = []
    unaccounted_total = 0
    for (family_id, role), variants in sorted(families.items()):
        variant_records: list[dict[str, object]] = []
        missing: list[str] = []
        dispositions: set[str] = set()
        for requirement in sorted(
            variants, key=lambda value: (value.evaluation_scope, value.variant_id)
        ):
            row = observed_by_key.get(requirement.key)
            if row is None:
                missing.append(requirement.variant_id)
                variant_records.append(
                    {
                        "variant_id": requirement.variant_id,
                        "evaluation_scope": requirement.evaluation_scope,
                        "closure_state": "unaccounted",
                        "disposition": "active",
                        "reason": "missing_evidence",
                        "artifact_sha256": "",
                    }
                )
                continue
            dispositions.add(row.disposition)
            variant_records.append(
                {
                    "variant_id": row.variant_id,
                    "evaluation_scope": row.evaluation_scope,
                    "closure_state": row.closure_state,
                    "disposition": row.disposition,
                    "reason": row.reason,
                    "artifact_sha256": row.artifact_sha256,
                    "reused_from": row.reused_from,
                }
            )
        unaccounted_total += len(missing)
        family_complete = not missing
        if not family_complete:
            family_status = "active"
        elif "pareto" in dispositions:
            family_status = "pareto"
        elif dispositions == {"blocked_asset"}:
            family_status = "blocked_asset"
        elif "information_limited" in dispositions:
            family_status = "information_limited"
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
                "missing_variants": missing,
                "variants": variant_records,
            }
        )

    return {
        "schema_version": "inpaint-method-family-closure-results-v4",
        "family_count": len(records),
        "required_variant_count": len(required),
        "accounted_variant_count": len(required) - unaccounted_total,
        "unaccounted_variant_count": unaccounted_total,
        "all_families_complete": unaccounted_total == 0,
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
            reused_from=str(row.get("reused_from") or ""),
        )
        for row in records
    )
