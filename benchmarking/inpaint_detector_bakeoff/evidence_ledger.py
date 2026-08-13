from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

from .contracts import COMBINATION_CLOSURE_STATES, ROLE_STATES
from .method_closure import MethodVariantRequirement
from .stage2 import _hard_gate_passes as _factorized_hard_gate_passes
from .stage2 import select_pareto_records
from .contracts import FactorizedRunRecord


ArtifactVariantExtractor = Callable[[Mapping[str, object], str], frozenset[str]]


@dataclass(frozen=True, slots=True)
class ArtifactVariantFact:
    disposition: str
    reason: str
    content_sha256: str
    content_identity_kind: str


def _is_sha256(value: object) -> bool:
    text = str(value or "")
    return (
        len(text) == 64
        and text == text.lower()
        and all(character in "0123456789abcdef" for character in text)
    )


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _required_int(payload: Mapping[str, object], field: str) -> int:
    value = payload.get(field)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"evidence artifact requires non-negative integer {field}")
    return value


def _require_unique(rows: Sequence[Mapping[str, object]], field: str, label: str) -> None:
    values = [str(row.get(field) or "") for row in rows]
    if any(not value for value in values):
        raise ValueError(f"{label} contains an empty {field}")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate {field}")


def _object_rows(payload: Mapping[str, object], field: str, label: str) -> list[Mapping[str, object]]:
    value = payload.get(field)
    if not isinstance(value, list) or any(not isinstance(row, Mapping) for row in value):
        raise ValueError(f"{label} must contain an object list in {field}")
    return list(value)


def _page_id_set(rows: Sequence[Mapping[str, object]], label: str) -> frozenset[str]:
    if not rows:
        raise ValueError(f"{label} requires a non-empty page inventory")
    _require_unique(rows, "page_id", label)
    return frozenset(str(row["page_id"]) for row in rows)


def _declared_page_ids(payload: Mapping[str, object], label: str) -> frozenset[str]:
    values = payload.get("page_ids")
    if not isinstance(values, list) or not values or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{label} requires a non-empty page_ids list")
    if len(values) != len(set(values)):
        raise ValueError(f"{label} contains duplicate page IDs")
    return frozenset(values)


def _factorized_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("factorized result must contain runs")
    field_and_map: dict[str, tuple[str, dict[str, str]]] = {
        "current-ctd": (
            "detector_id",
            {"current_ctd_raw": "raw", "current_ctd_refined": "refined"},
        ),
        "ballons-ctd": (
            "detector_id",
            {
                "ballons_ctd_raw": "raw",
                "ballons_ctd_refined": "refined",
                "ballons_ctd_native3px": "native3",
            },
        ),
        "sickzil": ("detector_id", {"sickzil_raw": "raw"}),
        "manga109-text": ("detector_id", {"manga109_text": "raw"}),
        "ctbd-text": ("detector_id", {"ctbd_raw": "raw"}),
        "ownership-roi-ctd": (
            "detector_id",
            {
                "ownership_roi_ctd": "raw",
                "ownership_roi_ctd_refined": "refined",
            },
        ),
        "ownership": (
            "ownership_id",
            {
                "control_text_prior": "block_region",
                "control_dual_ownership": "dual_ownership",
                "ballons_ctbd_content": "ctbd_content",
                "ysg_standard": "ysg_standard",
                "ysg_obb": "ysg_obb",
                "manga109_text": "manga109",
            },
        ),
        "bubble-silhouette": (
            "silhouette_id",
            {
                "pr2_validated": "pr2_validated",
                "ballons_native": "ballons_native",
                "ctbd_bubble": "ctbd_bubble",
                "manga109_balloon": "manga109_balloon",
                "ballons_pr2_union": "pair_union_ballons_pr2",
                "ballons_pr2_intersection": "pair_intersection_ballons_pr2",
                "two_of_four_consensus": "consensus_2_of_4",
                "three_of_four_consensus": "consensus_3_of_4",
            },
        ),
        "router": (
            "router_id",
            {"control_r0": "R0", **{variant: variant for variant in ("R1", "R2", "R3", "R4")}},
        ),
        "mask-expansion": (
            "expansion_id",
            {
                "raw": "raw",
                "refined": "refined",
                "native3px": "native3",
                "content_component": "content_component",
                "bubble_interior": "validated_interior",
                "lab_dilate1": "lab_dilate1",
                "lab_dilate2": "lab_dilate2",
                "lab_dilate3": "lab_dilate3",
                "lab_dilate4": "lab_dilate4",
            },
        ),
        "fill-backend": (
            "fill_id",
            {
                "current_lama": "current_lama",
                "ballons_lama": "ballons_lama",
                "robust_flat_median": "robust_flat_median",
                "planar_gradient": "planar_gradient",
                "telea": "telea",
                "conditional_hybrid": "conditional_hybrid",
                "skip": "skip",
            },
        ),
    }
    if family_id not in field_and_map:
        return frozenset()
    field, aliases = field_and_map[family_id]
    return frozenset(
        alias
        for row in runs
        if isinstance(row, Mapping)
        for alias in (aliases.get(str(row.get(field) or "")),)
        if alias
    )


def _stage1_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    candidate = str(payload.get("candidate") or "")
    variant = str(payload.get("variant") or "")
    if family_id == "ctd-synthetic-finetune":
        aliases = {"raw": "raw", "refined": "refined", "dilated": "native3"}
        normalized = aliases.get(variant, "")
        return (
            frozenset({normalized})
            if candidate == "ctd-synthetic-low-contrast-finetune-v4" and normalized
            else frozenset()
        )
    candidate_families = {
        "ballons-ctd-text-roi": "ownership-roi-ctd",
        "manga109-text": "manga109-text",
    }
    if candidate_families.get(candidate) != family_id:
        return frozenset()
    normalized = "refined" if variant == "refined" else "raw" if variant == "raw" else ""
    return frozenset({normalized}) if normalized else frozenset()


def _fusion_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        raise ValueError("fusion result must contain runs")
    if family_id == "detector-fusion":
        return frozenset(
            str(row.get("fusion") or "")
            for row in runs
            if isinstance(row, Mapping) and str(row.get("fusion") or "")
        )
    if family_id == "roi-trigger":
        variants = {"none"}
        variants.update(
            str(row.get("trigger") or "")
            for row in runs
            if isinstance(row, Mapping) and str(row.get("trigger") or "")
        )
        return frozenset(variants)
    single_aliases: dict[str, tuple[str, str]] = {
        "current_ctd_raw": ("current-ctd", "raw"),
        "current_ctd_refined": ("current-ctd", "refined"),
        "ballons_ctd_raw": ("ballons-ctd", "raw"),
        "ballons_ctd_refined": ("ballons-ctd", "refined"),
        "ballons_ctd_native3": ("ballons-ctd", "native3"),
        "ctbd_raw": ("ctbd-text", "raw"),
        "sickzil_raw": ("sickzil", "raw"),
        "manga109_text": ("manga109-text", "raw"),
        "ownership_roi_ctd": ("ownership-roi-ctd", "raw"),
        "ownership_roi_ctd_refined": ("ownership-roi-ctd", "refined"),
    }
    observed: set[str] = set()
    for row in runs:
        if not isinstance(row, Mapping) or str(row.get("fusion") or "") != "single":
            continue
        mapped = single_aliases.get(str(row.get("primary") or ""))
        if mapped is not None and mapped[0] == family_id:
            observed.add(mapped[1])
    return frozenset(observed)


def _semantic_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    policies = payload.get("policies")
    if family_id != "semantic-policy" or not isinstance(policies, list):
        return frozenset()
    return frozenset(
        str(row.get("policy_id") or "")
        for row in policies
        if isinstance(row, Mapping) and str(row.get("policy_id") or "")
    )


def _source_protection_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    if family_id not in {"exact-protection", "exact-protection-historical"}:
        return frozenset()
    aliases = {
        "c14-structure-risk-narrow-claim": "C14",
        "c15-post-expansion-protect-plus-c11-narrow": "C15",
        "c17-detector-verified-final-protect": "C17",
        "c18-product-expansion-matched-protect": "C18",
        "c19-accepted-seed-final-protect": "C19",
        "c21-expansion-reentry-protect": "C21",
        "c22-structure-risk-halo-narrow": "C22",
        "c23-structure-risk-narrow-addition": "C23",
    }
    variant = aliases.get(str(payload.get("candidate_id") or ""))
    return frozenset({variant}) if variant else frozenset()


_ROLE_RESULT_VARIANTS: dict[str, frozenset[str]] = {
    "ownership": frozenset({"rtdetr_pixel", "c13_reconciliation"}),
    "exact-protection": frozenset({"pr4_exact"}),
    "exact-composite": frozenset({"immutable_original_exact_mask"}),
}


def _role_result_variants(
    payload: Mapping[str, object], family_id: str
) -> frozenset[str]:
    allowed = _ROLE_RESULT_VARIANTS.get(family_id, frozenset())
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("role result must contain records")
    return frozenset(
        str(row.get("variant_id") or "")
        for row in records
        if isinstance(row, Mapping)
        and str(row.get("family_id") or "") == family_id
        and str(row.get("variant_id") or "") in allowed
    )


ARTIFACT_VARIANT_EXTRACTORS: dict[str, ArtifactVariantExtractor] = {
    "inpaint-factorized-results-v3": _factorized_variants,
    "inpaint-detector-fusion-results-v4": _fusion_variants,
    "inpaint-semantic-policy-results-v4": _semantic_variants,
    "inpaint-source-protection-reapply-v3": _source_protection_variants,
    "inpaint-detector-bakeoff-stage1-v1": _stage1_variants,
    "inpaint-method-role-results-v4": _role_result_variants,
}


def artifact_declared_variants(payload: Mapping[str, object], family_id: str) -> frozenset[str]:
    schema_version = str(payload.get("schema_version") or "")
    extractor = ARTIFACT_VARIANT_EXTRACTORS.get(schema_version)
    if extractor is None:
        raise ValueError(
            "unsupported evidence artifact schema for automatic variant proof: "
            f"{schema_version or '<empty>'}"
        )
    return extractor(payload, family_id)


def _validate_closure_ledger(
    payload: Mapping[str, object],
    *,
    physical_count_field: str,
) -> tuple[list[Mapping[str, object]], dict[str, Mapping[str, object]]]:
    ledger = _object_rows(payload, "closure_ledger", "result closure ledger")
    logical_count = _required_int(payload, "logical_combination_count")
    physical_count = _required_int(payload, physical_count_field)
    if len(ledger) != logical_count:
        raise ValueError("result logical count differs from closure ledger length")
    _require_unique(ledger, "logical_id", "result closure ledger")
    by_id = {str(row["logical_id"]): row for row in ledger}
    executed_count = 0
    for row in ledger:
        state = str(row.get("closure_state") or "")
        if state not in COMBINATION_CLOSURE_STATES:
            raise ValueError("result closure ledger contains an unknown state")
        selection = row.get("selection")
        if not isinstance(selection, Mapping) or not selection:
            raise ValueError("result closure ledger contains an invalid selection")
        reason = str(row.get("reason") or "")
        content = str(row.get("content_sha256") or "")
        reused_from = str(row.get("reused_from") or "")
        if state == "executed":
            executed_count += 1
            if not _is_sha256(content):
                raise ValueError("executed closure row requires content SHA")
            if reused_from:
                raise ValueError("executed closure row cannot declare reused_from")
        elif state == "reused_by_sha":
            if not _is_sha256(content) or not reused_from:
                raise ValueError("reused closure row requires content SHA and source")
        elif not reason:
            raise ValueError(f"{state} closure row requires a reason")
    if executed_count != physical_count:
        raise ValueError("result physical count differs from executed closure rows")
    for row in ledger:
        if str(row.get("closure_state") or "") != "reused_by_sha":
            continue
        source = by_id.get(str(row.get("reused_from") or ""))
        if source is None or str(source.get("closure_state") or "") != "executed":
            raise ValueError("reused closure source must be an executed logical row")
        if str(source.get("content_sha256") or "") != str(row.get("content_sha256") or ""):
            raise ValueError("reused closure content SHA differs from its source")
    return ledger, by_id


def _validate_run_status(row: Mapping[str, object], label: str) -> None:
    status = str(row.get("status") or "")
    if status not in ROLE_STATES - {"blocked_asset"}:
        raise ValueError(f"{label} contains an invalid or non-executed status")
    if row.get("oracle_only") is True and status == "pareto":
        raise ValueError("oracle-only result cannot be a pareto product candidate")
    if not isinstance(row.get("metrics"), Mapping):
        raise ValueError(f"{label} must contain metrics")


def _validate_factorized(payload: Mapping[str, object]) -> None:
    ledger, _ = _validate_closure_ledger(
        payload, physical_count_field="physical_combination_count"
    )
    runs = _object_rows(payload, "runs", "factorized result")
    if _required_int(payload, "combination_count") != len(runs):
        raise ValueError("factorized combination count differs from runs length")
    if len(runs) != _required_int(payload, "physical_combination_count"):
        raise ValueError("factorized physical count differs from runs length")
    _require_unique(runs, "run_id", "factorized runs")
    executed_ids = {
        str(row["logical_id"])
        for row in ledger
        if str(row.get("closure_state") or "") == "executed"
    }
    run_ids = {str(row["run_id"]) for row in runs}
    if run_ids != executed_ids:
        raise ValueError("factorized runs do not exactly match executed closure rows")
    pages = payload.get("pages")
    if not isinstance(pages, Mapping) or set(map(str, pages)) != run_ids:
        raise ValueError("factorized page results do not exactly match executed runs")
    records: list[FactorizedRunRecord] = []
    common_page_ids: frozenset[str] | None = None
    for row in runs:
        _validate_run_status(row, "factorized run")
        for field in (
            "detector_id",
            "ownership_id",
            "silhouette_id",
            "router_id",
            "expansion_id",
            "fill_id",
        ):
            if not str(row.get(field) or ""):
                raise ValueError(f"factorized run lacks {field}")
        page_rows = pages.get(str(row["run_id"]))
        if not isinstance(page_rows, list) or any(not isinstance(page, Mapping) for page in page_rows):
            raise ValueError("factorized run page results must be an object list")
        run_page_ids = _page_id_set(list(page_rows), "factorized run pages")
        if common_page_ids is None:
            common_page_ids = run_page_ids
        elif run_page_ids != common_page_ids:
            raise ValueError("factorized runs do not share one exact page inventory")
        page_count = row["metrics"].get("page_count")  # type: ignore[index]
        if not isinstance(page_count, int) or page_count != len(page_rows):
            raise ValueError("factorized metrics page count differs from page results")
        records.append(
            FactorizedRunRecord(
                run_id=str(row["run_id"]),
                detector_id=str(row["detector_id"]),
                ownership_id=str(row["ownership_id"]),
                silhouette_id=str(row["silhouette_id"]),
                router_id=str(row["router_id"]),
                expansion_id=str(row["expansion_id"]),
                fill_id=str(row["fill_id"]),
                oracle_only=bool(row.get("oracle_only", False)),
                status=str(row["status"]),
                metrics=row["metrics"],  # type: ignore[arg-type]
                closure_reason=str(row.get("closure_reason") or ""),
            )
        )
    recomputed = {record.run_id: record for record in select_pareto_records(records)}
    for declared in records:
        expected = recomputed[declared.run_id]
        if declared.status in {"pareto", "family_complete"} and not (
            _factorized_hard_gate_passes(declared.metrics)
        ):
            raise ValueError(
                "factorized finalist status is not proved by fail-closed metrics"
            )
        if declared.status != expected.status:
            raise ValueError(
                "factorized declared status differs from recomputed metrics/Pareto status"
            )


def _validate_fusion(payload: Mapping[str, object]) -> None:
    ledger, _ = _validate_closure_ledger(
        payload, physical_count_field="physical_output_count"
    )
    runs = _object_rows(payload, "runs", "fusion result")
    if len(runs) != _required_int(payload, "logical_combination_count"):
        raise ValueError("fusion logical count differs from runs length")
    if _required_int(payload, "unaccounted_combination_count") != 0:
        raise ValueError("fusion artifact contains unaccounted combinations")
    _require_unique(runs, "run_id", "fusion runs")
    if {str(row["run_id"]) for row in runs} != {
        str(row["logical_id"]) for row in ledger
    }:
        raise ValueError("fusion runs do not exactly match closure rows")
    ledger_by_id = {str(row["logical_id"]): row for row in ledger}
    _declared_page_ids(payload, "fusion result")
    for row in runs:
        _validate_run_status(row, "fusion run")
        for field in ("fusion", "primary"):
            if not str(row.get(field) or ""):
                raise ValueError(f"fusion run lacks {field}")
        output_sha = row["metrics"].get("output_mask_set_sha256")  # type: ignore[index]
        if not _is_sha256(output_sha):
            raise ValueError("fusion run lacks exact output-mask-set SHA")
        closure_sha = str(ledger_by_id[str(row["run_id"])].get("content_sha256") or "")
        if str(output_sha) != closure_sha:
            raise ValueError("fusion output SHA differs from closure content SHA")


def _validate_semantic(payload: Mapping[str, object]) -> None:
    policies = _object_rows(payload, "policies", "semantic result")
    if _required_int(payload, "policy_count") != len(policies):
        raise ValueError("semantic policy count differs from policies length")
    if _required_int(payload, "unaccounted_policy_count") != 0:
        raise ValueError("semantic artifact contains unaccounted policies")
    _declared_page_ids(payload, "semantic result")
    _require_unique(policies, "policy_id", "semantic policies")
    for row in policies:
        policy_id = str(row.get("policy_id") or "")
        oracle_only = row.get("oracle_only")
        if oracle_only is not (policy_id == "human_oracle"):
            raise ValueError("semantic oracle flag differs from policy identity")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("semantic policy must contain metrics")
        integer_fields = (
            "instance_count",
            "required_instance_count",
            "preserve_instance_count",
            "preserve_destructive_count",
            "ambiguous_instance_count",
            "ambiguous_destructive_count",
            "unavailable_instance_count",
        )
        values: dict[str, int] = {}
        for field in integer_fields:
            value = metrics.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"semantic metrics require non-negative integer {field}")
            values[field] = value
        if values["instance_count"] != (
            values["required_instance_count"]
            + values["preserve_instance_count"]
            + values["ambiguous_instance_count"]
        ):
            raise ValueError("semantic instance counts do not close")
        if values["preserve_destructive_count"] > values["preserve_instance_count"] or (
            values["ambiguous_destructive_count"]
            > values["ambiguous_instance_count"]
        ) or values["unavailable_instance_count"] > values["instance_count"]:
            raise ValueError("semantic destructive/unavailable counts exceed inventory")
        recall = metrics.get("required_translate_recall")
        required = values["required_instance_count"]
        if required:
            if not _finite_number(recall) or not 0.0 <= float(recall) <= 1.0:
                raise ValueError("semantic required recall is missing or invalid")
        elif recall is not None:
            raise ValueError("semantic zero-required policy must report null recall")
        for field in ("role_exact_accuracy", "action_exact_accuracy"):
            accuracy = metrics.get(field)
            if values["instance_count"]:
                if not _finite_number(accuracy) or not 0.0 <= float(accuracy) <= 1.0:
                    raise ValueError(f"semantic {field} is missing or invalid")
            elif accuracy is not None:
                raise ValueError(f"semantic empty inventory must report null {field}")
        blocked = not bool(oracle_only) and values["unavailable_instance_count"] > 0
        hard_pass = (
            not blocked
            and (required == 0 or float(recall) >= 1.0)
            and values["preserve_destructive_count"] == 0
            and values["ambiguous_destructive_count"] == 0
        )
        expected = "blocked_asset" if blocked else (
            "family_complete" if hard_pass else "dominated"
        )
        if str(row.get("status") or "") != expected:
            raise ValueError(
                "semantic declared status differs from fail-closed metric status"
            )
        reason = str(row.get("closure_reason") or "")
        if (expected == "family_complete" and reason) or (
            expected != "family_complete" and not reason
        ):
            raise ValueError("semantic closure reason differs from derived status")


_SUMMARY_ZERO_FIELDS = (
    "protected_edit_overlap",
    "ambiguous_edit_overlap",
    "ownership_leak_pixel_count",
    "preserve_edit_overlap",
    "false_edit_pixel_count",
    "missed_target_instance_count",
)


def _validate_stage_summary(summary: Mapping[str, object], page_count: int) -> str:
    if summary.get("page_count") != page_count:
        raise ValueError("artifact summary page count differs from pages length")
    required = (
        "aggregate_target_coverage",
        "minimum_component_coverage",
        "minimum_target_instance_edit_coverage",
        "target_instance_seed_recall",
        *_SUMMARY_ZERO_FIELDS,
    )
    if any(field not in summary for field in required):
        raise ValueError("artifact summary omits required closure metrics")
    numeric = [summary[field] for field in required if summary[field] is not None]
    if any(not _finite_number(value) for value in numeric):
        raise ValueError("artifact summary contains a non-finite closure metric")
    has_targets = int(summary.get("target_instance_count") or 0) > 0
    if has_targets and any(summary[field] is None for field in required[:4]):
        raise ValueError("artifact summary omits required target metrics")
    safety_pass = all(int(summary[field]) == 0 for field in _SUMMARY_ZERO_FIELDS)
    quality_pass = (
        not has_targets
        or (
            float(summary["aggregate_target_coverage"]) >= 0.98
            and float(summary["minimum_component_coverage"]) >= 0.98
            and float(summary["minimum_target_instance_edit_coverage"]) >= 0.98
            and float(summary["target_instance_seed_recall"]) >= 1.0
        )
    )
    return "family_complete" if safety_pass and quality_pass else "dominated"


def _validate_stage1(payload: Mapping[str, object]) -> None:
    pages = _object_rows(payload, "pages", "stage1 result")
    if not pages:
        raise ValueError("stage1 result must contain pages")
    _require_unique(pages, "page_id", "stage1 pages")
    for field in ("candidate", "variant"):
        if not str(payload.get(field) or ""):
            raise ValueError(f"stage1 result lacks {field}")
    model = payload.get("model")
    role = payload.get("role_candidate")
    summary = payload.get("summary")
    if not isinstance(model, Mapping) or not isinstance(role, Mapping) or not isinstance(summary, Mapping):
        raise ValueError("stage1 result lacks model, role candidate, or summary")
    if not _is_sha256(model.get("sha256")) or not _is_sha256(role.get("model_sha256")):
        raise ValueError("stage1 model provenance requires SHA-256 identities")
    if not _is_sha256(role.get("preprocessing_contract_sha256")):
        raise ValueError("stage1 preprocessing provenance requires SHA-256")
    for field in ("candidate_id", "provider", "role", "variant", "code_commit", "runtime_provider"):
        if not str(role.get(field) or ""):
            raise ValueError(f"stage1 role candidate lacks {field}")
    _validate_stage_summary(summary, len(pages))


def _validate_source_protection(payload: Mapping[str, object]) -> None:
    pages = _object_rows(payload, "pages", "source protection result")
    if not pages:
        raise ValueError("source protection result must contain pages")
    _require_unique(pages, "page_id", "source protection pages")
    if not str(payload.get("candidate_id") or ""):
        raise ValueError("source protection result lacks candidate_id")
    summary = payload.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("source protection result lacks summary")
    _validate_stage_summary(summary, len(pages))


def _validate_role_results(payload: Mapping[str, object]) -> None:
    records = _object_rows(payload, "records", "method role result")
    pages = _object_rows(payload, "pages", "method role result")
    if not records or not pages:
        raise ValueError("method role result requires non-empty records and pages")
    if _required_int(payload, "record_count") != len(records):
        raise ValueError("method role result count differs from records length")
    if _required_int(payload, "page_count") != len(pages):
        raise ValueError("method role result page count differs from pages length")
    if _required_int(payload, "unaccounted_record_count") != 0:
        raise ValueError("method role result contains unaccounted records")
    if not _is_sha256(payload.get("source_artifact_sha256")) or not str(
        payload.get("source_artifact_schema_version") or ""
    ):
        raise ValueError("method role result lacks source artifact provenance")
    _require_unique(pages, "page_id", "method role result pages")
    page_ids = {str(page["page_id"]) for page in pages}
    identities: set[tuple[str, str]] = set()
    for row in records:
        family_id = str(row.get("family_id") or "")
        variant_id = str(row.get("variant_id") or "")
        if variant_id not in _ROLE_RESULT_VARIANTS.get(family_id, frozenset()):
            raise ValueError("method role result declares an unsupported family variant")
        identity = (family_id, variant_id)
        if identity in identities:
            raise ValueError("method role result contains a duplicate family variant")
        identities.add(identity)
        if not str(row.get("source_result_id") or ""):
            raise ValueError("method role result lacks source_result_id")
        if not _is_sha256(row.get("content_sha256")):
            raise ValueError("method role result lacks a content SHA")
        record_pages = row.get("page_ids")
        if not isinstance(record_pages, list) or set(map(str, record_pages)) != page_ids:
            raise ValueError("method role result does not cover the full page inventory")
        _validate_run_status(row, "method role record")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("method role result lacks metrics")
        derived = _validate_stage_summary(metrics, len(pages))
        if str(row.get("status") or "") != derived:
            raise ValueError(
                "method role result status differs from its fail-closed metrics"
            )


def validate_evidence_artifact(payload: Mapping[str, object]) -> None:
    schema = str(payload.get("schema_version") or "")
    validators = {
        "inpaint-factorized-results-v3": _validate_factorized,
        "inpaint-detector-fusion-results-v4": _validate_fusion,
        "inpaint-semantic-policy-results-v4": _validate_semantic,
        "inpaint-source-protection-reapply-v3": _validate_source_protection,
        "inpaint-detector-bakeoff-stage1-v1": _validate_stage1,
        "inpaint-method-role-results-v4": _validate_role_results,
    }
    validator = validators.get(schema)
    if validator is None:
        raise ValueError(f"unsupported evidence artifact schema: {schema or '<empty>'}")
    validator(payload)


def _artifact_page_ids(payload: Mapping[str, object]) -> frozenset[str]:
    schema = str(payload.get("schema_version") or "")
    if schema == "inpaint-factorized-results-v3":
        pages = payload.get("pages")
        if not isinstance(pages, Mapping) or not pages:
            raise ValueError("factorized result lacks page results")
        first = next(iter(pages.values()))
        if not isinstance(first, list) or any(
            not isinstance(row, Mapping) for row in first
        ):
            raise ValueError("factorized result has an invalid page inventory")
        return _page_id_set(list(first), "factorized result pages")
    if schema in {
        "inpaint-detector-fusion-results-v4",
        "inpaint-semantic-policy-results-v4",
    }:
        return _declared_page_ids(payload, "evidence artifact")
    pages = _object_rows(payload, "pages", "evidence artifact")
    return _page_id_set(pages, "evidence artifact pages")


def _matching_runs(payload: Mapping[str, object], family_id: str, variant_id: str) -> list[Mapping[str, object]]:
    runs = payload.get("runs")
    if not isinstance(runs, list):
        return []
    matched: list[Mapping[str, object]] = []
    for row in runs:
        if not isinstance(row, Mapping):
            continue
        isolated = {**payload, "runs": [row]}
        if variant_id in artifact_declared_variants(isolated, family_id):
            matched.append(row)
    return matched


def _aggregate_disposition(rows: Sequence[Mapping[str, object]]) -> tuple[str, str]:
    if not rows:
        raise ValueError("artifact variant has no matching executed result rows")
    statuses = [str(row.get("status") or "") for row in rows]
    if "active" in statuses:
        status = "active"
    elif "pareto" in statuses:
        status = "pareto"
    elif "information_limited" in statuses:
        status = "information_limited"
    elif "family_complete" in statuses:
        status = "family_complete"
    elif set(statuses) == {"dominated"}:
        status = "dominated"
    else:
        raise ValueError("artifact variant has incompatible result dispositions")
    reasons = sorted(
        {str(row.get("closure_reason") or "") for row in rows if str(row.get("closure_reason") or "")}
    )
    return status, ";".join(reasons)


def artifact_variant_facts(payload: Mapping[str, object], family_id: str) -> dict[str, ArtifactVariantFact]:
    validate_evidence_artifact(payload)
    schema = str(payload.get("schema_version") or "")
    variants = artifact_declared_variants(payload, family_id)
    output: dict[str, ArtifactVariantFact] = {}
    if schema in {"inpaint-factorized-results-v3", "inpaint-detector-fusion-results-v4"}:
        for variant in variants:
            rows = _matching_runs(payload, family_id, variant)
            disposition, reason = _aggregate_disposition(rows)
            if schema == "inpaint-detector-fusion-results-v4":
                identities = sorted(str(row["metrics"]["output_mask_set_sha256"]) for row in rows)  # type: ignore[index]
                kind = "exact_output"
            else:
                identities = sorted(_canonical_sha256(row) for row in rows)
                kind = "artifact_record"
            output[variant] = ArtifactVariantFact(
                disposition, reason, _canonical_sha256(identities), kind
            )
        return output
    if schema == "inpaint-semantic-policy-results-v4":
        for row in payload["policies"]:  # type: ignore[index]
            if not isinstance(row, Mapping):
                continue
            variant = str(row["policy_id"])
            output[variant] = ArtifactVariantFact(
                str(row["status"]),
                str(row.get("closure_reason") or ""),
                _canonical_sha256(row),
                "artifact_record",
            )
        return output
    if schema == "inpaint-method-role-results-v4":
        for row in payload["records"]:  # type: ignore[index]
            if not isinstance(row, Mapping) or str(row.get("family_id") or "") != family_id:
                continue
            variant = str(row["variant_id"])
            output[variant] = ArtifactVariantFact(
                str(row["status"]),
                str(row.get("closure_reason") or ""),
                _canonical_sha256(
                    {
                        "source_artifact_sha256": payload["source_artifact_sha256"],
                        "source_result_id": row["source_result_id"],
                        "declared_content_sha256": row["content_sha256"],
                        "metrics": row["metrics"],
                        "page_ids": row["page_ids"],
                    }
                ),
                "artifact_record",
            )
        return output
    summary = payload.get("summary")
    pages = payload.get("pages")
    if not isinstance(summary, Mapping) or not isinstance(pages, list):
        raise ValueError("artifact lacks validated summary or pages")
    disposition = _validate_stage_summary(summary, len(pages))
    reason = "" if disposition == "family_complete" else "hard_gate_failed"
    content = _canonical_sha256(
        {
            "candidate": payload.get("candidate", payload.get("candidate_id")),
            "variant": payload.get("variant", ""),
            "summary": summary,
            "pages": pages,
            "model": payload.get("model"),
            "role_candidate": payload.get("role_candidate"),
        }
    )
    return {
        variant: ArtifactVariantFact(disposition, reason, content, "artifact_record")
        for variant in variants
    }


def artifact_variant_dispositions(payload: Mapping[str, object], family_id: str) -> dict[str, tuple[str, str]]:
    return {
        variant: (fact.disposition, fact.reason)
        for variant, fact in artifact_variant_facts(payload, family_id).items()
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scope_manifest_binding(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("scope manifest root must be an object")
    schema = str(payload.get("schema_version") or "")
    corpus = str(payload.get("corpus_id") or "")
    split = str(payload.get("split_role") or "")
    allowed_splits = frozenset(
        {
            "development",
            "development_source_only",
            "synthetic_known_ground_truth",
        }
    )
    if schema != "inpaint-factorized-source-manifest-v4" or split not in allowed_splits:
        raise ValueError("scope manifest is not an allowed source-only schema/split")
    if not corpus or payload.get("annotation_frozen_before_candidate") is not True:
        raise ValueError("scope manifest lacks frozen source-only identity")
    if payload.get("candidate_seen") not in {None, False}:
        raise ValueError("scope manifest was derived after candidate inspection")
    pages = _object_rows(payload, "pages", "scope manifest")
    page_ids = _page_id_set(pages, "scope manifest pages")
    for page in pages:
        if page.get("candidate_seen") not in {None, False} or any(
            page.get(field) is not True
            for field in (
                "target_extent_independent",
                "target_inventory_independent",
                "target_review_complete",
            )
        ):
            raise ValueError("scope manifest contains candidate-derived page annotation")
    return {
        "sha256": sha256_file(path),
        "schema_version": schema,
        "corpus_id": corpus,
        "split_role": split,
        "page_count": len(page_ids),
        "page_ids": sorted(page_ids),
        "page_inventory_sha256": _canonical_sha256(sorted(page_ids)),
    }


def merge_scope_manifest_binding(
    existing: Mapping[str, object],
    *,
    evaluation_scope: str,
    binding: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for scope, value in existing.items():
        if not isinstance(value, Mapping):
            raise ValueError(f"scope manifest binding is invalid: {scope}")
        output[str(scope)] = dict(value)
    prior = output.get(evaluation_scope)
    normalized = dict(binding)
    if prior is not None and prior != normalized:
        raise ValueError(
            f"canonical scope manifest rebinding is forbidden: {evaluation_scope}"
        )
    output[evaluation_scope] = normalized
    return {scope: output[scope] for scope in sorted(output)}


def evidence_key(family_id: str, role: str, variant_id: str, evaluation_scope: str) -> str:
    return "/".join((family_id, role, variant_id, evaluation_scope))


def merge_method_evidence(
    existing: Iterable[Mapping[str, object]],
    updates: Iterable[Mapping[str, object]],
    *,
    allow_replace: bool = False,
) -> tuple[dict[str, object], ...]:
    merged: dict[str, dict[str, object]] = {}
    for source_index, source in enumerate((existing, updates)):
        seen_in_source: set[str] = set()
        for value in source:
            row = dict(value)
            key = evidence_key(
                str(row.get("family_id") or ""),
                str(row.get("role") or ""),
                str(row.get("variant_id") or ""),
                str(row.get("evaluation_scope") or ""),
            )
            if not all(key.split("/")):
                raise ValueError("method evidence contains an empty identity")
            if key in seen_in_source:
                raise ValueError(f"duplicate method evidence identity: {key}")
            seen_in_source.add(key)
            prior = merged.get(key)
            if source_index == 1 and prior is not None and prior != row:
                if not allow_replace:
                    raise ValueError(
                        "method evidence identity already exists with different proof; "
                        "replacement requires explicit approval: " + key
                    )
                if prior.get("scope_manifest_sha256") != row.get("scope_manifest_sha256"):
                    raise ValueError("method evidence replacement cannot rebind its scope manifest")
            merged[key] = row
    return tuple(merged[key] for key in sorted(merged))


def evidence_rows_from_artifact(
    requirements: Iterable[MethodVariantRequirement],
    *,
    artifact_path: Path,
    scope_manifest_path: Path,
    family_id: str,
    variant_ids: frozenset[str],
    evaluation_scope: str,
) -> tuple[dict[str, object], ...]:
    return accounted_evidence_from_artifact(
        requirements,
        artifact_path=artifact_path,
        scope_manifest_path=scope_manifest_path,
        family_id=family_id,
        variant_ids=variant_ids,
        evaluation_scope=evaluation_scope,
    )


def accounted_evidence_from_artifact(
    requirements: Iterable[MethodVariantRequirement],
    *,
    artifact_path: Path,
    scope_manifest_path: Path,
    family_id: str,
    variant_ids: frozenset[str],
    evaluation_scope: str,
) -> tuple[dict[str, object], ...]:
    if not family_id.strip() or not variant_ids:
        raise ValueError("an explicit family and at least one variant are required")
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence artifact root must be an object")
    facts = artifact_variant_facts(payload, family_id)
    artifact_sha256 = sha256_file(artifact_path)
    scope_binding = scope_manifest_binding(scope_manifest_path)
    scope_manifest_sha256 = str(scope_binding["sha256"])
    if str(payload.get("manifest_sha256") or "") != scope_manifest_sha256:
        raise ValueError("evidence artifact manifest SHA differs from the sealed scope manifest")
    if _artifact_page_ids(payload) != frozenset(
        str(value) for value in scope_binding["page_ids"]  # type: ignore[index]
    ):
        raise ValueError(
            "evidence artifact page IDs differ from the canonical scope manifest"
        )
    registered = {
        requirement.variant_id: requirement
        for requirement in requirements
        if requirement.family_id == family_id
        and requirement.evaluation_scope == evaluation_scope
    }
    missing = sorted(variant_ids - set(registered))
    if missing:
        raise ValueError(
            "method evidence variant is not registered for the requested family/scope: "
            f"{missing[0]}"
        )
    unproved = sorted(variant_ids - set(facts))
    if unproved:
        raise ValueError(
            "requested method variant is not declared by the evidence artifact: "
            f"{unproved[0]}"
        )
    rows: list[dict[str, object]] = []
    for variant_id in sorted(variant_ids):
        requirement = registered[variant_id]
        fact = facts[variant_id]
        if fact.disposition == "blocked_asset":
            raise ValueError("blocked artifact status requires a separately hashed asset probe")
        rows.append(
            {
                "family_id": requirement.family_id,
                "role": requirement.role,
                "variant_id": requirement.variant_id,
                "evaluation_scope": requirement.evaluation_scope,
                "closure_state": "executed",
                "disposition": fact.disposition,
                "reason": fact.reason,
                "artifact_sha256": artifact_sha256,
                "artifact_schema_version": str(payload.get("schema_version") or ""),
                "artifact_name": artifact_path.name,
                "scope_manifest_sha256": scope_manifest_sha256,
                "content_sha256": fact.content_sha256,
                "content_identity_kind": fact.content_identity_kind,
                "reused_from": "",
                "blocker_probe_sha256": "",
            }
        )
    return tuple(rows)


def blocked_asset_evidence(
    requirements: Iterable[MethodVariantRequirement],
    *,
    scope_manifest_path: Path,
    blocker_probe_path: Path,
    family_id: str,
    variant_ids: frozenset[str],
    evaluation_scope: str,
) -> tuple[dict[str, object], ...]:
    binding = scope_manifest_binding(scope_manifest_path)
    scope_sha = str(binding["sha256"])
    probe = json.loads(blocker_probe_path.read_text(encoding="utf-8"))
    if not isinstance(probe, Mapping) or probe.get("schema_version") != "inpaint-blocked-asset-probe-v1":
        raise ValueError("blocked asset evidence requires a supported probe schema")
    if str(probe.get("family_id") or "") != family_id:
        raise ValueError("blocked asset probe family mismatch")
    if str(probe.get("evaluation_scope") or "") != evaluation_scope:
        raise ValueError("blocked asset probe scope mismatch")
    if str(probe.get("scope_manifest_sha256") or "") != scope_sha:
        raise ValueError("blocked asset probe scope manifest mismatch")
    if probe.get("status") != "blocked_asset":
        raise ValueError("blocked asset probe must declare blocked_asset status")
    if set(map(str, probe.get("variant_ids", []))) != set(variant_ids):
        raise ValueError("blocked asset probe variant set mismatch")
    for field in ("asset_id", "reason_code"):
        if not str(probe.get(field) or "").strip():
            raise ValueError(f"blocked asset probe lacks {field}")
    checks = probe.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(row, Mapping) for row in checks):
        raise ValueError("blocked asset probe requires concrete check records")
    probe_sha = sha256_file(blocker_probe_path)
    registered = {
        requirement.variant_id: requirement
        for requirement in requirements
        if requirement.family_id == family_id and requirement.evaluation_scope == evaluation_scope
    }
    missing = sorted(variant_ids - set(registered))
    if missing:
        raise ValueError(
            "method evidence variant is not registered for the requested family/scope: "
            f"{missing[0]}"
        )
    reason = str(probe["reason_code"])
    return tuple(
        {
            "family_id": registered[variant_id].family_id,
            "role": registered[variant_id].role,
            "variant_id": variant_id,
            "evaluation_scope": evaluation_scope,
            "closure_state": "blocked_asset",
            "disposition": "blocked_asset",
            "reason": reason,
            "artifact_sha256": "",
            "artifact_schema_version": "",
            "artifact_name": "",
            "scope_manifest_sha256": scope_sha,
            "content_sha256": "",
            "content_identity_kind": "",
            "reused_from": "",
            "blocker_probe_sha256": probe_sha,
        }
        for variant_id in sorted(variant_ids)
    )


def registry_evidence_adapter_gaps(
    requirements: Iterable[MethodVariantRequirement],
) -> tuple[dict[str, str], ...]:
    supported: dict[str, frozenset[str]] = {
        "current-ctd": frozenset({"raw", "refined"}),
        "ballons-ctd": frozenset({"raw", "refined", "native3"}),
        "sickzil": frozenset({"raw"}),
        "manga109-text": frozenset({"raw"}),
        "ctbd-text": frozenset({"raw"}),
        "ownership-roi-ctd": frozenset({"raw", "refined"}),
        "ctd-synthetic-finetune": frozenset({"raw", "refined", "native3"}),
        "detector-fusion": frozenset({"single", "or", "and", "gated_recovery"}),
        "roi-trigger": frozenset({"none", "always", "seed_missing", "raw_refined_disagreement", "source_seed_unavailable", "union"}),
        "semantic-policy": frozenset({"current_default", "detector_explicit_role", "ocr_semantic_hint", "explicit_role_consensus", "human_oracle"}),
        "ownership": frozenset({"block_region", "dual_ownership", "ctbd_content", "ysg_standard", "ysg_obb", "manga109"}),
        "bubble-silhouette": frozenset({"pr2_validated", "ballons_native", "ctbd_bubble", "manga109_balloon", "pair_union_ballons_pr2", "pair_intersection_ballons_pr2", "consensus_2_of_4", "consensus_3_of_4"}),
        "router": frozenset({"R0", "R1", "R2", "R3", "R4"}),
        "mask-expansion": frozenset({"raw", "refined", "native3", "content_component", "validated_interior", "lab_dilate1", "lab_dilate2", "lab_dilate3", "lab_dilate4"}),
        "exact-protection": frozenset({"C14", "C15", "C17", "C18", "C19", "C21", "C22", "C23"}),
        "exact-protection-historical": frozenset({"C14", "C15", "C17", "C18", "C19", "C21", "C22", "C23"}),
        "fill-backend": frozenset({"current_lama", "ballons_lama", "robust_flat_median", "planar_gradient", "telea", "conditional_hybrid", "skip"}),
        "exact-composite": frozenset({"immutable_original_exact_mask"}),
    }
    supported["ownership"] = supported["ownership"] | _ROLE_RESULT_VARIANTS["ownership"]
    supported["exact-protection"] = (
        supported["exact-protection"] | _ROLE_RESULT_VARIANTS["exact-protection"]
    )
    gaps = [
        {
            "family_id": row.family_id,
            "role": row.role,
            "variant_id": row.variant_id,
            "evaluation_scope": row.evaluation_scope,
        }
        for row in requirements
        if row.variant_id not in supported.get(row.family_id, frozenset())
    ]
    return tuple(
        sorted(
            gaps,
            key=lambda row: (
                row["family_id"], row["role"], row["variant_id"], row["evaluation_scope"]
            ),
        )
    )
