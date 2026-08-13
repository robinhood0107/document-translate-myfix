from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np

from .contracts import (
    COMBINATION_CLOSURE_STATES,
    ROLE_STATES,
    binary_mask,
    mask_sha256,
)
from .method_closure import MethodVariantRequirement
from .stage2 import _hard_gate_passes as _factorized_hard_gate_passes
from .stage2 import attach_reconstruction_control, select_pareto_records
from .contracts import FactorizedRunRecord
from .stage1 import summarize as summarize_stage1_pages
from .stage1 import (
    load_page_masks,
    load_stage1_manifest,
    positive_edit_from_claim,
    validate_source_only_manifest_v4,
)
from .stage2 import reconstruction_error, residue_score
from scripts.benchmark_inpaint_factorized_v3 import (
    aggregate_factorized_page_statistics,
)
from scripts.benchmark_inpaint_detector_fusions_v4 import (
    aggregate_fusion_page_statistics,
)
from scripts.benchmark_inpaint_semantic_policies_v4 import (
    aggregate_semantic_page_statistics,
)


ArtifactVariantExtractor = Callable[[Mapping[str, object], str], frozenset[str]]
ROOT = Path(__file__).resolve().parents[2]


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


def _logical_inventory_sha256(rows: Sequence[Mapping[str, object]]) -> str:
    inventory = sorted(
        (
            {
                "logical_id": str(row.get("logical_id") or ""),
                "selection": dict(row.get("selection") or {}),
            }
            for row in rows
        ),
        key=lambda row: row["logical_id"],
    )
    return _canonical_sha256(inventory)


def _require_logical_inventory_binding(
    payload: Mapping[str, object],
    ledger: Sequence[Mapping[str, object]],
    *,
    upstream_sha_field: str,
) -> None:
    if not _is_sha256(payload.get(upstream_sha_field)):
        raise ValueError(
            f"result requires a lowercase SHA-256 {upstream_sha_field} binding"
        )
    declared = str(payload.get("logical_inventory_sha256") or "")
    expected = _logical_inventory_sha256(ledger)
    if declared != expected:
        raise ValueError(
            "result logical inventory SHA differs from its full closure ledger"
        )


def _validate_upstream_logical_inventory(
    payload: Mapping[str, object],
    *,
    upstream_contract_path: Path | None,
    scope_manifest_sha256: str,
    expected_fusion_candidate_ids: frozenset[str] | None,
) -> None:
    schema = str(payload.get("schema_version") or "")
    if schema not in {
        "inpaint-factorized-results-v3",
        "inpaint-detector-fusion-results-v4",
    }:
        if upstream_contract_path is not None:
            raise ValueError(
                "this evidence artifact schema has no upstream matrix/spec contract"
            )
        return
    if upstream_contract_path is None:
        raise ValueError(
            "factorized/fusion evidence requires the exact upstream matrix/spec file"
        )
    upstream = json.loads(upstream_contract_path.read_text(encoding="utf-8"))
    if not isinstance(upstream, dict):
        raise ValueError("upstream matrix/spec root must be an object")
    upstream_sha256 = sha256_file(upstream_contract_path)
    if schema == "inpaint-factorized-results-v3":
        if upstream.get("schema_version") != "inpaint-factorized-matrix-v3":
            raise ValueError("factorized evidence requires matrix v3 upstream")
        if str(payload.get("matrix_sha256") or "") != upstream_sha256:
            raise ValueError("factorized artifact matrix SHA differs from upstream bytes")
        manifest_path = upstream.get("manifest")
        if manifest_path is not None:
            if not isinstance(manifest_path, str):
                raise ValueError("factorized matrix has an invalid sealed manifest path")
            bound_manifest = Path(manifest_path)
            if not bound_manifest.is_absolute():
                bound_manifest = upstream_contract_path.parent / bound_manifest
            if not bound_manifest.is_file():
                raise ValueError("factorized matrix has an invalid sealed manifest path")
            if sha256_file(bound_manifest) != scope_manifest_sha256:
                raise ValueError("factorized matrix is bound to a different scope manifest")
        axes = upstream.get("axes")
        controls = upstream.get("controls")
        if not isinstance(axes, dict) or not isinstance(controls, dict):
            raise ValueError("factorized matrix lacks axes or controls")
        from scripts.benchmark_inpaint_factorized_v3 import (  # noqa: PLC0415
            _declared_combinations,
            _prepare_closure_ledger,
        )

        combinations = _declared_combinations(upstream, axes, controls)
        expected_ledger, physical = _prepare_closure_ledger(
            combinations,
            matrix=upstream,
            manifest_sha256=scope_manifest_sha256,
        )
        expected_rows = [row.as_record() for row in expected_ledger]
        actual_ledger = payload.get("closure_ledger")
        if not isinstance(actual_ledger, list) or any(
            not isinstance(row, Mapping) for row in actual_ledger
        ):
            raise ValueError("factorized closure ledger is invalid")
        expected_fields = tuple(expected_rows[0]) if expected_rows else ()
        actual_core = [
            {field: row.get(field) for field in expected_fields}
            for row in actual_ledger
        ]
        if actual_core != expected_rows or any(
            set(row) - set(expected_fields) - {"runtime_diagnostics"}
            for row in actual_ledger
        ):
            raise ValueError(
                "factorized closure ledger differs from the upstream matrix inventory"
            )
        if payload.get("logical_combination_count") != len(combinations) or (
            payload.get("physical_combination_count") != len(physical)
        ):
            raise ValueError("factorized counts differ from the upstream matrix")
        return

    if upstream.get("schema_version") != "inpaint-detector-fusion-spec-v4":
        raise ValueError("fusion evidence requires fusion spec v4 upstream")
    if str(payload.get("spec_sha256") or "") != upstream_sha256:
        raise ValueError("fusion artifact spec SHA differs from upstream bytes")
    raw_candidates = upstream.get("candidates")
    if not isinstance(raw_candidates, dict) or not raw_candidates or any(
        not isinstance(value, dict) for value in raw_candidates.values()
    ):
        raise ValueError("fusion upstream spec lacks candidate definitions")
    if not expected_fusion_candidate_ids:
        raise ValueError(
            "fusion evidence requires a canonical registered candidate inventory"
        )
    actual_candidate_ids = frozenset(map(str, raw_candidates))
    if actual_candidate_ids != expected_fusion_candidate_ids:
        raise ValueError(
            "fusion upstream candidate set differs from the canonical registered "
            "provider/variant inventory"
        )
    from scripts.benchmark_inpaint_detector_fusions_v4 import (  # noqa: PLC0415
        _logical_runs,
    )

    roi_candidates = frozenset(
        str(candidate_id)
        for candidate_id, value in raw_candidates.items()
        if bool(value.get("roi_detector", False))
    )
    expected_runs = _logical_runs(tuple(map(str, raw_candidates)), roi_candidates)
    expected_inventory = {
        str(run["run_id"]): dict(run) for run in expected_runs
    }
    ledger = payload.get("closure_ledger")
    if not isinstance(ledger, list):
        raise ValueError("fusion artifact lacks closure ledger")
    actual_inventory = {
        str(row.get("logical_id") or ""): dict(row.get("selection") or {})
        for row in ledger
        if isinstance(row, Mapping)
    }
    if actual_inventory != expected_inventory or len(ledger) != len(expected_runs):
        raise ValueError(
            "fusion closure ledger differs from the upstream spec inventory"
        )
    if payload.get("logical_combination_count") != len(expected_runs):
        raise ValueError("fusion logical count differs from the upstream spec")


_FACTORIZED_FAMILY_IDS = frozenset(
    {
        "current-ctd",
        "ballons-ctd",
        "sickzil",
        "manga109-text",
        "ctbd-text",
        "ownership-roi-ctd",
        "ownership",
        "bubble-silhouette",
        "router",
        "mask-expansion",
        "fill-backend",
    }
)


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
        identities = payload.get("variant_output_identity")
        if candidate != "ctd-synthetic-low-contrast-finetune-v4" or not isinstance(
            identities, Mapping
        ):
            return frozenset()
        return frozenset(
            normalized
            for source_variant, normalized in aliases.items()
            if isinstance(identities.get(source_variant), Mapping)
        )
    easyocr_families = {
        "easyocr-craft": "easyocr-craft",
        "easyocr-dbnet18": "easyocr-dbnet18",
    }
    if easyocr_families.get(candidate) == family_id:
        aliases = {"raw": "raw", "refined": "refined", "dilated": "native3"}
        identities = payload.get("variant_output_identity")
        if not isinstance(identities, Mapping):
            return frozenset()
        return frozenset(
            normalized
            for source_variant, normalized in aliases.items()
            if isinstance(identities.get(source_variant), Mapping)
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


ARTIFACT_VARIANT_EXTRACTORS: dict[str, ArtifactVariantExtractor] = {
    "inpaint-factorized-results-v3": _factorized_variants,
    "inpaint-detector-fusion-results-v4": _fusion_variants,
    "inpaint-semantic-policy-results-v4": _semantic_variants,
    "inpaint-source-protection-reapply-v3": _source_protection_variants,
    "inpaint-detector-bakeoff-stage1-v1": _stage1_variants,
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


def _declared_complete_output_run_ids(
    payload: Mapping[str, object],
) -> frozenset[str]:
    binding = payload.get("output_artifact_inventory")
    if binding is None:
        return frozenset()
    if not isinstance(binding, Mapping):
        raise ValueError("output artifact inventory binding must be an object")
    values = binding.get("complete_run_ids")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("output artifact inventory lacks complete run IDs")
    if len(values) != len(set(values)):
        raise ValueError("output artifact inventory has duplicate complete run IDs")
    return frozenset(values)


def _declared_complete_runtime_run_ids(
    payload: Mapping[str, object],
) -> frozenset[str]:
    binding = payload.get("runtime_evidence_ledger")
    if binding is None:
        return frozenset()
    if not isinstance(binding, Mapping) or binding.get("role") != "runtime_evidence":
        raise ValueError("runtime evidence ledger binding must name its role")
    values = binding.get("complete_run_ids")
    if not isinstance(values, list) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError("runtime evidence ledger lacks complete run IDs")
    if len(values) != len(set(values)):
        raise ValueError("runtime evidence ledger has duplicate complete run IDs")
    return frozenset(values)


def _validate_factorized(payload: Mapping[str, object]) -> None:
    ledger, ledger_by_id = _validate_closure_ledger(
        payload, physical_count_field="physical_combination_count"
    )
    _require_logical_inventory_binding(
        payload, ledger, upstream_sha_field="matrix_sha256"
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
    complete_output_run_ids = _declared_complete_output_run_ids(payload)
    complete_runtime_run_ids = _declared_complete_runtime_run_ids(payload)
    if not complete_output_run_ids.issubset(run_ids):
        raise ValueError("factorized output inventory names an unknown run")
    if not complete_runtime_run_ids.issubset(run_ids):
        raise ValueError("factorized runtime ledger names an unknown run")
    declared_metrics_by_id: dict[str, Mapping[str, object]] = {}
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
        selection = row.get("selection")
        closure_selection = ledger_by_id[str(row["run_id"])].get("selection")
        if not isinstance(selection, Mapping) or dict(selection) != dict(
            closure_selection  # type: ignore[arg-type]
        ):
            raise ValueError(
                "factorized run selection differs from its logical closure inventory"
            )
        page_rows = pages.get(str(row["run_id"]))
        if not isinstance(page_rows, list) or any(not isinstance(page, Mapping) for page in page_rows):
            raise ValueError("factorized run page results must be an object list")
        run_page_ids = _page_id_set(list(page_rows), "factorized run pages")
        if common_page_ids is None:
            common_page_ids = run_page_ids
        elif run_page_ids != common_page_ids:
            raise ValueError("factorized runs do not share one exact page inventory")
        declared_metrics = row["metrics"]  # type: ignore[index]
        page_count = declared_metrics.get("page_count")
        if not isinstance(page_count, int) or page_count != len(page_rows):
            raise ValueError("factorized metrics page count differs from page results")
        try:
            canonical_metrics = aggregate_factorized_page_statistics(page_rows)
        except (KeyError, TypeError, ValueError) as error:
            if str(row.get("status") or "") not in {"information_limited"}:
                raise ValueError(
                    "factorized page rows lack canonical sufficient statistics"
                ) from error
            raise ValueError(
                "legacy factorized evidence is deliberately unavailable for closure"
            )
        declared_metrics_by_id[str(row["run_id"])] = declared_metrics
        expected_runtime_diagnostics = {
            "conditional_hybrid_overlap_conflict_pixel_count": int(
                canonical_metrics.get(
                    "conditional_hybrid_overlap_conflict_pixel_count", 0
                )
            ),
            "conditional_hybrid_overlap_fallback_page_count": int(
                canonical_metrics.get(
                    "conditional_hybrid_overlap_fallback_page_count", 0
                )
            ),
        }
        declared_runtime_diagnostics = ledger_by_id[str(row["run_id"])].get(
            "runtime_diagnostics"
        )
        if (
            declared_runtime_diagnostics is not None
            or any(expected_runtime_diagnostics.values())
        ) and (
            not isinstance(declared_runtime_diagnostics, Mapping)
            or dict(declared_runtime_diagnostics) != expected_runtime_diagnostics
        ):
            raise ValueError(
                "factorized closure runtime diagnostics differ from canonical pages"
            )
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
                metrics=canonical_metrics,
                closure_reason=str(row.get("closure_reason") or ""),
                selection={str(key): str(value) for key, value in selection.items()},
            )
        )
    records = attach_reconstruction_control(
        records,
        str(payload.get("reconstruction_control_run_id") or "") or None,
    )
    for record in records:
        if dict(declared_metrics_by_id[record.run_id]) != dict(record.metrics):
            raise ValueError(
                "factorized aggregate metrics differ from canonical page rows"
            )
    recomputed = {record.run_id: record for record in select_pareto_records(records)}
    for declared in records:
        expected = recomputed[declared.run_id]
        if declared.status in {"pareto", "family_complete"} and not _factorized_hard_gate_passes(
            declared.metrics
        ):
            raise ValueError(
                "factorized finalist status is not proved by fail-closed metrics"
            )
        expected_status = expected.status
        expected_reason = expected.closure_reason
        if (
            expected_status in {"pareto", "family_complete"}
            and declared.run_id not in complete_output_run_ids
        ):
            expected_status = "information_limited"
            expected_reason = "output_artifact_inventory_missing"
        elif (
            expected_status in {"pareto", "family_complete"}
            and declared.run_id not in complete_runtime_run_ids
        ):
            expected_status = "information_limited"
            expected_reason = "runtime_evidence_ledger_missing"
        if (
            declared.status in {"pareto", "family_complete"}
            and declared.run_id not in complete_output_run_ids
        ):
            raise ValueError(
                "factorized finalist lacks a complete output artifact inventory"
            )
        if (
            declared.status in {"pareto", "family_complete"}
            and declared.run_id not in complete_runtime_run_ids
        ):
            raise ValueError(
                "factorized finalist lacks a complete runtime evidence ledger"
            )
        if declared.status != expected_status:
            raise ValueError(
                "factorized declared status differs from recomputed metrics/Pareto status"
            )
        if declared.closure_reason != expected_reason:
            raise ValueError(
                "factorized closure reason differs from recomputed evidence status"
            )


def _validate_fusion(payload: Mapping[str, object]) -> None:
    ledger, ledger_by_id = _validate_closure_ledger(
        payload, physical_count_field="physical_output_count"
    )
    _require_logical_inventory_binding(
        payload, ledger, upstream_sha_field="spec_sha256"
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
    complete_output_run_ids = _declared_complete_output_run_ids(payload)
    fusion_run_ids = {str(row["run_id"]) for row in runs}
    if not complete_output_run_ids.issubset(fusion_run_ids):
        raise ValueError("fusion output inventory names an unknown run")
    _declared_page_ids(payload, "fusion result")
    pages = payload.get("pages")
    if not isinstance(pages, Mapping) or set(map(str, pages)) != {
        str(row["run_id"]) for row in runs
    }:
        raise ValueError("fusion page statistics do not exactly match runs")
    from scripts.benchmark_inpaint_detector_fusions_v4 import (  # noqa: PLC0415
        _hard_gate_passes as _fusion_hard_gate_passes,
        _product_mask_hard_gate_passes as _fusion_product_mask_hard_gate_passes,
        _seed_gate_passes as _fusion_seed_gate_passes,
        select_seed_admission_run_ids,
    )
    admission = payload.get("seed_admission")
    has_seed_admission = admission is not None
    if has_seed_admission and not isinstance(admission, Mapping):
        raise ValueError("fusion seed admission must be an object")

    for row in runs:
        _validate_run_status(row, "fusion run")
        for field in ("fusion", "primary"):
            if not str(row.get(field) or ""):
                raise ValueError(f"fusion run lacks {field}")
        closure_selection = ledger_by_id[str(row["run_id"])].get("selection")
        if not isinstance(closure_selection, Mapping):
            raise ValueError("fusion closure row lacks selection")
        actual_selection = {
            str(key): str(row.get(key) or "") for key in closure_selection
        }
        if actual_selection != dict(closure_selection):
            raise ValueError(
                "fusion run selection differs from its logical closure inventory"
            )
        metrics = row["metrics"]
        page_rows = pages.get(str(row["run_id"]))
        if not isinstance(page_rows, list) or any(
            not isinstance(page, dict) for page in page_rows
        ):
            raise ValueError("fusion run lacks canonical page statistics")
        if _page_id_set(page_rows, "fusion run pages") != _declared_page_ids(
            payload, "fusion result"
        ):
            raise ValueError("fusion run page IDs differ from declared inventory")
        try:
            canonical_metrics = aggregate_fusion_page_statistics(page_rows)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("fusion page statistics are incomplete") from error
        if dict(metrics) != canonical_metrics:  # type: ignore[arg-type]
            raise ValueError(
                "fusion aggregate metrics differ from canonical page statistics"
            )
        try:
            hard_pass = _fusion_hard_gate_passes(canonical_metrics)
            seed_eligible = _fusion_seed_gate_passes(canonical_metrics)
            product_mask_hard_pass = _fusion_product_mask_hard_gate_passes(
                canonical_metrics
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "fusion metrics do not satisfy the complete fail-closed schema"
            ) from error
        if has_seed_admission:
            if row.get("seed_eligible") is not seed_eligible:
                raise ValueError(
                    "fusion seed eligibility differs from canonical metrics"
                )
            if row.get("product_mask_hard_pass") is not product_mask_hard_pass:
                raise ValueError(
                    "fusion product-mask gate differs from canonical metrics"
                )
        information_limited = any(
            metrics.get(field) is not True  # type: ignore[union-attr]
            for field in (
                "target_extent_independent",
                "target_inventory_independent",
                "target_review_complete",
            )
        )
        output_complete = str(row["run_id"]) in complete_output_run_ids
        expected_status = (
            "family_complete"
            if hard_pass and output_complete
            else (
                "information_limited"
                if hard_pass and not output_complete or information_limited
                else "dominated"
            )
        )
        if str(row.get("status") or "") != expected_status:
            raise ValueError(
                "fusion declared status differs from fail-closed metric status"
            )
        expected_reason = (
            ""
            if hard_pass and output_complete
            else (
                "output_artifact_inventory_missing"
                if hard_pass and not output_complete
                else (
                    "target_extent_not_independent"
                    if metrics.get("target_extent_independent") is not True  # type: ignore[union-attr]
                    else (
                        "target_inventory_not_independent"
                        if metrics.get("target_inventory_independent") is not True  # type: ignore[union-attr]
                        else (
                            "target_review_incomplete"
                            if metrics.get("target_review_complete") is not True  # type: ignore[union-attr]
                            else "hard_gate_failed"
                        )
                    )
                )
            )
        )
        if str(row.get("closure_reason") or "") != expected_reason:
            raise ValueError(
                "fusion closure reason differs from fail-closed metric status"
            )
        output_sha = row["metrics"].get("output_mask_set_sha256")  # type: ignore[index]
        if not _is_sha256(output_sha):
            raise ValueError("fusion run lacks exact output-mask-set SHA")
        closure_sha = str(
            ledger_by_id[str(row["run_id"])].get("content_sha256") or ""
        )
        if str(output_sha) != closure_sha:
            raise ValueError("fusion output SHA differs from closure content SHA")
    if has_seed_admission:
        assert isinstance(admission, Mapping)
        if int(admission.get("runtime_detector_limit") or 0) != 2:
            raise ValueError(
                "fusion seed admission must enforce the two-detector limit"
            )
        selected = select_seed_admission_run_ids(
            [dict(row) for row in runs], limit=2
        )
        if set(map(str, admission.get("selected_run_ids", []))) != set(selected):
            raise ValueError("fusion seed admission differs from canonical selection")
        strict_available = any(bool(row.get("seed_eligible")) for row in runs)
        if admission.get("strict_seed_available") is not strict_available:
            raise ValueError("fusion strict-seed availability is inconsistent")
        for row in runs:
            admitted = str(row["run_id"]) in selected
            if row.get("seed_admitted") is not admitted:
                raise ValueError("fusion run seed-admitted flag is inconsistent")
            expected_kind = (
                "strict" if admitted and strict_available else
                "best_effort" if admitted else "not_selected"
            )
            if str(row.get("seed_admission_kind") or "") != expected_kind:
                raise ValueError("fusion run seed-admission kind is inconsistent")


def _validate_semantic(payload: Mapping[str, object]) -> None:
    policies = _object_rows(payload, "policies", "semantic result")
    if _required_int(payload, "policy_count") != len(policies):
        raise ValueError("semantic policy count differs from policies length")
    if _required_int(payload, "unaccounted_policy_count") != 0:
        raise ValueError("semantic artifact contains unaccounted policies")
    _declared_page_ids(payload, "semantic result")
    _require_unique(policies, "policy_id", "semantic policies")
    expected_policy_ids = frozenset(
        {
            "current_default",
            "detector_explicit_role",
            "ocr_semantic_hint",
            "ocr_provenance_verifier",
            "explicit_role_consensus",
            "human_oracle",
        }
    )
    actual_policy_ids = frozenset(str(row["policy_id"]) for row in policies)
    if actual_policy_ids != expected_policy_ids:
        raise ValueError("semantic artifact does not contain the full policy inventory")
    if str(payload.get("logical_inventory_sha256") or "") != _canonical_sha256(
        sorted(actual_policy_ids)
    ):
        raise ValueError("semantic logical inventory SHA is missing or mismatched")
    pages = payload.get("pages")
    if not isinstance(pages, Mapping) or set(map(str, pages)) != set(
        actual_policy_ids
    ):
        raise ValueError("semantic page statistics do not exactly match policies")
    declared_page_ids = _declared_page_ids(payload, "semantic result")
    for row in policies:
        policy_id = str(row.get("policy_id") or "")
        oracle_only = row.get("oracle_only")
        if oracle_only is not (policy_id == "human_oracle"):
            raise ValueError("semantic oracle flag differs from policy identity")
        metrics = row.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError("semantic policy must contain metrics")
        page_rows = pages.get(policy_id)
        if not isinstance(page_rows, list) or any(
            not isinstance(page, dict) for page in page_rows
        ):
            raise ValueError("semantic policy lacks canonical page statistics")
        if _page_id_set(page_rows, "semantic policy pages") != declared_page_ids:
            raise ValueError("semantic policy page IDs differ from declared inventory")
        try:
            canonical_metrics = aggregate_semantic_page_statistics(page_rows)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("semantic page statistics are incomplete") from error
        if dict(metrics) != canonical_metrics:
            raise ValueError(
                "semantic aggregate metrics differ from canonical page statistics"
            )
        integer_fields = (
            "instance_count",
            "required_instance_count",
            "preserve_instance_count",
            "preserve_destructive_count",
            "ambiguous_instance_count",
            "ambiguous_destructive_count",
            "unavailable_instance_count",
            "no_edit_page_count",
            "no_edit_false_translate_page_count",
            "no_edit_false_translate_region_count",
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
            and values["no_edit_false_translate_page_count"] == 0
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
    try:
        canonical_summary = summarize_stage1_pages(
            [dict(row) for row in pages]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "stage1 pages lack the canonical aggregation inputs"
        ) from error
    if dict(summary) != canonical_summary:
        raise ValueError("stage1 summary differs from canonical page aggregation")
    identities = payload.get("variant_output_identity")
    if not isinstance(identities, Mapping):
        raise ValueError("stage1 result lacks variant output identities")
    for variant in ("raw", "refined", "dilated"):
        identity = identities.get(variant)
        if not isinstance(identity, Mapping):
            raise ValueError(f"stage1 result lacks {variant} output identity")
        if not _is_sha256(identity.get("output_mask_set_sha256")):
            raise ValueError(f"stage1 {variant} output identity lacks SHA-256")
        if identity.get("page_count") != len(pages):
            raise ValueError(f"stage1 {variant} output page count differs")
    if str(payload.get("candidate") or "") == (
        "ctd-synthetic-low-contrast-finetune-v4"
    ):
        raw = identities["raw"]  # type: ignore[index]
        refined = identities["refined"]  # type: ignore[index]
        if (
            refined.get("provenance") != "exact_identity_reuse"
            or refined.get("independent_output") is not False
            or refined.get("source_variant") != "raw"
            or refined.get("source_output_mask_set_sha256")
            != raw.get("output_mask_set_sha256")
            or refined.get("output_mask_set_sha256")
            != raw.get("output_mask_set_sha256")
        ):
            raise ValueError(
                "synthetic fine-tune refined output is not exact raw identity reuse"
            )
    _validate_stage_summary(summary, len(pages))


def _validate_stage1_output_artifacts(
    payload: Mapping[str, object],
    artifact_path: Path,
) -> None:
    """Re-open every PNG that a stage-one result claims as evidence."""

    binding = payload.get("output_artifact_inventory")
    if not isinstance(binding, Mapping):
        raise ValueError("stage1 result lacks its output artifact inventory")
    relative = str(binding.get("relative_path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("stage1 output artifact inventory path must be relative")
    root = artifact_path.resolve().parent
    inventory_path = (root / relative).resolve()
    try:
        inventory_path.relative_to(root)
    except ValueError as error:
        raise ValueError("stage1 output artifact inventory escapes its run") from error
    if not inventory_path.is_file():
        raise ValueError("stage1 output artifact inventory is missing")
    if binding.get("artifact_sha256") != sha256_file(inventory_path):
        raise ValueError("stage1 output artifact inventory file SHA differs")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    if not isinstance(inventory, Mapping) or inventory.get("schema_version") != (
        "inpaint-detector-output-artifact-inventory-v1"
    ):
        raise ValueError("stage1 output artifact inventory schema differs")
    records = inventory.get("records")
    if not isinstance(records, list):
        raise ValueError("stage1 output artifact inventory lacks records")
    if binding.get("artifact_count") != len(records):
        raise ValueError("stage1 output artifact count differs")
    inventory_sha = _canonical_sha256(records)
    if (
        inventory.get("inventory_sha256") != inventory_sha
        or binding.get("inventory_sha256") != inventory_sha
    ):
        raise ValueError("stage1 output artifact inventory SHA differs")

    page_ids = {str(row.get("page_id") or "") for row in payload["pages"]}  # type: ignore[index]
    observed: set[tuple[str, str, str]] = set()
    by_variant: dict[str, list[dict[str, object]]] = {
        "raw": [],
        "refined": [],
        "dilated": [],
    }
    positive: list[dict[str, object]] = []
    for value in records:
        if not isinstance(value, Mapping):
            raise ValueError("stage1 output artifact record must be an object")
        page_id = str(value.get("page_id") or "")
        role = str(value.get("role") or "")
        variant = str(value.get("variant") or "")
        record_id = (role, variant, page_id)
        if page_id not in page_ids or record_id in observed:
            raise ValueError("stage1 output artifact identity is invalid or duplicate")
        observed.add(record_id)
        relative_mask = str(value.get("relative_path") or "")
        if not relative_mask or Path(relative_mask).is_absolute():
            raise ValueError("stage1 output mask path must be relative")
        mask_path = (root / relative_mask).resolve()
        try:
            mask_path.relative_to(root)
        except ValueError as error:
            raise ValueError("stage1 output mask escapes its run") from error
        if not mask_path.is_file() or value.get("artifact_sha256") != sha256_file(
            mask_path
        ):
            raise ValueError("stage1 output mask file SHA differs")
        decoded = cv2.imdecode(
            np.fromfile(mask_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if decoded is None or decoded.size == 0:
            raise ValueError("stage1 output mask cannot be decoded")
        unique = np.unique(decoded)
        if np.any((unique != 0) & (unique != 255)):
            raise ValueError("stage1 output mask is not strict binary")
        normalized = binary_mask(decoded)
        if value.get("binary_mask_sha256") != mask_sha256(normalized):
            raise ValueError("stage1 output decoded mask SHA differs")
        if value.get("pixel_count") != int(cv2.countNonZero(normalized)):
            raise ValueError("stage1 output mask pixel count differs")
        identity = {
            "page_id": page_id,
            "binary_mask_sha256": value["binary_mask_sha256"],
            "pixel_count": value["pixel_count"],
        }
        if role == "native_detector_mask" and variant in by_variant:
            by_variant[variant].append(identity)
        elif role == "positive_edit_mask" and variant == payload.get("variant"):
            positive.append(identity)
        else:
            raise ValueError("stage1 output artifact role or variant is invalid")

    identities = payload["variant_output_identity"]  # type: ignore[index]
    for variant, rows in by_variant.items():
        if {str(row["page_id"]) for row in rows} != page_ids:
            raise ValueError(f"stage1 {variant} output artifact pages differ")
        expected = _canonical_sha256(sorted(rows, key=lambda row: str(row["page_id"])))
        if identities[variant].get("output_mask_set_sha256") != expected:  # type: ignore[index]
            raise ValueError(f"stage1 {variant} output artifact identity differs")
    if {str(row["page_id"]) for row in positive} != page_ids:
        raise ValueError("stage1 positive edit artifact pages differ")
    positive_binding = payload.get("positive_edit_output_identity")
    expected_positive = _canonical_sha256(
        sorted(positive, key=lambda row: str(row["page_id"]))
    )
    if (
        not isinstance(positive_binding, Mapping)
        or positive_binding.get("page_count") != len(page_ids)
        or positive_binding.get("output_mask_set_sha256") != expected_positive
    ):
        raise ValueError("stage1 positive edit output artifact identity differs")


def _resolve_scope_artifact(manifest_path: Path, value: object) -> Path | None:
    if isinstance(value, Mapping):
        value = value.get("path")
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value.strip())
    if not path.is_absolute():
        path = manifest_path.resolve().parent / path
    return path.resolve()


def _decode_scope_image(path: Path, mode: int) -> np.ndarray:
    value = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
    if value is None or value.size == 0:
        raise ValueError(f"sealed scope artifact is unreadable: {path}")
    return np.ascontiguousarray(value)


def _assert_recomputed_fact(
    declared: Mapping[str, object], field: str, expected: object
) -> None:
    actual = declared.get(field)
    if isinstance(expected, float):
        if not _finite_number(actual) or not math.isclose(
            float(actual), expected, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(f"output {field} differs from sealed source artifacts")
        return
    if actual != expected:
        raise ValueError(f"output {field} differs from sealed source artifacts")


_RUNTIME_PAGE_FIELDS = (
    "runtime_seconds",
    "positive_lama_inference_count",
    "positive_lama_call_durations_seconds",
    "runtime_telemetry_complete",
    "cpu_fallback_count",
    "lama_runtime_provider",
    "lama_runtime_precision",
    "peak_vram_allocated_mib",
    "peak_vram_reserved_mib",
)

_RUNTIME_AGGREGATE_FIELDS = (
    "runtime_seconds",
    "positive_lama_inference_count",
    "maximum_positive_lama_inference_per_page",
    "runtime_telemetry_complete",
    "positive_lama_runtime_p95_seconds",
    "peak_vram_allocated_mib",
    "peak_vram_reserved_mib",
    "cpu_fallback_count",
    "lama_runtime_provider",
    "lama_runtime_precision",
)


def _validate_runtime_identity(identity: object) -> Mapping[str, object]:
    if not isinstance(identity, Mapping):
        raise ValueError("runtime evidence lacks a runtime identity")
    commit = str(identity.get("code_commit") or "")
    if len(commit) not in {40, 64} or commit != commit.lower() or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("runtime evidence lacks an exact code commit")
    for field in (
        "python_version",
        "platform",
        "numpy_version",
        "opencv_version",
        "requested_device",
        "requested_precision",
    ):
        if not str(identity.get(field) or "").strip():
            raise ValueError(f"runtime evidence identity lacks {field}")
    if (
        not isinstance(identity.get("tracked_worktree_clean"), bool)
        or not _is_sha256(identity.get("tracked_worktree_diff_sha256"))
        or identity.get("vram_measurement_scope")
        != "page_reset_then_run_max"
        or identity.get("lama_model_asset_id") != "lama_large_512px"
        or not isinstance(identity.get("lama_model_present"), bool)
    ):
        raise ValueError("runtime evidence lacks code/model/VRAM scope identity")
    model_sha = str(identity.get("lama_model_sha256") or "")
    registry_sha = str(identity.get("lama_model_registry_sha256") or "")
    if bool(identity["lama_model_present"]) != bool(_is_sha256(model_sha)):
        raise ValueError("runtime evidence LaMa model identity is invalid")
    if not _is_sha256(registry_sha):
        raise ValueError("runtime evidence lacks registry LaMa SHA")
    requested = str(identity.get("requested_device") or "").lower()
    if requested.startswith("cuda"):
        if identity.get("cuda_available") is not True:
            raise ValueError("CUDA runtime evidence does not prove CUDA availability")
        for field in ("torch_version", "torch_cuda_version", "gpu_name"):
            if not str(identity.get(field) or "").strip():
                raise ValueError(f"CUDA runtime evidence identity lacks {field}")
    return identity


def _reconstruct_runner_source(
    *, code_commit: str, source_path: str, patch_path: Path
) -> bytes:
    """Rebuild the executed runner from the sealed commit and runner-only patch."""

    try:
        baseline = subprocess.check_output(
            ["git", "show", f"{code_commit}:{source_path}"],
            cwd=ROOT,
            stderr=subprocess.PIPE,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("runtime runner commit source is unavailable") from error
    patch_bytes = patch_path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="inpaint-runtime-source-") as directory:
        temporary_root = Path(directory)
        reconstructed = temporary_root / source_path
        reconstructed.parent.mkdir(parents=True, exist_ok=True)
        reconstructed.write_bytes(baseline)
        if patch_bytes:
            try:
                applied = subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.autocrlf=false",
                        "-c",
                        "core.eol=lf",
                        "apply",
                        "--binary",
                        str(patch_path),
                    ],
                    cwd=temporary_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    check=False,
                )
            except OSError as error:
                raise ValueError("runtime runner patch cannot be applied") from error
            if applied.returncode != 0:
                raise ValueError("runtime runner patch cannot be applied")
        materialized = {
            path.relative_to(temporary_root).as_posix()
            for path in temporary_root.rglob("*")
            if path.is_file()
        }
        if materialized != {source_path} or not reconstructed.is_file():
            raise ValueError("runtime runner patch changes an unexpected path")
        return reconstructed.read_bytes()


def _tracked_patch_section(patch_bytes: bytes, source_path: str) -> bytes:
    marker = f"diff --git a/{source_path} b/{source_path}\n".encode("utf-8")
    start = patch_bytes.find(marker)
    if start < 0:
        return b""
    if patch_bytes.find(marker, start + len(marker)) >= 0:
        raise ValueError("runtime tracked diff repeats the runner source")
    end = patch_bytes.find(b"diff --git ", start + len(marker))
    return patch_bytes[start:] if end < 0 else patch_bytes[start:end]


def _validate_runtime_evidence_ledger(
    payload: Mapping[str, object],
    artifact_path: Path,
    *,
    schema: str,
    finalists: frozenset[str],
) -> None:
    if schema != "inpaint-factorized-results-v3":
        return
    binding = payload.get("runtime_evidence_ledger")
    if binding is None:
        if finalists:
            raise ValueError("finalist lacks a sealed runtime evidence ledger")
        return
    if not isinstance(binding, Mapping) or binding.get("role") != "runtime_evidence":
        raise ValueError("runtime evidence ledger binding is invalid")
    relative = str(binding.get("relative_path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("runtime evidence ledger path must be relative")
    root = artifact_path.resolve().parent
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError("runtime evidence ledger escapes its run") from error
    if not path.is_file() or binding.get("artifact_sha256") != sha256_file(path):
        raise ValueError("runtime evidence ledger file SHA differs")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    expected_schema = (
        "inpaint-factorized-runtime-evidence-v1"
    )
    if not isinstance(ledger, Mapping) or ledger.get("schema_version") != expected_schema:
        raise ValueError("runtime evidence ledger schema differs")
    identity = _validate_runtime_identity(ledger.get("runtime_identity"))
    source_binding = ledger.get("runtime_source_inventory")
    result_source_binding = payload.get("runtime_source_inventory")
    if (
        not isinstance(source_binding, Mapping)
        or not isinstance(result_source_binding, Mapping)
        or dict(source_binding) != dict(result_source_binding)
        or source_binding.get("role") != "runtime_source_inventory"
    ):
        raise ValueError("runtime source inventory binding differs")
    source_relative = str(source_binding.get("relative_path") or "")
    if not source_relative or Path(source_relative).is_absolute():
        raise ValueError("runtime source inventory path must be relative")
    source_path = (root / source_relative).resolve()
    try:
        source_path.relative_to(root)
    except ValueError as error:
        raise ValueError("runtime source inventory escapes its run") from error
    if (
        not source_path.is_file()
        or source_binding.get("artifact_sha256") != sha256_file(source_path)
    ):
        raise ValueError("runtime source inventory file SHA differs")
    source_inventory = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        not isinstance(source_inventory, Mapping)
        or source_inventory.get("schema_version")
        != "inpaint-factorized-runtime-source-inventory-v1"
    ):
        raise ValueError("runtime source inventory schema differs")
    source_records = source_inventory.get("records")
    model_inventory = source_inventory.get("lama_model")
    if (
        not isinstance(source_records, list)
        or any(not isinstance(row, Mapping) for row in source_records)
        or not isinstance(model_inventory, Mapping)
    ):
        raise ValueError("runtime source inventory records are invalid")
    source_canonical = {
        "code_commit": source_inventory.get("code_commit"),
        "tracked_worktree_clean": source_inventory.get("tracked_worktree_clean"),
        "records": source_records,
        "lama_model": dict(model_inventory),
    }
    source_inventory_sha = _canonical_sha256(source_canonical)
    if (
        source_inventory.get("inventory_sha256") != source_inventory_sha
        or source_binding.get("inventory_sha256") != source_inventory_sha
        or binding.get("runtime_source_inventory_sha256")
        != source_inventory_sha
    ):
        raise ValueError("runtime source inventory canonical SHA differs")
    if source_inventory.get("code_commit") != identity.get("code_commit"):
        raise ValueError("runtime source inventory commit differs")
    seen_source_roles: set[str] = set()
    patch_empty: bool | None = None
    tracked_patch_bytes: bytes | None = None
    runner_snapshot_bytes: bytes | None = None
    runner_patch_path: Path | None = None
    runner_patch_bytes: bytes | None = None
    for record in source_records:
        role = str(record.get("role") or "")
        relative_artifact = str(record.get("relative_path") or "")
        if role in seen_source_roles or role not in {
            "runner_source_snapshot",
            "tracked_diff_patch",
            "runner_diff_patch",
        } or not relative_artifact or Path(relative_artifact).is_absolute():
            raise ValueError("runtime source artifact record is invalid")
        if role in {"runner_source_snapshot", "runner_diff_patch"} and record.get(
            "source_path"
        ) != "scripts/benchmark_inpaint_factorized_v3.py":
            raise ValueError("runtime runner source path differs")
        seen_source_roles.add(role)
        artifact = (root / relative_artifact).resolve()
        try:
            artifact.relative_to(root)
        except ValueError as error:
            raise ValueError("runtime source artifact escapes its run") from error
        if not artifact.is_file() or record.get("artifact_sha256") != sha256_file(
            artifact
        ):
            raise ValueError("runtime source artifact SHA differs")
        artifact_bytes = artifact.read_bytes()
        if role == "runner_source_snapshot" and record.get(
            "source_bytes_sha256"
        ) != hashlib.sha256(artifact_bytes).hexdigest():
            raise ValueError("runtime runner source snapshot identity differs")
        if role == "runner_source_snapshot":
            runner_snapshot_bytes = artifact_bytes
        if role == "tracked_diff_patch":
            if record.get("byte_count") != len(artifact_bytes):
                raise ValueError("runtime tracked diff patch size differs")
            patch_empty = len(artifact_bytes) == 0
            tracked_patch_bytes = artifact_bytes
        if role == "runner_diff_patch":
            if record.get("byte_count") != len(artifact_bytes):
                raise ValueError("runtime runner diff patch size differs")
            runner_patch_path = artifact
            runner_patch_bytes = artifact_bytes
    if seen_source_roles != {
        "runner_source_snapshot",
        "tracked_diff_patch",
        "runner_diff_patch",
    }:
        raise ValueError("runtime source inventory is incomplete")
    if (
        patch_empty is None
        or source_inventory.get("tracked_worktree_clean") is not patch_empty
        or identity.get("tracked_worktree_clean") is not patch_empty
        or identity.get("tracked_worktree_diff_sha256")
        != next(
            str(row["artifact_sha256"])
            for row in source_records
            if row["role"] == "tracked_diff_patch"
        )
    ):
        raise ValueError("runtime tracked worktree identity differs from patch bytes")
    if (
        tracked_patch_bytes is None
        or runner_snapshot_bytes is None
        or runner_patch_path is None
        or runner_patch_bytes is None
    ):
        raise ValueError("runtime runner source inventory is incomplete")
    if _tracked_patch_section(
        tracked_patch_bytes, "scripts/benchmark_inpaint_factorized_v3.py"
    ) != runner_patch_bytes:
        raise ValueError("runtime runner patch differs from tracked diff patch")
    reconstructed_runner = _reconstruct_runner_source(
        code_commit=str(identity["code_commit"]),
        source_path="scripts/benchmark_inpaint_factorized_v3.py",
        patch_path=runner_patch_path,
    )
    if reconstructed_runner != runner_snapshot_bytes:
        raise ValueError("runtime runner snapshot differs from commit plus patch")
    actual_pre_sha = str(model_inventory.get("actual_pre_sha256") or "")
    actual_post_sha = str(model_inventory.get("actual_post_sha256") or "")
    if (
        not isinstance(model_inventory.get("present_pre"), bool)
        or not isinstance(model_inventory.get("present_post"), bool)
        or bool(model_inventory["present_pre"]) != _is_sha256(actual_pre_sha)
        or bool(model_inventory["present_post"]) != _is_sha256(actual_post_sha)
    ):
        raise ValueError("runtime LaMa source inventory has invalid asset SHA")
    if (
        model_inventory.get("asset_id") != identity.get("lama_model_asset_id")
        or model_inventory.get("registry_expected_sha256")
        != identity.get("lama_model_registry_sha256")
        or model_inventory.get("actual_pre_sha256")
        != identity.get("lama_model_sha256")
        or model_inventory.get("present_pre")
        is not identity.get("lama_model_present")
        or model_inventory.get("present_post")
        is not bool(model_inventory.get("actual_post_sha256"))
    ):
        raise ValueError("runtime LaMa source inventory differs from identity")
    from modules.utils.download import ModelDownloader, ModelID  # noqa: PLC0415

    registry_expected = str(
        ModelDownloader.registry[ModelID.LAMA_LARGE_512PX].sha256[0] or ""
    )
    if model_inventory.get("registry_expected_sha256") != registry_expected:
        raise ValueError("runtime LaMa registry expected SHA differs")
    if identity.get("requested_device", "").lower().startswith("cuda") and (
        model_inventory.get("actual_pre_sha256")
        != model_inventory.get("actual_post_sha256")
        or model_inventory.get("actual_pre_sha256")
        != model_inventory.get("registry_expected_sha256")
    ):
        raise ValueError("CUDA runtime LaMa bytes differ from registry SHA")
    run_values = ledger.get("runs")
    complete_values = ledger.get("complete_run_ids")
    if not isinstance(run_values, list) or any(
        not isinstance(row, Mapping) for row in run_values
    ):
        raise ValueError("runtime evidence run records are invalid")
    if not isinstance(complete_values, list) or any(
        not isinstance(value, str) or not value.strip() for value in complete_values
    ) or len(complete_values) != len(set(complete_values)):
        raise ValueError("runtime evidence complete run IDs are invalid")
    canonical = {
        "runtime_identity": dict(identity),
        "runtime_source_inventory": dict(source_binding),
        "runs": run_values,
        "complete_run_ids": complete_values,
        "positive_lama_inference_count": ledger.get(
            "positive_lama_inference_count"
        ),
    }
    ledger_sha = _canonical_sha256(canonical)
    if (
        ledger.get("ledger_sha256") != ledger_sha
        or binding.get("ledger_sha256") != ledger_sha
        or binding.get("complete_run_ids") != complete_values
        or binding.get("positive_lama_inference_count")
        != ledger.get("positive_lama_inference_count")
    ):
        raise ValueError("runtime evidence ledger binding differs")
    complete = frozenset(complete_values)
    runs = _object_rows(payload, "runs", "runtime evidence result")
    result_run_ids = [str(row["run_id"]) for row in runs]
    run_by_id = {str(row["run_id"]): row for row in runs}
    pages_by_run = payload.get("pages")
    if not isinstance(pages_by_run, Mapping):
        raise ValueError("runtime evidence result lacks page facts")
    ledger_by_id: dict[str, Mapping[str, object]] = {}
    for row in run_values:
        run_id = str(row.get("run_id") or "")
        if not run_id or run_id in ledger_by_id or run_id not in run_by_id:
            raise ValueError("runtime evidence names an invalid or duplicate run")
        if row.get("runtime_identity") != identity:
            raise ValueError("runtime evidence run identity differs from ledger identity")
        ledger_by_id[run_id] = row
    if set(ledger_by_id) != set(run_by_id) or complete != frozenset(run_by_id):
        raise ValueError("runtime evidence must cover every result run exactly")
    if not finalists.issubset(complete):
        raise ValueError("finalist runtime evidence ledger is incomplete")

    ledger_total = ledger.get("positive_lama_inference_count")
    if (
        not isinstance(ledger_total, int)
        or isinstance(ledger_total, bool)
        or ledger_total < 0
        or payload.get("positive_lama_inference_count") != ledger_total
    ):
        raise ValueError("runtime evidence total differs from result inference count")
    global_call_indices: set[int] = set()
    global_event_count = 0
    global_call_cursor = 1
    for run_id in result_run_ids:
        run = run_by_id[run_id]
        runtime_run = ledger_by_id[run_id]
        fill_id = str(run.get("fill_id") or "")
        result_pages = pages_by_run.get(run_id)
        runtime_pages = runtime_run.get("pages")
        if not isinstance(result_pages, list) or any(
            not isinstance(row, Mapping) for row in result_pages
        ) or not isinstance(runtime_pages, list) or any(
            not isinstance(row, Mapping) for row in runtime_pages
        ):
            raise ValueError("runtime evidence page records are invalid")
        result_by_page = {
            str(row.get("page_id") or ""): row for row in result_pages
        }
        runtime_by_page = {
            str(row.get("page_id") or ""): row for row in runtime_pages
        }
        if (
            not all(result_by_page)
            or len(result_by_page) != len(result_pages)
            or not all(runtime_by_page)
            or len(runtime_by_page) != len(runtime_pages)
            or set(result_by_page) != set(runtime_by_page)
        ):
            raise ValueError("runtime evidence page inventory differs")
        seen_call_indices: set[int] = set()
        ordered_call_indices: list[int] = []
        for page_id, result_page in result_by_page.items():
            declared_page = (
                result_page.get("canonical_statistics")
                if schema == "inpaint-factorized-results-v3"
                else result_page
            )
            if not isinstance(declared_page, Mapping):
                raise ValueError("runtime evidence result page lacks canonical facts")
            runtime_page = runtime_by_page[page_id]
            expected_summary = {
                field: declared_page.get(field) for field in _RUNTIME_PAGE_FIELDS
            }
            if runtime_page.get("summary") != expected_summary:
                raise ValueError("runtime page summary differs from result page facts")
            summary = runtime_page["summary"]
            if not isinstance(summary, Mapping):
                raise ValueError("runtime page summary must be an object")
            runtime_seconds = summary.get("runtime_seconds")
            count = summary.get("positive_lama_inference_count")
            durations = summary.get("positive_lama_call_durations_seconds")
            fallback_count = summary.get("cpu_fallback_count")
            if (
                not _finite_number(runtime_seconds)
                or float(runtime_seconds) < 0.0
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count < 0
                or count > 1
                or not isinstance(fallback_count, int)
                or isinstance(fallback_count, bool)
                or fallback_count < 0
                or summary.get("runtime_telemetry_complete") is not True
                or not isinstance(durations, list)
                or len(durations) != count
            ):
                raise ValueError("runtime page summary is incomplete or invalid")
            events = runtime_page.get("inference_events")
            if not isinstance(events, list) or len(events) != count or any(
                not isinstance(event, Mapping) for event in events
            ):
                raise ValueError("runtime inference event count differs")
            event_durations: list[float] = []
            event_providers: set[str] = set()
            event_precisions: set[str] = set()
            event_backends: set[str] = set()
            event_fallbacks = 0
            for event in events:
                call_index = event.get("call_index")
                duration = event.get("duration_seconds")
                if (
                    not isinstance(call_index, int)
                    or isinstance(call_index, bool)
                    or call_index < 1
                    or call_index in seen_call_indices
                    or call_index in global_call_indices
                    or not _finite_number(duration)
                    or float(duration) < 0.0
                    or not str(event.get("backend") or "").strip()
                    or not isinstance(event.get("cpu_fallback"), bool)
                ):
                    raise ValueError("runtime inference event is invalid")
                if call_index != global_call_cursor:
                    raise ValueError(
                        "runtime global call inventory order differs from result order"
                    )
                global_call_cursor += 1
                provider = str(event.get("provider") or "").strip()
                precision = str(event.get("precision") or "").strip()
                if not provider or not precision:
                    raise ValueError("runtime inference event lacks provider/precision")
                seen_call_indices.add(call_index)
                global_call_indices.add(call_index)
                global_event_count += 1
                ordered_call_indices.append(call_index)
                event_durations.append(float(duration))
                event_providers.add(provider)
                event_precisions.add(precision)
                event_backends.add(str(event["backend"]))
                event_fallbacks += int(bool(event["cpu_fallback"]))
            if event_durations != [float(value) for value in durations]:
                raise ValueError("runtime event durations differ from page summary")
            if sum(event_durations) > float(runtime_seconds) + 1e-12:
                raise ValueError("runtime inference duration exceeds page runtime")
            if event_fallbacks != fallback_count:
                raise ValueError("runtime fallback events differ from page summary")
            expected_provider = next(iter(event_providers), "")
            expected_precision = next(iter(event_precisions), "")
            if (
                len(event_providers) > 1
                or len(event_precisions) > 1
                or len(event_backends) > 1
                or (
                str(summary.get("lama_runtime_provider") or "") != expected_provider
                or str(summary.get("lama_runtime_precision") or "")
                != expected_precision
                )
            ):
                raise ValueError("runtime provider/precision differs from events")
            expected_backends = {
                "ballons_lama" if fill_id == "ballons_lama" else "current_lama"
            }
            if events and (
                fill_id == "mask_only"
                or not event_backends.issubset(expected_backends)
            ):
                raise ValueError("runtime inference backend differs from run selection")
            requested_device = str(identity.get("requested_device") or "").lower()
            requested_precision = str(
                identity.get("requested_precision") or ""
            ).lower()
            actual_cuda = expected_provider.lower().startswith("cuda")
            precision_aliases = {
                "bf16": {"bf16", "bfloat16", "torch.bfloat16"},
                "fp32": {"fp32", "float32", "torch.float32"},
            }
            precision_matches = expected_precision.lower() in precision_aliases.get(
                requested_precision, {requested_precision}
            )
            if events and (
                actual_cuda != requested_device.startswith("cuda")
                or not precision_matches
            ):
                raise ValueError("runtime provider/precision differs from requested runtime")
            allocated = summary.get("peak_vram_allocated_mib")
            reserved = summary.get("peak_vram_reserved_mib")
            cuda_runtime = bool(events) and expected_provider.lower().startswith("cuda")
            if cuda_runtime:
                if (
                    identity.get("lama_model_present") is not True
                    or not _is_sha256(identity.get("lama_model_sha256"))
                ):
                    raise ValueError("CUDA runtime evidence lacks LaMa weight SHA")
                if (
                    not _finite_number(allocated)
                    or not _finite_number(reserved)
                    or float(allocated) <= 0.0
                    or float(reserved) < float(allocated)
                ):
                    raise ValueError("CUDA runtime evidence has invalid peak VRAM")
            elif allocated is not None or reserved is not None:
                raise ValueError("non-CUDA runtime evidence claims peak VRAM")

        if ordered_call_indices:
            first_call = ordered_call_indices[0]
            if ordered_call_indices != list(
                range(first_call, first_call + len(ordered_call_indices))
            ):
                raise ValueError("runtime inference call ordinals are not contiguous")

        canonical_metrics = (
            aggregate_factorized_page_statistics(result_pages)
            if schema == "inpaint-factorized-results-v3"
            else aggregate_fusion_page_statistics(result_pages)
        )
        expected_aggregate = {
            field: canonical_metrics.get(field)
            for field in _RUNTIME_AGGREGATE_FIELDS
        }
        if runtime_run.get("aggregate") != expected_aggregate:
            raise ValueError("runtime aggregate differs from canonical page telemetry")
        declared_metrics = run.get("metrics")
        if not isinstance(declared_metrics, Mapping) or {
            field: declared_metrics.get(field) for field in _RUNTIME_AGGREGATE_FIELDS
        } != expected_aggregate:
            raise ValueError("runtime aggregate differs from declared run metrics")

    if (
        global_event_count != ledger_total
        or global_call_cursor != ledger_total + 1
        or sorted(global_call_indices) != list(range(1, ledger_total + 1))
    ):
        raise ValueError("runtime global call inventory differs from result total")


def _validate_finalist_output_artifacts(
    payload: Mapping[str, object],
    artifact_path: Path,
    scope_manifest_path: Path,
) -> None:
    """Re-open factorized/fusion output bytes before accepting finalist evidence."""

    schema = str(payload.get("schema_version") or "")
    if schema not in {
        "inpaint-factorized-results-v3",
        "inpaint-detector-fusion-results-v4",
    }:
        return
    runs = _object_rows(payload, "runs", "output evidence")
    finalists = {
        str(row.get("run_id") or "")
        for row in runs
        if str(row.get("status") or "") in {"pareto", "family_complete"}
    }
    binding = payload.get("output_artifact_inventory")
    if binding is None:
        if finalists:
            raise ValueError("finalist lacks a sealed output artifact inventory")
        return
    if not isinstance(binding, Mapping):
        raise ValueError("output artifact inventory binding must be an object")
    relative = str(binding.get("relative_path") or "")
    if not relative or Path(relative).is_absolute():
        raise ValueError("output artifact inventory path must be relative")
    root = artifact_path.resolve().parent
    inventory_path = (root / relative).resolve()
    try:
        inventory_path.relative_to(root)
    except ValueError as error:
        raise ValueError("output artifact inventory escapes its run") from error
    if not inventory_path.is_file():
        raise ValueError("output artifact inventory is missing")
    if binding.get("artifact_sha256") != sha256_file(inventory_path):
        raise ValueError("output artifact inventory file SHA differs")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    expected_schema = (
        "inpaint-factorized-output-artifact-inventory-v1"
        if schema == "inpaint-factorized-results-v3"
        else "inpaint-fusion-output-artifact-inventory-v1"
    )
    if not isinstance(inventory, Mapping) or inventory.get("schema_version") != expected_schema:
        raise ValueError("output artifact inventory schema differs")
    records = inventory.get("records")
    complete_values = inventory.get("complete_run_ids")
    if not isinstance(records, list) or any(
        not isinstance(value, Mapping) for value in records
    ):
        raise ValueError("output artifact inventory records are invalid")
    if not isinstance(complete_values, list) or any(
        not isinstance(value, str) or not value.strip() for value in complete_values
    ):
        raise ValueError("output artifact inventory complete run IDs are invalid")
    if len(complete_values) != len(set(complete_values)):
        raise ValueError("output artifact inventory repeats a complete run ID")
    canonical = {
        "records": records,
        "complete_run_ids": complete_values,
    }
    inventory_sha = _canonical_sha256(canonical)
    if (
        inventory.get("inventory_sha256") != inventory_sha
        or binding.get("inventory_sha256") != inventory_sha
        or binding.get("artifact_count") != len(records)
        or binding.get("complete_run_ids") != complete_values
    ):
        raise ValueError("output artifact inventory binding differs")

    run_by_id = {str(row["run_id"]): row for row in runs}
    complete_run_ids = frozenset(complete_values)
    if not complete_run_ids.issubset(run_by_id):
        raise ValueError("output artifact inventory names an unknown run")
    if not finalists.issubset(complete_run_ids):
        raise ValueError("finalist output artifact inventory is incomplete")
    _validate_runtime_evidence_ledger(
        payload,
        artifact_path,
        schema=schema,
        finalists=frozenset(finalists),
    )
    pages = payload.get("pages")
    if not isinstance(pages, Mapping):
        raise ValueError("output evidence lacks canonical pages")
    page_ids_by_run: dict[str, frozenset[str]] = {}
    for run_id, values in pages.items():
        if not isinstance(values, list) or any(
            not isinstance(value, Mapping) for value in values
        ):
            raise ValueError("output evidence page rows are invalid")
        page_ids_by_run[str(run_id)] = _page_id_set(
            list(values), "output evidence pages"
        )

    observed: dict[tuple[str, str, str], Mapping[str, object]] = {}
    seen_keys: set[tuple[str, str, str]] = set()
    decoded_sha: dict[tuple[str, str, str], str] = {}
    decoded_values: dict[tuple[str, str, str], np.ndarray] = {}
    for value in records:
        run_id = str(value.get("run_id") or "")
        page_id = str(value.get("page_id") or "")
        role = str(value.get("role") or "")
        key = (run_id, page_id, role)
        allowed_roles = (
            {
                "detector_seed_mask",
                "edit_mask",
                "final_mask",
                "candidate_image",
            }
            if schema == "inpaint-factorized-results-v3"
            else {"claim_mask", "edit_mask"}
        )
        if (
            run_id not in run_by_id
            or not page_id
            or page_id not in page_ids_by_run.get(run_id, frozenset())
            or role not in allowed_roles
            or key in seen_keys
        ):
            raise ValueError("output artifact identity is invalid or duplicate")
        seen_keys.add(key)
        if run_id not in finalists:
            continue
        relative_artifact = str(value.get("relative_path") or "")
        if not relative_artifact or Path(relative_artifact).is_absolute():
            raise ValueError("output artifact path must be relative")
        path = (root / relative_artifact).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError("output artifact escapes its run") from error
        if not path.is_file() or value.get("artifact_sha256") != sha256_file(path):
            raise ValueError("output artifact file SHA differs")
        mode = cv2.IMREAD_COLOR if role == "candidate_image" else cv2.IMREAD_GRAYSCALE
        decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), mode)
        if decoded is None or decoded.size == 0:
            raise ValueError("output artifact is not a decodable image")
        if role != "candidate_image" and not set(np.unique(decoded)).issubset(
            {0, 255}
        ):
            raise ValueError("output artifact mask is not strict binary")
        decoded = np.ascontiguousarray(decoded)
        pixel_sha = hashlib.sha256(decoded.tobytes()).hexdigest()
        if (
            value.get("pixel_sha256") != pixel_sha
            or value.get("shape") != list(decoded.shape)
            or value.get("dtype") != str(decoded.dtype)
        ):
            raise ValueError("output artifact decoded identity differs")
        if role != "candidate_image" and value.get(
            "foreground_pixel_count"
        ) != int(np.count_nonzero(decoded)):
            raise ValueError("output artifact foreground count differs")
        observed[key] = value
        decoded_sha[key] = pixel_sha
        decoded_values[key] = decoded

    manifest_payload = json.loads(scope_manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest_payload, Mapping):
        raise ValueError("sealed scope manifest root must be an object")
    raw_pages = manifest_payload.get("pages")
    if not isinstance(raw_pages, list) or any(
        not isinstance(value, Mapping) for value in raw_pages
    ):
        raise ValueError("sealed scope manifest lacks page records")
    raw_by_id = {str(value.get("page_id") or ""): value for value in raw_pages}
    stage_by_id = {
        page.page_id: page for page in load_stage1_manifest(scope_manifest_path)
    }

    for run_id in finalists:
        page_rows = pages.get(run_id)
        if not isinstance(page_rows, list) or any(
            not isinstance(row, Mapping) for row in page_rows
        ):
            raise ValueError("complete output run lacks canonical page rows")
        expected_roles = (
            (
                "detector_seed_mask",
                "edit_mask",
                "final_mask",
                "candidate_image",
            )
            if schema == "inpaint-factorized-results-v3"
            else ("claim_mask", "edit_mask")
        )
        for page in page_rows:
            page_id = str(page.get("page_id") or "")
            if any((run_id, page_id, role) not in observed for role in expected_roles):
                raise ValueError("complete output run lacks a page artifact")
            if schema == "inpaint-factorized-results-v3":
                canonical_page = page.get("canonical_statistics")
                if not isinstance(canonical_page, Mapping):
                    raise ValueError("factorized page lacks canonical output facts")
                expected = {
                    "detector_seed_mask": str(
                        canonical_page.get("detector_seed_mask_pixel_sha256") or ""
                    ),
                    "edit_mask": str(
                        canonical_page.get("output_edit_mask_pixel_sha256") or ""
                    ),
                    "final_mask": str(
                        canonical_page.get("final_mask_pixel_sha256") or ""
                    ),
                    "candidate_image": str(
                        canonical_page.get("candidate_pixel_sha256") or ""
                    ),
                }
            else:
                expected = {
                    "claim_mask": str(
                        page.get("output_claim_mask_pixel_sha256") or ""
                    ),
                    "edit_mask": str(
                        page.get("output_edit_mask_pixel_sha256") or ""
                    )
                }
            for role, expected_sha in expected.items():
                if decoded_sha[(run_id, page_id, role)] != expected_sha:
                    raise ValueError(
                        "output artifact pixel SHA differs from canonical page facts"
                    )
            edit_record = observed[(run_id, page_id, "edit_mask")]
            canonical_edit_count = (
                canonical_page.get("edit_pixel_count")
                if schema == "inpaint-factorized-results-v3"
                else page.get("edit_pixel_count")
            )
            if (
                not isinstance(canonical_edit_count, int)
                or isinstance(canonical_edit_count, bool)
                or canonical_edit_count < 0
                or edit_record.get("foreground_pixel_count")
                != canonical_edit_count
            ):
                raise ValueError(
                    "output edit mask pixel count differs from canonical page facts"
                )

            stage_page = stage_by_id.get(page_id)
            raw_page = raw_by_id.get(page_id)
            if stage_page is None or not isinstance(raw_page, Mapping):
                raise ValueError("output page is absent from the sealed source manifest")
            source = _decode_scope_image(
                Path(stage_page.source_image), cv2.IMREAD_COLOR
            )
            shape = source.shape[:2]
            existing_path = _resolve_scope_artifact(
                scope_manifest_path, raw_page.get("existing_source_edit_mask")
            )
            masks = load_page_masks(
                stage_page,
                shape,
                existing_edit_path=str(existing_path) if existing_path else None,
                strict_binary=True,
            )
            edit = decoded_values[(run_id, page_id, "edit_mask")]
            if edit.shape != shape:
                raise ValueError("output edit mask shape differs from sealed source")
            target_pixels = int(np.count_nonzero(masks.target))
            target_edit_pixels = int(
                np.count_nonzero((masks.target > 0) & (edit > 0))
            )
            common_facts = {
                "target_pixel_count": target_pixels,
                "target_edit_pixel_count": target_edit_pixels,
                "protected_overlap": int(
                    np.count_nonzero((edit > 0) & (masks.protected > 0))
                ),
                "ambiguous_overlap": int(
                    np.count_nonzero((edit > 0) & (masks.ambiguous > 0))
                ),
                "preserve_overlap": int(
                    np.count_nonzero((edit > 0) & (masks.preserve > 0))
                    if masks.preserve is not None
                    else 0
                ),
            }
            if schema == "inpaint-factorized-results-v3":
                seed = decoded_values[(run_id, page_id, "detector_seed_mask")]
                final_mask = decoded_values[(run_id, page_id, "final_mask")]
                candidate = decoded_values[(run_id, page_id, "candidate_image")]
                if seed.shape != shape or final_mask.shape != shape or candidate.shape[:2] != shape:
                    raise ValueError("factorized output shape differs from sealed source")
                baseline_path = _resolve_scope_artifact(
                    scope_manifest_path, raw_page.get("baseline")
                )
                baseline = (
                    _decode_scope_image(baseline_path, cv2.IMREAD_COLOR)
                    if baseline_path is not None
                    else source.copy()
                )
                baseline_mask_path = _resolve_scope_artifact(
                    scope_manifest_path, raw_page.get("baseline_mask")
                )
                baseline_mask = (
                    _decode_scope_image(
                        baseline_mask_path, cv2.IMREAD_GRAYSCALE
                    )
                    if baseline_mask_path is not None
                    else np.zeros(shape, np.uint8)
                )
                expected_final = cv2.bitwise_or(baseline_mask, edit)
                if not np.array_equal(final_mask, expected_final):
                    raise ValueError(
                        "factorized final mask differs from sealed baseline plus edit"
                    )
                if np.any(candidate[edit == 0] != baseline[edit == 0]):
                    raise ValueError(
                        "factorized candidate changed immutable pixels outside edit"
                    )
                changed = np.where(
                    np.any(candidate != source, axis=2), 255, 0
                ).astype(np.uint8)
                seed_scores: list[dict[str, object]] = []
                edit_scores: list[dict[str, object]] = []
                for instance in stage_page.target_instances:
                    if instance.priority != "required":
                        continue
                    instance_mask = _decode_scope_image(
                        Path(instance.mask_path), cv2.IMREAD_GRAYSCALE
                    )
                    pixels = int(np.count_nonzero(instance_mask))
                    seed_scores.append(
                        {
                            "instance_id": instance.instance_id,
                            "semantic_role": instance.semantic_role,
                            "seeded": bool(
                                np.count_nonzero(
                                    (seed > 0) & (instance_mask > 0)
                                )
                            ),
                        }
                    )
                    edit_scores.append(
                        {
                            "instance_id": instance.instance_id,
                            "coverage": (
                                float(
                                    np.count_nonzero(
                                        (edit > 0) & (instance_mask > 0)
                                    )
                                )
                                / float(pixels)
                                if pixels
                                else 0.0
                            ),
                        }
                    )
                clean_union = np.zeros(shape, np.uint8)
                overlap_seen = np.zeros(shape, np.uint8)
                overlap = np.zeros(shape, np.uint8)
                for region in masks.regions:
                    overlap[(overlap_seen > 0) & (region.ownership > 0)] = 255
                    overlap_seen[region.ownership > 0] = 255
                    if region.bubble_route_class in {"clean_flat", "clean_gradient"}:
                        clean_union[region.bubble_interior > 0] = 255
                broad_only = cv2.bitwise_and(edit, cv2.bitwise_not(seed))
                broad_false = (
                    int(np.count_nonzero((broad_only > 0) & (clean_union == 0)))
                    if masks.regions
                    else int(np.count_nonzero(broad_only))
                    if stage_page.bubble_route_class
                    not in {"clean_flat", "clean_gradient"}
                    else 0
                )
                overlap_edit = int(
                    np.count_nonzero((overlap > 0) & (edit > 0))
                )
                residue, _residue_sum, residue_count = residue_score(
                    source, candidate, masks.target
                )
                baseline_residue, _baseline_sum, _baseline_count = residue_score(
                    source, baseline, masks.target
                )
                known_path = _resolve_scope_artifact(
                    scope_manifest_path, raw_page.get("known_background")
                )
                reconstruction = (
                    reconstruction_error(
                        candidate,
                        _decode_scope_image(known_path, cv2.IMREAD_COLOR),
                        edit,
                    )
                    if known_path is not None
                    else None
                )
                factorized_facts: dict[str, object] = {
                    "target_extent_independent": stage_page.target_extent_independent,
                    "target_inventory_independent": stage_page.target_inventory_independent,
                    "target_review_complete": stage_page.target_review_complete,
                    "target_mask_provenance": stage_page.target_mask_provenance,
                    "no_edit": stage_page.no_edit,
                    "required_skip": not stage_page.no_edit and not np.any(edit),
                    "target_pixel_count": target_pixels,
                    "target_edit_pixel_count": target_edit_pixels,
                    "target_instance_seed_scores": seed_scores,
                    "target_instance_edit_scores": edit_scores,
                    "edit_pixel_count": int(np.count_nonzero(edit)),
                    "protected_structure_overlap_pixel_count": common_facts[
                        "protected_overlap"
                    ],
                    "protected_structure_changed_pixel_count": int(
                        np.count_nonzero((masks.protected > 0) & (changed > 0))
                    ),
                    "preserve_edit_overlap_pixel_count": common_facts[
                        "preserve_overlap"
                    ],
                    "ambiguous_structure_overlap_pixel_count": common_facts[
                        "ambiguous_overlap"
                    ],
                    "ambiguous_structure_changed_pixel_count": int(
                        np.count_nonzero((masks.ambiguous > 0) & (changed > 0))
                    ),
                    "outside_final_changed_pixel_count": int(
                        np.count_nonzero((changed > 0) & (final_mask == 0))
                    ),
                    "broad_route_false_positive_pixel_count": broad_false,
                    "conditional_hybrid_overlap_conflict_pixel_count": overlap_edit
                    if str(run_by_id[run_id].get("fill_id") or "")
                    == "conditional_hybrid"
                    else 0,
                    "authoritative_region_overlap_pixel_count": int(
                        np.count_nonzero(overlap)
                    )
                    if str(run_by_id[run_id].get("fill_id") or "")
                    == "conditional_hybrid"
                    else 0,
                    "authoritative_overlap_narrow_verified": (
                        not np.any((overlap > 0) & (edit > 0) & (seed == 0))
                    )
                    if str(run_by_id[run_id].get("fill_id") or "")
                    == "conditional_hybrid"
                    else False,
                    "residue_score": residue,
                    "baseline_residue_score": baseline_residue,
                    "residue_source_contrast_pixel_count": residue_count,
                    "reconstruction_mse": reconstruction,
                    "residue_gate_applicable": str(
                        run_by_id[run_id].get("fill_id") or ""
                    )
                    != "mask_only",
                    "detector_seed_mask_pixel_sha256": decoded_sha[
                        (run_id, page_id, "detector_seed_mask")
                    ],
                    "output_edit_mask_pixel_sha256": decoded_sha[
                        (run_id, page_id, "edit_mask")
                    ],
                    "final_mask_pixel_sha256": decoded_sha[
                        (run_id, page_id, "final_mask")
                    ],
                    "candidate_pixel_sha256": decoded_sha[
                        (run_id, page_id, "candidate_image")
                    ],
                }
                for field, expected_value in factorized_facts.items():
                    _assert_recomputed_fact(canonical_page, field, expected_value)
            else:
                claim = decoded_values[(run_id, page_id, "claim_mask")]
                if claim.shape != shape:
                    raise ValueError("fusion claim shape differs from sealed source")
                expected_edit = positive_edit_from_claim(claim, masks)
                if not np.array_equal(edit, expected_edit):
                    raise ValueError(
                        "fusion edit differs from claim and sealed source protection"
                    )
                scores: list[dict[str, object]] = []
                for instance in stage_page.target_instances:
                    if instance.priority != "required":
                        continue
                    instance_mask = _decode_scope_image(
                        Path(instance.mask_path), cv2.IMREAD_GRAYSCALE
                    )
                    pixels = int(np.count_nonzero(instance_mask))
                    scores.append(
                        {
                            "instance_id": instance.instance_id,
                            "seeded": bool(
                                np.count_nonzero(
                                    (claim > 0) & (instance_mask > 0)
                                )
                            ),
                            "coverage": (
                                float(
                                    np.count_nonzero(
                                        (edit > 0) & (instance_mask > 0)
                                    )
                                )
                                / float(pixels)
                                if pixels
                                else 0.0
                            ),
                        }
                    )
                fusion_facts: dict[str, object] = {
                    "target_pixel_count": target_pixels,
                    "target_edit_pixel_count": target_edit_pixels,
                    "target_instance_scores": scores,
                    "protected_edit_overlap": common_facts["protected_overlap"],
                    "ambiguous_edit_overlap": common_facts["ambiguous_overlap"],
                    "preserve_edit_overlap": common_facts["preserve_overlap"],
                    "ownership_leak_pixel_count": int(
                        np.count_nonzero((edit > 0) & (masks.ownership == 0))
                    ),
                    "false_edit_pixel_count": int(np.count_nonzero(edit))
                    if stage_page.no_edit
                    else 0,
                    "edit_pixel_count": int(np.count_nonzero(edit)),
                    "target_extent_independent": stage_page.target_extent_independent,
                    "target_inventory_independent": stage_page.target_inventory_independent,
                    "target_review_complete": stage_page.target_review_complete,
                    "target_mask_provenance": stage_page.target_mask_provenance,
                    "output_claim_mask_pixel_sha256": decoded_sha[
                        (run_id, page_id, "claim_mask")
                    ],
                    "output_edit_mask_pixel_sha256": decoded_sha[
                        (run_id, page_id, "edit_mask")
                    ],
                }
                for field, expected_value in fusion_facts.items():
                    _assert_recomputed_fact(page, field, expected_value)


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
    try:
        canonical_summary = summarize_stage1_pages(
            [dict(row) for row in pages]
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "source protection pages lack the canonical aggregation inputs"
        ) from error
    if dict(summary) != canonical_summary:
        raise ValueError(
            "source protection summary differs from canonical page aggregation"
        )
    _validate_stage_summary(summary, len(pages))


def validate_evidence_artifact(payload: Mapping[str, object]) -> None:
    schema = str(payload.get("schema_version") or "")
    validators = {
        "inpaint-factorized-results-v3": _validate_factorized,
        "inpaint-detector-fusion-results-v4": _validate_fusion,
        "inpaint-semantic-policy-results-v4": _validate_semantic,
        "inpaint-source-protection-reapply-v3": _validate_source_protection,
        "inpaint-detector-bakeoff-stage1-v1": _validate_stage1,
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
                identities = sorted(
                    ({
                        "output_mask_set_sha256": str(
                            row["metrics"]["output_mask_set_sha256"]  # type: ignore[index]
                        ),
                        "candidate_image_set_sha256": str(
                            row["metrics"]["candidate_image_set_sha256"]  # type: ignore[index]
                        ),
                    }
                    for row in rows
                    ),
                    key=lambda value: (
                    value["output_mask_set_sha256"],
                    value["candidate_image_set_sha256"],
                    ),
                )
                kind = "exact_output"
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
    if (
        schema == "inpaint-detector-bakeoff-stage1-v1"
        and family_id
        in {
            "ctd-synthetic-finetune",
            "easyocr-craft",
            "easyocr-dbnet18",
        }
    ):
        identities = payload["variant_output_identity"]  # type: ignore[index]
        summary = payload["summary"]  # type: ignore[index]
        pages = payload["pages"]  # type: ignore[index]
        disposition = _validate_stage_summary(summary, len(pages))
        reason = "" if disposition == "family_complete" else "hard_gate_failed"
        for source_variant, variant in (
            ("raw", "raw"),
            ("refined", "refined"),
            ("dilated", "native3"),
        ):
            identity = identities[source_variant]
            output[variant] = ArtifactVariantFact(
                disposition,
                reason,
                str(identity["output_mask_set_sha256"]),
                "exact_output",
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
    validated = validate_source_only_manifest_v4(path)
    return {
        "sha256": str(validated["manifest_sha256"]),
        "seal_sha256": str(validated["seal_sha256"]),
        "schema_version": str(validated["schema_version"]),
        "corpus_id": str(validated["corpus_id"]),
        "split_role": str(validated["split_role"]),
        "page_count": int(validated["page_count"]),
        "page_ids": list(validated["page_ids"]),
        "page_inventory_sha256": str(validated["page_inventory_sha256"]),
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
    upstream_contract_path: Path | None = None,
    expected_fusion_candidate_ids: frozenset[str] | None = None,
) -> tuple[dict[str, object], ...]:
    return accounted_evidence_from_artifact(
        requirements,
        artifact_path=artifact_path,
        scope_manifest_path=scope_manifest_path,
        family_id=family_id,
        variant_ids=variant_ids,
        evaluation_scope=evaluation_scope,
        upstream_contract_path=upstream_contract_path,
        expected_fusion_candidate_ids=expected_fusion_candidate_ids,
    )


def accounted_evidence_from_artifact(
    requirements: Iterable[MethodVariantRequirement],
    *,
    artifact_path: Path,
    scope_manifest_path: Path,
    family_id: str,
    variant_ids: frozenset[str],
    evaluation_scope: str,
    upstream_contract_path: Path | None = None,
    expected_fusion_candidate_ids: frozenset[str] | None = None,
) -> tuple[dict[str, object], ...]:
    if not family_id.strip() or not variant_ids:
        raise ValueError("an explicit family and at least one variant are required")
    requirements = tuple(requirements)
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("evidence artifact root must be an object")
    scope_binding = scope_manifest_binding(scope_manifest_path)
    scope_manifest_sha256 = str(scope_binding["sha256"])
    if str(payload.get("manifest_sha256") or "") != scope_manifest_sha256:
        raise ValueError("evidence artifact manifest SHA differs from the sealed scope manifest")
    facts = artifact_variant_facts(payload, family_id)
    if payload.get("schema_version") == "inpaint-detector-bakeoff-stage1-v1":
        _validate_stage1_output_artifacts(payload, artifact_path)
    _validate_finalist_output_artifacts(
        payload,
        artifact_path,
        scope_manifest_path,
    )
    artifact_sha256 = sha256_file(artifact_path)
    _validate_upstream_logical_inventory(
        payload,
        upstream_contract_path=upstream_contract_path,
        scope_manifest_sha256=scope_manifest_sha256,
        expected_fusion_candidate_ids=expected_fusion_candidate_ids,
    )
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
    if (
        str(payload.get("schema_version") or "")
        == "inpaint-factorized-results-v3"
        and any(facts[variant_id].disposition == "pareto" for variant_id in variant_ids)
    ):
        requested_role = registered[next(iter(sorted(variant_ids)))].role
        role_requirements = tuple(
            requirement
            for requirement in requirements
            if requirement.role == requested_role
            and requirement.evaluation_scope == evaluation_scope
            and requirement.family_id in _FACTORIZED_FAMILY_IDS
        )
        globally_unproved = sorted(
            (
                requirement.family_id,
                requirement.variant_id,
            )
            for requirement in role_requirements
            if requirement.variant_id
            not in artifact_declared_variants(payload, requirement.family_id)
        )
        if globally_unproved:
            family, variant = globally_unproved[0]
            raise ValueError(
                "factorized Pareto artifact does not cover the full registered "
                f"role inventory: {family}/{variant}"
            )
    rows: list[dict[str, object]] = []
    for variant_id in sorted(variant_ids):
        requirement = registered[variant_id]
        fact = facts[variant_id]
        if fact.disposition == "blocked_asset":
            raise ValueError("blocked artifact status requires a separately hashed asset probe")
        exact_identity_reuse = (
            str(payload.get("schema_version") or "")
            == "inpaint-detector-bakeoff-stage1-v1"
            and family_id == "ctd-synthetic-finetune"
            and variant_id == "refined"
        )
        rows.append(
            {
                "family_id": requirement.family_id,
                "role": requirement.role,
                "variant_id": requirement.variant_id,
                "evaluation_scope": requirement.evaluation_scope,
                "closure_state": (
                    "reused_by_sha" if exact_identity_reuse else "executed"
                ),
                "disposition": fact.disposition,
                "reason": fact.reason,
                "artifact_sha256": artifact_sha256,
                "artifact_schema_version": str(payload.get("schema_version") or ""),
                "artifact_name": artifact_path.name,
                "scope_manifest_sha256": scope_manifest_sha256,
                "content_sha256": fact.content_sha256,
                "content_identity_kind": fact.content_identity_kind,
                "reused_from": (
                    evidence_key(
                        family_id,
                        requirement.role,
                        "raw",
                        evaluation_scope,
                    )
                    if exact_identity_reuse
                    else ""
                ),
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
    for field in ("target", "provider", "asset_id", "reason_code"):
        if not str(probe.get(field) or "").strip():
            raise ValueError(f"blocked asset probe lacks {field}")
    checks = probe.get("checks")
    if not isinstance(checks, list) or not checks or any(not isinstance(row, Mapping) for row in checks):
        raise ValueError("blocked asset probe requires concrete check records")
    supported_check_kinds = frozenset(
        {
            "filesystem",
            "managed_asset_registry",
            "provider_availability",
            "runtime_import",
        }
    )
    for row in checks:
        kind = str(row.get("kind") or "")
        target = str(row.get("target") or "").strip()
        evidence = str(row.get("evidence") or "").strip()
        found = row.get("found")
        status = str(row.get("status") or "").strip()
        if kind not in supported_check_kinds:
            raise ValueError("blocked asset probe contains an unsupported check kind")
        if not target or not evidence:
            raise ValueError(
                "blocked asset probe check requires target and failure evidence"
            )
        explicit_failure = status in {"unavailable", "error"}
        if found is True or status == "success" or not (
            found is False or explicit_failure
        ):
            raise ValueError(
                "blocked asset probe check does not prove unavailable/error state"
            )
        if found not in {None, False}:
            raise ValueError("blocked asset probe check has an invalid found value")
        if status and status not in {"unavailable", "error"}:
            raise ValueError("blocked asset probe check has an invalid failure status")
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
        "easyocr-craft": frozenset({"raw", "refined", "native3"}),
        "easyocr-dbnet18": frozenset({"raw", "refined", "native3"}),
        "detector-fusion": frozenset({"single", "or", "and", "gated_recovery"}),
        "roi-trigger": frozenset({"none", "always", "seed_missing", "raw_refined_disagreement", "source_seed_unavailable", "union"}),
        "semantic-policy": frozenset(
            {
                "current_default",
                "detector_explicit_role",
                "ocr_semantic_hint",
                "ocr_provenance_verifier",
                "explicit_role_consensus",
                "human_oracle",
            }
        ),
        "ownership": frozenset({"block_region", "dual_ownership", "ctbd_content", "ysg_standard", "ysg_obb", "manga109"}),
        "bubble-silhouette": frozenset({"pr2_validated", "ballons_native", "ctbd_bubble", "manga109_balloon", "pair_union_ballons_pr2", "pair_intersection_ballons_pr2", "consensus_2_of_4", "consensus_3_of_4"}),
        "router": frozenset({"R0", "R1", "R2", "R3", "R4"}),
        "mask-expansion": frozenset({"raw", "refined", "native3", "content_component", "validated_interior", "lab_dilate1", "lab_dilate2", "lab_dilate3", "lab_dilate4"}),
        "exact-protection": frozenset({"C14", "C15", "C17", "C18", "C19", "C21", "C22", "C23"}),
        "exact-protection-historical": frozenset({"C14", "C15", "C17", "C18", "C19", "C21", "C22", "C23"}),
        "fill-backend": frozenset({"current_lama", "ballons_lama", "robust_flat_median", "planar_gradient", "telea", "conditional_hybrid", "skip"}),
    }
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
