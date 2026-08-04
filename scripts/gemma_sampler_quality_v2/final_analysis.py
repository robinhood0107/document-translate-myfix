"""Fail-closed final integrity and quality analysis for the single campaign."""

from __future__ import annotations

import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Iterable, Mapping, Sequence

from .incremental import (
    bind_incremental_ledger_to_records,
    validate_incremental_ledger,
)
from .judgment import JudgmentError, rank_sampler_results
from .protocol import (
    SEEDS,
    CampaignPlan,
    SamplerArm,
    SamplerTuple,
    campaign_plan,
    canonical_sha256,
)


FINAL_ANALYSIS_SCHEMA_VERSION = "gemma-sampler-final-analysis-v1"
FINAL_GATE_SCHEMA_VERSION = "gemma-sampler-required-gates-v1"


def _sampler(record: Mapping[str, Any]) -> SamplerTuple:
    value = record.get("sampler")
    if not isinstance(value, Mapping):
        raise JudgmentError("Final campaign record has no sampler tuple.")
    return SamplerTuple(
        float(value.get("temperature")),
        float(value.get("top_p")),
        int(value.get("top_k")),
        float(value.get("min_p")),
    )


def _case_ids(reference: Mapping[str, Any]) -> tuple[str, ...]:
    cases = reference.get("cases")
    if not isinstance(cases, list):
        raise JudgmentError("Final campaign analysis requires frozen reference cases.")
    case_ids = tuple(
        str(case.get("case_id") or "")
        for case in cases
        if isinstance(case, Mapping)
    )
    if not case_ids or len(case_ids) != len(cases) or len(set(case_ids)) != len(case_ids):
        raise JudgmentError("Final campaign reference has invalid case identities.")
    return case_ids


def _arm_map(arms: Iterable[SamplerArm]) -> dict[str, SamplerArm]:
    result: dict[str, SamplerArm] = {}
    for arm in arms:
        key = arm.sampler.key
        if key in result:
            raise JudgmentError("Final campaign origin contains duplicate sampler tuples.")
        result[key] = arm
    return result


def _manifest_completed(manifest: Mapping[str, Any], *, label: str) -> None:
    if str(manifest.get("status") or "") != "completed":
        raise JudgmentError(f"{label} managed artifact run is not completed.")


def _validate_origin_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_arms: Mapping[str, SamplerArm],
    case_ids: Sequence[str],
    seeds: Sequence[int],
    label: str,
) -> tuple[set[str], dict[str, set[str]]]:
    expected_cases = set(case_ids)
    expected_seeds = set(seeds)
    observed_cells: set[tuple[str, int, str]] = set()
    runtime_fingerprints: set[str] = set()
    request_hashes: dict[str, set[str]] = defaultdict(set)
    for record in records:
        sampler = _sampler(record)
        arm = expected_arms.get(sampler.key)
        if arm is None:
            raise JudgmentError(f"{label} contains a sampler tuple outside its sealed origin.")
        if dict(record.get("sampler") or {}) != arm.sampler.payload():
            raise JudgmentError(f"{label} sampler payload changed after execution.")
        if str(record.get("phase") or "") != arm.phase:
            raise JudgmentError(f"{label} response phase does not match the sealed arm.")
        if str(record.get("arm_key") or "") != arm.key:
            raise JudgmentError(f"{label} response arm identity changed.")
        seed = record.get("seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed not in expected_seeds:
            raise JudgmentError(f"{label} response has an unexpected seed.")
        case_id = str(record.get("case_id") or "")
        if case_id not in expected_cases:
            raise JudgmentError(f"{label} response has an unexpected reference case.")
        cell = (sampler.key, seed, case_id)
        if cell in observed_cells:
            raise JudgmentError(f"{label} contains duplicate sampler/seed/case evidence.")
        observed_cells.add(cell)
        runtime_fingerprint = str(record.get("runtime_fingerprint") or "")
        if not runtime_fingerprint:
            raise JudgmentError(f"{label} response has no runtime fingerprint.")
        runtime_fingerprints.add(runtime_fingerprint)
        request_identity = record.get("request_identity")
        if not isinstance(request_identity, Mapping):
            raise JudgmentError(f"{label} response has no fixed request identity.")
        request_hashes[case_id].add(canonical_sha256(request_identity))
    expected_cells = {
        (sampler_key, seed, case_id)
        for sampler_key in expected_arms
        for seed in seeds
        for case_id in case_ids
    }
    if observed_cells != expected_cells:
        raise JudgmentError(f"{label} does not contain every sealed sampler/seed/case exactly once.")
    return runtime_fingerprints, request_hashes


def validate_final_campaign_evidence(
    *,
    reference: Mapping[str, Any],
    r6_records: Sequence[Mapping[str, Any]],
    campaign_records: Sequence[Mapping[str, Any]],
    r6_manifest: Mapping[str, Any],
    campaign_manifest: Mapping[str, Any],
    campaign_status: Mapping[str, Any],
    campaign_plan_artifact: Mapping[str, Any],
    plan: CampaignPlan | None = None,
    seeds: Sequence[int] = SEEDS,
) -> dict[str, Any]:
    """Prove the exact final matrix and immutable runtime/request provenance."""

    selected_plan = plan or campaign_plan()
    case_ids = _case_ids(reference)
    reference_sha256 = str(reference.get("reference_sha256") or "")
    _manifest_completed(r6_manifest, label="r6")
    _manifest_completed(campaign_manifest, label="Campaign")
    if str(campaign_status.get("state") or "") != "WAITING_FOR_FINAL_JUDGMENT":
        raise JudgmentError("Campaign has not reached final judgment after successful cleanup.")
    if str(campaign_status.get("reference_sha256") or "") != reference_sha256:
        raise JudgmentError("Campaign status belongs to a different frozen reference.")
    if str(campaign_status.get("campaign_plan_sha256") or "") != selected_plan.sha256:
        raise JudgmentError("Campaign status plan hash differs from the sealed matrix.")
    if str(campaign_plan_artifact.get("campaign_plan_sha256") or "") != selected_plan.sha256:
        raise JudgmentError("Campaign plan artifact differs from the sealed matrix.")
    if campaign_plan_artifact.get("plan") != selected_plan.payload():
        raise JudgmentError("Campaign plan payload changed after execution.")
    if str(campaign_plan_artifact.get("reference_sha256") or "") != reference_sha256:
        raise JudgmentError("Campaign plan artifact belongs to a different frozen reference.")

    r6_arms = _arm_map(selected_plan.temperature_arms)
    campaign_arms = _arm_map(selected_plan.new_joint_arms + selected_plan.new_min_p_arms)
    if set(r6_arms) & set(campaign_arms):
        raise JudgmentError("Campaign new-response origin overlaps read-only r6 evidence.")
    per_arm = len(case_ids) * len(seeds)
    expected_r6 = len(r6_arms) * per_arm
    expected_campaign = len(campaign_arms) * per_arm
    expected_total = len(selected_plan.sampler_keys) * per_arm
    if len(r6_records) != expected_r6 or len(campaign_records) != expected_campaign:
        raise JudgmentError("Final response counts do not match the sealed campaign matrix.")
    logical_slots = [
        str(record.get("logical_slot") or "")
        for record in tuple(r6_records) + tuple(campaign_records)
    ]
    if any(not slot for slot in logical_slots) or len(set(logical_slots)) != len(logical_slots):
        raise JudgmentError("Final evidence contains duplicate or empty logical slot identities.")
    if int(campaign_status.get("reused_logical_slots") or 0) != expected_r6:
        raise JudgmentError("Campaign status r6 response count is invalid.")
    if int(campaign_status.get("completed_new_logical_slots") or 0) != expected_campaign:
        raise JudgmentError("Campaign status new response count is invalid.")
    if int(campaign_status.get("expected_evidence_logical_slots") or 0) != expected_total:
        raise JudgmentError("Campaign status total evidence count is invalid.")
    if tuple(campaign_status.get("sampler_keys") or ()) != selected_plan.sampler_keys:
        raise JudgmentError("Campaign status sampler tuple order changed.")
    if tuple(campaign_status.get("seeds") or ()) != tuple(seeds):
        raise JudgmentError("Campaign status seed contract changed.")

    r6_fingerprints, r6_requests = _validate_origin_records(
        r6_records,
        expected_arms=r6_arms,
        case_ids=case_ids,
        seeds=seeds,
        label="r6",
    )
    campaign_fingerprints, campaign_requests = _validate_origin_records(
        campaign_records,
        expected_arms=campaign_arms,
        case_ids=case_ids,
        seeds=seeds,
        label="Campaign",
    )
    all_fingerprints = r6_fingerprints | campaign_fingerprints
    if len(all_fingerprints) != 1:
        raise JudgmentError("Final evidence contains more than one Router runtime fingerprint.")
    fingerprint = next(iter(all_fingerprints))
    if str(campaign_status.get("r6_runtime_fingerprint") or "") != fingerprint:
        raise JudgmentError("Campaign status r6 runtime fingerprint changed.")
    if str(campaign_status.get("runtime_fingerprint") or "") != fingerprint:
        raise JudgmentError("Campaign status runtime fingerprint changed.")
    for case_id in case_ids:
        hashes = r6_requests.get(case_id, set()) | campaign_requests.get(case_id, set())
        if len(hashes) != 1:
            raise JudgmentError("Sampler arms did not preserve one fixed request identity per case.")
    return {
        "reference_sha256": reference_sha256,
        "campaign_plan_sha256": selected_plan.sha256,
        "runtime_fingerprint": fingerprint,
        "case_count": len(case_ids),
        "seed_count": len(seeds),
        "sampler_count": len(selected_plan.sampler_keys),
        "r6_response_count": len(r6_records),
        "new_response_count": len(campaign_records),
        "total_response_count": len(r6_records) + len(campaign_records),
    }


def _ordered_sampler_keys(arms: Iterable[SamplerArm]) -> list[str]:
    result: list[str] = []
    for arm in arms:
        if arm.sampler.key not in result:
            result.append(arm.sampler.key)
    return result


def _public_metric_row(row: Mapping[str, Any], *, case_count: int) -> dict[str, Any]:
    response_count = int(row.get("response_count") or 0)
    catastrophic = int(row.get("catastrophic") or 0)
    major = int(row.get("major") or 0)
    minor = int(row.get("minor") or 0)
    error_count = catastrophic + major + minor
    denominator = response_count if response_count > 0 else 1
    case_denominator = case_count if case_count > 0 else 1
    return {
        "sampler": dict(row.get("sampler") or {}),
        "sampler_key": str(row.get("sampler_key") or ""),
        "catastrophic": catastrophic,
        "major": major,
        "minor": minor,
        "unjudged": int(row.get("unjudged") or 0),
        "error_response_count": error_count,
        "clean_response_count": response_count - error_count,
        "unique_error_cases": int(row.get("unique_error_cases") or 0),
        "response_count": response_count,
        "catastrophic_rate": round(catastrophic / denominator, 8),
        "major_rate": round(major / denominator, 8),
        "minor_rate": round(minor / denominator, 8),
        "error_response_rate": round(error_count / denominator, 8),
        "unique_error_case_rate": round(
            int(row.get("unique_error_cases") or 0) / case_denominator,
            8,
        ),
        "naturalness_mean": row.get("naturalness_mean"),
        "latency_ms_mean": row.get("latency_ms_mean"),
        "completion_tokens_mean": row.get("completion_tokens_mean"),
    }


def _rank_rows_by_seed(
    records: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    *,
    sampler_keys: Sequence[str],
    seeds: Sequence[int],
    case_count: int,
) -> list[dict[str, Any]]:
    by_seed: dict[int, dict[str, Mapping[str, Any]]] = {}
    for seed in seeds:
        seed_records = [record for record in records if record.get("seed") == seed]
        rows = rank_sampler_results(seed_records, verdicts, scope="all")
        by_seed[seed] = {str(row.get("sampler_key") or ""): row for row in rows}
    result: list[dict[str, Any]] = []
    for sampler_key in sampler_keys:
        seed_rows: list[dict[str, Any]] = []
        for seed in seeds:
            row = by_seed[seed].get(sampler_key)
            if row is None:
                raise JudgmentError("Final seed analysis is missing a sampler tuple.")
            seed_rows.append(
                {
                    "seed": seed,
                    **_public_metric_row(row, case_count=case_count),
                }
            )
        result.append(
            {
                "sampler_key": sampler_key,
                "seeds": seed_rows,
                "catastrophic_spread": max(row["catastrophic"] for row in seed_rows)
                - min(row["catastrophic"] for row in seed_rows),
                "major_spread": max(row["major"] for row in seed_rows)
                - min(row["major"] for row in seed_rows),
                "minor_spread": max(row["minor"] for row in seed_rows)
                - min(row["minor"] for row in seed_rows),
            }
        )
    return result


def _normalize_gate_text(value: str) -> str:
    return "".join(unicodedata.normalize("NFKC", value).split()).casefold()


def _required_gate_results(
    *,
    reference: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
    sampler_keys: Sequence[str],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    if gates.get("schema_version") != FINAL_GATE_SCHEMA_VERSION:
        raise JudgmentError("Required gate manifest schema is not current.")
    if str(gates.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise JudgmentError("Required gate manifest belongs to a different frozen reference.")
    gate_rows = gates.get("gates")
    if not isinstance(gate_rows, list) or not gate_rows:
        raise JudgmentError("Required gate manifest has no gates.")
    cases = set(_case_ids(reference))
    by_cell: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for record in records:
        by_cell[(_sampler(record).key, int(record.get("seed")), str(record.get("case_id") or ""))] = record
    public: list[dict[str, Any]] = []
    private: list[dict[str, Any]] = []
    eligible = {key: True for key in sampler_keys}
    seen_gate_ids: set[str] = set()
    for gate in gate_rows:
        if not isinstance(gate, Mapping):
            raise JudgmentError("Required gate manifest contains an invalid gate.")
        gate_id = str(gate.get("gate_id") or "")
        case_id = str(gate.get("case_id") or "")
        required = gate.get("required_substrings")
        if (
            not gate_id
            or gate_id in seen_gate_ids
            or case_id not in cases
            or not isinstance(required, list)
            or not required
            or any(not isinstance(value, str) or not value for value in required)
        ):
            raise JudgmentError("Required gate manifest identity or token contract is invalid.")
        seen_gate_ids.add(gate_id)
        normalized_required = [_normalize_gate_text(value) for value in required]
        failures: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        excluded_keys: set[str] = set()
        for sampler_key in sampler_keys:
            for seed in seeds:
                record = by_cell.get((sampler_key, seed, case_id))
                if record is None:
                    raise JudgmentError("Required gate lacks one sampler/seed response.")
                translation = str(record.get("translation") or "")
                normalized = _normalize_gate_text(translation)
                verdict = verdicts.get(str(record.get("logical_slot") or ""))
                if not isinstance(verdict, Mapping):
                    raise JudgmentError("Required gate response has no final semantic verdict.")
                decision = str(verdict.get("decision") or "")
                missing = [
                    index
                    for index, required_value in enumerate(normalized_required)
                    if required_value not in normalized
                ]
                passed = not missing and decision in {"PASS", "MINOR"}
                evidence.append(
                    {
                        "sampler_key": sampler_key,
                        "seed": seed,
                        "passed": passed,
                        "missing_requirement_indexes": missing,
                        "decision": decision,
                        "category": str(verdict.get("category") or ""),
                        "translation": translation,
                        "raw_response": record.get("response") if not translation else None,
                    }
                )
                if not passed:
                    excluded_keys.add(sampler_key)
                    eligible[sampler_key] = False
                    failures.append(
                        {
                            "sampler_key": sampler_key,
                            "seed": seed,
                            "missing_requirement_indexes": missing,
                            "decision": decision,
                            "category": str(verdict.get("category") or ""),
                        }
                    )
        public.append(
            {
                "gate_id": gate_id,
                "eligible_sampler_count": len(sampler_keys) - len(excluded_keys),
                "excluded_sampler_count": len(excluded_keys),
                "excluded_sampler_keys": sorted(excluded_keys),
            }
        )
        private.append(
            {
                "gate_id": gate_id,
                "case_id": case_id,
                "failures": failures,
                "evidence": evidence,
            }
        )
    return public, private, eligible


def _private_error_clusters(
    *,
    reference: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    cases = {
        str(case.get("case_id") or ""): case
        for case in reference.get("cases", [])
        if isinstance(case, Mapping)
    }
    grouped: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for record in records:
        slot = str(record.get("logical_slot") or "")
        verdict = verdicts[slot]
        decision = str(verdict.get("decision") or "")
        if decision == "PASS":
            continue
        validation = record.get("response_validation")
        output_sha = (
            str(validation.get("translation_sha256") or "")
            if isinstance(validation, Mapping)
            else ""
        )
        raw_response = record.get("response")
        if not record.get("translation") and not isinstance(raw_response, Mapping):
            raise JudgmentError("Final error report lacks raw response evidence for a failed translation.")
        output_identity = output_sha or "raw-" + canonical_sha256(raw_response)
        case_id = str(record.get("case_id") or "")
        category = str(verdict.get("category") or "")
        key = (case_id, output_identity, decision, category)
        item = grouped.setdefault(
            key,
            {
                "case_id": case_id,
                "decision": decision,
                "category": category,
                "source_text": cases.get(case_id, {}).get("source_text"),
                "context_after_text": cases.get(case_id, {}).get("context_after_text"),
                "canonical_translation": cases.get(case_id, {}).get("canonical_translation"),
                "candidate_translation": str(record.get("translation") or ""),
                "raw_response": raw_response if not record.get("translation") else None,
                "occurrences": [],
            },
        )
        item["occurrences"].append(
            {
                "sampler_key": _sampler(record).key,
                "seed": record.get("seed"),
            }
        )
    return sorted(
        grouped.values(),
        key=lambda item: (
            {"CATASTROPHIC": 0, "MAJOR": 1, "MINOR": 2, "UNJUDGED": 3}.get(
                str(item.get("decision") or ""),
                4,
            ),
            str(item.get("case_id") or ""),
            str(item.get("candidate_translation") or ""),
        ),
    )


def build_final_campaign_analysis(
    *,
    reference: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    gates: Mapping[str, Any],
    evidence_summary: Mapping[str, Any],
    plan: CampaignPlan | None = None,
    seeds: Sequence[int] = SEEDS,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind every verdict, rank all arms, and produce public/private reports."""

    selected_plan = plan or campaign_plan()
    validate_incremental_ledger(ledger, reference=reference)
    verdicts = bind_incremental_ledger_to_records(reference, records, ledger)
    decisions = Counter(str(verdict.get("decision") or "") for verdict in verdicts.values())
    if decisions.get("UNJUDGED", 0) or decisions.get("REVIEW_REQUIRED", 0):
        raise JudgmentError("Final campaign still contains unresolved judgments.")
    ranked = rank_sampler_results(records, verdicts, scope="all")
    if len(ranked) != len(selected_plan.sampler_keys):
        raise JudgmentError("Final ranking does not contain every sampler tuple.")
    expected_per_sampler = len(_case_ids(reference)) * len(seeds)
    if any(int(row.get("response_count") or 0) != expected_per_sampler for row in ranked):
        raise JudgmentError("Final ranking does not contain both seeds for every reference case.")
    rank_by_key = {str(row.get("sampler_key") or ""): row for row in ranked}
    if set(rank_by_key) != set(selected_plan.sampler_keys):
        raise JudgmentError("Final ranking sampler identities differ from the sealed matrix.")
    public_gates, private_gates, eligible = _required_gate_results(
        reference=reference,
        records=records,
        verdicts=verdicts,
        gates=gates,
        sampler_keys=selected_plan.sampler_keys,
        seeds=seeds,
    )
    provisional = next(
        (str(row.get("sampler_key") or "") for row in ranked if eligible[str(row.get("sampler_key") or "")]),
        None,
    )
    baseline_key = SamplerTuple(0.7, 0.95, 64, 0.0).key
    case_count = len(_case_ids(reference))
    public_ranked = [
        _public_metric_row(row, case_count=case_count)
        for row in ranked
    ]
    public_by_key = {
        str(row.get("sampler_key") or ""): row
        for row in public_ranked
    }
    public = {
        "schema_version": FINAL_ANALYSIS_SCHEMA_VERSION,
        "state": "WAITING_FOR_USER_APPROVAL",
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "campaign_plan_sha256": selected_plan.sha256,
        "evidence": dict(evidence_summary),
        "verdict_totals": dict(sorted(decisions.items())),
        "ranked_arms": public_ranked,
        "temperature_rows": [
            deepcopy(public_by_key[key])
            for key in _ordered_sampler_keys(selected_plan.temperature_arms)
        ],
        "joint_top_p_top_k_rows": [
            deepcopy(public_by_key[key])
            for key in _ordered_sampler_keys(selected_plan.joint_arms)
        ],
        "min_p_rows": [
            deepcopy(public_by_key[key])
            for key in _ordered_sampler_keys(selected_plan.min_p_arms)
        ],
        "seed_rows": _rank_rows_by_seed(
            records,
            verdicts,
            sampler_keys=selected_plan.sampler_keys,
            seeds=seeds,
            case_count=case_count,
        ),
        "required_gates": public_gates,
        "provisional_candidate_sampler_key": provisional,
        "baseline_sampler_key": baseline_key,
        "baseline_rank": next(
            index + 1
            for index, row in enumerate(ranked)
            if str(row.get("sampler_key") or "") == baseline_key
        ),
        "product_promotion_allowed": False,
    }
    public["analysis_sha256"] = canonical_sha256(public)
    private = {
        "schema_version": FINAL_ANALYSIS_SCHEMA_VERSION,
        "public_analysis": public,
        "required_gate_evidence": private_gates,
        "error_clusters": _private_error_clusters(
            reference=reference,
            records=records,
            verdicts=verdicts,
        ),
    }
    private["private_analysis_sha256"] = canonical_sha256(private)
    return public, private
