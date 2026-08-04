"""Stable, private-only incremental judgment for the single sampler campaign.

The inference campaign is intentionally immutable while it is running.  This
module builds blind batches from whatever first-valid responses currently
exist and stores verdicts by ``case id + translated text hash``.  The stable
identity lets a verdict be reused when later sampler arms emit the same text,
without exposing sampler, seed, arm, or execution order to the evaluator.
"""

from __future__ import annotations

import hashlib
from copy import deepcopy
from typing import Any, Iterable, Mapping

from .judgment import (
    JUDGMENT_SCHEMA_VERSION,
    RESPONSE_VALIDATION_SCHEMA_VERSION,
    JudgmentError,
    apply_blind_judgments,
    build_blind_judgment_packet,
)
from .protocol import canonical_sha256


INCREMENTAL_LEDGER_SCHEMA_VERSION = "gemma-sampler-incremental-ledger-v1"
INCREMENTAL_PACKET_SCHEMA_VERSION = "gemma-sampler-incremental-packet-v1"
INCREMENTAL_AMENDMENT_SCHEMA_VERSION = "gemma-sampler-incremental-amendment-v1"
SEMANTIC_JUDGMENT_RULE_VERSION = "faithful-translation-quality-v1"
SEMANTIC_JUDGMENT_RULE = {
    "failure_dimensions": [
        "meaning",
        "person",
        "action",
        "relationship",
        "number",
        "negation",
        "question_or_statement",
        "identity",
        "consent_or_coercion",
        "censorship_or_weakening",
        "visible_mixed_token_damage",
        "naturalness",
    ],
    "allowed_differences": [
        "tone",
        "word_order",
        "honorific_style",
        "onomatopoeia",
        "meaning_preserving_synonym",
    ],
    "transport_is_diagnostic_when_translation_is_extractable": True,
    "exact_canonical_match_is_not_required": True,
}
_FINAL_DECISIONS = {"PASS", "MINOR", "MAJOR", "CATASTROPHIC"}
_FORBIDDEN_BLIND_KEYS = {
    "arm",
    "arm_key",
    "case_position",
    "current_sampler",
    "execution_order",
    "logical_slot",
    "recorded_utc",
    "run",
    "sampler",
    "sampler_key",
    "seed",
}


def _without_sha(payload: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(payload)
    result.pop(field, None)
    return result


def _seal_ledger(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["ledger_sha256"] = canonical_sha256(_without_sha(result, "ledger_sha256"))
    return result


def _seal_packet(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(payload))
    result["packet_sha256"] = canonical_sha256(_without_sha(result, "packet_sha256"))
    return result


def _forbidden_blind_key(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            normalized = str(key).strip().lower()
            if normalized in _FORBIDDEN_BLIND_KEYS:
                return normalized
            found = _forbidden_blind_key(nested)
            if found is not None:
                return found
    elif isinstance(value, (list, tuple)):
        for nested in value:
            found = _forbidden_blind_key(nested)
            if found is not None:
                return found
    return None


def new_incremental_ledger(reference: Mapping[str, Any]) -> dict[str, Any]:
    reference_sha256 = str(reference.get("reference_sha256") or "")
    if not reference_sha256:
        raise JudgmentError("Incremental judgment requires a frozen reference hash.")
    return _seal_ledger(
        {
            "schema_version": INCREMENTAL_LEDGER_SCHEMA_VERSION,
            "rule_version": SEMANTIC_JUDGMENT_RULE_VERSION,
            "rule": SEMANTIC_JUDGMENT_RULE,
            "reference_sha256": reference_sha256,
            "verdicts": {},
            "imports": [],
            "applied_batches": [],
            "amendments": [],
            "next_batch_number": 1,
            "pending_batch": None,
        }
    )


def validate_incremental_ledger(
    ledger: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
) -> None:
    if ledger.get("schema_version") != INCREMENTAL_LEDGER_SCHEMA_VERSION:
        raise JudgmentError("Incremental judgment ledger schema is not current.")
    if ledger.get("rule_version") != SEMANTIC_JUDGMENT_RULE_VERSION:
        raise JudgmentError("Incremental judgment ledger uses a different semantic rule version.")
    if ledger.get("rule") != SEMANTIC_JUDGMENT_RULE:
        raise JudgmentError("Incremental judgment ledger semantic rules changed.")
    if str(ledger.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise JudgmentError("Incremental judgment ledger belongs to a different frozen reference.")
    if str(ledger.get("ledger_sha256") or "") != canonical_sha256(
        _without_sha(ledger, "ledger_sha256")
    ):
        raise JudgmentError("Incremental judgment ledger integrity hash is invalid.")
    verdicts = ledger.get("verdicts")
    if not isinstance(verdicts, Mapping):
        raise JudgmentError("Incremental judgment ledger verdict collection is invalid.")
    for cluster_id, verdict in verdicts.items():
        if not str(cluster_id).startswith("cluster-") or not isinstance(verdict, Mapping):
            raise JudgmentError("Incremental judgment ledger contains an invalid cluster verdict.")
        if str(verdict.get("cluster_id") or "") != str(cluster_id):
            raise JudgmentError("Incremental judgment ledger cluster identity changed.")
        if str(verdict.get("decision") or "") not in _FINAL_DECISIONS:
            raise JudgmentError("Incremental judgment ledger contains a non-final verdict.")
    amendments = ledger.get("amendments", [])
    if not isinstance(amendments, list):
        raise JudgmentError("Incremental judgment ledger amendment history is invalid.")
    seen_amendments: set[str] = set()
    last_amended_verdicts: dict[str, Mapping[str, Any]] = {}
    for amendment in amendments:
        if not isinstance(amendment, Mapping):
            raise JudgmentError("Incremental judgment ledger contains an invalid amendment.")
        amendment_id = str(amendment.get("amendment_id") or "")
        changes = amendment.get("changes")
        reason = str(amendment.get("reason") or "").strip()
        if (
            not amendment_id.startswith("amendment-")
            or amendment_id in seen_amendments
            or not isinstance(changes, list)
            or not changes
            or not reason
            or not str(amendment.get("applied_utc") or "").strip()
        ):
            raise JudgmentError("Incremental judgment ledger amendment identity is invalid.")
        seen_amendments.add(amendment_id)
        if int(amendment.get("cluster_count") or 0) != len(changes):
            raise JudgmentError("Incremental judgment ledger amendment count changed.")
        if str(amendment.get("changes_sha256") or "") != canonical_sha256(changes):
            raise JudgmentError("Incremental judgment ledger amendment changes hash is invalid.")
        expected_id = "amendment-" + canonical_sha256(
            {
                "reference_sha256": str(ledger.get("reference_sha256") or ""),
                "rule_version": str(ledger.get("rule_version") or ""),
                "reason": reason,
                "changes": changes,
            }
        )[:24]
        if amendment_id != expected_id:
            raise JudgmentError("Incremental judgment ledger amendment id changed.")
        changed_clusters: set[str] = set()
        for change in changes:
            if not isinstance(change, Mapping):
                raise JudgmentError("Incremental judgment ledger amendment change is invalid.")
            before = change.get("before")
            after = change.get("after")
            if not isinstance(before, Mapping) or not isinstance(after, Mapping):
                raise JudgmentError("Incremental judgment ledger amendment evidence is invalid.")
            if str(change.get("before_sha256") or "") != canonical_sha256(before):
                raise JudgmentError("Incremental judgment ledger amendment before hash changed.")
            if str(change.get("after_sha256") or "") != canonical_sha256(after):
                raise JudgmentError("Incremental judgment ledger amendment after hash changed.")
            cluster_id = str(change.get("cluster_id") or "")
            if (
                not cluster_id.startswith("cluster-")
                or cluster_id in changed_clusters
                or cluster_id != str(before.get("cluster_id") or "")
                or cluster_id != str(after.get("cluster_id") or "")
                or str(change.get("case_id") or "") != str(before.get("case_id") or "")
                or str(change.get("case_id") or "") != str(after.get("case_id") or "")
                or not str(change.get("reason") or "").strip()
            ):
                raise JudgmentError("Incremental judgment ledger amendment binding changed.")
            changed_clusters.add(cluster_id)
            previous_after = last_amended_verdicts.get(cluster_id)
            if previous_after is not None and dict(before) != dict(previous_after):
                raise JudgmentError("Incremental judgment ledger amendment chain changed.")
            last_amended_verdicts[cluster_id] = after
    for cluster_id, latest_after in last_amended_verdicts.items():
        current = verdicts.get(cluster_id)
        if not isinstance(current, Mapping) or dict(current) != dict(latest_after):
            raise JudgmentError("Incremental judgment ledger final verdict diverged from its audit history.")


def incremental_verdict_sha256(verdict: Mapping[str, Any]) -> str:
    """Return the stable precondition hash used by audited amendments."""

    return canonical_sha256(dict(verdict))


def _merge_verdict(
    verdicts: dict[str, dict[str, Any]],
    verdict: Mapping[str, Any],
) -> None:
    cluster_id = str(verdict.get("cluster_id") or "")
    case_id = str(verdict.get("case_id") or "")
    if not cluster_id or not case_id:
        raise JudgmentError("Incremental verdict has no stable cluster or case identity.")
    normalized = {
        "cluster_id": cluster_id,
        "case_id": case_id,
        "decision": str(verdict.get("decision") or ""),
        "category": str(verdict.get("category") or ""),
        "naturalness": verdict.get("naturalness"),
        "automatic": bool(verdict.get("automatic", False)),
        "rule_version": SEMANTIC_JUDGMENT_RULE_VERSION,
    }
    if normalized["decision"] not in _FINAL_DECISIONS:
        raise JudgmentError("Incremental verdict must be final before it enters the ledger.")
    previous = verdicts.get(cluster_id)
    if previous is not None and dict(previous) != normalized:
        raise JudgmentError("Incremental judgment attempted to change an existing cluster verdict.")
    verdicts[cluster_id] = normalized


def _validate_completed_packet_against_reference(
    packet: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> None:
    cases_raw = reference.get("cases")
    if not isinstance(cases_raw, list):
        raise JudgmentError("Reusable judgment requires the frozen reference cases.")
    cases = {
        str(case.get("case_id") or ""): case
        for case in cases_raw
        if isinstance(case, Mapping)
    }
    if not cases or len(cases) != len(cases_raw):
        raise JudgmentError("Reusable judgment found invalid frozen reference identities.")
    rows = packet.get("rows")
    automatic = packet.get("automatic_pass_clusters")
    if not isinstance(rows, list) or not isinstance(automatic, list):
        raise JudgmentError("Reusable judgment packet is incomplete.")
    compared_fields = (
        "language",
        "source_text",
        "context_after_text",
        "canonical_translation",
        "required_meaning",
        "prohibited_changes",
    )
    for row in rows:
        if not isinstance(row, Mapping):
            raise JudgmentError("Reusable judgment packet contains an invalid row.")
        case_id = str(row.get("case_id") or "")
        case = cases.get(case_id)
        if case is None or any(row.get(field) != case.get(field) for field in compared_fields):
            raise JudgmentError("Reusable judgment packet does not match the frozen reference.")
        translation = row.get("candidate_translation")
        if not isinstance(translation, str):
            raise JudgmentError("Reusable judgment packet has no candidate translation.")
        output_sha = hashlib.sha256(translation.encode("utf-8")).hexdigest()
        expected_cluster = "cluster-" + canonical_sha256(
            {"case_id": case_id, "output_sha256": output_sha}
        )[:20]
        if str(row.get("cluster_id") or "") != expected_cluster:
            raise JudgmentError("Reusable judgment packet cluster identity changed.")
    for verdict in automatic:
        if not isinstance(verdict, Mapping):
            raise JudgmentError("Reusable automatic judgment is invalid.")
        case_id = str(verdict.get("case_id") or "")
        case = cases.get(case_id)
        if case is None or verdict.get("decision") != "PASS":
            raise JudgmentError("Reusable automatic judgment does not match the frozen reference.")
        canonical = str(case.get("canonical_translation") or "")
        output_sha = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expected_cluster = "cluster-" + canonical_sha256(
            {"case_id": case_id, "output_sha256": output_sha}
        )[:20]
        if str(verdict.get("cluster_id") or "") != expected_cluster:
            raise JudgmentError("Reusable automatic judgment cluster identity changed.")


def seed_ledger_from_completed_packet(
    ledger: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    packet: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    source_label: str,
) -> dict[str, Any]:
    """Import an already completed blind pass without weakening its binding."""

    validate_incremental_ledger(ledger, reference=reference)
    if packet.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
        raise JudgmentError("Reusable judgment packet schema is not current.")
    _validate_completed_packet_against_reference(packet, reference)
    resolved = apply_blind_judgments(packet, decisions)
    result = deepcopy(dict(ledger))
    verdicts = {str(key): dict(value) for key, value in dict(result["verdicts"]).items()}
    for verdict in resolved.values():
        _merge_verdict(verdicts, verdict)
    result["verdicts"] = verdicts
    provenance = {
        "source_label": str(source_label),
        "packet_sha256": canonical_sha256(packet),
        "decisions_sha256": canonical_sha256(decisions),
        "cluster_count": len(resolved),
        "rule_version": SEMANTIC_JUDGMENT_RULE_VERSION,
        "reference_sha256": str(reference.get("reference_sha256") or ""),
    }
    imports = [dict(item) for item in result.get("imports", []) if isinstance(item, Mapping)]
    if provenance not in imports:
        imports.append(provenance)
    result["imports"] = imports
    return _seal_ledger(result)


def _all_scope_packets(
    reference: Mapping[str, Any],
    records: list[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        build_blind_judgment_packet(reference, records, scope="tuning"),
        build_blind_judgment_packet(reference, records, scope="holdout"),
    )


def build_incremental_judgment_packet(
    reference: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    ledger: Mapping[str, Any],
    *,
    batch_size: int = 100,
) -> dict[str, Any]:
    """Build one bounded all-478 blind batch containing only unseen outputs."""

    validate_incremental_ledger(ledger, reference=reference)
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise JudgmentError("Incremental judgment batch size must be a positive integer.")
    record_list = list(records)
    packets = _all_scope_packets(reference, record_list)
    rows = [dict(row) for packet in packets for row in packet.get("rows", [])]
    by_cluster: dict[str, dict[str, Any]] = {}
    for row in rows:
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id or cluster_id in by_cluster:
            raise JudgmentError("All-478 blind packet contains duplicate or empty cluster identities.")
        by_cluster[cluster_id] = row
    verdicts = dict(ledger.get("verdicts") or {})
    for cluster_id, verdict in verdicts.items():
        row = by_cluster.get(str(cluster_id))
        if row is not None and str(verdict.get("case_id") or "") != str(row.get("case_id") or ""):
            raise JudgmentError("Reusable verdict no longer matches its frozen reference case.")
    pending = [row for cluster_id, row in by_cluster.items() if cluster_id not in verdicts]
    pending.sort(key=lambda row: canonical_sha256(str(row["cluster_id"])))
    selected = pending[:batch_size]
    automatic_pass = [item for packet in packets for item in packet.get("automatic_pass_clusters", [])]
    automatic_failures = [item for packet in packets for item in packet.get("automatic_verdicts", [])]
    unjudged = [item for packet in packets for item in packet.get("unjudged_responses", [])]
    payload = {
        "schema_version": INCREMENTAL_PACKET_SCHEMA_VERSION,
        "rule_version": SEMANTIC_JUDGMENT_RULE_VERSION,
        "rule": SEMANTIC_JUDGMENT_RULE,
        "reference_sha256": str(reference.get("reference_sha256") or ""),
        "arm_and_seed_hidden": True,
        "scope": "all-478",
        "observed_response_count": len(record_list),
        "observed_unique_cluster_count": len(rows) + len(automatic_pass),
        "reused_judged_cluster_count": len(rows) - len(pending),
        "pending_total_cluster_count": len(pending),
        "batch_cluster_count": len(selected),
        "automatic_exact_cluster_count": len(automatic_pass),
        "automatic_failure_response_count": sum(
            int(item.get("occurrence_count") or 0) for item in automatic_failures
        ),
        "unjudged_response_count": sum(int(item.get("occurrence_count") or 0) for item in unjudged),
        "rows": selected,
    }
    payload["packet_id"] = "batch-" + canonical_sha256(
        {
            "reference_sha256": payload["reference_sha256"],
            "rule_version": payload["rule_version"],
            "cluster_ids": [row["cluster_id"] for row in selected],
        }
    )[:24]
    return _seal_packet(payload)


def mark_pending_batch(
    ledger: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    packet: Mapping[str, Any],
    packet_file: str,
) -> dict[str, Any]:
    """Bind one persisted packet to the ledger until its decisions are applied."""

    validate_incremental_ledger(ledger, reference=reference)
    validate_incremental_packet(packet, reference=reference)
    if ledger.get("pending_batch") not in (None, {}):
        raise JudgmentError("Incremental judgment ledger already has a pending batch.")
    if not str(packet_file).startswith("judgment-batch-") or not str(packet_file).endswith(".json"):
        raise JudgmentError("Incremental judgment packet file name is invalid.")
    result = deepcopy(dict(ledger))
    result["pending_batch"] = {
        "packet_id": str(packet.get("packet_id") or ""),
        "packet_sha256": str(packet.get("packet_sha256") or ""),
        "packet_file": str(packet_file),
        "cluster_count": int(packet.get("batch_cluster_count") or 0),
    }
    return _seal_ledger(result)


def validate_incremental_packet(
    packet: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
) -> None:
    if packet.get("schema_version") != INCREMENTAL_PACKET_SCHEMA_VERSION:
        raise JudgmentError("Incremental judgment packet schema is not current.")
    if packet.get("rule_version") != SEMANTIC_JUDGMENT_RULE_VERSION or packet.get("rule") != SEMANTIC_JUDGMENT_RULE:
        raise JudgmentError("Incremental judgment packet semantic rules changed.")
    if str(packet.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise JudgmentError("Incremental judgment packet belongs to a different frozen reference.")
    if str(packet.get("packet_sha256") or "") != canonical_sha256(
        _without_sha(packet, "packet_sha256")
    ):
        raise JudgmentError("Incremental judgment packet integrity hash is invalid.")
    rows = packet.get("rows")
    if not isinstance(rows, list):
        raise JudgmentError("Incremental judgment packet has no blind row list.")
    if packet.get("arm_and_seed_hidden") is not True or packet.get("scope") != "all-478":
        raise JudgmentError("Incremental judgment packet is not an all-478 blind packet.")
    if int(packet.get("batch_cluster_count") or 0) != len(rows):
        raise JudgmentError("Incremental judgment packet row count changed.")
    cluster_ids: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise JudgmentError("Incremental judgment packet contains a non-object blind row.")
        cluster_id = str(row.get("cluster_id") or "")
        if not cluster_id.startswith("cluster-") or cluster_id in cluster_ids:
            raise JudgmentError("Incremental judgment packet contains an invalid cluster identity.")
        cluster_ids.append(cluster_id)
    expected_packet_id = "batch-" + canonical_sha256(
        {
            "reference_sha256": str(packet.get("reference_sha256") or ""),
            "rule_version": str(packet.get("rule_version") or ""),
            "cluster_ids": cluster_ids,
        }
    )[:24]
    if str(packet.get("packet_id") or "") != expected_packet_id:
        raise JudgmentError("Incremental judgment packet batch identity changed.")
    leaked_key = _forbidden_blind_key(packet)
    if leaked_key is not None:
        raise JudgmentError(
            f"Incremental judgment packet leaked sampler execution identity: {leaked_key}."
        )


def apply_incremental_judgments(
    ledger: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    packet: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    applied_utc: str,
) -> dict[str, Any]:
    """Atomically merge exactly one fully decided blind batch into the ledger."""

    validate_incremental_ledger(ledger, reference=reference)
    validate_incremental_packet(packet, reference=reference)
    pending = ledger.get("pending_batch")
    if not isinstance(pending, Mapping):
        raise JudgmentError("Incremental decisions require the ledger's pending batch binding.")
    if (
        str(pending.get("packet_id") or "") != str(packet.get("packet_id") or "")
        or str(pending.get("packet_sha256") or "") != str(packet.get("packet_sha256") or "")
    ):
        raise JudgmentError("Incremental decisions do not match the ledger's pending batch.")
    pseudo_packet = {
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "rows": list(packet.get("rows") or []),
        "automatic_pass_clusters": [],
    }
    resolved = apply_blind_judgments(pseudo_packet, decisions)
    if any(str(verdict.get("decision") or "") not in _FINAL_DECISIONS for verdict in resolved.values()):
        raise JudgmentError("Incremental judgment batch contains unresolved review decisions.")
    result = deepcopy(dict(ledger))
    verdicts = {str(key): dict(value) for key, value in dict(result["verdicts"]).items()}
    for verdict in resolved.values():
        _merge_verdict(verdicts, verdict)
    result["verdicts"] = verdicts
    history = [dict(item) for item in result.get("applied_batches", []) if isinstance(item, Mapping)]
    packet_id = str(packet.get("packet_id") or "")
    batch_record = {
        "packet_id": packet_id,
        "packet_sha256": str(packet.get("packet_sha256") or ""),
        "decisions_sha256": canonical_sha256(decisions),
        "cluster_count": len(resolved),
        "applied_utc": str(applied_utc),
    }
    if any(str(item.get("packet_id") or "") == packet_id for item in history):
        raise JudgmentError("Incremental judgment packet was already applied.")
    history.append(batch_record)
    result["applied_batches"] = history
    result["pending_batch"] = None
    current_number = result.get("next_batch_number")
    if isinstance(current_number, bool) or not isinstance(current_number, int) or current_number <= 0:
        raise JudgmentError("Incremental judgment ledger batch counter is invalid.")
    result["next_batch_number"] = current_number + 1
    return _seal_ledger(result)


def apply_incremental_amendments(
    ledger: Mapping[str, Any],
    *,
    reference: Mapping[str, Any],
    amendments: Mapping[str, Mapping[str, Any]],
    reason: str,
    applied_utc: str,
) -> dict[str, Any]:
    """Replace reviewed manual verdicts while preserving sealed before/after evidence."""

    validate_incremental_ledger(ledger, reference=reference)
    if ledger.get("pending_batch") not in (None, {}):
        raise JudgmentError("Incremental amendments require no pending blind batch.")
    normalized_reason = str(reason or "").strip()
    if not normalized_reason:
        raise JudgmentError("Incremental amendments require an overall reason.")
    if not amendments:
        raise JudgmentError("Incremental amendments require at least one cluster.")

    result = deepcopy(dict(ledger))
    verdicts = {str(key): dict(value) for key, value in dict(result["verdicts"]).items()}
    changes: list[dict[str, Any]] = []
    for cluster_id in sorted(amendments):
        requested = amendments[cluster_id]
        if not isinstance(requested, Mapping):
            raise JudgmentError("Incremental amendment entry is invalid.")
        previous = verdicts.get(str(cluster_id))
        if previous is None:
            raise JudgmentError("Incremental amendment references an unknown cluster verdict.")
        if bool(previous.get("automatic", False)):
            raise JudgmentError("Incremental amendment cannot replace an automatic verdict.")
        expected_previous = str(requested.get("expected_previous_sha256") or "")
        if expected_previous != incremental_verdict_sha256(previous):
            raise JudgmentError("Incremental amendment previous verdict hash does not match.")
        decision = str(requested.get("decision") or "")
        category = str(requested.get("category") or "").strip()
        naturalness = requested.get("naturalness")
        item_reason = str(requested.get("reason") or "").strip()
        if decision not in _FINAL_DECISIONS or not category or not item_reason:
            raise JudgmentError("Incremental amendment replacement is incomplete.")
        if (
            isinstance(naturalness, bool)
            or not isinstance(naturalness, (int, float))
            or not 0 <= float(naturalness) <= 5
        ):
            raise JudgmentError("Incremental amendment naturalness must be between 0 and 5.")
        replacement = {
            "cluster_id": str(cluster_id),
            "case_id": str(previous.get("case_id") or ""),
            "decision": decision,
            "category": category,
            "naturalness": naturalness,
            "automatic": False,
            "rule_version": SEMANTIC_JUDGMENT_RULE_VERSION,
        }
        if replacement == previous:
            raise JudgmentError("Incremental amendment does not change the existing verdict.")
        change = {
            "cluster_id": str(cluster_id),
            "case_id": replacement["case_id"],
            "reason": item_reason,
            "before_sha256": incremental_verdict_sha256(previous),
            "after_sha256": incremental_verdict_sha256(replacement),
            "before": previous,
            "after": replacement,
        }
        changes.append(change)
        verdicts[str(cluster_id)] = replacement

    amendment_id = "amendment-" + canonical_sha256(
        {
            "reference_sha256": str(result.get("reference_sha256") or ""),
            "rule_version": str(result.get("rule_version") or ""),
            "reason": normalized_reason,
            "changes": changes,
        }
    )[:24]
    history = [
        dict(item)
        for item in result.get("amendments", [])
        if isinstance(item, Mapping)
    ]
    if any(str(item.get("amendment_id") or "") == amendment_id for item in history):
        raise JudgmentError("Incremental amendment was already applied.")
    history.append(
        {
            "amendment_id": amendment_id,
            "reason": normalized_reason,
            "changes_sha256": canonical_sha256(changes),
            "cluster_count": len(changes),
            "applied_utc": str(applied_utc),
            "changes": changes,
        }
    )
    result["verdicts"] = verdicts
    result["amendments"] = history
    return _seal_ledger(result)


def bind_incremental_ledger_to_records(
    reference: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    ledger: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Bind the cumulative all-478 ledger to every actual response slot."""

    validate_incremental_ledger(ledger, reference=reference)
    cases_raw = reference.get("cases")
    if not isinstance(cases_raw, list):
        raise JudgmentError("Frozen reference has no case list for final binding.")
    cases = {
        str(case.get("case_id") or ""): case
        for case in cases_raw
        if isinstance(case, Mapping)
    }
    verdicts = dict(ledger.get("verdicts") or {})
    bound: dict[str, dict[str, Any]] = {}
    for record in records:
        slot = str(record.get("logical_slot") or "")
        case_id = str(record.get("case_id") or "")
        validation = record.get("response_validation")
        if not slot or slot in bound or case_id not in cases or not isinstance(validation, Mapping):
            raise JudgmentError("Final incremental binding found an invalid response identity.")
        if validation.get("schema_version") != RESPONSE_VALIDATION_SCHEMA_VERSION:
            raise JudgmentError("Final incremental binding requires the current quality view.")
        status = str(validation.get("status") or "")
        if status == "CATASTROPHIC":
            verdict = {
                "decision": "CATASTROPHIC",
                "category": str(validation.get("category") or "translation_quality_failure"),
                "naturalness": 0,
                "automatic": True,
            }
        elif status == "UNJUDGED":
            verdict = {
                "decision": "UNJUDGED",
                "category": str(validation.get("category") or "translation_unavailable"),
                "naturalness": None,
                "automatic": True,
            }
        elif status == "VALID":
            translation = record.get("translation")
            output_sha = str(validation.get("translation_sha256") or "")
            if not isinstance(translation, str) or not output_sha:
                raise JudgmentError("Valid response has no translated text for final binding.")
            if translation == str(cases[case_id].get("canonical_translation") or ""):
                verdict = {
                    "decision": "PASS",
                    "category": "exact_canonical",
                    "naturalness": 5,
                    "automatic": True,
                }
            else:
                cluster_id = "cluster-" + canonical_sha256(
                    {"case_id": case_id, "output_sha256": output_sha}
                )[:20]
                stored = verdicts.get(cluster_id)
                if not isinstance(stored, Mapping) or str(stored.get("case_id") or "") != case_id:
                    raise JudgmentError("Final campaign still has an unjudged unique translation cluster.")
                verdict = dict(stored)
        else:
            raise JudgmentError("Final incremental binding found an unknown validation status.")
        verdict["logical_slot"] = slot
        verdict["case_id"] = case_id
        bound[slot] = verdict
    return bound
