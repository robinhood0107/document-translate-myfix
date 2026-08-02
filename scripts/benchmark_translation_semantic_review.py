#!/usr/bin/env python3
"""Private, hash-bound text-first semantic review for translation benchmarks.

The checked-out approval contains only hashes, counts, review scope, and a
classification.  Source text, full dialogue context, translations, and any
optional page remain in the managed private archive.  A raw response mismatch
is therefore diagnostic; only a validated review can decide whether it is an
acceptable wording difference, requires a user decision, or is a semantic
regression.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SEMANTIC_REVIEW_SCHEMA_VERSION = "gemma-translation-semantic-approval-v2"
TEXT_FIRST_REVIEW_METHOD = "source-full-neighbor-text-first"
TEXT_ONLY_SCOPE = "text_only"
TEXT_PLUS_PAGE_SCOPE = "text_plus_target_page"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_PASS_CLASSIFICATIONS = frozenset(
    {
        "equivalent_wording",
        "formatting_only",
        "honorific_equivalent",
        "name_transliteration_equivalent",
        "style_or_register",
    }
)
_REJECT_CATEGORIES = frozenset(
    {
        "censorship_or_deletion",
        "sensitive_expression_softening",
        "negation_consent_or_coercion_change",
        "speaker_target_or_action_change",
        "relationship_or_number_change",
        "other_semantic_regression",
    }
)
_PASS_ATTESTATIONS = (
    "source_text_attested",
    "full_dialogue_context_attested",
    "neighbor_dialogue_context_attested",
    "baseline_translation_attested",
    "candidate_translation_attested",
    "semantic_equivalent",
    "no_deletion",
    "no_censorship",
    "sensitive_expression_preserved",
    "no_sensitive_expression_softening",
    "no_negation_change",
    "no_consent_change",
    "no_coercion_change",
    "no_speaker_change",
    "no_target_change",
    "no_action_change",
    "no_relationship_change",
    "no_number_change",
)


class SemanticReviewError(ValueError):
    """Raised when a semantic-review artifact cannot safely approve a run."""


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SemanticReviewError(f"{label} must be an object.")
    return value


def _nonempty_string(value: Any, *, label: str) -> str:
    text = str(value or "")
    if not text:
        raise SemanticReviewError(f"{label} is required.")
    return text


def _candidate_key(result: Mapping[str, Any]) -> str:
    candidate = _mapping(result.get("candidate"), label="candidate")
    return _nonempty_string(candidate.get("key"), label="candidate.key")


def _run_fingerprint(result: Mapping[str, Any]) -> str:
    candidate = _mapping(result.get("candidate"), label="candidate")
    runtime = result.get("runtime")
    return canonical_sha256(
        {
            "candidate": dict(candidate),
            "runtime": dict(runtime) if isinstance(runtime, Mapping) else {},
        }
    )


def _request_ledger_sha256(result: Mapping[str, Any]) -> str:
    ledger = _mapping(result.get("request_ledger"), label="request_ledger")
    return _nonempty_string(ledger.get("sha256"), label="request_ledger.sha256")


def _response_ledger(result: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], str]:
    ledger = _mapping(result.get("response_ledger"), label="response_ledger")
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise SemanticReviewError("response_ledger.rows must be a list.")
    normalized = [_mapping(row, label="response_ledger row") for row in rows]
    observed_sha256 = _nonempty_string(ledger.get("sha256"), label="response_ledger.sha256")
    expected_sha256 = canonical_sha256(normalized)
    if observed_sha256 != expected_sha256:
        raise SemanticReviewError("response_ledger.sha256 does not match its rows.")
    return normalized, observed_sha256


def validate_translation_response(response: Mapping[str, Any], *, index: int) -> str:
    """Validate the hard JSON/completion contract and return canonical content."""

    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise SemanticReviewError(f"response row {index} must contain exactly one choice.")
    choice = _mapping(choices[0], label=f"response row {index} choice")
    try:
        choice_index = int(choice.get("index"))
    except (TypeError, ValueError) as exc:
        raise SemanticReviewError(f"response row {index} choice index is invalid.") from exc
    if choice_index != 0:
        raise SemanticReviewError(f"response row {index} choice index must be zero.")
    if str(choice.get("finish_reason", "") or "") != "stop":
        raise SemanticReviewError(f"response row {index} did not finish with stop.")
    content = _nonempty_string(choice.get("content"), label=f"response row {index} content")
    try:
        decoded = json.loads(content)
    except json.JSONDecodeError:
        # Match the product parser: some llama.cpp builds expose a reasoning
        # prefix before an otherwise complete JSON object.  The complete
        # object is still required; a truncated or malformed tail cannot pass.
        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end <= start:
            raise SemanticReviewError(f"response row {index} content is not JSON.")
        try:
            decoded = json.loads(content[start : end + 1])
        except json.JSONDecodeError as exc:
            raise SemanticReviewError(f"response row {index} content is not JSON.") from exc
    if not isinstance(decoded, Mapping) or not isinstance(decoded.get("translation"), str):
        raise SemanticReviewError(
            f"response row {index} must contain a string translation field."
        )
    return content


def build_replay_comparison(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a content-only replay delta after all hard contracts pass."""

    baseline_request = _request_ledger_sha256(baseline)
    candidate_request = _request_ledger_sha256(candidate)
    if baseline_request != candidate_request:
        raise SemanticReviewError("request ledger differs between compared runs.")
    baseline_rows, baseline_response = _response_ledger(baseline)
    candidate_rows, candidate_response = _response_ledger(candidate)
    if len(baseline_rows) != len(candidate_rows):
        raise SemanticReviewError("response row count differs between compared runs.")

    mismatch_indices: list[int] = []
    for index, (baseline_row, candidate_row) in enumerate(zip(baseline_rows, candidate_rows)):
        baseline_content = validate_translation_response(baseline_row, index=index)
        candidate_content = validate_translation_response(candidate_row, index=index)
        if baseline_content != candidate_content:
            mismatch_indices.append(index)

    return {
        "comparison_kind": "replay-response",
        "baseline_key": _candidate_key(baseline),
        "candidate_key": _candidate_key(candidate),
        "request_ledger_sha256": baseline_request,
        "baseline_response_ledger_sha256": baseline_response,
        "candidate_response_ledger_sha256": candidate_response,
        "baseline_runtime_fingerprint": _run_fingerprint(baseline),
        "candidate_runtime_fingerprint": _run_fingerprint(candidate),
        "mismatch_indices": mismatch_indices,
        "mismatch_indices_sha256": canonical_sha256(mismatch_indices),
    }


def _snapshot_sha256(result: Mapping[str, Any], *, key: str) -> str:
    value = _nonempty_string(result.get(key), label=key)
    if not _SHA256_RE.fullmatch(value):
        raise SemanticReviewError(f"{key} must be a SHA-256 value.")
    return value


def _page_output_hashes(result: Mapping[str, Any]) -> list[str]:
    values = result.get("page_output_sha256")
    if not isinstance(values, list) or not values:
        raise SemanticReviewError("page_output_sha256 must be a non-empty list.")
    hashes = [str(value or "") for value in values]
    if not all(_SHA256_RE.fullmatch(value) for value in hashes):
        raise SemanticReviewError("page_output_sha256 contains an invalid value.")
    return hashes


def build_full_auto_comparison(
    *,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind upstream exactness and render completion without gating final pixels.

    A Turbo4 wording difference may change rendered pixels.  That final-output
    delta is recorded only as a diagnostic; detection/OCR/inpaint snapshot
    equality and successful render completion remain hard contracts.
    """

    baseline_request = _request_ledger_sha256(baseline)
    candidate_request = _request_ledger_sha256(candidate)
    if baseline_request != candidate_request:
        raise SemanticReviewError("request ledger differs between compared runs.")
    baseline_upstream = _snapshot_sha256(baseline, key="pre_translation_snapshot_sha256")
    candidate_upstream = _snapshot_sha256(candidate, key="pre_translation_snapshot_sha256")
    if baseline_upstream != candidate_upstream:
        raise SemanticReviewError("pre-translation snapshot differs between compared runs.")
    baseline_pages = _page_output_hashes(baseline)
    candidate_pages = _page_output_hashes(candidate)
    if len(baseline_pages) != len(candidate_pages):
        raise SemanticReviewError("full-auto page count differs between compared runs.")
    output_mismatch_indices = [
        index
        for index, (baseline_hash, candidate_hash) in enumerate(zip(baseline_pages, candidate_pages))
        if baseline_hash != candidate_hash
    ]
    return {
        "comparison_kind": "full-auto-render-completion",
        "baseline_key": _candidate_key(baseline),
        "candidate_key": _candidate_key(candidate),
        "request_ledger_sha256": baseline_request,
        "pre_translation_snapshot_sha256": baseline_upstream,
        "baseline_runtime_fingerprint": _run_fingerprint(baseline),
        "candidate_runtime_fingerprint": _run_fingerprint(candidate),
        "render_page_count": len(baseline_pages),
        "final_output_mismatch_count": len(output_mismatch_indices),
        "final_output_mismatch_indices_sha256": canonical_sha256(output_mismatch_indices),
        "mismatch_indices": [],
        "mismatch_indices_sha256": canonical_sha256([]),
    }


def _template_item(index: int) -> dict[str, Any]:
    return {
        "index": index,
        "decision": "REVIEW",
        "review_scope": TEXT_ONLY_SCOPE,
        "page_checked": False,
        "requires_user_confirmation": True,
        "classification": "",
        "rejection_category": "",
        "attestations": {name: False for name in _PASS_ATTESTATIONS},
    }


def build_review_template(
    *,
    protocol_version: str,
    stage: str,
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    required = [dict(item) for item in comparisons if item.get("mismatch_indices")]
    return {
        "schema_version": SEMANTIC_REVIEW_SCHEMA_VERSION,
        "protocol_version": protocol_version,
        "stage": stage,
        "review_method": TEXT_FIRST_REVIEW_METHOD,
        "decision": "REVIEW_REQUIRED",
        "comparison_bindings_sha256": canonical_sha256(required),
        "comparisons": [
            {
                **binding,
                "reviewed_count": 0,
                "unresolved_count": len(binding["mismatch_indices"]),
                "semantic_reject_count": 0,
                "items": [_template_item(index) for index in binding["mismatch_indices"]],
            }
            for binding in required
        ],
    }


def _item_outcome(row: Mapping[str, Any]) -> str:
    decision = str(row.get("decision", "") or "")
    if decision == "REVIEW":
        if row.get("requires_user_confirmation") is not True:
            raise SemanticReviewError("unresolved semantic item must require user confirmation.")
        return "REVIEW"
    if decision == "REJECT":
        category = str(row.get("rejection_category", "") or "")
        if category not in _REJECT_CATEGORIES:
            raise SemanticReviewError("semantic reject item lacks a valid rejection category.")
        return "REJECT"
    if decision != "PASS":
        raise SemanticReviewError("semantic approval item has an invalid decision.")
    scope = str(row.get("review_scope", "") or "")
    page_checked = row.get("page_checked")
    if scope not in {TEXT_ONLY_SCOPE, TEXT_PLUS_PAGE_SCOPE}:
        raise SemanticReviewError("semantic approval item has an invalid review scope.")
    if not isinstance(page_checked, bool) or page_checked != (scope == TEXT_PLUS_PAGE_SCOPE):
        raise SemanticReviewError("page_checked must match the declared review scope.")
    if row.get("requires_user_confirmation") is not False:
        raise SemanticReviewError("approved semantic item cannot require user confirmation.")
    if str(row.get("classification", "") or "") not in _PASS_CLASSIFICATIONS:
        raise SemanticReviewError("semantic approval item has an invalid PASS classification.")
    attestations = _mapping(row.get("attestations"), label="semantic approval attestations")
    missing = [name for name in _PASS_ATTESTATIONS if attestations.get(name) is not True]
    if missing:
        raise SemanticReviewError(
            "semantic approval item lacks required text-context attestation: "
            + ", ".join(missing)
        )
    return "PASS"


def _validate_review_rows(item: Mapping[str, Any], expected_indices: list[int]) -> tuple[int, int, int]:
    rows = item.get("items")
    if not isinstance(rows, list):
        raise SemanticReviewError("semantic approval items must be a list.")
    by_index: dict[int, Mapping[str, Any]] = {}
    for row in rows:
        mapping = _mapping(row, label="semantic approval item")
        try:
            index = int(mapping.get("index"))
        except (TypeError, ValueError) as exc:
            raise SemanticReviewError("semantic approval item index is invalid.") from exc
        if index in by_index:
            raise SemanticReviewError("semantic approval has duplicate mismatch indexes.")
        by_index[index] = mapping
    if sorted(by_index) != expected_indices:
        raise SemanticReviewError("semantic approval mismatch indexes do not match.")
    outcomes = [_item_outcome(by_index[index]) for index in expected_indices]
    reviewed = sum(outcome != "REVIEW" for outcome in outcomes)
    unresolved = sum(outcome == "REVIEW" for outcome in outcomes)
    rejected = sum(outcome == "REJECT" for outcome in outcomes)
    if int(item.get("reviewed_count", -1)) != reviewed:
        raise SemanticReviewError("semantic approval reviewed count does not match.")
    if int(item.get("unresolved_count", -1)) != unresolved:
        raise SemanticReviewError("semantic approval unresolved count does not match.")
    if int(item.get("semantic_reject_count", -1)) != rejected:
        raise SemanticReviewError("semantic approval reject count does not match.")
    return reviewed, unresolved, rejected


def validate_semantic_approval(
    approval: Mapping[str, Any],
    *,
    protocol_version: str,
    stage: str,
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate a private approval and return PASS, REVIEW_REQUIRED, or REJECT.

    Hash/schema/runtime/request contract drift raises an error.  A reviewed
    semantic regression is an intentional REJECT; an unresolved item remains
    REVIEW_REQUIRED so the caller can ask the user without rerunning the GPU.
    """

    required = [dict(item) for item in comparisons if item.get("mismatch_indices")]
    if not required:
        return {
            "status": "NOT_REQUIRED",
            "mismatch_count": 0,
            "comparison_count": 0,
            "unresolved_count": 0,
            "semantic_reject_count": 0,
            "bindings": [],
        }
    payload = _mapping(approval, label="semantic approval")
    if payload.get("schema_version") != SEMANTIC_REVIEW_SCHEMA_VERSION:
        raise SemanticReviewError("semantic approval schema version does not match.")
    if payload.get("protocol_version") != protocol_version:
        raise SemanticReviewError("semantic approval protocol version does not match.")
    if payload.get("stage") != stage:
        raise SemanticReviewError("semantic approval stage does not match.")
    if payload.get("review_method") != TEXT_FIRST_REVIEW_METHOD:
        raise SemanticReviewError("semantic approval review method does not match.")
    if payload.get("comparison_bindings_sha256") != canonical_sha256(required):
        raise SemanticReviewError("semantic approval comparison binding does not match.")
    observed = payload.get("comparisons")
    if not isinstance(observed, list) or len(observed) != len(required):
        raise SemanticReviewError("semantic approval comparison count does not match.")

    mismatch_count = 0
    unresolved_count = 0
    semantic_reject_count = 0
    for expected, actual_value in zip(required, observed):
        actual = _mapping(actual_value, label="semantic approval comparison")
        for key, value in expected.items():
            if actual.get(key) != value:
                raise SemanticReviewError(f"semantic approval {key} does not match.")
        expected_indices = [int(index) for index in expected["mismatch_indices"]]
        _, unresolved, rejected = _validate_review_rows(actual, expected_indices)
        mismatch_count += len(expected_indices)
        unresolved_count += unresolved
        semantic_reject_count += rejected

    status = (
        "REJECT"
        if semantic_reject_count
        else "REVIEW_REQUIRED"
        if unresolved_count
        else "PASS"
    )
    if payload.get("decision") != status:
        raise SemanticReviewError("semantic approval decision does not match reviewed items.")
    return {
        "status": status,
        "mismatch_count": mismatch_count,
        "comparison_count": len(required),
        "unresolved_count": unresolved_count,
        "semantic_reject_count": semantic_reject_count,
        "bindings": required,
    }


def evaluate_semantic_review(
    *,
    protocol_version: str,
    stage: str,
    comparisons: Sequence[Mapping[str, Any]],
    approval: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not any(item.get("mismatch_indices") for item in comparisons):
        return {
            "status": "NOT_REQUIRED",
            "mismatch_count": 0,
            "comparison_count": 0,
            "unresolved_count": 0,
            "semantic_reject_count": 0,
            "bindings": [],
        }
    if approval is None:
        template = build_review_template(
            protocol_version=protocol_version,
            stage=stage,
            comparisons=comparisons,
        )
        return {
            "status": "REVIEW_REQUIRED",
            "mismatch_count": sum(len(item["mismatch_indices"]) for item in comparisons),
            "comparison_count": len(template["comparisons"]),
            "unresolved_count": sum(len(item["mismatch_indices"]) for item in comparisons),
            "semantic_reject_count": 0,
            "bindings": [dict(item) for item in comparisons if item.get("mismatch_indices")],
            "template": template,
        }
    return validate_semantic_approval(
        approval,
        protocol_version=protocol_version,
        stage=stage,
        comparisons=comparisons,
    )


def load_semantic_approval(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticReviewError("semantic approval is unreadable.") from exc
    return dict(_mapping(payload, label="semantic approval"))
