"""Blind, private-only response validation and sampler-quality ranking."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping, Sequence

from .corpus import CorpusError
from .protocol import SamplerTuple, canonical_sha256


JUDGMENT_SCHEMA_VERSION = "gemma-sampler-judgment-v2"
_CHANNEL_OR_REASONING = re.compile(
    r"<\|channel|<channel\||<\|(?:analysis|thought|reasoning)|</?(?:analysis|thought|reasoning)\b",
    re.IGNORECASE,
)
_MIXED_TOKEN_DAMAGE = re.compile(r"[가-힣][A-Za-z]+[가-힣]")
_VALID_DECISIONS = {"PASS", "MINOR", "MAJOR", "CATASTROPHIC", "REVIEW_REQUIRED"}
_SEVERITY_ORDER = {"PASS": 0, "MINOR": 1, "MAJOR": 2, "CATASTROPHIC": 3}


class JudgmentError(RuntimeError):
    """Raised when a response or private review ledger violates v2 rules."""


class _DuplicateKeyError(ValueError):
    pass


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKeyError(str(key))
        result[key] = value
    return result


@dataclass(frozen=True)
class ResponseVerdict:
    status: str
    category: str
    translation: str
    translation_sha256: str
    message: str

    @property
    def catastrophic(self) -> bool:
        return self.status == "CATASTROPHIC"

    def payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "category": self.category,
            "translation_sha256": self.translation_sha256,
            "message": self.message,
        }


def _catastrophic(category: str, message: str) -> ResponseVerdict:
    return ResponseVerdict(
        status="CATASTROPHIC",
        category=category,
        translation="",
        translation_sha256="",
        message=message,
    )


def _choice_content(envelope: Mapping[str, Any]) -> tuple[str, str | None]:
    choices = envelope.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise JudgmentError("response choice count is not exactly one")
    choice = choices[0]
    if not isinstance(choice, Mapping) or choice.get("index") != 0:
        raise JudgmentError("response choice index is not zero")
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop":
        raise JudgmentError("response did not finish with stop")
    content: Any = choice.get("content")
    if content is None and isinstance(choice.get("message"), Mapping):
        content = choice["message"].get("content")
    if not isinstance(content, str):
        raise JudgmentError("response content is not a string")
    return content, str(finish_reason)


def _strict_json_object(content: str) -> Mapping[str, Any]:
    decoder = json.JSONDecoder(object_pairs_hook=_unique_object)
    start = len(content) - len(content.lstrip())
    try:
        parsed, end = decoder.raw_decode(content, start)
    except _DuplicateKeyError as exc:
        raise JudgmentError("response JSON contains duplicate keys") from exc
    except json.JSONDecodeError as exc:
        raise JudgmentError("response content is not one JSON object") from exc
    if content[end:].strip():
        raise JudgmentError("response JSON has trailing data")
    if not isinstance(parsed, Mapping):
        raise JudgmentError("response JSON root is not an object")
    return parsed


def validate_response_envelope(envelope: Mapping[str, Any]) -> ResponseVerdict:
    """Classify raw Router output before any product sanitizer could alter it."""

    try:
        content, _finish_reason = _choice_content(envelope)
    except JudgmentError as exc:
        return _catastrophic("json_schema_order_count_or_finish", str(exc))
    if _CHANNEL_OR_REASONING.search(content):
        return _catastrophic("raw_channel_or_reasoning_leak", "raw channel or reasoning token leaked")
    try:
        payload = _strict_json_object(content)
    except JudgmentError as exc:
        return _catastrophic("json_schema_order_count_or_finish", str(exc))
    if set(payload) != {"translation"}:
        return _catastrophic("json_schema_order_count_or_finish", "response schema is not exactly translation")
    translation = payload.get("translation")
    if not isinstance(translation, str):
        return _catastrophic("json_schema_order_count_or_finish", "translation is not a string")
    if not translation.strip():
        return _catastrophic("censorship_or_deletion", "translation is empty")
    if _CHANNEL_OR_REASONING.search(translation):
        return _catastrophic("raw_channel_or_reasoning_leak", "translation contains raw channel or reasoning token")
    if _MIXED_TOKEN_DAMAGE.search(translation):
        return _catastrophic("mixed_token_corruption", "translation contains Korean/Latin token corruption")
    return ResponseVerdict(
        status="VALID",
        category="",
        translation=translation,
        translation_sha256=hashlib.sha256(translation.encode("utf-8")).hexdigest(),
        message="",
    )


def _case_by_id(reference: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    cases = reference.get("cases")
    if not isinstance(cases, list):
        raise CorpusError("Private reference cases are missing.")
    indexed = {str(case.get("case_id") or ""): case for case in cases if isinstance(case, Mapping)}
    if not indexed or len(indexed) != len(cases):
        raise CorpusError("Private reference case ids are invalid.")
    return indexed


def _record_sampler(record: Mapping[str, Any]) -> SamplerTuple:
    raw = record.get("sampler")
    if not isinstance(raw, Mapping):
        raise JudgmentError("Response record is missing sampler identity.")
    return SamplerTuple(
        float(raw.get("temperature")),
        float(raw.get("top_p")),
        int(raw.get("top_k")),
        float(raw.get("min_p")),
    )


def build_blind_judgment_packet(
    reference: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    scope: str = "tuning",
    allowed_sampler_keys: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Group exact outputs while hiding arm, sampler, seed, and run ordering."""

    if scope not in {"tuning", "holdout"}:
        raise JudgmentError("Judgment scope must be tuning or holdout.")
    cases = _case_by_id(reference)
    allowed = set(allowed_sampler_keys or ())
    candidate_sampler_keys: set[str] = set()
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    automatic_counts: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        case_id = str(record.get("case_id") or "")
        case = cases.get(case_id)
        if case is None or case.get("split") != scope:
            continue
        sampler = _record_sampler(record)
        if allowed and sampler.key not in allowed:
            continue
        candidate_sampler_keys.add(sampler.key)
        validation = record.get("response_validation")
        if not isinstance(validation, Mapping):
            raise JudgmentError("Response record lacks raw response validation.")
        status = str(validation.get("status") or "")
        logical_slot = str(record.get("logical_slot") or "")
        if not logical_slot:
            raise JudgmentError("Response record lacks logical slot identity.")
        if status == "CATASTROPHIC":
            automatic_counts[
                (case_id, str(validation.get("category") or "json_schema_order_count_or_finish"))
            ] += 1
            continue
        if status != "VALID":
            raise JudgmentError("Response record has an unknown validation status.")
        output_sha = str(validation.get("translation_sha256") or "")
        translation = record.get("translation")
        if not output_sha or not isinstance(translation, str):
            raise JudgmentError("Valid response record lacks its private translation.")
        grouped[(case_id, output_sha)].append(record)
    rows: list[dict[str, Any]] = []
    auto_pass_clusters: list[dict[str, Any]] = []
    for (case_id, output_sha), entries in grouped.items():
        case = cases[case_id]
        translation = str(entries[0]["translation"])
        cluster_id = "cluster-" + canonical_sha256(
            {"case_id": case_id, "output_sha256": output_sha}
        )[:20]
        slots = sorted(str(entry["logical_slot"]) for entry in entries)
        if translation == str(case.get("canonical_translation") or ""):
            auto_pass_clusters.append(
                {
                    "cluster_id": cluster_id,
                    "case_id": case_id,
                    "decision": "PASS",
                    "category": "exact_canonical",
                    "naturalness": 5,
                    "automatic": True,
                    "occurrence_count": len(slots),
                }
            )
            continue
        rows.append(
            {
                "cluster_id": cluster_id,
                "case_id": case_id,
                "language": case.get("language"),
                "source_text": case.get("source_text"),
                "context_after_text": case.get("context_after_text"),
                "canonical_translation": case.get("canonical_translation"),
                "required_meaning": list(case.get("required_meaning") or []),
                "prohibited_changes": list(case.get("prohibited_changes") or []),
                "candidate_translation": translation,
                "occurrence_count": len(slots),
                "decision": "",
                "category": "",
                "naturalness": None,
                "note": "",
            }
        )
    rows.sort(key=lambda row: hashlib.sha256(str(row["cluster_id"]).encode("utf-8")).hexdigest())
    return {
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "scope": scope,
        "arm_and_seed_hidden": True,
        "candidate_sampler_keys_sha256": canonical_sha256(sorted(candidate_sampler_keys)),
        "pending_cluster_count": len(rows),
        "automatic_verdicts": [
            {
                "case_id": case_id,
                "decision": "CATASTROPHIC",
                "category": category,
                "naturalness": 0,
                "automatic": True,
                "occurrence_count": count,
            }
            for (case_id, category), count in sorted(automatic_counts.items())
        ],
        "automatic_pass_clusters": auto_pass_clusters,
        "rows": rows,
    }


def apply_blind_judgments(
    packet: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return logical-slot verdicts with each cluster decision propagated."""

    if packet.get("schema_version") != JUDGMENT_SCHEMA_VERSION:
        raise JudgmentError("Judgment packet schema version is not v2.")
    rows = packet.get("rows")
    automatic_pass = packet.get("automatic_pass_clusters")
    if not isinstance(rows, list) or not isinstance(automatic_pass, list):
        raise JudgmentError("Judgment packet is incomplete.")
    expected = {str(row.get("cluster_id") or "") for row in rows if isinstance(row, Mapping)}
    if set(decisions) != expected:
        raise JudgmentError("Blind judgments must cover every pending output cluster exactly once.")
    result: dict[str, dict[str, Any]] = {}
    for verdict in automatic_pass:
        if not isinstance(verdict, Mapping):
            raise JudgmentError("Automatic judgment record is invalid.")
        cluster_id = str(verdict.get("cluster_id") or "")
        if not cluster_id:
            raise JudgmentError("Automatic judgment has no cluster identity.")
        result[cluster_id] = dict(verdict)
    for row in rows:
        if not isinstance(row, Mapping):
            raise JudgmentError("Judgment row is invalid.")
        cluster_id = str(row.get("cluster_id") or "")
        decision = decisions[cluster_id]
        verdict = str(decision.get("decision") or "").upper()
        category = str(decision.get("category") or "").strip()
        naturalness = decision.get("naturalness")
        if verdict not in _VALID_DECISIONS or not category:
            raise JudgmentError("Blind judgment requires a valid decision and category.")
        if verdict != "REVIEW_REQUIRED":
            if isinstance(naturalness, bool) or not isinstance(naturalness, int) or not 0 <= naturalness <= 5:
                raise JudgmentError("Blind judgment naturalness must be an integer from 0 to 5.")
        result[cluster_id] = {
            "cluster_id": cluster_id,
            "case_id": str(row.get("case_id") or ""),
            "decision": verdict,
            "category": category,
            "naturalness": naturalness,
            "automatic": False,
        }
    return result


def bind_cluster_verdicts_to_records(
    records: Iterable[Mapping[str, Any]],
    packet: Mapping[str, Any],
    cluster_decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Bind hidden-cluster decisions to actual slots after the blind pass."""

    record_list = list(records)
    by_cluster = apply_blind_judgments(packet, cluster_decisions)
    candidate_sampler_keys = sorted(_record_sampler(record).key for record in record_list)
    expected_sampler_set_sha = canonical_sha256(sorted(set(candidate_sampler_keys)))
    if str(packet.get("candidate_sampler_keys_sha256") or "") != expected_sampler_set_sha:
        raise JudgmentError("Blind judgment packet does not match the selected sampler tuple set.")
    result: dict[str, dict[str, Any]] = {}
    for record in record_list:
        validation = record.get("response_validation")
        if not isinstance(validation, Mapping):
            continue
        slot = str(record.get("logical_slot") or "")
        case_id = str(record.get("case_id") or "")
        if not slot:
            raise JudgmentError("Response record lacks logical slot during judgment binding.")
        if validation.get("status") == "CATASTROPHIC":
            result[slot] = {
                "logical_slot": slot,
                "case_id": case_id,
                "decision": "CATASTROPHIC",
                "category": str(validation.get("category") or "json_schema_order_count_or_finish"),
                "naturalness": 0,
                "automatic": True,
            }
            continue
        if validation.get("status") != "VALID":
            raise JudgmentError("Response record has an unknown validation status during binding.")
        output_sha = str(validation.get("translation_sha256") or "")
        cluster_id = "cluster-" + canonical_sha256(
            {"case_id": case_id, "output_sha256": output_sha}
        )[:20]
        verdict = by_cluster.get(cluster_id)
        if verdict is None:
            raise JudgmentError("Valid response has no matching blind output cluster verdict.")
        bound = dict(verdict)
        bound["logical_slot"] = slot
        result[slot] = bound
    return result


def rank_sampler_results(
    records: Iterable[Mapping[str, Any]],
    verdicts: Mapping[str, Mapping[str, Any]],
    *,
    scope: str = "tuning",
) -> list[dict[str, Any]]:
    """Rank only fully judged responses using the specified lexicographic rule."""

    aggregates: dict[str, dict[str, Any]] = {}
    for record in records:
        if str(record.get("split") or "") != scope:
            continue
        sampler = _record_sampler(record)
        entry = aggregates.setdefault(
            sampler.key,
            {
                "sampler": sampler.payload(),
                "sampler_key": sampler.key,
                "catastrophic": 0,
                "major": 0,
                "minor": 0,
                "unique_error_cases": set(),
                "naturalness_values": [],
                "latency_ms_values": [],
                "completion_tokens": [],
                "response_count": 0,
            },
        )
        slot = str(record.get("logical_slot") or "")
        verdict = verdicts.get(slot)
        if verdict is None:
            raise JudgmentError("Cannot rank sampler results with unresolved responses.")
        decision = str(verdict.get("decision") or "")
        if decision == "REVIEW_REQUIRED":
            raise JudgmentError("Cannot rank sampler results with review-required responses.")
        if decision not in _SEVERITY_ORDER:
            raise JudgmentError("Sampler verdict contains an invalid severity.")
        entry["response_count"] += 1
        if decision == "CATASTROPHIC":
            entry["catastrophic"] += 1
        elif decision == "MAJOR":
            entry["major"] += 1
        elif decision == "MINOR":
            entry["minor"] += 1
        if decision != "PASS":
            entry["unique_error_cases"].add(str(record.get("case_id") or ""))
        naturalness = verdict.get("naturalness")
        if isinstance(naturalness, int) and not isinstance(naturalness, bool):
            entry["naturalness_values"].append(naturalness)
        latency = record.get("latency_ms")
        if isinstance(latency, (int, float)) and not isinstance(latency, bool):
            entry["latency_ms_values"].append(float(latency))
        completion_tokens = record.get("completion_tokens")
        if isinstance(completion_tokens, int) and not isinstance(completion_tokens, bool):
            entry["completion_tokens"].append(completion_tokens)
    ranked: list[dict[str, Any]] = []
    for entry in aggregates.values():
        naturalness_values = entry.pop("naturalness_values")
        latency_values = entry.pop("latency_ms_values")
        completion_values = entry.pop("completion_tokens")
        entry["unique_error_cases"] = len(entry["unique_error_cases"])
        entry["naturalness_mean"] = (
            sum(naturalness_values) / len(naturalness_values) if naturalness_values else 0.0
        )
        entry["latency_ms_mean"] = (
            sum(latency_values) / len(latency_values) if latency_values else None
        )
        entry["completion_tokens_mean"] = (
            sum(completion_values) / len(completion_values) if completion_values else None
        )
        ranked.append(entry)
    return sorted(
        ranked,
        key=lambda item: (
            item["catastrophic"],
            item["major"],
            item["minor"],
            item["unique_error_cases"],
            -item["naturalness_mean"],
            float(item["latency_ms_mean"])
            if isinstance(item["latency_ms_mean"], (int, float))
            and math.isfinite(float(item["latency_ms_mean"]))
            else float("inf"),
            float(item["completion_tokens_mean"])
            if isinstance(item["completion_tokens_mean"], (int, float))
            and math.isfinite(float(item["completion_tokens_mean"]))
            else float("inf"),
            item["sampler_key"],
        ),
    )


def open_holdout_packet(
    reference: Mapping[str, Any],
    records: Iterable[Mapping[str, Any]],
    *,
    provisional_sampler: SamplerTuple,
    baseline_sampler: SamplerTuple,
    tuning_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Open only provisional-winner versus baseline holdout after tuning closes."""

    if tuning_report.get("scope") != "tuning" or tuning_report.get("status") != "PROVISIONAL_WINNER":
        raise JudgmentError("Holdout stays sealed until a tuning provisional winner is recorded.")
    if str(tuning_report.get("reference_sha256") or "") != str(reference.get("reference_sha256") or ""):
        raise JudgmentError("Holdout tuning report belongs to a different frozen reference.")
    if str(tuning_report.get("provisional_sampler_key") or "") != provisional_sampler.key:
        raise JudgmentError("Holdout candidate does not match the recorded provisional winner.")
    return build_blind_judgment_packet(
        reference,
        records,
        scope="holdout",
        allowed_sampler_keys=(provisional_sampler.key, baseline_sampler.key),
    )


def public_rank_summary(
    ranked: Sequence[Mapping[str, Any]],
    *,
    scope: str,
    reference_sha256: str,
) -> dict[str, Any]:
    """Produce a tracked-report-safe aggregate that contains no text or paths."""

    def _optional_metric(value: Any) -> float | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        numeric = float(value)
        return round(numeric, 4) if math.isfinite(numeric) else None

    return {
        "schema_version": JUDGMENT_SCHEMA_VERSION,
        "scope": scope,
        "reference_sha256": reference_sha256,
        "rows": [
            {
                "sampler": dict(item.get("sampler") or {}),
                "sampler_key": str(item.get("sampler_key") or ""),
                "catastrophic": int(item.get("catastrophic") or 0),
                "major": int(item.get("major") or 0),
                "minor": int(item.get("minor") or 0),
                "unique_error_cases": int(item.get("unique_error_cases") or 0),
                "naturalness_mean": _optional_metric(item.get("naturalness_mean")),
                "latency_ms_mean": _optional_metric(item.get("latency_ms_mean")),
                "completion_tokens_mean": _optional_metric(item.get("completion_tokens_mean")),
                "response_count": int(item.get("response_count") or 0),
            }
            for item in ranked
        ],
    }
