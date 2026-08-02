"""Private 758-occurrence to 478-case reference workflow for sampler v2.

Nothing in this module emits corpus text to stdout or to tracked reports.  The
caller supplies an ignored source-manifest and receives only private JSON
documents under the managed validation archive.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .protocol import (
    CORPUS_CASE_COUNT,
    CORPUS_OCCURRENCE_COUNT,
    HOLDOUT_CASE_COUNT,
    TUNING_CASE_COUNT,
    canonical_sha256,
)
from .storage import atomic_write_json, read_json


REFERENCE_SCHEMA_VERSION = "gemma-sampler-reference-v2"
SOURCE_MANIFEST_SCHEMA_VERSION = "gemma-sampler-source-manifest-v2"
BLIND_REVIEW_SCHEMA_VERSION = "gemma-sampler-blind-review-v2"


class CorpusError(RuntimeError):
    """Raised when the sealed reference workflow cannot prove its inputs."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_path(archive_root: Path, relative: str) -> Path:
    candidate = (archive_root / str(relative)).resolve()
    try:
        candidate.relative_to(archive_root.resolve())
    except ValueError as exc:
        raise CorpusError("Private source manifest path escaped the validation archive.") from exc
    if candidate.is_symlink():
        raise CorpusError("Private source snapshot cannot be a symbolic link.")
    if not candidate.is_file():
        raise CorpusError("Private source snapshot is missing.")
    return candidate


def _block_mapping(block: Any) -> Mapping[str, Any]:
    if isinstance(block, Mapping) and block.get("type") == "textblock":
        data = block.get("data")
        return data if isinstance(data, Mapping) else {}
    return block if isinstance(block, Mapping) else {}


def _raw_text(block: Any) -> str:
    data = _block_mapping(block)
    # `text` is intentionally the original OCR field.  A previous translated
    # `normalized_text` must never substitute for it in this reference corpus.
    return str(data.get("text") or "").strip()


def _prior_outputs(block: Any) -> dict[str, str]:
    data = _block_mapping(block)
    result: dict[str, str] = {}
    for key in ("translation", "normalized_translation", "translated_text"):
        value = data.get(key)
        if isinstance(value, str) and value.strip():
            result[key] = value
    return result


def _as_object(path: Path) -> Mapping[str, Any]:
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise CorpusError("Private JSON root must be an object.")
    return value


def load_source_manifest(path: Path, *, archive_root: Path) -> dict[str, Any]:
    """Load and validate a private source manifest without revealing paths."""

    try:
        manifest_relative_path = path.relative_to(archive_root).as_posix()
    except ValueError as exc:
        raise CorpusError("Private source manifest path escaped the validation archive.") from exc
    manifest_path = _archive_path(archive_root, manifest_relative_path)
    manifest = _as_object(manifest_path)
    if manifest.get("schema_version") != SOURCE_MANIFEST_SCHEMA_VERSION:
        raise CorpusError("Private source manifest schema version is not v2.")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise CorpusError("Private source manifest has no sources.")
    source_ids: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for source in sources:
        if not isinstance(source, Mapping):
            raise CorpusError("Private source manifest has an invalid source record.")
        source_id = str(source.get("source_id") or "").strip()
        if not source_id or source_id in source_ids:
            raise CorpusError("Private source manifest source ids must be unique.")
        source_ids.add(source_id)
        snapshot_path = _archive_path(archive_root, str(source.get("snapshot_path") or ""))
        expected_sha = str(source.get("sha256") or "").lower()
        actual_sha = _sha256_file(snapshot_path)
        if len(expected_sha) != 64 or expected_sha != actual_sha:
            raise CorpusError("Private source snapshot SHA-256 does not match its manifest.")
        language = str(source.get("language") or "").strip()
        if not language:
            raise CorpusError("Private source manifest language is required.")
        expected_occurrences = source.get("expected_occurrences")
        if not isinstance(expected_occurrences, int) or expected_occurrences <= 0:
            raise CorpusError("Private source manifest occurrence count is required.")
        normalized.append(
            {
                "source_id": source_id,
                "snapshot_path": snapshot_path,
                "snapshot_sha256": actual_sha,
                "language": language,
                "expected_occurrences": expected_occurrences,
            }
        )
    return {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "path_sha256": _sha256_file(manifest_path),
        "sources": normalized,
    }


@dataclass(frozen=True)
class Occurrence:
    source_id: str
    page_index: int
    block_index: int
    language: str
    source_text: str
    context_before: str
    context_after: str
    source_snapshot_sha256: str
    prior_outputs: Mapping[str, str]

    @property
    def identity_material(self) -> dict[str, str]:
        return {
            "language": self.language,
            "source_text": self.source_text,
            "context_after_text": self.context_after,
        }


def _iter_snapshot_occurrences(source: Mapping[str, Any]) -> Iterable[Occurrence]:
    payload = _as_object(Path(source["snapshot_path"]))
    pages = payload.get("pages")
    if not isinstance(pages, list):
        raise CorpusError("Private source snapshot has no pages list.")
    count = 0
    for page_index, page in enumerate(pages, start=1):
        if not isinstance(page, Mapping):
            continue
        raw_blocks = page.get("blocks")
        if not isinstance(raw_blocks, list):
            continue
        nonempty: list[tuple[int, Any, str]] = []
        for block_index, block in enumerate(raw_blocks, start=1):
            text = _raw_text(block)
            if text:
                nonempty.append((block_index, block, text))
        for index, (block_index, block, text) in enumerate(nonempty):
            before = nonempty[index - 1][2] if index > 0 else ""
            after = nonempty[index + 1][2] if index + 1 < len(nonempty) else ""
            count += 1
            yield Occurrence(
                source_id=str(source["source_id"]),
                page_index=page_index,
                block_index=block_index,
                language=str(source["language"]),
                source_text=text,
                context_before=before,
                context_after=after,
                source_snapshot_sha256=str(source["snapshot_sha256"]),
                prior_outputs=_prior_outputs(block),
            )
    if count != int(source["expected_occurrences"]):
        raise CorpusError("Private source snapshot occurrence count changed.")


def _case_id(identity: Mapping[str, str]) -> str:
    return "case-" + canonical_sha256(identity)[:20]


def _holdout_ids_from_case_ids(case_ids: Sequence[str]) -> set[str]:
    ordered = sorted(
        case_ids,
        key=lambda item: hashlib.sha256(f"sampler-v2-holdout|{item}".encode("utf-8")).hexdigest(),
    )
    return set(ordered[:HOLDOUT_CASE_COUNT])


def _assign_splits(cases: list[dict[str, Any]]) -> None:
    holdout_ids = _holdout_ids_from_case_ids([str(case["case_id"]) for case in cases])
    for case in cases:
        case["split"] = "holdout" if case["case_id"] in holdout_ids else "tuning"
    tuning = sum(case["split"] == "tuning" for case in cases)
    holdout = sum(case["split"] == "holdout" for case in cases)
    if tuning != TUNING_CASE_COUNT or holdout != HOLDOUT_CASE_COUNT:
        raise CorpusError("Frozen corpus split did not produce 382 tuning and 96 holdout cases.")


def build_reference_draft(
    source_manifest_path: Path,
    *,
    archive_root: Path,
    strict_counts: bool = True,
) -> dict[str, Any]:
    """Build an editable private reference draft from raw OCR snapshots.

    The only de-duplication identity is language + original source text + next
    non-empty adjacent original text.  Prior translations are retained in
    private provenance but are never read as canonical answers or weights.
    """

    source_manifest = load_source_manifest(source_manifest_path, archive_root=archive_root)
    grouped: dict[str, list[Occurrence]] = defaultdict(list)
    for source in source_manifest["sources"]:
        for occurrence in _iter_snapshot_occurrences(source):
            grouped[_case_id(occurrence.identity_material)].append(occurrence)
    occurrence_count = sum(len(items) for items in grouped.values())
    if strict_counts and occurrence_count != CORPUS_OCCURRENCE_COUNT:
        raise CorpusError("v2 reference requires exactly 758 private occurrences.")
    if strict_counts and len(grouped) != CORPUS_CASE_COUNT:
        raise CorpusError("v2 reference identity did not produce exactly 478 cases.")

    cases: list[dict[str, Any]] = []
    for case_id, occurrences in sorted(grouped.items()):
        first = occurrences[0]
        identity = first.identity_material
        if any(item.identity_material != identity for item in occurrences):
            raise CorpusError("Private corpus group identity is internally inconsistent.")
        cases.append(
            {
                "case_id": case_id,
                "split": "unassigned",
                "language": first.language,
                "source_text": first.source_text,
                "context_after_text": first.context_after,
                "context_before_examples": sorted(
                    {item.context_before for item in occurrences if item.context_before}
                ),
                "canonical_translation": "",
                "required_meaning": [],
                "prohibited_changes": [
                    "censorship_or_deletion",
                    "name_masking",
                    "negation_question_or_force_change",
                    "speaker_target_action_relationship_number_identity_change",
                ],
                "terminology_basis": [],
                "acceptable_alternatives": [],
                "confidence": "",
                "review_status": "PENDING_CANONICAL",
                "flags": [],
                "provenance": [
                    {
                        "source_id": item.source_id,
                        "page_index": item.page_index,
                        "block_index": item.block_index,
                        "snapshot_sha256": item.source_snapshot_sha256,
                        "prior_outputs": dict(item.prior_outputs),
                    }
                    for item in occurrences
                ],
            }
        )
    if strict_counts:
        _assign_splits(cases)
    else:
        for index, case in enumerate(cases):
            case["split"] = "holdout" if index % 5 == 0 else "tuning"
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "state": "DRAFT_CANONICAL",
        "source_manifest_sha256": source_manifest["path_sha256"],
        "case_identity": "language+source_text+context_after_text",
        "occurrence_count": occurrence_count,
        "case_count": len(cases),
        "cases": cases,
    }


def write_reference_draft(path: Path, draft: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(draft))


def _reference_cases(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    if reference.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise CorpusError("Private reference schema version is not v2.")
    cases = reference.get("cases")
    if not isinstance(cases, list) or not cases:
        raise CorpusError("Private reference has no cases.")
    result = [dict(case) for case in cases if isinstance(case, Mapping)]
    if len(result) != len(cases):
        raise CorpusError("Private reference has an invalid case record.")
    if len({str(case.get("case_id") or "") for case in result}) != len(result):
        raise CorpusError("Private reference case ids must be unique.")
    return result


def apply_canonical_answers(
    reference: Mapping[str, Any],
    answers: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the agent's first-pass canonical answers without copying old output."""

    result = dict(reference)
    cases = _reference_cases(reference)
    expected_ids = {str(case["case_id"]) for case in cases}
    if set(answers) != expected_ids:
        raise CorpusError("Canonical answers must cover every private reference case exactly once.")
    updated: list[dict[str, Any]] = []
    for case in cases:
        answer = answers[str(case["case_id"])]
        if not isinstance(answer, Mapping):
            raise CorpusError("Each canonical answer must be an object.")
        canonical = str(answer.get("canonical_translation") or "").strip()
        required = answer.get("required_meaning")
        forbidden = answer.get("prohibited_changes")
        confidence = str(answer.get("confidence") or "").lower()
        if not canonical or not isinstance(required, list) or not required:
            raise CorpusError("Each canonical answer needs translation and required meaning.")
        if not isinstance(forbidden, list) or not forbidden:
            raise CorpusError("Each canonical answer needs prohibited-change criteria.")
        if confidence not in {"high", "medium", "low"}:
            raise CorpusError("Canonical confidence must be high, medium, or low.")
        required_values = [str(item) for item in required if str(item).strip()]
        forbidden_values = [str(item) for item in forbidden if str(item).strip()]
        if not required_values or not forbidden_values:
            raise CorpusError("Canonical meaning and prohibited-change criteria cannot be empty.")
        updated_case = dict(case)
        updated_case.update(
            {
                "canonical_translation": canonical,
                "required_meaning": required_values,
                "prohibited_changes": forbidden_values,
                "terminology_basis": [
                    str(item) for item in answer.get("terminology_basis", []) if str(item).strip()
                ],
                "acceptable_alternatives": [
                    str(item)
                    for item in answer.get("acceptable_alternatives", [])
                    if str(item).strip()
                ],
                "confidence": confidence,
                "review_status": "PENDING_BLIND_REVIEW",
                "flags": list(case.get("flags") or []),
            }
        )
        updated.append(updated_case)
    result["state"] = "PENDING_BLIND_REVIEW"
    result["cases"] = updated
    return result


def build_blind_review_packet(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Create an order-hidden second pass without canonical or old outputs."""

    cases = _reference_cases(reference)
    if any(case.get("review_status") != "PENDING_BLIND_REVIEW" for case in cases):
        raise CorpusError("Blind review starts only after every canonical answer is present.")
    rows: list[dict[str, Any]] = []
    for case in cases:
        blind_id = "blind-" + hashlib.sha256(
            f"sampler-v2-blind|{case['case_id']}".encode("utf-8")
        ).hexdigest()[:20]
        rows.append(
            {
                "blind_id": blind_id,
                "language": case["language"],
                "source_text": case["source_text"],
                "context_after_text": case["context_after_text"],
                "context_before_examples": list(case.get("context_before_examples") or []),
                "independent_translation": "",
                "required_meaning": [],
                "confidence": "",
                "ocr_damaged": False,
            }
        )
    rows.sort(key=lambda row: hashlib.sha256(str(row["blind_id"]).encode("utf-8")).hexdigest())
    return {
        "schema_version": BLIND_REVIEW_SCHEMA_VERSION,
        "reference_case_count": len(rows),
        "rows": rows,
    }


def apply_blind_review(
    reference: Mapping[str, Any],
    blind_packet: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply independent answers and flag every disagreement or weak source."""

    if blind_packet.get("schema_version") != BLIND_REVIEW_SCHEMA_VERSION:
        raise CorpusError("Blind review packet schema version is not v2.")
    rows = blind_packet.get("rows")
    if not isinstance(rows, list):
        raise CorpusError("Blind review packet rows are missing.")
    by_blind = {str(row.get("blind_id") or ""): row for row in rows if isinstance(row, Mapping)}
    if len(by_blind) != len(rows) or "" in by_blind:
        raise CorpusError("Blind review packet has duplicate or empty hidden identities.")
    if set(decisions) != set(by_blind):
        raise CorpusError("Blind review decisions must cover each hidden row exactly once.")
    cases = _reference_cases(reference)
    case_by_blind = {
        "blind-" + hashlib.sha256(f"sampler-v2-blind|{case['case_id']}".encode("utf-8")).hexdigest()[:20]: case
        for case in cases
    }
    updated: list[dict[str, Any]] = []
    for blind_id, row in by_blind.items():
        case = case_by_blind.get(blind_id)
        if case is None:
            raise CorpusError("Blind review packet does not match the private reference.")
        decision = decisions[blind_id]
        independent = str(decision.get("independent_translation") or "").strip()
        confidence = str(decision.get("confidence") or "").lower()
        ocr_damaged = bool(decision.get("ocr_damaged", False))
        if not independent or confidence not in {"high", "medium", "low"}:
            raise CorpusError("Each blind review needs a translation and confidence.")
        flags = list(case.get("flags") or [])
        if independent != str(case.get("canonical_translation") or ""):
            flags.append("BLIND_TRANSLATION_DIFFERENCE")
        if confidence == "low" or str(case.get("confidence") or "") == "low":
            flags.append("LOW_CONFIDENCE")
        if ocr_damaged:
            flags.append("OCR_DAMAGED")
        updated_case = dict(case)
        updated_case["blind_review"] = {
            "independent_translation": independent,
            "required_meaning": [
                str(item) for item in decision.get("required_meaning", []) if str(item).strip()
            ],
            "confidence": confidence,
            "ocr_damaged": ocr_damaged,
        }
        updated_case["flags"] = sorted(set(flags))
        updated_case["review_status"] = "FLAGGED" if flags else "PENDING_USER_SAMPLE"
        updated.append(updated_case)
    result = dict(reference)
    result["state"] = "PENDING_RESOLUTION"
    result["cases"] = sorted(updated, key=lambda case: str(case["case_id"]))
    return result


def flagged_cases(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        case
        for case in _reference_cases(reference)
        if list(case.get("flags") or []) or case.get("review_status") == "FLAGGED"
    ]


def build_resolution_packet(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Private packet for the user-visible disagreement/low-confidence gate."""

    rows = []
    for case in flagged_cases(reference):
        rows.append(
            {
                "case_id": case["case_id"],
                "source_text": case["source_text"],
                "context_after_text": case["context_after_text"],
                "canonical_translation": case["canonical_translation"],
                "blind_review": dict(case.get("blind_review") or {}),
                "flags": list(case.get("flags") or []),
                "resolution": "",
            }
        )
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": "flag-resolution",
        "flagged_case_count": len(rows),
        "rows": rows,
    }


def apply_resolutions(
    reference: Mapping[str, Any],
    resolutions: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Resolve all flags explicitly; no automatic equivalence inference occurs."""

    cases = _reference_cases(reference)
    expected = {str(case["case_id"]) for case in cases if case.get("flags")}
    if set(resolutions) != expected:
        raise CorpusError("Every flagged reference case needs an explicit resolution.")
    updated: list[dict[str, Any]] = []
    for case in cases:
        case_id = str(case["case_id"])
        updated_case = dict(case)
        if case_id in resolutions:
            resolution = resolutions[case_id]
            decision = str(resolution.get("decision") or "").upper()
            if decision not in {"APPROVE_CANONICAL", "REPLACE_CANONICAL", "EXCLUDE"}:
                raise CorpusError("Reference resolution decision is invalid.")
            if decision == "EXCLUDE":
                raise CorpusError("The v2 corpus must have zero unresolved cases; exclusion is not allowed.")
            if decision == "REPLACE_CANONICAL":
                replacement = str(resolution.get("canonical_translation") or "").strip()
                if not replacement:
                    raise CorpusError("Canonical replacement is empty.")
                updated_case["canonical_translation"] = replacement
            updated_case["flags"] = []
            updated_case["review_status"] = "PENDING_USER_SAMPLE"
            updated_case["resolution_note"] = str(resolution.get("note") or "").strip()
        updated.append(updated_case)
    result = dict(reference)
    result["state"] = "PENDING_USER_SAMPLE"
    result["cases"] = updated
    return result


def select_user_sample(reference: Mapping[str, Any], *, sample_size: int = 24) -> dict[str, Any]:
    """Create deterministic, stratified non-flagged cases for user review."""

    cases = _reference_cases(reference)
    candidates = [
        case
        for case in cases
        if not case.get("flags") and case.get("review_status") == "PENDING_USER_SAMPLE"
    ]
    if len(candidates) < sample_size:
        raise CorpusError("Not enough resolved non-flagged cases for user sampling.")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in candidates:
        length = len(str(case.get("source_text") or ""))
        length_bucket = "short" if length <= 12 else "medium" if length <= 40 else "long"
        buckets[(str(case.get("language") or ""), length_bucket)].append(case)
    for bucket_cases in buckets.values():
        bucket_cases.sort(
            key=lambda case: hashlib.sha256(str(case["case_id"]).encode("utf-8")).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    ordered_buckets = sorted(buckets)
    positions = {key: 0 for key in ordered_buckets}
    cursor = 0
    while len(selected) < sample_size:
        available = [key for key in ordered_buckets if positions[key] < len(buckets[key])]
        if not available:
            raise CorpusError("Unable to construct the deterministic user sample.")
        key = available[cursor % len(available)]
        choices = buckets[key]
        candidate = choices[positions[key]]
        positions[key] += 1
        selected.append(candidate)
        cursor += 1
    rows = [
        {
            "case_id": case["case_id"],
            "language": case["language"],
            "source_text": case["source_text"],
            "context_after_text": case["context_after_text"],
            "canonical_translation": case["canonical_translation"],
            "required_meaning": list(case.get("required_meaning") or []),
            "prohibited_changes": list(case.get("prohibited_changes") or []),
            "decision": "",
            "note": "",
        }
        for case in selected
    ]
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "kind": "user-stratified-sample",
        "sample_size": sample_size,
        "sample_sha256": canonical_sha256([row["case_id"] for row in rows]),
        "rows": rows,
    }


def freeze_reference(
    reference: Mapping[str, Any],
    *,
    user_sample: Mapping[str, Any],
    user_approval: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal only a fully-resolved, explicitly user-approved v2 reference."""

    cases = _reference_cases(reference)
    if len(cases) != CORPUS_CASE_COUNT:
        raise CorpusError("v2 reference cannot freeze before it has exactly 478 cases.")
    if any(case.get("flags") for case in cases):
        raise CorpusError("v2 reference cannot freeze with unresolved flagged cases.")
    if any(case.get("review_status") != "PENDING_USER_SAMPLE" for case in cases):
        raise CorpusError("v2 reference contains unresolved review states.")
    if any(
        not str(case.get("canonical_translation") or "").strip()
        or not list(case.get("required_meaning") or [])
        for case in cases
    ):
        raise CorpusError("v2 reference contains incomplete canonical answers.")
    if user_sample.get("schema_version") != REFERENCE_SCHEMA_VERSION:
        raise CorpusError("User sample schema version is not v2.")
    if user_sample.get("kind") != "user-stratified-sample":
        raise CorpusError("Reference freeze requires the generated stratified user sample.")
    rows = user_sample.get("rows")
    if not isinstance(rows, list) or len(rows) != 24 or not all(isinstance(row, Mapping) for row in rows):
        raise CorpusError("v2 reference requires the 24-case user sample.")
    expected_sample = select_user_sample(reference)
    expected_rows = expected_sample["rows"]
    expected_case_ids = [str(row["case_id"]) for row in expected_rows]
    supplied_case_ids = [str(row.get("case_id") or "") for row in rows]
    expected_sample_sha = str(expected_sample["sample_sha256"])
    if user_sample.get("sample_size") != len(expected_rows):
        raise CorpusError("User sample size changed before reference freeze.")
    if supplied_case_ids != expected_case_ids:
        raise CorpusError("User sample cases or order changed before reference freeze.")
    if str(user_sample.get("sample_sha256") or "") != expected_sample_sha:
        raise CorpusError("User sample identity changed before reference freeze.")
    # The user must inspect the generated source/context/canonical fields, not
    # a packet with the same case ids but altered content.  Only the explicit
    # decision and free-form note are mutable after sample generation.
    immutable_fields = (
        "case_id",
        "language",
        "source_text",
        "context_after_text",
        "canonical_translation",
        "required_meaning",
        "prohibited_changes",
    )
    for supplied, expected in zip(rows, expected_rows, strict=True):
        if any(supplied.get(field) != expected.get(field) for field in immutable_fields):
            raise CorpusError("User sample content changed before reference freeze.")
    if any(str(row.get("decision") or "").upper() != "PASS" for row in rows):
        raise CorpusError("Every user sample row must explicitly pass before freeze.")
    if not bool(user_approval.get("approved")):
        raise CorpusError("Reference freeze requires explicit user approval.")
    if str(user_approval.get("sample_sha256") or "") != expected_sample_sha:
        raise CorpusError("User approval does not bind the selected sample.")
    frozen_cases = []
    for case in cases:
        finalized = dict(case)
        finalized["review_status"] = "APPROVED"
        frozen_cases.append(finalized)
    result = dict(reference)
    result.update(
        {
            "state": "FROZEN",
            "cases": frozen_cases,
            "user_sample_sha256": expected_sample_sha,
            "reference_sha256": canonical_sha256(
                {
                    "schema_version": REFERENCE_SCHEMA_VERSION,
                    "case_identity": result.get("case_identity"),
                    "cases": frozen_cases,
                }
            ),
        }
    )
    return result


def reference_summary(reference: Mapping[str, Any]) -> dict[str, Any]:
    """Return public-safe counts and hashes only; never corpus content or paths."""

    cases = _reference_cases(reference)
    split_counts = {
        "tuning": sum(case.get("split") == "tuning" for case in cases),
        "holdout": sum(case.get("split") == "holdout" for case in cases),
    }
    return {
        "schema_version": REFERENCE_SCHEMA_VERSION,
        "state": str(reference.get("state") or ""),
        "occurrence_count": int(reference.get("occurrence_count") or 0),
        "case_count": len(cases),
        "split_counts": split_counts,
        "flagged_case_count": len(flagged_cases(reference)),
        "reference_sha256": str(reference.get("reference_sha256") or ""),
    }
