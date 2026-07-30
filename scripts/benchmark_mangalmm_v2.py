#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


ROOT = Path(__file__).resolve().parents[1]
EVALUATION_PROTOCOL_VERSION = "mangalmm-v2-evaluation-v1"
ANNOTATION_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SAFE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,95}$")

ALLOWED_SPLITS = frozenset(
    {"development", "holdout", "negative_control", "final"}
)
ALLOWED_ROLES = frozenset(
    {
        "dialogue_bubble",
        "dialogue_free",
        "narration",
        "ui_or_sign",
        "sfx",
        "decorative",
        "ambiguous",
    }
)
ALLOWED_ACTIONS = frozenset({"translate_inpaint", "preserve", "review"})
ALLOWED_BUBBLE_TYPES = frozenset(
    {
        "opaque",
        "black",
        "gray",
        "translucent",
        "borderless",
        "none",
        "unknown",
    }
)


class BenchmarkContractError(ValueError):
    pass


@dataclass(frozen=True)
class HistoricalAuditEntry:
    commit: str
    strategy: str
    decision: str
    reusable: tuple[str, ...]
    prohibited: tuple[str, ...]
    evidence_file: str
    required_needles: tuple[str, ...]


HISTORICAL_AUDIT: tuple[HistoricalAuditEntry, ...] = (
    HistoricalAuditEntry(
        commit="cdc92547a03ba44277d7f94fdefaf11b413f118b",
        strategy="detector-block crop requests",
        decision="failed_contract_mismatch",
        reusable=("OpenAI-compatible request envelope",),
        prohibited=("block crop OCR",),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=("resolve_block_crop_bbox", "_process_block"),
    ),
    HistoricalAuditEntry(
        commit="6f96d1b5283f24bea5cf1bd0eb26b75251f91b2f",
        strategy="overlapping page tiles and rescue macros",
        decision="failed_coordinate_and_dedup_complexity",
        reusable=(),
        prohibited=("page tiles", "overlap", "rescue macros"),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=("page_tile", "rescue_macro", "_build_rescue_units"),
    ),
    HistoricalAuditEntry(
        commit="673ceceded545798e16d0136330c9dfbd61e7d01",
        strategy="hardened tile coordinate remapping",
        decision="failed_same_architecture",
        reusable=("axis-aware coordinate diagnostics",),
        prohibited=("tile matching",),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=("response_bbox_2d", "page_tile", "rescue_macro"),
    ),
    HistoricalAuditEntry(
        commit="21272140db076531a4ece92495dc390766383126",
        strategy="single full-page spotting request",
        decision="valid_direction",
        reusable=("full-page request", "global region-to-block matching"),
        prohibited=(),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=('"page_full"', "_assign_regions_to_blocks"),
    ),
    HistoricalAuditEntry(
        commit="0f1c1bdbbec224326a4d743e020cc46332e53278",
        strategy="PNG request and strict coordinate scaling",
        decision="valid_contract_hardening",
        reusable=("PNG input", "scale_x", "scale_y", "strict payload diagnostics"),
        prohibited=(),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=('data:image/png;base64', 'cv2.imencode(".png"', "scale_x"),
    ),
    HistoricalAuditEntry(
        commit="8a6ba642274c4cbf9b360b0fcecab50c0629e667",
        strategy="adaptive Optimal+ full-page profiles",
        decision="failed_dense_capacity_reduction",
        reusable=("bounded retry telemetry",),
        prohibited=("lower dense resolution", "lower dense token capacity"),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=(
            "DEFAULT_MANGALMM_DENSE_SHORT_SIDE = 900",
            "DEFAULT_MANGALMM_DENSE_MAX_LONG_SIDE = 1270",
            "DEFAULT_MANGALMM_DENSE_PROFILE_TOKENS = 1024",
        ),
    ),
    HistoricalAuditEntry(
        commit="e31cc4e42e8238e595b2a63debf0066608f54604",
        strategy="unbounded accepted-request read timeout",
        decision="evidence_for_long_running_requests_only",
        reusable=("separate connect and read timeout",),
        prohibited=("unbounded read timeout",),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=("return (float(self.request_timeout_sec), None)",),
    ),
    HistoricalAuditEntry(
        commit="d65b3b24c5b03cc7636d92ebb86099698b46ec98",
        strategy="broad decorative glyph string stripping",
        decision="failed_role_normalization_conflation",
        reusable=("raw text preservation",),
        prohibited=("broad glyph deletion as semantic classification",),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=("normalize_decorative_ocr_text(raw_text)", "raw_text_content"),
    ),
    HistoricalAuditEntry(
        commit="ae2a90d001933382a0a87502528681e62b94f795",
        strategy="dual-resident Paddle and Manga hybrid selection",
        decision="failed_runtime_and_quality_gate",
        reusable=("stage telemetry",),
        prohibited=("dual residency", "automatic selector", "hidden Paddle fallback"),
        evidence_file="modules/ocr/selection.py",
        required_needles=(
            'resident_engines = ("PaddleOCR VL", "MangaLMM")',
            "requires_sidecar_collection = True",
        ),
    ),
    HistoricalAuditEntry(
        commit="0724b98e02bcc9c23d6d03f3123e70349df0d8e1",
        strategy="hybrid benchmark closure",
        decision="failed_closed",
        reusable=("measured failure evidence",),
        prohibited=("reopening the same selector without a new hypothesis",),
        evidence_file="docs/benchmark/workflow-split-runtime/results-history-ko.md",
        required_needles=("1664.021", "failed_closed"),
    ),
    HistoricalAuditEntry(
        commit="dade2d90034f95b796cfa84f5f8e263d4cdd64cb",
        strategy="direct one-shot manual profile",
        decision="insufficient_capacity_not_model_disqualification",
        reusable=("full-page direct mode",),
        prohibited=("one-shot 256-token product contract",),
        evidence_file="modules/ocr/mangalmm_ocr.py",
        required_needles=(
            'DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS = 256',
            'self.contract_mode = "direct_manual"',
        ),
    ),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _run_git(repo_root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise BenchmarkContractError(
            f"git {' '.join(args)} failed: {detail or completed.returncode}"
        )
    return completed.stdout


def audit_history(
    repo_root: Path = ROOT,
    *,
    git_reader: Callable[..., str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    if git_reader is None:
        git_reader = lambda root, *args: _run_git(root, *args)
    verified: list[dict[str, Any]] = []
    for entry in HISTORICAL_AUDIT:
        resolved_commit = git_reader(
            repo_root,
            "rev-parse",
            entry.commit,
        ).strip()
        if resolved_commit != entry.commit:
            raise BenchmarkContractError(
                f"Historical commit mismatch for {entry.commit}: {resolved_commit}"
            )
        content = git_reader(
            repo_root,
            "show",
            f"{entry.commit}:{entry.evidence_file}",
        )
        missing = [
            needle for needle in entry.required_needles if needle not in content
        ]
        if missing:
            raise BenchmarkContractError(
                f"Historical evidence drift for {entry.commit}: missing {missing}"
            )
        verified.append(
            {
                "commit": entry.commit,
                "strategy": entry.strategy,
                "decision": entry.decision,
                "reusable": list(entry.reusable),
                "prohibited": list(entry.prohibited),
                "evidence_file": entry.evidence_file,
                "evidence_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                "status": "verified",
            }
        )
    payload = {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "audit_entry_count": len(verified),
        "entries": verified,
    }
    payload["audit_sha256"] = canonical_json_sha256(payload)
    return payload


def _require_object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise BenchmarkContractError(f"{label} must be a JSON object.")
    return value


def _require_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise BenchmarkContractError(f"{label} must be a JSON array.")
    return value


def _require_safe_id(value: Any, label: str) -> str:
    text = str(value or "")
    if not SAFE_ID_RE.fullmatch(text):
        raise BenchmarkContractError(
            f"{label} must match {SAFE_ID_RE.pattern!r}."
        )
    return text


def _require_sha256(value: Any, label: str) -> str:
    text = str(value or "")
    if not SHA256_RE.fullmatch(text):
        raise BenchmarkContractError(f"{label} must be lowercase SHA-256.")
    return text


def _resolve_existing_file(value: Any, label: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise BenchmarkContractError(f"{label} is required.")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise BenchmarkContractError(f"{label} does not exist: {path}")
    return path


def _path_is_inside(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def require_external_path(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if _path_is_inside(resolved, ROOT):
        raise BenchmarkContractError(
            f"{label} must stay outside the Git working tree: {resolved}"
        )
    return resolved


def _validate_number_sequence(
    value: Any,
    *,
    expected_length: int | None,
    minimum_length: int,
    label: str,
) -> list[float]:
    values = _require_list(value, label)
    if expected_length is not None and len(values) != expected_length:
        raise BenchmarkContractError(
            f"{label} must contain exactly {expected_length} numbers."
        )
    if len(values) < minimum_length:
        raise BenchmarkContractError(
            f"{label} must contain at least {minimum_length} numbers."
        )
    numbers: list[float] = []
    for index, item in enumerate(values):
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise BenchmarkContractError(
                f"{label}[{index}] must be numeric."
            ) from exc
        if not math.isfinite(number):
            raise BenchmarkContractError(
                f"{label}[{index}] must be finite."
            )
        numbers.append(number)
    return numbers


def validate_annotation(
    annotation: dict[str, Any],
    *,
    case_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    if annotation.get("schema_version") != ANNOTATION_SCHEMA_VERSION:
        raise BenchmarkContractError(
            f"annotation {case_id} uses an unsupported schema version."
        )
    if annotation.get("case_id") != case_id:
        raise BenchmarkContractError(
            f"annotation case_id mismatch for {case_id}."
        )
    if annotation.get("source_sha256") != source_sha256:
        raise BenchmarkContractError(
            f"annotation source_sha256 mismatch for {case_id}."
        )

    regions = _require_list(annotation.get("regions"), f"{case_id}.regions")
    region_ids: set[str] = set()
    role_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    for index, raw_region in enumerate(regions):
        region = _require_object(raw_region, f"{case_id}.regions[{index}]")
        region_id = _require_safe_id(
            region.get("region_id"),
            f"{case_id}.regions[{index}].region_id",
        )
        if region_id in region_ids:
            raise BenchmarkContractError(
                f"Duplicate region_id {region_id!r} in {case_id}."
            )
        region_ids.add(region_id)

        bbox = region.get("bbox_xyxy")
        polygon = region.get("polygon")
        if bbox is None and polygon is None:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} requires bbox_xyxy or polygon."
            )
        if bbox is not None:
            x1, y1, x2, y2 = _validate_number_sequence(
                bbox,
                expected_length=4,
                minimum_length=4,
                label=f"{case_id}.{region_id}.bbox_xyxy",
            )
            if x2 <= x1 or y2 <= y1:
                raise BenchmarkContractError(
                    f"{case_id}.{region_id}.bbox_xyxy has invalid bounds."
                )
        if polygon is not None:
            points = _require_list(
                polygon,
                f"{case_id}.{region_id}.polygon",
            )
            if len(points) < 3:
                raise BenchmarkContractError(
                    f"{case_id}.{region_id}.polygon needs at least three points."
                )
            for point_index, point in enumerate(points):
                _validate_number_sequence(
                    point,
                    expected_length=2,
                    minimum_length=2,
                    label=f"{case_id}.{region_id}.polygon[{point_index}]",
                )

        role = str(region.get("semantic_role") or "")
        action = str(region.get("processing_action") or "")
        bubble_type = str(region.get("bubble_type") or "")
        if role not in ALLOWED_ROLES:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} has invalid semantic_role {role!r}."
            )
        if action not in ALLOWED_ACTIONS:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} has invalid processing_action {action!r}."
            )
        if bubble_type not in ALLOWED_BUBBLE_TYPES:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} has invalid bubble_type {bubble_type!r}."
            )
        if not isinstance(region.get("human_translation_expected"), bool):
            raise BenchmarkContractError(
                f"{case_id}.{region_id}.human_translation_expected must be boolean."
            )
        human_translation_expected = bool(region["human_translation_expected"])
        if not isinstance(region.get("original_text", ""), str):
            raise BenchmarkContractError(
                f"{case_id}.{region_id}.original_text must be a string."
            )
        if role in {"sfx", "decorative"} and action != "preserve":
            raise BenchmarkContractError(
                f"{case_id}.{region_id} must preserve role {role!r}."
            )
        if role == "ambiguous" and action != "review":
            raise BenchmarkContractError(
                f"{case_id}.{region_id} must route ambiguous text to review."
            )
        if action == "translate_inpaint" and not human_translation_expected:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} cannot translate text excluded by the human reference."
            )
        if action == "preserve" and human_translation_expected:
            raise BenchmarkContractError(
                f"{case_id}.{region_id} cannot preserve required meaning text."
            )
        role_counts[role] = role_counts.get(role, 0) + 1
        action_counts[action] = action_counts.get(action, 0) + 1

    return {
        "region_count": len(regions),
        "role_counts": dict(sorted(role_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
    }


def validate_evaluation_manifest(
    manifest: dict[str, Any],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    if manifest.get("protocol_version") != EVALUATION_PROTOCOL_VERSION:
        raise BenchmarkContractError("Unsupported MangaLMM v2 protocol_version.")
    corpus_id = _require_safe_id(manifest.get("corpus_id"), "corpus_id")
    cases = _require_list(manifest.get("cases"), "cases")
    if not cases:
        raise BenchmarkContractError("cases must not be empty.")

    case_ids: set[str] = set()
    source_hashes: set[str] = set()
    summaries: list[dict[str, Any]] = []
    for index, raw_case in enumerate(cases):
        case = _require_object(raw_case, f"cases[{index}]")
        case_id = _require_safe_id(case.get("case_id"), f"cases[{index}].case_id")
        if case_id in case_ids:
            raise BenchmarkContractError(f"Duplicate case_id {case_id!r}.")
        case_ids.add(case_id)

        split = str(case.get("split") or "")
        if split not in ALLOWED_SPLITS:
            raise BenchmarkContractError(
                f"{case_id}.split must be one of {sorted(ALLOWED_SPLITS)}."
            )
        if split == "holdout" and case.get("frozen_before_candidate_run") is not True:
            raise BenchmarkContractError(
                f"Holdout {case_id} must be frozen before candidate execution."
            )

        source_sha256 = _require_sha256(
            case.get("source_sha256"),
            f"{case_id}.source_sha256",
        )
        if source_sha256 in source_hashes:
            raise BenchmarkContractError(
                f"Duplicate source_sha256 {source_sha256!r} in {case_id}."
            )
        source_hashes.add(source_sha256)
        annotation_sha256 = _require_sha256(
            case.get("annotation_sha256"),
            f"{case_id}.annotation_sha256",
        )
        source_path = _resolve_existing_file(
            case.get("source_image"),
            f"{case_id}.source_image",
        )
        annotation_path = require_external_path(
            _resolve_existing_file(
                case.get("annotation"),
                f"{case_id}.annotation",
            ),
            f"{case_id}.annotation",
        )
        if verify_files and sha256_file(source_path) != source_sha256:
            raise BenchmarkContractError(
                f"{case_id}.source_image SHA-256 mismatch."
            )
        if verify_files and sha256_file(annotation_path) != annotation_sha256:
            raise BenchmarkContractError(
                f"{case_id}.annotation SHA-256 mismatch."
            )
        try:
            annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise BenchmarkContractError(
                f"{case_id}.annotation is not valid UTF-8 JSON."
            ) from exc
        annotation_summary = validate_annotation(
            _require_object(annotation, f"{case_id}.annotation"),
            case_id=case_id,
            source_sha256=source_sha256,
        )
        summaries.append(
            {
                "case_id": case_id,
                "split": split,
                "source_sha256": source_sha256,
                "annotation_sha256": annotation_sha256,
                **annotation_summary,
            }
        )

    summary = {
        "protocol_version": EVALUATION_PROTOCOL_VERSION,
        "corpus_id": corpus_id,
        "case_count": len(summaries),
        "split_counts": {
            split: sum(1 for item in summaries if item["split"] == split)
            for split in sorted(ALLOWED_SPLITS)
            if any(item["split"] == split for item in summaries)
        },
        "cases": summaries,
    }
    summary["contract_sha256"] = canonical_json_sha256(summary)
    return summary


def load_external_manifest(path: Path) -> dict[str, Any]:
    manifest_path = require_external_path(path, "Evaluation manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(
            f"Unable to read evaluation manifest: {manifest_path}"
        ) from exc
    return _require_object(payload, "manifest")


def write_external_json(path: Path, payload: Any) -> Path:
    output_path = require_external_path(path, "Benchmark output")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    return output_path


def _print_json(payload: Any) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MangaLMM v2 historical audit and external evaluation contract."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    history = subparsers.add_parser(
        "audit-history",
        help="Verify the historical commit evidence without model requests.",
    )
    history.add_argument("--output", type=Path)

    validate = subparsers.add_parser(
        "validate-manifest",
        help="Validate a frozen external evaluation manifest and its file hashes.",
    )
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--output", type=Path)
    validate.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help=(
            "Skip source/annotation SHA verification while still requiring external "
            "annotation files and valid schemas; never valid for a benchmark run."
        ),
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "audit-history":
            payload = audit_history()
        elif args.command == "validate-manifest":
            manifest = load_external_manifest(args.manifest)
            payload = validate_evaluation_manifest(
                manifest,
                verify_files=not args.skip_file_hashes,
            )
            payload["file_hashes_verified"] = not args.skip_file_hashes
        else:
            raise BenchmarkContractError(f"Unsupported command: {args.command}")
        if getattr(args, "output", None):
            write_external_json(args.output, payload)
        _print_json(payload)
        return 0
    except BenchmarkContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
