#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EVALUATION_PROTOCOL_VERSION = "mangalmm-v2-evaluation-v1"
ANNOTATION_SCHEMA_VERSION = 1
AUDIT_SCHEMA_VERSION = 1
DETECTOR_SNAPSHOT_SCHEMA_VERSION = 1
PROFILE_RUN_SCHEMA_VERSION = 1
PROFILE_RUN_PROTOCOL_VERSION = "mangalmm-v2-fullpage-profile-v1"
DEFAULT_MAX_IDLE_GPU_USED_MIB = 2048.0
OFFICIAL_MANGA_OCR_PROMPT = (
    "Please perform OCR on this image and output the recognized Japanese "
    "text along with its position (grounding)."
)
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
class FullPageProfile:
    profile_id: str
    short_side: int
    long_side: int
    max_pixels: int
    context_size: int
    max_completion_tokens: int = 4096

    def capacity_key(self) -> tuple[int, int, int, int]:
        return (
            int(self.short_side),
            int(self.long_side),
            int(self.max_pixels),
            int(self.context_size),
        )


STANDARD_FULL_PAGE_PROFILE = FullPageProfile(
    profile_id="standard-1728-8192",
    short_side=1224,
    long_side=1728,
    max_pixels=2_116_800,
    context_size=8192,
)
HIGH_FULL_PAGE_PROFILE = FullPageProfile(
    profile_id="high-2048-12288",
    short_side=1451,
    long_side=2048,
    max_pixels=2_971_648,
    context_size=12288,
)
FULL_PAGE_PROFILES: tuple[FullPageProfile, ...] = (
    STANDARD_FULL_PAGE_PROFILE,
    HIGH_FULL_PAGE_PROFILE,
)
PROFILE_ROUND_ORDER: tuple[tuple[str, ...], ...] = (
    (
        STANDARD_FULL_PAGE_PROFILE.profile_id,
        HIGH_FULL_PAGE_PROFILE.profile_id,
    ),
    (
        HIGH_FULL_PAGE_PROFILE.profile_id,
        STANDARD_FULL_PAGE_PROFILE.profile_id,
    ),
)


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


def validate_profile_matrix(
    profiles: Iterable[FullPageProfile] = FULL_PAGE_PROFILES,
) -> dict[str, Any]:
    values = tuple(profiles)
    if len(values) != 2:
        raise BenchmarkContractError(
            "The locked MangaLMM profile matrix must contain exactly two profiles."
        )
    ids = [profile.profile_id for profile in values]
    if len(ids) != len(set(ids)):
        raise BenchmarkContractError("MangaLMM profile IDs must be unique.")
    for profile in values:
        _require_safe_id(profile.profile_id, "profile_id")
        if profile.short_side <= 0 or profile.long_side < profile.short_side:
            raise BenchmarkContractError(
                f"Invalid image capacity for profile {profile.profile_id}."
            )
        if profile.max_pixels < profile.short_side * profile.long_side:
            raise BenchmarkContractError(
                f"{profile.profile_id}.max_pixels may not reduce its declared "
                "full-page dimensions."
            )
        if profile.context_size < profile.max_completion_tokens:
            raise BenchmarkContractError(
                f"{profile.profile_id}.context_size is smaller than completion capacity."
            )

    standard, high = values
    for label, standard_value, high_value in (
        ("short_side", standard.short_side, high.short_side),
        ("long_side", standard.long_side, high.long_side),
        ("max_pixels", standard.max_pixels, high.max_pixels),
        ("context_size", standard.context_size, high.context_size),
        (
            "max_completion_tokens",
            standard.max_completion_tokens,
            high.max_completion_tokens,
        ),
    ):
        if high_value < standard_value:
            raise BenchmarkContractError(
                f"Dense/high profile capacity regression: {label} "
                f"{high_value} < {standard_value}."
            )
    if tuple(ids) != PROFILE_ROUND_ORDER[0]:
        raise BenchmarkContractError(
            "The first round must use the locked standard-then-high order."
        )
    if tuple(reversed(ids)) != PROFILE_ROUND_ORDER[1]:
        raise BenchmarkContractError(
            "The second round must reverse the profile order."
        )
    payload = {
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "official_prompt": OFFICIAL_MANGA_OCR_PROMPT,
        "official_prompt_sha256": hashlib.sha256(
            OFFICIAL_MANGA_OCR_PROMPT.encode("utf-8")
        ).hexdigest(),
        "profiles": [
            {
                "profile_id": profile.profile_id,
                "short_side": profile.short_side,
                "long_side": profile.long_side,
                "max_pixels": profile.max_pixels,
                "context_size": profile.context_size,
                "max_completion_tokens": profile.max_completion_tokens,
            }
            for profile in values
        ],
        "round_order": [list(round_order) for round_order in PROFILE_ROUND_ORDER],
    }
    payload["contract_sha256"] = canonical_json_sha256(payload)
    return payload


def profile_by_id(profile_id: str) -> FullPageProfile:
    for profile in FULL_PAGE_PROFILES:
        if profile.profile_id == profile_id:
            return profile
    raise BenchmarkContractError(f"Unknown MangaLMM profile: {profile_id!r}")


def _validated_bbox(
    value: Any,
    *,
    label: str,
    image_shape_hw: tuple[int, int] | None = None,
) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    elif isinstance(value, tuple):
        value = list(value)
    numbers = _validate_number_sequence(
        value,
        expected_length=4,
        minimum_length=4,
        label=label,
    )
    coords = [int(round(number)) for number in numbers]
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise BenchmarkContractError(f"{label} has invalid bounds.")
    if image_shape_hw is not None:
        image_h, image_w = image_shape_hw
        if x1 < 0 or y1 < 0 or x2 > image_w or y2 > image_h:
            raise BenchmarkContractError(
                f"{label} lies outside image bounds {image_w}x{image_h}."
            )
    return coords


def validate_detector_snapshot(
    snapshot: Mapping[str, Any],
    *,
    case_id: str,
    source_sha256: str,
) -> dict[str, Any]:
    if snapshot.get("schema_version") != DETECTOR_SNAPSHOT_SCHEMA_VERSION:
        raise BenchmarkContractError(
            f"detector snapshot {case_id} uses an unsupported schema version."
        )
    if snapshot.get("case_id") != case_id:
        raise BenchmarkContractError(
            f"detector snapshot case_id mismatch for {case_id}."
        )
    if snapshot.get("source_sha256") != source_sha256:
        raise BenchmarkContractError(
            f"detector snapshot source_sha256 mismatch for {case_id}."
        )
    decoded_sha256 = _require_sha256(
        snapshot.get("source_decoded_sha256"),
        f"{case_id}.source_decoded_sha256",
    )
    detector_fingerprint = _require_sha256(
        snapshot.get("detector_fingerprint"),
        f"{case_id}.detector_fingerprint",
    )
    detector_identity = _require_object(
        snapshot.get("detector_identity"),
        f"{case_id}.detector_identity",
    )
    if not detector_identity:
        raise BenchmarkContractError(
            f"{case_id}.detector_identity must not be empty."
        )
    shape_values = _validate_number_sequence(
        snapshot.get("source_shape_hw"),
        expected_length=2,
        minimum_length=2,
        label=f"{case_id}.source_shape_hw",
    )
    image_shape_hw = tuple(int(round(value)) for value in shape_values)
    if any(value <= 0 for value in image_shape_hw):
        raise BenchmarkContractError(
            f"{case_id}.source_shape_hw must be positive."
        )

    blocks = _require_list(snapshot.get("blocks"), f"{case_id}.blocks")
    block_ids: set[str] = set()
    for index, raw_block in enumerate(blocks):
        block = _require_object(raw_block, f"{case_id}.blocks[{index}]")
        block_id = _require_safe_id(
            block.get("block_id"),
            f"{case_id}.blocks[{index}].block_id",
        )
        if block_id in block_ids:
            raise BenchmarkContractError(
                f"Duplicate detector block_id {block_id!r} in {case_id}."
            )
        block_ids.add(block_id)
        _validated_bbox(
            block.get("text_bbox_xyxy"),
            label=f"{case_id}.{block_id}.text_bbox_xyxy",
            image_shape_hw=image_shape_hw,
        )
        if block.get("bubble_bbox_xyxy") is not None:
            _validated_bbox(
                block.get("bubble_bbox_xyxy"),
                label=f"{case_id}.{block_id}.bubble_bbox_xyxy",
                image_shape_hw=image_shape_hw,
            )
        if not isinstance(block.get("text_class", ""), str):
            raise BenchmarkContractError(
                f"{case_id}.{block_id}.text_class must be a string."
            )
    return {
        "block_count": len(blocks),
        "source_shape_hw": list(image_shape_hw),
        "source_decoded_sha256": decoded_sha256,
        "detector_fingerprint": detector_fingerprint,
        "detector_identity_sha256": canonical_json_sha256(detector_identity),
    }


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
        detector_summary: dict[str, Any] = {}
        detector_snapshot_value = case.get("detector_snapshot")
        detector_snapshot_sha_value = case.get("detector_snapshot_sha256")
        if (
            detector_snapshot_value is not None
            or detector_snapshot_sha_value is not None
        ):
            detector_snapshot_sha256 = _require_sha256(
                detector_snapshot_sha_value,
                f"{case_id}.detector_snapshot_sha256",
            )
            detector_snapshot_path = require_external_path(
                _resolve_existing_file(
                    detector_snapshot_value,
                    f"{case_id}.detector_snapshot",
                ),
                f"{case_id}.detector_snapshot",
            )
            if (
                verify_files
                and sha256_file(detector_snapshot_path)
                != detector_snapshot_sha256
            ):
                raise BenchmarkContractError(
                    f"{case_id}.detector_snapshot SHA-256 mismatch."
                )
            try:
                detector_snapshot = json.loads(
                    detector_snapshot_path.read_text(encoding="utf-8")
                )
            except (OSError, json.JSONDecodeError) as exc:
                raise BenchmarkContractError(
                    f"{case_id}.detector_snapshot is not valid UTF-8 JSON."
                ) from exc
            detector_summary = validate_detector_snapshot(
                _require_object(
                    detector_snapshot,
                    f"{case_id}.detector_snapshot",
                ),
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
                **(
                    {"detector_snapshot": detector_summary}
                    if detector_summary
                    else {}
                ),
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


def validate_profile_run_manifest(
    manifest: dict[str, Any],
    *,
    verify_files: bool = True,
) -> dict[str, Any]:
    summary = validate_evaluation_manifest(
        manifest,
        verify_files=verify_files,
    )
    missing = [
        str(case.get("case_id") or f"cases[{index}]")
        for index, case in enumerate(_require_list(manifest.get("cases"), "cases"))
        if not isinstance(case, dict)
        or case.get("detector_snapshot") is None
        or case.get("detector_snapshot_sha256") is None
    ]
    if missing:
        raise BenchmarkContractError(
            "Profile runs require a frozen detector snapshot for every case: "
            + ", ".join(missing)
        )
    profile_contract = validate_profile_matrix()
    return {
        **summary,
        "profile_contract_sha256": profile_contract["contract_sha256"],
        "profile_run_ready": True,
    }


def load_external_manifest(path: Path) -> dict[str, Any]:
    manifest_path = require_external_path(path, "Evaluation manifest")
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(
            f"Unable to read evaluation manifest: {manifest_path}"
        ) from exc
    return _require_object(payload, "manifest")


def build_profile_execution_plan(
    manifest: Mapping[str, Any],
    *,
    split: str = "development",
    case_ids: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    if split not in ALLOWED_SPLITS:
        raise BenchmarkContractError(
            f"split must be one of {sorted(ALLOWED_SPLITS)}."
        )
    requested_ids = {
        _require_safe_id(case_id, "case_id")
        for case_id in (case_ids or [])
    }
    cases = [
        dict(case)
        for case in _require_list(manifest.get("cases"), "cases")
        if isinstance(case, dict)
        and case.get("split") == split
        and (
            not requested_ids
            or str(case.get("case_id") or "") in requested_ids
        )
    ]
    found_ids = {str(case.get("case_id") or "") for case in cases}
    missing = sorted(requested_ids - found_ids)
    if missing:
        raise BenchmarkContractError(
            f"Requested cases are absent from split {split!r}: {missing}"
        )
    if not cases:
        raise BenchmarkContractError(
            f"No MangaLMM cases are available for split {split!r}."
        )
    cases.sort(key=lambda case: str(case.get("case_id") or ""))
    plan: list[dict[str, Any]] = []
    for round_index, profile_ids in enumerate(PROFILE_ROUND_ORDER, start=1):
        for profile_id in profile_ids:
            plan.append(
                {
                    "round": round_index,
                    "profile_id": profile_id,
                    "case_ids": [
                        str(case["case_id"])
                        for case in cases
                    ],
                }
            )
    return plan


def _region_bbox(region: Mapping[str, Any]) -> list[float] | None:
    raw_bbox = region.get("bbox_xyxy")
    if raw_bbox is not None:
        try:
            bbox = [float(value) for value in raw_bbox]
        except (TypeError, ValueError):
            return None
        return bbox if len(bbox) == 4 else None
    polygon = region.get("polygon")
    if not isinstance(polygon, list) or len(polygon) < 3:
        return None
    try:
        xs = [float(point[0]) for point in polygon]
        ys = [float(point[1]) for point in polygon]
    except (TypeError, ValueError, IndexError):
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _intersection_area_float(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0,
        min(a[3], b[3]) - max(a[1], b[1]),
    )


def _bbox_area_float(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _normalize_ocr_text(value: Any) -> str:
    return re.sub(
        r"\s+",
        "",
        unicodedata.normalize("NFC", str(value or "")).strip(),
    )


def evaluate_profile_regions(
    *,
    annotation: Mapping[str, Any],
    predicted_regions: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    expected = [
        dict(region)
        for region in _require_list(annotation.get("regions"), "annotation.regions")
        if isinstance(region, dict)
    ]
    predicted = [dict(region) for region in predicted_regions]
    pairs: list[tuple[float, float, float, int, int]] = []
    for expected_index, expected_region in enumerate(expected):
        expected_box = _region_bbox(expected_region)
        if expected_box is None:
            continue
        expected_area = max(1.0, _bbox_area_float(expected_box))
        for predicted_index, predicted_region in enumerate(predicted):
            predicted_box = _region_bbox(predicted_region)
            if predicted_box is None:
                continue
            predicted_area = max(1.0, _bbox_area_float(predicted_box))
            intersection = _intersection_area_float(
                expected_box,
                predicted_box,
            )
            if intersection <= 0:
                continue
            union = expected_area + predicted_area - intersection
            iou = intersection / max(1.0, union)
            expected_coverage = intersection / expected_area
            predicted_coverage = intersection / predicted_area
            if iou < 0.05 and expected_coverage < 0.20:
                continue
            score = max(iou, expected_coverage * 0.80)
            pairs.append(
                (
                    score,
                    expected_coverage,
                    predicted_coverage,
                    expected_index,
                    predicted_index,
                )
            )
    pairs.sort(reverse=True)
    matched_expected: set[int] = set()
    matched_predicted: set[int] = set()
    matches: list[dict[str, Any]] = []
    for score, expected_coverage, predicted_coverage, expected_index, predicted_index in pairs:
        if (
            expected_index in matched_expected
            or predicted_index in matched_predicted
        ):
            continue
        matched_expected.add(expected_index)
        matched_predicted.add(predicted_index)
        expected_region = expected[expected_index]
        predicted_region = predicted[predicted_index]
        expected_text = str(expected_region.get("original_text", "") or "")
        predicted_text = str(
            predicted_region.get("text")
            or predicted_region.get("text_content")
            or ""
        )
        matches.append(
            {
                "region_id": str(expected_region.get("region_id") or ""),
                "semantic_role": str(
                    expected_region.get("semantic_role") or ""
                ),
                "processing_action": str(
                    expected_region.get("processing_action") or ""
                ),
                "predicted_region_index": predicted_index,
                "overlap_score": round(score, 6),
                "expected_coverage": round(expected_coverage, 6),
                "predicted_coverage": round(predicted_coverage, 6),
                "expected_text": expected_text,
                "predicted_text": predicted_text,
                "normalized_text_exact": (
                    bool(_normalize_ocr_text(expected_text))
                    and _normalize_ocr_text(expected_text)
                    == _normalize_ocr_text(predicted_text)
                ),
            }
        )
    unmatched_expected = [
        {
            "region_id": str(region.get("region_id") or ""),
            "semantic_role": str(region.get("semantic_role") or ""),
            "processing_action": str(region.get("processing_action") or ""),
            "original_text": str(region.get("original_text") or ""),
        }
        for index, region in enumerate(expected)
        if index not in matched_expected
    ]
    unmatched_predicted = [
        {
            "predicted_region_index": index,
            "bbox_xyxy": _region_bbox(region),
            "text": str(
                region.get("text")
                or region.get("text_content")
                or ""
            ),
        }
        for index, region in enumerate(predicted)
        if index not in matched_predicted
    ]
    predicted_keys: dict[tuple[Any, ...], int] = {}
    for region in predicted:
        bbox = _region_bbox(region)
        key = (
            tuple(round(value, 3) for value in bbox)
            if bbox is not None
            else (),
            _normalize_ocr_text(
                region.get("text")
                or region.get("text_content")
                or ""
            ),
        )
        predicted_keys[key] = predicted_keys.get(key, 0) + 1
    exact_duplicate_count = sum(
        count - 1
        for count in predicted_keys.values()
        if count > 1
    )
    required_expected_ids = {
        str(region.get("region_id") or "")
        for region in expected
        if region.get("processing_action") == "translate_inpaint"
    }
    matched_required_ids = {
        str(match.get("region_id") or "")
        for match in matches
        if str(match.get("region_id") or "") in required_expected_ids
    }
    return {
        "expected_region_count": len(expected),
        "predicted_region_count": len(predicted),
        "matched_region_count": len(matches),
        "required_region_count": len(required_expected_ids),
        "matched_required_region_count": len(matched_required_ids),
        "coverage_gap": bool(required_expected_ids - matched_required_ids),
        "exact_predicted_duplicate_count": exact_duplicate_count,
        "matches": matches,
        "unmatched_expected": unmatched_expected,
        "unmatched_predicted": unmatched_predicted,
    }


def detect_repetition(text: str) -> bool:
    raw = str(text or "")
    text_values: list[str] = []
    for match in re.finditer(
        r'"text_content"\s*:\s*("(?:\\.|[^"\\])*")',
        raw,
        flags=re.DOTALL,
    ):
        try:
            value = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        normalized_value = _normalize_ocr_text(value)
        if normalized_value:
            text_values.append(normalized_value)
    value_counts: dict[str, int] = {}
    for value in text_values:
        value_counts[value] = value_counts.get(value, 0) + 1
        if len(value) >= 4 and value_counts[value] >= 8:
            return True
        if len(value) >= 128:
            for width in (2, 3, 4, 6, 8):
                chunks = [
                    value[index : index + width]
                    for index in range(0, len(value) - width + 1, width)
                ]
                if max(Counter(chunks).values(), default=0) >= 12:
                    return True

    normalized = re.sub(r"\s+", "", raw)
    if text_values or len(normalized) < 512:
        return False
    for width in (8, 12, 16):
        chunks = [
            normalized[index : index + width]
            for index in range(0, len(normalized) - width + 1, width)
        ]
        if max(Counter(chunks).values(), default=0) >= 20:
            return True
    return False


def profile_escalation_reasons(
    *,
    request_metadata: Mapping[str, Any],
    evaluation: Mapping[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if str(request_metadata.get("finish_reason") or "").lower() == "length":
        reasons.append("finish_reason_length")
    if str(request_metadata.get("parser_error_code") or ""):
        reasons.append("parser_error")
    if detect_repetition(str(request_metadata.get("raw_response") or "")):
        reasons.append("repetition")
    if evaluation.get("coverage_gap") is True:
        reasons.append("coverage_gap")
    return sorted(set(reasons))


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


@contextlib.contextmanager
def temporary_environment(
    updates: Mapping[str, str],
) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def query_gpu_snapshot() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi.exe") or shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "reason": "nvidia-smi unavailable"}
    completed = subprocess.run(
        [
            executable,
            "--query-gpu=memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=15,
    )
    if completed.returncode != 0 or not completed.stdout.strip():
        return {
            "available": False,
            "reason": completed.stderr.strip() or "nvidia-smi failed",
        }
    values = [
        item.strip()
        for item in completed.stdout.splitlines()[0].split(",")
    ]
    if len(values) < 4:
        return {
            "available": False,
            "reason": "unexpected nvidia-smi output",
        }
    try:
        total, used, free = (float(values[index]) for index in range(3))
        utilization = float(values[3])
    except ValueError:
        return {
            "available": False,
            "reason": "invalid nvidia-smi numeric output",
        }
    return {
        "available": True,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "utilization_percent": utilization,
    }


def require_idle_gpu(
    snapshot: Mapping[str, Any],
    *,
    max_used_mib: float,
) -> None:
    if snapshot.get("available") is not True:
        raise BenchmarkContractError(
            "GPU inventory is unavailable: "
            + str(snapshot.get("reason") or "unknown reason")
        )
    used = float(snapshot.get("memory_used_mib", math.inf))
    if used > float(max_used_mib):
        raise BenchmarkContractError(
            "GPU preflight refused the live MangaLMM run: "
            f"{used:.0f} MiB > {float(max_used_mib):.0f} MiB."
        )


@dataclass
class _UIStub:
    value_mappings: dict[str, str]

    def tr(self, text: str) -> str:
        return text


class _BenchmarkSettings:
    def __init__(
        self,
        profile: FullPageProfile,
        *,
        request_timeout_sec: int = 300,
    ) -> None:
        self.profile = profile
        self.request_timeout_sec = int(request_timeout_sec)
        self.ui = _UIStub(value_mappings={})

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "detector":
            return "RT-DETR-v2"
        if tool_type == "ocr":
            return "MangaLMM"
        raise KeyError(tool_type)

    def is_gpu_enabled(self) -> bool:
        return True

    def get_credentials(self, _provider_name: str) -> dict[str, Any]:
        return {}

    def get_mangalmm_ocr_settings(self) -> dict[str, Any]:
        return {
            "server_url": "http://127.0.0.1:28081/v1",
            "max_completion_tokens": self.profile.max_completion_tokens,
            "parallel_workers": 1,
            "request_timeout_sec": self.request_timeout_sec,
            "raw_response_logging": False,
            "safe_resize": True,
            "max_pixels": self.profile.max_pixels,
            "max_long_side": self.profile.long_side,
        }


def _build_profile_engine(
    profile: FullPageProfile,
    settings: _BenchmarkSettings,
):
    from modules.ocr.mangalmm_ocr import MangaLMMOCREngine, ResizePlan

    class _ProfileEngine(MangaLMMOCREngine):
        STANDARD_PROMPT = OFFICIAL_MANGA_OCR_PROMPT
        DENSE_PROMPT = OFFICIAL_MANGA_OCR_PROMPT

        def _build_resize_plan(
            self,
            image_shape,
            *,
            profile: str,
            block_count: int,
            small_block_ratio: float,
            text_cover_ratio: float,
            max_completion_tokens_override: int | None = None,
        ):
            image_h, image_w = image_shape[:2]
            short_side = float(min(image_w, image_h))
            long_side = float(max(image_w, image_h))
            page_area = float(max(1, image_w * image_h))
            base_scale = min(
                1.0,
                float(self._benchmark_profile.short_side)
                / short_side
                if short_side > 0
                else 1.0,
                float(self._benchmark_profile.long_side)
                / long_side
                if long_side > 0
                else 1.0,
                math.sqrt(
                    float(self._benchmark_profile.max_pixels) / page_area
                ),
            )
            request_w = max(1, int(round(image_w * base_scale)))
            request_h = max(1, int(round(image_h * base_scale)))
            request_tokens = (
                int(max_completion_tokens_override)
                if max_completion_tokens_override is not None
                else int(self._benchmark_profile.max_completion_tokens)
            )
            return ResizePlan(
                profile=profile,
                original_shape=(int(image_h), int(image_w)),
                request_shape=(int(request_h), int(request_w)),
                base_scale=float(base_scale),
                scale_x=request_w / float(max(1, image_w)),
                scale_y=request_h / float(max(1, image_h)),
                max_completion_tokens=request_tokens,
                block_count=int(block_count),
                small_block_ratio=float(small_block_ratio),
                text_cover_ratio=float(text_cover_ratio),
            )

    engine = _ProfileEngine()
    engine._benchmark_profile = profile
    engine.initialize(
        settings,
        source_lang_english="Japanese",
        selected_ocr_mode="MangaLMM",
    )
    return engine


def _load_json_file(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BenchmarkContractError(f"Unable to read {label}: {path}") from exc
    return _require_object(payload, label)


def _load_profile_case(case: Mapping[str, Any]) -> dict[str, Any]:
    import imkit as imk

    from app.projects.stage_checkpoints import decoded_image_sha256
    from modules.utils.ocr_debug import ensure_three_channel
    from modules.utils.textblock import TextBlock

    case_id = str(case["case_id"])
    source_path = _resolve_existing_file(
        case.get("source_image"),
        f"{case_id}.source_image",
    )
    image = imk.read_image(str(source_path))
    if image is None:
        raise BenchmarkContractError(
            f"Unable to decode source image for {case_id}: {source_path}"
        )
    image = ensure_three_channel(image)
    annotation_path = require_external_path(
        _resolve_existing_file(
            case.get("annotation"),
            f"{case_id}.annotation",
        ),
        f"{case_id}.annotation",
    )
    snapshot_path = require_external_path(
        _resolve_existing_file(
            case.get("detector_snapshot"),
            f"{case_id}.detector_snapshot",
        ),
        f"{case_id}.detector_snapshot",
    )
    annotation = _load_json_file(
        annotation_path,
        f"{case_id}.annotation",
    )
    snapshot = _load_json_file(
        snapshot_path,
        f"{case_id}.detector_snapshot",
    )
    expected_shape = [
        int(value)
        for value in snapshot.get("source_shape_hw", [])
    ]
    if expected_shape != [int(image.shape[0]), int(image.shape[1])]:
        raise BenchmarkContractError(
            f"{case_id}.detector_snapshot image shape drift."
        )
    if decoded_image_sha256(image) != snapshot.get(
        "source_decoded_sha256"
    ):
        raise BenchmarkContractError(
            f"{case_id}.detector_snapshot decoded image SHA-256 drift."
        )
    blocks = []
    for raw_block in snapshot.get("blocks", []):
        text_bbox = raw_block["text_bbox_xyxy"]
        bubble_bbox = raw_block.get("bubble_bbox_xyxy")
        blocks.append(
            TextBlock(
                text_bbox=text_bbox,
                bubble_bbox=bubble_bbox,
                text_class=str(raw_block.get("text_class") or ""),
                block_id=str(raw_block["block_id"]),
            )
        )
    return {
        "case_id": case_id,
        "source_path": source_path,
        "image": image,
        "annotation": annotation,
        "snapshot": snapshot,
        "blocks": blocks,
    }


def _serialize_profile_block(block: Any) -> dict[str, Any]:
    def box(value: Any) -> list[int] | None:
        if value is None:
            return None
        try:
            values = [int(round(float(item))) for item in value]
        except (TypeError, ValueError):
            return None
        return values if len(values) == 4 else None

    return {
        "block_id": str(getattr(block, "block_id", "") or ""),
        "text_bbox_xyxy": box(getattr(block, "xyxy", None)),
        "bubble_bbox_xyxy": box(getattr(block, "bubble_xyxy", None)),
        "text_class": str(getattr(block, "text_class", "") or ""),
        "text": str(getattr(block, "text", "") or ""),
        "ocr_status": str(getattr(block, "ocr_status", "") or ""),
        "ocr_empty_reason": str(
            getattr(block, "ocr_empty_reason", "") or ""
        ),
        "ocr_regions": list(getattr(block, "ocr_regions", []) or []),
        "merge_split_diagnostics": dict(
            getattr(block, "merge_split_diagnostics", {}) or {}
        ),
        "semantic_role": str(
            getattr(block, "semantic_role", "") or ""
        ),
        "processing_action": str(
            getattr(block, "processing_action", "") or ""
        ),
    }


def _write_profile_overlay(
    *,
    image: Any,
    blocks: Iterable[Any],
    regions: Iterable[Mapping[str, Any]],
    output_path: Path,
) -> None:
    import numpy as np
    from PIL import Image, ImageDraw

    canvas = Image.fromarray(np.asarray(image).astype(np.uint8))
    draw = ImageDraw.Draw(canvas)
    for block in blocks:
        text_box = getattr(block, "xyxy", None)
        bubble_box = getattr(block, "bubble_xyxy", None)
        if bubble_box is not None:
            draw.rectangle(
                [int(value) for value in bubble_box],
                outline=(0, 128, 255),
                width=3,
            )
        if text_box is not None:
            draw.rectangle(
                [int(value) for value in text_box],
                outline=(40, 220, 40),
                width=3,
            )
    for region in regions:
        bbox = _region_bbox(region)
        if bbox is not None:
            draw.rectangle(
                [int(round(value)) for value in bbox],
                outline=(255, 64, 64),
                width=3,
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, format="PNG")


def _prepare_output_directory(path: Path) -> Path:
    output_dir = require_external_path(path, "Profile output directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BenchmarkContractError(
            f"Profile output directory must be empty: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _profile_result_without_raw(
    metadata: Mapping[str, Any],
    *,
    raw_response_file: str,
    raw_response_sha256: str,
) -> dict[str, Any]:
    result = dict(metadata)
    result.pop("raw_response", None)
    result["raw_response_file"] = raw_response_file
    result["raw_response_sha256"] = raw_response_sha256
    return result


def _run_profile_unit(
    *,
    profile: FullPageProfile,
    round_index: int,
    cases: list[Mapping[str, Any]],
    output_dir: Path,
    timeout_sec: int,
) -> dict[str, Any]:
    from modules.ocr.local_runtime import LocalOCRRuntimeManager

    settings = _BenchmarkSettings(
        profile,
        request_timeout_sec=timeout_sec,
    )
    manager = LocalOCRRuntimeManager()
    progress_events: list[dict[str, Any]] = []
    runtime_started_at = time.perf_counter()
    case_results: list[dict[str, Any]] = []
    runtime_contract: dict[str, Any] = {}
    failure = ""
    with temporary_environment(
        {"MANGALMM_LLAMA_CTX_SIZE": str(profile.context_size)}
    ):
        try:
            manager.ensure_engine(
                "MangaLMM",
                settings,
                timeout_sec=timeout_sec,
                progress_callback=lambda event: progress_events.append(
                    dict(event)
                ),
            )
            startup_seconds = time.perf_counter() - runtime_started_at
            contract = manager._mangalmm_runtime_contract()
            runtime_contract = {
                "fingerprint": contract.fingerprint,
                "command_sha256": contract.command_sha256,
                "ready_manifest_sha256": contract.ready_manifest_sha256,
                "llama_image_ref": contract.llama_image_ref,
                "llama_image_id": contract.llama_image_id,
                "model_volume": contract.volume_name,
                "runtime_options": dict(contract.runtime_options),
            }
            engine = _build_profile_engine(profile, settings)
            for case in cases:
                loaded = _load_profile_case(case)
                case_id = str(loaded["case_id"])
                blocks = loaded["blocks"]
                request_started_at = time.perf_counter()
                engine.process_image(loaded["image"], blocks)
                request_seconds = time.perf_counter() - request_started_at
                evaluation = evaluate_profile_regions(
                    annotation=loaded["annotation"],
                    predicted_regions=engine.last_page_regions,
                )
                reasons = profile_escalation_reasons(
                    request_metadata=engine.last_request_metadata,
                    evaluation=evaluation,
                )
                artifact_dir = (
                    output_dir
                    / "_internal"
                    / f"round-{round_index}"
                    / profile.profile_id
                    / case_id
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                raw_response = str(
                    engine.last_request_metadata.get(
                        "raw_response",
                        "",
                    )
                    or ""
                )
                raw_path = artifact_dir / "raw-response.txt"
                raw_path.write_text(raw_response, encoding="utf-8")
                raw_sha = sha256_file(raw_path)
                overlay_path = artifact_dir / "mapped-overlay.png"
                _write_profile_overlay(
                    image=loaded["image"],
                    blocks=blocks,
                    regions=engine.last_page_regions,
                    output_path=overlay_path,
                )
                overlay_sha = sha256_file(overlay_path)
                metadata = _profile_result_without_raw(
                    engine.last_request_metadata,
                    raw_response_file=str(
                        raw_path.relative_to(output_dir)
                    ),
                    raw_response_sha256=raw_sha,
                )
                case_result = {
                    "round": round_index,
                    "profile_id": profile.profile_id,
                    "case_id": case_id,
                    "request_seconds": round(request_seconds, 6),
                    "request": metadata,
                    "attempts": list(engine.last_attempt_history),
                    "regions": list(engine.last_page_regions),
                    "blocks": [
                        _serialize_profile_block(block)
                        for block in blocks
                    ],
                    "shadow_regions": list(engine.last_shadow_regions),
                    "merge_split_diagnostics": list(
                        engine.last_merge_split_diagnostics
                    ),
                    "evaluation": evaluation,
                    "higher_profile_eligible_reasons": reasons,
                    "overlay_file": str(
                        overlay_path.relative_to(output_dir)
                    ),
                    "overlay_sha256": overlay_sha,
                }
                case_result["result_sha256"] = canonical_json_sha256(
                    case_result
                )
                write_external_json(
                    artifact_dir / "result.json",
                    case_result,
                )
                case_results.append(case_result)
        except Exception as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                manager.shutdown()
            except Exception as shutdown_exc:
                if not failure:
                    raise BenchmarkContractError(
                        "MangaLMM runtime stop failed: "
                        f"{type(shutdown_exc).__name__}: {shutdown_exc}"
                    ) from shutdown_exc
    return {
        "round": round_index,
        "profile_id": profile.profile_id,
        "startup_seconds": round(
            locals().get("startup_seconds", 0.0),
            6,
        ),
        "runtime_contract": runtime_contract,
        "progress_events": progress_events,
        "cases": case_results,
        "post_stop_gpu": query_gpu_snapshot(),
    }


def build_blind_profile_review(
    *,
    run_units: Iterable[Mapping[str, Any]],
    output_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    profile_ids = [profile.profile_id for profile in FULL_PAGE_PROFILES]
    if secrets.randbelow(2):
        profile_ids.reverse()
    label_by_profile = {
        profile_id: chr(ord("A") + index)
        for index, profile_id in enumerate(profile_ids)
    }
    rows: list[dict[str, Any]] = []
    for unit in run_units:
        profile_id = str(unit["profile_id"])
        label = label_by_profile[profile_id]
        for case in unit.get("cases", []):
            case_id = str(case["case_id"])
            round_index = int(case["round"])
            source_overlay = output_dir / str(case["overlay_file"])
            blind_overlay = (
                output_dir
                / "blind"
                / f"round-{round_index}"
                / case_id
                / f"{label}-overlay.png"
            )
            blind_overlay.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_overlay, blind_overlay)
            blind_overlay_sha = sha256_file(blind_overlay)
            rows.append(
                {
                    "round": round_index,
                    "case_id": case_id,
                    "candidate": label,
                    "regions": list(case.get("regions", [])),
                    "blocks": list(case.get("blocks", [])),
                    "evaluation": dict(case.get("evaluation", {})),
                    "overlay_file": str(
                        blind_overlay.relative_to(output_dir)
                    ),
                    "overlay_sha256": blind_overlay_sha,
                    "review": {
                        "meaning_text_recall": "",
                        "merge_split": "",
                        "coordinate_quality": "",
                        "ocr_text_quality": "",
                        "notes": "",
                    },
                }
            )
    rows.sort(
        key=lambda row: (
            int(row["round"]),
            str(row["case_id"]),
            str(row["candidate"]),
        )
    )
    blind_payload = {
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "candidate_labels": sorted(label_by_profile.values()),
        "rows": rows,
        "unblind_forbidden_until_review_complete": True,
    }
    blind_payload["review_sha256"] = canonical_json_sha256(blind_payload)
    key_payload = {
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "profile_by_candidate": {
            label: profile
            for profile, label in sorted(label_by_profile.items())
        },
        "review_sha256": blind_payload["review_sha256"],
    }
    key_payload["key_sha256"] = canonical_json_sha256(key_payload)
    return blind_payload, key_payload


def run_fullpage_profiles(
    *,
    manifest_path: Path,
    output_dir: Path,
    split: str,
    case_ids: Iterable[str],
    timeout_sec: int,
    max_idle_gpu_used_mib: float,
) -> dict[str, Any]:
    manifest = load_external_manifest(manifest_path)
    manifest_summary = validate_profile_run_manifest(
        manifest,
        verify_files=True,
    )
    execution_plan = build_profile_execution_plan(
        manifest,
        split=split,
        case_ids=case_ids,
    )
    selected_ids = set(execution_plan[0]["case_ids"])
    case_by_id = {
        str(case["case_id"]): case
        for case in manifest["cases"]
        if str(case.get("case_id") or "") in selected_ids
    }
    destination = _prepare_output_directory(output_dir)
    initial_gpu = query_gpu_snapshot()
    require_idle_gpu(
        initial_gpu,
        max_used_mib=max_idle_gpu_used_mib,
    )
    run_units: list[dict[str, Any]] = []
    for unit in execution_plan:
        profile = profile_by_id(str(unit["profile_id"]))
        run_units.append(
            _run_profile_unit(
                profile=profile,
                round_index=int(unit["round"]),
                cases=[
                    case_by_id[case_id]
                    for case_id in unit["case_ids"]
                ],
                output_dir=destination,
                timeout_sec=timeout_sec,
            )
        )
    blind_payload, key_payload = build_blind_profile_review(
        run_units=run_units,
        output_dir=destination,
    )
    write_external_json(destination / "blind-review.json", blind_payload)
    write_external_json(destination / "unblind-key.json", key_payload)
    payload = {
        "schema_version": PROFILE_RUN_SCHEMA_VERSION,
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "manifest_contract_sha256": manifest_summary["contract_sha256"],
        "profile_contract": validate_profile_matrix(),
        "split": split,
        "case_ids": sorted(selected_ids),
        "initial_gpu": initial_gpu,
        "max_idle_gpu_used_mib": float(max_idle_gpu_used_mib),
        "run_units": run_units,
        "blind_review_sha256": blind_payload["review_sha256"],
        "unblind_key_sha256": key_payload["key_sha256"],
    }
    payload["result_sha256"] = canonical_json_sha256(payload)
    write_external_json(destination / "profile-results.json", payload)
    return payload


def _resolve_result_artifact(
    *,
    result_root: Path,
    relative_value: Any,
    label: str,
) -> Path:
    relative_text = str(relative_value or "").replace("\\", "/")
    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise BenchmarkContractError(
            f"{label} must be a safe relative artifact path."
        )
    resolved = (result_root / relative).resolve()
    if not _path_is_inside(resolved, result_root):
        raise BenchmarkContractError(
            f"{label} escapes the result directory."
        )
    if not resolved.is_file():
        raise BenchmarkContractError(f"{label} is missing: {resolved}")
    return resolved


def validate_profile_results_file(path: Path) -> dict[str, Any]:
    result_path = require_external_path(path, "Profile results")
    payload = _load_json_file(result_path, "profile results")
    if payload.get("schema_version") != PROFILE_RUN_SCHEMA_VERSION:
        raise BenchmarkContractError(
            "Unsupported MangaLMM profile result schema."
        )
    if payload.get("protocol_version") != PROFILE_RUN_PROTOCOL_VERSION:
        raise BenchmarkContractError(
            "Unsupported MangaLMM profile result protocol."
        )
    expected_result_sha = _require_sha256(
        payload.get("result_sha256"),
        "result_sha256",
    )
    unsigned_payload = dict(payload)
    unsigned_payload.pop("result_sha256", None)
    if canonical_json_sha256(unsigned_payload) != expected_result_sha:
        raise BenchmarkContractError(
            "Profile result SHA-256 mismatch."
        )
    if payload.get("profile_contract") != validate_profile_matrix():
        raise BenchmarkContractError(
            "Profile result matrix does not match the locked contract."
        )
    _require_sha256(
        payload.get("manifest_contract_sha256"),
        "manifest_contract_sha256",
    )
    run_units = _require_list(payload.get("run_units"), "run_units")
    expected_units = [
        (round_index, profile_id)
        for round_index, profile_ids in enumerate(
            PROFILE_ROUND_ORDER,
            start=1,
        )
        for profile_id in profile_ids
    ]
    actual_units = [
        (
            int(_require_object(unit, f"run_units[{index}]").get("round", 0)),
            str(unit.get("profile_id") or ""),
        )
        for index, unit in enumerate(run_units)
    ]
    if actual_units != expected_units:
        raise BenchmarkContractError(
            "Profile result AB/BA unit order is invalid."
        )
    expected_case_ids = sorted(
        _require_safe_id(case_id, "case_ids")
        for case_id in _require_list(payload.get("case_ids"), "case_ids")
    )
    if not expected_case_ids or len(set(expected_case_ids)) != len(
        expected_case_ids
    ):
        raise BenchmarkContractError(
            "Profile result case_ids must be non-empty and unique."
        )

    result_root = result_path.parent.resolve()
    verified_raw = 0
    verified_overlays = 0
    verified_cases = 0
    for unit_index, raw_unit in enumerate(run_units):
        unit = _require_object(raw_unit, f"run_units[{unit_index}]")
        unit_cases = _require_list(
            unit.get("cases"),
            f"run_units[{unit_index}].cases",
        )
        unit_case_ids = [
            str(
                _require_object(
                    raw_case,
                    f"run_units[{unit_index}].cases[{case_index}]",
                ).get("case_id")
                or ""
            )
            for case_index, raw_case in enumerate(unit_cases)
        ]
        if unit_case_ids != expected_case_ids:
            raise BenchmarkContractError(
                "Profile result case order or coverage is invalid."
            )
        for case_index, raw_case in enumerate(unit_cases):
            case = _require_object(
                raw_case,
                f"run_units[{unit_index}].cases[{case_index}]",
            )
            expected_case_sha = _require_sha256(
                case.get("result_sha256"),
                f"run_units[{unit_index}].cases[{case_index}].result_sha256",
            )
            unsigned_case = dict(case)
            unsigned_case.pop("result_sha256", None)
            if canonical_json_sha256(unsigned_case) != expected_case_sha:
                raise BenchmarkContractError(
                    f"Case result SHA-256 mismatch: {case.get('case_id')}"
                )
            request = _require_object(
                case.get("request"),
                f"{case.get('case_id')}.request",
            )
            raw_path = _resolve_result_artifact(
                result_root=result_root,
                relative_value=request.get("raw_response_file"),
                label=f"{case.get('case_id')}.raw_response_file",
            )
            expected_raw_sha = _require_sha256(
                request.get("raw_response_sha256"),
                f"{case.get('case_id')}.raw_response_sha256",
            )
            if sha256_file(raw_path) != expected_raw_sha:
                raise BenchmarkContractError(
                    f"Raw response SHA-256 mismatch: {case.get('case_id')}"
                )
            overlay_path = _resolve_result_artifact(
                result_root=result_root,
                relative_value=case.get("overlay_file"),
                label=f"{case.get('case_id')}.overlay_file",
            )
            expected_overlay_sha = _require_sha256(
                case.get("overlay_sha256"),
                f"{case.get('case_id')}.overlay_sha256",
            )
            if sha256_file(overlay_path) != expected_overlay_sha:
                raise BenchmarkContractError(
                    f"Overlay SHA-256 mismatch: {case.get('case_id')}"
                )
            verified_cases += 1
            verified_raw += 1
            verified_overlays += 1

    blind_path = _resolve_result_artifact(
        result_root=result_root,
        relative_value="blind-review.json",
        label="blind-review.json",
    )
    blind_payload = _load_json_file(blind_path, "blind review")
    blind_sha = _require_sha256(
        blind_payload.get("review_sha256"),
        "blind-review.review_sha256",
    )
    unsigned_blind = dict(blind_payload)
    unsigned_blind.pop("review_sha256", None)
    if canonical_json_sha256(unsigned_blind) != blind_sha:
        raise BenchmarkContractError("Blind review SHA-256 mismatch.")
    if blind_sha != _require_sha256(
        payload.get("blind_review_sha256"),
        "blind_review_sha256",
    ):
        raise BenchmarkContractError(
            "Blind review does not match the profile result."
        )
    blind_rows = _require_list(blind_payload.get("rows"), "blind-review.rows")
    candidate_labels = [
        str(value)
        for value in _require_list(
            blind_payload.get("candidate_labels"),
            "blind-review.candidate_labels",
        )
    ]
    if candidate_labels != ["A", "B"]:
        raise BenchmarkContractError(
            "Blind review candidate labels are invalid."
        )
    expected_blind_rows = len(expected_units) * len(expected_case_ids)
    if len(blind_rows) != expected_blind_rows:
        raise BenchmarkContractError(
            "Blind review row coverage is invalid."
        )
    verified_blind_overlays = 0
    actual_blind_row_keys: list[tuple[int, str, str]] = []
    for row_index, raw_row in enumerate(blind_rows):
        row = _require_object(
            raw_row,
            f"blind-review.rows[{row_index}]",
        )
        forbidden_keys = {
            "profile_id",
            "profile",
            "request_seconds",
            "startup_seconds",
            "runtime_contract",
            "unblind_key",
        }
        leaked_keys = sorted(forbidden_keys.intersection(row))
        if leaked_keys:
            raise BenchmarkContractError(
                "Blind review leaks hidden candidate metadata: "
                + ", ".join(leaked_keys)
            )
        row_key = (
            int(row.get("round", 0) or 0),
            _require_safe_id(
                row.get("case_id"),
                f"blind-review.rows[{row_index}].case_id",
            ),
            str(row.get("candidate") or ""),
        )
        actual_blind_row_keys.append(row_key)
        blind_overlay_path = _resolve_result_artifact(
            result_root=result_root,
            relative_value=row.get("overlay_file"),
            label=f"blind-review.rows[{row_index}].overlay_file",
        )
        expected_blind_overlay_sha = _require_sha256(
            row.get("overlay_sha256"),
            f"blind-review.rows[{row_index}].overlay_sha256",
        )
        if sha256_file(blind_overlay_path) != expected_blind_overlay_sha:
            raise BenchmarkContractError(
                "Blind overlay SHA-256 mismatch: "
                f"row {row_index}"
            )
        verified_blind_overlays += 1
    expected_blind_row_keys = sorted(
        (
            round_index,
            case_id,
            candidate,
        )
        for round_index in range(1, len(PROFILE_ROUND_ORDER) + 1)
        for case_id in expected_case_ids
        for candidate in candidate_labels
    )
    if actual_blind_row_keys != expected_blind_row_keys:
        raise BenchmarkContractError(
            "Blind review candidate mapping or row order is invalid."
        )

    key_path = _resolve_result_artifact(
        result_root=result_root,
        relative_value="unblind-key.json",
        label="unblind-key.json",
    )
    key_payload = _load_json_file(key_path, "unblind key")
    key_sha = _require_sha256(
        key_payload.get("key_sha256"),
        "unblind-key.key_sha256",
    )
    unsigned_key = dict(key_payload)
    unsigned_key.pop("key_sha256", None)
    if canonical_json_sha256(unsigned_key) != key_sha:
        raise BenchmarkContractError("Unblind key SHA-256 mismatch.")
    if key_sha != _require_sha256(
        payload.get("unblind_key_sha256"),
        "unblind_key_sha256",
    ):
        raise BenchmarkContractError(
            "Unblind key does not match the profile result."
        )
    if key_payload.get("review_sha256") != blind_sha:
        raise BenchmarkContractError(
            "Unblind key references a different blind review."
        )
    profile_by_candidate = _require_object(
        key_payload.get("profile_by_candidate"),
        "unblind-key.profile_by_candidate",
    )
    if set(profile_by_candidate) != set(candidate_labels) or set(
        str(value) for value in profile_by_candidate.values()
    ) != {profile.profile_id for profile in FULL_PAGE_PROFILES}:
        raise BenchmarkContractError(
            "Unblind key is not a complete candidate/profile bijection."
        )
    return {
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "result_sha256": expected_result_sha,
        "run_unit_count": len(run_units),
        "case_result_count": verified_cases,
        "raw_response_count": verified_raw,
        "overlay_count": verified_overlays,
        "blind_overlay_count": verified_blind_overlays,
        "status": "verified",
        "payload": payload,
    }


def summarize_profile_results(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {
        profile.profile_id: {
            "profile_id": profile.profile_id,
            "request_seconds": [],
            "startup_seconds": [],
            "round_request_seconds": {},
            "required_total": 0,
            "required_matched": 0,
            "required_normalized_exact": 0,
            "required_exact_available": True,
            "parser_error_count": 0,
            "length_finish_count": 0,
            "exact_predicted_duplicate_count": 0,
            "coverage_gap_runs": [],
            "case_runs": [],
        }
        for profile in FULL_PAGE_PROFILES
    }
    for raw_unit in _require_list(payload.get("run_units"), "run_units"):
        unit = _require_object(raw_unit, "run_unit")
        profile_id = str(unit.get("profile_id") or "")
        if profile_id not in aggregate:
            raise BenchmarkContractError(
                f"Unexpected result profile: {profile_id}"
            )
        target = aggregate[profile_id]
        target["startup_seconds"].append(
            float(unit.get("startup_seconds", 0.0) or 0.0)
        )
        round_index = int(unit.get("round", 0) or 0)
        round_total = 0.0
        for raw_case in _require_list(unit.get("cases"), "unit.cases"):
            case = _require_object(raw_case, "case_result")
            request_seconds = float(
                case.get("request_seconds", 0.0) or 0.0
            )
            round_total += request_seconds
            target["request_seconds"].append(request_seconds)
            evaluation = _require_object(
                case.get("evaluation"),
                "case.evaluation",
            )
            required_count = int(
                evaluation.get("required_region_count", 0) or 0
            )
            matched_count = int(
                evaluation.get("matched_required_region_count", 0) or 0
            )
            matches = [
                match
                for match in evaluation.get("matches", [])
                if isinstance(match, dict)
            ]
            exact_available = all(
                "processing_action" in match
                for match in matches
            )
            exact_required = (
                sum(
                    1
                    for match in matches
                    if match.get("processing_action")
                    == "translate_inpaint"
                    and match.get("normalized_text_exact") is True
                )
                if exact_available
                else 0
            )
            request = _require_object(case.get("request"), "case.request")
            parser_error = bool(
                str(request.get("parser_error_code") or "")
            )
            length_finish = (
                str(request.get("finish_reason") or "").lower()
                == "length"
            )
            target["required_total"] += required_count
            target["required_matched"] += matched_count
            target["required_normalized_exact"] += min(
                exact_required,
                matched_count,
            )
            target["required_exact_available"] = bool(
                target["required_exact_available"]
                and exact_available
            )
            target["parser_error_count"] += int(parser_error)
            target["length_finish_count"] += int(length_finish)
            target["exact_predicted_duplicate_count"] += int(
                evaluation.get(
                    "exact_predicted_duplicate_count",
                    0,
                )
                or 0
            )
            if matched_count != required_count:
                target["coverage_gap_runs"].append(
                    {
                        "round": round_index,
                        "case_id": str(case.get("case_id") or ""),
                        "matched": matched_count,
                        "required": required_count,
                    }
                )
            target["case_runs"].append(
                {
                    "round": round_index,
                    "case_id": str(case.get("case_id") or ""),
                    "request_seconds": round(request_seconds, 6),
                    "matched_required": matched_count,
                    "required": required_count,
                    "normalized_exact_required": (
                        min(exact_required, matched_count)
                        if exact_available
                        else None
                    ),
                    "parser_error": parser_error,
                    "finish_reason": str(
                        request.get("finish_reason") or ""
                    ),
                }
            )
        target["round_request_seconds"][str(round_index)] = round(
            round_total,
            6,
        )

    summaries: list[dict[str, Any]] = []
    for profile_id, target in aggregate.items():
        request_values = list(target.pop("request_seconds"))
        startup_values = list(target.pop("startup_seconds"))
        eligible = (
            target["required_total"] > 0
            and target["required_matched"] == target["required_total"]
            and target["parser_error_count"] == 0
            and target["length_finish_count"] == 0
        )
        summaries.append(
            {
                **target,
                "median_request_seconds": round(
                    statistics.median(request_values),
                    6,
                ),
                "total_request_seconds": round(
                    sum(request_values),
                    6,
                ),
                "median_startup_seconds": round(
                    statistics.median(startup_values),
                    6,
                ),
                "required_recall": round(
                    target["required_matched"]
                    / max(1, target["required_total"]),
                    6,
                ),
                "automatic_quality_eligible": eligible,
                "manual_meaning_review_required": (
                    eligible
                    and (
                        not target["required_exact_available"]
                        or target["required_normalized_exact"]
                        != target["required_total"]
                    )
                ),
            }
        )
    summaries.sort(key=lambda item: str(item["profile_id"]))
    eligible_profiles = [
        item["profile_id"]
        for item in summaries
        if item["automatic_quality_eligible"]
    ]
    recall_leader = max(
        summaries,
        key=lambda item: (
            float(item["required_recall"]),
            -float(item["total_request_seconds"]),
        ),
    )
    decision = (
        "manual_review_required"
        if eligible_profiles
        else "no_profile_passed_required_coverage"
    )
    result = {
        "protocol_version": PROFILE_RUN_PROTOCOL_VERSION,
        "source_result_sha256": str(payload.get("result_sha256") or ""),
        "decision": decision,
        "eligible_profiles": eligible_profiles,
        "development_recall_leader_without_promotion": recall_leader[
            "profile_id"
        ],
        "profiles": summaries,
    }
    result["decision_sha256"] = canonical_json_sha256(result)
    return result


def freeze_detector_snapshot(
    *,
    source_image: Path,
    case_id: str,
    output: Path,
    source_lang_english: str,
) -> dict[str, Any]:
    import imkit as imk

    from app.projects.stage_checkpoints import (
        build_detection_fingerprint,
        build_detection_identity,
        decoded_image_sha256,
    )
    from modules.detection.processor import TextBlockDetector
    from modules.utils.ocr_debug import ensure_three_channel

    safe_case_id = _require_safe_id(case_id, "case_id")
    source_path = source_image.expanduser().resolve()
    if not source_path.is_file():
        raise BenchmarkContractError(
            f"source_image does not exist: {source_path}"
        )
    image = imk.read_image(str(source_path))
    if image is None:
        raise BenchmarkContractError(
            f"Unable to decode source_image: {source_path}"
        )
    image = ensure_three_channel(image)
    settings = _BenchmarkSettings(STANDARD_FULL_PAGE_PROFILE)
    identity = build_detection_identity(
        settings,
        source_lang_english=source_lang_english,
    )
    if identity is None:
        raise BenchmarkContractError(
            "The selected detector does not provide a frozen identity."
        )
    decoded_sha = decoded_image_sha256(image)
    detector = TextBlockDetector(settings)
    blocks = detector.detect(image) or []
    block_payloads: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        text_bbox = _validated_bbox(
            getattr(block, "xyxy", None),
            label=f"{safe_case_id}.block-{index:04d}.text_bbox_xyxy",
            image_shape_hw=(int(image.shape[0]), int(image.shape[1])),
        )
        bubble_value = getattr(block, "bubble_xyxy", None)
        bubble_bbox = (
            _validated_bbox(
                bubble_value,
                label=(
                    f"{safe_case_id}.block-{index:04d}.bubble_bbox_xyxy"
                ),
                image_shape_hw=(int(image.shape[0]), int(image.shape[1])),
            )
            if bubble_value is not None
            else None
        )
        block_payloads.append(
            {
                "block_id": f"block-{index:04d}",
                "text_bbox_xyxy": text_bbox,
                "bubble_bbox_xyxy": bubble_bbox,
                "text_class": str(
                    getattr(block, "text_class", "") or ""
                ),
            }
        )
    payload = {
        "schema_version": DETECTOR_SNAPSHOT_SCHEMA_VERSION,
        "case_id": safe_case_id,
        "source_sha256": sha256_file(source_path),
        "source_decoded_sha256": decoded_sha,
        "source_shape_hw": [int(image.shape[0]), int(image.shape[1])],
        "detector_identity": identity,
        "detector_fingerprint": build_detection_fingerprint(
            source_sha256=decoded_sha,
            identity=identity,
        ),
        "blocks": block_payloads,
    }
    payload["snapshot_sha256"] = canonical_json_sha256(payload)
    write_external_json(output, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "MangaLMM v2 historical audit, frozen detector contract, and "
            "managed full-page profile benchmark."
        )
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

    freeze = subparsers.add_parser(
        "freeze-detector-snapshot",
        help=(
            "Run the current RT-DETR-v2 detector once and write a frozen "
            "external geometry snapshot."
        ),
    )
    freeze.add_argument("--source-image", type=Path, required=True)
    freeze.add_argument("--case-id", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    freeze.add_argument(
        "--source-language",
        choices=("Japanese", "Chinese", "English"),
        default="Japanese",
    )

    profiles = subparsers.add_parser(
        "run-fullpage-profiles",
        help=(
            "Run the locked 1728/8192 and 2048/12288 profiles in AB/BA "
            "order using managed MangaLMM and frozen detector snapshots."
        ),
    )
    profiles.add_argument("--manifest", type=Path, required=True)
    profiles.add_argument("--output-dir", type=Path, required=True)
    profiles.add_argument(
        "--split",
        choices=tuple(sorted(ALLOWED_SPLITS)),
        default="development",
    )
    profiles.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Optional neutral case ID; repeat to select multiple cases.",
    )
    profiles.add_argument("--timeout-sec", type=int, default=300)
    profiles.add_argument(
        "--max-idle-gpu-used-mib",
        type=float,
        default=DEFAULT_MAX_IDLE_GPU_USED_MIB,
        help=(
            "Reject the live run when baseline GPU memory is above this "
            "value. The project-approved default is 2048 MiB."
        ),
    )

    validate_results = subparsers.add_parser(
        "validate-profile-results",
        help=(
            "Verify the locked AB/BA profile result, raw responses, "
            "overlays, blind review, and unblind key without model requests."
        ),
    )
    validate_results.add_argument(
        "--results",
        type=Path,
        required=True,
    )
    validate_results.add_argument("--output", type=Path)

    analyze_results = subparsers.add_parser(
        "analyze-profile-results",
        help=(
            "Verify and summarize a locked profile result without model "
            "requests or automatic product promotion."
        ),
    )
    analyze_results.add_argument(
        "--results",
        type=Path,
        required=True,
    )
    analyze_results.add_argument("--output", type=Path)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    display_payload: Any | None = None
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
        elif args.command == "freeze-detector-snapshot":
            payload = freeze_detector_snapshot(
                source_image=args.source_image,
                case_id=args.case_id,
                output=args.output,
                source_lang_english=args.source_language,
            )
        elif args.command == "run-fullpage-profiles":
            if args.timeout_sec < 1 or args.timeout_sec > 3600:
                raise BenchmarkContractError(
                    "--timeout-sec must be between 1 and 3600."
                )
            if (
                not math.isfinite(args.max_idle_gpu_used_mib)
                or args.max_idle_gpu_used_mib <= 0
            ):
                raise BenchmarkContractError(
                    "--max-idle-gpu-used-mib must be positive."
                )
            payload = run_fullpage_profiles(
                manifest_path=args.manifest,
                output_dir=args.output_dir,
                split=args.split,
                case_ids=args.case_id,
                timeout_sec=args.timeout_sec,
                max_idle_gpu_used_mib=args.max_idle_gpu_used_mib,
            )
            display_payload = {
                "protocol_version": payload["protocol_version"],
                "result_file": str(
                    (args.output_dir / "profile-results.json").resolve()
                ),
                "result_sha256": payload["result_sha256"],
                "case_ids": payload["case_ids"],
                "run_unit_count": len(payload["run_units"]),
                "status": "completed",
            }
        elif args.command == "validate-profile-results":
            verified = validate_profile_results_file(args.results)
            payload = {
                key: value
                for key, value in verified.items()
                if key != "payload"
            }
        elif args.command == "analyze-profile-results":
            verified = validate_profile_results_file(args.results)
            payload = summarize_profile_results(verified["payload"])
        else:
            raise BenchmarkContractError(f"Unsupported command: {args.command}")
        if (
            getattr(args, "output", None)
            and args.command not in {"freeze-detector-snapshot"}
        ):
            write_external_json(args.output, payload)
        _print_json(display_payload if display_payload is not None else payload)
        return 0
    except BenchmarkContractError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
