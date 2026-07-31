#!/usr/bin/env python3
"""Build a source-first, blind review for the three managed OCR strategies.

This benchmark-only tool never starts Docker or performs inference.  It locks
source images and detector geometry before candidate results are imported,
normalizes existing route outputs, and builds a Git-external A/B/C review
package.  Semantic correctness is accepted only from a completed human review;
text similarity is diagnostic and is never used as the quality decision.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
import secrets
import shutil
import sys
import unicodedata
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_VERSION = "ocr-three-way-human-truth-v1"
CORPUS_SCHEMA_VERSION = 1
TRUTH_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
REVIEW_SCHEMA_VERSION = 1

ROUTE_IMAGE_MAX_PIXELS = {
    "paddle_crop": 1_003_520,
    "paddle_spotting_full_page": 1_605_632,
    "mangalmm_full_page": 2_116_800,
}
ROUTE_PROMPT_MODES = {
    "paddle_crop": "OCR:",
    "paddle_spotting_full_page": "Spotting:",
    "mangalmm_full_page": "mangalmm_official_full_page",
}

ROUTES = (
    "paddle_crop",
    "paddle_spotting_full_page",
    "mangalmm_full_page",
)
BLIND_LABELS = ("A", "B", "C")
ALLOWED_SPLITS = {
    "development",
    "stress",
    "holdout",
    "negative_control",
    "final",
}
ALLOWED_LANGUAGES = {"ja", "en", "zh"}
ALLOWED_ROLES = {
    "dialogue_bubble",
    "dialogue_free",
    "narration",
    "ui_or_sign",
    "sfx",
    "decorative",
    "ambiguous",
}
ALLOWED_ACTIONS = {"translate_inpaint", "preserve", "review"}
ALLOWED_CONFIDENCE = {"high", "medium", "low"}
ALLOWED_ASSET_KINDS = {
    "ocr-crop",
    "raw-mask",
    "final-mask",
    "cleaned-crop",
    "render",
    "diff",
    "detector-overlay",
    "mask-overlay",
    "cleanup-delta",
    "debug-metadata",
}
REVIEW_DECISIONS = {"yes", "no", "uncertain", "not_applicable"}
SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

TRUTH_STATE_FILENAME = "truth_state.json"
TRUTH_LOCK_FILENAME = "truth.lock.json"
TRUTH_SOURCE_REVIEW_FILENAME = "source-review.html"
TRUTH_CSV_FILENAME = "truth-entry.csv"
RUN_FILENAME = "normalized_run.json"
BLIND_KEY_FILENAME = "blind_key.json"
BLIND_PAYLOAD_FILENAME = "blind_payload.json"
REVIEW_CSV_FILENAME = "region-review.csv"
REVIEW_HTML_FILENAME = "index.html"
FINAL_METRICS_FILENAME = "final_metrics.json"
FINAL_REPORT_FILENAME = "final_report-ko.md"
SOURCE_BINDING_FILENAME = "source-bindings.json"
CSV_FORMULA_PREFIXES = ("=", "+", "-", "@")

REVIEW_ID_COLUMNS = (
    "row_number",
    "row_id",
    "row_kind",
    "page_id",
    "truth_region_id",
    "truth_region_source",
    "language",
    "truth_transcription",
    "truth_semantic_role",
    "truth_processing_action",
    "truth_confidence",
    "truth_bbox_xyxy",
    "source_page",
    "source_crop",
)
REVIEW_ROUTE_FIELDS = (
    "text",
    "bbox_xyxy",
    "raw_region_ids",
    "semantic_role",
    "processing_action",
    "geometry_status",
    "assets_json",
    "transcription_correct",
    "semantic_correct",
    "role_action_correct",
    "merge_split_error",
    "destructive_edit",
    "false_positive",
    "notes",
)

TRUTH_CSV_COLUMNS = (
    "page_id",
    "truth_region_id",
    "region_source",
    "detector_block_ids_json",
    "bbox_xyxy_json",
    "detector_text_class",
    "direction",
    "transcription",
    "semantic_role",
    "processing_action",
    "confidence",
    "meaning_notes",
)

REVIEW_DECISION_SUFFIXES = {
    "transcription_correct",
    "semantic_correct",
    "role_action_correct",
    "merge_split_error",
    "destructive_edit",
    "false_positive",
    "notes",
}


class ContractError(ValueError):
    """Raised when benchmark evidence violates a locked contract."""


class IncompleteReviewError(ContractError):
    def __init__(self, errors: Sequence[str]) -> None:
        self.errors = tuple(errors)
        super().__init__(
            f"Blind review is incomplete or invalid ({len(self.errors)} errors)."
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    _ensure_finite_json(payload, label="canonical JSON payload")
    encoded = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _ensure_finite_json(payload: Any, *, label: str) -> None:
    if isinstance(payload, float) and not math.isfinite(payload):
        raise ContractError(f"Non-finite number in {label}.")
    if isinstance(payload, dict):
        for value in payload.values():
            _ensure_finite_json(value, label=label)
    elif isinstance(payload, list):
        for value in payload:
            _ensure_finite_json(value, label=label)


def read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"Unable to read JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ContractError(f"Expected JSON object: {path}")
    _ensure_finite_json(payload, label=str(path))
    return payload


def write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _csv_encode_cell(value: Any) -> str:
    """Keep CSV cells inert in spreadsheets while preserving a reversible value."""
    raw = str(value if value is not None else "")
    if raw.startswith("'") or raw.lstrip().startswith(CSV_FORMULA_PREFIXES):
        return "'" + raw
    return raw


def _csv_decode_cell(value: Any) -> str:
    raw = str(value if value is not None else "")
    if raw.startswith("''"):
        return raw[1:]
    if raw.startswith("'") and raw[1:].lstrip().startswith(CSV_FORMULA_PREFIXES):
        return raw[1:]
    return raw


def _is_inside_repo(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    root = ROOT.resolve()
    return resolved == root or root in resolved.parents


def require_external_path(path: Path, *, label: str) -> Path:
    resolved = path.expanduser().resolve(strict=False)
    if _is_inside_repo(resolved):
        raise ContractError(f"{label} must remain outside the Git repository: {resolved}")
    return resolved


def require_safe_id(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip()
    if not SAFE_ID_RE.fullmatch(normalized):
        raise ContractError(f"Invalid {label}: {value!r}")
    return normalized


def require_sha256(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not SHA256_RE.fullmatch(normalized):
        raise ContractError(f"Invalid SHA-256 for {label}.")
    return normalized


def require_asset_kind(value: Any) -> str:
    kind = require_safe_id(value, label="OCR unit asset kind")
    base_kind = re.sub(r"^region-[0-9]{4}-", "", kind)
    if base_kind not in ALLOWED_ASSET_KINDS:
        raise ContractError(f"Unsupported OCR unit asset kind: {kind}")
    return kind


def verify_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    path = require_external_path(Path(str(record.get("path", ""))), label=label)
    if not path.is_file():
        raise ContractError(f"Missing {label}: {path}")
    expected = require_sha256(record.get("sha256"), label=label)
    actual = sha256_file(path)
    if actual != expected:
        raise ContractError(
            f"SHA-256 mismatch for {label}: expected {expected}, got {actual}"
        )
    return path


def _bbox(value: Any, *, label: str, width: int, height: int) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError(f"{label} must be [x1, y1, x2, y2].")
    try:
        x1, y1, x2, y2 = [int(round(float(item))) for item in value]
    except (TypeError, ValueError) as exc:
        raise ContractError(f"{label} contains a non-numeric coordinate.") from exc
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ContractError(
            f"{label} is outside {width}x{height}: {[x1, y1, x2, y2]}"
        )
    return [x1, y1, x2, y2]


def _bbox_area(box: Sequence[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(
        0, int(box[3]) - int(box[1])
    )


def _bbox_intersection(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _geometry_score(left: Sequence[int], right: Sequence[int]) -> float:
    intersection = _bbox_intersection(left, right)
    if not intersection:
        return 0.0
    smaller = min(_bbox_area(left), _bbox_area(right))
    return intersection / smaller if smaller else 0.0


def _union_bbox(boxes: Iterable[Sequence[int]]) -> list[int]:
    materialized = [list(box) for box in boxes]
    if not materialized:
        return [0, 0, 1, 1]
    return [
        min(box[0] for box in materialized),
        min(box[1] for box in materialized),
        max(box[2] for box in materialized),
        max(box[3] for box in materialized),
    ]


def _image_size(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - product environments include Pillow
        raise ContractError("Pillow is required to inspect benchmark images.") from exc
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _copy_crop(source: Path, bbox: Sequence[int], destination: Path) -> None:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise ContractError("Pillow is required to create review crops.") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        crop = image.crop(tuple(int(value) for value in bbox))
        temporary = destination.with_name(destination.name + ".tmp.png")
        crop.save(temporary, format="PNG")
        os.replace(temporary, destination)


def _load_detector_blocks(
    path: Path,
    *,
    page_id: str,
    source_sha256: str,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    payload = read_json(path)
    embedded_sha = str(payload.get("source_sha256", "") or "").lower()
    if embedded_sha and embedded_sha != source_sha256:
        raise ContractError(f"Detector source hash mismatch for page {page_id}.")
    raw_blocks = payload.get("blocks")
    if not isinstance(raw_blocks, list) or not raw_blocks:
        raise ContractError(f"Detector snapshot has no blocks for page {page_id}.")
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_blocks):
        if not isinstance(raw, dict):
            raise ContractError(f"Invalid detector block {page_id}:{index}.")
        block_id = require_safe_id(
            raw.get("block_id") or f"detector-{index:04d}",
            label="detector block ID",
        )
        if block_id in seen:
            raise ContractError(f"Duplicate detector block ID on {page_id}: {block_id}")
        seen.add(block_id)
        box = _bbox(raw.get("xyxy"), label=f"{page_id}:{block_id}", width=width, height=height)
        bubble = raw.get("bubble_xyxy")
        blocks.append(
            {
                "block_id": block_id,
                "block_index": index,
                "bbox_xyxy": box,
                "bubble_xyxy": (
                    _bbox(
                        bubble,
                        label=f"{page_id}:{block_id}:bubble",
                        width=width,
                        height=height,
                    )
                    if bubble
                    else None
                ),
                "text_class": str(raw.get("text_class", "") or ""),
                "direction": str(raw.get("direction", "") or ""),
            }
        )
    return blocks


def validate_corpus_manifest(path: Path) -> dict[str, Any]:
    path = require_external_path(path, label="corpus manifest")
    payload = read_json(path)
    if payload.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise ContractError("Unsupported corpus manifest schema.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ContractError("Corpus manifest protocol mismatch.")
    require_safe_id(payload.get("suite_id"), label="suite ID")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ContractError("Corpus manifest must contain at least one page.")
    seen: set[str] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise ContractError("Corpus page must be an object.")
        page_id = require_safe_id(page.get("page_id"), label="page ID")
        if page.get("source_page_id") is not None:
            require_safe_id(page.get("source_page_id"), label="source page ID")
        if page_id in seen:
            raise ContractError(f"Duplicate page ID: {page_id}")
        seen.add(page_id)
        if page.get("split") not in ALLOWED_SPLITS:
            raise ContractError(f"Invalid split for {page_id}.")
        if page.get("language") not in ALLOWED_LANGUAGES:
            raise ContractError(f"Invalid language for {page_id}.")
        source_path = verify_file_record(page.get("source_image", {}), label=f"source {page_id}")
        width, height = _image_size(source_path)
        source_record = page["source_image"]
        if int(source_record.get("width", 0)) != width or int(
            source_record.get("height", 0)
        ) != height:
            raise ContractError(f"Image dimensions changed for {page_id}.")
        detector_path = verify_file_record(
            page.get("detector_snapshot", {}),
            label=f"detector snapshot {page_id}",
        )
        _load_detector_blocks(
            detector_path,
            page_id=page_id,
            source_sha256=source_record["sha256"],
            width=width,
            height=height,
        )
    expected_digest = payload.get("manifest_sha256")
    if not expected_digest:
        raise ContractError("Corpus manifest canonical hash is required.")
    digest_payload = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    actual_digest = canonical_sha256(digest_payload)
    if expected_digest and require_sha256(expected_digest, label="manifest") != actual_digest:
        raise ContractError("Corpus manifest canonical hash mismatch.")
    return payload


def build_corpus_manifest(spec_path: Path, output_path: Path) -> dict[str, Any]:
    spec_file = require_external_path(spec_path, label="corpus build spec")
    output = require_external_path(output_path, label="corpus manifest output")
    if output.exists():
        raise ContractError(f"Corpus manifest output already exists: {output}")
    spec = read_json(spec_file)
    if spec.get("schema_version") != 1:
        raise ContractError("Unsupported corpus build spec schema.")
    suite_id = require_safe_id(spec.get("suite_id"), label="suite ID")
    groups = spec.get("groups")
    if not isinstance(groups, list) or not groups:
        raise ContractError("Corpus build spec must contain groups.")
    pages: list[dict[str, Any]] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, dict):
            raise ContractError("Corpus build group must be an object.")
        split = str(group.get("split", ""))
        language = str(group.get("language", ""))
        if split not in ALLOWED_SPLITS or language not in ALLOWED_LANGUAGES:
            raise ContractError("Corpus build group has invalid split or language.")
        source_dir = require_external_path(
            Path(str(group.get("source_dir", ""))), label="corpus source directory"
        )
        detector_dir = require_external_path(
            Path(str(group.get("detector_results", ""))),
            label="detector results directory",
        )
        requested = group.get("page_ids")
        if not isinstance(requested, list) or not requested:
            raise ContractError("Corpus build group page_ids must be a non-empty list.")
        prefix = str(group.get("page_id_prefix", "") or "")
        for raw_page_id in requested:
            source_stem = require_safe_id(raw_page_id, label="source page ID")
            page_id = require_safe_id(f"{prefix}{source_stem}", label="page ID")
            if page_id in seen:
                raise ContractError(f"Duplicate generated page ID: {page_id}")
            seen.add(page_id)
            source_matches = [
                path
                for path in source_dir.glob(f"{source_stem}.*")
                if path.suffix.lower()
                in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
            ]
            if len(source_matches) != 1:
                raise ContractError(
                    f"Expected exactly one source image for {source_stem}, got {len(source_matches)}."
                )
            source_path = source_matches[0].resolve()
            detector_candidates = (
                detector_dir / source_stem / "result.json",
                detector_dir / f"{source_stem}.json",
            )
            detector_path = next((path for path in detector_candidates if path.is_file()), None)
            if detector_path is None:
                raise ContractError(f"Missing detector snapshot for {source_stem}.")
            width, height = _image_size(source_path)
            source_sha = sha256_file(source_path)
            _load_detector_blocks(
                detector_path,
                page_id=page_id,
                source_sha256=source_sha,
                width=width,
                height=height,
            )
            pages.append(
                {
                    "page_id": page_id,
                    "source_page_id": source_stem,
                    "split": split,
                    "language": language,
                    "source_image": {
                        "path": str(source_path),
                        "sha256": source_sha,
                        "width": width,
                        "height": height,
                    },
                    "detector_snapshot": {
                        "path": str(detector_path.resolve()),
                        "sha256": sha256_file(detector_path),
                    },
                }
            )
    manifest: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": suite_id,
        "created_at": utc_now(),
        "build_spec_path": str(spec_file),
        "build_spec_sha256": sha256_file(spec_file),
        "pages": pages,
    }
    manifest["manifest_sha256"] = canonical_sha256(manifest)
    write_json(output, manifest)
    validate_corpus_manifest(output)
    return manifest


def init_truth(corpus_manifest: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = require_external_path(corpus_manifest, label="corpus manifest")
    output = require_external_path(output_dir, label="truth output")
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"Truth output must be empty: {output}")
    manifest = validate_corpus_manifest(manifest_path)
    output.mkdir(parents=True, exist_ok=True)
    pages_dir = output / "pages"
    page_assets = output / "assets" / "pages"
    crop_assets = output / "assets" / "crops"
    page_records: list[dict[str, Any]] = []
    for page in manifest["pages"]:
        page_id = page["page_id"]
        source_path = Path(page["source_image"]["path"]).expanduser().resolve()
        detector_path = Path(page["detector_snapshot"]["path"]).expanduser().resolve()
        width = int(page["source_image"]["width"])
        height = int(page["source_image"]["height"])
        blocks = _load_detector_blocks(
            detector_path,
            page_id=page_id,
            source_sha256=page["source_image"]["sha256"],
            width=width,
            height=height,
        )
        copied_page = page_assets / f"{page_id}{source_path.suffix.lower()}"
        copied_page.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, copied_page)
        if sha256_file(copied_page) != page["source_image"]["sha256"]:
            raise ContractError(f"Source image changed while freezing page {page_id}.")
        truth_regions: list[dict[str, Any]] = []
        for index, block in enumerate(blocks):
            truth_id = f"{page_id}-det-{index:04d}"
            crop_path = crop_assets / page_id / f"{truth_id}.png"
            _copy_crop(copied_page, block["bbox_xyxy"], crop_path)
            truth_regions.append(
                {
                    "truth_region_id": truth_id,
                    "region_source": "detector",
                    "detector_block_ids": [block["block_id"]],
                    "bbox_xyxy": block["bbox_xyxy"],
                    "detector_text_class": block["text_class"],
                    "direction": block["direction"],
                    "transcription": "",
                    "semantic_role": "",
                    "processing_action": "",
                    "confidence": "",
                    "meaning_notes": "",
                    "crop_asset": crop_path.relative_to(output).as_posix(),
                }
            )
        truth_page = {
            "schema_version": TRUTH_SCHEMA_VERSION,
            "protocol_version": PROTOCOL_VERSION,
            "page_id": page_id,
            "split": page["split"],
            "language": page["language"],
            "source_sha256": page["source_image"]["sha256"],
            "source_asset": copied_page.relative_to(output).as_posix(),
            "image_width": width,
            "image_height": height,
            "regions": truth_regions,
            "page_notes": "",
        }
        truth_path = pages_dir / f"{page_id}.json"
        write_json(truth_path, truth_page)
        page_records.append(
            {
                "page_id": page_id,
                "truth_file": truth_path.relative_to(output).as_posix(),
                "detector_block_count": len(blocks),
            }
        )
    state = {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": manifest["suite_id"],
        "created_at": utc_now(),
        "source_first": True,
        "candidate_results_visible": False,
        "corpus_manifest_path": str(manifest_path),
        "corpus_manifest_file_sha256": sha256_file(manifest_path),
        "corpus_manifest_canonical_sha256": canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ),
        "pages": page_records,
        "instructions": (
            "Read the source page and enlarged crop only. Fill every region, add "
            "human_extra regions when detector geometry missed meaningful text, then lock."
        ),
    }
    write_json(output / TRUTH_STATE_FILENAME, state)
    write_truth_source_review(output)
    export_truth_csv(output)
    return state


def _truth_files(truth_dir: Path, state: Mapping[str, Any]) -> list[Path]:
    files: list[Path] = []
    for page in state.get("pages", []):
        if not isinstance(page, dict):
            raise ContractError("Invalid truth state page record.")
        relative = Path(str(page.get("truth_file", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("Truth file path escapes the truth directory.")
        path = (truth_dir / relative).resolve()
        if truth_dir.resolve() not in path.parents:
            raise ContractError("Truth file path escapes the truth directory.")
        if not path.is_file():
            raise ContractError(f"Missing truth page: {path}")
        files.append(path)
    return files


def refresh_truth_crops(truth_dir: Path) -> dict[str, Any]:
    truth = require_external_path(truth_dir, label="truth directory")
    if (truth / TRUTH_LOCK_FILENAME).exists():
        raise ContractError("Locked truth crops cannot be regenerated.")
    state = read_json(truth / TRUTH_STATE_FILENAME)
    updated = 0
    for path in _truth_files(truth, state):
        page = read_json(path)
        source_relative = Path(str(page.get("source_asset", "")))
        if source_relative.is_absolute() or ".." in source_relative.parts:
            raise ContractError("Truth source asset escapes the truth directory.")
        source = (truth / source_relative).resolve()
        if not source.is_file() or truth.resolve() not in source.parents:
            raise ContractError(f"Missing truth source asset: {source}")
        changed = False
        for region in page.get("regions", []):
            truth_id = require_safe_id(
                region.get("truth_region_id"), label="truth region ID"
            )
            crop_relative = Path(str(region.get("crop_asset", "") or ""))
            if not crop_relative.parts:
                crop_relative = Path("assets") / "crops" / page["page_id"] / f"{truth_id}.png"
                region["crop_asset"] = crop_relative.as_posix()
                changed = True
            if crop_relative.is_absolute() or ".." in crop_relative.parts:
                raise ContractError("Truth crop asset escapes the truth directory.")
            destination = (truth / crop_relative).resolve()
            if truth.resolve() not in destination.parents:
                raise ContractError("Truth crop asset escapes the truth directory.")
            _copy_crop(source, region["bbox_xyxy"], destination)
            updated += 1
        if changed:
            write_json(path, page)
    write_truth_source_review(truth)
    return {"page_count": len(state.get("pages", [])), "crop_count": updated}


def write_truth_source_review(truth_dir: Path) -> Path:
    truth = require_external_path(truth_dir, label="truth directory")
    state = read_json(truth / TRUTH_STATE_FILENAME)
    sections: list[str] = []
    for page_path in _truth_files(truth, state):
        page = read_json(page_path)
        cards: list[str] = []
        for region in page.get("regions", []):
            crop = html.escape(str(region.get("crop_asset", "")))
            cards.append(
                "<article class='crop'>"
                f"<h3>{html.escape(str(region.get('truth_region_id', '')))}</h3>"
                f"<img src='{crop}' alt='source crop'>"
                f"<p>bbox: {html.escape(json.dumps(region.get('bbox_xyxy', [])))}</p>"
                f"<p>detector: {html.escape(','.join(str(value) for value in region.get('detector_block_ids', [])))}</p>"
                "</article>"
            )
        sections.append(
            "<section class='page'>"
            f"<h2>{html.escape(str(page.get('page_id', '')))}</h2>"
            f"<img class='full' src='{html.escape(str(page.get('source_asset', '')))}' alt='source page'>"
            "<div class='grid'>" + "".join(cards) + "</div></section>"
        )
    document = """<!doctype html><html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR source-first truth review</title><style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f3f3f3;color:#111}
.page{background:#fff;padding:18px;margin-bottom:30px;border-radius:12px}
.full{max-width:700px;max-height:900px;border:1px solid #aaa}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;margin-top:16px}
.crop{border:1px solid #ccc;padding:10px;border-radius:8px;overflow:auto}
.crop img{max-width:100%;max-height:420px;image-rendering:auto}
</style></head><body>
<h1>원본 선판독 정답 작성용</h1>
<p>이 페이지에는 OCR 후보 결과가 없습니다. 원본과 detector crop만 보고 pages/*.json을 작성합니다.</p>
""" + "".join(sections) + "</body></html>\n"
    output = truth / TRUTH_SOURCE_REVIEW_FILENAME
    output.write_text(document, encoding="utf-8")
    return output


def _truth_csv_rows(
    truth_dir: Path,
    state: Mapping[str, Any],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for page_path in _truth_files(truth_dir, state):
        page = read_json(page_path)
        for region in page.get("regions", []):
            if not isinstance(region, dict):
                raise ContractError(f"Invalid truth region in {page_path}.")
            rows.append(
                {
                    "page_id": str(page.get("page_id", "")),
                    "truth_region_id": str(region.get("truth_region_id", "")),
                    "region_source": str(region.get("region_source", "")),
                    "detector_block_ids_json": json.dumps(
                        region.get("detector_block_ids", []), ensure_ascii=False
                    ),
                    "bbox_xyxy_json": json.dumps(
                        region.get("bbox_xyxy", []), ensure_ascii=False
                    ),
                    "detector_text_class": str(
                        region.get("detector_text_class", "") or ""
                    ),
                    "direction": str(region.get("direction", "") or ""),
                    "transcription": str(region.get("transcription", "") or ""),
                    "semantic_role": str(region.get("semantic_role", "") or ""),
                    "processing_action": str(
                        region.get("processing_action", "") or ""
                    ),
                    "confidence": str(region.get("confidence", "") or ""),
                    "meaning_notes": str(region.get("meaning_notes", "") or ""),
                }
            )
    return rows


def _read_truth_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != TRUTH_CSV_COLUMNS:
                raise ContractError("Truth CSV columns or order changed.")
            return [
                {key: _csv_decode_cell(value) for key, value in row.items()}
                for row in reader
            ]
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise ContractError(f"Unable to read truth CSV: {path}") from exc


def export_truth_csv(truth_dir: Path, output_path: Path | None = None) -> dict[str, Any]:
    truth = require_external_path(truth_dir, label="truth directory")
    state = read_json(truth / TRUTH_STATE_FILENAME)
    output = (
        require_external_path(output_path, label="truth CSV output")
        if output_path is not None
        else truth / TRUTH_CSV_FILENAME
    )
    rows = _truth_csv_rows(truth, state)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=TRUTH_CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(
            {
                column: _csv_encode_cell(row.get(column, ""))
                for column in TRUTH_CSV_COLUMNS
            }
            for row in rows
        )
    return {"truth_csv": str(output), "row_count": len(rows), "sha256": sha256_file(output)}


def _parse_csv_json_list(value: str, *, label: str) -> list[Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ContractError(f"Invalid JSON list in {label}.") from exc
    if not isinstance(parsed, list):
        raise ContractError(f"{label} must be a JSON list.")
    return parsed


def import_truth_csv(truth_dir: Path, csv_path: Path) -> dict[str, Any]:
    truth = require_external_path(truth_dir, label="truth directory")
    source_csv = require_external_path(csv_path, label="truth CSV")
    if (truth / TRUTH_LOCK_FILENAME).exists():
        raise ContractError("Locked truth cannot be updated from CSV.")
    state = read_json(truth / TRUTH_STATE_FILENAME)
    page_paths = _truth_files(truth, state)
    pages: dict[str, dict[str, Any]] = {}
    paths: dict[str, Path] = {}
    for path in page_paths:
        page = read_json(path)
        page_id = str(page["page_id"])
        pages[page_id] = page
        paths[page_id] = path
    original_detector_regions = {
        (page_id, str(region["truth_region_id"])): region
        for page_id, page in pages.items()
        for region in page.get("regions", [])
        if isinstance(region, dict) and region.get("region_source") == "detector"
    }
    imported: dict[str, list[dict[str, Any]]] = {page_id: [] for page_id in pages}
    seen_ids: set[tuple[str, str]] = set()
    for row_index, row in enumerate(_read_truth_csv(source_csv), start=2):
        page_id = require_safe_id(row.get("page_id"), label=f"truth CSV row {row_index} page ID")
        if page_id not in pages:
            raise ContractError(f"Truth CSV row {row_index} has an unknown page: {page_id}")
        truth_id = require_safe_id(
            row.get("truth_region_id"), label=f"truth CSV row {row_index} region ID"
        )
        identity = (page_id, truth_id)
        if identity in seen_ids:
            raise ContractError(f"Duplicate truth CSV region: {page_id}:{truth_id}")
        seen_ids.add(identity)
        region_source = str(row.get("region_source", ""))
        if region_source not in {"detector", "human_extra"}:
            raise ContractError(f"Invalid region source in truth CSV row {row_index}.")
        detector_ids = _parse_csv_json_list(
            row.get("detector_block_ids_json", ""),
            label=f"truth CSV row {row_index} detector IDs",
        )
        detector_ids = [
            require_safe_id(value, label=f"truth CSV row {row_index} detector ID")
            for value in detector_ids
        ]
        bbox = _bbox(
            _parse_csv_json_list(
                row.get("bbox_xyxy_json", ""),
                label=f"truth CSV row {row_index} bbox",
            ),
            label=f"truth CSV row {row_index} bbox",
            width=int(pages[page_id]["image_width"]),
            height=int(pages[page_id]["image_height"]),
        )
        original = original_detector_regions.get(identity)
        if original is not None:
            if region_source != "detector" or detector_ids != original["detector_block_ids"]:
                raise ContractError(
                    f"Detector truth identity changed in CSV: {page_id}:{truth_id}"
                )
        elif region_source != "human_extra" or detector_ids:
            raise ContractError(
                f"New truth CSV regions must be human_extra: {page_id}:{truth_id}"
            )
        imported[page_id].append(
            {
                "truth_region_id": truth_id,
                "region_source": region_source,
                "detector_block_ids": detector_ids,
                "bbox_xyxy": bbox,
                "detector_text_class": str(row.get("detector_text_class", "") or ""),
                "direction": str(row.get("direction", "") or ""),
                "transcription": str(row.get("transcription", "") or ""),
                "semantic_role": str(row.get("semantic_role", "") or ""),
                "processing_action": str(row.get("processing_action", "") or ""),
                "confidence": str(row.get("confidence", "") or ""),
                "meaning_notes": str(row.get("meaning_notes", "") or ""),
                "crop_asset": (
                    Path("assets") / "crops" / page_id / f"{truth_id}.png"
                ).as_posix(),
            }
        )
    missing_detector = sorted(set(original_detector_regions) - seen_ids)
    if missing_detector:
        raise ContractError(
            f"Truth CSV omitted detector regions: {missing_detector[:10]}"
        )
    for page_id, page in pages.items():
        page["regions"] = imported[page_id]
        write_json(paths[page_id], page)
    refresh_truth_crops(truth)
    canonical_csv = export_truth_csv(truth)
    return {
        "page_count": len(pages),
        "region_count": sum(len(regions) for regions in imported.values()),
        "truth_csv": canonical_csv,
    }


def _validate_truth_page(page: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    page_id = str(page.get("page_id", ""))
    width = int(page.get("image_width", 0) or 0)
    height = int(page.get("image_height", 0) or 0)
    if width <= 0 or height <= 0:
        errors.append(f"{page_id}: invalid image dimensions")
        return errors
    regions = page.get("regions")
    if not isinstance(regions, list) or not regions:
        return [f"{page_id}: no truth regions"]
    seen_truth: set[str] = set()
    seen_detector: set[str] = set()
    for index, region in enumerate(regions):
        prefix = f"{page_id}:region-{index}"
        if not isinstance(region, dict):
            errors.append(f"{prefix}: not an object")
            continue
        try:
            truth_id = require_safe_id(region.get("truth_region_id"), label="truth region ID")
            if truth_id in seen_truth:
                errors.append(f"{prefix}: duplicate truth_region_id {truth_id}")
            seen_truth.add(truth_id)
            _bbox(region.get("bbox_xyxy"), label=prefix, width=width, height=height)
        except ContractError as exc:
            errors.append(str(exc))
        source = str(region.get("region_source", ""))
        if source not in {"detector", "human_extra"}:
            errors.append(f"{prefix}: invalid region_source")
        detector_ids = region.get("detector_block_ids")
        if not isinstance(detector_ids, list):
            errors.append(f"{prefix}: detector_block_ids must be a list")
            detector_ids = []
        if source == "detector" and not detector_ids:
            errors.append(f"{prefix}: detector region has no detector block")
        if source == "human_extra" and detector_ids:
            errors.append(f"{prefix}: human_extra must not claim detector blocks")
        for detector_id in detector_ids:
            normalized = str(detector_id)
            if normalized in seen_detector:
                errors.append(f"{prefix}: detector block reused: {normalized}")
            seen_detector.add(normalized)
        role = str(region.get("semantic_role", ""))
        action = str(region.get("processing_action", ""))
        confidence = str(region.get("confidence", ""))
        transcription = str(region.get("transcription", ""))
        if role not in ALLOWED_ROLES:
            errors.append(f"{prefix}: semantic_role is incomplete")
        if action not in ALLOWED_ACTIONS:
            errors.append(f"{prefix}: processing_action is incomplete")
        if confidence not in ALLOWED_CONFIDENCE:
            errors.append(f"{prefix}: confidence is incomplete")
        if not transcription.strip() and not (
            role in {"decorative", "ambiguous"} and action in {"preserve", "review"}
        ):
            errors.append(f"{prefix}: transcription is incomplete")
    return errors


def _truth_manifest(
    truth: Path,
    state: Mapping[str, Any],
) -> dict[str, Any]:
    manifest_path = require_external_path(
        Path(str(state.get("corpus_manifest_path", ""))),
        label="truth corpus manifest",
    )
    if sha256_file(manifest_path) != state.get("corpus_manifest_file_sha256"):
        raise ContractError("Truth corpus manifest changed.")
    manifest = validate_corpus_manifest(manifest_path)
    canonical_manifest_sha = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if canonical_manifest_sha != state.get("corpus_manifest_canonical_sha256"):
        raise ContractError("Truth corpus manifest contract changed.")
    if manifest.get("suite_id") != state.get("suite_id"):
        raise ContractError("Truth suite differs from its corpus manifest.")
    manifest_pages = {str(page["page_id"]): page for page in manifest["pages"]}
    state_pages: dict[str, Mapping[str, Any]] = {}
    for record in state.get("pages", []):
        if not isinstance(record, dict):
            raise ContractError("Invalid truth state page record.")
        page_id = require_safe_id(record.get("page_id"), label="truth state page ID")
        if page_id in state_pages:
            raise ContractError(f"Duplicate truth state page: {page_id}")
        state_pages[page_id] = record
    if set(state_pages) != set(manifest_pages):
        raise ContractError("Truth state page set differs from its corpus manifest.")
    for page_id, manifest_page in manifest_pages.items():
        expected_count = len(_detector_blocks_for_page(manifest_page))
        if state_pages[page_id].get("detector_block_count") != expected_count:
            raise ContractError(f"Truth detector block count changed for {page_id}.")
    return manifest


def _validate_truth_against_manifest(
    page: Mapping[str, Any],
    manifest_page: Mapping[str, Any],
) -> list[str]:
    page_id = str(manifest_page["page_id"])
    errors: list[str] = []
    source = manifest_page["source_image"]
    expected_identity = (
        page_id,
        manifest_page["split"],
        manifest_page["language"],
        source["sha256"],
        int(source["width"]),
        int(source["height"]),
    )
    actual_identity = (
        str(page.get("page_id", "")),
        page.get("split"),
        page.get("language"),
        page.get("source_sha256"),
        page.get("image_width"),
        page.get("image_height"),
    )
    if actual_identity != expected_identity:
        errors.append(f"{page_id}: truth page identity changed")
    detector_blocks = _detector_blocks_for_page(manifest_page)
    expected = {block["block_id"]: block for block in detector_blocks}
    actual: dict[str, Mapping[str, Any]] = {}
    for region in page.get("regions", []):
        if not isinstance(region, dict) or region.get("region_source") != "detector":
            continue
        detector_ids = region.get("detector_block_ids")
        if not isinstance(detector_ids, list) or len(detector_ids) != 1:
            continue
        detector_id = str(detector_ids[0])
        actual[detector_id] = region
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        unexpected = sorted(set(actual) - set(expected))
        errors.append(
            f"{page_id}: detector truth set changed; missing={missing}, "
            f"unexpected={unexpected}"
        )
    for detector_id in sorted(set(actual).intersection(expected)):
        region = actual[detector_id]
        block = expected[detector_id]
        expected_truth_id = f"{page_id}-det-{block['block_index']:04d}"
        if (
            region.get("truth_region_id") != expected_truth_id
            or region.get("bbox_xyxy") != block["bbox_xyxy"]
            or str(region.get("detector_text_class", "")) != block["text_class"]
            or str(region.get("direction", "")) != block["direction"]
        ):
            errors.append(f"{page_id}: detector truth geometry changed: {detector_id}")
    return errors


def lock_truth(truth_dir: Path) -> dict[str, Any]:
    truth = require_external_path(truth_dir, label="truth directory")
    state_path = truth / TRUTH_STATE_FILENAME
    state = read_json(state_path)
    if not state.get("source_first") or state.get("candidate_results_visible") is not False:
        raise ContractError("Truth state is not source-first.")
    lock_path = truth / TRUTH_LOCK_FILENAME
    if lock_path.exists():
        raise ContractError("Truth is already locked; copy it to revise it.")
    manifest = _truth_manifest(truth, state)
    manifest_pages = {str(page["page_id"]): page for page in manifest["pages"]}
    errors: list[str] = []
    refresh_truth_crops(truth)
    truth_csv_path = truth / TRUTH_CSV_FILENAME
    if not truth_csv_path.is_file():
        errors.append("truth-entry.csv is missing; export it before locking")
    elif _read_truth_csv(truth_csv_path) != _truth_csv_rows(truth, state):
        errors.append(
            "truth-entry.csv and page JSON differ; import or export the intended truth first"
        )
    page_hashes: list[dict[str, str]] = []
    asset_hashes: list[dict[str, str]] = []
    for path in _truth_files(truth, state):
        page = read_json(path)
        errors.extend(_validate_truth_page(page))
        manifest_page = manifest_pages.get(str(page.get("page_id", "")))
        if manifest_page is None:
            errors.append(f"{page.get('page_id')}: page is absent from the corpus manifest")
        else:
            errors.extend(_validate_truth_against_manifest(page, manifest_page))
        asset_relatives = [str(page.get("source_asset", ""))] + [
            str(region.get("crop_asset", ""))
            for region in page.get("regions", [])
            if isinstance(region, dict)
        ]
        for relative_text in asset_relatives:
            relative = Path(relative_text)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{page.get('page_id')}: asset path escapes truth directory")
                continue
            asset = (truth / relative).resolve()
            if not asset.is_file() or truth.resolve() not in asset.parents:
                errors.append(f"{page.get('page_id')}: missing asset {relative_text}")
                continue
            asset_hashes.append(
                {"relative_path": relative.as_posix(), "sha256": sha256_file(asset)}
            )
            if (
                manifest_page is not None
                and relative_text == str(page.get("source_asset", ""))
                and sha256_file(asset) != manifest_page["source_image"]["sha256"]
            ):
                errors.append(f"{page.get('page_id')}: source asset changed")
        page_hashes.append(
            {
                "page_id": str(page.get("page_id", "")),
                "relative_path": path.relative_to(truth).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if errors:
        raise IncompleteReviewError(errors)
    page_hashes.sort(key=lambda item: item["page_id"])
    asset_hashes.sort(key=lambda item: item["relative_path"])
    lock = {
        "schema_version": TRUTH_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": state["suite_id"],
        "locked_at": utc_now(),
        "source_first": True,
        "candidate_results_visible_before_lock": False,
        "truth_state_sha256": sha256_file(state_path),
        "truth_csv_sha256": sha256_file(truth_csv_path) if truth_csv_path.is_file() else "",
        "corpus_manifest_file_sha256": state["corpus_manifest_file_sha256"],
        "page_files": page_hashes,
        "asset_files": asset_hashes,
    }
    lock["truth_contract_sha256"] = canonical_sha256(lock)
    write_json(lock_path, lock)
    return lock


def validate_locked_truth(truth_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth = require_external_path(truth_dir, label="truth directory")
    state = read_json(truth / TRUTH_STATE_FILENAME)
    lock = read_json(truth / TRUTH_LOCK_FILENAME)
    if (
        lock.get("suite_id") != state.get("suite_id")
        or lock.get("protocol_version") != PROTOCOL_VERSION
        or state.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ContractError("Truth state and lock identity differ.")
    manifest = _truth_manifest(truth, state)
    manifest_pages = {str(page["page_id"]): page for page in manifest["pages"]}
    expected_contract = str(lock.get("truth_contract_sha256", ""))
    contract_payload = {key: value for key, value in lock.items() if key != "truth_contract_sha256"}
    if require_sha256(expected_contract, label="truth contract") != canonical_sha256(contract_payload):
        raise ContractError("Truth lock contract was modified.")
    if sha256_file(truth / TRUTH_STATE_FILENAME) != lock.get("truth_state_sha256"):
        raise ContractError("Truth state changed after locking.")
    truth_csv_path = truth / TRUTH_CSV_FILENAME
    if (
        not truth_csv_path.is_file()
        or sha256_file(truth_csv_path) != lock.get("truth_csv_sha256")
    ):
        raise ContractError("Truth CSV changed after locking.")
    for record in lock.get("asset_files", []):
        if not isinstance(record, dict):
            raise ContractError("Invalid truth asset lock record.")
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("Truth asset lock path escapes truth directory.")
        asset = (truth / relative).resolve()
        if (
            not asset.is_file()
            or truth.resolve() not in asset.parents
            or sha256_file(asset) != record.get("sha256")
        ):
            raise ContractError(f"Truth asset changed after locking: {relative}")
    locked_files = {
        item["relative_path"]: item for item in lock.get("page_files", []) if isinstance(item, dict)
    }
    pages: list[dict[str, Any]] = []
    for path in _truth_files(truth, state):
        relative = path.relative_to(truth).as_posix()
        record = locked_files.get(relative)
        if not record or sha256_file(path) != record.get("sha256"):
            raise ContractError(f"Truth page changed after locking: {relative}")
        page = read_json(path)
        errors = _validate_truth_page(page)
        manifest_page = manifest_pages.get(str(page.get("page_id", "")))
        if manifest_page is None:
            errors.append(f"{page.get('page_id')}: page is absent from the corpus manifest")
        else:
            errors.extend(_validate_truth_against_manifest(page, manifest_page))
        if errors:
            raise IncompleteReviewError(errors)
        pages.append(page)
    if len(pages) != len(locked_files):
        raise ContractError("Truth lock page set mismatch.")
    return lock, pages


def _runtime_contract(path: Path, route: str) -> dict[str, Any]:
    contract_path = require_external_path(path, label="runtime contract")
    payload = read_json(contract_path)
    if payload.get("schema_version") != 1:
        raise ContractError("Unsupported runtime contract schema.")
    if payload.get("route_id") != route:
        raise ContractError("Runtime contract route mismatch.")
    if str(payload.get("backend", "")).strip().lower() != "llama.cpp":
        raise ContractError("Managed OCR benchmark backend must be llama.cpp.")
    expected_fingerprint = require_sha256(
        payload.get("fingerprint_sha256"), label="runtime fingerprint"
    )
    for field in ("model_sha256", "mmproj_sha256", "command_sha256"):
        digest = require_sha256(payload.get(field), label=field)
        if digest == "0" * 64:
            raise ContractError(f"Runtime {field} still contains a placeholder digest.")
    image_digest = str(payload.get("image_digest", ""))
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise ContractError("Runtime image_digest must be an immutable sha256 digest.")
    if image_digest == "sha256:" + "0" * 64:
        raise ContractError("Runtime image_digest still contains a placeholder digest.")
    try:
        image_max_pixels = int(payload.get("image_max_pixels", 0))
    except (TypeError, ValueError) as exc:
        raise ContractError("Runtime image_max_pixels must be an integer.") from exc
    if image_max_pixels != ROUTE_IMAGE_MAX_PIXELS[route]:
        raise ContractError(
            f"Runtime image_max_pixels does not match the official {route} contract."
        )
    prompt_mode = str(payload.get("prompt_mode", "")).strip()
    if not prompt_mode:
        raise ContractError("Runtime prompt_mode is required.")
    expected_prompt = ROUTE_PROMPT_MODES.get(route)
    if expected_prompt is not None and prompt_mode != expected_prompt:
        raise ContractError(f"Runtime prompt_mode does not match {route}.")
    if not isinstance(payload.get("special_tokens"), bool):
        raise ContractError("Runtime special_tokens must be a boolean.")
    expected_special_tokens = route == "paddle_spotting_full_page"
    if payload["special_tokens"] != expected_special_tokens:
        raise ContractError(
            f"Runtime special_tokens does not match the official {route} contract."
        )
    fingerprint_payload = {
        key: value for key, value in payload.items() if key != "fingerprint_sha256"
    }
    if canonical_sha256(fingerprint_payload) != expected_fingerprint:
        raise ContractError("Runtime fingerprint does not match its contract fields.")
    return payload


def _primary_source_result_paths(
    route: str,
    source_root: Path,
    source_page_id: str,
) -> list[Path]:
    if route == "paddle_crop":
        candidates = [source_root / source_page_id / "result.json"]
    elif route == "paddle_spotting_full_page":
        candidates = [
            source_root
            / "detector-fused-comparison-v1"
            / f"{source_page_id}.json",
            source_root / "geometry-audit" / f"{source_page_id}.json",
            source_root / f"{source_page_id}_spotting.json",
        ]
    elif route == "mangalmm_full_page":
        candidates = [source_root / source_page_id / "result.json"]
    else:  # pragma: no cover - callers validate the route first
        raise ContractError(f"Unknown OCR route: {route}")
    resolved_root = source_root.resolve()
    paths: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved_root not in resolved.parents or not resolved.is_file():
            raise ContractError(f"Missing primary source result: {candidate}")
        paths.append(resolved)
    return paths


def create_source_bindings(
    *,
    route: str,
    corpus_manifest: Path,
    source_results: Path,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise ContractError(f"Unknown OCR route: {route}")
    manifest_path = require_external_path(corpus_manifest, label="corpus manifest")
    source_root = require_external_path(source_results, label="source results")
    output = source_root / SOURCE_BINDING_FILENAME
    if output.exists():
        raise ContractError(f"Source binding already exists: {output}")
    manifest = validate_corpus_manifest(manifest_path)
    pages = []
    for page in manifest["pages"]:
        source_page_id = str(page.get("source_page_id", page["page_id"]))
        pages.append(
            {
                "page_id": page["page_id"],
                "source_page_id": source_page_id,
                "source_sha256": page["source_image"]["sha256"],
                "image_width": int(page["source_image"]["width"]),
                "image_height": int(page["source_image"]["height"]),
                "source_result_files": [
                    _source_file_record(source_root, result_path)
                    for result_path in _primary_source_result_paths(
                        route, source_root, source_page_id
                    )
                ],
            }
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": PROTOCOL_VERSION,
        "route_id": route,
        "suite_id": manifest["suite_id"],
        "corpus_manifest_canonical_sha256": canonical_sha256(
            {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        ),
        "attestation": "manual_source_result_binding",
        "pages": pages,
    }
    payload["binding_contract_sha256"] = canonical_sha256(payload)
    write_json(output, payload)
    return payload


def _validate_source_bindings(
    *,
    route: str,
    manifest: Mapping[str, Any],
    source_root: Path,
) -> Path:
    path = source_root / SOURCE_BINDING_FILENAME
    payload = read_json(path)
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_version") != PROTOCOL_VERSION
        or payload.get("route_id") != route
        or payload.get("suite_id") != manifest.get("suite_id")
        or payload.get("attestation") != "manual_source_result_binding"
    ):
        raise ContractError("Source binding identity is invalid.")
    expected_contract = require_sha256(
        payload.get("binding_contract_sha256"), label="source binding contract"
    )
    contract_payload = {
        key: value for key, value in payload.items() if key != "binding_contract_sha256"
    }
    if canonical_sha256(contract_payload) != expected_contract:
        raise ContractError("Source binding contract changed.")
    manifest_contract = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if payload.get("corpus_manifest_canonical_sha256") != manifest_contract:
        raise ContractError("Source binding corpus manifest changed.")
    bound_pages = payload.get("pages")
    if not isinstance(bound_pages, list):
        raise ContractError("Source binding pages are invalid.")
    expected_pages = {
        str(page["page_id"]): {
            "page_id": page["page_id"],
            "source_page_id": str(page.get("source_page_id", page["page_id"])),
            "source_sha256": page["source_image"]["sha256"],
            "image_width": int(page["source_image"]["width"]),
            "image_height": int(page["source_image"]["height"]),
        }
        for page in manifest["pages"]
    }
    actual_pages: dict[str, Mapping[str, Any]] = {}
    for record in bound_pages:
        if not isinstance(record, dict):
            raise ContractError("Invalid source binding page record.")
        page_id = require_safe_id(record.get("page_id"), label="source binding page ID")
        if page_id in actual_pages:
            raise ContractError(f"Duplicate source binding page: {page_id}")
        result_files = record.get("source_result_files")
        if not isinstance(result_files, list) or not result_files:
            raise ContractError(f"Source binding lacks result evidence for {page_id}.")
        source_page_id = str(record.get("source_page_id", ""))
        expected_result_paths = {
            path.relative_to(source_root.resolve()).as_posix()
            for path in _primary_source_result_paths(route, source_root, source_page_id)
        }
        actual_result_paths: set[str] = set()
        for result_record in result_files:
            if not isinstance(result_record, dict):
                raise ContractError(f"Invalid source result binding for {page_id}.")
            relative = Path(str(result_record.get("relative_path", "")))
            if relative.is_absolute() or ".." in relative.parts:
                raise ContractError("Source result binding escapes its root.")
            relative_text = relative.as_posix()
            if relative_text in actual_result_paths:
                raise ContractError(f"Duplicate source result binding: {relative_text}")
            actual_result_paths.add(relative_text)
            result_path = (source_root / relative).resolve()
            expected_sha = require_sha256(
                result_record.get("sha256"),
                label=f"source binding result {relative_text}",
            )
            if (
                source_root.resolve() not in result_path.parents
                or not result_path.is_file()
                or sha256_file(result_path) != expected_sha
            ):
                raise ContractError(f"Bound source result changed: {relative_text}")
        if actual_result_paths != expected_result_paths:
            raise ContractError(f"Source result binding set changed for {page_id}.")
        actual_pages[page_id] = {
            key: value for key, value in record.items() if key != "source_result_files"
        }
    if actual_pages != expected_pages:
        raise ContractError("Source bindings do not match the locked corpus pages.")
    return path


def _source_file_record(root: Path, path: Path) -> dict[str, Any]:
    resolved = require_external_path(path, label="route source result")
    try:
        relative = resolved.relative_to(root)
    except ValueError as exc:
        raise ContractError("Route source file escapes source results root.") from exc
    if ".." in relative.parts:
        raise ContractError("Route source file uses parent traversal.")
    return {"relative_path": relative.as_posix(), "sha256": sha256_file(resolved)}


def _normalize_unit_assets(
    source_root: Path,
    raw_assets: Any,
    used_files: list[Path],
) -> dict[str, dict[str, str]]:
    """Normalize optional visual evidence without making it part of OCR scoring."""

    if raw_assets in (None, {}):
        return {}
    if not isinstance(raw_assets, dict):
        raise ContractError("OCR unit assets must be an object.")
    normalized: dict[str, dict[str, str]] = {}
    for raw_kind, raw_record in raw_assets.items():
        kind = require_asset_kind(raw_kind)
        if isinstance(raw_record, str):
            relative = Path(raw_record)
            supplied_sha = ""
        elif isinstance(raw_record, dict):
            relative = Path(str(raw_record.get("path", "")))
            supplied_sha = str(raw_record.get("sha256", "") or "").lower()
        else:
            raise ContractError(f"Invalid OCR unit asset record: {kind}")
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ContractError(f"OCR unit asset escapes source results: {kind}")
        path = (source_root / relative).resolve()
        if source_root.resolve() not in path.parents or not path.is_file():
            raise ContractError(f"Missing OCR unit asset: {path}")
        actual_sha = sha256_file(path)
        if supplied_sha and require_sha256(supplied_sha, label=kind) != actual_sha:
            raise ContractError(f"OCR unit asset SHA-256 mismatch: {kind}")
        used_files.append(path)
        normalized[kind] = {
            "relative_path": relative.as_posix(),
            "sha256": actual_sha,
        }
    return normalized


def _detector_blocks_for_page(page: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = page["source_image"]
    return _load_detector_blocks(
        Path(page["detector_snapshot"]["path"]),
        page_id=page["page_id"],
        source_sha256=source["sha256"],
        width=int(source["width"]),
        height=int(source["height"]),
    )


def _require_source_basename(
    value: Any,
    page: Mapping[str, Any],
    *,
    label: str,
) -> None:
    supplied = PureWindowsPath(str(value or "")).name
    expected = Path(str(page["source_image"]["path"])).name
    if not supplied or supplied.casefold() != expected.casefold():
        raise ContractError(
            f"{label} is not bound to the locked source image for {page['page_id']}."
        )


def _canonical_unit(
    *,
    unit_id: str,
    detector_block_ids: Sequence[str],
    bbox_xyxy: Sequence[int],
    text: str,
    raw_region_ids: Sequence[str],
    geometry_status: str,
    semantic_role: str = "",
    processing_action: str = "",
    assets: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    return {
        "canonical_unit_id": unit_id,
        "detector_block_ids": list(detector_block_ids),
        "bbox_xyxy": list(bbox_xyxy),
        "text": str(text or ""),
        "raw_region_ids": list(raw_region_ids),
        "geometry_status": geometry_status,
        "semantic_role": semantic_role,
        "processing_action": processing_action,
        "assets": dict(assets or {}),
    }


def _import_crop_page(
    source_root: Path,
    page: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    page_id = page["page_id"]
    source_page_id = str(page.get("source_page_id", page_id))
    result_path = source_root / source_page_id / "result.json"
    payload = read_json(result_path)
    if payload.get("status") != "success" or payload.get("error"):
        status = "failure"
    else:
        status = "success"
    if str(payload.get("source_sha256", "")).lower() != page["source_image"]["sha256"]:
        raise ContractError(f"Crop source hash mismatch for {page_id}.")
    width = int(page["source_image"]["width"])
    height = int(page["source_image"]["height"])
    blocks = payload.get("blocks")
    if not isinstance(blocks, list):
        raise ContractError(f"Crop result lacks a blocks list for {page_id}.")
    detector_blocks = _detector_blocks_for_page(page)
    detector_by_id = {block["block_id"]: block for block in detector_blocks}
    seen_blocks: set[str] = set()
    used_files = [result_path]
    raw_regions: list[dict[str, Any]] = []
    units: list[dict[str, Any]] = []
    for index, raw in enumerate(blocks):
        if not isinstance(raw, dict):
            raise ContractError(f"Invalid crop block on {page_id}.")
        block_id = require_safe_id(raw.get("block_id"), label="crop block ID")
        if block_id in seen_blocks:
            raise ContractError(f"Duplicate crop block on {page_id}: {block_id}")
        seen_blocks.add(block_id)
        detector_block = detector_by_id.get(block_id)
        if detector_block is None:
            raise ContractError(f"Unknown crop detector block on {page_id}: {block_id}")
        box = _bbox(
            raw.get("xyxy"),
            label=f"crop:{page_id}:{block_id}",
            width=width,
            height=height,
        )
        if box != detector_block["bbox_xyxy"]:
            raise ContractError(f"Crop detector geometry changed on {page_id}: {block_id}")
        region_id = f"crop-{index:04d}"
        text = str(raw.get("text", "") or "")
        raw_regions.append(
            {
                "raw_region_id": region_id,
                "bbox_xyxy": box,
                "text": text,
                "detector_block_ids": [block_id],
                "geometry_source": "detector_crop",
            }
        )
        units.append(
            _canonical_unit(
                unit_id=block_id,
                detector_block_ids=[block_id],
                bbox_xyxy=box,
                text=text,
                raw_region_ids=[region_id],
                geometry_status="detector_exact",
                semantic_role=str(raw.get("semantic_role", "") or ""),
                processing_action=str(raw.get("processing_action", "") or ""),
                assets=_normalize_unit_assets(
                    source_root, raw.get("assets"), used_files
                ),
            )
        )
    missing_blocks = sorted(set(detector_by_id) - seen_blocks)
    for block_id in missing_blocks:
        detector_block = detector_by_id[block_id]
        units.append(
            _canonical_unit(
                unit_id=block_id,
                detector_block_ids=[block_id],
                bbox_xyxy=detector_block["bbox_xyxy"],
                text="",
                raw_region_ids=[],
                geometry_status="missing_detector_output",
            )
        )
    if missing_blocks:
        status = "failure"
    profile = payload.get("page_profile", {})
    performance = profile.get("performance", {}) if isinstance(profile, dict) else {}
    page_result = {
        "page_id": page_id,
        "source_sha256": page["source_image"]["sha256"],
        "image_width": width,
        "image_height": height,
        "status": status,
        "elapsed_seconds": float(payload.get("request_seconds", 0.0) or 0.0),
        "attempt_count": int(performance.get("http_attempt_count", len(units)) or 0),
        "retry_count": int(performance.get("http_retry_count", 0) or 0),
        "attempt_telemetry_complete": "http_attempt_count" in performance,
        "parser_error_count": 0,
        "length_error_count": 0,
        "raw_regions": raw_regions,
        "canonical_units": units,
        "diagnostics": {
            "ocr_status": status,
            "missing_detector_block_ids": missing_blocks,
        },
    }
    return page_result, used_files


def _import_spotting_page(
    source_root: Path,
    page: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    page_id = page["page_id"]
    source_page_id = str(page.get("source_page_id", page_id))
    fused_path = (
        source_root / "detector-fused-comparison-v1" / f"{source_page_id}.json"
    )
    geometry_path = source_root / "geometry-audit" / f"{source_page_id}.json"
    raw_path = source_root / f"{source_page_id}_spotting.json"
    fused = read_json(fused_path)
    geometry = read_json(geometry_path)
    raw = read_json(raw_path)
    width = int(page["source_image"]["width"])
    height = int(page["source_image"]["height"])
    detector_blocks = _detector_blocks_for_page(page)
    detector_by_id = {block["block_id"]: block for block in detector_blocks}
    used_files = [fused_path, geometry_path, raw_path]
    shape_hw = geometry.get("shape_hw")
    if not isinstance(shape_hw, list) or shape_hw != [height, width]:
        raise ContractError(f"Spotting image dimensions changed for {page_id}.")
    if raw.get("image_width") != width or raw.get("image_height") != height:
        raise ContractError(f"Spotting raw image dimensions changed for {page_id}.")
    _require_source_basename(raw.get("input"), page, label="Spotting input")
    spotting = geometry.get("spotting")
    if not isinstance(spotting, list):
        raise ContractError(f"Spotting geometry lacks a region list for {page_id}.")
    raw_regions: list[dict[str, Any]] = []
    for index, region in enumerate(spotting):
        if not isinstance(region, dict):
            raise ContractError(f"Invalid Spotting region on {page_id}: {index}")
        box = _bbox(
            region.get("bbox_xyxy"),
            label=f"spotting:{page_id}:{index}",
            width=width,
            height=height,
        )
        raw_regions.append(
            {
                "raw_region_id": f"spot-{index:04d}",
                "bbox_xyxy": box,
                "text": str(region.get("text", "") or ""),
                "detector_block_ids": [],
                "geometry_source": "paddle_normalized_0_1000",
            }
        )
    units: list[dict[str, Any]] = []
    claimed: set[int] = set()
    fused_blocks = fused.get("blocks")
    if not isinstance(fused_blocks, list):
        raise ContractError(f"Spotting fusion lacks a blocks list for {page_id}.")
    if fused.get("detector_block_count") != len(detector_blocks):
        raise ContractError(f"Spotting detector block count changed for {page_id}.")
    if fused.get("page") is not None and str(fused["page"]) != source_page_id:
        raise ContractError(f"Spotting fusion page changed for {page_id}.")
    seen_blocks: set[str] = set()
    for block in fused_blocks:
        if not isinstance(block, dict):
            raise ContractError(f"Invalid Spotting fused block on {page_id}.")
        block_id = require_safe_id(block.get("block_id"), label="spotting block ID")
        if block_id in seen_blocks:
            raise ContractError(f"Duplicate Spotting fused block on {page_id}: {block_id}")
        seen_blocks.add(block_id)
        detector_block = detector_by_id.get(block_id)
        if detector_block is None:
            raise ContractError(f"Unknown Spotting detector block on {page_id}: {block_id}")
        if block.get("page") is not None and str(block["page"]) != source_page_id:
            raise ContractError(f"Spotting block page changed for {page_id}: {block_id}")
        fused_box = _bbox(
            block.get("xyxy"),
            label=f"spotting-block:{page_id}:{block_id}",
            width=width,
            height=height,
        )
        if fused_box != detector_block["bbox_xyxy"]:
            raise ContractError(
                f"Spotting detector geometry changed on {page_id}: {block_id}"
            )
        raw_indices = block.get("spot_indices")
        if not isinstance(raw_indices, list) or any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in raw_indices
        ):
            raise ContractError(f"Invalid Spotting region indices on {page_id}: {block_id}")
        if len(raw_indices) != len(set(raw_indices)):
            raise ContractError(f"Duplicate Spotting region index on {page_id}: {block_id}")
        indices = list(raw_indices)
        for index in indices:
            if index < 0 or index >= len(raw_regions):
                raise ContractError(f"Invalid Spotting region index on {page_id}.")
            if index in claimed:
                raise ContractError(
                    f"Spotting region is assigned to multiple blocks on {page_id}: "
                    f"{index}"
                )
            claimed.add(index)
            raw_regions[index]["detector_block_ids"] = [block_id]
        if not _matches_normalized_segment_permutation(
            str(block.get("spotting_text", "") or ""),
            [raw_regions[index]["text"] for index in indices],
        ):
            raise ContractError(
                f"Spotting fused text differs from its raw regions on {page_id}: "
                f"{block_id}"
            )
        units.append(
            _canonical_unit(
                unit_id=block_id,
                detector_block_ids=[block_id],
                bbox_xyxy=fused_box,
                text=str(block.get("spotting_text", "") or ""),
                raw_region_ids=[f"spot-{index:04d}" for index in indices],
                geometry_status=("compound_safe" if len(indices) > 1 else "one_to_one"),
                semantic_role=str(block.get("semantic_role", "") or ""),
                processing_action=str(block.get("processing_action", "") or ""),
                assets=_normalize_unit_assets(
                    source_root, block.get("assets"), used_files
                ),
            )
        )
    for index, region in enumerate(raw_regions):
        if index in claimed:
            continue
        units.append(
            _canonical_unit(
                unit_id=f"spot-extra-{index:04d}",
                detector_block_ids=[],
                bbox_xyxy=region["bbox_xyxy"],
                text=region["text"],
                raw_region_ids=[region["raw_region_id"]],
                geometry_status="full_page_only",
                assets=_normalize_unit_assets(
                    source_root,
                    spotting[index].get("assets"),
                    used_files,
                ),
            )
        )
    if seen_blocks != set(detector_by_id):
        missing = sorted(set(detector_by_id) - seen_blocks)
        raise ContractError(f"Spotting fusion is missing detector blocks on {page_id}: {missing}")
    mapped_count = sum(bool(block.get("spot_indices")) for block in fused_blocks)
    if fused.get("mapped_detector_count") != mapped_count:
        raise ContractError(f"Spotting mapped detector count changed for {page_id}.")
    finish_reason = str(raw.get("finish_reason", "") or "")
    if finish_reason not in {"stop", "length"}:
        raise ContractError(f"Invalid Spotting finish reason on {page_id}.")
    page_result = {
        "page_id": page_id,
        "source_sha256": page["source_image"]["sha256"],
        "image_width": width,
        "image_height": height,
        "status": "success" if finish_reason != "length" else "failure",
        "elapsed_seconds": float(raw.get("elapsed_seconds", 0.0) or 0.0),
        "attempt_count": 0,
        "retry_count": 0,
        "attempt_telemetry_complete": False,
        "parser_error_count": 0,
        "length_error_count": int(finish_reason == "length"),
        "raw_regions": raw_regions,
        "canonical_units": units,
        "diagnostics": {
            "finish_reason": finish_reason,
            "attempt_telemetry": "historical_final_response_only",
            "detector_block_count": int(fused.get("detector_block_count", 0)),
            "mapped_detector_count": int(fused.get("mapped_detector_count", 0)),
            "spotting_region_count": len(raw_regions),
        },
    }
    return page_result, used_files


def _reading_order(regions: list[dict[str, Any]], direction: str) -> list[dict[str, Any]]:
    if direction.lower() == "vertical":
        return sorted(
            regions,
            key=lambda region: (
                -((region["bbox_xyxy"][0] + region["bbox_xyxy"][2]) / 2),
                (region["bbox_xyxy"][1] + region["bbox_xyxy"][3]) / 2,
                region["raw_region_id"],
            ),
        )
    return sorted(
        regions,
        key=lambda region: (
            (region["bbox_xyxy"][1] + region["bbox_xyxy"][3]) / 2,
            (region["bbox_xyxy"][0] + region["bbox_xyxy"][2]) / 2,
            region["raw_region_id"],
        ),
    )


def _assign_regions_to_detector(
    regions: list[dict[str, Any]],
    blocks: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    assignments = {block["block_id"]: [] for block in blocks}
    extras: list[dict[str, Any]] = []
    ambiguous: list[dict[str, Any]] = []
    for region in regions:
        candidates = sorted(
            (
                (_geometry_score(region["bbox_xyxy"], block["bbox_xyxy"]), block)
                for block in blocks
            ),
            key=lambda item: (item[0], -item[1]["block_index"]),
            reverse=True,
        )
        if not candidates or candidates[0][0] < 0.50:
            extras.append(region)
            continue
        close = [item for item in candidates if item[0] >= 0.50]
        if len(close) > 1 and close[1][0] >= candidates[0][0] - 0.05:
            ambiguous.append(
                {
                    "raw_region_id": region["raw_region_id"],
                    "candidate_block_ids": [item[1]["block_id"] for item in close],
                    "reason": "one_region_multiple_detector_blocks",
                }
            )
            continue
        block_id = candidates[0][1]["block_id"]
        region["detector_block_ids"] = [block_id]
        assignments[block_id].append(region)
    return assignments, extras, ambiguous


def _import_manga_page(
    source_root: Path,
    page: Mapping[str, Any],
) -> tuple[dict[str, Any], list[Path]]:
    page_id = page["page_id"]
    source_page_id = str(page.get("source_page_id", page_id))
    result_path = source_root / source_page_id / "result.json"
    payload = read_json(result_path)
    width = int(page["source_image"]["width"])
    height = int(page["source_image"]["height"])
    used_files = [result_path]
    blocks = _detector_blocks_for_page(page)
    _require_source_basename(payload.get("image"), page, label="MangaLMM input")
    if payload.get("detector_block_count") is not None and payload.get(
        "detector_block_count"
    ) != len(blocks):
        raise ContractError(f"MangaLMM detector block count changed for {page_id}.")
    regions = payload.get("regions")
    if not isinstance(regions, list):
        raise ContractError(f"MangaLMM result lacks a regions list for {page_id}.")
    raw_regions: list[dict[str, Any]] = []
    for index, raw in enumerate(regions):
        if not isinstance(raw, dict):
            raise ContractError(f"Invalid MangaLMM region on {page_id}: {index}")
        box = _bbox(
            raw.get("bbox_xyxy"),
            label=f"manga:{page_id}:{index}",
            width=width,
            height=height,
        )
        raw_regions.append(
            {
                "raw_region_id": f"manga-{index:04d}",
                "bbox_xyxy": box,
                "text": str(raw.get("text", "") or ""),
                "detector_block_ids": [],
                "geometry_source": "mangalmm_full_page",
                "assets": _normalize_unit_assets(
                    source_root, raw.get("assets"), used_files
                ),
            }
        )
    assignments, extras, ambiguous = _assign_regions_to_detector(raw_regions, blocks)
    units: list[dict[str, Any]] = []
    for block in blocks:
        assigned = _reading_order(assignments[block["block_id"]], block["direction"])
        if not assigned:
            continue
        units.append(
            _canonical_unit(
                unit_id=block["block_id"],
                detector_block_ids=[block["block_id"]],
                bbox_xyxy=block["bbox_xyxy"],
                text="\n".join(region["text"] for region in assigned if region["text"]),
                raw_region_ids=[region["raw_region_id"] for region in assigned],
                geometry_status=("compound_safe" if len(assigned) > 1 else "one_to_one"),
                assets={
                    f"region-{region_index:04d}-{kind}": record
                    for region_index, region in enumerate(assigned)
                    for kind, record in region.get("assets", {}).items()
                },
            )
        )
    for region in extras:
        units.append(
            _canonical_unit(
                unit_id=f"manga-extra-{region['raw_region_id'].split('-')[-1]}",
                detector_block_ids=[],
                bbox_xyxy=region["bbox_xyxy"],
                text=region["text"],
                raw_region_ids=[region["raw_region_id"]],
                geometry_status="full_page_only",
                assets=region.get("assets", {}),
            )
        )
    regions_by_id = {region["raw_region_id"]: region for region in raw_regions}
    for item in ambiguous:
        region = regions_by_id[str(item["raw_region_id"])]
        units.append(
            _canonical_unit(
                unit_id=(
                    "manga-ambiguous-"
                    f"{region['raw_region_id'].split('-')[-1]}"
                ),
                detector_block_ids=[],
                bbox_xyxy=region["bbox_xyxy"],
                text=region["text"],
                raw_region_ids=[region["raw_region_id"]],
                geometry_status="ambiguous_multi_detector",
                semantic_role="ambiguous",
                processing_action="review",
                assets=region.get("assets", {}),
            )
        )
    attempts = payload.get("attempts")
    if not isinstance(attempts, list) or not attempts or any(
        not isinstance(attempt, dict) for attempt in attempts
    ):
        raise ContractError(f"MangaLMM result lacks valid attempts for {page_id}.")
    for index, attempt in enumerate(attempts):
        finish_reason = str(attempt.get("finish_reason", "") or "")
        has_explicit_error = bool(
            attempt.get("parser_error_code") or attempt.get("error")
        )
        if finish_reason not in {"stop", "length"} and not has_explicit_error:
            raise ContractError(
                f"MangaLMM attempt lacks an outcome on {page_id}: {index}"
            )
    parser_errors = sum(
        bool(attempt.get("parser_error_code"))
        for attempt in attempts
        if isinstance(attempt, dict)
    )
    length_errors = sum(
        attempt.get("finish_reason") == "length"
        for attempt in attempts
        if isinstance(attempt, dict)
    )
    failure = payload.get("failure")
    if failure is not None and not isinstance(failure, str):
        raise ContractError(f"Invalid MangaLMM failure field on {page_id}.")
    reported_merge_split = payload.get("merge_split_diagnostics", [])
    if not isinstance(reported_merge_split, list):
        raise ContractError(f"Invalid MangaLMM merge/split diagnostics on {page_id}.")
    page_result = {
        "page_id": page_id,
        "source_sha256": page["source_image"]["sha256"],
        "image_width": width,
        "image_height": height,
        "status": "failure" if failure else "success",
        "elapsed_seconds": float(payload.get("elapsed_seconds", 0.0) or 0.0),
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "attempt_telemetry_complete": True,
        "parser_error_count": int(parser_errors),
        "length_error_count": int(length_errors),
        "raw_regions": raw_regions,
        "canonical_units": units,
        "diagnostics": {
            "failure": failure,
            "ambiguous_regions": ambiguous,
            "reported_merge_split": reported_merge_split,
            "detector_block_count": len(blocks),
            "matched_detector_count": len([unit for unit in units if unit["detector_block_ids"]]),
        },
    }
    return page_result, used_files


IMPORTERS = {
    "paddle_crop": _import_crop_page,
    "paddle_spotting_full_page": _import_spotting_page,
    "mangalmm_full_page": _import_manga_page,
}


def import_existing_run(
    *,
    route: str,
    corpus_manifest: Path,
    source_results: Path,
    runtime_contract: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if route not in ROUTES:
        raise ContractError(f"Unknown OCR route: {route}")
    manifest_path = require_external_path(corpus_manifest, label="corpus manifest")
    source_root = require_external_path(source_results, label="source results")
    output = require_external_path(output_dir, label="normalized run output")
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"Normalized run output must be empty: {output}")
    manifest = validate_corpus_manifest(manifest_path)
    source_binding_path = _validate_source_bindings(
        route=route,
        manifest=manifest,
        source_root=source_root,
    )
    runtime = _runtime_contract(runtime_contract, route)
    importer = IMPORTERS[route]
    pages: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = [
        _source_file_record(source_root, source_binding_path)
    ]
    for page in manifest["pages"]:
        page_result, used_files = importer(source_root, page)
        pages.append(page_result)
        source_files.extend(_source_file_record(source_root, path) for path in used_files)
    source_file_map: dict[str, dict[str, Any]] = {}
    for record in source_files:
        relative = str(record["relative_path"])
        previous = source_file_map.get(relative)
        if previous is not None and previous != record:
            raise ContractError(f"Conflicting source evidence record: {relative}")
        source_file_map[relative] = record
    source_files = sorted(source_file_map.values(), key=lambda item: item["relative_path"])
    result = {
        "schema_version": RUN_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": manifest["suite_id"],
        "route_id": route,
        "created_at": utc_now(),
        "corpus_manifest_file_sha256": sha256_file(manifest_path),
        "corpus_manifest_path": str(manifest_path),
        "source_results_root": str(source_root),
        "source_files": source_files,
        "runtime_contract": runtime,
        "runtime_contract_path": str(
            require_external_path(runtime_contract, label="runtime contract")
        ),
        "runtime_contract_file_sha256": sha256_file(
            require_external_path(runtime_contract, label="runtime contract")
        ),
        "pages": pages,
    }
    result["normalized_contract_sha256"] = canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / RUN_FILENAME, result)
    return result


def validate_run(path: Path) -> dict[str, Any]:
    run_path = require_external_path(path, label="normalized run")
    payload = read_json(run_path)
    if payload.get("schema_version") != RUN_SCHEMA_VERSION:
        raise ContractError("Unsupported normalized run schema.")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ContractError("Normalized run protocol mismatch.")
    route = str(payload.get("route_id", ""))
    if route not in ROUTES:
        raise ContractError("Normalized run route is invalid.")
    runtime = payload.get("runtime_contract")
    if not isinstance(runtime, dict) or runtime.get("backend") != "llama.cpp":
        raise ContractError("Normalized run is not a managed llama.cpp result.")
    expected_contract = require_sha256(
        payload.get("normalized_contract_sha256"),
        label="normalized run contract",
    )
    contract_payload = {
        key: value for key, value in payload.items() if key != "normalized_contract_sha256"
    }
    if canonical_sha256(contract_payload) != expected_contract:
        raise ContractError("Normalized run was modified.")
    manifest_path = require_external_path(
        Path(str(payload.get("corpus_manifest_path", ""))),
        label="normalized run corpus manifest",
    )
    if sha256_file(manifest_path) != payload.get("corpus_manifest_file_sha256"):
        raise ContractError("Normalized run corpus manifest changed.")
    manifest = validate_corpus_manifest(manifest_path)
    if payload.get("suite_id") != manifest.get("suite_id"):
        raise ContractError("Normalized run suite ID differs from its corpus manifest.")
    runtime_path = require_external_path(
        Path(str(payload.get("runtime_contract_path", ""))),
        label="normalized run runtime contract",
    )
    if sha256_file(runtime_path) != payload.get("runtime_contract_file_sha256"):
        raise ContractError("Normalized run runtime contract changed.")
    if _runtime_contract(runtime_path, route) != runtime:
        raise ContractError("Embedded runtime contract differs from source evidence.")
    source_root = require_external_path(
        Path(str(payload.get("source_results_root", ""))),
        label="normalized run source root",
    )
    source_records = payload.get("source_files")
    if not isinstance(source_records, list) or not source_records:
        raise ContractError("Normalized run has no source evidence records.")
    source_record_map: dict[str, str] = {}
    for record in source_records:
        if not isinstance(record, dict):
            raise ContractError("Invalid normalized source file record.")
        relative = Path(str(record.get("relative_path", "")))
        if relative.is_absolute() or ".." in relative.parts:
            raise ContractError("Normalized source path escapes source root.")
        source_path = (source_root / relative).resolve()
        if source_root.resolve() not in source_path.parents:
            raise ContractError("Normalized source path escapes source root.")
        expected_source_sha = require_sha256(
            record.get("sha256"), label=f"source evidence {relative}"
        )
        if relative.as_posix() in source_record_map:
            raise ContractError(f"Duplicate normalized source record: {relative}")
        source_record_map[relative.as_posix()] = expected_source_sha
        if not source_path.is_file() or sha256_file(source_path) != expected_source_sha:
            raise ContractError(f"Normalized source evidence changed: {relative}")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ContractError("Normalized run has no pages.")
    manifest_pages = {str(page["page_id"]): page for page in manifest["pages"]}
    seen_pages: set[str] = set()
    for page in pages:
        if not isinstance(page, dict):
            raise ContractError("Normalized run page must be an object.")
        page_id = require_safe_id(page.get("page_id"), label="normalized page ID")
        if page_id in seen_pages:
            raise ContractError(f"Duplicate normalized page: {page_id}")
        seen_pages.add(page_id)
        manifest_page = manifest_pages.get(page_id)
        if manifest_page is None:
            raise ContractError(f"Normalized run contains an unexpected page: {page_id}")
        width = int(manifest_page["source_image"]["width"])
        height = int(manifest_page["source_image"]["height"])
        if (
            page.get("source_sha256") != manifest_page["source_image"]["sha256"]
            or int(page.get("image_width", 0) or 0) != width
            or int(page.get("image_height", 0) or 0) != height
        ):
            raise ContractError(f"Normalized page identity changed: {page_id}")
        if page.get("status") not in {"success", "failure"}:
            raise ContractError(f"Invalid normalized page status: {page_id}")
        if not isinstance(page.get("attempt_telemetry_complete"), bool):
            raise ContractError(
                f"Normalized page attempt telemetry state is invalid: {page_id}"
            )
        elapsed_seconds = page.get("elapsed_seconds")
        if (
            not isinstance(elapsed_seconds, (int, float))
            or isinstance(elapsed_seconds, bool)
            or not math.isfinite(float(elapsed_seconds))
            or float(elapsed_seconds) < 0
        ):
            raise ContractError(
                f"Invalid normalized page counter {page_id}:elapsed_seconds"
            )
        integer_counters: dict[str, int] = {}
        for counter in (
            "attempt_count",
            "retry_count",
            "parser_error_count",
            "length_error_count",
        ):
            value = page.get(counter)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ContractError(
                    f"Invalid normalized page counter {page_id}:{counter}"
                )
            integer_counters[counter] = value
        if integer_counters["retry_count"] > integer_counters["attempt_count"]:
            raise ContractError(f"Retry count exceeds attempts on {page_id}.")
        units = page.get("canonical_units")
        raw_regions = page.get("raw_regions")
        if not isinstance(units, list) or not isinstance(raw_regions, list):
            raise ContractError(f"Normalized page {page_id} lacks OCR regions.")
        known_detector_ids = {
            block["block_id"] for block in _detector_blocks_for_page(manifest_page)
        }
        raw_ids: set[str] = set()
        for raw_region in raw_regions:
            if not isinstance(raw_region, dict):
                raise ContractError(f"Invalid raw OCR region on {page_id}.")
            raw_id = require_safe_id(
                raw_region.get("raw_region_id"), label="raw OCR region ID"
            )
            if raw_id in raw_ids:
                raise ContractError(f"Duplicate raw OCR region on {page_id}: {raw_id}")
            raw_ids.add(raw_id)
            _bbox(
                raw_region.get("bbox_xyxy"),
                label=f"raw OCR region {page_id}:{raw_id}",
                width=width,
                height=height,
            )
            raw_detector_ids = raw_region.get("detector_block_ids")
            if not isinstance(raw_detector_ids, list) or len(raw_detector_ids) != len(
                set(raw_detector_ids)
            ):
                raise ContractError(
                    f"Invalid raw OCR detector IDs on {page_id}:{raw_id}."
                )
            if any(
                str(detector_id) not in known_detector_ids
                for detector_id in raw_detector_ids
            ):
                raise ContractError(
                    f"Raw OCR region references unknown detector geometry: {page_id}:{raw_id}"
                )
        unit_ids: set[str] = set()
        referenced_raw_ids: set[str] = set()
        claimed_detector_ids: set[str] = set()
        for unit in units:
            if not isinstance(unit, dict):
                raise ContractError(f"Invalid canonical unit on {page_id}.")
            unit_id = require_safe_id(
                unit.get("canonical_unit_id"), label="canonical unit ID"
            )
            if unit_id in unit_ids:
                raise ContractError(f"Duplicate canonical unit on {page_id}: {unit_id}")
            unit_ids.add(unit_id)
            _bbox(
                unit.get("bbox_xyxy"),
                label=f"canonical OCR unit {page_id}:{unit_id}",
                width=width,
                height=height,
            )
            referenced_raw = unit.get("raw_region_ids")
            if not isinstance(referenced_raw, list) or any(
                str(raw_id) not in raw_ids for raw_id in referenced_raw
            ):
                raise ContractError(
                    f"Canonical unit references unknown raw regions: {page_id}:{unit_id}"
                )
            if len(referenced_raw) != len(set(referenced_raw)) or any(
                str(raw_id) in referenced_raw_ids for raw_id in referenced_raw
            ):
                raise ContractError(
                    f"Canonical units reuse raw OCR regions: {page_id}:{unit_id}"
                )
            referenced_raw_ids.update(str(raw_id) for raw_id in referenced_raw)
            unit_detector_ids = unit.get("detector_block_ids")
            if not isinstance(unit_detector_ids, list) or len(unit_detector_ids) != len(
                set(unit_detector_ids)
            ):
                raise ContractError(
                    f"Canonical unit detector IDs are invalid: {page_id}:{unit_id}"
                )
            for detector_id in unit_detector_ids:
                detector_id = require_safe_id(
                    detector_id, label="canonical detector block ID"
                )
                if detector_id not in known_detector_ids:
                    raise ContractError(
                        f"Canonical unit references unknown detector geometry: "
                        f"{page_id}:{unit_id}"
                    )
                if detector_id in claimed_detector_ids:
                    raise ContractError(
                        f"Canonical units reuse detector geometry: {page_id}:{detector_id}"
                    )
                claimed_detector_ids.add(detector_id)
            role = str(unit.get("semantic_role", "") or "")
            action = str(unit.get("processing_action", "") or "")
            if role and role not in ALLOWED_ROLES:
                raise ContractError(f"Invalid OCR semantic role: {page_id}:{unit_id}")
            if action and action not in ALLOWED_ACTIONS:
                raise ContractError(f"Invalid OCR processing action: {page_id}:{unit_id}")
            assets = unit.get("assets", {})
            if not isinstance(assets, dict):
                raise ContractError(f"Invalid OCR assets: {page_id}:{unit_id}")
            for kind, asset in assets.items():
                require_asset_kind(kind)
                if not isinstance(asset, dict):
                    raise ContractError(f"Invalid OCR asset record: {page_id}:{unit_id}")
                relative = Path(str(asset.get("relative_path", "")))
                expected_sha = require_sha256(
                    asset.get("sha256"), label=f"OCR asset {page_id}:{unit_id}:{kind}"
                )
                if source_record_map.get(relative.as_posix()) != expected_sha:
                    raise ContractError(
                        f"OCR asset is not locked source evidence: {page_id}:{unit_id}:{kind}"
                    )
    if seen_pages != set(manifest_pages):
        missing = sorted(set(manifest_pages) - seen_pages)
        raise ContractError(f"Normalized run is missing corpus pages: {missing}")
    return payload


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", str(value or ""))
    return "".join(
        character
        for character in normalized
        if not character.isspace() and not unicodedata.category(character).startswith("P")
    )


def _matches_normalized_segment_permutation(
    fused: str,
    segments: Sequence[str],
) -> bool:
    target = normalize_text(fused)
    normalized_segments = tuple(
        segment for value in segments if (segment := normalize_text(value))
    )
    if sum(len(segment) for segment in normalized_segments) != len(target):
        return False
    memo: set[tuple[int, tuple[int, ...]]] = set()

    def walk(offset: int, remaining: tuple[int, ...]) -> bool:
        if not remaining:
            return offset == len(target)
        state = (offset, remaining)
        if state in memo:
            return False
        memo.add(state)
        tried: set[str] = set()
        for position, segment_index in enumerate(remaining):
            segment = normalized_segments[segment_index]
            if segment in tried or not target.startswith(segment, offset):
                continue
            tried.add(segment)
            if walk(
                offset + len(segment),
                remaining[:position] + remaining[position + 1 :],
            ):
                return True
        return False

    return walk(0, tuple(range(len(normalized_segments))))


def levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def normalized_character_accuracy(truth: str, candidate: str) -> float:
    expected = normalize_text(truth)
    actual = normalize_text(candidate)
    if not expected:
        return 1.0 if not actual else 0.0
    distance = levenshtein_distance(expected, actual)
    return max(0.0, 1.0 - distance / len(expected))


def _truth_regions(pages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in pages:
        for region in page["regions"]:
            rows.append(
                {
                    **region,
                    "page_id": page["page_id"],
                    "language": page["language"],
                    "split": page["split"],
                    "source_asset": page["source_asset"],
                }
            )
    return rows


def _route_pages(run: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(page["page_id"]): page
        for page in run.get("pages", [])
        if isinstance(page, dict)
    }


def _match_truth_to_units(
    truth_region: Mapping[str, Any],
    units: Sequence[Mapping[str, Any]],
    *,
    detectorless_matches: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[Mapping[str, Any]], str]:
    detector_ids = set(str(value) for value in truth_region.get("detector_block_ids", []))
    if detector_ids:
        safe_matches = [
            unit
            for unit in units
            if (
                unit.get("detector_block_ids")
                and set(str(value) for value in unit["detector_block_ids"])
                <= detector_ids
            )
        ]
        if not safe_matches:
            has_unsafe_intersection = any(
                detector_ids.intersection(
                    str(value) for value in unit.get("detector_block_ids", [])
                )
                for unit in units
            )
            return (
                [],
                "ambiguous_cross_truth_unit"
                if has_unsafe_intersection
                else "missing_detector_unit",
            )
        covered = set().union(
            *(
                set(str(value) for value in unit.get("detector_block_ids", []))
                for unit in safe_matches
            )
        )
        if not detector_ids.issubset(covered):
            return safe_matches, "detector_partial"
        status = (
            "detector_exact"
            if len(safe_matches) == 1
            else "detector_compound"
        )
        return safe_matches, status
    if detectorless_matches is not None:
        matches = list(detectorless_matches)
        if not matches:
            return [], "missing_full_page_region"
        return matches, (
            "full_page_geometry" if len(matches) == 1 else "full_page_compound"
        )
    scored = sorted(
        (
            (_geometry_score(truth_region["bbox_xyxy"], unit["bbox_xyxy"]), unit)
            for unit in units
            if not unit.get("detector_block_ids")
        ),
        key=lambda item: item[0],
        reverse=True,
    )
    matches = [item[1] for item in scored if item[0] >= 0.50]
    if not matches:
        return [], "missing_full_page_region"
    return matches, "full_page_geometry" if len(matches) == 1 else "full_page_compound"


def _assign_detectorless_units_to_human_truth(
    truth_regions: Sequence[Mapping[str, Any]],
    units: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    human_regions = [
        region
        for region in truth_regions
        if region.get("region_source") == "human_extra"
        and not region.get("detector_block_ids")
    ]
    assignments = {
        str(region["truth_region_id"]): [] for region in human_regions
    }
    for unit in units:
        if unit.get("detector_block_ids"):
            continue
        scores = sorted(
            (
                (
                    _geometry_score(region["bbox_xyxy"], unit["bbox_xyxy"]),
                    str(region["truth_region_id"]),
                )
                for region in human_regions
            ),
            key=lambda item: (-item[0], item[1]),
        )
        if not scores or scores[0][0] < 0.50:
            continue
        if len(scores) > 1 and math.isclose(
            scores[0][0], scores[1][0], rel_tol=0.0, abs_tol=1e-9
        ):
            continue
        assignments[scores[0][1]].append(unit)
    return assignments


def _blind_geometry_status(value: Any) -> str:
    status = str(value or "").lower()
    if "ambiguous" in status:
        return "ambiguous"
    if "missing" in status or "partial" in status:
        return "missing_or_partial"
    if "compound" in status:
        return "compound"
    if status in {"detector_exact", "one_to_one", "full_page_geometry"}:
        return "matched"
    if status == "full_page_only":
        return "unmatched_extra"
    return "other"


def _join_unit_text(units: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(str(unit.get("text", "") or "") for unit in units if unit.get("text"))


def _review_headers() -> list[str]:
    headers = list(REVIEW_ID_COLUMNS)
    for label in BLIND_LABELS:
        headers.extend(f"{label}_{field}" for field in REVIEW_ROUTE_FIELDS)
    return headers


def _empty_review_decisions(row: dict[str, str], label: str) -> None:
    row[f"{label}_transcription_correct"] = ""
    row[f"{label}_semantic_correct"] = ""
    row[f"{label}_role_action_correct"] = ""
    row[f"{label}_merge_split_error"] = ""
    row[f"{label}_destructive_edit"] = ""
    row[f"{label}_false_positive"] = ""
    row[f"{label}_notes"] = ""


def _cluster_extra_units(
    extras: list[tuple[str, str, str, Mapping[str, Any]]],
) -> list[tuple[str, dict[str, Mapping[str, Any]]]]:
    clusters: list[tuple[str, dict[str, Mapping[str, Any]]]] = []
    for page_id, label, _route, unit in extras:
        best_index = -1
        best_score = 0.0
        for index, (cluster_page_id, cluster) in enumerate(clusters):
            if cluster_page_id != page_id:
                continue
            if label in cluster:
                continue
            score = max(
                _geometry_score(unit["bbox_xyxy"], candidate["bbox_xyxy"])
                for candidate in cluster.values()
            )
            if score >= 0.50 and score > best_score:
                best_index = index
                best_score = score
        if best_index < 0:
            clusters.append((page_id, {label: unit}))
        else:
            clusters[best_index][1][label] = unit
    return clusters


def _copy_review_asset(
    source: Path,
    output: Path,
    relative: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    destination = output / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    if expected_sha256 is not None:
        expected = require_sha256(expected_sha256, label=f"review asset {relative}")
        if sha256_file(destination) != expected:
            raise ContractError(f"Review asset changed while copying: {relative}")
    return relative.as_posix()


def _locked_asset_sha256(asset_hashes: Mapping[str, str], relative: str) -> str:
    digest = asset_hashes.get(relative)
    if digest is None:
        raise ContractError(f"Truth asset is absent from its lock: {relative}")
    return require_sha256(digest, label=f"locked truth asset {relative}")


def _copy_candidate_assets(
    *,
    units: Sequence[Mapping[str, Any]],
    source_root: Path,
    output: Path,
    label: str,
    page_id: str,
    row_id: str,
) -> dict[str, str]:
    copied: dict[str, str] = {}
    public_row_id = require_safe_id(row_id, label="review row ID")
    for unit_index, unit in enumerate(units):
        unit_id = require_safe_id(
            unit.get("canonical_unit_id"), label="candidate asset unit ID"
        )
        public_unit_id = f"unit-{unit_index:04d}"
        assets = unit.get("assets", {})
        if not isinstance(assets, dict):
            raise ContractError(f"Invalid candidate assets: {page_id}:{unit_id}")
        for kind, record in assets.items():
            require_asset_kind(kind)
            if not isinstance(record, dict):
                raise ContractError(f"Invalid candidate asset record: {page_id}:{unit_id}")
            source_relative = Path(str(record.get("relative_path", "")))
            if source_relative.is_absolute() or ".." in source_relative.parts:
                raise ContractError("Candidate asset path escapes source results.")
            source = (source_root / source_relative).resolve()
            if source_root.resolve() not in source.parents or not source.is_file():
                raise ContractError(f"Missing candidate asset: {source}")
            if sha256_file(source) != record.get("sha256"):
                raise ContractError(f"Candidate asset changed: {source_relative}")
            suffix = source.suffix.lower() or ".bin"
            destination_relative = (
                Path("assets")
                / "candidates"
                / label
                / page_id
                / public_row_id
                / public_unit_id
                / f"{kind}{suffix}"
            )
            key = f"{public_unit_id}:{kind}"
            copied[key] = _copy_review_asset(
                source,
                output,
                destination_relative,
                expected_sha256=str(record.get("sha256", "")),
            )
    return copied


def _review_asset_records(review_root: Path) -> list[dict[str, str]]:
    assets_root = review_root / "assets"
    if not assets_root.is_dir() or assets_root.is_symlink():
        raise ContractError("Blind review assets directory is missing or invalid.")
    resolved_assets = assets_root.resolve()
    records: list[dict[str, str]] = []
    for path in sorted(assets_root.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ContractError(f"Blind review asset cannot be a symlink: {path}")
        if not path.is_file():
            continue
        resolved = path.resolve()
        if resolved_assets not in resolved.parents:
            raise ContractError(f"Blind review asset escapes its root: {path}")
        records.append(
            {
                "relative_path": path.relative_to(review_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    if not records:
        raise ContractError("Blind review contains no visual evidence assets.")
    return records


def _write_review_csv(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_review_headers())
        writer.writeheader()
        writer.writerows(
            {
                column: _csv_encode_cell(row.get(column, ""))
                for column in _review_headers()
            }
            for row in rows
        )


def _review_locked_content_sha256(rows: Sequence[Mapping[str, str]]) -> str:
    locked_columns = [
        column
        for column in _review_headers()
        if not any(
            column.endswith(f"_{suffix}") for suffix in REVIEW_DECISION_SUFFIXES
        )
    ]
    payload = [
        {column: str(row.get(column, "")) for column in locked_columns}
        for row in rows
    ]
    return canonical_sha256(payload)


def _run_statistics(run: Mapping[str, Any]) -> dict[str, Any]:
    pages = [page for page in run.get("pages", []) if isinstance(page, dict)]
    return {
        "page_count": len(pages),
        "successful_page_count": sum(page.get("status") == "success" for page in pages),
        "elapsed_seconds": sum(float(page.get("elapsed_seconds", 0.0) or 0.0) for page in pages),
        "attempt_count": sum(int(page.get("attempt_count", 0) or 0) for page in pages),
        "retry_count": sum(int(page.get("retry_count", 0) or 0) for page in pages),
        "attempt_telemetry_incomplete_page_count": sum(
            page.get("attempt_telemetry_complete") is not True for page in pages
        ),
        "parser_error_count": sum(
            int(page.get("parser_error_count", 0) or 0) for page in pages
        ),
        "length_error_count": sum(
            int(page.get("length_error_count", 0) or 0) for page in pages
        ),
        "raw_region_count": sum(len(page.get("raw_regions", [])) for page in pages),
        "canonical_unit_count": sum(
            len(page.get("canonical_units", [])) for page in pages
        ),
        "runtime_fingerprint_sha256": str(
            run.get("runtime_contract", {}).get("fingerprint_sha256", "")
        ),
        "page_status": {
            str(page.get("page_id", "")): str(page.get("status", ""))
            for page in pages
        },
    }


def _review_html(rows: Sequence[Mapping[str, str]]) -> str:
    cards: list[str] = []
    for row in rows:
        route_cards = []
        for label in BLIND_LABELS:
            try:
                candidate_assets = json.loads(row[f"{label}_assets_json"] or "{}")
            except json.JSONDecodeError:
                candidate_assets = {}
            asset_cards: list[str] = []
            if isinstance(candidate_assets, dict):
                for asset_name, asset_path in candidate_assets.items():
                    escaped_name = html.escape(str(asset_name))
                    escaped_path = html.escape(str(asset_path), quote=True)
                    if Path(str(asset_path)).suffix.lower() in {
                        ".png",
                        ".jpg",
                        ".jpeg",
                        ".webp",
                        ".bmp",
                        ".gif",
                    }:
                        asset_cards.append(
                            f"<figure><figcaption>{escaped_name}</figcaption>"
                            f"<img src='{escaped_path}' alt='{escaped_name}'></figure>"
                        )
                    else:
                        asset_cards.append(
                            f"<p><a href='{escaped_path}'>{escaped_name}</a></p>"
                        )
            route_cards.append(
                "<section class='candidate'>"
                f"<h3>{label}</h3>"
                f"<pre>{html.escape(row[f'{label}_text'])}</pre>"
                f"<p>bbox: {html.escape(row[f'{label}_bbox_xyxy'])}</p>"
                f"<p>geometry: {html.escape(row[f'{label}_geometry_status'])}</p>"
                f"<p>role/action: {html.escape(row[f'{label}_semantic_role'])} / "
                f"{html.escape(row[f'{label}_processing_action'])}</p>"
                + "".join(asset_cards)
                + "</section>"
            )
        crop = html.escape(row["source_crop"])
        page = html.escape(row["source_page"])
        cards.append(
            "<article class='row'>"
            f"<h2>#{row['row_number']} {html.escape(row['page_id'])} / "
            f"{html.escape(row['truth_region_id'])}</h2>"
            f"<details><summary>원본 페이지</summary><img src='{page}' alt='source page'></details>"
            f"<img src='{crop}' alt='source crop'>"
            f"<p><strong>Codex truth:</strong> {html.escape(row['truth_transcription'])}</p>"
            f"<p>truth bbox: {html.escape(row['truth_bbox_xyxy'])}</p>"
            f"<p>role/action: {html.escape(row['truth_semantic_role'])} / "
            f"{html.escape(row['truth_processing_action'])}</p>"
            "<div class='candidates'>" + "".join(route_cards) + "</div>"
            "</article>"
        )
    return """<!doctype html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>OCR three-way blind review</title>
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#f5f5f5;color:#111}
.row{background:white;padding:18px;margin:0 0 22px;border-radius:12px}
.row img{max-width:520px;max-height:720px;border:1px solid #bbb}
.candidates{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}
.candidate{border:1px solid #ccc;padding:10px;border-radius:8px;overflow:auto}
pre{white-space:pre-wrap;word-break:break-word}
@media(max-width:900px){.candidates{grid-template-columns:1fr}}
</style></head><body>
<h1>세 OCR A/B/C 블라인드 검수</h1>
<p>후보명과 속도는 숨겨져 있습니다. 판정은 region-review.csv에 기록합니다.</p>
""" + "".join(cards) + "</body></html>\n"


def make_review(
    *,
    truth_dir: Path,
    runs: Sequence[Path],
    output_dir: Path,
) -> dict[str, Any]:
    truth = require_external_path(truth_dir, label="truth directory")
    output = require_external_path(output_dir, label="review output")
    if output.exists() and any(output.iterdir()):
        raise ContractError(f"Review output must be empty: {output}")
    truth_lock, truth_pages = validate_locked_truth(truth)
    truth_asset_hashes = {
        str(record["relative_path"]): str(record["sha256"])
        for record in truth_lock.get("asset_files", [])
        if isinstance(record, dict)
    }
    loaded_runs = [validate_run(path) for path in runs]
    by_route = {run["route_id"]: run for run in loaded_runs}
    if set(by_route) != set(ROUTES) or len(loaded_runs) != len(ROUTES):
        raise ContractError("Exactly one normalized run for every OCR route is required.")
    suite_id = truth_lock["suite_id"]
    if any(run.get("suite_id") != suite_id for run in loaded_runs):
        raise ContractError("Truth and normalized run suite IDs differ.")
    if any(
        run.get("corpus_manifest_file_sha256")
        != truth_lock.get("corpus_manifest_file_sha256")
        for run in loaded_runs
    ):
        raise ContractError("Truth and normalized runs do not use the same corpus manifest.")
    shuffled = list(ROUTES)
    secrets.SystemRandom().shuffle(shuffled)
    label_to_route = dict(zip(BLIND_LABELS, shuffled, strict=True))
    route_to_label = {route: label for label, route in label_to_route.items()}
    run_pages = {route: _route_pages(run) for route, run in by_route.items()}
    source_roots = {
        route: require_external_path(
            Path(str(by_route[route]["source_results_root"])),
            label=f"{route} source results",
        )
        for route in ROUTES
    }
    expected_pages = {str(page["page_id"]) for page in truth_pages}
    for route, page_map in run_pages.items():
        if set(page_map) != expected_pages:
            raise ContractError(f"Page set mismatch for route {route}.")
    output.mkdir(parents=True, exist_ok=True)
    review_assets = output / "assets"
    rows: list[dict[str, str]] = []
    claimed_units: dict[tuple[str, str], set[str]] = {
        (route, page_id): set() for route in ROUTES for page_id in expected_pages
    }
    truth_by_page = {page["page_id"]: page for page in truth_pages}
    all_truth_regions = _truth_regions(truth_pages)
    detectorless_assignments: dict[
        tuple[str, str, str], list[Mapping[str, Any]]
    ] = {}
    for route in ROUTES:
        for page_id, truth_page in truth_by_page.items():
            assigned = _assign_detectorless_units_to_human_truth(
                truth_page["regions"],
                run_pages[route][page_id]["canonical_units"],
            )
            for truth_region_id, units in assigned.items():
                detectorless_assignments[(route, page_id, truth_region_id)] = units
    row_number = 0
    for truth_region in all_truth_regions:
        row_number += 1
        page_id = truth_region["page_id"]
        row_id = f"truth-{truth_region['truth_region_id']}"
        source_crop = truth / str(truth_region["crop_asset"])
        crop_relative = Path("assets") / "truth" / page_id / source_crop.name
        crop_asset = str(truth_region["crop_asset"])
        copied_crop = _copy_review_asset(
            source_crop,
            output,
            crop_relative,
            expected_sha256=_locked_asset_sha256(truth_asset_hashes, crop_asset),
        )
        source_page = truth / str(truth_region["source_asset"])
        page_relative = Path("assets") / "pages" / page_id / source_page.name
        copied_page = page_relative.as_posix()
        if not (output / page_relative).exists():
            source_asset = str(truth_region["source_asset"])
            copied_page = _copy_review_asset(
                source_page,
                output,
                page_relative,
                expected_sha256=_locked_asset_sha256(
                    truth_asset_hashes, source_asset
                ),
            )
        row: dict[str, str] = {
            "row_number": str(row_number),
            "row_id": row_id,
            "row_kind": "truth",
            "page_id": page_id,
            "truth_region_id": str(truth_region["truth_region_id"]),
            "truth_region_source": str(truth_region["region_source"]),
            "language": str(truth_region["language"]),
            "truth_transcription": str(truth_region["transcription"]),
            "truth_semantic_role": str(truth_region["semantic_role"]),
            "truth_processing_action": str(truth_region["processing_action"]),
            "truth_confidence": str(truth_region["confidence"]),
            "truth_bbox_xyxy": json.dumps(truth_region["bbox_xyxy"]),
            "source_page": copied_page,
            "source_crop": copied_crop,
        }
        for label in BLIND_LABELS:
            route = label_to_route[label]
            units = run_pages[route][page_id]["canonical_units"]
            detectorless_matches = (
                detectorless_assignments.get(
                    (route, page_id, str(truth_region["truth_region_id"])),
                    [],
                )
                if not truth_region.get("detector_block_ids")
                else None
            )
            matched, geometry_status = _match_truth_to_units(
                truth_region,
                units,
                detectorless_matches=detectorless_matches,
            )
            for unit in matched:
                claimed_units[(route, page_id)].add(str(unit["canonical_unit_id"]))
            row[f"{label}_text"] = _join_unit_text(matched)
            row[f"{label}_bbox_xyxy"] = (
                json.dumps(_union_bbox(unit["bbox_xyxy"] for unit in matched))
                if matched
                else ""
            )
            raw_region_count = sum(
                len(unit.get("raw_region_ids", [])) for unit in matched
            )
            row[f"{label}_raw_region_ids"] = ",".join(
                f"raw-{index:04d}" for index in range(raw_region_count)
            )
            row[f"{label}_semantic_role"] = ",".join(
                sorted(
                    {
                        str(unit.get("semantic_role", ""))
                        for unit in matched
                        if unit.get("semantic_role")
                    }
                )
            )
            row[f"{label}_processing_action"] = ",".join(
                sorted(
                    {
                        str(unit.get("processing_action", ""))
                        for unit in matched
                        if unit.get("processing_action")
                    }
                )
            )
            row[f"{label}_geometry_status"] = _blind_geometry_status(
                geometry_status
            )
            row[f"{label}_assets_json"] = json.dumps(
                _copy_candidate_assets(
                    units=matched,
                    source_root=source_roots[route],
                    output=output,
                    label=label,
                    page_id=page_id,
                    row_id=row_id,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
            _empty_review_decisions(row, label)
        rows.append(row)
    extras: list[tuple[str, str, str, Mapping[str, Any]]] = []
    for route in ROUTES:
        label = route_to_label[route]
        for page_id, page_result in run_pages[route].items():
            for unit in page_result["canonical_units"]:
                unit_id = str(unit["canonical_unit_id"])
                if unit_id not in claimed_units[(route, page_id)]:
                    extras.append((page_id, label, route, unit))
    for cluster_index, (page_id, cluster) in enumerate(
        _cluster_extra_units(extras), start=1
    ):
        row_number += 1
        truth_page = truth_by_page[page_id]
        row_id = f"extra-{page_id}-{cluster_index:04d}"
        source_page = truth / str(truth_page["source_asset"])
        extra_crop = review_assets / "extras" / page_id / f"extra-{cluster_index:04d}.png"
        row = {
            "row_number": str(row_number),
            "row_id": row_id,
            "row_kind": "candidate_extra",
            "page_id": page_id,
            "truth_region_id": "",
            "truth_region_source": "",
            "language": str(truth_page["language"]),
            "truth_transcription": "",
            "truth_semantic_role": "",
            "truth_processing_action": "",
            "truth_confidence": "",
            "truth_bbox_xyxy": "",
            "source_page": (
                Path("assets") / "pages" / page_id / source_page.name
            ).as_posix(),
            "source_crop": extra_crop.relative_to(output).as_posix(),
        }
        page_destination = output / row["source_page"]
        if not page_destination.exists():
            source_asset = str(truth_page["source_asset"])
            _copy_review_asset(
                source_page,
                output,
                Path(row["source_page"]),
                expected_sha256=_locked_asset_sha256(
                    truth_asset_hashes, source_asset
                ),
            )
        _copy_crop(
            page_destination,
            _union_bbox(unit["bbox_xyxy"] for unit in cluster.values()),
            extra_crop,
        )
        for label in BLIND_LABELS:
            unit = cluster.get(label)
            row[f"{label}_text"] = str(unit.get("text", "") if unit else "")
            row[f"{label}_bbox_xyxy"] = (
                json.dumps(unit.get("bbox_xyxy", [])) if unit else ""
            )
            row[f"{label}_raw_region_ids"] = (
                ",".join(
                    f"raw-{index:04d}"
                    for index, _value in enumerate(unit.get("raw_region_ids", []))
                )
                if unit
                else ""
            )
            row[f"{label}_semantic_role"] = str(
                unit.get("semantic_role", "") if unit else ""
            )
            row[f"{label}_processing_action"] = str(
                unit.get("processing_action", "") if unit else ""
            )
            row[f"{label}_geometry_status"] = (
                _blind_geometry_status(unit.get("geometry_status"))
                if unit
                else "absent"
            )
            row[f"{label}_assets_json"] = json.dumps(
                _copy_candidate_assets(
                    units=[unit] if unit else [],
                    source_root=source_roots[label_to_route[label]],
                    output=output,
                    label=label,
                    page_id=page_id,
                    row_id=row_id,
                ),
                ensure_ascii=False,
                sort_keys=True,
            )
            _empty_review_decisions(row, label)
        rows.append(row)
    _write_review_csv(output / REVIEW_CSV_FILENAME, rows)
    (output / REVIEW_HTML_FILENAME).write_text(_review_html(rows), encoding="utf-8")
    key = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": suite_id,
        "created_at": utc_now(),
        "truth_contract_sha256": truth_lock["truth_contract_sha256"],
        "label_to_route": label_to_route,
        "run_contract_sha256": {
            route: by_route[route]["normalized_contract_sha256"] for route in ROUTES
        },
        "route_run_statistics": {
            route: _run_statistics(by_route[route]) for route in ROUTES
        },
    }
    private_dir = output / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    key_path = private_dir / BLIND_KEY_FILENAME
    write_json(key_path, key)
    try:
        key_path.chmod(0o600)
    except OSError:
        pass
    payload = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": suite_id,
        "truth_contract_sha256": truth_lock["truth_contract_sha256"],
        "row_count": len(rows),
        "truth_row_count": sum(row["row_kind"] == "truth" for row in rows),
        "candidate_extra_row_count": sum(
            row["row_kind"] == "candidate_extra" for row in rows
        ),
        "review_csv_sha256_at_creation": sha256_file(output / REVIEW_CSV_FILENAME),
        "review_locked_content_sha256": _review_locked_content_sha256(rows),
        "review_html_sha256": sha256_file(output / REVIEW_HTML_FILENAME),
        "blind_key_sha256": sha256_file(key_path),
        "review_asset_files": _review_asset_records(output),
    }
    payload["review_assets_contract_sha256"] = canonical_sha256(
        payload["review_asset_files"]
    )
    payload["blind_payload_contract_sha256"] = canonical_sha256(payload)
    write_json(private_dir / BLIND_PAYLOAD_FILENAME, payload)
    return payload


def _read_review_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != _review_headers():
            raise ContractError("Review CSV columns or order changed.")
        return [
            {key: _csv_decode_cell(value) for key, value in row.items()}
            for row in reader
        ]


def validate_review_rows(rows: Sequence[Mapping[str, str]]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        row_id = str(row.get("row_id", ""))
        if not row_id or row_id in seen:
            errors.append(f"row {index}: missing or duplicate row_id")
        seen.add(row_id)
        kind = row.get("row_kind")
        if kind not in {"truth", "candidate_extra"}:
            errors.append(f"row {index}: invalid row_kind")
            continue
        for label in BLIND_LABELS:
            has_text = bool(str(row.get(f"{label}_text", "")).strip())
            required_fields = (
                ("transcription_correct", kind == "truth"),
                ("semantic_correct", kind == "truth"),
                ("role_action_correct", kind == "truth"),
                ("merge_split_error", has_text),
                ("destructive_edit", has_text),
                ("false_positive", kind == "candidate_extra" and has_text),
            )
            for field, required in required_fields:
                value = str(row.get(f"{label}_{field}", "")).strip()
                if not required and not value:
                    continue
                if value not in REVIEW_DECISIONS:
                    errors.append(
                        f"row {index} {label}_{field}: expected one of "
                        f"{sorted(REVIEW_DECISIONS)}"
                    )
            if kind == "truth" and not has_text:
                if (
                    str(row.get("truth_transcription", "")).strip()
                    and str(
                        row.get(f"{label}_transcription_correct", "")
                    ).strip()
                    == "yes"
                ):
                    errors.append(
                        f"row {index} {label}_transcription_correct: "
                        "missing OCR text cannot be correct"
                    )
                if (
                    row.get("truth_processing_action") == "translate_inpaint"
                    and str(row.get(f"{label}_semantic_correct", "")).strip()
                    == "yes"
                ):
                    errors.append(
                        f"row {index} {label}_semantic_correct: "
                        "missing meaning text cannot be correct"
                    )
    return errors


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _review_decision(row: Mapping[str, str], label: str, field: str) -> str:
    return str(row.get(f"{label}_{field}", "")).strip()


def finalize_review(review_dir: Path) -> dict[str, Any]:
    review = require_external_path(review_dir, label="review directory")
    payload = read_json(review / "private" / BLIND_PAYLOAD_FILENAME)
    if (
        payload.get("schema_version") != REVIEW_SCHEMA_VERSION
        or payload.get("protocol_version") != PROTOCOL_VERSION
    ):
        raise ContractError("Blind review payload contract is invalid.")
    payload_contract = require_sha256(
        payload.get("blind_payload_contract_sha256"),
        label="blind review payload contract",
    )
    unsigned_payload = {
        key: value
        for key, value in payload.items()
        if key != "blind_payload_contract_sha256"
    }
    if canonical_sha256(unsigned_payload) != payload_contract:
        raise ContractError("Blind review payload changed after creation.")
    key_path = review / "private" / BLIND_KEY_FILENAME
    if sha256_file(key_path) != payload.get("blind_key_sha256"):
        raise ContractError("Blind key changed after review creation.")
    key = read_json(key_path)
    label_to_route = key.get("label_to_route")
    if not isinstance(label_to_route, dict) or set(label_to_route) != set(BLIND_LABELS):
        raise ContractError("Blind key mapping is invalid.")
    if set(label_to_route.values()) != set(ROUTES):
        raise ContractError("Blind key does not cover all OCR routes.")
    if (
        key.get("suite_id") != payload.get("suite_id")
        or key.get("truth_contract_sha256")
        != payload.get("truth_contract_sha256")
    ):
        raise ContractError("Blind review truth or suite contract changed.")
    if sha256_file(review / REVIEW_HTML_FILENAME) != payload.get("review_html_sha256"):
        raise ContractError("Blind review HTML changed after creation.")
    asset_records = _review_asset_records(review)
    if asset_records != payload.get("review_asset_files") or canonical_sha256(
        asset_records
    ) != payload.get("review_assets_contract_sha256"):
        raise ContractError("Blind review visual evidence assets changed after creation.")
    rows = _read_review_rows(review / REVIEW_CSV_FILENAME)
    if len(rows) != int(payload.get("row_count", -1)):
        raise ContractError("Review row count changed.")
    if _review_locked_content_sha256(rows) != payload.get(
        "review_locked_content_sha256"
    ):
        raise ContractError("Blind review evidence columns changed after creation.")
    errors = validate_review_rows(rows)
    if errors:
        raise IncompleteReviewError(errors)
    route_metrics: dict[str, dict[str, Any]] = {}
    for label, route in label_to_route.items():
        truth_rows = [row for row in rows if row["row_kind"] == "truth"]
        meaning_rows = [
            row
            for row in truth_rows
            if row["truth_processing_action"] == "translate_inpaint"
        ]
        human_extra_meaning_rows = [
            row
            for row in meaning_rows
            if row.get("truth_region_source") == "human_extra"
        ]
        extra_rows = [
            row
            for row in rows
            if row["row_kind"] == "candidate_extra" and row[f"{label}_text"].strip()
        ]
        exact_count = sum(
            normalize_text(row["truth_transcription"])
            == normalize_text(row[f"{label}_text"])
            for row in truth_rows
        )
        character_accuracies = [
            normalized_character_accuracy(
                row["truth_transcription"], row[f"{label}_text"]
            )
            for row in truth_rows
        ]
        semantic_yes = sum(
            bool(row[f"{label}_text"].strip())
            and _review_decision(row, label, "semantic_correct") == "yes"
            for row in meaning_rows
        )
        transcription_yes = sum(
            (
                bool(row[f"{label}_text"].strip())
                or not row["truth_transcription"].strip()
            )
            and _review_decision(row, label, "transcription_correct") == "yes"
            for row in truth_rows
        )
        role_action_yes = sum(
            _review_decision(row, label, "role_action_correct") == "yes"
            for row in truth_rows
        )
        false_positive_count = sum(
            _review_decision(row, label, "false_positive") == "yes"
            for row in extra_rows
        )
        destructive_count = sum(
            _review_decision(row, label, "destructive_edit") == "yes"
            for row in rows
            if row[f"{label}_text"].strip()
        )
        merge_split_count = sum(
            _review_decision(row, label, "merge_split_error") == "yes"
            for row in rows
            if row[f"{label}_text"].strip()
        )
        page_ids = sorted({row["page_id"] for row in truth_rows})
        route_stats = key.get("route_run_statistics", {}).get(route)
        if not isinstance(route_stats, dict):
            raise ContractError(f"Missing private run statistics for {route}.")
        page_status = route_stats.get("page_status", {})
        if not isinstance(page_status, dict):
            raise ContractError(f"Missing private page status for {route}.")
        complete_pages = 0
        for page_id in page_ids:
            page_truth = [row for row in truth_rows if row["page_id"] == page_id]
            page_meaning = [
                row
                for row in page_truth
                if row["truth_processing_action"] == "translate_inpaint"
            ]
            page_review_rows = [row for row in rows if row["page_id"] == page_id]
            if (
                page_status.get(page_id) == "success"
                and all(
                    _review_decision(row, label, "transcription_correct") == "yes"
                    for row in page_truth
                )
                and all(
                    _review_decision(row, label, "role_action_correct") == "yes"
                    for row in page_truth
                )
                and all(
                    _review_decision(row, label, "semantic_correct") == "yes"
                    for row in page_meaning
                )
                and all(
                    _review_decision(row, label, "merge_split_error") == "no"
                    and _review_decision(row, label, "destructive_edit")
                    in {"no", "not_applicable"}
                    for row in page_review_rows
                    if row[f"{label}_text"].strip()
                )
                and all(
                    _review_decision(row, label, "false_positive") == "no"
                    for row in page_review_rows
                    if row["row_kind"] == "candidate_extra"
                    and row[f"{label}_text"].strip()
                )
            ):
                complete_pages += 1
        human_extra_semantic_yes = sum(
            _review_decision(row, label, "semantic_correct") == "yes"
            for row in human_extra_meaning_rows
        )
        route_metrics[route] = {
            "run_statistics": route_stats,
            "page_complete_count": complete_pages,
            "page_complete_accuracy": _ratio(complete_pages, len(page_ids)),
            "truth_region_count": len(truth_rows),
            "meaning_region_count": len(meaning_rows),
            "normalized_exact_count": exact_count,
            "normalized_exact_accuracy": _ratio(exact_count, len(truth_rows)),
            "normalized_character_accuracy": (
                sum(character_accuracies) / len(character_accuracies)
                if character_accuracies
                else None
            ),
            "reviewed_transcription_correct_count": transcription_yes,
            "reviewed_transcription_correct_accuracy": _ratio(
                transcription_yes, len(truth_rows)
            ),
            "reviewed_role_action_correct_count": role_action_yes,
            "reviewed_role_action_correct_accuracy": _ratio(
                role_action_yes, len(truth_rows)
            ),
            "reviewed_semantic_correct_count": semantic_yes,
            "meaning_text_recall": _ratio(semantic_yes, len(meaning_rows)),
            "full_page_only_meaning_region_count": len(human_extra_meaning_rows),
            "full_page_only_meaning_recall": _ratio(
                human_extra_semantic_yes, len(human_extra_meaning_rows)
            ),
            "candidate_extra_count": len(extra_rows),
            "reviewed_false_positive_count": false_positive_count,
            "merge_split_error_count": merge_split_count,
            "destructive_edit_count": destructive_count,
        }
    result = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": payload["suite_id"],
        "finalized_at": utc_now(),
        "quality_decision_source": "completed_blind_human_review",
        "automatic_similarity_is_not_semantic_truth": True,
        "truth_contract_sha256": payload["truth_contract_sha256"],
        "run_contract_sha256": key["run_contract_sha256"],
        "review_assets_contract_sha256": payload[
            "review_assets_contract_sha256"
        ],
        "completed_review_csv_sha256": sha256_file(
            review / REVIEW_CSV_FILENAME
        ),
        "route_metrics": route_metrics,
    }
    write_json(review / FINAL_METRICS_FILENAME, result)
    lines = [
        "# 세 OCR 사람 기준 최종 검수",
        "",
        "> 문자열 유사도는 진단값일 뿐입니다. 의미 품질 수치는 완료된 블라인드 검수에서만 계산했습니다.",
        "",
        "| 경로 | 구조 성공 | 페이지 완전 | 글자 정확 | 문자 정확도 | 의미 recall | 추가 영역 | 오탐 | merge/split | 파괴 편집 | 요청 시간 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for route in ROUTES:
        metrics = route_metrics[route]
        lines.append(
            "| {route} | {success}/{pages} | {complete:.2%} | {exact:.2%} | "
            "{char:.2%} | {semantic:.2%} | {extra} | {false_positive} | "
            "{merge} | {destructive} | {elapsed:.3f}s |".format(
                route=route,
                success=metrics["run_statistics"]["successful_page_count"],
                pages=metrics["run_statistics"]["page_count"],
                complete=metrics["page_complete_accuracy"] or 0.0,
                exact=metrics["normalized_exact_accuracy"] or 0.0,
                char=metrics["normalized_character_accuracy"] or 0.0,
                semantic=metrics["meaning_text_recall"] or 0.0,
                extra=metrics["candidate_extra_count"],
                false_positive=metrics["reviewed_false_positive_count"],
                merge=metrics["merge_split_error_count"],
                destructive=metrics["destructive_edit_count"],
                elapsed=metrics["run_statistics"]["elapsed_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "속도와 후보명은 검수 완료 뒤에만 공개됐습니다. 기본값 변경은 이 보고서와 사용자 승인 후 별도 제품 PR에서만 수행합니다.",
            "",
        ]
    )
    (review / FINAL_REPORT_FILENAME).write_text("\n".join(lines), encoding="utf-8")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_manifest_parser = subparsers.add_parser("build-manifest")
    build_manifest_parser.add_argument("--spec", type=Path, required=True)
    build_manifest_parser.add_argument("--output", type=Path, required=True)

    validate_manifest_parser = subparsers.add_parser("validate-manifest")
    validate_manifest_parser.add_argument("--manifest", type=Path, required=True)

    init_parser = subparsers.add_parser("init-truth")
    init_parser.add_argument("--manifest", type=Path, required=True)
    init_parser.add_argument("--output", type=Path, required=True)

    lock_parser = subparsers.add_parser("lock-truth")
    lock_parser.add_argument("--truth-dir", type=Path, required=True)

    refresh_parser = subparsers.add_parser("refresh-truth-crops")
    refresh_parser.add_argument("--truth-dir", type=Path, required=True)

    export_truth_parser = subparsers.add_parser("export-truth-csv")
    export_truth_parser.add_argument("--truth-dir", type=Path, required=True)
    export_truth_parser.add_argument("--output", type=Path)

    import_truth_parser = subparsers.add_parser("import-truth-csv")
    import_truth_parser.add_argument("--truth-dir", type=Path, required=True)
    import_truth_parser.add_argument("--csv", type=Path, required=True)

    validate_truth_parser = subparsers.add_parser("validate-truth")
    validate_truth_parser.add_argument("--truth-dir", type=Path, required=True)

    bind_parser = subparsers.add_parser("bind-source-results")
    bind_parser.add_argument("--route", choices=ROUTES, required=True)
    bind_parser.add_argument("--manifest", type=Path, required=True)
    bind_parser.add_argument("--source-results", type=Path, required=True)

    import_parser = subparsers.add_parser("import-existing")
    import_parser.add_argument("--route", choices=ROUTES, required=True)
    import_parser.add_argument("--manifest", type=Path, required=True)
    import_parser.add_argument("--source-results", type=Path, required=True)
    import_parser.add_argument("--runtime-contract", type=Path, required=True)
    import_parser.add_argument("--output", type=Path, required=True)

    validate_run_parser = subparsers.add_parser("validate-run")
    validate_run_parser.add_argument("--run", type=Path, required=True)

    review_parser = subparsers.add_parser("make-review")
    review_parser.add_argument("--truth-dir", type=Path, required=True)
    review_parser.add_argument("--run", type=Path, action="append", required=True)
    review_parser.add_argument("--output", type=Path, required=True)

    finalize_parser = subparsers.add_parser("finalize-review")
    finalize_parser.add_argument("--review-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "build-manifest":
            result = build_corpus_manifest(args.spec, args.output)
        elif args.command == "validate-manifest":
            result = validate_corpus_manifest(args.manifest)
        elif args.command == "init-truth":
            result = init_truth(args.manifest, args.output)
        elif args.command == "lock-truth":
            result = lock_truth(args.truth_dir)
        elif args.command == "refresh-truth-crops":
            result = refresh_truth_crops(args.truth_dir)
        elif args.command == "export-truth-csv":
            result = export_truth_csv(args.truth_dir, args.output)
        elif args.command == "import-truth-csv":
            result = import_truth_csv(args.truth_dir, args.csv)
        elif args.command == "validate-truth":
            lock, pages = validate_locked_truth(args.truth_dir)
            result = {"truth_contract_sha256": lock["truth_contract_sha256"], "page_count": len(pages)}
        elif args.command == "bind-source-results":
            result = create_source_bindings(
                route=args.route,
                corpus_manifest=args.manifest,
                source_results=args.source_results,
            )
        elif args.command == "import-existing":
            result = import_existing_run(
                route=args.route,
                corpus_manifest=args.manifest,
                source_results=args.source_results,
                runtime_contract=args.runtime_contract,
                output_dir=args.output,
            )
        elif args.command == "validate-run":
            result = validate_run(args.run)
        elif args.command == "make-review":
            result = make_review(
                truth_dir=args.truth_dir,
                runs=args.run,
                output_dir=args.output,
            )
        elif args.command == "finalize-review":
            result = finalize_review(args.review_dir)
        else:  # pragma: no cover
            raise ContractError(f"Unknown command: {args.command}")
    except (ContractError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        if isinstance(exc, IncompleteReviewError):
            for error in exc.errors[:50]:
                print(f"  - {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
