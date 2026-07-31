#!/usr/bin/env python3
"""Validate and score Git-external COO SFX shadow evidence.

This benchmark-only tool never starts Docker, loads a model, or changes product
routing. It validates official COO ABCNetv2 prediction exports, compares CPU and
CUDA geometry, and evaluates a review-only SFX signal against source-first
locked truth. Automatic preserve decisions are intentionally out of scope.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence

import benchmark_ocr_three_way_human_truth as three_way


PROTOCOL_VERSION = "coo-sfx-shadow-v1"
PREDICTION_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1
EXPECTED_MODEL = "COO-ABCNetv2"
EXPECTED_MODEL_SHA256 = (
    "25d33d9dc033a65c888e99ef25dbdfadd5b2ae7bf8d3b18e8e85a093956ea6e2"
)
EXPECTED_SOURCE_COMMIT = "d8028f015b8ce99a4dd798427342f97087529357"
IMAGE_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
ALLOWED_DEVICES = {"cpu", "cuda"}
ALLOWED_NORMALIZATION = {"cpu": "BN", "cuda": "SyncBN"}
SFX_ROLES = {"sfx", "decorative"}
MEANING_ACTION = "translate_inpaint"


class ContractError(three_way.ContractError):
    """Raised when COO evidence violates the benchmark contract."""


def _finite_number(value: Any, *, label: str) -> float:
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ContractError(f"{label} must be a finite number.")
    return float(value)


def _nonnegative_number(value: Any, *, label: str) -> float:
    number = _finite_number(value, label=label)
    if number < 0:
        raise ContractError(f"{label} must be non-negative.")
    return number


def _digest(value: Any, *, label: str) -> str:
    normalized = str(value or "").strip().lower()
    if not IMAGE_DIGEST_RE.fullmatch(normalized):
        raise ContractError(f"{label} must be sha256:<64 lowercase hex>.")
    return normalized


def _sha256(value: Any, *, label: str) -> str:
    try:
        return three_way.require_sha256(value, label=label)
    except three_way.ContractError as exc:
        raise ContractError(str(exc)) from exc


def _bbox_float(
    value: Any,
    *,
    label: str,
    width: int,
    height: int,
) -> tuple[list[float], bool]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ContractError(f"{label} must be [x1, y1, x2, y2].")
    raw = [_finite_number(item, label=label) for item in value]
    x1, y1, x2, y2 = raw
    if not (x1 < x2 and y1 < y2):
        raise ContractError(f"{label} is degenerate: {raw}")
    tolerance_x = max(1.0, width * 0.10)
    tolerance_y = max(1.0, height * 0.10)
    if not (
        -tolerance_x <= x1 <= width + tolerance_x
        and -tolerance_y <= y1 <= height + tolerance_y
        and -tolerance_x <= x2 <= width + tolerance_x
        and -tolerance_y <= y2 <= height + tolerance_y
    ):
        raise ContractError(f"{label} is implausibly outside {width}x{height}: {raw}")
    box = [
        min(max(x1, 0.0), float(width)),
        min(max(y1, 0.0), float(height)),
        min(max(x2, 0.0), float(width)),
        min(max(y2, 0.0), float(height)),
    ]
    if not (box[0] < box[2] and box[1] < box[3]):
        raise ContractError(f"{label} is outside the visible page: {raw}")
    return box, box != raw


def _polygon(
    value: Any,
    *,
    label: str,
    width: int,
    height: int,
) -> tuple[list[list[float]], bool]:
    if not isinstance(value, list) or len(value) < 4:
        raise ContractError(f"{label} must contain at least four points.")
    points: list[list[float]] = []
    clipped = False
    tolerance_x = max(1.0, width * 0.10)
    tolerance_y = max(1.0, height * 0.10)
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ContractError(f"{label}[{index}] must be [x, y].")
        x = _finite_number(point[0], label=f"{label}[{index}].x")
        y = _finite_number(point[1], label=f"{label}[{index}].y")
        if not (
            -tolerance_x <= x <= width + tolerance_x
            and -tolerance_y <= y <= height + tolerance_y
        ):
            raise ContractError(
                f"{label}[{index}] is implausibly outside {width}x{height}: {[x, y]}"
            )
        visible = [
            min(max(x, 0.0), float(width)),
            min(max(y, 0.0), float(height)),
        ]
        clipped = clipped or visible != [x, y]
        points.append(visible)
    return points, clipped


def _manifest_pages_by_sha(
    manifest: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for page in manifest["pages"]:
        source_sha = str(page["source_image"]["sha256"])
        if source_sha in result:
            raise ContractError(
                "Corpus manifest contains duplicate source image SHA-256 values."
            )
        result[source_sha] = page
    return result


def validate_predictions(
    predictions_path: Path,
    corpus_manifest: Path,
) -> dict[str, Any]:
    predictions_file = three_way.require_external_path(
        predictions_path, label="COO prediction export"
    )
    manifest_file = three_way.require_external_path(
        corpus_manifest, label="COO corpus manifest"
    )
    try:
        payload = three_way.read_json(predictions_file)
        manifest = three_way.validate_corpus_manifest(manifest_file)
    except three_way.ContractError as exc:
        raise ContractError(str(exc)) from exc

    if payload.get("schema_version") != PREDICTION_SCHEMA_VERSION:
        raise ContractError("Unsupported COO prediction schema.")
    if payload.get("model") != EXPECTED_MODEL:
        raise ContractError("Unexpected COO model identity.")
    if (
        _sha256(payload.get("model_sha256"), label="COO model")
        != EXPECTED_MODEL_SHA256
    ):
        raise ContractError("Unexpected COO model SHA-256.")
    if str(payload.get("source_commit", "")).strip() != EXPECTED_SOURCE_COMMIT:
        raise ContractError("Unexpected COO source commit.")
    runtime_image_digest = _digest(
        payload.get("runtime_image_digest"), label="COO runtime image digest"
    )
    threshold = _finite_number(payload.get("threshold"), label="COO threshold")
    if not 0 < threshold < 1:
        raise ContractError("COO threshold must be between zero and one.")
    device = str(payload.get("device", "")).strip()
    if device not in ALLOWED_DEVICES:
        raise ContractError(f"Unsupported COO device: {device!r}")
    normalization = str(payload.get("normalization", "")).strip()
    if normalization != ALLOWED_NORMALIZATION[device]:
        raise ContractError(
            f"COO {device} evidence must use {ALLOWED_NORMALIZATION[device]}."
        )
    torch_version = str(payload.get("torch_version", "")).strip()
    if not torch_version:
        raise ContractError("COO torch version is missing.")
    cuda_runtime_version = str(payload.get("cuda_runtime_version", "")).strip()
    if not cuda_runtime_version:
        raise ContractError("COO CUDA runtime version is missing.")
    cuda_device_name = payload.get("cuda_device_name")
    if device == "cuda" and not str(cuda_device_name or "").strip():
        raise ContractError("CUDA evidence is missing the GPU device name.")
    if device == "cpu" and cuda_device_name is not None:
        raise ContractError("CPU evidence must not claim a CUDA device.")

    model_load_seconds = _nonnegative_number(
        payload.get("model_load_seconds"), label="COO model load time"
    )
    process_elapsed_seconds = _nonnegative_number(
        payload.get("process_elapsed_seconds"), label="COO process time"
    )
    cuda_peak_allocated = payload.get("cuda_peak_allocated_bytes")
    cuda_peak_reserved = payload.get("cuda_peak_reserved_bytes")
    if device == "cuda":
        cuda_peak_allocated = int(
            _nonnegative_number(
                cuda_peak_allocated, label="CUDA peak allocated bytes"
            )
        )
        cuda_peak_reserved = int(
            _nonnegative_number(cuda_peak_reserved, label="CUDA peak reserved bytes")
        )
        if cuda_peak_allocated > cuda_peak_reserved:
            raise ContractError("CUDA allocated bytes exceed reserved bytes.")
    elif cuda_peak_allocated is not None or cuda_peak_reserved is not None:
        raise ContractError("CPU evidence must not contain CUDA peak memory.")

    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ContractError("COO prediction export has no pages.")
    manifest_by_sha = _manifest_pages_by_sha(manifest)
    seen_page_ids: set[str] = set()
    normalized_pages: list[dict[str, Any]] = []
    for page_index, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ContractError(f"COO page {page_index} must be an object.")
        source_sha = _sha256(
            page.get("source_sha256"), label=f"COO page {page_index} source"
        )
        manifest_page = manifest_by_sha.get(source_sha)
        if manifest_page is None:
            raise ContractError(
                f"COO page {page_index} is not bound to the corpus manifest."
            )
        page_id = str(manifest_page["page_id"])
        if page_id in seen_page_ids:
            raise ContractError(f"Duplicate COO page: {page_id}")
        seen_page_ids.add(page_id)
        width = int(manifest_page["source_image"]["width"])
        height = int(manifest_page["source_image"]["height"])
        if (
            int(page.get("width", 0) or 0) != width
            or int(page.get("height", 0) or 0) != height
        ):
            raise ContractError(f"COO page dimensions changed: {page_id}")
        elapsed_seconds = _nonnegative_number(
            page.get("elapsed_seconds"), label=f"COO page time {page_id}"
        )
        regions = page.get("regions")
        if not isinstance(regions, list):
            raise ContractError(f"COO regions are missing on {page_id}.")
        if int(page.get("region_count", -1)) != len(regions):
            raise ContractError(f"COO region count changed on {page_id}.")
        seen_region_ids: set[str] = set()
        normalized_regions: list[dict[str, Any]] = []
        for region_index, region in enumerate(regions):
            if not isinstance(region, dict):
                raise ContractError(
                    f"COO region {page_id}:{region_index} must be an object."
                )
            region_id = three_way.require_safe_id(
                region.get("region_id"), label="COO region ID"
            )
            if region_id in seen_region_ids:
                raise ContractError(f"Duplicate COO region on {page_id}: {region_id}")
            seen_region_ids.add(region_id)
            score = _finite_number(
                region.get("score"), label=f"COO score {page_id}:{region_id}"
            )
            if not 0 <= score <= 1:
                raise ContractError(f"COO score is outside [0, 1]: {page_id}:{region_id}")
            box, bbox_clipped = _bbox_float(
                region.get("bbox_xyxy"),
                label=f"COO bbox {page_id}:{region_id}",
                width=width,
                height=height,
            )
            polygon, polygon_clipped = _polygon(
                region.get("polygon_xy"),
                label=f"COO polygon {page_id}:{region_id}",
                width=width,
                height=height,
            )
            normalized_regions.append(
                {
                    "region_id": region_id,
                    "score": score,
                    "bbox_xyxy": box,
                    "polygon_xy": polygon,
                    "geometry_clipped": bbox_clipped or polygon_clipped,
                }
            )
        normalized_pages.append(
            {
                "page_id": page_id,
                "source_sha256": source_sha,
                "width": width,
                "height": height,
                "elapsed_seconds": elapsed_seconds,
                "regions": normalized_regions,
            }
        )

    normalized = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "suite_id": manifest["suite_id"],
        "manifest_sha256": manifest["manifest_sha256"],
        "predictions_file_sha256": three_way.sha256_file(predictions_file),
        "model": EXPECTED_MODEL,
        "model_sha256": EXPECTED_MODEL_SHA256,
        "source_commit": EXPECTED_SOURCE_COMMIT,
        "runtime_image_digest": runtime_image_digest,
        "threshold": threshold,
        "device": device,
        "normalization": normalization,
        "torch_version": torch_version,
        "cuda_runtime_version": cuda_runtime_version,
        "cuda_device_name": cuda_device_name,
        "cuda_peak_allocated_bytes": cuda_peak_allocated,
        "cuda_peak_reserved_bytes": cuda_peak_reserved,
        "model_load_seconds": model_load_seconds,
        "process_elapsed_seconds": process_elapsed_seconds,
        "pages": sorted(normalized_pages, key=lambda item: item["page_id"]),
    }
    normalized["clipped_region_count"] = sum(
        bool(region["geometry_clipped"])
        for page in normalized["pages"]
        for region in page["regions"]
    )
    normalized["prediction_contract_sha256"] = three_way.canonical_sha256(normalized)
    return normalized


def _box_area(box: Sequence[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection_area(left: Sequence[float], right: Sequence[float]) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )


def bbox_iou(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = _intersection_area(left, right)
    union = _box_area(left) + _box_area(right) - intersection
    return intersection / union if union else 0.0


def bbox_overlap_strength(left: Sequence[float], right: Sequence[float]) -> float:
    intersection = _intersection_area(left, right)
    smaller = min(_box_area(left), _box_area(right))
    return intersection / smaller if smaller else 0.0


def _greedy_match(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
) -> list[tuple[int, int, float]]:
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_region in enumerate(left):
        for right_index, right_region in enumerate(right):
            candidates.append(
                (
                    bbox_iou(
                        left_region["bbox_xyxy"], right_region["bbox_xyxy"]
                    ),
                    left_index,
                    right_index,
                )
            )
    used_left: set[int] = set()
    used_right: set[int] = set()
    matches: list[tuple[int, int, float]] = []
    for overlap, left_index, right_index in sorted(candidates, reverse=True):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        matches.append((left_index, right_index, overlap))
    return matches


def _speedup_percent(baseline: float, candidate: float) -> float | None:
    if baseline <= 0:
        return None
    return (baseline - candidate) / baseline * 100.0


def compare_devices(
    *,
    cpu_predictions: Path,
    cuda_predictions: Path,
    corpus_manifest: Path,
    output_dir: Path,
) -> dict[str, Any]:
    cpu = validate_predictions(cpu_predictions, corpus_manifest)
    cuda = validate_predictions(cuda_predictions, corpus_manifest)
    if cpu["device"] != "cpu" or cuda["device"] != "cuda":
        raise ContractError("Device comparison requires CPU evidence then CUDA evidence.")
    identical_fields = (
        "suite_id",
        "manifest_sha256",
        "model",
        "model_sha256",
        "source_commit",
        "runtime_image_digest",
        "threshold",
        "torch_version",
        "cuda_runtime_version",
    )
    for field in identical_fields:
        if cpu[field] != cuda[field]:
            raise ContractError(f"CPU/CUDA evidence differs in {field}.")
    cpu_pages = {page["page_id"]: page for page in cpu["pages"]}
    cuda_pages = {page["page_id"]: page for page in cuda["pages"]}
    if set(cpu_pages) != set(cuda_pages):
        raise ContractError("CPU/CUDA page sets differ.")

    page_results: list[dict[str, Any]] = []
    all_ious: list[float] = []
    max_bbox_delta = 0.0
    counts_equal = True
    for page_id in sorted(cpu_pages):
        cpu_page = cpu_pages[page_id]
        cuda_page = cuda_pages[page_id]
        if cpu_page["source_sha256"] != cuda_page["source_sha256"]:
            raise ContractError(f"CPU/CUDA source SHA differs: {page_id}")
        matches = _greedy_match(cpu_page["regions"], cuda_page["regions"])
        counts_equal = counts_equal and (
            len(cpu_page["regions"]) == len(cuda_page["regions"])
        )
        page_max_delta = 0.0
        for cpu_index, cuda_index, overlap in matches:
            all_ious.append(overlap)
            cpu_box = cpu_page["regions"][cpu_index]["bbox_xyxy"]
            cuda_box = cuda_page["regions"][cuda_index]["bbox_xyxy"]
            page_max_delta = max(
                page_max_delta,
                max(abs(left - right) for left, right in zip(cpu_box, cuda_box)),
            )
        max_bbox_delta = max(max_bbox_delta, page_max_delta)
        page_minimum_iou = min(
            (match[2] for match in matches),
            default=(1.0 if not cpu_page["regions"] and not cuda_page["regions"] else 0.0),
        )
        page_results.append(
            {
                "page_id": page_id,
                "cpu_region_count": len(cpu_page["regions"]),
                "cuda_region_count": len(cuda_page["regions"]),
                "matched_region_count": len(matches),
                "minimum_matched_bbox_iou": page_minimum_iou,
                "maximum_bbox_coordinate_delta_px": page_max_delta,
                "cpu_inference_seconds": cpu_page["elapsed_seconds"],
                "cuda_inference_seconds": cuda_page["elapsed_seconds"],
            }
        )
    minimum_iou = min(all_ious, default=(1.0 if counts_equal else 0.0))
    geometry_equivalent = (
        counts_equal
        and cpu["clipped_region_count"] == cuda["clipped_region_count"]
        and minimum_iou >= 0.999
        and max_bbox_delta <= 0.1
    )
    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "comparison": "cpu_vs_cuda",
        "suite_id": cpu["suite_id"],
        "manifest_sha256": cpu["manifest_sha256"],
        "threshold": cpu["threshold"],
        "model_sha256": cpu["model_sha256"],
        "source_commit": cpu["source_commit"],
        "runtime_image_digest": cpu["runtime_image_digest"],
        "page_count": len(page_results),
        "region_counts_equal": counts_equal,
        "minimum_matched_bbox_iou": minimum_iou,
        "maximum_bbox_coordinate_delta_px": max_bbox_delta,
        "geometry_equivalent": geometry_equivalent,
        "cpu": {
            "normalization": cpu["normalization"],
            "model_load_seconds": cpu["model_load_seconds"],
            "inference_seconds": sum(
                page["elapsed_seconds"] for page in cpu["pages"]
            ),
            "process_elapsed_seconds": cpu["process_elapsed_seconds"],
            "clipped_region_count": cpu["clipped_region_count"],
        },
        "cuda": {
            "device_name": cuda["cuda_device_name"],
            "normalization": cuda["normalization"],
            "model_load_seconds": cuda["model_load_seconds"],
            "inference_seconds": sum(
                page["elapsed_seconds"] for page in cuda["pages"]
            ),
            "process_elapsed_seconds": cuda["process_elapsed_seconds"],
            "peak_allocated_bytes": cuda["cuda_peak_allocated_bytes"],
            "peak_reserved_bytes": cuda["cuda_peak_reserved_bytes"],
            "clipped_region_count": cuda["clipped_region_count"],
        },
        "inference_speedup_percent": _speedup_percent(
            sum(page["elapsed_seconds"] for page in cpu["pages"]),
            sum(page["elapsed_seconds"] for page in cuda["pages"]),
        ),
        "process_speedup_percent": _speedup_percent(
            cpu["process_elapsed_seconds"], cuda["process_elapsed_seconds"]
        ),
        "pages": page_results,
        "promotion_allowed": False,
        "promotion_blockers": [
            "COO remains a benchmark-only review signal.",
            "Locked source-first truth and route-level benefit are not part of device equivalence.",
            "ABCNetv2 product licensing requires separate clearance.",
        ],
    }
    result["report_sha256"] = three_way.canonical_sha256(result)
    output = three_way.require_external_path(output_dir, label="COO comparison output")
    output.mkdir(parents=True, exist_ok=True)
    three_way.write_json(output / "device-comparison.json", result)
    _write_device_report(output / "device-comparison-ko.md", result)
    return result


def _write_device_report(path: Path, result: Mapping[str, Any]) -> None:
    cpu = result["cpu"]
    cuda = result["cuda"]
    lines = [
        "# COO CPU/CUDA 일치성",
        "",
        f"- pages: {result['page_count']}",
        f"- threshold: {result['threshold']}",
        f"- geometry equivalent: {result['geometry_equivalent']}",
        f"- minimum matched bbox IoU: {result['minimum_matched_bbox_iou']:.6f}",
        f"- maximum coordinate delta: {result['maximum_bbox_coordinate_delta_px']:.6f}px",
        f"- CPU inference: {cpu['inference_seconds']:.3f}s",
        f"- CUDA inference: {cuda['inference_seconds']:.3f}s",
        f"- inference speedup: {result['inference_speedup_percent']:.3f}%",
        f"- CPU process: {cpu['process_elapsed_seconds']:.3f}s",
        f"- CUDA process: {cuda['process_elapsed_seconds']:.3f}s",
        f"- process speedup: {result['process_speedup_percent']:.3f}%",
        f"- CUDA peak reserved: {cuda['peak_reserved_bytes'] / 1048576:.1f} MiB",
        "",
        "이 보고서는 장치 일치성만 증명한다. 제품 적용과 SFX 품질 승격은 허용하지 않는다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _truth_pages_by_id(
    truth_pages: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    return {str(page["page_id"]): page for page in truth_pages}


def _coo_hits(
    truth_box: Sequence[float],
    regions: Sequence[Mapping[str, Any]],
    *,
    minimum_overlap: float,
) -> list[str]:
    return [
        str(region["region_id"])
        for region in regions
        if bbox_overlap_strength(truth_box, region["bbox_xyxy"]) >= minimum_overlap
    ]


def evaluate_shadow(
    *,
    predictions_path: Path,
    corpus_manifest: Path,
    truth_dir: Path,
    output_dir: Path,
    normalized_runs: Sequence[Path] = (),
    minimum_overlap: float = 0.30,
) -> dict[str, Any]:
    if not 0 < minimum_overlap <= 1:
        raise ContractError("Minimum overlap must be in (0, 1].")
    predictions = validate_predictions(predictions_path, corpus_manifest)
    try:
        truth_lock, truth_pages = three_way.validate_locked_truth(truth_dir)
    except three_way.ContractError as exc:
        raise ContractError(str(exc)) from exc
    truth_by_id = _truth_pages_by_id(truth_pages)
    prediction_page_ids = {page["page_id"] for page in predictions["pages"]}
    if not prediction_page_ids.issubset(truth_by_id):
        raise ContractError("COO predictions include pages outside locked truth.")

    region_rows: list[dict[str, Any]] = []
    role_counts: Counter[str] = Counter()
    role_hit_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()
    action_hit_counts: Counter[str] = Counter()
    for prediction_page in predictions["pages"]:
        truth_page = truth_by_id[prediction_page["page_id"]]
        for truth_region in truth_page["regions"]:
            box = [float(value) for value in truth_region["bbox_xyxy"]]
            hit_ids = _coo_hits(
                box,
                prediction_page["regions"],
                minimum_overlap=minimum_overlap,
            )
            role = str(truth_region["semantic_role"])
            action = str(truth_region["processing_action"])
            detector_text_class = str(
                truth_region.get("detector_text_class", "") or ""
            )
            is_bubble = "bubble" in detector_text_class.casefold()
            safe_review_signal = bool(hit_ids) and not is_bubble
            role_counts[role] += 1
            action_counts[action] += 1
            if hit_ids:
                role_hit_counts[role] += 1
                action_hit_counts[action] += 1
            region_rows.append(
                {
                    "page_id": prediction_page["page_id"],
                    "truth_region_id": truth_region["truth_region_id"],
                    "semantic_role": role,
                    "processing_action": action,
                    "detector_text_class": detector_text_class,
                    "coo_region_ids": hit_ids,
                    "coo_signal": bool(hit_ids),
                    "review_only_nonbubble_signal": safe_review_signal,
                    "automatic_preserve": False,
                }
            )

    run_metrics: dict[str, Any] = {}
    for run_path in normalized_runs:
        try:
            run = three_way.validate_run(run_path)
            run_manifest = three_way.validate_corpus_manifest(
                Path(str(run["corpus_manifest_path"]))
            )
        except three_way.ContractError as exc:
            raise ContractError(str(exc)) from exc
        if run_manifest["manifest_sha256"] != predictions["manifest_sha256"]:
            raise ContractError("Normalized OCR run uses a different corpus manifest.")
        route = str(run["route_id"])
        if route in run_metrics:
            raise ContractError(f"Duplicate normalized OCR route: {route}")
        units_by_page = {
            str(page["page_id"]): page["canonical_units"] for page in run["pages"]
        }
        baseline_sfx_edit = 0
        caught_sfx_edit = 0
        meaningful_review = 0
        for row in region_rows:
            detector_ids = set(
                next(
                    region["detector_block_ids"]
                    for region in truth_by_id[row["page_id"]]["regions"]
                    if region["truth_region_id"] == row["truth_region_id"]
                )
            )
            matching_units = [
                unit
                for unit in units_by_page.get(row["page_id"], [])
                if detector_ids.intersection(unit.get("detector_block_ids", []))
            ]
            route_actions = {
                str(unit.get("processing_action", "") or "")
                for unit in matching_units
            }
            risky_sfx = (
                row["semantic_role"] in SFX_ROLES
                and MEANING_ACTION in route_actions
            )
            if risky_sfx:
                baseline_sfx_edit += 1
                if row["review_only_nonbubble_signal"]:
                    caught_sfx_edit += 1
            if (
                row["processing_action"] == MEANING_ACTION
                and row["review_only_nonbubble_signal"]
            ):
                meaningful_review += 1
        run_metrics[route] = {
            "baseline_sfx_or_decorative_auto_edit_count": baseline_sfx_edit,
            "caught_by_review_only_nonbubble_signal_count": caught_sfx_edit,
            "meaningful_text_sent_to_review_count": meaningful_review,
            "meaningful_text_auto_hidden_count": 0,
        }

    result = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "evaluation": "locked_truth_sfx_shadow",
        "suite_id": predictions["suite_id"],
        "manifest_sha256": predictions["manifest_sha256"],
        "truth_contract_sha256": truth_lock["truth_contract_sha256"],
        "prediction_contract_sha256": predictions["prediction_contract_sha256"],
        "minimum_bbox_overlap_strength": minimum_overlap,
        "page_count": len(predictions["pages"]),
        "truth_region_count": len(region_rows),
        "role_counts": dict(sorted(role_counts.items())),
        "role_signal_counts": dict(sorted(role_hit_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "action_signal_counts": dict(sorted(action_hit_counts.items())),
        "sfx_or_decorative_count": sum(
            role_counts[role] for role in SFX_ROLES
        ),
        "sfx_or_decorative_signal_count": sum(
            role_hit_counts[role] for role in SFX_ROLES
        ),
        "meaningful_text_signal_count": action_hit_counts[MEANING_ACTION],
        "review_only_nonbubble_signal_count": sum(
            bool(row["review_only_nonbubble_signal"]) for row in region_rows
        ),
        "automatic_preserve_count": 0,
        "meaningful_text_auto_hidden_count": 0,
        "route_metrics": run_metrics,
        "region_diagnostics": region_rows,
        "promotion_allowed": False,
        "promotion_blockers": [
            "User review of locked source-first truth is incomplete.",
            "The signal is review-only and has not proven net route-level benefit.",
            "ABCNetv2 product licensing requires separate clearance.",
        ],
    }
    result["report_sha256"] = three_way.canonical_sha256(result)
    output = three_way.require_external_path(output_dir, label="COO shadow output")
    output.mkdir(parents=True, exist_ok=True)
    three_way.write_json(output / "shadow-evaluation.json", result)
    _write_shadow_report(output / "shadow-evaluation-ko.md", result)
    return result


def _write_shadow_report(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# COO SFX shadow 평가",
        "",
        f"- pages: {result['page_count']}",
        f"- truth regions: {result['truth_region_count']}",
        f"- SFX/decorative: {result['sfx_or_decorative_count']}",
        f"- SFX/decorative signaled: {result['sfx_or_decorative_signal_count']}",
        f"- meaningful text signaled: {result['meaningful_text_signal_count']}",
        f"- automatic preserve: {result['automatic_preserve_count']}",
        f"- meaningful text auto-hidden: {result['meaningful_text_auto_hidden_count']}",
        "",
        "COO 신호는 말풍선 밖 영역을 review로 낮추는 데만 사용했다. 자동 preserve 또는 자동 삭제 권한은 없다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--predictions", type=Path, required=True)
    validate_parser.add_argument("--manifest", type=Path, required=True)

    compare_parser = subparsers.add_parser("compare-devices")
    compare_parser.add_argument("--cpu", type=Path, required=True)
    compare_parser.add_argument("--cuda", type=Path, required=True)
    compare_parser.add_argument("--manifest", type=Path, required=True)
    compare_parser.add_argument("--output", type=Path, required=True)

    evaluate_parser = subparsers.add_parser("evaluate-shadow")
    evaluate_parser.add_argument("--predictions", type=Path, required=True)
    evaluate_parser.add_argument("--manifest", type=Path, required=True)
    evaluate_parser.add_argument("--truth-dir", type=Path, required=True)
    evaluate_parser.add_argument("--run", type=Path, action="append", default=[])
    evaluate_parser.add_argument("--minimum-overlap", type=float, default=0.30)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate":
            result = validate_predictions(args.predictions, args.manifest)
        elif args.command == "compare-devices":
            result = compare_devices(
                cpu_predictions=args.cpu,
                cuda_predictions=args.cuda,
                corpus_manifest=args.manifest,
                output_dir=args.output,
            )
        elif args.command == "evaluate-shadow":
            result = evaluate_shadow(
                predictions_path=args.predictions,
                corpus_manifest=args.manifest,
                truth_dir=args.truth_dir,
                output_dir=args.output,
                normalized_runs=args.run,
                minimum_overlap=args.minimum_overlap,
            )
        else:  # pragma: no cover
            raise ContractError(f"Unknown command: {args.command}")
    except (ContractError, three_way.ContractError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
