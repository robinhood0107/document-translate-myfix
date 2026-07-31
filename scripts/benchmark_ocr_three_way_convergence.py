#!/usr/bin/env python3
"""Replay current MangaLMM debug evidence into the locked OCR review contract.

The live debug runner records the selected full-page response plus detector
geometry.  This tool replays only parsing and reconciliation; it never starts
Docker or performs model inference.  The result is a normalised run accepted
by ``benchmark_ocr_three_way_human_truth.py`` so current product logic can be
compared with the already locked source-first truth without re-reading the
model output by hand.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import benchmark_ocr_three_way_human_truth as three_way
from modules.ocr.mangalmm_full_page.engine import MangaLMMOCREngine
from modules.ocr.mangalmm_full_page.image_policy import ResizePlan
from modules.ocr.mangalmm_full_page.response_parser import (
    MangaLMMResponseContractError,
    parse_mangalmm_response,
)
from modules.utils.textblock import TextBlock


REPLAY_SCHEMA_VERSION = 1
DECISION_FIELDS = tuple(sorted(three_way.REVIEW_DECISION_SUFFIXES))


class ReplayContractError(three_way.ContractError):
    """Raised when live debug evidence cannot be replayed safely."""


def _finite_float(value: object, *, label: str) -> float:
    if isinstance(value, bool):
        raise ReplayContractError(f"{label} must be a finite number.")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ReplayContractError(f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ReplayContractError(f"{label} must be a finite number.")
    return parsed


def _positive_int(value: object, *, label: str) -> int:
    if isinstance(value, bool):
        raise ReplayContractError(f"{label} must be a positive integer.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ReplayContractError(f"{label} must be a positive integer.") from exc
    if parsed <= 0:
        raise ReplayContractError(f"{label} must be a positive integer.")
    return parsed


def _box(value: object, *, label: str) -> list[int]:
    if not isinstance(value, list) or len(value) != 4:
        raise ReplayContractError(f"{label} must contain four coordinates.")
    result = [int(_finite_float(item, label=label)) for item in value]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ReplayContractError(f"{label} must be non-empty.")
    return result


def _optional_box(value: object, *, label: str) -> list[int] | None:
    if value is None:
        return None
    return _box(value, label=label)


def _resize_plan(request: Mapping[str, Any]) -> ResizePlan:
    original_shape = request.get("original_shape")
    request_shape = request.get("request_shape")
    if not isinstance(original_shape, list) or len(original_shape) != 2:
        raise ReplayContractError("Manga debug request lacks original_shape.")
    if not isinstance(request_shape, list) or len(request_shape) != 2:
        raise ReplayContractError("Manga debug request lacks request_shape.")
    return ResizePlan(
        profile=str(request.get("resize_profile", "") or "standard"),
        original_shape=(
            _positive_int(original_shape[0], label="original height"),
            _positive_int(original_shape[1], label="original width"),
        ),
        request_shape=(
            _positive_int(request_shape[0], label="request height"),
            _positive_int(request_shape[1], label="request width"),
        ),
        base_scale=_finite_float(
            request.get("base_scale"), label="Manga base scale"
        ),
        scale_x=_finite_float(request.get("scale_x"), label="Manga scale_x"),
        scale_y=_finite_float(request.get("scale_y"), label="Manga scale_y"),
        max_completion_tokens=_positive_int(
            request.get("max_completion_tokens", 4096),
            label="Manga completion token limit",
        ),
        block_count=max(0, int(request.get("block_count", 0) or 0)),
        small_block_ratio=_finite_float(
            request.get("small_block_ratio", 0.0),
            label="Manga small block ratio",
        ),
        text_cover_ratio=_finite_float(
            request.get("text_cover_ratio", 0.0),
            label="Manga text cover ratio",
        ),
    )


def _text_blocks(
    detector_blocks: Sequence[Mapping[str, Any]],
) -> list[TextBlock]:
    blocks: list[TextBlock] = []
    for index, record in enumerate(detector_blocks):
        text_box = _box(record.get("bbox_xyxy"), label=f"detector block {index}")
        bubble_box = _optional_box(
            record.get("bubble_xyxy"),
            label=f"detector bubble {index}",
        )
        blocks.append(
            TextBlock(
                text_bbox=np.asarray(text_box, dtype=np.int32),
                bubble_bbox=(
                    np.asarray(bubble_box, dtype=np.int32)
                    if bubble_box is not None
                    else None
                ),
                text_class=str(record.get("text_class", "") or ""),
                source_lang="ja",
                direction=str(record.get("direction", "") or ""),
                block_id=str(record.get("block_id", "") or ""),
                semantic_role=str(record.get("semantic_role", "") or ""),
                processing_action=str(
                    record.get("processing_action", "") or ""
                ),
            )
        )
    return blocks


def _debug_geometry_matches(
    debug_blocks: object,
    detector_blocks: Sequence[Mapping[str, Any]],
    *,
    page_id: str,
) -> None:
    if not isinstance(debug_blocks, list) or len(debug_blocks) != len(
        detector_blocks
    ):
        raise ReplayContractError(
            f"Live detector block count changed on {page_id}."
        )
    for index, (debug, detector) in enumerate(
        zip(debug_blocks, detector_blocks, strict=True)
    ):
        if not isinstance(debug, dict):
            raise ReplayContractError(
                f"Invalid live detector block on {page_id}:{index}."
            )
        debug_box = _box(
            debug.get("xyxy"), label=f"live detector {page_id}:{index}"
        )
        detector_box = _box(
            detector.get("bbox_xyxy"),
            label=f"locked detector {page_id}:{index}",
        )
        if debug_box != detector_box:
            raise ReplayContractError(
                f"Live detector geometry changed on {page_id}:{index}."
            )


def _region_key(
    *, bbox: Sequence[object], response_bbox: Sequence[object], text: object
) -> tuple[object, ...]:
    return (
        tuple(round(float(value), 6) for value in bbox),
        tuple(round(float(value), 6) for value in response_bbox),
        str(text or "").strip(),
    )


def _replay_page(
    *,
    page: Mapping[str, Any],
    debug_root: Path,
    engine: MangaLMMOCREngine,
) -> tuple[dict[str, Any], list[Path]]:
    page_id = str(page["page_id"])
    summary_path = debug_root / page_id / "page_summary.json"
    raw_path = debug_root / page_id / "raw_response.txt"
    summary = three_way.read_json(summary_path)
    if Path(str(summary.get("image", ""))).stem != Path(
        str(page["source_image"]["path"])
    ).stem:
        raise ReplayContractError(f"Manga debug image changed on {page_id}.")

    source_path = Path(str(summary.get("source_path", "")))
    if not source_path.is_file():
        raise ReplayContractError(f"Manga debug source is missing on {page_id}.")
    if three_way.sha256_file(source_path) != page["source_image"]["sha256"]:
        raise ReplayContractError(f"Manga debug source hash changed on {page_id}.")

    detector_blocks = three_way._detector_blocks_for_page(page)
    _debug_geometry_matches(
        summary.get("blocks"), detector_blocks, page_id=page_id
    )
    request = summary.get("request")
    attempts = summary.get("attempts")
    if not isinstance(request, dict):
        raise ReplayContractError(f"Manga debug request is missing on {page_id}.")
    if not isinstance(attempts, list) or not attempts or any(
        not isinstance(attempt, dict) for attempt in attempts
    ):
        raise ReplayContractError(f"Manga debug attempts are missing on {page_id}.")
    raw_response = str(request.get("raw_response", "") or "")
    if raw_path.read_text(encoding="utf-8") != raw_response:
        raise ReplayContractError(f"Manga raw response changed on {page_id}.")
    try:
        parsed = parse_mangalmm_response(raw_response)
    except MangaLMMResponseContractError as exc:
        raise ReplayContractError(
            f"Selected Manga response no longer parses on {page_id}: {exc.code}"
        ) from exc

    width = int(page["source_image"]["width"])
    height = int(page["source_image"]["height"])
    plan = _resize_plan(request)
    if plan.original_shape != (height, width):
        raise ReplayContractError(
            f"Manga resize origin changed on {page_id}."
        )
    mapped = engine._map_regions_to_page_coords(
        list(parsed.regions),
        (0, 0, width, height),
        (height, width),
        plan,
        "page_full",
    )
    blocks = _text_blocks(detector_blocks)
    assignments = engine._assign_regions_to_blocks(mapped, blocks)

    raw_ids_by_object = {
        id(region): f"manga-live-{index:04d}"
        for index, region in enumerate(mapped)
    }
    raw_records: list[dict[str, Any]] = []
    raw_records_by_id: dict[str, dict[str, Any]] = {}
    for region in mapped:
        raw_id = raw_ids_by_object[id(region)]
        record = {
            "raw_region_id": raw_id,
            "bbox_xyxy": list(region.bbox_xyxy),
            "text": str(region.text or ""),
            "detector_block_ids": [],
            "geometry_source": "mangalmm_full_page_live_replay",
        }
        raw_records.append(record)
        raw_records_by_id[raw_id] = record

    units: list[dict[str, Any]] = []
    referenced_raw_ids: set[str] = set()
    for index, detector in enumerate(detector_blocks):
        items = assignments.get(index, [])
        if not items:
            continue
        raw_ids = [
            raw_ids_by_object[id(item["region"])]
            for item in items
        ]
        block_id = str(detector["block_id"])
        for raw_id in raw_ids:
            raw_records_by_id[raw_id]["detector_block_ids"] = [block_id]
        referenced_raw_ids.update(raw_ids)
        units.append(
            three_way._canonical_unit(
                unit_id=block_id,
                detector_block_ids=[block_id],
                bbox_xyxy=detector["bbox_xyxy"],
                text="\n".join(
                    str(item["region"].text or "").strip()
                    for item in items
                    if str(item["region"].text or "").strip()
                ),
                raw_region_ids=raw_ids,
                geometry_status=(
                    "compound_safe" if len(items) > 1 else "one_to_one"
                ),
                semantic_role=str(detector.get("semantic_role", "") or ""),
                processing_action=str(
                    detector.get("processing_action", "") or ""
                ),
            )
        )

    raw_lookup: dict[tuple[object, ...], deque[str]] = defaultdict(deque)
    for region in mapped:
        raw_lookup[
            _region_key(
                bbox=region.bbox_xyxy_float,
                response_bbox=region.response_bbox_2d or [],
                text=region.text,
            )
        ].append(raw_ids_by_object[id(region)])
    shadow_ids: set[str] = set()
    for shadow in engine.last_shadow_regions:
        key = _region_key(
            bbox=shadow.get("bbox_xyxy_float", []),
            response_bbox=shadow.get("response_bbox_2d", []),
            text=shadow.get("text", ""),
        )
        matches = raw_lookup.get(key)
        if not matches:
            raise ReplayContractError(
                f"Manga shadow region cannot be bound on {page_id}."
            )
        raw_id = matches.popleft()
        if raw_id in referenced_raw_ids or raw_id in shadow_ids:
            continue
        shadow_ids.add(raw_id)
        candidate_ids = [
            str(value)
            for value in shadow.get("candidate_block_ids", [])
            if str(value)
        ]
        raw_records_by_id[raw_id]["detector_block_ids"] = candidate_ids
        reason = str(shadow.get("reason", "") or "full_page_only")
        units.append(
            three_way._canonical_unit(
                unit_id=f"manga-live-extra-{raw_id.rsplit('-', 1)[-1]}",
                detector_block_ids=[],
                bbox_xyxy=raw_records_by_id[raw_id]["bbox_xyxy"],
                text=raw_records_by_id[raw_id]["text"],
                raw_region_ids=[raw_id],
                geometry_status=reason,
                semantic_role=("ambiguous" if candidate_ids else ""),
                processing_action=("review" if candidate_ids else ""),
            )
        )

    # Any remaining raw region was intentionally collapsed as a near-exact
    # duplicate by the product reconciler.  It remains in raw evidence but
    # must not reappear as a candidate-only review unit.
    collapsed_raw_ids = {
        str(record["raw_region_id"])
        for record in raw_records
        if str(record["raw_region_id"])
        not in referenced_raw_ids | shadow_ids
    }

    parser_errors = sum(bool(item.get("parser_error_code")) for item in attempts)
    length_errors = sum(
        str(item.get("finish_reason", "") or "").lower() == "length"
        for item in attempts
    )
    elapsed_seconds = _finite_float(
        summary.get("elapsed_seconds", 0.0),
        label=f"Manga elapsed time {page_id}",
    )
    if elapsed_seconds < 0:
        raise ReplayContractError(f"Manga elapsed time is negative on {page_id}.")
    failure = str(summary.get("failure", "") or "")
    page_result = {
        "page_id": page_id,
        "source_sha256": page["source_image"]["sha256"],
        "image_width": width,
        "image_height": height,
        "status": "failure" if failure else "success",
        "elapsed_seconds": float(elapsed_seconds),
        "attempt_count": len(attempts),
        "retry_count": max(0, len(attempts) - 1),
        "attempt_telemetry_complete": True,
        "parser_error_count": int(parser_errors),
        "length_error_count": int(length_errors),
        "raw_regions": raw_records,
        "canonical_units": units,
        "diagnostics": {
            "replay_schema_version": REPLAY_SCHEMA_VERSION,
            "failure": failure,
            "response_kind": parsed.response_kind,
            "detector_block_count": len(detector_blocks),
            "matched_detector_count": sum(bool(items) for items in assignments.values()),
            "shadow_region_count": len(engine.last_shadow_regions),
            "collapsed_raw_region_count": len(collapsed_raw_ids),
            "merge_split_diagnostics": list(
                engine.last_merge_split_diagnostics
            ),
            "timing_available": "elapsed_seconds" in summary,
        },
    }
    return page_result, [summary_path, raw_path]


def replay_manga_debug_run(
    *,
    corpus_manifest: Path,
    debug_root: Path,
    runtime_contract: Path,
    output_dir: Path,
) -> dict[str, Any]:
    manifest_path = three_way.require_external_path(
        corpus_manifest, label="corpus manifest"
    )
    source_root = three_way.require_external_path(
        debug_root, label="Manga debug root"
    )
    runtime_path = three_way.require_external_path(
        runtime_contract, label="runtime contract"
    )
    output = three_way.require_external_path(
        output_dir, label="Manga replay output"
    )
    if output.exists() and any(output.iterdir()):
        raise ReplayContractError(f"Manga replay output must be empty: {output}")
    manifest = three_way.validate_corpus_manifest(manifest_path)
    runtime = three_way._runtime_contract(
        runtime_path, "mangalmm_full_page"
    )
    engine = MangaLMMOCREngine()
    pages: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    for page in manifest["pages"]:
        page_result, evidence = _replay_page(
            page=page,
            debug_root=source_root,
            engine=engine,
        )
        pages.append(page_result)
        source_files.extend(
            three_way._source_file_record(source_root, path)
            for path in evidence
        )
    source_files.sort(key=lambda item: str(item["relative_path"]))
    result: dict[str, Any] = {
        "schema_version": three_way.RUN_SCHEMA_VERSION,
        "protocol_version": three_way.PROTOCOL_VERSION,
        "suite_id": manifest["suite_id"],
        "route_id": "mangalmm_full_page",
        "created_at": three_way.utc_now(),
        "corpus_manifest_file_sha256": three_way.sha256_file(manifest_path),
        "corpus_manifest_path": str(manifest_path),
        "source_results_root": str(source_root),
        "source_files": source_files,
        "runtime_contract": runtime,
        "runtime_contract_path": str(runtime_path),
        "runtime_contract_file_sha256": three_way.sha256_file(runtime_path),
        "pages": pages,
    }
    result["normalized_contract_sha256"] = three_way.canonical_sha256(result)
    output.mkdir(parents=True, exist_ok=True)
    three_way.write_json(output / three_way.RUN_FILENAME, result)
    three_way.validate_run(output / three_way.RUN_FILENAME)
    return result


def _review_route_labels(review_root: Path) -> dict[str, str]:
    key = three_way.read_json(
        review_root / "private" / three_way.BLIND_KEY_FILENAME
    )
    label_to_route = key.get("label_to_route")
    if not isinstance(label_to_route, dict) or set(label_to_route) != set(
        three_way.BLIND_LABELS
    ):
        raise ReplayContractError("Blind review key has an invalid route map.")
    route_to_label = {
        str(route): str(label) for label, route in label_to_route.items()
    }
    if set(route_to_label) != set(three_way.ROUTES):
        raise ReplayContractError("Blind review key does not contain all routes.")
    return route_to_label


def _evidence_signature(
    row: Mapping[str, str], label: str
) -> tuple[str, ...]:
    return tuple(
        str(row.get(f"{label}_{field}", "") or "")
        for field in (
            "text",
            "bbox_xyxy",
            "semantic_role",
            "processing_action",
            "geometry_status",
        )
    )


def _copy_decisions(
    *,
    source: Mapping[str, str],
    source_label: str,
    target: dict[str, str],
    target_label: str,
    fields: Sequence[str],
) -> int:
    copied = 0
    for field in fields:
        source_value = str(source.get(f"{source_label}_{field}", "") or "")
        target_key = f"{target_label}_{field}"
        target_value = str(target.get(target_key, "") or "")
        if not source_value:
            continue
        if target_value and target_value != source_value:
            raise ReplayContractError(
                f"Target review already has a conflicting decision: {target_key}"
            )
        target[target_key] = source_value
        copied += not bool(target_value)
    return copied


def transfer_review_decisions(
    *,
    completed_review: Path,
    target_review: Path,
) -> dict[str, Any]:
    source_root = three_way.require_external_path(
        completed_review, label="completed review"
    )
    target_root = three_way.require_external_path(
        target_review, label="target review"
    )
    source_metrics_path = source_root / three_way.FINAL_METRICS_FILENAME
    source_metrics = three_way.read_json(source_metrics_path)
    source_csv = source_root / three_way.REVIEW_CSV_FILENAME
    target_csv = target_root / three_way.REVIEW_CSV_FILENAME
    if source_metrics.get("completed_review_csv_sha256") != three_way.sha256_file(
        source_csv
    ):
        raise ReplayContractError("Completed review CSV no longer matches its metrics.")
    source_rows = three_way._read_review_rows(source_csv)
    source_errors = three_way.validate_review_rows(source_rows)
    if source_errors:
        raise ReplayContractError(
            f"Source review is not complete ({len(source_errors)} errors)."
        )
    target_rows = three_way._read_review_rows(target_csv)
    source_labels = _review_route_labels(source_root)
    target_labels = _review_route_labels(target_root)

    source_truth = {
        (str(row["page_id"]), str(row["truth_region_id"])): row
        for row in source_rows
        if row["row_kind"] == "truth"
    }
    source_extras: dict[
        tuple[str, str, tuple[str, ...]], deque[dict[str, str]]
    ] = defaultdict(deque)
    for row in source_rows:
        if row["row_kind"] != "candidate_extra":
            continue
        for route, label in source_labels.items():
            if not str(row.get(f"{label}_text", "") or "").strip():
                continue
            source_extras[
                (str(row["page_id"]), route, _evidence_signature(row, label))
            ].append(row)

    copied_by_route = {route: 0 for route in three_way.ROUTES}
    unchanged_truth_by_route = {route: 0 for route in three_way.ROUTES}
    matched_extra_by_route = {route: 0 for route in three_way.ROUTES}
    for row in target_rows:
        if row["row_kind"] == "truth":
            source = source_truth.get(
                (str(row["page_id"]), str(row["truth_region_id"]))
            )
            if source is None:
                continue
            for route in three_way.ROUTES:
                source_label = source_labels[route]
                target_label = target_labels[route]
                source_signature = _evidence_signature(source, source_label)
                target_signature = _evidence_signature(row, target_label)
                if source_signature[0] != target_signature[0]:
                    continue
                fields = ["transcription_correct", "semantic_correct"]
                if source_signature[2:4] == target_signature[2:4]:
                    fields.append("role_action_correct")
                if source_signature[4] == target_signature[4]:
                    fields.extend(("merge_split_error", "destructive_edit"))
                if source_signature == target_signature:
                    fields.append("notes")
                copied_by_route[route] += _copy_decisions(
                    source=source,
                    source_label=source_label,
                    target=row,
                    target_label=target_label,
                    fields=fields,
                )
                unchanged_truth_by_route[route] += 1
            continue

        for route in three_way.ROUTES:
            target_label = target_labels[route]
            if not str(row.get(f"{target_label}_text", "") or "").strip():
                continue
            key = (
                str(row["page_id"]),
                route,
                _evidence_signature(row, target_label),
            )
            candidates = source_extras.get(key)
            if not candidates:
                continue
            source = candidates.popleft()
            copied_by_route[route] += _copy_decisions(
                source=source,
                source_label=source_labels[route],
                target=row,
                target_label=target_label,
                fields=DECISION_FIELDS,
            )
            matched_extra_by_route[route] += 1

    three_way._write_review_csv(target_csv, target_rows)
    pending = three_way.validate_review_rows(target_rows)
    result: dict[str, Any] = {
        "schema_version": 1,
        "protocol_version": three_way.PROTOCOL_VERSION,
        "completed_review": str(source_root),
        "completed_review_metrics_sha256": three_way.sha256_file(
            source_metrics_path
        ),
        "target_review": str(target_root),
        "copied_decision_cells_by_route": copied_by_route,
        "unchanged_truth_rows_by_route": unchanged_truth_by_route,
        "matched_candidate_extra_rows_by_route": matched_extra_by_route,
        "pending_error_count": len(pending),
        "pending_errors": pending,
        "target_review_csv_sha256_after_transfer": three_way.sha256_file(
            target_csv
        ),
    }
    result["transfer_contract_sha256"] = three_way.canonical_sha256(result)
    three_way.write_json(
        target_root / "private" / "decision-transfer.json", result
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay MangaLMM live debug evidence into the locked OCR benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    replay = subparsers.add_parser("replay-manga-debug")
    replay.add_argument("--corpus-manifest", type=Path, required=True)
    replay.add_argument("--debug-root", type=Path, required=True)
    replay.add_argument("--runtime-contract", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    transfer = subparsers.add_parser("transfer-decisions")
    transfer.add_argument("--completed-review", type=Path, required=True)
    transfer.add_argument("--target-review", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "replay-manga-debug":
            result = replay_manga_debug_run(
                corpus_manifest=args.corpus_manifest,
                debug_root=args.debug_root,
                runtime_contract=args.runtime_contract,
                output_dir=args.output_dir,
            )
        else:
            result = transfer_review_decisions(
                completed_review=args.completed_review,
                target_review=args.target_review,
            )
    except (ReplayContractError, three_way.ContractError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.command == "replay-manga-debug":
        output_payload = {
            "route_id": result["route_id"],
            "page_count": len(result["pages"]),
            "normalized_run": str(
                Path(args.output_dir) / three_way.RUN_FILENAME
            ),
        }
    else:
        output_payload = {
            "pending_error_count": result["pending_error_count"],
            "copied_decision_cells_by_route": result[
                "copied_decision_cells_by_route"
            ],
            "decision_transfer": str(
                Path(args.target_review) / "private" / "decision-transfer.json"
            ),
        }
    print(json.dumps(output_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
