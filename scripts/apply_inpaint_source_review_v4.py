#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)
from scripts.validation_artifact_harness import default_archive_root  # noqa: E402


SCHEMA_VERSION = "inpaint-factorized-source-decisions-v4"
REVIEW_SCHEMA_VERSION = "inpaint-source-review-decisions-v4"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_mask(path: str) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return np.where(image > 0, 255, 0).astype(np.uint8)


def _write_mask(path: Path, mask: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", mask)
    if not success:
        raise ValueError(f"unable to encode mask: {path}")
    encoded.tofile(path)
    return str(path.resolve())


def _indexed(values: object, field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{field} must be an array")
    result: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{field} entry must be an object")
        identity = str(value.get(field) or "").strip()
        if not identity or identity in result:
            raise ValueError(f"{field} is empty or duplicated: {identity}")
        result[identity] = value
    return result


def _normalized_review_decisions(
    ledger: dict[str, Any], decisions: dict[str, Any]
) -> dict[str, dict[str, str]]:
    if ledger.get("candidate_seen") is not False:
        raise ValueError("review ledger must be candidate blind")
    if decisions.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("unsupported source review decisions schema")
    if decisions.get("candidate_seen") is not False:
        raise ValueError("review decisions must be frozen before candidates")
    ledger_rows = _indexed(ledger.get("rows"), "review_id")
    decision_rows = _indexed(decisions.get("decisions"), "review_id")
    if set(ledger_rows) != set(decision_rows):
        missing = sorted(set(ledger_rows).difference(decision_rows))
        extra = sorted(set(decision_rows).difference(ledger_rows))
        raise ValueError(f"review decisions differ from ledger: missing={missing} extra={extra}")
    normalized: dict[str, dict[str, str]] = {}
    for review_id, value in decision_rows.items():
        action = str(value.get("decision") or "").strip().lower()
        role = str(value.get("semantic_role") or "").strip().lower()
        if action not in {"required", "preserve", "ambiguous"}:
            raise ValueError(f"invalid review decision: {review_id}={action}")
        if action == "required" and role not in {
            "dialogue_bubble", "dialogue_free", "narration", "ui_or_sign"
        }:
            raise ValueError(f"required review lacks a text role: {review_id}")
        if action == "preserve" and role not in {"sfx", "decorative"}:
            raise ValueError(f"preserve review lacks a preserve role: {review_id}")
        if action == "ambiguous" and role != "ambiguous":
            raise ValueError(f"ambiguous review must use ambiguous role: {review_id}")
        normalized[review_id] = {"decision": action, "semantic_role": role}
    return normalized


def apply_source_review(
    proposals_path: Path,
    ledger_path: Path,
    review_decisions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    proposals = _read_json(proposals_path)
    if proposals.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported source proposal schema")
    if proposals.get("candidate_seen") is not False:
        raise ValueError("source proposals must be candidate blind")
    ledger = _read_json(ledger_path)
    review = _normalized_review_decisions(
        ledger, _read_json(review_decisions_path)
    )
    review_by_instance = {
        (str(row["page_id"]), str(row["instance_id"])): review[review_id]
        for review_id, row in _indexed(ledger.get("rows"), "review_id").items()
    }
    used_reviews: set[tuple[str, str]] = set()
    page_payloads: list[dict[str, Any]] = []
    counts = {"required": 0, "preserve": 0, "ambiguous": 0}
    for page in proposals.get("pages", []):
        if not isinstance(page, dict):
            raise ValueError("proposal page must be an object")
        page_id = str(page.get("page_id") or "").strip()
        page_dir = output_dir / "pages" / page_id
        instances: list[dict[str, Any]] = []
        instance_masks: dict[str, np.ndarray] = {}
        shape: tuple[int, int] | None = None
        for raw_instance in page.get("target_instances", []):
            instance = dict(raw_instance)
            instance_id = str(instance.get("instance_id") or "").strip()
            mask = _read_mask(str(instance.get("mask_path") or ""))
            if shape is None:
                shape = mask.shape
            elif mask.shape != shape:
                raise ValueError(f"instance shape mismatch: {page_id}/{instance_id}")
            record = review_by_instance.get((page_id, instance_id))
            if record is not None:
                used_reviews.add((page_id, instance_id))
                instance["semantic_role"] = record["semantic_role"]
                decision = record["decision"]
                if decision == "required":
                    instance.update(processing_action="translate_inpaint", priority="required")
                elif decision == "preserve":
                    instance.update(processing_action="preserve", priority="optional")
                else:
                    instance.update(processing_action="review", priority="ambiguous")
                instance["source_reviewed"] = True
            instance_masks[instance_id] = mask
            instances.append(instance)
        if shape is None:
            shape = _read_mask(str(page.get("ownership_mask") or "")).shape
        unions = {
            "required": np.zeros(shape, np.uint8),
            "preserve": np.zeros(shape, np.uint8),
            "ambiguous": np.zeros(shape, np.uint8),
        }
        region_ambiguous: dict[str, np.ndarray] = {
            str(region.get("region_id") or ""): np.zeros(shape, np.uint8)
            for region in page.get("regions", [])
        }
        for instance in instances:
            mask = instance_masks[str(instance["instance_id"])]
            priority = str(instance.get("priority") or "")
            bucket = {
                "required": "required",
                "optional": "preserve",
                "ambiguous": "ambiguous",
            }.get(priority)
            if bucket is None:
                raise ValueError(f"invalid reviewed priority: {priority}")
            if np.any((unions[bucket] > 0) & (mask > 0)):
                raise ValueError(f"overlapping reviewed instances: {page_id}")
            unions[bucket][mask > 0] = 255
            counts[bucket] += 1
            if bucket == "ambiguous":
                region_id = str(instance.get("region_id") or "")
                region_ambiguous[region_id][mask > 0] = 255
        occupied = np.where(
            (unions["required"] > 0)
            | (unions["preserve"] > 0)
            | (unions["ambiguous"] > 0),
            255,
            0,
        ).astype(np.uint8)
        protected = _read_mask(str(page.get("protected_structure_mask") or ""))
        protected[occupied > 0] = 0
        regions: list[dict[str, Any]] = []
        for raw_region in page.get("regions", []):
            region = dict(raw_region)
            region_id = str(region.get("region_id") or "")
            local_ambiguous = region_ambiguous[region_id]
            local_protected = _read_mask(str(region.get("protected_structure_mask") or ""))
            local_protected[occupied > 0] = 0
            region["ambiguous_structure_mask"] = _write_mask(
                page_dir / "regions" / region_id / "ambiguous.png",
                local_ambiguous,
            )
            region["protected_structure_mask"] = _write_mask(
                page_dir / "regions" / region_id / "protected.png",
                local_protected,
            )
            region["source_reviewed"] = True
            regions.append(region)
        if not regions:
            region_id = "region-page-empty"
            empty = np.zeros(shape, np.uint8)
            regions.append(
                {
                    "region_id": region_id,
                    "bubble_route_class": "ambiguous",
                    "bubble_interior_mask": _write_mask(
                        page_dir / "regions" / region_id / "bubble-interior.png",
                        empty,
                    ),
                    "ownership_mask": _write_mask(
                        page_dir / "regions" / region_id / "ownership.png", empty
                    ),
                    "protected_structure_mask": _write_mask(
                        page_dir / "regions" / region_id / "protected.png", empty
                    ),
                    "ambiguous_structure_mask": _write_mask(
                        page_dir / "regions" / region_id / "ambiguous.png", empty
                    ),
                    "corner_protect_mask": _write_mask(
                        page_dir / "regions" / region_id / "corner.png", empty
                    ),
                    "source_reviewed": True,
                    "proposal": {
                        "empty_no_edit_page": True,
                        "candidate_seen": False,
                    },
                }
            )
        result = dict(page)
        target_path = (
            _write_mask(page_dir / "target-text.png", unions["required"])
            if np.any(unions["required"])
            else None
        )
        result.update(
            target_text_mask=target_path,
            preserve_mask=_write_mask(page_dir / "preserve.png", unions["preserve"]),
            ambiguous_structure_mask=_write_mask(page_dir / "ambiguous-structure.png", unions["ambiguous"]),
            protected_structure_mask=_write_mask(page_dir / "protected-structure.png", protected),
            target_instances=instances,
            regions=regions,
            expected_edit="required" if np.any(unions["required"]) else "none",
            reviewed_source_only=True,
            review_complete=True,
            candidate_seen=False,
        )
        page_payloads.append(result)
    if used_reviews != set(review_by_instance):
        unused = sorted(set(review_by_instance).difference(used_reviews))
        raise ValueError(f"review rows did not match source instances: {unused}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": proposals.get("corpus_id"),
        "candidate_seen": False,
        "review_complete": True,
        "review_ledger": str(ledger_path.resolve()),
        "review_decisions": str(review_decisions_path.resolve()),
        "counts": counts,
        "pages": page_payloads,
    }
    output_path = output_dir / "source-decisions-v4-reviewed.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def validate_reviewed_manifest(path: Path) -> None:
    for page in load_stage1_manifest(path):
        source = cv2.imdecode(
            np.fromfile(page.source_image, dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if source is None or source.size == 0:
            raise FileNotFoundError(page.source_image)
        load_page_masks(page, source.shape[:2])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply candidate-blind semantic review decisions to v4 masks."
    )
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    archive = default_archive_root().resolve()
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(archive)
    except ValueError as exc:
        raise ValueError("reviewed output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    payload = apply_source_review(
        args.proposals.resolve(),
        args.review_ledger.resolve(),
        args.review_decisions.resolve(),
        output_dir,
    )
    print(json.dumps({"pages": len(payload["pages"]), "counts": payload["counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
