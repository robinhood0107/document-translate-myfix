#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
    manifest_page_artifact_sha256,
    source_manifest_page_inventory_sha256,
    validate_source_only_manifest_v4,
)
from scripts.validation_artifact_harness import default_archive_root  # noqa: E402


SCHEMA_VERSION = "inpaint-factorized-source-manifest-v4"
LEDGER_SCHEMA_VERSION = "inpaint-independent-target-review-ledger-v4"
DECISIONS_SCHEMA_VERSION = "inpaint-independent-target-review-decisions-v4"
MANUAL_INVENTORY_SCHEMA_VERSION = "inpaint-independent-manual-inventory-v4"
SEAL_SCHEMA_VERSION = "inpaint-factorized-manifest-seal-v4-independent"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_mask(path: object, shape: tuple[int, int] | None = None) -> np.ndarray:
    if path is None or not str(path).strip():
        if shape is None:
            raise FileNotFoundError(path)
        return np.zeros(shape, np.uint8)
    value = cv2.imdecode(np.fromfile(Path(str(path)), dtype=np.uint8), 0)
    if value is None or value.size == 0:
        raise FileNotFoundError(path)
    if shape is not None and value.shape != shape:
        raise ValueError("independent target mask shape mismatch")
    return np.where(value > 0, 255, 0).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manual_inventory_instances(
    inventory_path: Path,
    *,
    page_id: str,
    source_path: Path,
    shape: tuple[int, int],
    page_dir: Path,
    regions: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    payload = _read_json(inventory_path)
    if payload.get("schema_version") != MANUAL_INVENTORY_SCHEMA_VERSION:
        raise ValueError(f"unsupported manual inventory: {inventory_path}")
    if payload.get("candidate_seen") is not False or payload.get("source_reviewed") is not True:
        raise ValueError(f"manual inventory must be source-only reviewed: {page_id}")
    if str(payload.get("page_id") or "") != page_id:
        raise ValueError(f"manual inventory page mismatch: {page_id}")
    expected_sha = str(payload.get("source_sha256") or "").lower()
    if not expected_sha or expected_sha != _sha256(source_path).lower():
        raise ValueError(f"manual inventory source SHA mismatch: {page_id}")
    raw_instances = payload.get("instances")
    if not isinstance(raw_instances, list) or not raw_instances:
        raise ValueError(f"manual inventory needs reviewed instances: {page_id}")

    occupied = np.zeros(shape, np.uint8)
    required = np.zeros(shape, np.uint8)
    preserve = np.zeros(shape, np.uint8)
    ambiguous = np.zeros(shape, np.uint8)
    output: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_instances:
        if not isinstance(raw, dict):
            raise ValueError(f"manual inventory instance must be an object: {page_id}")
        instance_id = str(raw.get("instance_id") or "")
        if not instance_id or instance_id in seen_ids:
            raise ValueError(f"invalid manual inventory instance id: {page_id}={instance_id}")
        seen_ids.add(instance_id)
        region_id = str(raw.get("region_id") or "")
        region = regions.get(region_id)
        if region is None:
            raise ValueError(f"manual inventory needs authoritative ownership: {page_id}={region_id}")
        priority = str(raw.get("priority") or "")
        if priority not in {"required", "optional", "ambiguous"}:
            raise ValueError(f"invalid manual inventory priority: {page_id}={priority}")
        mask = _read_mask(raw.get("mask_path"), shape)
        ownership = _read_mask(region["ownership_mask"], shape)
        mask[ownership == 0] = 0
        for field in (
            "protected_structure_mask",
            "ambiguous_structure_mask",
            "corner_protect_mask",
        ):
            mask[_read_mask(region[field], shape) > 0] = 0
        mask[occupied > 0] = 0
        if not np.any(mask):
            raise ValueError(f"manual inventory instance became empty: {page_id}={instance_id}")
        occupied[mask > 0] = 255
        if priority == "required":
            bucket = required
            default_role = "dialogue_bubble"
            action = "translate_inpaint"
        elif priority == "optional":
            bucket = preserve
            default_role = "sfx"
            action = "preserve"
        else:
            bucket = ambiguous
            default_role = "ambiguous"
            action = "review"
        bucket[mask > 0] = 255
        destination = page_dir / "instances" / f"manual-{instance_id}.png"
        output.append(
            {
                "instance_id": f"manual-{instance_id}",
                "region_id": region_id,
                "mask_path": _write_mask(destination, mask),
                "semantic_role": str(raw.get("semantic_role") or default_role),
                "processing_action": action,
                "priority": priority,
                "source_reviewed": True,
            }
        )
    if not np.any(required):
        raise ValueError(f"manual inventory has no required text: {page_id}")
    return output, required, preserve, ambiguous


def _write_mask(path: Path, value: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(".png", np.asarray(value))
    if not success:
        raise OSError(f"unable to encode mask: {path}")
    encoded.tofile(path)
    return str(path.resolve())


def _index(values: object, key: str) -> dict[str, dict[str, Any]]:
    if not isinstance(values, list):
        raise ValueError(f"{key} records must be an array")
    output: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict):
            raise ValueError(f"{key} record must be an object")
        identity = str(value.get(key) or "")
        if not identity or identity in output:
            raise ValueError(f"invalid or duplicate {key}: {identity}")
        output[identity] = value
    return output


def _empty_region(
    page_dir: Path,
    shape: tuple[int, int],
    region_id: str,
    *,
    ownership: np.ndarray | None = None,
) -> dict[str, Any]:
    empty = np.zeros(shape, np.uint8)
    exact_ownership = empty if ownership is None else np.asarray(ownership)
    region_dir = page_dir / "regions" / region_id
    return {
        "region_id": region_id,
        "bubble_route_class": "ambiguous",
        "bubble_interior_mask": _write_mask(region_dir / "bubble-interior.png", empty),
        "ownership_mask": _write_mask(
            region_dir / "ownership.png", exact_ownership
        ),
        "protected_structure_mask": _write_mask(region_dir / "protected.png", empty),
        "ambiguous_structure_mask": _write_mask(region_dir / "ambiguous.png", empty),
        "corner_protect_mask": _write_mask(region_dir / "corner.png", empty),
        "source_reviewed": True,
    }


def apply_independent_target_review(
    semantic_manifest_path: Path,
    ledger_path: Path,
    decisions_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    semantic = _read_json(semantic_manifest_path)
    ledger = _read_json(ledger_path)
    decisions = _read_json(decisions_path)
    if semantic.get("schema_version") != "inpaint-factorized-source-decisions-v4":
        raise ValueError("unsupported source-only semantic review manifest")
    if (
        semantic.get("candidate_seen") is not False
        or semantic.get("review_complete") is not True
    ):
        raise ValueError("semantic review input is not source-only review complete")
    semantic_pages = semantic.get("pages")
    if not isinstance(semantic_pages, list) or not semantic_pages:
        raise ValueError("semantic review input requires reviewed pages")
    for semantic_page in semantic_pages:
        if (
            not isinstance(semantic_page, dict)
            or semantic_page.get("candidate_seen") is not False
            or semantic_page.get("reviewed_source_only") is not True
            or semantic_page.get("review_complete") is not True
        ):
            raise ValueError("semantic review page is not source-only review complete")
    if ledger.get("schema_version") != LEDGER_SCHEMA_VERSION:
        raise ValueError("unsupported independent target review ledger")
    if (
        str(ledger.get("semantic_manifest") or "").strip()
        and Path(str(ledger["semantic_manifest"])).resolve()
        != semantic_manifest_path.resolve()
    ):
        raise ValueError("independent target review ledger binds another semantic manifest")
    if ledger.get("semantic_manifest_sha256") != _sha256(semantic_manifest_path):
        raise ValueError("independent target review ledger semantic manifest SHA differs")

    if decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported independent target review decisions")
    if ledger.get("candidate_seen") is not False or decisions.get("candidate_seen") is not False:
        raise ValueError("independent target review must be candidate blind")
    if decisions.get("review_complete") is not True:
        raise ValueError("independent target review decisions are incomplete")
    declared_ledger = str(decisions.get("review_ledger") or "").strip()
    if not declared_ledger or Path(declared_ledger).resolve() != ledger_path.resolve():
        raise ValueError("independent target decisions bind a different review ledger")
    if decisions.get("review_ledger_sha256") != _sha256(ledger_path):
        raise ValueError("independent target decisions review ledger SHA differs")

    source_index: dict[str, dict[str, Any]] = {}
    needs_source_index = any(
        not all(field in page for field in ("path", "height", "width"))
        for page in semantic_pages
    )
    if needs_source_index:
        declared_source_index = str(ledger.get("source_index") or "").strip()
        if not declared_source_index:
            raise ValueError("independent target review ledger lacks its source index")
        source_index_path = Path(declared_source_index).resolve()
        if ledger.get("source_index_sha256") != _sha256(source_index_path):
            raise ValueError("independent target review ledger source index SHA differs")
        source_index_payload = _read_json(source_index_path)
        source_index = _index(source_index_payload.get("pages"), "page_id")
        if set(source_index) != {
            str(page.get("page_id") or "") for page in semantic_pages
        }:
            raise ValueError("independent target review source index page set differs")

    pages = _index(semantic_pages, "page_id")
    rows = _index(ledger.get("rows"), "review_id")
    selected = _index(decisions.get("decisions"), "review_id")
    if set(selected) != set(rows):
        raise ValueError("independent target decisions differ from review ledger")
    inventory = _index(decisions.get("full_page_inventory"), "page_id")
    pending = {
        str(value.get("page_id") or "")
        for value in ledger.get("full_page_inventory_pending", [])
    }
    if set(inventory) != pending:
        raise ValueError("full-page inventory decisions differ from review ledger")

    rows_by_page: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for review_id, row in rows.items():
        page_id = str(row.get("page_id") or "")
        if page_id not in pages:
            raise ValueError(f"review row references unknown page: {page_id}")
        rows_by_page.setdefault(page_id, []).append((row, selected[review_id]))

    output_pages: list[dict[str, Any]] = []
    total_instances = {"required": 0, "preserve": 0, "ambiguous": 0}
    for page_id, source_page in pages.items():
        page = dict(source_page)
        if page_id in source_index:
            source_record = source_index[page_id]
            for field in (
                "path",
                "height",
                "width",
                "source_sha256",
                "set_id",
                "paired_reference",
            ):
                if field in source_record:
                    page[field] = source_record[field]
        height, width = int(page["height"]), int(page["width"])
        shape = (height, width)
        page_dir = output_dir / "pages" / page_id
        page_rows = rows_by_page.get(page_id, [])
        unowned_ownership = np.zeros(shape, np.uint8)
        required = np.zeros(shape, np.uint8)
        preserve = np.zeros(shape, np.uint8)
        ambiguous = np.zeros(shape, np.uint8)

        inventory_entry = inventory.get(page_id)
        inventory_status = (
            str(inventory_entry.get("status") or "") if inventory_entry else ""
        )
        if inventory_entry and inventory_status != "complete_with_reviewed_rows":
            entry = inventory_entry
            manual = str(entry.get("manual_inventory_path") or "")
            if manual:
                region_map = {
                    str(value.get("region_id") or ""): dict(value)
                    for value in page.get("regions", [])
                }
                instances, required, preserve, ambiguous = _manual_inventory_instances(
                    Path(manual),
                    page_id=page_id,
                    source_path=Path(str(page["path"])),
                    shape=shape,
                    page_dir=page_dir,
                    regions=region_map,
                )
                page["target_instances"] = instances
                page["target_text_mask"] = _write_mask(
                    page_dir / "target-text.png", required
                )
                page["preserve_mask"] = _write_mask(page_dir / "preserve.png", preserve)
                page["ambiguous_structure_mask"] = _write_mask(
                    page_dir / "ambiguous-structure.png", ambiguous
                )
            else:
                # The independent full-page inventory supersedes the old
                # detector-derived semantic proposal.  A reviewed no-target
                # page must not retain circular target instances.
                page["target_instances"] = []
                page["target_text_mask"] = None
            page["target_mask_provenance"] = "source_only_full_page_inventory_review"
        else:
            instances: list[dict[str, Any]] = []
            region_map = {
                str(value.get("region_id") or ""): dict(value)
                for value in page.get("regions", [])
            }
            for row, decision in sorted(page_rows, key=lambda value: value[0]["review_id"]):
                semantic_decision = str(decision.get("semantic") or "")
                if semantic_decision == "not_text":
                    continue
                extent = str(decision.get("extent") or "")
                if semantic_decision == "ambiguous" and extent == "reject":
                    extent_path = row.get("location_seed")
                elif extent == "manual":
                    extent_path = decision.get("manual_extent_path")
                    if not extent_path:
                        raise ValueError(
                            f"manual review extent missing: {row['review_id']}"
                        )
                elif extent == "location":
                    extent_path = row.get("location_seed")
                else:
                    variants = row.get("extent_variants")
                    if not isinstance(variants, dict) or extent not in variants:
                        raise ValueError(
                            f"selected extent unavailable: {row['review_id']}={extent}"
                        )
                    extent_path = variants[extent]
                mask = _read_mask(extent_path, shape)
                region_id = str(row.get("region_id") or "")
                region = region_map.get(region_id)
                hard_protect = np.zeros(shape, np.uint8)
                if region is not None:
                    # A candidate-blind human target/preserve decision is more
                    # authoritative than the old derived structure proxy.  The
                    # explicitly reviewed ambiguous and corner masks remain
                    # hard exclusions.
                    for field in (
                        "ambiguous_structure_mask",
                        "corner_protect_mask",
                    ):
                        hard_protect[_read_mask(region[field], shape) > 0] = 255
                    ownership = _read_mask(region["ownership_mask"], shape)
                    mask[ownership == 0] = 0
                mask[hard_protect > 0] = 0
                if not np.any(mask):
                    raise ValueError(f"selected review extent became empty: {row['review_id']}")
                if semantic_decision == "required":
                    bucket = required
                    priority = "required"
                    action = "translate_inpaint"
                    role = str(row.get("semantic_role_proposal") or "dialogue_free")
                    if role not in {"dialogue_bubble", "dialogue_free", "narration", "ui_or_sign"}:
                        role = "dialogue_free"
                elif semantic_decision == "preserve":
                    bucket = preserve
                    priority = "optional"
                    action = "preserve"
                    role = "sfx"
                else:
                    bucket = ambiguous
                    priority = "ambiguous"
                    action = "review"
                    role = "ambiguous"
                mask[(required > 0) | (preserve > 0) | (ambiguous > 0)] = 0
                if not np.any(mask):
                    continue
                if region_id == "region-unowned-review":
                    unowned_ownership[mask > 0] = 255
                bucket[mask > 0] = 255
                instance_id = f"independent-{row['review_id']}"
                mask_path = _write_mask(page_dir / "instances" / f"{instance_id}.png", mask)
                instances.append(
                    {
                        "instance_id": instance_id,
                        "region_id": region_id,
                        "mask_path": mask_path,
                        "semantic_role": role,
                        "processing_action": action,
                        "priority": priority,
                        "source_reviewed": True,
                    }
                )
            if np.any(unowned_ownership):
                unowned_region = _empty_region(
                    page_dir,
                    shape,
                    "region-unowned-review",
                    ownership=unowned_ownership,
                )
                page["regions"] = [*page.get("regions", []), unowned_region]
            if not region_map:
                empty_region = _empty_region(page_dir, shape, "region-page-empty")
                page["regions"] = [empty_region]
                region_map = {"region-page-empty": empty_region}
            page["target_instances"] = instances
            page["target_text_mask"] = (
                _write_mask(page_dir / "target-text.png", required)
                if np.any(required)
                else None
            )
            page["preserve_mask"] = _write_mask(page_dir / "preserve.png", preserve)
            page["target_mask_provenance"] = (
                "source_only_full_page_inventory_review"
                if inventory_status == "complete_with_reviewed_rows"
                else "paired_location_source_only_extent_review"
            )

        required_mask = _read_mask(page.get("target_text_mask"), shape)
        preserve_mask = _read_mask(page.get("preserve_mask"), shape)
        semantic_ambiguous = ambiguous
        semantic_union = cv2.bitwise_or(
            cv2.bitwise_or(required_mask, preserve_mask), semantic_ambiguous
        )
        protected = _read_mask(source_page.get("protected_structure_mask"), shape)
        protected[semantic_union > 0] = 0
        source_ambiguous = _read_mask(
            source_page.get("ambiguous_structure_mask"), shape
        )
        normalized_ambiguous = cv2.bitwise_or(source_ambiguous, semantic_ambiguous)
        normalized_ambiguous[(required_mask > 0) | (preserve_mask > 0)] = 0
        ownership = _read_mask(source_page.get("ownership_mask"), shape)
        ownership[unowned_ownership > 0] = 255
        page["protected_structure_mask"] = _write_mask(
            page_dir / "protected-structure.png", protected
        )
        page["ambiguous_structure_mask"] = _write_mask(
            page_dir / "ambiguous-structure.png", normalized_ambiguous
        )
        page["ownership_mask"] = _write_mask(
            page_dir / "ownership.png", ownership
        )
        normalized_regions: list[dict[str, Any]] = []
        for raw_region in page.get("regions", []):
            region = dict(raw_region)
            region_id = str(region.get("region_id") or "")
            region_dir = page_dir / "regions" / region_id
            region_protected = _read_mask(region.get("protected_structure_mask"), shape)
            region_protected[semantic_union > 0] = 0
            region_ambiguous = _read_mask(region.get("ambiguous_structure_mask"), shape)
            region_ambiguous[semantic_union > 0] = 0
            region["protected_structure_mask"] = _write_mask(
                region_dir / "protected.png", region_protected
            )
            region["ambiguous_structure_mask"] = _write_mask(
                region_dir / "ambiguous.png", region_ambiguous
            )
            normalized_regions.append(region)
        page["regions"] = normalized_regions

        for instance in page.get("target_instances", []):
            priority = str(instance.get("priority") or "")
            key = "preserve" if priority == "optional" else priority
            if key in total_instances:
                total_instances[key] += 1
        page["expected_edit"] = (
            "required"
            if any(
                str(value.get("priority") or "") == "required"
                for value in page.get("target_instances", [])
            )
            else "none"
        )
        page.update(
            annotation_basis="source_only_v4",
            target_extent_independent=True,
            target_inventory_independent=True,
            target_review_complete=True,
            annotation_frozen_before_candidate=True,
            reviewed_source_only=True,
            candidate_seen=False,
        )
        output_pages.append(page)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus_id": semantic.get("corpus_id"),
        "split_role": "development",
        "annotation_frozen_before_candidate": True,
        "candidate_seen": False,
        "target_extent_independent": True,
        "target_inventory_independent": True,
        "target_review_complete": True,
        "review_ledger": str(ledger_path.resolve()),
        "review_decisions": str(decisions_path.resolve()),
        "instance_counts": total_instances,
        "pages": output_pages,
    }
    output_path = output_dir / "source-manifest-v4-independent.json"
    for page in output_pages:
        page.setdefault("existing_source_edit_mask", None)
        page["artifact_sha256"] = manifest_page_artifact_sha256(output_path, page)
        page["source_sha256"] = page["artifact_sha256"]["path"]
    page_ids = sorted(str(page["page_id"]) for page in output_pages)
    payload["page_count"] = len(page_ids)
    payload["page_ids"] = page_ids
    payload["page_inventory_sha256"] = source_manifest_page_inventory_sha256(
        output_pages
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def seal_independent_manifest(
    manifest_path: Path,
    review_decisions_path: Path,
) -> dict[str, Any]:
    payload = _read_json(manifest_path)
    decisions = _read_json(review_decisions_path)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported independent manifest seal input")
    if decisions.get("schema_version") != DECISIONS_SCHEMA_VERSION:
        raise ValueError("unsupported independent review decisions seal input")
    if not all(
        (
            payload.get("annotation_frozen_before_candidate") is True,
            payload.get("candidate_seen") is False,
            payload.get("target_extent_independent") is True,
            payload.get("target_inventory_independent") is True,
            payload.get("target_review_complete") is True,
        )
    ):
        raise ValueError("independent manifest is not source-only review complete")
    if (
        decisions.get("candidate_seen") is not False
        or decisions.get("review_complete") is not True
    ):
        raise ValueError("independent review decisions are not complete")
    seal = {
        "schema_version": SEAL_SCHEMA_VERSION,
        "manifest_sha256": _sha256(manifest_path),
        "review_decisions_sha256": _sha256(review_decisions_path),
        "corpus_id": payload.get("corpus_id"),
        "instance_counts": payload.get("instance_counts"),
        "annotation_frozen_before_candidate": True,
        "candidate_seen": False,
        "candidate_generated": False,
        "source_only_review_complete": True,
        "target_extent_independent": True,
        "target_inventory_independent": True,
    }
    seal_path = manifest_path.with_suffix(manifest_path.suffix + ".seal.json")
    seal_path.write_text(
        json.dumps(seal, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return seal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Apply a complete independent target review.")
    parser.add_argument("--semantic-manifest", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--review-decisions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output_dir = args.output_dir.resolve()
    try:
        output_dir.relative_to(default_archive_root().resolve())
    except ValueError as exc:
        raise ValueError("independent manifest output must stay in the private archive") from exc
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = apply_independent_target_review(
        args.semantic_manifest.resolve(),
        args.review_ledger.resolve(),
        args.review_decisions.resolve(),
        output_dir,
    )
    manifest_path = output_dir / "source-manifest-v4-independent.json"
    seal_independent_manifest(manifest_path, args.review_decisions.resolve())
    validate_source_only_manifest_v4(manifest_path)
    print(json.dumps({"pages": len(payload["pages"]), "counts": payload["instance_counts"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
