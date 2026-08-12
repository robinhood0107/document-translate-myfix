#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inpaint-independent-target-review-decisions-v4"
OVERRIDES_SCHEMA_VERSION = "inpaint-independent-review-overrides-v4"
KNOWN_INVENTORY_SOURCES = frozenset(
    {
        "paired_location_with_human_semantic_region",
    }
)
SOURCE_PRIORITY_INVENTORY = "source_manifest_location_aid_only"
EXTENTS = frozenset(
    {
        "strict",
        "balanced",
        "edge_supported",
        "location_dilate1",
        "location_dilate2",
        "location",
        "manual",
        "reject",
    }
)
SEMANTICS = frozenset({"required", "preserve", "ambiguous", "not_text"})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _sheet_number(review_id: str, rows_per_sheet: int) -> int:
    try:
        value = int(review_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"invalid review id: {review_id}") from exc
    return value // rows_per_sheet + 1


def extend_source_location_review_overrides(
    ledger_path: Path,
    base_overrides_path: Path,
    output_path: Path,
    *,
    reviewed_sheet_through: int,
    source_location_default_extent: str,
    reject_review_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    base = _read_json(base_overrides_path)
    if ledger.get("candidate_seen") is not False or base.get("candidate_seen") is not False:
        raise ValueError("independent target review must be candidate blind")
    if base.get("schema_version") != OVERRIDES_SCHEMA_VERSION:
        raise ValueError("unsupported independent review overrides")
    sheets = ledger.get("review_sheets")
    rows = ledger.get("rows")
    if not isinstance(sheets, list) or not isinstance(rows, list):
        raise ValueError("review ledger rows and sheets must be arrays")
    if reviewed_sheet_through != len(sheets):
        raise ValueError(
            "source-only review must end at the final sheet: "
            f"through={reviewed_sheet_through} total={len(sheets)}"
        )
    if source_location_default_extent not in EXTENTS - {"manual", "reject"}:
        raise ValueError("invalid source location default extent")
    known_rows = {str(row.get("review_id") or ""): row for row in rows}
    if not all(known_rows):
        raise ValueError("review ledger contains invalid review ids")
    unknown_rejects = sorted(set(reject_review_ids).difference(known_rows))
    if unknown_rejects:
        raise ValueError(f"reject overrides reference unknown reviews: {unknown_rejects}")

    effective = copy.deepcopy(base)
    effective["reviewed_sheets"] = list(range(1, reviewed_sheet_through + 1))
    effective["source_location_default_extent"] = source_location_default_extent
    row_overrides = effective.setdefault("row_overrides", {})
    if not isinstance(row_overrides, dict):
        raise ValueError("row_overrides must be an object")
    for review_id in reject_review_ids:
        if (
            str(known_rows[review_id].get("inventory_source") or "")
            != SOURCE_PRIORITY_INVENTORY
        ):
            raise ValueError(f"only source-location rows can use this reject helper: {review_id}")
        row_overrides[review_id] = {"extent": "reject", "semantic": "not_text"}
    effective["full_page_inventory"] = [
        {
            "page_id": str(row.get("page_id") or ""),
            "status": "complete_with_reviewed_rows",
        }
        for row in ledger.get("full_page_inventory_pending", [])
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return effective


def build_review_decisions(
    ledger_path: Path,
    overrides_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    overrides = _read_json(overrides_path)
    if ledger.get("candidate_seen") is not False or overrides.get("candidate_seen") is not False:
        raise ValueError("independent target review must be candidate blind")
    if overrides.get("schema_version") != OVERRIDES_SCHEMA_VERSION:
        raise ValueError("unsupported independent review overrides")
    rows = ledger.get("rows")
    sheets = ledger.get("review_sheets")
    if not isinstance(rows, list) or not isinstance(sheets, list):
        raise ValueError("review ledger rows and sheets must be arrays")
    reviewed_sheets = {int(value) for value in overrides.get("reviewed_sheets", [])}
    expected_sheets = set(range(1, len(sheets) + 1))
    if reviewed_sheets != expected_sheets:
        missing = sorted(expected_sheets.difference(reviewed_sheets))
        extra = sorted(reviewed_sheets.difference(expected_sheets))
        raise ValueError(f"source-only review sheets incomplete: missing={missing} extra={extra}")
    rows_per_sheet = int(overrides.get("rows_per_sheet") or 8)
    raw_overrides = overrides.get("row_overrides")
    if not isinstance(raw_overrides, dict):
        raise ValueError("row_overrides must be an object")
    known_ids = {str(value.get("review_id") or "") for value in rows}
    if not all(known_ids) or len(known_ids) != len(rows):
        raise ValueError("review ledger contains invalid review ids")
    extras = sorted(set(raw_overrides).difference(known_ids))
    if extras:
        raise ValueError(f"row overrides reference unknown reviews: {extras}")

    decisions: list[dict[str, str]] = []
    for raw in rows:
        review_id = str(raw["review_id"])
        sheet = _sheet_number(review_id, rows_per_sheet)
        if sheet not in reviewed_sheets:
            raise ValueError(f"review row is on an unreviewed sheet: {review_id}")
        override = raw_overrides.get(review_id)
        if override is None:
            inventory_source = str(raw.get("inventory_source") or "")
            if inventory_source == SOURCE_PRIORITY_INVENTORY:
                priority = str(raw.get("source_priority_proposal") or "")
                if priority not in {"required", "optional"}:
                    raise ValueError(
                        f"source location row needs an explicit decision: {review_id}"
                    )
                extent = str(overrides.get("source_location_default_extent") or "balanced")
                semantic = "required" if priority == "required" else "preserve"
            elif inventory_source not in KNOWN_INVENTORY_SOURCES:
                raise ValueError(f"unowned review row needs an explicit decision: {review_id}")
            else:
                extent = str(overrides.get("known_default_extent") or "balanced")
                semantic = "required"
        else:
            if not isinstance(override, dict):
                raise ValueError(f"row override must be an object: {review_id}")
            extent = str(override.get("extent") or "")
            semantic = str(override.get("semantic") or "")
        if extent not in EXTENTS or semantic not in SEMANTICS:
            raise ValueError(f"invalid review decision: {review_id}={extent}/{semantic}")
        manual_path = (
            str(override.get("manual_extent_path") or "")
            if isinstance(override, dict)
            else ""
        )
        if extent == "manual" and not manual_path:
            raise ValueError(f"manual review extent missing: {review_id}")
        if extent != "manual" and manual_path:
            raise ValueError(f"non-manual extent must not provide a mask: {review_id}")
        decision = {
            "review_id": review_id,
            "extent": extent,
            "semantic": semantic,
        }
        if manual_path:
            decision["manual_extent_path"] = manual_path
        decisions.append(decision)

    inventory = overrides.get("full_page_inventory")
    if not isinstance(inventory, list):
        raise ValueError("full_page_inventory must be an array")
    expected_pages = {
        str(value.get("page_id") or "")
        for value in ledger.get("full_page_inventory_pending", [])
    }
    actual_pages = {str(value.get("page_id") or "") for value in inventory}
    if actual_pages != expected_pages:
        missing = sorted(expected_pages.difference(actual_pages))
        extra = sorted(actual_pages.difference(expected_pages))
        raise ValueError(f"full-page inventory incomplete: missing={missing} extra={extra}")
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_seen": False,
        "decisions": decisions,
        "full_page_inventory": inventory,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build complete independent decisions after all source-only sheets were reviewed."
    )
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--effective-overrides-output", type=Path)
    parser.add_argument("--reviewed-sheet-through", type=int)
    parser.add_argument("--source-location-default-extent")
    parser.add_argument("--reject-review-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    overrides_path = args.overrides.resolve()
    extension_requested = any(
        (
            args.effective_overrides_output,
            args.reviewed_sheet_through,
            args.source_location_default_extent,
            args.reject_review_id,
        )
    )
    if extension_requested:
        if not all(
            (
                args.effective_overrides_output,
                args.reviewed_sheet_through,
                args.source_location_default_extent,
            )
        ):
            parser.error("override extension requires output, final sheet, and default extent")
        extend_source_location_review_overrides(
            args.review_ledger.resolve(),
            overrides_path,
            args.effective_overrides_output.resolve(),
            reviewed_sheet_through=args.reviewed_sheet_through,
            source_location_default_extent=args.source_location_default_extent,
            reject_review_ids=tuple(args.reject_review_id),
        )
        overrides_path = args.effective_overrides_output.resolve()
    payload = build_review_decisions(
        args.review_ledger.resolve(), overrides_path, args.output.resolve()
    )
    print(json.dumps({"decisions": len(payload["decisions"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
