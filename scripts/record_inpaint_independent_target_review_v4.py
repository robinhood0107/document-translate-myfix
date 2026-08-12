#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inpaint-independent-target-review-decisions-v4"
EXTENT_CHOICES = frozenset(
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
SEMANTIC_CHOICES = frozenset({"required", "preserve", "ambiguous", "not_text"})


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def record_independent_target_review(
    ledger_path: Path,
    decisions_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    if ledger.get("candidate_seen") is not False:
        raise ValueError("independent target review must be candidate blind")
    if ledger.get("review_complete") is not False:
        raise ValueError("review ledger must still be pending")
    raw_decisions = _read_json(decisions_path)
    if raw_decisions.get("candidate_seen") is not False:
        raise ValueError("review decisions must be candidate blind")
    decisions = raw_decisions.get("decisions")
    inventory = raw_decisions.get("full_page_inventory")
    if not isinstance(decisions, list) or not isinstance(inventory, list):
        raise ValueError("review decisions and full-page inventory must be arrays")
    known_rows = {str(row.get("review_id") or ""): row for row in ledger.get("rows", [])}
    decided_rows = {str(row.get("review_id") or ""): row for row in decisions}
    if not all(known_rows) or len(known_rows) != len(ledger.get("rows", [])):
        raise ValueError("review ledger contains invalid review ids")
    if set(decided_rows) != set(known_rows):
        missing = sorted(set(known_rows).difference(decided_rows))
        extra = sorted(set(decided_rows).difference(known_rows))
        raise ValueError(f"review decisions differ from ledger: missing={missing} extra={extra}")
    pending_pages = {
        str(row.get("page_id") or "")
        for row in ledger.get("full_page_inventory_pending", [])
    }
    decided_pages = {str(row.get("page_id") or "") for row in inventory}
    if decided_pages != pending_pages:
        missing = sorted(pending_pages.difference(decided_pages))
        extra = sorted(decided_pages.difference(pending_pages))
        raise ValueError(f"full-page inventory differs: missing={missing} extra={extra}")

    normalized: list[dict[str, str]] = []
    for review_id in sorted(decided_rows):
        row = decided_rows[review_id]
        extent = str(row.get("extent") or "").strip().lower()
        semantic = str(row.get("semantic") or "").strip().lower()
        if extent not in EXTENT_CHOICES:
            raise ValueError(f"invalid extent choice: {review_id}={extent}")
        if semantic not in SEMANTIC_CHOICES:
            raise ValueError(f"invalid semantic choice: {review_id}={semantic}")
        if semantic in {"not_text", "ambiguous"} and extent != "reject":
            raise ValueError(f"non-edit semantic must reject extent: {review_id}")
        if semantic in {"required", "preserve"} and extent == "reject":
            raise ValueError(f"text semantic requires an extent: {review_id}")
        manual_path = str(row.get("manual_extent_path") or "").strip()
        if extent == "manual":
            if not manual_path or not Path(manual_path).is_file():
                raise ValueError(f"manual review extent missing: {review_id}")
        elif manual_path:
            raise ValueError(f"non-manual extent must not provide a mask: {review_id}")
        normalized_row = {
            "review_id": review_id,
            "extent": extent,
            "semantic": semantic,
        }
        if manual_path:
            normalized_row["manual_extent_path"] = manual_path
        normalized.append(normalized_row)
    normalized_inventory: list[dict[str, str]] = []
    for row in sorted(inventory, key=lambda value: str(value.get("page_id") or "")):
        page_id = str(row.get("page_id") or "")
        status = str(row.get("status") or "").strip().lower()
        if status not in {
            "complete_no_required_text",
            "complete_with_manual_inventory",
            "complete_with_reviewed_rows",
        }:
            raise ValueError(f"invalid full-page inventory status: {page_id}={status}")
        manual_path = str(row.get("manual_inventory_path") or "").strip()
        if status == "complete_with_manual_inventory":
            if not manual_path or not Path(manual_path).is_file():
                raise ValueError(f"manual inventory artifact missing: {page_id}")
            manual_payload = _read_json(Path(manual_path))
            if (
                manual_payload.get("schema_version")
                != "inpaint-independent-manual-inventory-v4"
                or manual_payload.get("candidate_seen") is not False
                or manual_payload.get("source_reviewed") is not True
                or str(manual_payload.get("page_id") or "") != page_id
            ):
                raise ValueError(f"invalid source-only manual inventory: {page_id}")
            if not isinstance(manual_payload.get("instances"), list) or not manual_payload["instances"]:
                raise ValueError(f"manual inventory has no instances: {page_id}")
        elif manual_path:
            raise ValueError(f"row-reviewed inventory must not provide a mask: {page_id}")
        normalized_inventory.append(
            {
                "page_id": page_id,
                "status": status,
                "manual_inventory_path": manual_path,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_seen": False,
        "review_complete": True,
        "review_ledger": str(ledger_path.resolve()),
        "decisions": normalized,
        "full_page_inventory": normalized_inventory,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Record a complete independent target review.")
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    payload = record_independent_target_review(
        args.review_ledger.resolve(), args.decisions.resolve(), args.output.resolve()
    )
    print(
        json.dumps(
            {
                "decisions": len(payload["decisions"]),
                "full_page_inventory": len(payload["full_page_inventory"]),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
