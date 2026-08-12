#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "inpaint-source-review-decisions-v4"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _expanded_ids(values: list[str]) -> set[str]:
    result: set[str] = set()
    for raw in values:
        for token in raw.split(","):
            value = token.strip()
            if not value:
                continue
            if "-" in value and not value.startswith("review-"):
                start, end = value.split("-", 1)
                for number in range(int(start), int(end) + 1):
                    result.add(f"review-{number:04d}")
            elif value.isdigit():
                result.add(f"review-{int(value):04d}")
            else:
                result.add(value)
    return result


def record_source_review(
    ledger_path: Path,
    output_path: Path,
    *,
    preserve_ids: set[str],
    ambiguous_ids: set[str] = frozenset(),
    ui_ids: set[str] = frozenset(),
    narration_ids: set[str] = frozenset(),
) -> dict[str, Any]:
    ledger = _read_json(ledger_path)
    if ledger.get("candidate_seen") is not False:
        raise ValueError("review ledger must be candidate blind")
    rows = ledger.get("rows")
    if not isinstance(rows, list):
        raise ValueError("review ledger rows must be an array")
    known = {str(row.get("review_id") or "") for row in rows}
    selected = preserve_ids | ambiguous_ids | ui_ids | narration_ids
    unknown = sorted(selected.difference(known))
    if unknown:
        raise ValueError(f"unknown review ids: {unknown}")
    if preserve_ids & ambiguous_ids:
        raise ValueError("preserve and ambiguous review ids overlap")
    if (ui_ids | narration_ids) & (preserve_ids | ambiguous_ids):
        raise ValueError("required role overrides overlap preserve/ambiguous ids")
    decisions: list[dict[str, str]] = []
    for row in rows:
        review_id = str(row.get("review_id") or "")
        proposal_role = str(row.get("semantic_role_proposal") or "")
        if review_id in preserve_ids:
            decision, role = "preserve", "sfx"
        elif review_id in ambiguous_ids:
            decision, role = "ambiguous", "ambiguous"
        elif review_id in ui_ids:
            decision, role = "required", "ui_or_sign"
        elif review_id in narration_ids:
            decision, role = "required", "narration"
        else:
            decision = "required"
            role = (
                "dialogue_bubble"
                if proposal_role == "dialogue_bubble"
                else "dialogue_free"
            )
        decisions.append(
            {
                "review_id": review_id,
                "decision": decision,
                "semantic_role": role,
            }
        )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "candidate_seen": False,
        "review_complete": True,
        "review_ledger": str(ledger_path.resolve()),
        "decisions": decisions,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record candidate-blind source semantic review decisions."
    )
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preserve", action="append", default=[])
    parser.add_argument("--ambiguous", action="append", default=[])
    parser.add_argument("--ui", action="append", default=[])
    parser.add_argument("--narration", action="append", default=[])
    args = parser.parse_args(argv)
    payload = record_source_review(
        args.review_ledger.resolve(),
        args.output.resolve(),
        preserve_ids=_expanded_ids(args.preserve),
        ambiguous_ids=_expanded_ids(args.ambiguous),
        ui_ids=_expanded_ids(args.ui),
        narration_ids=_expanded_ids(args.narration),
    )
    counts: dict[str, int] = {}
    for value in payload["decisions"]:
        decision = value["decision"]
        counts[decision] = counts.get(decision, 0) + 1
    print(json.dumps({"rows": len(payload["decisions"]), "counts": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
