#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.update_inpaint_method_evidence_v4 import update_evidence  # noqa: E402
from benchmarking.inpaint_detector_bakeoff.evidence_ledger import (  # noqa: E402
    scope_manifest_binding_cache,
)


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Atomically bind multiple method-family evidence operations."
    )
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args(argv)
    registry_path = args.registry.resolve()
    evidence_path = args.evidence.resolve()
    plan = _read_json(args.plan.resolve())
    if plan.get("schema_version") != "inpaint-method-evidence-batch-plan-v1":
        raise ValueError("unsupported method evidence batch plan schema")
    operations = plan.get("operations")
    if not isinstance(operations, list) or not operations or any(
        not isinstance(row, dict) for row in operations
    ):
        raise ValueError("method evidence batch plan requires operations")
    partial = evidence_path.with_name(f".{evidence_path.name}.batch-partial")
    if partial.exists():
        raise FileExistsError("method evidence batch partial already exists")
    partial.parent.mkdir(parents=True, exist_ok=True)
    if evidence_path.exists():
        shutil.copyfile(evidence_path, partial)
    else:
        partial.write_text(
            json.dumps(
                {
                    "schema_version": "inpaint-method-family-evidence-v4",
                    "scope_manifests": {},
                    "evidence": [],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    try:
        with scope_manifest_binding_cache():
            for index, row in enumerate(operations, start=1):
                variants = row.get("variants")
                if not isinstance(variants, list) or not variants or any(
                    not isinstance(value, str) or not value for value in variants
                ):
                    raise ValueError(f"batch operation {index} lacks variants")
                payload = update_evidence(
                    registry_path=registry_path,
                    evidence_path=partial,
                    artifact_path=Path(str(row["artifact"])).resolve(),
                    scope_manifest_path=Path(str(row["scope_manifest"])).resolve(),
                    family_id=str(row["family"]),
                    variant_ids=frozenset(variants),
                    evaluation_scope=str(row["scope"]),
                    upstream_contract_path=(
                        Path(str(row["upstream_contract"])).resolve()
                        if row.get("upstream_contract")
                        else None
                    ),
                    invalid_reason=(
                        str(row["invalid_reason"])
                        if row.get("invalid_reason")
                        else None
                    ),
                    invalid_parent_record_kind=(
                        str(row["invalid_parent_kind"])
                        if row.get("invalid_parent_kind")
                        else None
                    ),
                    invalid_parent_record_id=(
                        str(row["invalid_parent_id"])
                        if row.get("invalid_parent_id")
                        else None
                    ),
                    allow_replace=bool(row.get("replace", False)),
                )
                partial.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
                    encoding="utf-8",
                )
                print(f"evidence-batch {index}/{len(operations)} {row['family']}")
        partial.replace(evidence_path)
    except BaseException:
        raise
    print(evidence_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
