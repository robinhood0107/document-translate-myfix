#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.contracts import (  # noqa: E402
    FactorizedRunRecord,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    select_pareto_records,
)


SCHEMA_VERSION = "inpaint-factorized-results-v3"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _record(payload: dict[str, Any]) -> FactorizedRunRecord:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("factorized run must contain metrics")
    return FactorizedRunRecord(
        run_id=str(payload["run_id"]),
        detector_id=str(payload["detector_id"]),
        ownership_id=str(payload["ownership_id"]),
        silhouette_id=str(payload["silhouette_id"]),
        router_id=str(payload["router_id"]),
        expansion_id=str(payload["expansion_id"]),
        fill_id=str(payload["fill_id"]),
        oracle_only=bool(payload.get("oracle_only", False)),
        status="active",
        metrics=metrics,
        closure_reason=str(payload.get("closure_reason") or ""),
    )


def merge_results(result_paths: list[Path]) -> dict[str, Any]:
    if not result_paths:
        raise ValueError("at least one factorized result is required")
    manifest_sha: str | None = None
    matrix_sha: str | None = None
    records: list[FactorizedRunRecord] = []
    pages: dict[str, list[dict[str, object]]] = {}
    inference_count = 0
    source_paths: list[str] = []
    for path in result_paths:
        payload = _read_json(path)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"unsupported result schema: {path}")
        current_manifest = str(payload.get("manifest_sha256") or "")
        current_matrix = str(payload.get("matrix_sha256") or "")
        if not current_manifest or not current_matrix:
            raise ValueError(f"result lacks manifest or matrix SHA: {path}")
        if manifest_sha is None:
            manifest_sha = current_manifest
            matrix_sha = current_matrix
        elif current_manifest != manifest_sha or current_matrix != matrix_sha:
            raise ValueError("factorized results have different manifest or matrix SHAs")
        raw_runs = payload.get("runs")
        raw_pages = payload.get("pages")
        if not isinstance(raw_runs, list) or not isinstance(raw_pages, dict):
            raise ValueError(f"result has invalid runs/pages: {path}")
        for raw in raw_runs:
            if not isinstance(raw, dict):
                raise ValueError(f"result run must be an object: {path}")
            record = _record(raw)
            if record.run_id in pages:
                raise ValueError(f"duplicate factorized run id: {record.run_id}")
            rows = raw_pages.get(record.run_id)
            if not isinstance(rows, list):
                raise ValueError(f"result lacks page rows for {record.run_id}")
            records.append(record)
            pages[record.run_id] = rows
        inference_count += int(payload.get("positive_lama_inference_count", 0) or 0)
        source_paths.append(str(path.resolve()))
    ranked = select_pareto_records(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": manifest_sha,
        "matrix_sha256": matrix_sha,
        "combination_count": len(ranked),
        "positive_lama_inference_count": inference_count,
        "merged_isolated_results": True,
        "source_result_paths": source_paths,
        "runs": [record.as_record() for record in ranked],
        "pages": pages,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge bounded v3 factorized runs and recompute Pareto status."
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result_paths = sorted(args.input_root.resolve().glob("*/factorized-results.json"))
    payload = merge_results(result_paths)
    _write_json(args.output.resolve(), payload)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
