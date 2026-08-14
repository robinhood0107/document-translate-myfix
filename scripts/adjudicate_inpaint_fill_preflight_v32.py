#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    RELATIVE_PRODUCT_SAFETY_ZERO_FIELDS,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-fill-preflight-v32"
CATEGORY = "40-inpaint-mask-render"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _one_run(payload: Mapping[str, object], run_id: str) -> Mapping[str, object]:
    rows = payload.get("runs")
    if not isinstance(rows, list):
        raise ValueError("factorized result lacks runs")
    matches = [
        row
        for row in rows
        if isinstance(row, Mapping) and str(row.get("run_id") or "") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"factorized run must resolve exactly once: {run_id}")
    return matches[0]


def adjudicate_fill_preflight(
    *,
    factorized_path: Path,
    run_id: str,
    source_manifest_path: Path,
    product_summary_path: Path,
) -> dict[str, object]:
    factorized = _read_json(factorized_path)
    source_manifest = _read_json(source_manifest_path)
    summary = _read_json(product_summary_path)
    if factorized.get("schema_version") != "inpaint-factorized-results-v3":
        raise ValueError("fill preflight requires factorized results v3")
    if source_manifest.get("schema_version") != "inpaint-factorized-source-manifest-v4":
        raise ValueError("fill preflight requires source manifest v4")
    if str(factorized.get("manifest_sha256") or "") != _sha256(
        source_manifest_path
    ):
        raise ValueError("factorized result is not bound to the supplied manifest")
    manifest_corpora = summary.get("manifest_corpora")
    if not isinstance(manifest_corpora, Mapping) or len(manifest_corpora) != 1:
        raise ValueError("product summary requires one canonical manifest corpus")
    product_scope = next(iter(manifest_corpora.values()))
    if not isinstance(product_scope, Mapping):
        raise ValueError("product summary manifest corpus is invalid")
    source_annotation_sha = str(
        source_manifest.get("source_annotation_manifest_sha256") or ""
    )
    if (
        not source_annotation_sha
        or str(product_scope.get("parent_manifest_sha256") or "")
        != source_annotation_sha
    ):
        raise ValueError("product and factorized evidence use different annotations")
    run = _one_run(factorized, run_id)
    metrics = run.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("factorized run lacks metrics")
    if int(metrics.get("page_count", -1)) != int(source_manifest.get("page_count", -2)):
        raise ValueError("factorized and manifest page counts differ")

    failures: list[str] = []
    safety: dict[str, int | float | None] = {}
    for field in RELATIVE_PRODUCT_SAFETY_ZERO_FIELDS:
        value = metrics.get(field)
        safety[field] = value if isinstance(value, (int, float)) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value != 0:
            failures.append(f"safety_nonzero:{field}")
    runtime = summary.get("inpainter_runtime")
    if not isinstance(runtime, Mapping):
        failures.append("product_runtime_missing")
    else:
        if str(runtime.get("actual_device") or "").lower() != "cuda":
            failures.append("product_runtime_not_cuda")
        if str(runtime.get("actual_precision") or "").lower() != "bf16":
            failures.append("product_runtime_not_bf16")
        if runtime.get("cpu_fallback_used") is not False:
            failures.append("product_runtime_cpu_fallback")
    summary_checks = {
        "success_count": int(summary.get("success_count", -1)),
        "failure_count": int(summary.get("failure_count", -1)),
        "cpu_fallback_count": int(summary.get("cpu_fallback_count", -1)),
        "required_skipped_block_count": int(
            summary.get("required_skipped_block_count", -1)
        ),
        "protected_structure_changed_pixel_count_exact": int(
            summary.get("protected_structure_changed_pixel_count_exact", -1)
        ),
        "ambiguous_structure_changed_pixel_count_exact": int(
            summary.get("ambiguous_structure_changed_pixel_count_exact", -1)
        ),
        "changed_outside_final_mask_pixel_count_exact": int(
            summary.get("changed_outside_final_mask_pixel_count_exact", -1)
        ),
        "unexpected_none_edit_count": int(summary.get("unexpected_none_edit_count", -1)),
    }
    expected_pages = int(source_manifest["page_count"])
    if summary_checks["success_count"] != expected_pages:
        failures.append("product_page_count_incomplete")
    if summary_checks["failure_count"] != 0:
        failures.append("product_page_failure")
    for field in (
        "cpu_fallback_count",
        "required_skipped_block_count",
        "protected_structure_changed_pixel_count_exact",
        "ambiguous_structure_changed_pixel_count_exact",
        "changed_outside_final_mask_pixel_count_exact",
        "unexpected_none_edit_count",
    ):
        if summary_checks[field] != 0:
            failures.append(f"product_summary_nonzero:{field}")
    admitted = not failures
    return {
        "schema_version": "inpaint-fill-preflight-adjudication-v32",
        "manifest_sha256": str(factorized["manifest_sha256"]),
        "source_annotation_manifest_sha256": source_annotation_sha,
        "factorized_artifact_sha256": _sha256(factorized_path),
        "factorized_run_id": run_id,
        "product_summary_sha256": _sha256(product_summary_path),
        "fill_candidate": "conditional_refill_existing",
        "edit_mask_identity_required": True,
        "fill_candidate_admitted": admitted,
        "cuda_stage2_authorized": admitted,
        "gate_failures": sorted(set(failures)),
        "factorized_safety_metrics": safety,
        "product_summary_checks": summary_checks,
        "product_runtime": {
            "actual_device": str(runtime.get("actual_device") or "")
            if isinstance(runtime, Mapping)
            else "",
            "actual_precision": str(runtime.get("actual_precision") or "")
            if isinstance(runtime, Mapping)
            else "",
            "cpu_fallback_used": runtime.get("cpu_fallback_used")
            if isinstance(runtime, Mapping)
            else None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed before spending CUDA on the v3.2 fill-only candidate."
    )
    parser.add_argument("--factorized-result", type=Path, required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--product-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = adjudicate_fill_preflight(
        factorized_path=args.factorized_result.resolve(),
        run_id=str(args.run),
        source_manifest_path=args.source_manifest.resolve(),
        product_summary_path=args.product_summary.resolve(),
    )
    output = output_root / "fill-preflight-adjudication.json"
    temporary = output.with_name(f".{output.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output)
    if managed is not None:
        managed.complete(
            metadata={
                "manifest_sha256": payload["manifest_sha256"],
                "fill_candidate_admitted": payload["fill_candidate_admitted"],
            }
        )
        mismatches = managed.verify()
        if mismatches:
            raise RuntimeError(
                "managed artifact verification failed: " + "; ".join(mismatches)
            )
        print(managed.run_root)
    else:
        print(output)
    return 0 if payload["fill_candidate_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
