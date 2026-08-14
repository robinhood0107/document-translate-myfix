#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    evaluate_relative_product_gate,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-relative-product-v32"
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


def _run(payload: dict[str, Any], run_id: str) -> dict[str, Any]:
    rows = payload.get("runs")
    if not isinstance(rows, list):
        raise ValueError("factorized result lacks runs")
    matches = [
        row
        for row in rows
        if isinstance(row, dict) and str(row.get("run_id") or "") == run_id
    ]
    if len(matches) != 1:
        raise ValueError(f"factorized run must exist exactly once: {run_id}")
    return matches[0]


def adjudicate_relative_product(
    *,
    baseline_path: Path,
    baseline_run_id: str,
    candidate_path: Path,
    candidate_run_id: str,
    candidate_kind: str,
    balanced_preflight_path: Path | None = None,
) -> dict[str, object]:
    baseline = _read_json(baseline_path)
    candidate = _read_json(candidate_path)
    if baseline.get("schema_version") != "inpaint-factorized-results-v3" or (
        candidate.get("schema_version") != "inpaint-factorized-results-v3"
    ):
        raise ValueError("relative adjudication requires factorized results v3")
    if baseline.get("manifest_sha256") != candidate.get("manifest_sha256"):
        raise ValueError("relative candidate and baseline use different manifests")
    baseline_run = _run(baseline, baseline_run_id)
    candidate_run = _run(candidate, candidate_run_id)
    baseline_metrics = baseline_run.get("metrics")
    candidate_metrics = candidate_run.get("metrics")
    baseline_pages = baseline.get("pages", {}).get(baseline_run_id)
    candidate_pages = candidate.get("pages", {}).get(candidate_run_id)
    if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
        raise ValueError("relative runs require metrics")
    if not isinstance(baseline_pages, list) or not isinstance(candidate_pages, list):
        raise ValueError("relative runs require page statistics")
    kind = str(candidate_kind).strip().lower()
    semantic_provider = "none"
    seed_admitted = False
    balanced_preflight_sha256 = ""
    if kind == "balanced":
        if balanced_preflight_path is None:
            raise ValueError("balanced relative candidate requires a sealed preflight")
        preflight = _read_json(balanced_preflight_path)
        if (
            preflight.get("schema_version")
            != "inpaint-balanced-preflight-adjudication-v32"
            or preflight.get("manifest_sha256") != baseline.get("manifest_sha256")
            or preflight.get("balanced_candidate_admitted") is not True
            or preflight.get("seed_admitted") is not True
        ):
            raise ValueError("balanced preflight does not admit this sealed scope")
        semantic_provider = str(preflight.get("semantic_provider") or "")
        if not semantic_provider:
            raise ValueError("balanced preflight lacks semantic provider provenance")
        seed_admitted = True
        balanced_preflight_sha256 = _sha256(balanced_preflight_path)
    elif kind == "fill_only" and balanced_preflight_path is not None:
        raise ValueError("fill-only relative candidate must not bind a balanced preflight")
    gate = evaluate_relative_product_gate(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        baseline_pages=baseline_pages,
        candidate_pages=candidate_pages,
        candidate_kind=kind,
    )
    return {
        "schema_version": "inpaint-relative-product-adjudication-v32",
        "manifest_sha256": str(baseline["manifest_sha256"]),
        "baseline": {
            "artifact_sha256": _sha256(baseline_path),
            "run_id": baseline_run_id,
            "output_mask_set_sha256": str(
                baseline_metrics.get("output_mask_set_sha256") or ""
            ),
        },
        "candidate": {
            "artifact_sha256": _sha256(candidate_path),
            "run_id": candidate_run_id,
            "output_mask_set_sha256": str(
                candidate_metrics.get("output_mask_set_sha256") or ""
            ),
        },
        "baseline_sha256": _sha256(baseline_path),
        "balanced_preflight_sha256": balanced_preflight_sha256,
        "seed_admitted": seed_admitted,
        "provenance": {
            "detector": str(
                (candidate_run.get("selection") or {}).get("detector")
                if isinstance(candidate_run.get("selection"), dict)
                else ""
            ),
            "semantic_provider": semantic_provider,
            "router": str(
                (candidate_run.get("selection") or {}).get("router")
                if isinstance(candidate_run.get("selection"), dict)
                else ""
            ),
            "fill": str(
                (candidate_run.get("selection") or {}).get("fill")
                if isinstance(candidate_run.get("selection"), dict)
                else ""
            ),
        },
        **gate,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Adjudicate one inpaint candidate relative to a sealed PR6 baseline."
    )
    parser.add_argument("--baseline-result", type=Path, required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-result", type=Path, required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument(
        "--candidate-kind", choices=("balanced", "fill_only"), required=True
    )
    parser.add_argument(
        "--balanced-preflight",
        type=Path,
        help="Required hash-bound preflight for a balanced candidate; forbidden for fill-only.",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    payload = adjudicate_relative_product(
        baseline_path=args.baseline_result.resolve(),
        baseline_run_id=str(args.baseline_run),
        candidate_path=args.candidate_result.resolve(),
        candidate_run_id=str(args.candidate_run),
        candidate_kind=str(args.candidate_kind),
        balanced_preflight_path=(
            args.balanced_preflight.resolve()
            if args.balanced_preflight is not None
            else None
        ),
    )
    output_path = output_root / "relative-product-adjudication.json"
    temporary = output_path.with_name(f".{output_path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(output_path)
    if managed is not None:
        managed.complete(
            metadata={
                "manifest_sha256": payload["manifest_sha256"],
                "relative_product_pass": payload["relative_product_pass"],
            }
        )
        mismatches = managed.verify()
        if mismatches:
            raise RuntimeError(
                "managed artifact verification failed: " + "; ".join(mismatches)
            )
        print(managed.run_root)
    else:
        print(output_path)
    return 0 if payload["relative_product_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
