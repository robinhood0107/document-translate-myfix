#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    residue_score,
    restrict_candidate_to_final_mask,
    score_stage2_page,
)
from scripts.benchmark_inpaint_positive_mask_stage2 import (  # noqa: E402
    _read_bgr,
    _read_mask,
    _review_panel,
    _write_image,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-final-protection-composite-v3"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage1_manifest(
    stage1_result: dict[str, object],
    manifest_sha256: str,
) -> None:
    if stage1_result.get("schema_version") != "inpaint-source-protection-reapply-v3":
        raise ValueError("unsupported source-protection stage1 result")
    if stage1_result.get("manifest_sha256") != manifest_sha256:
        raise ValueError("stage1 result manifest SHA mismatch")


def _page_gate_failures(
    page_id: str,
    metrics: dict[str, object],
    *,
    residue: float | None,
    baseline_residue: float | None,
) -> list[str]:
    failures: list[str] = []
    if int(metrics["protected_changed_pixel_count"]):
        failures.append(f"{page_id}:protected_structure_changed")
    if int(metrics["ambiguous_changed_pixel_count"]):
        failures.append(f"{page_id}:ambiguous_structure_changed")
    if int(metrics["changed_outside_detector_mask_pixel_count"]):
        failures.append(f"{page_id}:changed_outside_final_mask")
    if (
        residue is not None
        and baseline_residue is not None
        and float(residue) > float(baseline_residue) + 1e-12
    ):
        failures.append(f"{page_id}:residue_worse_than_product_baseline")
    coverage = metrics["target_detector_coverage"]
    if coverage is not None and float(coverage) < 0.98:
        failures.append(f"{page_id}:target_coverage_below_98pct")
    minimum = metrics["minimum_target_component_coverage"]
    if minimum is not None and float(minimum) < 0.98:
        failures.append(f"{page_id}:target_component_below_98pct")
    return failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Restore immutable source pixels removed by a source-only final "
            "protection mask, without running a new fill backend."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--product-run", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-quality-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.resolve()
    product_run = args.product_run.resolve()
    stage1_run = args.stage1_run.resolve()
    output, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output.mkdir(parents=True, exist_ok=True)
    try:
        manifest_sha256 = _sha256(manifest)
        stage1_result_path = stage1_run / "stage1-results.json"
        stage1_result = json.loads(stage1_result_path.read_text(encoding="utf-8"))
        _validate_stage1_manifest(stage1_result, manifest_sha256)
        candidate_id = str(stage1_result["candidate_id"])
        rows: list[dict[str, object]] = []
        failures: list[str] = []
        aggregate_sum = 0.0
        aggregate_count = 0
        baseline_sum = 0.0
        baseline_count = 0
        for page in load_stage1_manifest(manifest):
            source = _read_bgr(page.source_image)
            shape = source.shape[:2]
            masks = load_page_masks(page, shape)
            product_image = _read_bgr(
                product_run
                / "corpus-a1"
                / "cleaned_images"
                / f"{page.page_id}_cleaned.png"
            )
            product_mask = _read_mask(
                product_run
                / "corpus-a1"
                / "final_masks"
                / f"{page.page_id}_final_mask.png",
                shape,
            )
            restricted_mask = _read_mask(
                stage1_run
                / "replacement_edit_masks"
                / f"{page.page_id}_replacement_edit.png",
                shape,
            )
            candidate = restrict_candidate_to_final_mask(
                source,
                product_image,
                product_mask,
                restricted_mask,
            )
            metrics, changed = score_stage2_page(
                source,
                candidate,
                restricted_mask,
                masks,
                baseline=product_image,
            )
            residue, residue_sum, residue_pixels = residue_score(
                source,
                candidate,
                masks.target,
            )
            baseline_residue, baseline_residue_sum, baseline_residue_pixels = (
                residue_score(source, product_image, masks.target)
            )
            restored = (product_mask > 0) & (restricted_mask <= 0)
            metrics.update(
                {
                    "page_id": page.page_id,
                    "restored_source_pixel_count": int(np.count_nonzero(restored)),
                    "residue_score": residue,
                    "baseline_residue_score": baseline_residue,
                    "residue_score_delta_from_baseline": (
                        float(residue) - float(baseline_residue)
                        if residue is not None and baseline_residue is not None
                        else None
                    ),
                    "inference_call_count": 0,
                }
            )
            rows.append(metrics)
            failures.extend(
                _page_gate_failures(
                    page.page_id,
                    metrics,
                    residue=residue,
                    baseline_residue=baseline_residue,
                )
            )
            if residue is not None and residue_pixels:
                aggregate_sum += float(residue_sum)
                aggregate_count += int(residue_pixels)
            if baseline_residue is not None and baseline_residue_pixels:
                baseline_sum += float(baseline_residue_sum)
                baseline_count += int(baseline_residue_pixels)
            _write_image(
                output / "candidate_images" / f"{page.page_id}_candidate.png",
                candidate,
            )
            _write_image(
                output / "final_masks" / f"{page.page_id}_final_mask.png",
                restricted_mask,
            )
            _write_image(
                output / "changed_masks" / f"{page.page_id}_changed.png",
                changed,
            )
            _write_image(
                output / "restored_source_masks" / f"{page.page_id}_restored.png",
                np.where(restored, 255, 0).astype(np.uint8),
            )
            review = output / "review"
            review.mkdir(parents=True, exist_ok=True)
            _review_panel(
                source,
                product_image,
                candidate,
                restricted_mask,
                masks.target,
                masks.protected,
            ).save(review / f"{page.page_id}_review.png")

        aggregate = aggregate_sum / aggregate_count if aggregate_count else None
        baseline_aggregate = (
            baseline_sum / baseline_count if baseline_count else None
        )
        if (
            aggregate is not None
            and baseline_aggregate is not None
            and aggregate > baseline_aggregate
        ):
            failures.append("aggregate:residue_worse_than_product_baseline")
        result = {
            "schema_version": "inpaint-final-protection-composite-v3",
            "candidate_id": candidate_id,
            "manifest_sha256": manifest_sha256,
            "product_summary_sha256": _sha256(product_run / "metrics" / "summary.json"),
            "stage1_result_sha256": _sha256(stage1_result_path),
            "page_count": len(rows),
            "inference_call_count": 0,
            "aggregate_residue_score": aggregate,
            "baseline_aggregate_residue_score": baseline_aggregate,
            "required_gate_failures": sorted(set(failures)),
            "required_gate_failure_count": len(set(failures)),
            "pages": rows,
        }
        (output / "stage2-results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        if managed is not None:
            managed.complete(
                metadata={
                    "candidate_id": result["candidate_id"],
                    "required_gate_failure_count": result[
                        "required_gate_failure_count"
                    ],
                }
            )
        if args.require_quality_gates and failures:
            return 1
        return 0
    except Exception as exc:
        if managed is not None:
            managed.fail(exc, metadata={"manifest": manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
