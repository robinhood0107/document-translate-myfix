#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    composite_replacement_result,
    residue_score,
    score_stage2_page,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge  # noqa: E402
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.benchmark_inpaint_detector_bakeoff import (  # noqa: E402
    _existing_edit_paths,
)
from scripts.benchmark_inpaint_positive_mask_stage2 import (  # noqa: E402
    _entries,
    _path_value,
    _read_bgr,
    _read_mask,
    _review_panel,
    _write_image,
    _write_json,
)
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-replacement-mask-stage2-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one immutable-original LaMa call per page with a detector-"
            "verified source replacement mask."
        )
    )
    parser.add_argument("--evaluation-manifest", type=Path, required=True)
    parser.add_argument("--routing-manifest", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-quality-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    evaluation_manifest = args.evaluation_manifest.resolve()
    routing_manifest = args.routing_manifest.resolve()
    stage1_run = args.stage1_run.resolve()
    stage1_result = stage1_run / "stage1-results.json"
    replacement_root = stage1_run / "replacement_edit_masks"
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        evaluation_pages = load_stage1_manifest(evaluation_manifest)
        evaluation_entries = _entries(evaluation_manifest)
        routing_entries = _entries(routing_manifest)
        routing_existing = _existing_edit_paths(routing_manifest)
        if set(evaluation_entries) != set(routing_entries):
            raise ValueError("evaluation and routing manifests have different pages")

        inpainter = SourceLaMaLarge(
            device=args.device,
            precision=args.precision,
            inpaint_size=args.inpaint_size,
        )
        inpainter.ensure_loaded()
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        rows: list[dict[str, object]] = []
        failures: list[str] = []
        aggregate_sum = 0.0
        aggregate_count = 0
        baseline_sum = 0.0
        baseline_count = 0
        source_edit_sum = 0.0
        source_edit_count = 0
        baseline_source_edit_sum = 0.0
        baseline_source_edit_count = 0
        inference_count = 0
        diagnostics_before = len(inpainter.run_diagnostics)
        for page in evaluation_pages:
            entry = evaluation_entries[page.page_id]
            source = _read_bgr(page.source_image)
            shape = source.shape[:2]
            masks = load_page_masks(page, shape)
            baseline_path = _path_value(entry.get("baseline"))
            baseline_mask_path = _path_value(entry.get("baseline_mask"))
            existing_path = routing_existing.get(page.page_id)
            if not baseline_path or not baseline_mask_path or not existing_path:
                raise ValueError(f"sealed replacement inputs missing for {page.page_id}")
            baseline = _read_bgr(baseline_path)
            baseline_mask = _read_mask(baseline_mask_path, shape)
            existing_source = _read_mask(existing_path, shape)
            replacement_edit = _read_mask(
                replacement_root / f"{page.page_id}_replacement_edit.png",
                shape,
            )
            page_inference_count = 0
            if np.any(replacement_edit):
                generated_rgb = inpainter.memory_safe_inpaint(
                    cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
                    replacement_edit,
                )
                generated = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
                inference_count += 1
                page_inference_count = 1
            else:
                generated = source
            candidate, final_mask = composite_replacement_result(
                source,
                baseline,
                generated,
                replacement_edit,
                baseline_mask,
                existing_source,
            )
            metrics, changed = score_stage2_page(
                source,
                candidate,
                final_mask,
                masks,
                baseline=baseline,
            )
            source_edit_score, _source_edit_sum, source_edit_pixels = residue_score(
                source,
                candidate,
                existing_source,
            )
            (
                baseline_source_edit_score,
                _baseline_source_edit_sum,
                baseline_source_edit_pixels,
            ) = residue_score(source, baseline, existing_source)
            safe_baseline = (baseline_mask > 0) & (existing_source == 0)
            metrics.update(
                {
                    "page_id": page.page_id,
                    "replacement_edit_pixel_count": int(
                        np.count_nonzero(replacement_edit)
                    ),
                    "retained_non_source_baseline_pixel_count": int(
                        np.count_nonzero(safe_baseline)
                    ),
                    "replacement_inference_call_count": page_inference_count,
                    "source_edit_residue_score": source_edit_score,
                    "baseline_source_edit_residue_score": baseline_source_edit_score,
                    "source_edit_residue_delta_from_baseline": (
                        float(source_edit_score) - float(baseline_source_edit_score)
                        if source_edit_score is not None
                        and baseline_source_edit_score is not None
                        else None
                    ),
                    "source_edit_residue_contrast_pixel_count": source_edit_pixels,
                }
            )
            rows.append(metrics)
            _write_image(
                output_root / "candidate_images" / f"{page.page_id}_candidate.png",
                candidate,
            )
            _write_image(
                output_root / "final_masks" / f"{page.page_id}_final_mask.png",
                final_mask,
            )
            _write_image(
                output_root / "changed_masks" / f"{page.page_id}_changed.png",
                changed,
            )
            review_root = output_root / "review"
            review_root.mkdir(parents=True, exist_ok=True)
            _review_panel(
                source,
                baseline,
                candidate,
                final_mask,
                masks.target,
                masks.protected,
            ).save(review_root / f"{page.page_id}_review.png")

            if int(metrics["protected_changed_pixel_count"]) != 0:
                failures.append(f"{page.page_id}:protected_structure_changed")
            if int(metrics["ambiguous_changed_pixel_count"]) != 0:
                failures.append(f"{page.page_id}:ambiguous_structure_changed")
            if int(metrics["changed_outside_detector_mask_pixel_count"]) != 0:
                failures.append(f"{page.page_id}:changed_outside_final_mask")
            coverage = metrics["target_detector_coverage"]
            if coverage is not None and float(coverage) < 0.98:
                failures.append(f"{page.page_id}:target_coverage_below_98pct")
            minimum = metrics["minimum_target_component_coverage"]
            if minimum is not None and float(minimum) < 0.98:
                failures.append(
                    f"{page.page_id}:target_component_coverage_below_98pct"
                )
            delta = metrics["residue_score_delta_from_baseline"]
            if delta is not None and float(delta) > 0.0:
                failures.append(f"{page.page_id}:residue_worse_than_baseline")
            source_edit_delta = metrics["source_edit_residue_delta_from_baseline"]
            if source_edit_delta is not None and float(source_edit_delta) > 0.0:
                failures.append(
                    f"{page.page_id}:source_edit_residue_worse_than_baseline"
                )
            score = metrics["residue_score"]
            baseline_score = metrics["baseline_residue_score"]
            count = int(metrics["residue_source_contrast_pixel_count"])
            if score is not None and count:
                aggregate_sum += float(score) * count
                aggregate_count += count
            if baseline_score is not None and count:
                baseline_sum += float(baseline_score) * count
                baseline_count += count
            if source_edit_score is not None and source_edit_pixels:
                source_edit_sum += float(source_edit_score) * source_edit_pixels
                source_edit_count += source_edit_pixels
            if baseline_source_edit_score is not None and baseline_source_edit_pixels:
                baseline_source_edit_sum += (
                    float(baseline_source_edit_score) * baseline_source_edit_pixels
                )
                baseline_source_edit_count += baseline_source_edit_pixels

        aggregate = aggregate_sum / aggregate_count if aggregate_count else None
        baseline_aggregate = (
            baseline_sum / baseline_count if baseline_count else None
        )
        source_edit_aggregate = (
            source_edit_sum / source_edit_count if source_edit_count else None
        )
        baseline_source_edit_aggregate = (
            baseline_source_edit_sum / baseline_source_edit_count
            if baseline_source_edit_count
            else None
        )
        if (
            aggregate is not None
            and baseline_aggregate is not None
            and aggregate >= baseline_aggregate
        ):
            failures.append("aggregate:residue_not_reduced_from_baseline")
        if (
            source_edit_aggregate is not None
            and baseline_source_edit_aggregate is not None
            and source_edit_aggregate > baseline_source_edit_aggregate
        ):
            failures.append(
                "aggregate:source_edit_residue_worse_than_baseline"
            )
        diagnostics = inpainter.run_diagnostics[diagnostics_before:]
        cpu_fallback_count = sum(
            int(bool(row.get("cpu_fallback_used", False))) for row in diagnostics
        )
        if cpu_fallback_count:
            failures.append("aggregate:cpu_fallback_used")

        summary = {
            "schema_version": "inpaint-replacement-mask-stage2-v1",
            "evaluation_manifest_sha256": _sha256(evaluation_manifest),
            "routing_manifest_sha256": _sha256(routing_manifest),
            "stage1_result_sha256": _sha256(stage1_result),
            "lama_model_sha256": _sha256(
                Path(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX))
            ),
            "page_count": len(rows),
            "device": args.device,
            "precision": args.precision,
            "replacement_inference_call_count": inference_count,
            "cpu_fallback_count": cpu_fallback_count,
            "aggregate_residue_score": aggregate,
            "baseline_aggregate_residue_score": baseline_aggregate,
            "source_edit_aggregate_residue_score": source_edit_aggregate,
            "baseline_source_edit_aggregate_residue_score": (
                baseline_source_edit_aggregate
            ),
            "protected_changed_pixel_count": sum(
                int(row["protected_changed_pixel_count"]) for row in rows
            ),
            "ambiguous_changed_pixel_count": sum(
                int(row["ambiguous_changed_pixel_count"]) for row in rows
            ),
            "changed_outside_final_mask_pixel_count": sum(
                int(row["changed_outside_detector_mask_pixel_count"]) for row in rows
            ),
            "peak_vram_allocated_mib": (
                float(torch.cuda.max_memory_allocated()) / (1024.0 * 1024.0)
                if args.device == "cuda"
                else None
            ),
            "peak_vram_reserved_mib": (
                float(torch.cuda.max_memory_reserved()) / (1024.0 * 1024.0)
                if args.device == "cuda"
                else None
            ),
            "quality_gate_failures": failures,
            "diagnostics": diagnostics,
            "pages": rows,
        }
        _write_json(output_root / "stage2-results.json", summary)
        if managed is not None:
            managed.complete(
                metadata={
                    "stage1_result_sha256": summary["stage1_result_sha256"],
                    "quality_gate_failure_count": len(failures),
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError(
                    "managed artifact verification failed: " + "; ".join(mismatches)
                )
            print(str(managed.run_root))
        else:
            print(str(output_root))
        print(
            json.dumps(
                {
                    key: value
                    for key, value in summary.items()
                    if key not in {"pages", "diagnostics"}
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1 if args.require_quality_gates and failures else 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"evaluation_manifest": evaluation_manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
