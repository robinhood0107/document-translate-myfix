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
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    composite_positive_result,
    score_stage2_page,
)
from modules.inpainting.source_lama_blockwise import SourceLaMaLarge  # noqa: E402
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-positive-mask-stage2-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.partial")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _path_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, dict):
        path = value.get("path")
        if isinstance(path, str) and path.strip():
            return path.strip()
    return None


def _entries(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("page_id") or ""): row
        for row in payload.get("pages", [])
        if isinstance(row, dict)
    }


def _read_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(path)
    return image


def _read_mask(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None or mask.size == 0:
        raise FileNotFoundError(path)
    if mask.shape != shape:
        raise ValueError(f"mask shape mismatch: {mask.shape} != {shape}")
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"failed to write image: {path}")


def _overlay(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = image.copy()
    selected = mask > 0
    if np.any(selected):
        tint = np.empty_like(result)
        tint[:] = color
        result[selected] = np.round(
            result[selected].astype(np.float32) * 0.45
            + tint[selected].astype(np.float32) * 0.55
        ).astype(np.uint8)
    return result


def _review_panel(
    source: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    final_mask: np.ndarray,
    target: np.ndarray,
    protected: np.ndarray,
) -> Image.Image:
    overlay = _overlay(source, final_mask, (255, 0, 0))
    overlay = _overlay(overlay, target, (0, 255, 0))
    overlay = _overlay(overlay, protected, (0, 0, 255))
    images = [
        Image.fromarray(cv2.cvtColor(value, cv2.COLOR_BGR2RGB))
        for value in (source, baseline, candidate, overlay)
    ]
    labels = ("source", "rewritten PR3", "positive LaMa", "mask/target/protect")
    thumb_size = (360, 540)
    label_height = 30
    panel = Image.new(
        "RGB",
        (thumb_size[0] * len(images), thumb_size[1] + label_height),
        "white",
    )
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(zip(images, labels)):
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = index * thumb_size[0] + (thumb_size[0] - image.width) // 2
        y = (thumb_size[1] - image.height) // 2
        panel.paste(image, (x, y))
        draw.text((index * thumb_size[0] + 8, thumb_size[1] + 8), label, fill="black")
    return panel


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply one page-level LaMa call per detector-positive mask and "
            "composite only exact edit pixels onto a sealed baseline."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage1-run", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--require-quality-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.resolve()
    stage1_run = args.stage1_run.resolve()
    stage1_result = stage1_run / "stage1-results.json"
    edit_root = stage1_run / "positive_edit_masks"
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest)
        entries = _entries(manifest)
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
        inference_count = 0
        diagnostics_before = len(inpainter.run_diagnostics)
        for page in pages:
            entry = entries[page.page_id]
            source = _read_bgr(page.source_image)
            shape = source.shape[:2]
            masks = load_page_masks(page, shape)
            baseline_path = _path_value(entry.get("baseline"))
            baseline_mask_path = _path_value(entry.get("baseline_mask"))
            if not baseline_path or not baseline_mask_path:
                raise ValueError(f"sealed baseline is missing for {page.page_id}")
            baseline = _read_bgr(baseline_path)
            baseline_mask = _read_mask(baseline_mask_path, shape)
            positive_edit = _read_mask(
                edit_root / f"{page.page_id}_positive_edit.png",
                shape,
            )
            if np.any(positive_edit):
                generated_rgb = inpainter.memory_safe_inpaint(
                    cv2.cvtColor(source, cv2.COLOR_BGR2RGB),
                    positive_edit,
                )
                generated = cv2.cvtColor(generated_rgb, cv2.COLOR_RGB2BGR)
                inference_count += 1
            else:
                generated = source
            candidate, final_mask = composite_positive_result(
                baseline,
                generated,
                positive_edit,
                baseline_mask,
            )
            metrics, changed = score_stage2_page(
                source,
                candidate,
                final_mask,
                masks,
                baseline=baseline,
            )
            metrics.update(
                {
                    "page_id": page.page_id,
                    "positive_edit_pixel_count": int(
                        np.count_nonzero(positive_edit)
                    ),
                    "positive_inference_call_count": int(np.any(positive_edit)),
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

            score = metrics["residue_score"]
            count = int(metrics["residue_source_contrast_pixel_count"])
            baseline_score = metrics["baseline_residue_score"]
            if score is not None and count:
                aggregate_sum += float(score) * count
                aggregate_count += count
            if baseline_score is not None and count:
                baseline_sum += float(baseline_score) * count
                baseline_count += count
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

        aggregate = aggregate_sum / aggregate_count if aggregate_count else None
        baseline_aggregate = (
            baseline_sum / baseline_count if baseline_count else None
        )
        if (
            aggregate is not None
            and baseline_aggregate is not None
            and aggregate >= baseline_aggregate
        ):
            failures.append("aggregate:residue_not_reduced_from_baseline")
        diagnostics = inpainter.run_diagnostics[diagnostics_before:]
        cpu_fallback_count = sum(
            int(bool(row.get("cpu_fallback_used", False))) for row in diagnostics
        )
        if cpu_fallback_count:
            failures.append("aggregate:cpu_fallback_used")

        summary = {
            "schema_version": "inpaint-positive-mask-stage2-v1",
            "manifest_sha256": _sha256(manifest),
            "stage1_result_sha256": _sha256(stage1_result),
            "lama_model_sha256": _sha256(
                Path(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX))
            ),
            "page_count": len(rows),
            "device": args.device,
            "precision": args.precision,
            "positive_inference_call_count": inference_count,
            "cpu_fallback_count": cpu_fallback_count,
            "aggregate_residue_score": aggregate,
            "baseline_aggregate_residue_score": baseline_aggregate,
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
                    "manifest_sha256": summary["manifest_sha256"],
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
                {key: value for key, value in summary.items() if key not in {"pages", "diagnostics"}},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1 if args.require_quality_gates and failures else 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
