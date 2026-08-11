#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarking.inpaint_detector_bakeoff.ballons_e2e import (  # noqa: E402
    BallonsEndToEndReference,
)
from benchmarking.inpaint_detector_bakeoff.stage1 import (  # noqa: E402
    load_page_masks,
    load_stage1_manifest,
)
from benchmarking.inpaint_detector_bakeoff.stage2 import (  # noqa: E402
    score_stage2_page,
)
from modules.utils.download import ModelDownloader, ModelID  # noqa: E402
from scripts.validation_artifact_harness import (  # noqa: E402
    select_managed_output_directory,
)


FAMILY = "inpaint-ballons-e2e-reference-v2"
CATEGORY = "40-inpaint-mask-render"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        nested = value.get("path")
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None


def _manifest_entries(path: Path) -> dict[str, dict[str, object]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        str(row.get("page_id") or ""): row
        for row in payload.get("pages", [])
        if isinstance(row, dict)
    }


def _read_bgr(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise FileNotFoundError(f"unable to read image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"unable to write image: {path}")


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
    candidate: np.ndarray,
    detector: np.ndarray,
    target: np.ndarray,
    protected: np.ndarray,
) -> Image.Image:
    left = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    middle = cv2.cvtColor(candidate, cv2.COLOR_BGR2RGB)
    right_bgr = _overlay(source, detector, (255, 0, 0))
    right_bgr = _overlay(right_bgr, target, (0, 255, 0))
    right_bgr = _overlay(right_bgr, protected, (0, 0, 255))
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    images = [Image.fromarray(value) for value in (left, middle, right)]
    thumb_size = (420, 600)
    label_height = 30
    panel = Image.new("RGB", (thumb_size[0] * 3, thumb_size[1] + label_height), "white")
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(
        zip(images, ("source", "Ballons candidate", "claim/target/protect"))
    ):
        image.thumbnail(thumb_size, Image.Resampling.LANCZOS)
        x = index * thumb_size[0] + (thumb_size[0] - image.width) // 2
        y = (thumb_size[1] - image.height) // 2
        panel.paste(image, (x, y))
        draw.text((index * thumb_size[0] + 8, thumb_size[1] + 8), label, fill="black")
    return panel


def _repo_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the pinned Ballons CTD plus LaMa end-to-end reference on an annotated manifest.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ballons-root", type=Path, required=True)
    parser.add_argument("--detector-model", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")
    parser.add_argument("--detect-size", type=int, default=1280)
    parser.add_argument("--inpaint-size", type=int, default=1536)
    parser.add_argument("--require-quality-gates", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = args.manifest.resolve()
    ballons_root = args.ballons_root.resolve()
    detector_model = Path(
        args.detector_model or ModelDownloader.primary_path(ModelID.CTD_TORCH)
    ).resolve()
    output_root, managed = select_managed_output_directory(
        family=FAMILY,
        category=CATEGORY,
        explicit_output_directory=args.output_dir,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        pages = load_stage1_manifest(manifest)
        entries = _manifest_entries(manifest)
        reference = BallonsEndToEndReference(
            ballons_root=ballons_root,
            detector_model_path=detector_model,
            device=args.device,
            precision=args.precision,
            detect_size=args.detect_size,
            inpaint_size=args.inpaint_size,
        )
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        records: list[dict[str, object]] = []
        aggregate_residue_sum = 0.0
        aggregate_residue_count = 0
        baseline_residue_sum = 0.0
        baseline_residue_count = 0
        quality_failures: list[str] = []
        for page in pages:
            source = _read_bgr(page.source_image)
            masks = load_page_masks(page, source.shape[:2])
            entry = entries[page.page_id]
            baseline_path = _path_value(entry.get("baseline"))
            baseline = _read_bgr(baseline_path) if baseline_path else None
            result = reference.infer(source)
            metrics, changed = score_stage2_page(
                source,
                result.image_bgr,
                result.detector_mask,
                masks,
                baseline=baseline,
            )
            metrics.update({"page_id": page.page_id, "runtime": result.runtime})
            records.append(metrics)

            candidate_root = output_root / "candidate_images"
            mask_root = output_root / "detector_masks"
            changed_root = output_root / "changed_masks"
            review_root = output_root / "review"
            _write_image(candidate_root / f"{page.page_id}_candidate.png", result.image_bgr)
            _write_image(mask_root / f"{page.page_id}_detector_mask.png", result.detector_mask)
            _write_image(changed_root / f"{page.page_id}_changed.png", changed)
            review_root.mkdir(parents=True, exist_ok=True)
            _review_panel(
                source,
                result.image_bgr,
                result.detector_mask,
                masks.target,
                masks.protected,
            ).save(review_root / f"{page.page_id}_review.png")

            score = metrics["residue_score"]
            count = int(metrics["residue_source_contrast_pixel_count"])
            if score is not None and count:
                aggregate_residue_sum += float(score) * count
                aggregate_residue_count += count
            baseline_score = metrics["baseline_residue_score"]
            if baseline_score is not None and count:
                baseline_residue_sum += float(baseline_score) * count
                baseline_residue_count += count

            if int(metrics["protected_changed_pixel_count"]) != 0:
                quality_failures.append(f"{page.page_id}:protected_structure_changed")
            if int(metrics["ambiguous_changed_pixel_count"]) != 0:
                quality_failures.append(f"{page.page_id}:ambiguous_structure_changed")
            if int(metrics["changed_outside_detector_mask_pixel_count"]) != 0:
                quality_failures.append(
                    f"{page.page_id}:changed_outside_detector_mask"
                )
            coverage = metrics["target_detector_coverage"]
            if coverage is not None and float(coverage) < 0.98:
                quality_failures.append(f"{page.page_id}:target_coverage_below_98pct")
            minimum = metrics["minimum_target_component_coverage"]
            if minimum is not None and float(minimum) < 0.98:
                quality_failures.append(f"{page.page_id}:target_component_coverage_below_98pct")
            delta = metrics["residue_score_delta_from_baseline"]
            if delta is not None and float(delta) > 0.0:
                quality_failures.append(f"{page.page_id}:residue_worse_than_baseline")
            if int(result.runtime["cpu_fallback_count"]) != 0:
                quality_failures.append(f"{page.page_id}:cpu_fallback_used")

        aggregate_score = (
            aggregate_residue_sum / aggregate_residue_count
            if aggregate_residue_count
            else None
        )
        baseline_aggregate = (
            baseline_residue_sum / baseline_residue_count
            if baseline_residue_count
            else None
        )
        if (
            aggregate_score is not None
            and baseline_aggregate is not None
            and aggregate_score >= baseline_aggregate
        ):
            quality_failures.append("aggregate:residue_not_reduced_from_baseline")

        summary = {
            "schema_version": "inpaint-ballons-e2e-stage2-v1",
            "manifest_sha256": _sha256(manifest),
            "ballons_reference_head": _repo_head(ballons_root),
            "detector_model_sha256": _sha256(detector_model),
            "lama_model_sha256": _sha256(
                Path(ModelDownloader.primary_path(ModelID.LAMA_LARGE_512PX))
            ),
            "page_count": len(records),
            "device": args.device,
            "precision": args.precision,
            "aggregate_residue_score": aggregate_score,
            "baseline_aggregate_residue_score": baseline_aggregate,
            "protected_changed_pixel_count": sum(
                int(row["protected_changed_pixel_count"]) for row in records
            ),
            "ambiguous_changed_pixel_count": sum(
                int(row["ambiguous_changed_pixel_count"]) for row in records
            ),
            "changed_outside_detector_mask_pixel_count": sum(
                int(row["changed_outside_detector_mask_pixel_count"]) for row in records
            ),
            "cpu_fallback_count": sum(
                int(row["runtime"]["cpu_fallback_count"]) for row in records
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
            "quality_gate_failures": quality_failures,
            "pages": records,
        }
        _write_json(output_root / "stage2-results.json", summary)
        if managed is not None:
            managed.complete(
                metadata={
                    "manifest_sha256": summary["manifest_sha256"],
                    "ballons_reference_head": summary["ballons_reference_head"],
                    "quality_gate_failure_count": len(quality_failures),
                }
            )
            mismatches = managed.verify()
            if mismatches:
                raise RuntimeError("managed artifact verification failed: " + "; ".join(mismatches))
            print(str(managed.run_root))
        else:
            print(str(output_root))
        print(
            json.dumps(
                {key: value for key, value in summary.items() if key != "pages"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1 if args.require_quality_gates and quality_failures else 0
    except BaseException as error:
        if managed is not None:
            managed.fail(error, metadata={"manifest": manifest.name})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
