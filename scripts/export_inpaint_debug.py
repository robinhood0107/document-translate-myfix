#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imkit as imk
import numpy as np

from modules.detection.processor import TextBlockDetector
from modules.inpainting.source_lama_blockwise import source_lama_blockwise_inpaint
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    ensure_three_channel,
)
from modules.utils.mask_inpaint_mode import DEFAULT_MASK_INPAINT_MODE
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.inpainting_runtime import inpainter_default_settings, normalize_inpainter_key


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class _UIStub:
    value_mappings: dict[str, str]

    def tr(self, text: str) -> str:
        return text


class _SettingsStub:
    def __init__(self, *, use_gpu: bool) -> None:
        self._use_gpu = use_gpu
        self.ui = _UIStub(
            value_mappings={
                "Resize": "Resize",
                "Original": "Original",
                "Crop": "Crop",
            }
        )

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "detector":
            return "RT-DETR-v2"
        if tool_type == "inpainter":
            return "lama_large_512px"
        raise KeyError(tool_type)

    def get_mask_inpaint_mode(self) -> str:
        return DEFAULT_MASK_INPAINT_MODE

    def is_gpu_enabled(self) -> bool:
        return self._use_gpu

    def get_hd_strategy_settings(self) -> dict:
        return {"strategy": "Resize", "resize_limit": 960}

    def get_mask_refiner_settings(self) -> dict:
        return {
            "mask_refiner": "legacy_bbox",
            "mask_inpaint_mode": DEFAULT_MASK_INPAINT_MODE,
            "keep_existing_lines": False,
        }

    def get_inpainter_runtime_settings(self, inpainter_key: str | None = None) -> dict:
        normalized = normalize_inpainter_key(inpainter_key or "lama_large_512px")
        return inpainter_default_settings(normalized)


def _iter_sample_images(input_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and fnmatch.fnmatch(path.name, pattern)
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imk.write_image(str(path), ensure_three_channel(image))


def _mask_to_rgb(mask: np.ndarray | None, image_shape: tuple[int, ...]) -> np.ndarray:
    if mask is None:
        normalized = np.zeros(image_shape[:2], dtype=np.uint8)
    else:
        normalized = np.asarray(mask)
        if normalized.ndim == 3:
            normalized = normalized[:, :, 0]
        normalized = np.where(normalized > 0, 255, 0).astype(np.uint8)
    return np.stack([normalized] * 3, axis=-1)


def _build_mask_overlay(image: np.ndarray, mask: np.ndarray | None, *, color=(255, 0, 0), alpha: float = 0.35) -> np.ndarray:
    base = ensure_three_channel(image).astype(np.float32)
    overlay = base.copy()
    if mask is not None:
        mask_pixels = np.asarray(mask) > 0
        if np.any(mask_pixels):
            tint = np.array(color, dtype=np.float32)
            overlay[mask_pixels] = base[mask_pixels] * (1.0 - alpha) + tint * alpha
    return np.clip(np.round(overlay), 0, 255).astype(np.uint8)


def _build_compare_panel(
    *,
    source: np.ndarray,
    legacy_base_mask: np.ndarray | None,
    hard_box_rescue_mask: np.ndarray | None,
    mask_overlay: np.ndarray,
    cleaned: np.ndarray,
) -> np.ndarray:
    pieces = [
        ensure_three_channel(source),
        _mask_to_rgb(legacy_base_mask, source.shape),
        _mask_to_rgb(hard_box_rescue_mask, source.shape),
        ensure_three_channel(mask_overlay),
        ensure_three_channel(cleaned),
    ]
    return np.concatenate(pieces, axis=1)


def _build_metrics_payload(image_path: Path, metadata: dict, mask_details: dict) -> dict:
    blocks = metadata.get("blocks", [])
    legacy_fill_ratios = [float(block.get("legacy_fill_ratio", 0.0) or 0.0) for block in blocks]
    rescue_fill_ratios = [float(block.get("rescue_fill_ratio", 0.0) or 0.0) for block in blocks]
    return {
        "image": image_path.name,
        "relative_parent": image_path.parent.name,
        "detector_key": metadata.get("detector_key", ""),
        "inpainter": metadata.get("inpainter", ""),
        "block_count": int(metadata.get("block_count", 0) or 0),
        "hard_box_applied_count": int(metadata.get("hard_box_applied_count", 0) or 0),
        "hard_box_reason_totals": dict(metadata.get("hard_box_reason_totals", {}) or {}),
        "legacy_base_mask_pixel_count": int(metadata.get("legacy_base_mask_pixel_count", 0) or 0),
        "hard_box_rescue_mask_pixel_count": int(metadata.get("hard_box_rescue_mask_pixel_count", 0) or 0),
        "final_mask_pixel_count": int(metadata.get("final_mask_pixel_count", 0) or 0),
        "legacy_fill_ratio_values": legacy_fill_ratios,
        "rescue_fill_ratio_values": rescue_fill_ratios,
        "legacy_fill_ratio_avg": float(sum(legacy_fill_ratios) / len(legacy_fill_ratios)) if legacy_fill_ratios else 0.0,
        "rescue_fill_ratio_avg": float(sum(rescue_fill_ratios) / len(rescue_fill_ratios)) if rescue_fill_ratios else 0.0,
        "refiner_backend": metadata.get("refiner_backend", ""),
        "refiner_device": metadata.get("refiner_device", ""),
        "blocks": blocks,
        "mask_details": {
            "mask_refiner": mask_details.get("mask_refiner", "legacy_bbox"),
            "mask_inpaint_mode": mask_details.get("mask_inpaint_mode", DEFAULT_MASK_INPAINT_MODE),
            "fallback_used": bool(mask_details.get("fallback_used", False)),
        },
    }


def _process_image(
    *,
    image_path: Path,
    output_root: Path,
    input_root: Path,
    detector: TextBlockDetector,
    inpainter,
    settings: _SettingsStub,
) -> dict:
    image = imk.read_image(str(image_path))
    if image is None:
        raise RuntimeError("failed to read image")
    image = ensure_three_channel(image)
    blocks = detector.detect(image) or []

    detector_key = getattr(detector, "detector", None) or settings.get_tool_selection("detector")
    detector_engine = detector.last_engine_name or ""
    detector_device = detector.last_device or resolve_device(settings.is_gpu_enabled(), backend="onnx")
    mask_details = generate_mask(
        image,
        blocks,
        settings=settings.get_mask_refiner_settings(),
        return_details=True,
    )
    legacy_base_mask = mask_details.get("legacy_base_mask")
    hard_box_rescue_mask = mask_details.get("hard_box_rescue_mask")
    final_mask = mask_details.get("final_mask")

    if final_mask is not None and np.any(final_mask):
        cleaned = source_lama_blockwise_inpaint(
            image,
            final_mask,
            blocks,
            inpainter,
            get_config(settings),
            check_need_inpaint=True,
        )
        cleaned = ensure_three_channel(cleaned)
    else:
        cleaned = image.copy()

    cleanup_stats = {
        "applied": bool(mask_details.get("hard_box_applied_count", 0)),
        "component_count": int(mask_details.get("hard_box_rescue_mask_pixel_count", 0) or 0),
        "block_count": int(mask_details.get("hard_box_applied_count", 0) or 0),
    }
    metadata = build_inpaint_debug_metadata(
        image_path=str(image_path),
        run_type="sample_inpaint_only",
        detector_key=detector_key,
        detector_engine=detector_engine,
        device=detector_device,
        inpainter=settings.get_tool_selection("inpainter"),
        hd_strategy="Resize",
        blocks=blocks,
        raw_mask=legacy_base_mask,
        final_mask=final_mask,
        final_mask_pre_expand=mask_details.get("final_mask_pre_expand"),
        final_mask_post_expand=mask_details.get("final_mask_post_expand"),
        cleanup_delta=hard_box_rescue_mask,
        cleanup_stats=cleanup_stats,
        mask_refiner=str(mask_details.get("mask_refiner", "legacy_bbox") or "legacy_bbox"),
        protect_mask_applied=False,
        protect_mask=mask_details.get("protect_mask"),
        refiner_backend=str(mask_details.get("refiner_backend", "legacy_bbox_rescue") or "legacy_bbox_rescue"),
        refiner_device=str(mask_details.get("refiner_device", "cpu") or "cpu"),
        inpainter_backend=str(get_inpainter_runtime(settings)["backend"] or "torch"),
        legacy_base_mask=legacy_base_mask,
        hard_box_rescue_mask=hard_box_rescue_mask,
        hard_box_applied_count=int(mask_details.get("hard_box_applied_count", 0) or 0),
        hard_box_reason_totals=dict(mask_details.get("hard_box_reason_totals", {}) or {}),
    )
    metrics = _build_metrics_payload(image_path, metadata, mask_details)

    rel_parent = image_path.parent.relative_to(input_root) if image_path.parent != input_root else Path("root")
    image_output = output_root / rel_parent / image_path.stem
    source_path = image_output / "source.png"
    legacy_mask_path = image_output / "legacy_base_mask.png"
    rescue_mask_path = image_output / "hard_box_rescue_mask.png"
    final_mask_path = image_output / "final_mask.png"
    overlay_path = image_output / "mask_overlay.png"
    cleaned_path = image_output / "cleaned.png"
    panel_path = image_output / "compare_panel.png"
    metrics_path = image_output / "metrics.json"

    mask_overlay = _build_mask_overlay(image, final_mask)
    panel = _build_compare_panel(
        source=image,
        legacy_base_mask=legacy_base_mask,
        hard_box_rescue_mask=hard_box_rescue_mask,
        mask_overlay=mask_overlay,
        cleaned=cleaned,
    )

    _write_image(source_path, image)
    _write_image(legacy_mask_path, _mask_to_rgb(legacy_base_mask, image.shape))
    _write_image(rescue_mask_path, _mask_to_rgb(hard_box_rescue_mask, image.shape))
    _write_image(final_mask_path, _mask_to_rgb(final_mask, image.shape))
    _write_image(overlay_path, mask_overlay)
    _write_image(cleaned_path, cleaned)
    _write_image(panel_path, panel)
    _write_json(metrics_path, metrics)

    return {
        "image": image_path.name,
        "relative_parent": rel_parent.as_posix(),
        "source": source_path,
        "legacy_base_mask": legacy_mask_path,
        "hard_box_rescue_mask": rescue_mask_path,
        "final_mask": final_mask_path,
        "mask_overlay": overlay_path,
        "cleaned": cleaned_path,
        "compare_panel": panel_path,
        "metrics": metrics_path,
        "block_count": int(metadata.get("block_count", 0) or 0),
        "hard_box_applied_count": int(metadata.get("hard_box_applied_count", 0) or 0),
        "legacy_base_mask_pixel_count": int(metadata.get("legacy_base_mask_pixel_count", 0) or 0),
        "hard_box_rescue_mask_pixel_count": int(metadata.get("hard_box_rescue_mask_pixel_count", 0) or 0),
        "final_mask_pixel_count": int(metadata.get("final_mask_pixel_count", 0) or 0),
        "legacy_fill_ratio_values": list(metrics.get("legacy_fill_ratio_values", [])),
        "rescue_fill_ratio_values": list(metrics.get("rescue_fill_ratio_values", [])),
    }


def _write_index(root_output: Path, records: list[dict], summary: dict) -> None:
    def rel(path: Path) -> str:
        return path.relative_to(root_output).as_posix()

    lines = [
        "# Sample Inpaint-Only Verification",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"Detector: `{summary['detector_key']}`  ",
        f"Inpainter: `{summary['inpainter']}`  ",
        f"Use GPU: `{summary['use_gpu']}`",
        "",
        "| Image | Group | Source | Legacy | Rescue | Final | Overlay | Cleaned | Panel | Metrics | Hard Box | Final Mask Px |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: |",
    ]
    for record in records:
        lines.append(
            f"| `{record['image']}` | `{record['relative_parent']}` | [source]({rel(record['source'])}) | "
            f"[legacy]({rel(record['legacy_base_mask'])}) | [rescue]({rel(record['hard_box_rescue_mask'])}) | "
            f"[final]({rel(record['final_mask'])}) | [overlay]({rel(record['mask_overlay'])}) | "
            f"[cleaned]({rel(record['cleaned'])}) | [panel]({rel(record['compare_panel'])}) | "
            f"[metrics]({rel(record['metrics'])}) | {record['hard_box_applied_count']} | {record['final_mask_pixel_count']} |"
        )
    (root_output / "index.md").write_text("\n".join(lines), encoding="utf-8")


def _copy_review_samples(root_output: Path, records: list[dict], review_count: int) -> list[dict]:
    review_dir = root_output / "review_samples"
    review_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        records,
        key=lambda item: (
            int(item["hard_box_applied_count"]),
            int(item["hard_box_rescue_mask_pixel_count"]),
            int(item["final_mask_pixel_count"]),
        ),
        reverse=True,
    )
    copied = []
    for index, record in enumerate(ranked[:review_count], start=1):
        panel_target = review_dir / f"{index:02d}_{record['image']}_panel.png"
        metrics_target = review_dir / f"{index:02d}_{record['image']}_metrics.json"
        shutil.copy2(record["compare_panel"], panel_target)
        shutil.copy2(record["metrics"], metrics_target)
        copied.append(
            {
                "image": record["image"],
                "panel": panel_target.relative_to(root_output).as_posix(),
                "metrics": metrics_target.relative_to(root_output).as_posix(),
                "hard_box_applied_count": int(record["hard_box_applied_count"]),
            }
        )
    return copied


def main() -> int:
    parser = argparse.ArgumentParser(description="Run detection -> legacy bbox rescue -> source LaMa on Sample images without OCR or translation.")
    parser.add_argument("--input-dir", default=str(ROOT / "Sample"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--glob", default="*")
    parser.add_argument("--review-count", type=int, default=3)
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir).resolve() if args.output_dir else ROOT / "banchmark_result_log" / "inpaint_debug" / f"{timestamp}_sample-inpaint-only"
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = _SettingsStub(use_gpu=args.use_gpu)
    detector = TextBlockDetector(settings)
    runtime = get_inpainter_runtime(settings)
    inpainter_cls = inpaint_map[runtime["key"]]
    device = resolve_device(args.use_gpu, backend=runtime["backend"])
    inpainter = inpainter_cls(
        device,
        backend=runtime["backend"],
        runtime_device=runtime.get("device", device),
        inpaint_size=runtime.get("inpaint_size"),
        precision=runtime.get("precision"),
    )

    image_paths = _iter_sample_images(input_dir, args.glob)
    records = [
        _process_image(
            image_path=image_path,
            output_root=output_dir,
            input_root=input_dir,
            detector=detector,
            inpainter=inpainter,
            settings=settings,
        )
        for image_path in image_paths
    ]
    review_samples = _copy_review_samples(output_dir, records, max(0, args.review_count))

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "input_dir": str(input_dir),
        "detector_key": settings.get_tool_selection("detector"),
        "inpainter": settings.get_tool_selection("inpainter"),
        "mask_inpaint_mode": settings.get_mask_inpaint_mode(),
        "use_gpu": bool(args.use_gpu),
        "glob": args.glob,
        "image_count": len(records),
        "hard_box_applied_image_count": sum(1 for record in records if int(record["hard_box_applied_count"]) > 0),
        "total_hard_box_applied_count": sum(int(record["hard_box_applied_count"]) for record in records),
        "total_legacy_base_mask_pixel_count": sum(int(record["legacy_base_mask_pixel_count"]) for record in records),
        "total_hard_box_rescue_mask_pixel_count": sum(int(record["hard_box_rescue_mask_pixel_count"]) for record in records),
        "total_final_mask_pixel_count": sum(int(record["final_mask_pixel_count"]) for record in records),
        "review_samples": review_samples,
        "images": [
            {
                "image": record["image"],
                "group": record["relative_parent"],
                "hard_box_applied_count": int(record["hard_box_applied_count"]),
                "legacy_base_mask_pixel_count": int(record["legacy_base_mask_pixel_count"]),
                "hard_box_rescue_mask_pixel_count": int(record["hard_box_rescue_mask_pixel_count"]),
                "final_mask_pixel_count": int(record["final_mask_pixel_count"]),
                "legacy_fill_ratio_values": list(record["legacy_fill_ratio_values"]),
                "rescue_fill_ratio_values": list(record["rescue_fill_ratio_values"]),
            }
            for record in records
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    _write_index(output_dir, records, summary)
    print(output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
