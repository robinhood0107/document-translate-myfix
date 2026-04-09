#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import imkit as imk
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modules.detection.processor import TextBlockDetector
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.utils.inpaint_cleanup import refine_bubble_residue_inpaint
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    ensure_three_channel,
    export_inpaint_debug_artifacts,
)
from modules.utils.pipeline_config import get_config, inpaint_map

DEBUG_EXPORT_SETTINGS = {
    "export_detector_overlay": True,
    "export_raw_mask": True,
    "export_mask_overlay": True,
    "export_cleanup_mask_delta": True,
    "export_debug_metadata": True,
}


@dataclass
class _UIStub:
    value_mappings: dict[str, str]

    def tr(self, text: str) -> str:
        return text


class _SettingsStub:
    def __init__(self, *, inpainter: str, use_gpu: bool) -> None:
        self._inpainter = inpainter
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
            return self._inpainter
        raise KeyError(tool_type)

    def is_gpu_enabled(self) -> bool:
        return self._use_gpu

    def get_hd_strategy_settings(self) -> dict:
        return {"strategy": "Resize", "resize_limit": 960}


def _build_cleanup_delta(raw_mask, final_mask):
    if final_mask is None:
        return None
    final_arr = np.asarray(final_mask)
    if final_arr.ndim == 3:
        final_arr = final_arr[:, :, 0]
    if raw_mask is None:
        raw_arr = np.zeros_like(final_arr, dtype=np.uint8)
    else:
        raw_arr = np.asarray(raw_mask)
        if raw_arr.ndim == 3:
            raw_arr = raw_arr[:, :, 0]
    return np.where((final_arr > 0) & (raw_arr <= 0), 255, 0).astype(np.uint8)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imk.write_image(str(path), ensure_three_channel(image))


def _iter_sample_images(corpus_dir: Path, pattern: str) -> list[Path]:
    return sorted(
        path
        for path in corpus_dir.iterdir()
        if path.is_file() and fnmatch.fnmatch(path.name, pattern)
    )


def _process_image(
    image_path: Path,
    corpus_output: Path,
    detector: TextBlockDetector,
    inpainter,
    settings: _SettingsStub,
):
    image = imk.read_image(str(image_path))
    if image is None:
        raise RuntimeError("failed to read image")
    image = ensure_three_channel(image)
    blocks = detector.detect(image) or []
    detector_key = settings.get_tool_selection("detector")
    detector_engine = detector.last_engine_name or ""
    detector_device = detector.last_device or resolve_device(settings.is_gpu_enabled(), backend="onnx")
    raw_mask = None
    final_mask = None
    cleanup_stats = {"applied": False, "component_count": 0, "block_count": 0}
    cleaned = image.copy()

    if blocks:
        mask = generate_mask(image, blocks)
        if mask is not None and np.any(mask):
            raw_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
            cleaned = inpainter(image, raw_mask, get_config(settings))
            cleaned = imk.convert_scale_abs(cleaned)
            cleaned, final_mask, cleanup_stats = refine_bubble_residue_inpaint(
                cleaned,
                raw_mask,
                blocks,
                inpainter,
                get_config(settings),
            )
        else:
            final_mask = mask

    cleanup_delta = _build_cleanup_delta(raw_mask, final_mask)
    metadata = build_inpaint_debug_metadata(
        image_path=str(image_path),
        run_type="sample_debug",
        detector_key=detector_key,
        detector_engine=detector_engine,
        device=detector_device,
        inpainter=settings.get_tool_selection("inpainter"),
        hd_strategy="Resize",
        blocks=blocks,
        raw_mask=raw_mask,
        cleanup_delta=cleanup_delta,
        cleanup_stats=cleanup_stats,
    )
    export_inpaint_debug_artifacts(
        export_root=str(corpus_output),
        archive_bname="",
        page_base_name=image_path.stem,
        image=image,
        blocks=blocks,
        export_settings=DEBUG_EXPORT_SETTINGS,
        raw_mask=raw_mask,
        cleanup_delta=cleanup_delta,
        metadata=metadata,
    )

    source_path = corpus_output / "source_images" / f"{image_path.stem}_source{image_path.suffix}"
    cleaned_path = corpus_output / "cleaned_images" / f"{image_path.stem}_cleaned{image_path.suffix}"
    _write_image(source_path, image)
    _write_image(cleaned_path, cleaned)
    return {
        "image": image_path.name,
        "source": source_path,
        "cleaned": cleaned_path,
        "detector_overlay": corpus_output / "detector_overlays" / f"{image_path.stem}_detector_overlay.png",
        "raw_mask": corpus_output / "raw_masks" / f"{image_path.stem}_raw_mask.png",
        "mask_overlay": corpus_output / "mask_overlays" / f"{image_path.stem}_mask_overlay.png",
        "cleanup_delta": corpus_output / "cleanup_mask_delta" / f"{image_path.stem}_cleanup_delta.png",
        "metadata": corpus_output / "debug_metadata" / f"{image_path.stem}_debug.json",
        "block_count": len(blocks),
        "cleanup_applied": bool(cleanup_stats.get("applied", False)),
        "cleanup_component_count": int(cleanup_stats.get("component_count", 0) or 0),
        "cleanup_block_count": int(cleanup_stats.get("block_count", 0) or 0),
    }


def _write_index(root_output: Path, records_by_corpus: dict[str, list[dict]], summary: dict) -> None:
    lines = [
        "# Inpaint Debug Export",
        "",
        f"Generated: `{summary['generated_at']}`",
        "",
        f"Detector: `{summary['detector_key']}`  ",
        f"Inpainter: `{summary['inpainter']}`  ",
        f"HD Strategy: `{summary['hd_strategy']}`  ",
        f"Use GPU: `{summary['use_gpu']}`",
        "",
        "## How To Review",
        "",
        "- Detector issue: compare `source`, `detector overlay`, and `metadata`.",
        "- Mask issue: compare `raw mask`, `mask overlay`, `cleanup delta`, and `metadata`.",
        "- Inpainter issue: compare `cleaned`, `raw mask`, `cleanup delta`, and `metadata`.",
        "",
    ]
    for corpus_name, records in records_by_corpus.items():
        lines.extend([f"## {corpus_name}", "", "| Image | Source | Detector | Raw Mask | Mask Overlay | Cleanup Delta | Cleaned | Metadata | Blocks | Cleanup |", "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |"])
        for record in records:
            def rel(path: Path) -> str:
                return path.relative_to(root_output).as_posix()
            cleanup_text = "yes" if record["cleanup_applied"] else "no"
            lines.append(
                f"| `{record['image']}` | [source]({rel(record['source'])}) | [detector]({rel(record['detector_overlay'])}) | [raw mask]({rel(record['raw_mask'])}) | [overlay]({rel(record['mask_overlay'])}) | [delta]({rel(record['cleanup_delta'])}) | [cleaned]({rel(record['cleaned'])}) | [metadata]({rel(record['metadata'])}) | {record['block_count']} | {cleanup_text} |"
            )
        lines.append("")
    (root_output / "index.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export inpaint debug artifacts for Sample corpora.")
    parser.add_argument("--glob", default="*", help="Glob pattern for sample filenames.")
    parser.add_argument("--inpainter", default="AOT", choices=sorted(inpaint_map.keys()))
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    settings = _SettingsStub(inpainter=args.inpainter, use_gpu=args.use_gpu)
    detector = TextBlockDetector(settings)
    inpainter_cls = inpaint_map[args.inpainter]
    inpainter = inpainter_cls(resolve_device(args.use_gpu, backend="onnx"), backend="onnx")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_output = ROOT / "banchmark_result_log" / "inpaint_debug" / f"{timestamp}_sample-debug-export"
    root_output.mkdir(parents=True, exist_ok=True)

    records_by_corpus: dict[str, list[dict]] = {}
    failures: list[dict] = []
    total_images = 0

    for corpus_name in ("japan", "China"):
        corpus_dir = ROOT / "Sample" / corpus_name
        corpus_output = root_output / corpus_name.lower()
        corpus_output.mkdir(parents=True, exist_ok=True)
        records: list[dict] = []
        for image_path in _iter_sample_images(corpus_dir, args.glob):
            total_images += 1
            try:
                record = _process_image(image_path, corpus_output, detector, inpainter, settings)
                records.append(record)
            except Exception as exc:
                failures.append(
                    {
                        "corpus": corpus_name,
                        "image": image_path.name,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
        records_by_corpus[corpus_name.lower()] = records

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "detector_key": settings.get_tool_selection("detector"),
        "inpainter": args.inpainter,
        "hd_strategy": "Resize",
        "use_gpu": bool(args.use_gpu),
        "glob": args.glob,
        "total_images": total_images,
        "success_count": sum(len(records) for records in records_by_corpus.values()),
        "failure_count": len(failures),
        "failures": failures,
        "corpora": {
            corpus: {
                "image_count": len(records),
                "cleanup_applied_count": sum(1 for record in records if record["cleanup_applied"]),
                "total_blocks": sum(record["block_count"] for record in records),
            }
            for corpus, records in records_by_corpus.items()
        },
    }
    metrics_dir = root_output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_index(root_output, records_by_corpus, summary)
    print(root_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
