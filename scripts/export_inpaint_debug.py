#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import json
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imkit as imk
import numpy as np

from modules.detection.processor import TextBlockDetector
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.source_parity_vendor import source_parity_blockwise_inpaint
from modules.utils.mask_inpaint_mode import normalize_mask_inpaint_mode
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    ensure_three_channel,
    export_inpaint_debug_artifacts,
)
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.inpainting_runtime import inpainter_default_settings, normalize_inpainter_key
from modules.utils.mask_inpaint_mode import (
    DEFAULT_MASK_INPAINT_MODE,
    MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE,
    MASK_INPAINT_MODE_SOURCE_PARITY,
)

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
    def __init__(
        self,
        *,
        inpainter: str,
        use_gpu: bool,
        mask_refiner: str = "legacy_bbox",
        keep_existing_lines: bool = False,
        ctd_detect_size: int = 1280,
        ctd_det_rearrange_max_batches: int = 4,
        ctd_font_size_multiplier: float = 1.0,
        ctd_font_size_max: int = -1,
        ctd_font_size_min: int = -1,
        ctd_mask_dilate_size: int = 2,
        mask_inpaint_mode: str = DEFAULT_MASK_INPAINT_MODE,
    ) -> None:
        self._inpainter = inpainter
        self._use_gpu = use_gpu
        self._mask_refiner = mask_refiner
        self._keep_existing_lines = keep_existing_lines
        self._ctd_detect_size = ctd_detect_size
        self._ctd_det_rearrange_max_batches = ctd_det_rearrange_max_batches
        self._ctd_font_size_multiplier = ctd_font_size_multiplier
        self._ctd_font_size_max = ctd_font_size_max
        self._ctd_font_size_min = ctd_font_size_min
        self._ctd_mask_dilate_size = ctd_mask_dilate_size
        self._mask_inpaint_mode = mask_inpaint_mode
        self.ui = _UIStub(
            value_mappings={
                "Resize": "Resize",
                "Original": "Original",
                "Crop": "Crop",
            }
        )

    def get_tool_selection(self, tool_type: str) -> str:
        if tool_type == "detector":
            if self._mask_inpaint_mode == MASK_INPAINT_MODE_SOURCE_PARITY:
                return "Source Parity CTD"
            return "RT-DETR-v2"
        if tool_type == "inpainter":
            if self._mask_inpaint_mode in {MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE, MASK_INPAINT_MODE_SOURCE_PARITY}:
                return "lama_large_512px"
            return self._inpainter
        raise KeyError(tool_type)

    def get_mask_inpaint_mode(self) -> str:
        return self._mask_inpaint_mode

    def is_gpu_enabled(self) -> bool:
        return self._use_gpu

    def get_hd_strategy_settings(self) -> dict:
        return {"strategy": "Resize", "resize_limit": 960}

    def get_mask_refiner_settings(self) -> dict:
        mask_refiner = self._mask_refiner
        if self._mask_inpaint_mode == MASK_INPAINT_MODE_SOURCE_PARITY:
            mask_refiner = "ctd"
        elif self._mask_inpaint_mode == MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE:
            mask_refiner = "legacy_bbox"
        return {
            "mask_refiner": mask_refiner,
            "ctd_detect_size": self._ctd_detect_size,
            "ctd_det_rearrange_max_batches": self._ctd_det_rearrange_max_batches,
            "ctd_device": "cuda" if self._use_gpu else "cpu",
            "ctd_font_size_multiplier": self._ctd_font_size_multiplier,
            "ctd_font_size_max": self._ctd_font_size_max,
            "ctd_font_size_min": self._ctd_font_size_min,
            "ctd_mask_dilate_size": self._ctd_mask_dilate_size,
            "keep_existing_lines": False if self._mask_inpaint_mode in {MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE, MASK_INPAINT_MODE_SOURCE_PARITY} else self._keep_existing_lines,
            "mask_inpaint_mode": self._mask_inpaint_mode,
        }

    def get_inpainter_runtime_settings(self, inpainter_key: str | None = None) -> dict:
        normalized = normalize_inpainter_key(inpainter_key or self._inpainter)
        return inpainter_default_settings(normalized)


def _build_cleanup_delta(base_mask, final_mask):
    if final_mask is None:
        return None
    final_arr = np.asarray(final_mask)
    if final_arr.ndim == 3:
        final_arr = final_arr[:, :, 0]
    if base_mask is None:
        base_arr = np.zeros_like(final_arr, dtype=np.uint8)
    else:
        base_arr = np.asarray(base_mask)
        if base_arr.ndim == 3:
            base_arr = base_arr[:, :, 0]
    return np.where((final_arr > 0) & (base_arr <= 0), 255, 0).astype(np.uint8)


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
    detector_key = getattr(detector, "detector", None) or settings.get_tool_selection("detector")
    detector_engine = detector.last_engine_name or ""
    detector_device = detector.last_device or resolve_device(settings.is_gpu_enabled(), backend="onnx")
    raw_mask = None
    final_mask = None
    cleanup_stats = {"applied": False, "component_count": 0, "block_count": 0}
    cleaned = image.copy()
    mask_details = {}

    if blocks:
        mask_details = generate_mask(
            image,
            blocks,
            settings=settings.get_mask_refiner_settings(),
            return_details=True,
            precomputed_mask_details=getattr(detector, "last_mask_details", None),
        )
        mask = mask_details["final_mask"]
        if mask is not None and np.any(mask):
            raw_mask = mask_details["raw_mask"]
            if normalize_mask_inpaint_mode(settings.get_mask_inpaint_mode()) in {MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE, MASK_INPAINT_MODE_SOURCE_PARITY}:
                cleaned = source_parity_blockwise_inpaint(image, mask, blocks, inpainter, get_config(settings), check_need_inpaint=True)
            else:
                cleaned = inpainter(image, mask, get_config(settings))
            cleaned = imk.convert_scale_abs(cleaned)
            final_mask = mask
        else:
            final_mask = mask

    cleanup_delta = _build_cleanup_delta(mask_details.get("final_mask", final_mask), final_mask)
    runtime = get_inpainter_runtime(settings)
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
        final_mask=final_mask,
        final_mask_pre_expand=mask_details.get("final_mask_pre_expand"),
        final_mask_post_expand=mask_details.get("final_mask_post_expand"),
        residue_mask=cleanup_stats.get("residue_mask") if cleanup_stats else None,
        cleanup_delta=cleanup_delta,
        cleanup_stats=cleanup_stats,
        mask_refiner=str(mask_details.get("mask_refiner", "legacy_bbox") or "legacy_bbox"),
        protect_mask_applied=bool(mask_details.get("keep_existing_lines", False)),
        protect_mask=mask_details.get("protect_mask"),
        refiner_backend=str(mask_details.get("refiner_backend", "legacy") or "legacy"),
        refiner_device=str(mask_details.get("refiner_device", "cpu") or "cpu"),
        inpainter_backend=str(runtime.get("backend", "unknown") or "unknown"),
    )
    export_inpaint_debug_artifacts(
        export_root=str(corpus_output),
        archive_bname="",
        page_base_name=image_path.stem,
        image=image,
        blocks=blocks,
        export_settings=DEBUG_EXPORT_SETTINGS,
        raw_mask=raw_mask,
        mask_overlay_mask=mask_details.get("final_mask_post_expand", mask_details.get("final_mask", final_mask)),
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
    parser.add_argument("--inpainter", default="AOT", choices=["AOT", "lama_large_512px", "lama_mpe"])
    parser.add_argument("--mask-inpaint-mode", default=DEFAULT_MASK_INPAINT_MODE, choices=[MASK_INPAINT_MODE_RTDETR_LEGACY_BBOX_SOURCE, MASK_INPAINT_MODE_SOURCE_PARITY])
    parser.add_argument("--use-gpu", action="store_true")
    args = parser.parse_args()

    settings = _SettingsStub(inpainter=args.inpainter, use_gpu=args.use_gpu, mask_inpaint_mode=args.mask_inpaint_mode)
    detector = TextBlockDetector(settings)
    runtime = get_inpainter_runtime(settings, args.inpainter)
    inpainter_cls = inpaint_map[runtime["key"]]
    device = resolve_device(args.use_gpu, backend=runtime["backend"])
    inpainter = inpainter_cls(
        device,
        backend=runtime["backend"],
        runtime_device=runtime.get("device", device),
        inpaint_size=runtime.get("inpaint_size"),
        precision=runtime.get("precision"),
    )

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
        "inpainter": settings.get_tool_selection("inpainter"),
        "mask_inpaint_mode": args.mask_inpaint_mode,
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
