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
from PIL import Image, ImageDraw

from modules.detection.processor import TextBlockDetector
from modules.inpainting.runtime_contract import inspect_learned_inpainter_runtime
from modules.inpainting.source_lama_blockwise import source_lama_blockwise_inpaint
from modules.rendering.render import get_best_render_area
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.utils.inpaint_composite import composite_with_edit_mask, normalize_edit_mask
from modules.utils.inpaint_cleanup import (
    apply_duplicate_bubble_inner_fill,
    refine_bubble_residue_inpaint,
)
from modules.utils.inpaint_debug import (
    build_inpaint_debug_metadata,
    ensure_three_channel,
    export_inpaint_debug_artifacts,
)
from modules.utils.pipeline_config import get_config, get_inpainter_runtime, inpaint_map
from modules.utils.inpainting_runtime import (
    inpainter_default_settings,
    is_lama_family_inpainter,
    normalize_inpainter_key,
)
from scripts.validation_artifact_harness import select_managed_output_directory

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
        mask_refiner: str = "ctd",
        keep_existing_lines: bool = True,
        ctd_detect_size: int = 1280,
        ctd_det_rearrange_max_batches: int = 4,
        ctd_font_size_multiplier: float = 1.0,
        ctd_font_size_max: int = -1,
        ctd_font_size_min: int = -1,
        ctd_mask_dilate_size: int = 2,
        hd_strategy: str = "Original",
        developer_performance_mode: bool = False,
        resize_limit: int = 960,
        crop_margin: int = 512,
        crop_trigger_size: int = 512,
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
        self._hd_strategy = str(hd_strategy or "Original")
        self._developer_performance_mode = bool(developer_performance_mode)
        self._resize_limit = int(resize_limit)
        self._crop_margin = int(crop_margin)
        self._crop_trigger_size = int(crop_trigger_size)
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
        return {
            "strategy": self._hd_strategy,
            "resize_limit": self._resize_limit,
            "crop_margin": self._crop_margin,
            "crop_trigger_size": self._crop_trigger_size,
            "developer_performance_mode": self._developer_performance_mode,
        }

    def get_mask_refiner_settings(self) -> dict:
        return {
            "mask_refiner": self._mask_refiner,
            "ctd_detect_size": self._ctd_detect_size,
            "ctd_det_rearrange_max_batches": self._ctd_det_rearrange_max_batches,
            "ctd_device": "cuda" if self._use_gpu else "cpu",
            "ctd_font_size_multiplier": self._ctd_font_size_multiplier,
            "ctd_font_size_max": self._ctd_font_size_max,
            "ctd_font_size_min": self._ctd_font_size_min,
            "ctd_mask_dilate_size": self._ctd_mask_dilate_size,
            "keep_existing_lines": self._keep_existing_lines,
        }

    def get_inpainter_runtime_settings(self, inpainter_key: str | None = None) -> dict:
        normalized = normalize_inpainter_key(inpainter_key or self._inpainter)
        return inpainter_default_settings(normalized)


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


def _build_changed_pixel_stats(source_image, cleaned_image, final_mask) -> dict[str, int]:
    if source_image is None or cleaned_image is None or final_mask is None:
        return {
            "changed_pixel_count": 0,
            "changed_inside_final_mask_pixel_count": 0,
            "changed_outside_final_mask_pixel_count": 0,
            "changed_pixel_count_exact": 0,
            "changed_inside_final_mask_pixel_count_exact": 0,
            "changed_outside_final_mask_pixel_count_exact": 0,
        }
    source_arr = np.asarray(source_image)
    cleaned_arr = np.asarray(cleaned_image)
    if source_arr.shape != cleaned_arr.shape:
        return {
            "changed_pixel_count": 0,
            "changed_inside_final_mask_pixel_count": 0,
            "changed_outside_final_mask_pixel_count": 0,
            "changed_pixel_count_exact": 0,
            "changed_inside_final_mask_pixel_count_exact": 0,
            "changed_outside_final_mask_pixel_count_exact": 0,
        }
    mask_arr = np.asarray(final_mask)
    if mask_arr.ndim == 3:
        mask_arr = mask_arr[:, :, 0]
    changed = np.any(np.abs(cleaned_arr.astype(np.int16) - source_arr.astype(np.int16)) > 2, axis=2)
    changed_exact = np.any(cleaned_arr != source_arr, axis=2)
    return {
        "changed_pixel_count": int(np.count_nonzero(changed)),
        "changed_inside_final_mask_pixel_count": int(np.count_nonzero(changed & (mask_arr > 0))),
        "changed_outside_final_mask_pixel_count": int(np.count_nonzero(changed & (mask_arr <= 0))),
        "changed_pixel_count_exact": int(np.count_nonzero(changed_exact)),
        "changed_inside_final_mask_pixel_count_exact": int(
            np.count_nonzero(changed_exact & (mask_arr > 0))
        ),
        "changed_outside_final_mask_pixel_count_exact": int(
            np.count_nonzero(changed_exact & (mask_arr <= 0))
        ),
    }


def _build_text_anchor_mask(image_shape, blocks) -> np.ndarray:
    anchor_mask = np.zeros(image_shape[:2], dtype=np.uint8)
    for block in list(blocks or []):
        anchor = getattr(block, "_mask_anchor_xyxy", None)
        if anchor is None:
            continue
        try:
            x1, y1, x2, y2 = [int(v) for v in list(anchor)[:4]]
        except (TypeError, ValueError):
            continue
        x1 = max(0, min(image_shape[1], x1))
        x2 = max(0, min(image_shape[1], x2))
        y1 = max(0, min(image_shape[0], y1))
        y2 = max(0, min(image_shape[0], y2))
        if x2 > x1 and y2 > y1:
            anchor_mask[y1:y2, x1:x2] = 255
    return anchor_mask


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
    *,
    auto_max_font_profile: str = "current",
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
    mask_details = {}
    inpaint_diagnostics: list[dict] = []

    if blocks:
        get_best_render_area(
            blocks,
            image,
            auto_max_font_profile=auto_max_font_profile,
        )
        mask_details = generate_mask(image, blocks, settings=settings.get_mask_refiner_settings(), return_details=True)
        mask = mask_details["final_mask"]
        if mask is not None and np.any(mask):
            raw_mask = mask_details["raw_mask"]
            config = get_config(settings)
            cleaned, inpaint_edit_mask, inpaint_diagnostics = source_lama_blockwise_inpaint(
                image,
                mask,
                blocks,
                inpainter,
                config,
                check_need_inpaint=True,
                return_edit_mask=True,
                return_diagnostics=True,
                protected_corner_mask=mask_details.get("protected_corner_mask"),
            )
            mask = np.where((mask > 0) | (inpaint_edit_mask > 0), 255, 0).astype(np.uint8)
            cleaned, final_mask, cleanup_stats = refine_bubble_residue_inpaint(
                cleaned,
                mask,
                blocks,
                inpainter,
                config,
                protected_corner_mask=mask_details.get("protected_corner_mask"),
            )
            cleaned, final_mask, cleanup_stats = apply_duplicate_bubble_inner_fill(
                cleaned,
                final_mask,
                mask_details,
                cleanup_stats,
            )
            protected_corner_mask = normalize_edit_mask(
                mask_details.get("protected_corner_mask"),
                image.shape,
            )
            final_mask = np.where(
                (normalize_edit_mask(final_mask, image.shape) > 0)
                & (protected_corner_mask <= 0),
                255,
                0,
            ).astype(np.uint8)
            cleaned = composite_with_edit_mask(image, cleaned, final_mask)
        else:
            final_mask = mask

    cleanup_delta = _build_cleanup_delta(raw_mask, final_mask)
    runtime = get_inpainter_runtime(settings)
    config = get_config(settings)
    metadata = build_inpaint_debug_metadata(
        image_path=str(image_path),
        run_type="sample_debug",
        detector_key=detector_key,
        detector_engine=detector_engine,
        device=detector_device,
        inpainter=settings.get_tool_selection("inpainter"),
        hd_strategy=str(config.hd_strategy),
        blocks=blocks,
        raw_mask=raw_mask,
        final_mask=final_mask,
        final_mask_pre_expand=mask_details.get("final_mask_pre_expand"),
        final_mask_post_expand=mask_details.get("final_mask_post_expand"),
        residue_mask=cleanup_stats.get("residue_mask") if isinstance(cleanup_stats, dict) else None,
        cleanup_delta=cleanup_delta,
        cleanup_stats=cleanup_stats,
        mask_refiner=str(mask_details.get("mask_refiner", "legacy_bbox") or "legacy_bbox"),
        protect_mask_applied=bool(mask_details.get("keep_existing_lines", False)),
        protect_mask=mask_details.get("protect_mask"),
        refiner_backend=str(mask_details.get("refiner_backend", "legacy") or "legacy"),
        refiner_device=str(mask_details.get("refiner_device", "cpu") or "cpu"),
        inpainter_backend=str(runtime.get("backend", "unknown") or "unknown"),
        legacy_base_mask=mask_details.get("legacy_base_mask"),
        hard_box_rescue_mask=mask_details.get("hard_box_rescue_mask"),
        hard_box_applied_count=int(mask_details.get("hard_box_applied_count", 0) or 0),
        hard_box_reason_totals=dict(mask_details.get("hard_box_reason_totals", {}) or {}),
        mask_quality_policy=str(mask_details.get("mask_quality_policy", "") or ""),
        mask_policy_bubble_clamp_applied_count=int(
            mask_details.get("mask_policy_bubble_clamp_applied_count", 0) or 0
        ),
        mask_policy_bubble_silhouette_applied_count=int(
            mask_details.get("mask_policy_bubble_silhouette_applied_count", 0) or 0
        ),
        mask_policy_bubble_silhouette_fallback_count=int(
            mask_details.get("mask_policy_bubble_silhouette_fallback_count", 0) or 0
        ),
        mask_policy_text_free_glyph_applied_count=int(
            mask_details.get("mask_policy_text_free_glyph_applied_count", 0) or 0
        ),
        mask_policy_removed_pixel_count=int(mask_details.get("mask_policy_removed_pixel_count", 0) or 0),
        mask_policy_outside_bubble_removed_pixel_count=int(
            mask_details.get("mask_policy_outside_bubble_removed_pixel_count", 0) or 0
        ),
        ctd_legacy_rectangle_rescue_disabled=bool(
            mask_details.get("ctd_legacy_rectangle_rescue_disabled", False)
        ),
        text_free_image_glyph_rescue_count=int(
            mask_details.get("text_free_image_glyph_rescue_count", 0) or 0
        ),
        text_free_image_glyph_rescue_mask_pixel_count=int(
            mask_details.get("text_free_image_glyph_rescue_mask_pixel_count", 0) or 0
        ),
        mask_policy_version=str(mask_details.get("mask_policy_version", "") or ""),
        mask_candidate_source=str(mask_details.get("mask_candidate_source", "") or ""),
        mask_decision=str(mask_details.get("mask_decision", "") or ""),
        mask_reject_reason=str(mask_details.get("mask_reject_reason", "") or ""),
        mask_score_outside_change=float(mask_details.get("mask_score_outside_change", 0.0) or 0.0),
        mask_score_outline_damage=float(mask_details.get("mask_score_outline_damage", 0.0) or 0.0),
        mask_score_residue=float(mask_details.get("mask_score_residue", 0.0) or 0.0),
        mask_score_color_delta=float(mask_details.get("mask_score_color_delta", 0.0) or 0.0),
        ui_panel_mode=str(mask_details.get("ui_panel_mode", "") or ""),
        ui_panel_preview_path=str(mask_details.get("ui_panel_preview_path", "") or ""),
    )
    changed_stats = _build_changed_pixel_stats(image, cleaned, final_mask)
    metadata.update(changed_stats)
    normalized_final_mask = (
        np.where(np.asarray(final_mask) > 0, 255, 0).astype(np.uint8)
        if final_mask is not None
        else np.zeros(image.shape[:2], dtype=np.uint8)
    )
    bubble_cap_mask = np.where(
        np.asarray(
            mask_details.get("bubble_interior_cap_mask", np.zeros(image.shape[:2], dtype=np.uint8))
        )
        > 0,
        255,
        0,
    ).astype(np.uint8)
    protected_corner_mask = np.where(
        np.asarray(
            mask_details.get("protected_corner_mask", np.zeros(image.shape[:2], dtype=np.uint8))
        )
        > 0,
        255,
        0,
    ).astype(np.uint8)
    text_anchor_mask = _build_text_anchor_mask(image.shape, blocks)
    changed_exact = np.any(np.asarray(cleaned) != np.asarray(image), axis=2)
    metadata.update(
        {
            "bubble_block_count": sum(
                1
                for block in blocks
                if str(getattr(block, "text_class", "") or "") == "text_bubble"
                and getattr(block, "bubble_xyxy", None) is not None
            ),
            "protected_corner_mask_pixel_count": int(np.count_nonzero(protected_corner_mask)),
            "protected_corner_final_mask_pixel_count": int(
                np.count_nonzero((protected_corner_mask > 0) & (normalized_final_mask > 0))
            ),
            "protected_corner_changed_pixel_count": int(
                np.count_nonzero((protected_corner_mask > 0) & changed_exact)
            ),
            "text_anchor_final_mask_pixel_count": int(
                np.count_nonzero((text_anchor_mask > 0) & (normalized_final_mask > 0))
            ),
            "text_anchor_changed_pixel_count": int(
                np.count_nonzero((text_anchor_mask > 0) & changed_exact)
            ),
        }
    )
    metadata["inpaint_runtime_diagnostics"] = list(inpaint_diagnostics)
    metadata["inpaint_runtime_inference_call_count"] = len(inpaint_diagnostics)
    metadata["inpaint_runtime_cpu_fallback_count"] = sum(
        1 for item in inpaint_diagnostics if bool(item.get("cpu_fallback_used", False))
    )
    export_inpaint_debug_artifacts(
        export_root=str(corpus_output),
        archive_bname="",
        page_base_name=image_path.stem,
        image=image,
        blocks=blocks,
        export_settings=DEBUG_EXPORT_SETTINGS,
        raw_mask=raw_mask,
        mask_overlay_mask=final_mask,
        cleanup_delta=cleanup_delta,
        metadata=metadata,
    )

    source_path = corpus_output / "source_images" / f"{image_path.stem}_source.png"
    cleaned_path = corpus_output / "cleaned_images" / f"{image_path.stem}_cleaned.png"
    final_mask_path = corpus_output / "final_masks" / f"{image_path.stem}_final_mask.png"
    bubble_cap_path = corpus_output / "bubble_interior_caps" / f"{image_path.stem}_bubble_cap.png"
    protected_corner_path = corpus_output / "protected_corner_masks" / f"{image_path.stem}_protected_corners.png"
    _write_image(source_path, image)
    _write_image(cleaned_path, cleaned)
    _write_image(final_mask_path, normalized_final_mask)
    _write_image(bubble_cap_path, bubble_cap_mask)
    _write_image(protected_corner_path, protected_corner_mask)
    return {
        "image": image_path.name,
        "source": source_path,
        "cleaned": cleaned_path,
        "final_mask": final_mask_path,
        "bubble_interior_cap": bubble_cap_path,
        "protected_corner_mask": protected_corner_path,
        "detector_overlay": corpus_output / "detector_overlays" / f"{image_path.stem}_detector_overlay.png",
        "raw_mask": corpus_output / "raw_masks" / f"{image_path.stem}_raw_mask.png",
        "mask_overlay": corpus_output / "mask_overlays" / f"{image_path.stem}_mask_overlay.png",
        "cleanup_delta": corpus_output / "cleanup_mask_delta" / f"{image_path.stem}_cleanup_delta.png",
        "metadata": corpus_output / "debug_metadata" / f"{image_path.stem}_debug.json",
        "block_count": len(blocks),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)) if final_mask is not None else 0,
        "hd_strategy": str(config.hd_strategy),
        "auto_max_font_profile": str(auto_max_font_profile),
        "refiner_backend": str(mask_details.get("refiner_backend", "") or ""),
        "refiner_device": str(mask_details.get("refiner_device", "") or ""),
        "inpaint_runtime_diagnostics": list(inpaint_diagnostics),
        "inpaint_runtime_inference_call_count": len(inpaint_diagnostics),
        "inpaint_runtime_cpu_fallback_count": sum(
            1 for item in inpaint_diagnostics if bool(item.get("cpu_fallback_used", False))
        ),
        "bubble_block_count": int(metadata["bubble_block_count"]),
        "bubble_silhouette_applied_count": int(
            mask_details.get("mask_policy_bubble_silhouette_applied_count", 0) or 0
        ),
        "bubble_silhouette_fallback_count": int(
            mask_details.get("mask_policy_bubble_silhouette_fallback_count", 0) or 0
        ),
        "protected_corner_mask_pixel_count": int(metadata["protected_corner_mask_pixel_count"]),
        "protected_corner_final_mask_pixel_count": int(
            metadata["protected_corner_final_mask_pixel_count"]
        ),
        "protected_corner_changed_pixel_count": int(
            metadata["protected_corner_changed_pixel_count"]
        ),
        "text_anchor_final_mask_pixel_count": int(metadata["text_anchor_final_mask_pixel_count"]),
        "text_anchor_changed_pixel_count": int(metadata["text_anchor_changed_pixel_count"]),
        **changed_stats,
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


def _write_contact_sheet(
    root_output: Path,
    records_by_corpus: dict[str, list[dict]],
    *,
    record_key: str,
    filename: str,
) -> None:
    entries = [
        (corpus, record)
        for corpus, records in records_by_corpus.items()
        for record in records
        if record.get(record_key)
    ]
    if not entries:
        return

    thumb_width, thumb_height = 300, 420
    label_height = 32
    columns = min(4, len(entries))
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new(
        "RGB",
        (columns * thumb_width, rows * (thumb_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    for index, (corpus, record) in enumerate(entries):
        path = Path(record[record_key])
        with Image.open(path) as source:
            preview = source.convert("RGB")
            preview.thumbnail((thumb_width, thumb_height))
        column = index % columns
        row = index // columns
        x = column * thumb_width + (thumb_width - preview.width) // 2
        y = row * (thumb_height + label_height)
        sheet.paste(preview, (x, y))
        draw.text(
            (column * thumb_width + 4, y + thumb_height + 5),
            f"{corpus}/{record['image']}",
            fill="black",
        )
    sheet.save(root_output / filename, format="PNG")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export inpaint debug artifacts for Sample corpora.")
    parser.add_argument("--glob", default="*", help="Glob pattern for sample filenames.")
    parser.add_argument(
        "--corpus",
        choices=("all", "japan", "china"),
        default="all",
        help="Sample corpus to process. Repeatable --input paths replace corpus selection.",
    )
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        default=[],
        help="Private image path to process. Repeat for multiple images.",
    )
    parser.add_argument("--inpainter", default="AOT", choices=["AOT", "lama_large_512px", "lama_mpe"])
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument(
        "--require-cuda-lama",
        action="store_true",
        help="Fail unless Original-mode LaMa and both mask/refiner runtimes are CUDA-only.",
    )
    parser.add_argument(
        "--require-rounded-bubble-gate",
        action="store_true",
        help=(
            "Fail unless every selected rounded-bubble regression image passes the "
            "protected-corner gates; intended for curated private target pages."
        ),
    )
    parser.add_argument(
        "--require-image-count",
        type=int,
        default=None,
        help="Fail unless exactly this many images are selected and completed.",
    )
    parser.add_argument(
        "--auto-max-font-profile",
        choices=("current", "strong"),
        default="current",
        help="Render-area profile to apply before generating the inpaint mask.",
    )
    parser.add_argument(
        "--hd-strategy",
        choices=("Original", "Resize", "Crop"),
        default="Original",
        help="HD strategy; Resize/Crop take effect only with developer performance mode.",
    )
    parser.add_argument(
        "--developer-performance-mode",
        action="store_true",
        help="Explicitly enable the product's Resize/Crop performance strategies.",
    )
    parser.add_argument("--resize-limit", type=int, default=960)
    parser.add_argument("--crop-margin", type=int, default=512)
    parser.add_argument("--crop-trigger-size", type=int, default=512)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory. Without it, a classified private validation "
            "artifact run is created automatically."
        ),
    )
    return parser


def _required_gate_failures(
    summary: dict,
    records_by_corpus: dict[str, list[dict]],
    *,
    require_cuda_lama: bool,
    require_rounded_bubble_gate: bool,
    required_image_count: int | None = None,
) -> list[str]:
    failures: list[str] = []
    image_count = int(summary.get("image_count", 0) or 0)
    success_count = int(summary.get("success_count", 0) or 0)
    if image_count <= 0:
        failures.append("no_input_images")
    if success_count != image_count:
        failures.append("success_count_mismatch")
    if required_image_count is not None and image_count != int(required_image_count):
        failures.append(
            f"image_count_mismatch:{image_count}!={int(required_image_count)}"
        )
    if require_cuda_lama:
        runtime = dict(summary.get("inpainter_runtime") or {})
        if not str(summary.get("inpainter", "")).startswith("lama"):
            failures.append("inpainter_not_lama")
        if not bool(summary.get("use_gpu", False)):
            failures.append("gpu_not_requested")
        if str(summary.get("hd_strategy", "")) != "Original":
            failures.append("hd_strategy_not_original")
        if not str(runtime.get("actual_device", "")).lower().startswith("cuda"):
            failures.append("inpainter_not_cuda")
        if not bool(runtime.get("device_verified_from_model", False)):
            failures.append("inpainter_cuda_not_model_verified")
        if bool(runtime.get("cpu_fallback_used", False)) or int(
            summary.get("cpu_fallback_count", 0) or 0
        ) > 0:
            failures.append("cpu_fallback_detected")
        if int(summary.get("non_cuda_refiner_count", 0) or 0) > 0:
            failures.append("non_cuda_refiner_detected")
        if int(summary.get("zero_block_count", 0) or 0) > 0:
            failures.append("empty_detection_detected")
        if int(summary.get("empty_final_mask_count", 0) or 0) > 0:
            failures.append("empty_final_mask_detected")
        if int(summary.get("runtime_inference_call_count", 0) or 0) <= 0:
            failures.append("no_inpaint_inference")

    if require_rounded_bubble_gate:
        for corpus_name, records in records_by_corpus.items():
            for record in records:
                image_name = (
                    f"{corpus_name}/{str(record.get('image', 'image'))}"
                )
                if int(record.get("bubble_block_count", 0) or 0) <= 0:
                    failures.append(f"{image_name}:missing_bubble")
                if int(record.get("bubble_silhouette_fallback_count", 0) or 0) > 0:
                    failures.append(f"{image_name}:silhouette_fallback")
                if int(record.get("protected_corner_mask_pixel_count", 0) or 0) <= 0:
                    failures.append(f"{image_name}:empty_protected_corner_mask")
                if int(record.get("protected_corner_final_mask_pixel_count", 0) or 0) != 0:
                    failures.append(f"{image_name}:protected_corner_mask_overlap")
                if int(record.get("protected_corner_changed_pixel_count", 0) or 0) != 0:
                    failures.append(f"{image_name}:protected_corner_changed")
                if int(record.get("text_anchor_final_mask_pixel_count", 0) or 0) <= 0:
                    failures.append(f"{image_name}:empty_text_anchor_mask")
                if int(record.get("text_anchor_changed_pixel_count", 0) or 0) <= 0:
                    failures.append(f"{image_name}:unchanged_text_anchor")
                if int(record.get("changed_outside_final_mask_pixel_count_exact", 0) or 0) != 0:
                    failures.append(f"{image_name}:changed_outside_final_mask")
    return failures


def main() -> int:
    args = _build_argument_parser().parse_args()

    root_output, artifact_run = select_managed_output_directory(
        family="inpaint-debug-export",
        category="40-inpaint-mask-render",
        explicit_output_directory=args.output_dir,
    )
    root_output.mkdir(parents=True, exist_ok=True)
    try:
        settings = _SettingsStub(
            inpainter=args.inpainter,
            use_gpu=args.use_gpu,
            hd_strategy=args.hd_strategy,
            developer_performance_mode=args.developer_performance_mode,
            resize_limit=args.resize_limit,
            crop_margin=args.crop_margin,
            crop_trigger_size=args.crop_trigger_size,
        )
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
        runtime_report = {
            "inpainter_key": str(runtime["key"]),
            "backend": str(runtime.get("backend", "") or ""),
            "requested_device": str(runtime.get("device", device) or device),
            "actual_device": str(getattr(inpainter, "runtime_device", getattr(inpainter, "device", "")) or ""),
            "actual_precision": str(getattr(inpainter, "precision", "") or ""),
            "cpu_fallback_used": False,
        }
        if is_lama_family_inpainter(runtime["key"]):
            runtime_report.update(
                inspect_learned_inpainter_runtime(
                    inpainter,
                    inpainter_key=str(runtime["key"]),
                    requested_device=str(runtime.get("device", device) or device),
                    requested_precision=str(runtime.get("precision", "bf16") or "bf16"),
                )
            )

        records_by_corpus: dict[str, list[dict]] = {}
        failures: list[dict] = []
        total_images = 0

        if args.input:
            corpus_inputs = {"private": [path.expanduser() for path in args.input]}
        else:
            selected_corpora = (
                ("japan", "China")
                if args.corpus == "all"
                else (("japan",) if args.corpus == "japan" else ("China",))
            )
            corpus_inputs = {
                corpus_name: _iter_sample_images(ROOT / "Sample" / corpus_name, args.glob)
                for corpus_name in selected_corpora
            }

        for corpus_name, image_paths in corpus_inputs.items():
            corpus_output = root_output / corpus_name.lower()
            corpus_output.mkdir(parents=True, exist_ok=True)
            records: list[dict] = []
            for image_path in image_paths:
                total_images += 1
                try:
                    if not image_path.is_file():
                        raise FileNotFoundError(str(image_path))
                    record = _process_image(
                        image_path,
                        corpus_output,
                        detector,
                        inpainter,
                        settings,
                        auto_max_font_profile=args.auto_max_font_profile,
                    )
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
            "hd_strategy": str(get_config(settings).hd_strategy),
            "auto_max_font_profile": args.auto_max_font_profile,
            "use_gpu": bool(args.use_gpu),
            "corpus": args.corpus,
            "glob": args.glob,
            "image_count": total_images,
            "total_images": total_images,
            "success_count": sum(len(records) for records in records_by_corpus.values()),
            "failure_count": len(failures),
            "failures": failures,
            "inpainter_runtime": runtime_report,
            "runtime_inference_call_count": sum(
                record["inpaint_runtime_inference_call_count"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "cpu_fallback_count": sum(
                record["inpaint_runtime_cpu_fallback_count"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "non_cuda_refiner_count": sum(
                1
                for records in records_by_corpus.values()
                for record in records
                if record["block_count"] > 0
                and not record["refiner_device"].lower().startswith("cuda")
            ),
            "zero_block_count": sum(
                1
                for records in records_by_corpus.values()
                for record in records
                if record["block_count"] <= 0
            ),
            "empty_final_mask_count": sum(
                1
                for records in records_by_corpus.values()
                for record in records
                if record["final_mask_pixel_count"] <= 0
            ),
            "bubble_silhouette_fallback_count": sum(
                record["bubble_silhouette_fallback_count"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "protected_corner_final_mask_pixel_count": sum(
                record["protected_corner_final_mask_pixel_count"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "protected_corner_changed_pixel_count": sum(
                record["protected_corner_changed_pixel_count"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "changed_outside_final_mask_pixel_count_exact": sum(
                record["changed_outside_final_mask_pixel_count_exact"]
                for records in records_by_corpus.values()
                for record in records
            ),
            "empty_text_anchor_edit_count": sum(
                1
                for records in records_by_corpus.values()
                for record in records
                if record["block_count"] > 0
                and (
                    record["text_anchor_final_mask_pixel_count"] <= 0
                    or record["text_anchor_changed_pixel_count"] <= 0
                )
            ),
            "corpora": {
                corpus: {
                    "image_count": len(records),
                    "cleanup_applied_count": sum(1 for record in records if record["cleanup_applied"]),
                    "total_blocks": sum(record["block_count"] for record in records),
                }
                for corpus, records in records_by_corpus.items()
            },
        }
        gate_failures = _required_gate_failures(
            summary,
            records_by_corpus,
            require_cuda_lama=bool(args.require_cuda_lama),
            require_rounded_bubble_gate=bool(args.require_rounded_bubble_gate),
            required_image_count=args.require_image_count,
        )
        summary["required_gate_failure_count"] = len(gate_failures)
        summary["required_gate_failures"] = gate_failures
        metrics_dir = root_output / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_index(root_output, records_by_corpus, summary)
        _write_contact_sheet(
            root_output,
            records_by_corpus,
            record_key="cleaned",
            filename="cleaned_contact_sheet.png",
        )
        _write_contact_sheet(
            root_output,
            records_by_corpus,
            record_key="mask_overlay",
            filename="mask_contact_sheet.png",
        )
        _write_contact_sheet(
            root_output,
            records_by_corpus,
            record_key="final_mask",
            filename="final_mask_contact_sheet.png",
        )
        _write_contact_sheet(
            root_output,
            records_by_corpus,
            record_key="protected_corner_mask",
            filename="protected_corner_contact_sheet.png",
        )
        if artifact_run is not None:
            artifact_run.complete(
                metadata={
                    "input_image_count": total_images,
                    "success_count": summary["success_count"],
                    "failure_count": summary["failure_count"],
                    "inpainter": args.inpainter,
                    "use_gpu": bool(args.use_gpu),
                }
            )
        print(root_output)
        return 1 if failures or gate_failures else 0
    except BaseException as exc:
        if artifact_run is not None:
            artifact_run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
