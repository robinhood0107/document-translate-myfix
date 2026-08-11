#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import fnmatch
import json
import math
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import imkit as imk
import cv2
import numpy as np
from PIL import Image, ImageDraw

from modules.detection.processor import TextBlockDetector
from modules.inpainting.runtime_contract import inspect_learned_inpainter_runtime
from modules.inpainting.source_lama_blockwise import (
    source_lama_blockwise_inpaint,
    source_lama_blockwise_inpaint_result,
)
from modules.rendering.render import get_best_render_area
from modules.utils.device import resolve_device
from modules.utils.image_utils import generate_mask
from modules.utils.inpaint_composite import composite_with_edit_mask, normalize_edit_mask
from modules.utils.inpaint_evidence import combine_evidence_patches
from modules.utils.inpaint_cleanup import (
    apply_duplicate_bubble_inner_fill,
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
from scripts.inpaint_eval_contract import (
    EvalManifest,
    EvalPageSpec,
    InpaintEvalManifestError,
    build_quality_metrics,
    derive_blind_review_seed,
    load_binary_mask,
    load_eval_source_array,
    load_eval_manifests,
    load_rgb_reference_array,
    pixel_sha256,
    sha256_file,
    verify_eval_page_spec,
    write_blind_review_jsonl,
    write_comparison_and_blind_panels,
)
from scripts.validation_artifact_harness import select_managed_output_directory

DEBUG_EXPORT_SETTINGS = {
    "export_detector_overlay": True,
    "export_raw_mask": True,
    "export_mask_overlay": True,
    "export_cleanup_mask_delta": True,
    "export_debug_metadata": True,
}

SAFE_PROCESSING_CAUSE_CODES = frozenset(
    {
        "evaluation_image_channels_invalid",
        "evaluation_image_shape_mismatch",
        "evaluation_mask_shape_mismatch",
        "evaluation_pre_composite_shape_mismatch",
        "evaluation_reference_hash_mismatch",
        "evaluation_reference_image_invalid",
        "evaluation_reference_shape_mismatch",
        "evaluation_reference_unreadable",
        "inpaint_input_image_missing",
        "inpaint_output_directory_not_empty",
        "manifest_already_finalized",
        "manifest_corpus_id_invalid",
        "manifest_count_mismatch",
        "manifest_dimension_mismatch",
        "manifest_dimensions_invalid",
        "manifest_duplicate_corpus_id",
        "manifest_duplicate_page_id",
        "manifest_duplicate_page_id_global",
        "manifest_duplicate_source_hash",
        "manifest_expected_count_invalid",
        "manifest_expected_edit_basis_invalid",
        "manifest_expected_edit_decisions_seal_invalid",
        "manifest_expected_edit_invalid",
        "manifest_file_missing",
        "manifest_file_unreadable",
        "manifest_finalization_basis_invalid",
        "manifest_finalization_decision_invalid",
        "manifest_finalization_decisions_invalid",
        "manifest_finalization_decisions_schema_invalid",
        "manifest_finalization_decisions_unreadable",
        "manifest_finalization_duplicate_page",
        "manifest_finalization_expected_edit_invalid",
        "manifest_finalization_incomplete",
        "manifest_finalization_no_optional_pages",
        "manifest_finalization_optional_remaining",
        "manifest_finalization_output_directory_invalid",
        "manifest_finalization_output_exists",
        "manifest_finalization_output_unwritable",
        "manifest_finalization_overwrite_parent",
        "manifest_finalization_parent_changed",
        "manifest_finalization_page_id_invalid",
        "manifest_finalization_page_set_mismatch",
        "manifest_finalization_parent_mismatch",
        "manifest_hash_invalid",
        "manifest_hash_mismatch",
        "manifest_holdout_not_source_review_finalized",
        "manifest_image_invalid",
        "manifest_page_id_invalid",
        "manifest_page_invalid",
        "manifest_page_schema_key_invalid",
        "manifest_page_unknown_key",
        "manifest_pages_invalid",
        "manifest_parent_seal_invalid",
        "manifest_path_missing",
        "manifest_path_unresolvable",
        "manifest_reference_invalid",
        "manifest_reference_unknown_key",
        "manifest_root_invalid",
        "manifest_schema_unsupported",
        "manifest_seal_invalid",
        "manifest_seal_mismatch",
        "manifest_size_mismatch",
        "manifest_source_lock_invalid",
        "manifest_source_lock_mismatch",
        "manifest_source_missing",
        "manifest_annotation_masks_overlap",
        "manifest_split_role_invalid",
        "manifest_unknown_key",
        "manifest_unreadable",
    }
)
SAFE_RUNTIME_PHASES = frozenset(
    {"block", "bubble_erase", "full", "generic"}
)
SAFE_RUNTIME_STATUSES = frozenset(
    {
        "completed",
        "completed_after_roi_retry",
        "failed",
        "failed_after_roi_retry",
        "failed_during_roi_retry",
        "failed_no_smaller_roi",
        "running",
    }
)
SAFE_RUNTIME_ERASE_MODES = frozenset(
    {
        "bubble_flat_fill",
        "bubble_gradient_fill",
        "bubble_lama_fallback",
        "bubble_skipped",
        "bubble_telea",
        "text_free_lama",
    }
)
SAFE_RUNTIME_PRECISIONS = frozenset(
    {"bf16", "bfloat16", "float16", "float32", "fp16", "fp32"}
)
SAFE_RUNTIME_DTYPES = frozenset(
    {"torch.bfloat16", "torch.float16", "torch.float32"}
)
SAFE_RUNTIME_PROVIDERS = frozenset(
    {"CPUExecutionProvider", "CUDAExecutionProvider", "TensorrtExecutionProvider"}
)
SAFE_RUNTIME_CONTRACT_VERSIONS = frozenset({"cuda-learned-inpaint-v1"})
SAFE_RUNTIME_RETRY_POLICIES = frozenset({"single-tighter-roi-v1"})
SAFE_RUNTIME_INPAINTER_KEYS = frozenset({"lama_large_512px"})
SOURCE_REVIEW_FINALIZATION_ROLES = frozenset(
    {"final-holdout-primary", "final-holdout-reserve"}
)
_DROP_RUNTIME_VALUE = object()
QUALITY_GATE_REQUIRED_FIELDS = (
    "outside_changed_pixel_count_exact",
    "protected_structure_changed_pixel_count_exact",
    "protected_structure_annotation_available",
    "protected_structure_annotation_changed_pixel_count_exact",
    "residue_pass_truncated_block_count",
    "residue_target_is_annotation",
    "erase_mode_distribution",
    "erase_skipped_reason_distribution",
)
REQUIRED_ERASE_SKIP_REASONS = frozenset(
    {
        "bubble_interior_cap_source_seed_unavailable",
        "bubble_interior_cap_source_seed_partially_suppressed",
        "bubble_protected_source_seed_unavailable",
        "bubble_residual_source_seed_unavailable",
        "line_art_source_seed_unavailable",
        "microtexture_source_seed_unavailable",
        "microtexture_source_seed_partially_suppressed",
        "text_prior_unavailable_source_seed_unavailable",
    }
)


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


def _reset_cuda_peak_metrics(device: str) -> bool:
    if not str(device or "").lower().startswith("cuda"):
        return False
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(torch.device(device))
            return True
    except (ImportError, RuntimeError, ValueError):
        pass
    return False


def _read_cuda_peak_metrics(device: str) -> dict[str, float | bool]:
    metrics: dict[str, float | bool] = {
        "peak_vram_allocated_mb": 0.0,
        "peak_vram_reserved_mb": 0.0,
        "peak_vram_metrics_available": False,
    }
    if not str(device or "").lower().startswith("cuda"):
        return metrics
    try:
        import torch

        if torch.cuda.is_available():
            torch_device = torch.device(device)
            metrics["peak_vram_allocated_mb"] = float(
                torch.cuda.max_memory_allocated(torch_device) / (1024 * 1024)
            )
            metrics["peak_vram_reserved_mb"] = float(
                torch.cuda.max_memory_reserved(torch_device) / (1024 * 1024)
            )
            metrics["peak_vram_metrics_available"] = True
    except (ImportError, RuntimeError, ValueError):
        pass
    return metrics


def _safe_reference_mask(
    page_spec: EvalPageSpec | None,
    attribute: str,
    image_shape: tuple[int, ...],
) -> np.ndarray | None:
    if page_spec is None:
        return None
    reference = getattr(page_spec, attribute, None)
    if reference is None:
        return None
    return load_binary_mask(reference, image_shape)


def _write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imk.write_image(str(path), ensure_three_channel(image))


def _write_structure_change_contact_sheet(
    source_image: np.ndarray,
    candidate_image: np.ndarray,
    changed_protected_mask: np.ndarray,
    output_path: Path,
    *,
    overlay_mask_on_candidate: bool = False,
) -> Path | None:
    def fit_crop(crop: Image.Image, size: tuple[int, int]) -> Image.Image:
        width, height = crop.size
        scale = min(size[0] / max(1, width), size[1] / max(1, height))
        resized = (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        )
        return crop.resize(resized, Image.Resampling.NEAREST)

    binary = np.where(changed_protected_mask > 0, 1, 0).astype(np.uint8)
    component_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary,
        8,
        cv2.CV_32S,
    )
    rows: list[tuple[int, int, int, int, int]] = []
    for component_index in range(1, component_count):
        x, y, width, height, area = [
            int(value) for value in stats[component_index]
        ]
        if area > 0:
            rows.append((area, x, y, width, height))
    if not rows:
        return None
    rows.sort(reverse=True)
    tile_width = 512
    tile_height = 256
    sheet = Image.new("RGB", (tile_width, tile_height * min(64, len(rows))), "white")
    draw = ImageDraw.Draw(sheet)
    source_rgb = Image.fromarray(ensure_three_channel(source_image).astype(np.uint8))
    candidate_pixels = ensure_three_channel(candidate_image).astype(np.uint8).copy()
    if overlay_mask_on_candidate:
        overlay = changed_protected_mask > 0
        candidate_pixels[overlay] = (
            candidate_pixels[overlay].astype(np.uint16) + np.asarray([255, 0, 0])
        ) // 2
    candidate_rgb = Image.fromarray(candidate_pixels.astype(np.uint8))
    for row_index, (area, x, y, width, height) in enumerate(rows[:64]):
        padding = max(8, int(round(max(width, height) * 0.5)))
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(source_rgb.width, x + width + padding)
        y2 = min(source_rgb.height, y + height + padding)
        source_crop = source_rgb.crop((x1, y1, x2, y2))
        candidate_crop = candidate_rgb.crop((x1, y1, x2, y2))
        source_crop = fit_crop(source_crop, (248, 218))
        candidate_crop = fit_crop(candidate_crop, (248, 218))
        row_y = row_index * tile_height
        sheet.paste(source_crop, (0, row_y + 30))
        sheet.paste(candidate_crop, (256, row_y + 30))
        draw.text((4, row_y + 6), f"source area={area} xyxy={x},{y},{x + width},{y + height}", fill="black")
        draw.text(
            (260, row_y + 6),
            "source + protect" if overlay_mask_on_candidate else "candidate",
            fill="black",
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, format="PNG")
    return output_path


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
    page_spec: EvalPageSpec | None = None,
    public_corpus_id: str | None = None,
    public_page_id: str | None = None,
    runtime_device: str | None = None,
    peak_vram_reset_succeeded: bool = False,
):
    resolved_corpus_id = (
        page_spec.corpus_id
        if page_spec is not None
        else str(public_corpus_id or "private")
    )
    resolved_page_id = (
        page_spec.page_id
        if page_spec is not None
        else str(public_page_id or "direct-001")
    )
    page_started = perf_counter()
    stage_timings: dict[str, float] = {
        "read_seconds": 0.0,
        "detect_seconds": 0.0,
        "render_and_mask_seconds": 0.0,
        "inpaint_seconds": 0.0,
        "cleanup_seconds": 0.0,
    }
    read_started = perf_counter()
    image = (
        load_eval_source_array(page_spec)
        if page_spec is not None
        else imk.read_image(str(image_path))
    )
    if image is None:
        raise RuntimeError("failed to read image")
    image = ensure_three_channel(image)
    stage_timings["read_seconds"] = perf_counter() - read_started
    detect_started = perf_counter()
    blocks = detector.detect(image) or []
    stage_timings["detect_seconds"] = perf_counter() - detect_started
    detector_key = settings.get_tool_selection("detector")
    detector_engine = detector.last_engine_name or ""
    detector_device = detector.last_device or resolve_device(settings.is_gpu_enabled(), backend="onnx")
    raw_mask = None
    final_mask = None
    cleanup_stats = {"applied": False, "component_count": 0, "block_count": 0}
    cleaned = image.copy()
    pre_final_composite = image.copy()
    mask_details = {}
    inpaint_diagnostics: list[dict] = []
    routing_evidence = ()

    if blocks:
        render_mask_started = perf_counter()
        get_best_render_area(
            blocks,
            image,
            auto_max_font_profile=auto_max_font_profile,
        )
        mask_details = generate_mask(image, blocks, settings=settings.get_mask_refiner_settings(), return_details=True)
        stage_timings["render_and_mask_seconds"] = perf_counter() - render_mask_started
        mask = mask_details["final_mask"]
        if mask is not None and np.any(mask):
            raw_mask = mask_details["raw_mask"]
            config = get_config(settings)
            inpaint_started = perf_counter()
            blockwise_result = source_lama_blockwise_inpaint_result(
                image,
                mask,
                blocks,
                inpainter,
                config,
                raw_source_mask=mask_details.get(
                    "positive_claim_raw_mask",
                    raw_mask,
                ),
                check_need_inpaint=True,
                protected_corner_mask=mask_details.get("protected_corner_mask"),
            )
            cleaned = blockwise_result.image
            inpaint_edit_mask = blockwise_result.edit_mask
            inpaint_diagnostics = blockwise_result.diagnostics
            routing_evidence = blockwise_result.evidence
            stage_timings["inpaint_seconds"] = perf_counter() - inpaint_started
            mask = np.where((mask > 0) | (inpaint_edit_mask > 0), 255, 0).astype(np.uint8)
            cleanup_started = perf_counter()
            final_mask = mask
            cleanup_stats = {
                "autonomous_residue_cleanup": "disabled"
            }
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
            pre_final_composite = cleaned.copy()
            cleaned = composite_with_edit_mask(image, cleaned, final_mask)
            stage_timings["cleanup_seconds"] = perf_counter() - cleanup_started
        else:
            final_mask = mask

    cleanup_delta = _build_cleanup_delta(raw_mask, final_mask)
    runtime = get_inpainter_runtime(settings)
    config = get_config(settings)
    evaluation_final_mask = normalize_edit_mask(final_mask, image.shape)
    explicit_residue_target = _safe_reference_mask(
        page_spec,
        "target_text_mask",
        image.shape,
    )
    annotated_protected_structure = _safe_reference_mask(
        page_spec,
        "protected_structure_mask",
        image.shape,
    )
    annotated_ambiguous_structure = _safe_reference_mask(
        page_spec,
        "ambiguous_structure_mask",
        image.shape,
    )
    protected_structure_is_annotation = annotated_protected_structure is not None
    source_text_cap = imk.dilate(
        normalize_edit_mask(raw_mask, image.shape),
        np.ones((5, 5), dtype=np.uint8),
        iterations=1,
    )
    derived_protected_structure = np.where(
        (
            (
                normalize_edit_mask(
                    mask_details.get("protect_mask"),
                    image.shape,
                )
                > 0
            )
            & (source_text_cap <= 0)
        )
        | (
            normalize_edit_mask(
                mask_details.get("protected_corner_mask"),
                image.shape,
            )
            > 0
        ),
        255,
        0,
    ).astype(np.uint8)
    routing_structure_protect = combine_evidence_patches(
        routing_evidence,
        "structure_protect",
        image.shape,
    )
    routing_source_owned = combine_evidence_patches(
        routing_evidence,
        "source_owned",
        image.shape,
    )
    routing_source_raw_owned = combine_evidence_patches(
        routing_evidence,
        "source_raw_owned",
        image.shape,
    )
    routing_ownership_protect = combine_evidence_patches(
        routing_evidence,
        "ownership_protect",
        image.shape,
    )
    routing_positive_claim = combine_evidence_patches(
        routing_evidence,
        "positive_claim",
        image.shape,
    )
    routing_positive_edit = combine_evidence_patches(
        routing_evidence,
        "positive_edit",
        image.shape,
    )
    routing_claim_providers = sorted(
        {
            provider
            for item in routing_evidence
            for provider in item.claim_providers
        }
    )
    evaluation_protected_structure = (
        annotated_protected_structure
        if annotated_protected_structure is not None
        else derived_protected_structure
    )
    quality_metrics = build_quality_metrics(
        image,
        cleaned,
        evaluation_final_mask,
        residue_target_mask=(
            explicit_residue_target
            if explicit_residue_target is not None
            else raw_mask
        ),
        residue_target_is_annotation=explicit_residue_target is not None,
        protected_structure_mask=evaluation_protected_structure,
        pre_composite_candidate_image=pre_final_composite,
    )
    if page_spec is not None and page_spec.baseline is not None:
        baseline_image = load_rgb_reference_array(
            page_spec.baseline,
            image.shape,
        )
        baseline_final_mask = load_binary_mask(
            page_spec.baseline_mask,
            image.shape,
        )
        baseline_quality_metrics = build_quality_metrics(
            image,
            baseline_image,
            baseline_final_mask,
            residue_target_mask=(
                explicit_residue_target
                if explicit_residue_target is not None
                else raw_mask
            ),
            residue_target_is_annotation=explicit_residue_target is not None,
        )
        for field in (
            "residue_pixel_count",
            "residue_ratio",
            "residue_score",
            "residue_score_sum",
            "residue_source_contrast_pixel_count",
            "residue_target_coverage",
            "residue_target_minimum_component_coverage",
        ):
            quality_metrics[f"baseline_{field}"] = baseline_quality_metrics[field]
        candidate_score = quality_metrics.get("residue_score")
        baseline_score = quality_metrics.get("baseline_residue_score")
        quality_metrics["residue_score_delta_from_baseline"] = (
            float(candidate_score) - float(baseline_score)
            if candidate_score is not None and baseline_score is not None
            else None
        )
        quality_metrics["residue_pixel_count_delta_from_baseline"] = int(
            quality_metrics["residue_pixel_count"]
        ) - int(quality_metrics["baseline_residue_pixel_count"])
    else:
        for field in (
            "residue_pixel_count",
            "residue_ratio",
            "residue_score",
            "residue_score_sum",
            "residue_source_contrast_pixel_count",
            "residue_target_coverage",
            "residue_target_minimum_component_coverage",
        ):
            quality_metrics[f"baseline_{field}"] = None
        quality_metrics["residue_score_delta_from_baseline"] = None
        quality_metrics["residue_pixel_count_delta_from_baseline"] = None
    quality_metrics["protected_structure_source"] = (
        "private_annotation"
        if protected_structure_is_annotation
        else "derived_line_protection"
    )
    changed_exact_for_structure = np.any(
        np.asarray(cleaned)[:, :, :3] != np.asarray(image)[:, :, :3],
        axis=2,
    )
    quality_metrics.update(
        {
            "positive_claim_raw_pixel_count": int(
                np.count_nonzero(
                    normalize_edit_mask(
                        mask_details.get("positive_claim_raw_mask"),
                        image.shape,
                    )
                )
            ),
            "positive_claim_runtime": dict(
                mask_details.get("positive_claim_runtime", {}) or {}
            ),
            "protected_structure_annotation_available": bool(
                protected_structure_is_annotation
            ),
            "ambiguous_structure_annotation_available": bool(
                annotated_ambiguous_structure is not None
            ),
            "ambiguous_structure_changed_pixel_count_exact": (
                int(
                    np.count_nonzero(
                        (annotated_ambiguous_structure > 0)
                        & changed_exact_for_structure
                    )
                )
                if annotated_ambiguous_structure is not None
                else None
            ),
            "protected_structure_annotation_changed_pixel_count_exact": (
                int(
                    np.count_nonzero(
                        (annotated_protected_structure > 0)
                        & changed_exact_for_structure
                    )
                )
                if annotated_protected_structure is not None
                else None
            ),
            "derived_protected_structure_pixel_count": int(
                np.count_nonzero(derived_protected_structure)
            ),
            "derived_protected_structure_changed_pixel_count_exact": int(
                np.count_nonzero(
                    (derived_protected_structure > 0)
                    & changed_exact_for_structure
                )
            ),
            "routing_structure_protect_pixel_count": int(
                np.count_nonzero(routing_structure_protect)
            ),
            "routing_source_owned_pixel_count": int(
                np.count_nonzero(routing_source_owned)
            ),
            "routing_source_raw_owned_pixel_count": int(
                np.count_nonzero(routing_source_raw_owned)
            ),
            "routing_ownership_protect_pixel_count": int(
                np.count_nonzero(routing_ownership_protect)
            ),
            "routing_positive_claim_pixel_count": int(
                np.count_nonzero(routing_positive_claim)
            ),
            "routing_positive_edit_pixel_count": int(
                np.count_nonzero(routing_positive_edit)
            ),
            "routing_claim_providers": routing_claim_providers,
            "routing_structure_changed_pixel_count_exact": int(
                np.count_nonzero(
                    (routing_structure_protect > 0)
                    & changed_exact_for_structure
                )
            ),
        }
    )
    public_image_path = f"{resolved_corpus_id}/{resolved_page_id}"
    metadata = build_inpaint_debug_metadata(
        image_path=public_image_path,
        run_type="manifest_debug" if page_spec is not None else "sample_debug",
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
        mask_score_outside_change=float(
            mask_details.get("mask_score_outside_change", 0.0) or 0.0
        ),
        mask_score_outline_damage=float(
            mask_details.get("mask_score_outline_damage", 0.0) or 0.0
        ),
        mask_score_residue=float(
            mask_details.get("mask_score_residue", 0.0) or 0.0
        ),
        mask_score_color_delta=float(
            mask_details.get("mask_score_color_delta", 0.0) or 0.0
        ),
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
    inference_diagnostics = [
        item
        for item in inpaint_diagnostics
        if bool(item.get("is_inference", True))
    ]
    metadata["inpaint_runtime_inference_call_count"] = len(inference_diagnostics)
    metadata["inpaint_runtime_cpu_fallback_count"] = sum(
        1
        for item in inference_diagnostics
        if bool(item.get("cpu_fallback_used", False))
    )
    erase_mode_distribution = Counter(
        str(getattr(block, "_erase_mode", "") or "unassigned")
        for block in blocks
    )
    erase_skipped_reason_distribution = Counter(
        str(getattr(block, "_erase_skipped_reason", "") or "")
        for block in blocks
        if str(getattr(block, "_erase_skipped_reason", "") or "")
    )
    resolved_runtime_device = str(
        runtime_device
        or getattr(inpainter, "runtime_device", getattr(inpainter, "device", ""))
        or ""
    )
    peak_vram = _read_cuda_peak_metrics(resolved_runtime_device)
    peak_vram["peak_vram_reset_succeeded"] = bool(
        peak_vram_reset_succeeded
    )
    pipeline_elapsed_seconds = perf_counter() - page_started
    metadata.update(quality_metrics)
    metadata["evaluation_scores"] = {
        "outside_change": quality_metrics.get(
            "pre_composite_outside_change_ratio"
        ),
        "outline_damage": quality_metrics.get(
            "pre_composite_outline_damage_ratio"
        ),
        "residue": quality_metrics.get("residue_score"),
        "color_delta": quality_metrics.get("color_delta_score"),
    }
    metadata["evaluation_score_availability"] = {
        key: value is not None
        for key, value in metadata["evaluation_scores"].items()
    }
    metadata.update(peak_vram)
    metadata.update(
        {
            "corpus_id": resolved_corpus_id,
            "page_id": resolved_page_id,
            "expected_edit": page_spec.expected_edit if page_spec is not None else "required",
            "source_sha256": (
                page_spec.source.sha256
                if page_spec is not None
                else sha256_file(image_path)
            ),
            "source_pixel_sha256": pixel_sha256(image),
            "source_size_bytes": (
                page_spec.size_bytes
                if page_spec is not None
                else image_path.stat().st_size
            ),
            "stage_timings_seconds": stage_timings,
            "pipeline_elapsed_seconds": pipeline_elapsed_seconds,
            "block_runtime_seconds": [
                {
                    "block_index": item.get("block_index"),
                    "phase": str(item.get("phase", "") or ""),
                    "elapsed_seconds": float(item.get("elapsed_seconds", 0.0) or 0.0),
                }
                for item in inpaint_diagnostics
                if item.get("block_index") is not None
                and float(item.get("elapsed_seconds", 0.0) or 0.0) >= 0.0
            ],
            "erase_mode_distribution": dict(sorted(erase_mode_distribution.items())),
            "erase_skipped_reason_distribution": dict(
                sorted(erase_skipped_reason_distribution.items())
            ),
            "residue_pass_truncated_block_count": int(
                cleanup_stats.get("residue_pass_truncated_block_count", 0) or 0
            ),
        }
    )

    page_base_name = resolved_page_id
    source_path = corpus_output / "source_images" / f"{page_base_name}_source.png"
    cleaned_path = corpus_output / "cleaned_images" / f"{page_base_name}_cleaned.png"
    final_mask_path = corpus_output / "final_masks" / f"{page_base_name}_final_mask.png"
    bubble_cap_path = corpus_output / "bubble_interior_caps" / f"{page_base_name}_bubble_cap.png"
    protected_corner_path = corpus_output / "protected_corner_masks" / f"{page_base_name}_protected_corners.png"
    routing_structure_path = corpus_output / "routing_structure_masks" / f"{page_base_name}_routing_structure.png"
    routing_source_owned_path = corpus_output / "routing_source_owned_masks" / f"{page_base_name}_routing_source_owned.png"
    routing_source_raw_owned_path = corpus_output / "routing_source_raw_owned_masks" / f"{page_base_name}_routing_source_raw_owned.png"
    routing_ownership_protect_path = corpus_output / "routing_ownership_protect_masks" / f"{page_base_name}_routing_ownership_protect.png"
    routing_positive_claim_path = corpus_output / "routing_positive_claim_masks" / f"{page_base_name}_routing_positive_claim.png"
    routing_positive_edit_path = corpus_output / "routing_positive_edit_masks" / f"{page_base_name}_routing_positive_edit.png"
    positive_claim_raw_path = corpus_output / "positive_claim_raw_masks" / f"{page_base_name}_positive_claim_raw.png"
    changed_routing_structure_path = corpus_output / "routing_structure_changes" / f"{page_base_name}_routing_structure_changed.png"
    structure_contact_sheet_path = corpus_output / "routing_structure_contact_sheets" / f"{page_base_name}_routing_structure_changes.png"
    structure_source_contact_sheet_path = corpus_output / "routing_structure_source_contact_sheets" / f"{page_base_name}_routing_structure_source.png"
    derived_structure_path = corpus_output / "derived_structure_proxy_masks" / f"{page_base_name}_derived_structure_proxy.png"
    changed_derived_structure_path = corpus_output / "derived_structure_proxy_changes" / f"{page_base_name}_derived_structure_proxy_changed.png"
    derived_structure_contact_sheet_path = corpus_output / "derived_structure_proxy_contact_sheets" / f"{page_base_name}_derived_structure_proxy_changes.png"
    ambiguous_structure_path = corpus_output / "ambiguous_structure_masks" / f"{page_base_name}_ambiguous_structure.png"
    changed_ambiguous_structure_path = corpus_output / "ambiguous_structure_changes" / f"{page_base_name}_ambiguous_structure_changed.png"
    ambiguous_structure_contact_sheet_path = corpus_output / "ambiguous_structure_contact_sheets" / f"{page_base_name}_ambiguous_structure_changes.png"
    write_started = perf_counter()
    cleaned_for_write = ensure_three_channel(cleaned)
    _write_image(source_path, image)
    _write_image(cleaned_path, cleaned_for_write)
    _write_image(final_mask_path, normalized_final_mask)
    _write_image(bubble_cap_path, bubble_cap_mask)
    _write_image(protected_corner_path, protected_corner_mask)
    changed_routing_structure = np.where(
        (routing_structure_protect > 0) & changed_exact_for_structure,
        255,
        0,
    ).astype(np.uint8)
    changed_derived_structure = np.where(
        (derived_protected_structure > 0) & changed_exact_for_structure,
        255,
        0,
    ).astype(np.uint8)
    normalized_ambiguous_structure = normalize_edit_mask(
        annotated_ambiguous_structure,
        image.shape,
    )
    changed_ambiguous_structure = np.where(
        (normalized_ambiguous_structure > 0) & changed_exact_for_structure,
        255,
        0,
    ).astype(np.uint8)
    _write_image(routing_structure_path, routing_structure_protect)
    _write_image(routing_source_owned_path, routing_source_owned)
    _write_image(routing_source_raw_owned_path, routing_source_raw_owned)
    _write_image(routing_ownership_protect_path, routing_ownership_protect)
    _write_image(routing_positive_claim_path, routing_positive_claim)
    _write_image(routing_positive_edit_path, routing_positive_edit)
    _write_image(
        positive_claim_raw_path,
        normalize_edit_mask(
            mask_details.get("positive_claim_raw_mask"),
            image.shape,
        ),
    )
    _write_image(changed_routing_structure_path, changed_routing_structure)
    _write_image(derived_structure_path, derived_protected_structure)
    _write_image(changed_derived_structure_path, changed_derived_structure)
    _write_image(ambiguous_structure_path, normalized_ambiguous_structure)
    _write_image(
        changed_ambiguous_structure_path,
        changed_ambiguous_structure,
    )
    written_structure_contact_sheet = _write_structure_change_contact_sheet(
        image,
        cleaned_for_write,
        changed_routing_structure,
        structure_contact_sheet_path,
    )
    written_structure_source_contact_sheet = _write_structure_change_contact_sheet(
        image,
        image,
        routing_structure_protect,
        structure_source_contact_sheet_path,
        overlay_mask_on_candidate=True,
    )
    written_derived_structure_contact_sheet = _write_structure_change_contact_sheet(
        image,
        cleaned_for_write,
        changed_derived_structure,
        derived_structure_contact_sheet_path,
    )
    written_ambiguous_structure_contact_sheet = _write_structure_change_contact_sheet(
        image,
        cleaned_for_write,
        changed_ambiguous_structure,
        ambiguous_structure_contact_sheet_path,
    )
    with Image.open(cleaned_path) as saved_cleaned:
        cleaned_artifact_pixels = np.asarray(saved_cleaned.convert("RGB")).copy()
    with Image.open(final_mask_path) as saved_final_mask:
        final_mask_artifact_pixels = np.where(
            np.asarray(saved_final_mask.convert("L")) > 0,
            255,
            0,
        ).astype(np.uint8)
    metadata.update(
        {
            "primary_artifact_write_seconds": perf_counter() - write_started,
            "cleaned_sha256": sha256_file(cleaned_path),
            "final_mask_sha256": sha256_file(final_mask_path),
            "cleaned_pixel_sha256": pixel_sha256(cleaned_artifact_pixels),
            "final_mask_pixel_sha256": pixel_sha256(
                final_mask_artifact_pixels
            ),
        }
    )
    baseline_cleaned_sha256 = (
        page_spec.baseline.sha256
        if page_spec is not None and page_spec.baseline is not None
        else None
    )
    baseline_final_mask_sha256 = (
        page_spec.baseline_mask.sha256
        if page_spec is not None and page_spec.baseline_mask is not None
        else None
    )
    baseline_cleaned_pixel_sha256 = (
        pixel_sha256(
            load_rgb_reference_array(page_spec.baseline, cleaned_for_write.shape)
        )
        if page_spec is not None and page_spec.baseline is not None
        else None
    )
    baseline_final_mask_pixel_sha256 = (
        pixel_sha256(
            load_binary_mask(page_spec.baseline_mask, normalized_final_mask.shape)
        )
        if page_spec is not None and page_spec.baseline_mask is not None
        else None
    )
    metadata.update(
        {
            "baseline_cleaned_sha256": baseline_cleaned_sha256,
            "baseline_final_mask_sha256": baseline_final_mask_sha256,
            "baseline_cleaned_pixel_sha256": baseline_cleaned_pixel_sha256,
            "baseline_final_mask_pixel_sha256": (
                baseline_final_mask_pixel_sha256
            ),
            "cleaned_matches_baseline_sha256": (
                metadata["cleaned_sha256"] == baseline_cleaned_sha256
                if baseline_cleaned_sha256 is not None
                else None
            ),
            "final_mask_matches_baseline_sha256": (
                metadata["final_mask_sha256"] == baseline_final_mask_sha256
                if baseline_final_mask_sha256 is not None
                else None
            ),
            "cleaned_matches_baseline_pixel_sha256": (
                metadata["cleaned_pixel_sha256"]
                == baseline_cleaned_pixel_sha256
                if baseline_cleaned_pixel_sha256 is not None
                else None
            ),
            "final_mask_matches_baseline_pixel_sha256": (
                metadata["final_mask_pixel_sha256"]
                == baseline_final_mask_pixel_sha256
                if baseline_final_mask_pixel_sha256 is not None
                else None
            ),
        }
    )
    export_inpaint_debug_artifacts(
        export_root=str(corpus_output),
        archive_bname="",
        page_base_name=page_base_name,
        image=image,
        blocks=blocks,
        export_settings=DEBUG_EXPORT_SETTINGS,
        raw_mask=raw_mask,
        mask_overlay_mask=final_mask,
        cleanup_delta=cleanup_delta,
        metadata=metadata,
    )

    return {
        "image": resolved_page_id,
        "page_id": resolved_page_id,
        "expected_edit": page_spec.expected_edit if page_spec is not None else "required",
        "source": source_path,
        "cleaned": cleaned_path,
        "final_mask": final_mask_path,
        "bubble_interior_cap": bubble_cap_path,
        "protected_corner_mask": protected_corner_path,
        "routing_structure_mask": routing_structure_path,
        "routing_source_owned_mask": routing_source_owned_path,
        "routing_source_raw_owned_mask": routing_source_raw_owned_path,
        "routing_ownership_protect_mask": routing_ownership_protect_path,
        "routing_positive_claim_mask": routing_positive_claim_path,
        "routing_positive_edit_mask": routing_positive_edit_path,
        "positive_claim_raw_mask": positive_claim_raw_path,
        "routing_structure_changed_mask": changed_routing_structure_path,
        "routing_structure_contact_sheet": written_structure_contact_sheet,
        "routing_structure_source_contact_sheet": written_structure_source_contact_sheet,
        "derived_structure_proxy_mask": derived_structure_path,
        "derived_structure_proxy_changed_mask": changed_derived_structure_path,
        "derived_structure_proxy_contact_sheet": written_derived_structure_contact_sheet,
        "ambiguous_structure_mask": ambiguous_structure_path,
        "ambiguous_structure_changed_mask": changed_ambiguous_structure_path,
        "ambiguous_structure_contact_sheet": written_ambiguous_structure_contact_sheet,
        "detector_overlay": corpus_output / "detector_overlays" / f"{page_base_name}_detector_overlay.png",
        "raw_mask": corpus_output / "raw_masks" / f"{page_base_name}_raw_mask.png",
        "mask_overlay": corpus_output / "mask_overlays" / f"{page_base_name}_mask_overlay.png",
        "cleanup_delta": corpus_output / "cleanup_mask_delta" / f"{page_base_name}_cleanup_delta.png",
        "metadata": corpus_output / "debug_metadata" / f"{page_base_name}_debug.json",
        "source_sha256": metadata["source_sha256"],
        "source_pixel_sha256": metadata["source_pixel_sha256"],
        "source_size_bytes": metadata["source_size_bytes"],
        "cleaned_sha256": metadata["cleaned_sha256"],
        "final_mask_sha256": metadata["final_mask_sha256"],
        "cleaned_pixel_sha256": metadata["cleaned_pixel_sha256"],
        "final_mask_pixel_sha256": metadata["final_mask_pixel_sha256"],
        "baseline_cleaned_sha256": metadata["baseline_cleaned_sha256"],
        "baseline_final_mask_sha256": metadata["baseline_final_mask_sha256"],
        "baseline_cleaned_pixel_sha256": metadata[
            "baseline_cleaned_pixel_sha256"
        ],
        "baseline_final_mask_pixel_sha256": metadata[
            "baseline_final_mask_pixel_sha256"
        ],
        "cleaned_matches_baseline_sha256": metadata[
            "cleaned_matches_baseline_sha256"
        ],
        "final_mask_matches_baseline_sha256": metadata[
            "final_mask_matches_baseline_sha256"
        ],
        "cleaned_matches_baseline_pixel_sha256": metadata[
            "cleaned_matches_baseline_pixel_sha256"
        ],
        "final_mask_matches_baseline_pixel_sha256": metadata[
            "final_mask_matches_baseline_pixel_sha256"
        ],
        "block_count": len(blocks),
        "final_mask_pixel_count": int(np.count_nonzero(final_mask)) if final_mask is not None else 0,
        "hd_strategy": str(config.hd_strategy),
        "auto_max_font_profile": str(auto_max_font_profile),
        "refiner_backend": str(mask_details.get("refiner_backend", "") or ""),
        "refiner_device": str(mask_details.get("refiner_device", "") or ""),
        "inpaint_runtime_diagnostics": list(inpaint_diagnostics),
        "inpaint_runtime_inference_call_count": len(inference_diagnostics),
        "inpaint_runtime_cpu_fallback_count": sum(
            1
            for item in inference_diagnostics
            if bool(item.get("cpu_fallback_used", False))
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
        **quality_metrics,
        **peak_vram,
        "stage_timings_seconds": dict(stage_timings),
        "block_runtime_seconds": list(metadata["block_runtime_seconds"]),
        "pipeline_elapsed_seconds": pipeline_elapsed_seconds,
        "erase_mode_distribution": dict(sorted(erase_mode_distribution.items())),
        "erase_skipped_reason_distribution": dict(
            sorted(erase_skipped_reason_distribution.items())
        ),
        "residue_pass_truncated_block_count": int(
            cleanup_stats.get("residue_pass_truncated_block_count", 0) or 0
        ),
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
    parser.add_argument(
        "--manifest",
        action="append",
        type=Path,
        default=[],
        help=(
            "Sealed private evaluation manifest. Repeat for multiple neutral-ID "
            "corpora; cannot be combined with --input."
        ),
    )
    parser.add_argument(
        "--blind-review-duplicate-count",
        type=int,
        default=0,
        help="Deterministically repeat this many blinded pages for reviewer consistency.",
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
        "--require-quality-gates",
        action="store_true",
        help=(
            "Fail on outside-mask changes, protected-structure damage, cleanup "
            "truncation, or annotated target coverage below 98 percent."
        ),
    )
    parser.add_argument(
        "--require-baseline-parity",
        action="store_true",
        help=(
            "Fail unless every manifest page has locked cleaned/final-mask "
            "references and both artifact SHA-256 values match exactly."
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
    require_quality_gates: bool = False,
    require_baseline_parity: bool = False,
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
    if str(summary.get("input_mode", "")) == "manifest":
        for corpus_name, contract in dict(
            summary.get("manifest_corpora") or {}
        ).items():
            expected = int(dict(contract or {}).get("expected_count", 0) or 0)
            completed = len(records_by_corpus.get(str(corpus_name), []))
            if completed != expected:
                failures.append(
                    f"{corpus_name}:manifest_success_count_mismatch:{completed}!={expected}"
                )
        for corpus_name, records in records_by_corpus.items():
            for record in records:
                page_name = f"{corpus_name}/{record.get('page_id', 'page')}"
                expected_edit = str(record.get("expected_edit", "required") or "required")
                if expected_edit == "required":
                    if int(record.get("block_count", 0) or 0) <= 0:
                        failures.append(f"{page_name}:expected_edit_missing_detection")
                    if int(record.get("final_mask_pixel_count", 0) or 0) <= 0:
                        failures.append(f"{page_name}:expected_edit_empty_mask")
                elif expected_edit == "none" and int(
                    record.get("final_mask_pixel_count", 0) or 0
                ) > 0:
                    failures.append(f"{page_name}:unexpected_edit_mask")
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
        if int(summary.get("peak_vram_unavailable_count", 1) or 0) > 0:
            failures.append("cuda_peak_vram_metrics_unavailable")
        if int(summary.get("peak_vram_reset_failure_count", 1) or 0) > 0:
            failures.append("cuda_peak_vram_reset_failed")
        if int(
            summary.get("cuda_memory_diagnostics_unavailable_count", 1) or 0
        ) > 0:
            failures.append("cuda_inference_memory_diagnostics_unavailable")
        if int(
            summary.get(
                "required_zero_block_count",
                summary.get("zero_block_count", 0),
            )
            or 0
        ) > 0:
            failures.append("empty_detection_detected")
        if int(
            summary.get(
                "required_empty_final_mask_count",
                summary.get("empty_final_mask_count", 0),
            )
            or 0
        ) > 0:
            failures.append("empty_final_mask_detected")
        if (
            int(summary.get("expected_edit_active_count", 1) or 0) > 0
            and int(summary.get("runtime_inference_call_count", 0) or 0) <= 0
        ):
            failures.append("no_inpaint_inference")

    if require_quality_gates:
        if str(summary.get("input_mode", "")) == "manifest":
            for corpus_name, raw_contract in dict(
                summary.get("manifest_corpora") or {}
            ).items():
                contract = dict(raw_contract or {})
                if str(contract.get("split_role", "")) not in (
                    SOURCE_REVIEW_FINALIZATION_ROLES
                ):
                    continue
                parent_seal = str(
                    contract.get("parent_manifest_sha256", "") or ""
                )
                decisions_seal = str(
                    contract.get("expected_edit_decisions_sha256", "") or ""
                )
                valid_hex = frozenset("0123456789abcdef")
                source_review_finalized = (
                    len(parent_seal) == 64
                    and set(parent_seal) <= valid_hex
                    and contract.get("expected_edit_basis")
                    == "source-only-review"
                    and len(decisions_seal) == 64
                    and set(decisions_seal) <= valid_hex
                )
                if not source_review_finalized:
                    failures.append(
                        f"{corpus_name}:holdout_not_source_review_finalized"
                    )
        for corpus_name, records in records_by_corpus.items():
            manifest_schema_version = int(
                dict(summary.get("manifest_corpora") or {})
                .get(corpus_name, {})
                .get("schema_version", 1)
                or 1
            )
            for record in records:
                page_name = f"{corpus_name}/{record.get('page_id', 'page')}"
                for field in QUALITY_GATE_REQUIRED_FIELDS:
                    if field not in record:
                        failures.append(
                            f"{page_name}:quality_metric_missing:{field}"
                        )
                if str(record.get("expected_edit", "") or "") == "optional":
                    failures.append(f"{page_name}:expected_edit_optional_not_final")
                if int(
                    record.get("outside_changed_pixel_count_exact", 0)
                    or 0
                ) != 0:
                    failures.append(f"{page_name}:changed_outside_final_mask")
                if not bool(
                    record.get("protected_structure_annotation_available", False)
                ):
                    failures.append(
                        f"{page_name}:protected_structure_annotation_missing"
                    )
                elif int(
                    record.get(
                        "protected_structure_annotation_changed_pixel_count_exact",
                        0,
                    )
                    or 0
                ) != 0:
                    failures.append(f"{page_name}:protected_structure_changed")
                if manifest_schema_version >= 2 and not bool(
                    record.get("ambiguous_structure_annotation_available", False)
                ):
                    failures.append(
                        f"{page_name}:ambiguous_structure_annotation_missing"
                    )
                if int(
                    record.get("residue_pass_truncated_block_count", 0) or 0
                ) != 0:
                    failures.append(f"{page_name}:cleanup_truncated")
                if bool(record.get("residue_target_is_annotation", False)):
                    target_pixel_count = record.get("residue_target_pixel_count")
                    has_target_pixels = (
                        target_pixel_count is not None
                        and int(target_pixel_count) > 0
                    )
                    if target_pixel_count is None or has_target_pixels:
                        coverage = record.get("residue_target_coverage")
                        if coverage is None or float(coverage) < 0.98:
                            failures.append(
                                f"{page_name}:target_coverage_below_98pct"
                            )
                        minimum_component_coverage = record.get(
                            "residue_target_minimum_component_coverage"
                        )
                        if (
                            minimum_component_coverage is None
                            or float(minimum_component_coverage) < 0.98
                        ):
                            failures.append(
                                f"{page_name}:target_component_coverage_below_98pct"
                            )
                    if (
                        has_target_pixels
                        and
                        manifest_schema_version >= 2
                        and str(
                            record.get("expected_edit", "required") or "required"
                        )
                        == "required"
                    ):
                        candidate_residue_score = record.get("residue_score")
                        baseline_residue_score = record.get(
                            "baseline_residue_score"
                        )
                        if (
                            candidate_residue_score is None
                            or baseline_residue_score is None
                        ):
                            failures.append(
                                f"{page_name}:baseline_residue_metric_missing"
                            )
                        elif float(candidate_residue_score) > float(
                            baseline_residue_score
                        ) + 1e-12:
                            failures.append(
                                f"{page_name}:residue_worse_than_baseline"
                            )
                if (
                    str(record.get("expected_edit", "required") or "required")
                    == "required"
                    and sum(
                        int(
                            dict(
                                record.get(
                                    "erase_skipped_reason_distribution"
                                )
                                or {}
                            ).get(reason, 0)
                            or 0
                        )
                        for reason in REQUIRED_ERASE_SKIP_REASONS
                    )
                    > 0
                ):
                    failures.append(f"{page_name}:required_bubble_erase_skipped")
        if any(
            int(dict(contract or {}).get("schema_version", 1) or 1) >= 2
            for contract in dict(summary.get("manifest_corpora") or {}).values()
        ):
            aggregate_residue_score = summary.get("aggregate_residue_score")
            baseline_aggregate_residue_score = summary.get(
                "baseline_aggregate_residue_score"
            )
            if (
                aggregate_residue_score is None
                or baseline_aggregate_residue_score is None
            ):
                failures.append("aggregate:baseline_residue_metric_missing")
            elif float(aggregate_residue_score) >= float(
                baseline_aggregate_residue_score
            ) - 1e-12:
                failures.append("aggregate:residue_not_reduced_from_baseline")

    if require_baseline_parity:
        for corpus_name, records in records_by_corpus.items():
            for record in records:
                page_name = f"{corpus_name}/{record.get('page_id', 'page')}"
                if record.get("cleaned_matches_baseline_sha256") is None:
                    failures.append(f"{page_name}:baseline_cleaned_unavailable")
                elif not bool(record.get("cleaned_matches_baseline_sha256")):
                    failures.append(f"{page_name}:baseline_cleaned_sha_mismatch")
                if record.get("cleaned_matches_baseline_pixel_sha256") is None:
                    failures.append(
                        f"{page_name}:baseline_cleaned_pixel_unavailable"
                    )
                elif not bool(
                    record.get("cleaned_matches_baseline_pixel_sha256")
                ):
                    failures.append(
                        f"{page_name}:baseline_cleaned_pixel_mismatch"
                    )
                if record.get("final_mask_matches_baseline_sha256") is None:
                    failures.append(f"{page_name}:baseline_final_mask_unavailable")
                elif not bool(record.get("final_mask_matches_baseline_sha256")):
                    failures.append(f"{page_name}:baseline_final_mask_sha_mismatch")
                if record.get("final_mask_matches_baseline_pixel_sha256") is None:
                    failures.append(
                        f"{page_name}:baseline_final_mask_pixel_unavailable"
                    )
                elif not bool(
                    record.get("final_mask_matches_baseline_pixel_sha256")
                ):
                    failures.append(
                        f"{page_name}:baseline_final_mask_pixel_mismatch"
                    )

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


def _write_page_metrics_jsonl(
    root_output: Path,
    records_by_corpus: dict[str, list[dict]],
) -> Path:
    metrics_dir = root_output / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    output = metrics_dir / "pages.jsonl"
    retained_fields = (
        "page_id",
        "expected_edit",
        "source_sha256",
        "source_pixel_sha256",
        "source_size_bytes",
        "cleaned_sha256",
        "final_mask_sha256",
        "cleaned_pixel_sha256",
        "final_mask_pixel_sha256",
        "baseline_cleaned_sha256",
        "baseline_final_mask_sha256",
        "baseline_cleaned_pixel_sha256",
        "baseline_final_mask_pixel_sha256",
        "cleaned_matches_baseline_sha256",
        "final_mask_matches_baseline_sha256",
        "cleaned_matches_baseline_pixel_sha256",
        "final_mask_matches_baseline_pixel_sha256",
        "block_count",
        "final_mask_pixel_count",
        "inpaint_runtime_inference_call_count",
        "inpaint_runtime_cpu_fallback_count",
        "bubble_silhouette_fallback_count",
        "protected_corner_final_mask_pixel_count",
        "protected_corner_changed_pixel_count",
        "changed_outside_final_mask_pixel_count_exact",
        "outside_pixel_count",
        "outside_changed_pixel_count_exact",
        "outside_change_ratio",
        "pre_composite_outside_changed_pixel_count_exact",
        "pre_composite_outside_change_ratio",
        "residue_target_pixel_count",
        "residue_target_covered_pixel_count",
        "residue_target_coverage",
        "residue_target_component_coverages",
        "residue_target_minimum_component_coverage",
        "residue_target_is_annotation",
        "residue_target_source",
        "residue_source_contrast_pixel_count",
        "residue_pixel_count",
        "residue_ratio",
        "residue_score",
        "residue_score_sum",
        "baseline_residue_pixel_count",
        "baseline_residue_ratio",
        "baseline_residue_score",
        "baseline_residue_score_sum",
        "baseline_residue_source_contrast_pixel_count",
        "baseline_residue_target_coverage",
        "baseline_residue_target_minimum_component_coverage",
        "residue_score_delta_from_baseline",
        "residue_pixel_count_delta_from_baseline",
        "protected_structure_pixel_count",
        "protected_structure_changed_pixel_count_exact",
        "protected_structure_source",
        "protected_structure_annotation_available",
        "protected_structure_annotation_changed_pixel_count_exact",
        "ambiguous_structure_annotation_available",
        "ambiguous_structure_changed_pixel_count_exact",
        "derived_protected_structure_pixel_count",
        "derived_protected_structure_changed_pixel_count_exact",
        "routing_structure_protect_pixel_count",
        "routing_source_owned_pixel_count",
        "routing_source_raw_owned_pixel_count",
        "routing_ownership_protect_pixel_count",
        "routing_positive_claim_pixel_count",
        "routing_positive_edit_pixel_count",
        "positive_claim_raw_pixel_count",
        "positive_claim_runtime",
        "routing_claim_providers",
        "routing_structure_changed_pixel_count_exact",
        "outline_damage_ratio",
        "pre_composite_protected_structure_changed_pixel_count_exact",
        "pre_composite_outline_damage_ratio",
        "color_delta_mean",
        "color_delta_p95",
        "color_delta_score",
        "cleanup_component_count",
        "cleanup_block_count",
        "residue_pass_truncated_block_count",
        "erase_mode_distribution",
        "erase_skipped_reason_distribution",
        "stage_timings_seconds",
        "block_runtime_seconds",
        "pipeline_elapsed_seconds",
        "peak_vram_allocated_mb",
        "peak_vram_reserved_mb",
        "peak_vram_metrics_available",
        "peak_vram_reset_succeeded",
        "inpaint_runtime_diagnostics",
    )
    rows: list[dict] = []
    for corpus_id, records in sorted(records_by_corpus.items()):
        for record in sorted(records, key=lambda item: str(item.get("page_id", ""))):
            row = {"corpus_id": corpus_id}
            for field in retained_fields:
                if field not in record:
                    continue
                if field == "inpaint_runtime_diagnostics":
                    row[field] = _sanitize_runtime_diagnostics(record[field])
                else:
                    row[field] = record[field]
            rows.append(row)
    output.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return output


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(values, percentile)) if values else 0.0


def _sanitized_traceback_summary(exc: BaseException) -> list[dict[str, object]]:
    return [
        {
            "file": Path(frame.filename).name,
            "line": int(frame.lineno),
            "function": str(frame.name),
        }
        for frame in traceback.extract_tb(exc.__traceback__)[-8:]
    ]


def _sanitize_runtime_device(value: object) -> object:
    if not isinstance(value, str):
        return _DROP_RUNTIME_VALUE
    candidate = value.strip().lower()
    if candidate in {"cpu", "cuda"}:
        return candidate
    prefix, separator, index = candidate.partition(":")
    if prefix == "cuda" and separator and index.isdigit() and len(index) <= 2:
        return candidate
    return _DROP_RUNTIME_VALUE


def _sanitize_runtime_number(
    value: object,
    *,
    integer: bool,
    maximum: float,
    nullable: bool = False,
) -> object:
    if value is None and nullable:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return _DROP_RUNTIME_VALUE
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0 or numeric > maximum:
        return _DROP_RUNTIME_VALUE
    if integer:
        if not numeric.is_integer():
            return _DROP_RUNTIME_VALUE
        return int(numeric)
    return numeric


def _sanitize_runtime_bbox(value: object, *, nullable: bool) -> object:
    if value is None and nullable:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return _DROP_RUNTIME_VALUE
    normalized: list[int] = []
    for coordinate in value:
        sanitized = _sanitize_runtime_number(
            coordinate,
            integer=True,
            maximum=2_147_483_647,
        )
        if sanitized is _DROP_RUNTIME_VALUE:
            return _DROP_RUNTIME_VALUE
        normalized.append(int(sanitized))
    return normalized


def _sanitize_runtime_value(field: str, value: object) -> object:
    enum_values = {
        "phase": SAFE_RUNTIME_PHASES,
        "status": SAFE_RUNTIME_STATUSES,
        "erase_mode": SAFE_RUNTIME_ERASE_MODES,
        "actual_precision": SAFE_RUNTIME_PRECISIONS,
        "model_parameter_dtype": SAFE_RUNTIME_DTYPES,
        "contract_version": SAFE_RUNTIME_CONTRACT_VERSIONS,
        "retry_policy": SAFE_RUNTIME_RETRY_POLICIES,
        "inpainter_key": SAFE_RUNTIME_INPAINTER_KEYS,
    }
    if field in enum_values:
        return value if isinstance(value, str) and value in enum_values[field] else _DROP_RUNTIME_VALUE
    if field in {"actual_device", "model_parameter_device"}:
        return _sanitize_runtime_device(value)
    if field in {
        "cpu_fallback_used",
        "cuda_memory_diagnostics_available",
        "cuda_memory_diagnostics_unavailable",
        "device_verified_from_model",
        "fp32_promotion_eligible",
        "is_inference",
    }:
        return value if isinstance(value, bool) else _DROP_RUNTIME_VALUE
    if field == "block_index":
        return _sanitize_runtime_number(
            value,
            integer=True,
            maximum=10_000_000,
            nullable=True,
        )
    if field in {"mask_pixel_count"}:
        return _sanitize_runtime_number(
            value,
            integer=True,
            maximum=2**63 - 1,
        )
    if field == "oom_retry_count":
        sanitized = _sanitize_runtime_number(
            value,
            integer=True,
            maximum=1,
        )
        return sanitized
    if field == "elapsed_seconds":
        return _sanitize_runtime_number(
            value,
            integer=False,
            maximum=86_400.0,
        )
    if field in {
        "cuda_memory_allocated_mb",
        "cuda_memory_reserved_mb",
        "page_peak_vram_allocated_mb",
        "page_peak_vram_reserved_mb",
    }:
        return _sanitize_runtime_number(
            value,
            integer=False,
            maximum=1_048_576.0,
        )
    if field == "mask_bbox":
        return _sanitize_runtime_bbox(value, nullable=True)
    if field == "oom_retry_roi":
        return _sanitize_runtime_bbox(value, nullable=True)
    if field == "session_providers":
        if not isinstance(value, (list, tuple)) or len(value) > 8:
            return _DROP_RUNTIME_VALUE
        providers = list(value)
        if not all(
            isinstance(provider, str) and provider in SAFE_RUNTIME_PROVIDERS
            for provider in providers
        ):
            return _DROP_RUNTIME_VALUE
        return providers
    return _DROP_RUNTIME_VALUE


def _sanitize_runtime_diagnostics(value: object) -> list[dict[str, object]]:
    if not isinstance(value, (list, tuple)):
        return []
    sanitized: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        safe_item: dict[str, object] = {}
        for field, raw_value in item.items():
            safe_value = _sanitize_runtime_value(str(field), raw_value)
            if safe_value is not _DROP_RUNTIME_VALUE:
                safe_item[str(field)] = safe_value
        if safe_item:
            sanitized.append(safe_item)
    return sanitized


def _processing_failure_record(
    *,
    corpus_id: str,
    page_id: str,
    exc: BaseException,
) -> dict[str, object]:
    cause_code: str | None = None
    if isinstance(exc, InpaintEvalManifestError):
        candidate_code = exc.code
    else:
        candidate_code = str(exc)
    if candidate_code in SAFE_PROCESSING_CAUSE_CODES:
        cause_code = candidate_code
    return {
        "corpus_id": corpus_id,
        "page_id": page_id,
        "error_code": "processing_failed",
        "cause_code": cause_code,
        "exception_type": type(exc).__name__,
        "traceback_summary": _sanitized_traceback_summary(exc),
    }


def _holdout_manifest_preflight_error(
    manifests: tuple[EvalManifest, ...],
) -> InpaintEvalManifestError | None:
    for manifest in manifests:
        if manifest.split_role not in SOURCE_REVIEW_FINALIZATION_ROLES:
            continue
        finalized = (
            manifest.parent_manifest_sha256 is not None
            and manifest.expected_edit_basis == "source-only-review"
            and manifest.expected_edit_decisions_sha256 is not None
            and all(page.expected_edit != "optional" for page in manifest.pages)
        )
        if not finalized:
            return InpaintEvalManifestError(
                "manifest_holdout_not_source_review_finalized",
                corpus_id=manifest.corpus_id,
            )
    return None


def main() -> int:
    args = _build_argument_parser().parse_args()

    root_output, artifact_run = select_managed_output_directory(
        family="inpaint-debug-export",
        category="40-inpaint-mask-render",
        explicit_output_directory=args.output_dir,
    )
    try:
        root_output.mkdir(parents=True, exist_ok=True)
        if next(root_output.iterdir(), None) is not None:
            output_error = InpaintEvalManifestError(
                "inpaint_output_directory_not_empty"
            )
            if artifact_run is not None:
                artifact_run.fail(output_error)
            print(output_error.code, file=sys.stderr)
            return 1
        manifest_mode = bool(args.manifest)
        manifest_error: InpaintEvalManifestError | None = None
        manifests = ()
        if args.input and args.manifest:
            manifest_error = InpaintEvalManifestError(
                "manifest_and_direct_input_conflict"
            )
        elif args.blind_review_duplicate_count < 0:
            manifest_error = InpaintEvalManifestError(
                "blind_duplicate_count_invalid"
            )
        elif manifest_mode:
            try:
                manifests = load_eval_manifests(args.manifest)
                manifest_error = _holdout_manifest_preflight_error(manifests)
            except InpaintEvalManifestError as exc:
                manifest_error = exc
        if manifest_error is not None:
            failure_summary = {
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "input_mode": "manifest" if manifest_mode else "direct",
                "image_count": 0,
                "success_count": 0,
                "failure_count": 1,
                "failures": [manifest_error.as_record()],
                "traceback_summary": manifest_error.as_record(),
                "required_gate_failure_count": 1,
                "required_gate_failures": [manifest_error.code],
            }
            metrics_dir = root_output / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "summary.json").write_text(
                json.dumps(failure_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if artifact_run is not None:
                artifact_run.fail(manifest_error)
            print(root_output)
            return 1

        if manifest_mode:
            corpus_inputs: dict[
                str,
                list[tuple[Path, EvalPageSpec | None, str]],
            ] = {
                manifest.corpus_id: [
                    (page.source.path, page, page.page_id)
                    for page in manifest.pages
                ]
                for manifest in manifests
            }
        elif args.input:
            corpus_inputs = {
                "private": [
                    (path.expanduser(), None, f"direct-{index:03d}")
                    for index, path in enumerate(args.input, start=1)
                ]
            }
        else:
            selected_corpora = (
                ("japan", "China")
                if args.corpus == "all"
                else (("japan",) if args.corpus == "japan" else ("China",))
            )
            corpus_inputs = {
                corpus_name: [
                    (
                        path,
                        None,
                        f"sample-{corpus_name.lower()}-{index:03d}",
                    )
                    for index, path in enumerate(
                        _iter_sample_images(
                            ROOT / "Sample" / corpus_name,
                            args.glob,
                        ),
                        start=1,
                    )
                ]
                for corpus_name in selected_corpora
            }
        selected_count = sum(len(items) for items in corpus_inputs.values())
        blind_eligible_count = (
            sum(
                1
                for manifest in manifests
                for page in manifest.pages
                if page.baseline is not None
            )
            if manifest_mode
            else 0
        )
        if args.blind_review_duplicate_count > blind_eligible_count:
            duplicate_error = InpaintEvalManifestError(
                "blind_duplicate_count_out_of_range"
            )
            metrics_dir = root_output / "metrics"
            metrics_dir.mkdir(parents=True, exist_ok=True)
            (metrics_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "generated_at": datetime.now().isoformat(timespec="seconds"),
                        "input_mode": "manifest" if manifest_mode else "direct",
                        "image_count": selected_count,
                        "success_count": 0,
                        "failure_count": 1,
                        "failures": [duplicate_error.as_record()],
                        "traceback_summary": duplicate_error.as_record(),
                        "required_gate_failure_count": 1,
                        "required_gate_failures": [duplicate_error.code],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if artifact_run is not None:
                artifact_run.fail(duplicate_error)
            print(root_output)
            return 1

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
        panel_records: list[dict] = []
        total_images = 0
        for corpus_name, input_specs in corpus_inputs.items():
            corpus_output = root_output / corpus_name.lower()
            corpus_output.mkdir(parents=True, exist_ok=True)
            records: list[dict] = []
            for image_path, page_spec, public_page_id in input_specs:
                total_images += 1
                try:
                    if page_spec is not None:
                        verify_eval_page_spec(page_spec)
                    if not image_path.is_file():
                        raise FileNotFoundError("inpaint_input_image_missing")
                    page_runtime_device = str(
                        runtime_report.get("actual_device", "") or ""
                    )
                    peak_vram_reset_succeeded = _reset_cuda_peak_metrics(
                        page_runtime_device
                    )
                    record = _process_image(
                        image_path,
                        corpus_output,
                        detector,
                        inpainter,
                        settings,
                        auto_max_font_profile=args.auto_max_font_profile,
                        page_spec=page_spec,
                        public_corpus_id=corpus_name.lower(),
                        public_page_id=public_page_id,
                        runtime_device=page_runtime_device,
                        peak_vram_reset_succeeded=(
                            peak_vram_reset_succeeded
                        ),
                    )
                    record["corpus_id"] = corpus_name.lower()
                    panel_record = write_comparison_and_blind_panels(
                        root_output=root_output,
                        corpus_id=corpus_name.lower(),
                        page_id=str(record["page_id"]),
                        source_path=Path(record["source"]),
                        baseline_path=(
                            page_spec.baseline.path
                            if page_spec is not None and page_spec.baseline is not None
                            else None
                        ),
                        baseline_sha256=(
                            page_spec.baseline.sha256
                            if page_spec is not None and page_spec.baseline is not None
                            else None
                        ),
                        candidate_path=Path(record["cleaned"]),
                        final_mask_path=Path(record["final_mask"]),
                    )
                    record["comparison_panel"] = panel_record["comparison_panel"]
                    if panel_record["blind_eligible"]:
                        panel_records.append(panel_record)
                    records.append(record)
                except Exception as exc:
                    failures.append(
                        _processing_failure_record(
                            corpus_id=corpus_name.lower(),
                            page_id=public_page_id,
                            exc=exc,
                        )
                    )
            records_by_corpus[corpus_name.lower()] = records

        all_records = [
            record
            for records in records_by_corpus.values()
            for record in records
        ]
        erase_mode_distribution: Counter[str] = Counter()
        erase_skipped_reason_distribution: Counter[str] = Counter()
        block_timings: list[float] = []
        for record in all_records:
            erase_mode_distribution.update(record.get("erase_mode_distribution", {}))
            erase_skipped_reason_distribution.update(
                record.get("erase_skipped_reason_distribution", {})
            )
            block_timings.extend(
                float(item.get("elapsed_seconds", 0.0) or 0.0)
                for item in record.get("block_runtime_seconds", [])
                if float(item.get("elapsed_seconds", 0.0) or 0.0) > 0.0
            )
        aggregate_residue_score_sum = sum(
            float(record.get("residue_score_sum", 0.0) or 0.0)
            for record in all_records
        )
        aggregate_residue_source_count = sum(
            int(record.get("residue_source_contrast_pixel_count", 0) or 0)
            for record in all_records
        )
        baseline_aggregate_residue_score_sum = sum(
            float(record.get("baseline_residue_score_sum", 0.0) or 0.0)
            for record in all_records
        )
        baseline_aggregate_residue_source_count = sum(
            int(
                record.get(
                    "baseline_residue_source_contrast_pixel_count",
                    0,
                )
                or 0
            )
            for record in all_records
        )
        summary = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "input_mode": "manifest" if manifest_mode else ("direct" if args.input else "sample"),
            "detector_key": settings.get_tool_selection("detector"),
            "inpainter": args.inpainter,
            "hd_strategy": str(get_config(settings).hd_strategy),
            "auto_max_font_profile": args.auto_max_font_profile,
            "use_gpu": bool(args.use_gpu),
            "corpus": "manifest" if manifest_mode else args.corpus,
            "glob": (
                None
                if manifest_mode or args.input
                else ("*" if args.glob == "*" else "<redacted>")
            ),
            "manifest_corpora": (
                {
                    manifest.corpus_id: {
                        "schema_version": manifest.schema_version,
                        "expected_count": manifest.expected_count,
                        "manifest_sha256": manifest.manifest_sha256,
                        "split_role": manifest.split_role,
                        "source_lock_git_sha": manifest.source_lock_git_sha,
                        "parent_manifest_sha256": (
                            manifest.parent_manifest_sha256
                        ),
                        "expected_edit_basis": manifest.expected_edit_basis,
                        "expected_edit_decisions_sha256": (
                            manifest.expected_edit_decisions_sha256
                        ),
                    }
                    for manifest in manifests
                }
                if manifest_mode
                else {}
            ),
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
            "expected_edit_required_count": sum(
                1 for record in all_records if record["expected_edit"] == "required"
            ),
            "expected_edit_none_count": sum(
                1 for record in all_records if record["expected_edit"] == "none"
            ),
            "expected_edit_optional_count": sum(
                1 for record in all_records if record["expected_edit"] == "optional"
            ),
            "expected_edit_active_count": sum(
                1 for record in all_records if record["expected_edit"] != "none"
            ),
            "required_zero_block_count": sum(
                1
                for record in all_records
                if record["expected_edit"] == "required" and record["block_count"] <= 0
            ),
            "required_empty_final_mask_count": sum(
                1
                for record in all_records
                if record["expected_edit"] == "required"
                and record["final_mask_pixel_count"] <= 0
            ),
            "unexpected_none_edit_count": sum(
                1
                for record in all_records
                if record["expected_edit"] == "none"
                and record["final_mask_pixel_count"] > 0
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
            "protected_structure_changed_pixel_count_exact": sum(
                int(record["protected_structure_changed_pixel_count_exact"])
                for record in all_records
            ),
            "protected_structure_annotation_available_count": sum(
                1
                for record in all_records
                if bool(record.get("protected_structure_annotation_available", False))
            ),
            "protected_structure_annotation_changed_pixel_count_exact": sum(
                int(
                    record.get(
                        "protected_structure_annotation_changed_pixel_count_exact",
                        0,
                    )
                    or 0
                )
                for record in all_records
            ),
            "ambiguous_structure_annotation_available_count": sum(
                1
                for record in all_records
                if bool(record.get("ambiguous_structure_annotation_available", False))
            ),
            "ambiguous_structure_changed_pixel_count_exact": sum(
                int(
                    record.get(
                        "ambiguous_structure_changed_pixel_count_exact",
                        0,
                    )
                    or 0
                )
                for record in all_records
            ),
            "derived_protected_structure_changed_pixel_count_exact": sum(
                int(
                    record.get(
                        "derived_protected_structure_changed_pixel_count_exact",
                        0,
                    )
                    or 0
                )
                for record in all_records
            ),
            "routing_structure_changed_pixel_count_exact": sum(
                int(
                    record.get(
                        "routing_structure_changed_pixel_count_exact",
                        0,
                    )
                    or 0
                )
                for record in all_records
            ),
            "routing_source_owned_pixel_count": sum(
                int(record.get("routing_source_owned_pixel_count", 0) or 0)
                for record in all_records
            ),
            "routing_source_raw_owned_pixel_count": sum(
                int(record.get("routing_source_raw_owned_pixel_count", 0) or 0)
                for record in all_records
            ),
            "routing_ownership_protect_pixel_count": sum(
                int(record.get("routing_ownership_protect_pixel_count", 0) or 0)
                for record in all_records
            ),
            "routing_positive_claim_pixel_count": sum(
                int(record.get("routing_positive_claim_pixel_count", 0) or 0)
                for record in all_records
            ),
            "routing_positive_edit_pixel_count": sum(
                int(record.get("routing_positive_edit_pixel_count", 0) or 0)
                for record in all_records
            ),
            "positive_claim_raw_pixel_count": sum(
                int(record.get("positive_claim_raw_pixel_count", 0) or 0)
                for record in all_records
            ),
            "positive_claim_runtime_status_distribution": dict(
                sorted(
                    Counter(
                        str(
                            (record.get("positive_claim_runtime") or {}).get(
                                "status",
                                "",
                            )
                            or "unknown"
                        )
                        for record in all_records
                    ).items()
                )
            ),
            "routing_claim_providers": sorted(
                {
                    str(provider)
                    for record in all_records
                    for provider in (record.get("routing_claim_providers") or [])
                    if str(provider)
                }
            ),
            "residue_pixel_count": sum(
                int(record["residue_pixel_count"])
                for record in all_records
            ),
            "residue_source_contrast_pixel_count": sum(
                int(record["residue_source_contrast_pixel_count"])
                for record in all_records
            ),
            "residue_target_minimum_component_coverage": min(
                (
                    float(record["residue_target_minimum_component_coverage"])
                    for record in all_records
                    if record.get("residue_target_minimum_component_coverage")
                    is not None
                ),
                default=None,
            ),
            "aggregate_residue_score": (
                aggregate_residue_score_sum / aggregate_residue_source_count
                if aggregate_residue_source_count > 0
                else None
            ),
            "baseline_aggregate_residue_score": (
                baseline_aggregate_residue_score_sum
                / baseline_aggregate_residue_source_count
                if baseline_aggregate_residue_source_count > 0
                else None
            ),
            "aggregate_residue_score_delta_from_baseline": (
                (
                    aggregate_residue_score_sum
                    / aggregate_residue_source_count
                )
                - (
                    baseline_aggregate_residue_score_sum
                    / baseline_aggregate_residue_source_count
                )
                if aggregate_residue_source_count > 0
                and baseline_aggregate_residue_source_count > 0
                else None
            ),
            "residue_pass_truncated_block_count": sum(
                int(record["residue_pass_truncated_block_count"])
                for record in all_records
            ),
            "erase_mode_distribution": dict(sorted(erase_mode_distribution.items())),
            "erase_skipped_reason_distribution": dict(
                sorted(erase_skipped_reason_distribution.items())
            ),
            "required_skipped_block_count": sum(
                sum(
                    int(
                        dict(
                            record.get(
                                "erase_skipped_reason_distribution"
                            )
                            or {}
                        ).get(reason, 0)
                        or 0
                    )
                    for reason in REQUIRED_ERASE_SKIP_REASONS
                )
                for record in all_records
                if record["expected_edit"] == "required"
            ),
            "page_processing_seconds_total": sum(
                float(record["pipeline_elapsed_seconds"])
                for record in all_records
            ),
            "page_processing_seconds_p95": _percentile(
                [float(record["pipeline_elapsed_seconds"]) for record in all_records],
                95,
            ),
            "block_processing_seconds_p95": _percentile(block_timings, 95),
            "peak_vram_allocated_mb": max(
                (float(record["peak_vram_allocated_mb"]) for record in all_records),
                default=0.0,
            ),
            "peak_vram_reserved_mb": max(
                (float(record["peak_vram_reserved_mb"]) for record in all_records),
                default=0.0,
            ),
            "peak_vram_unavailable_count": sum(
                1
                for record in all_records
                if not bool(record.get("peak_vram_metrics_available", False))
            ),
            "peak_vram_reset_failure_count": sum(
                1
                for record in all_records
                if not bool(record.get("peak_vram_reset_succeeded", False))
            ),
            "cuda_memory_diagnostics_unavailable_count": sum(
                1
                for record in all_records
                for item in record.get("inpaint_runtime_diagnostics", [])
                if bool(item.get("is_inference", True))
                and (
                    item.get("cuda_memory_diagnostics_available") is not True
                    or bool(
                        item.get(
                            "cuda_memory_diagnostics_unavailable",
                            False,
                        )
                    )
                )
            ),
            "oom_retry_count": sum(
                int(item.get("oom_retry_count", 0) or 0)
                for record in all_records
                for item in record.get("inpaint_runtime_diagnostics", [])
            ),
            "baseline_cleaned_mismatch_count": sum(
                1
                for record in all_records
                if record.get("cleaned_matches_baseline_sha256") is False
            ),
            "baseline_final_mask_mismatch_count": sum(
                1
                for record in all_records
                if record.get("final_mask_matches_baseline_sha256") is False
            ),
            "baseline_cleaned_pixel_mismatch_count": sum(
                1
                for record in all_records
                if record.get("cleaned_matches_baseline_pixel_sha256") is False
            ),
            "baseline_final_mask_pixel_mismatch_count": sum(
                1
                for record in all_records
                if record.get("final_mask_matches_baseline_pixel_sha256")
                is False
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
            require_quality_gates=bool(args.require_quality_gates),
            require_baseline_parity=bool(args.require_baseline_parity),
        )
        if args.blind_review_duplicate_count > len(panel_records):
            gate_failures.append(
                "blind_review_duplicate_shortfall:"
                f"{len(panel_records)}<{args.blind_review_duplicate_count}"
            )
        review_path = None
        key_path = None
        if panel_records and args.blind_review_duplicate_count <= len(panel_records):
            review_path, key_path = write_blind_review_jsonl(
                root_output,
                panel_records,
                duplicate_count=args.blind_review_duplicate_count,
                assignment_seed=derive_blind_review_seed(
                    manifests,
                    (
                        str(record["candidate_sha256"])
                        for record in panel_records
                    ),
                ),
            )
        summary["required_gate_failure_count"] = len(gate_failures)
        summary["required_gate_failures"] = gate_failures
        summary["blind_review_page_count"] = len(panel_records)
        summary["blind_review_path"] = (
            review_path.relative_to(root_output).as_posix()
            if review_path is not None
            else None
        )
        summary["blind_key_path"] = (
            key_path.relative_to(root_output).as_posix()
            if key_path is not None
            else None
        )
        metrics_dir = root_output / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_page_metrics_jsonl(root_output, records_by_corpus)
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
            artifact_metadata = {
                "input_image_count": total_images,
                "success_count": summary["success_count"],
                "failure_count": summary["failure_count"],
                "required_gate_failure_count": len(gate_failures),
                "inpainter": args.inpainter,
                "use_gpu": bool(args.use_gpu),
            }
            if failures or gate_failures:
                artifact_run.fail(
                    RuntimeError("inpaint_evaluation_gate_failed"),
                    metadata=artifact_metadata,
                )
            else:
                artifact_run.complete(metadata=artifact_metadata)
        print(root_output)
        return 1 if failures or gate_failures else 0
    except BaseException as exc:
        if artifact_run is not None:
            artifact_run.fail(exc)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
