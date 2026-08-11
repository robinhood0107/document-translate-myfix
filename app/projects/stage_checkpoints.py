from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import copy
import hashlib
import json
import logging
import os
import platform
import re
import zlib

import msgpack
import numpy as np
import onnxruntime as ort

from app.projects.checkpoint_store import (
    ProjectCheckpointError,
    ProjectCheckpointStore,
    checkpoint_reference_for_save,
    normalize_checkpoint_reference,
)
from app.projects.parsers import ProjectDecoder, ProjectEncoder
from app.projects.project_types import PROJECT_KIND_SINGLE
from modules.ocr.persistent_cache import (
    apply_raw_ocr_result,
    canonical_sha256,
    validate_raw_ocr_result,
)
from modules.ocr.common.result_contract import (
    OCR_PROCESSING_CONTRACT_SCHEMA_VERSION,
)
from modules.utils.device import get_providers, resolve_device
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock
from modules.inpainting.runtime_contract import (
    INPAINT_RETRY_POLICY_VERSION,
    INPAINT_RUNTIME_CONTRACT_VERSION,
)


logger = logging.getLogger(__name__)


PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_TRANSLATION_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION = 4
PROJECT_RENDER_CHECKPOINT_SCHEMA_VERSION = 3
DETECTION_PREPROCESS_SCHEMA_VERSION = "rtdetr-v2-rgb-640-f32-v1"
DETECTION_POSTPROCESS_SCHEMA_VERSION = (
    "comic-text-bubble-blocks-detector-provenance-v2"
)
DETECTION_SORT_SCHEMA_VERSION = "sort-blk-list-v1"
DETECTION_MASK_SCHEMA_VERSION = "precomputed-mask-details-v1"
DETECTION_RENDER_AREA_SCHEMA_VERSION = "detected-bubble-render-area-v2"
DETECTION_FONT_SCHEMA_VERSION = "font-onnx-512-cv-color-v1"
OCR_POSTPROCESS_SCHEMA_VERSION = (
    "text-first-exact-canonical-quality-retry-drop-guards-v2"
)
TRANSLATION_STATE_SCHEMA_VERSION = "ctpr-block-translation-state-v1"
INPAINT_INPUT_SCHEMA_VERSION = (
    "semantic-action-mask-deterministic-ordered-input-brush-v5"
)
INPAINT_CLEANUP_SCHEMA_VERSION = "bubble-residue-duplicate-fill-cuda-v3"
INPAINT_ARTIFACT_SCHEMA_VERSION = "lossless-zlib-array-v2"
INPAINT_BLOCK_STATE_SCHEMA_VERSION = "inpaint-block-state-v1"
RENDER_INPUT_SCHEMA_VERSION = "translation-inpaint-style-layout-v1"
RENDER_SANITIZER_SCHEMA_VERSION = "strict-symbol-rich-text-v1"
RENDER_OUTPUT_SCHEMA_VERSION = "encoded-output-object-v1"
_RENDER_EXPORT_IDENTITY_KEYS = frozenset(
    {
        "resolved_automatic_output_target",
        "resolved_automatic_output_image_format",
        "resolved_automatic_output_archive_format",
        "resolved_automatic_output_archive_image_format",
        "resolved_automatic_output_archive_compression_level",
    }
)

_DETECTION_OBJECT_ROLE = "detection-result"
_OCR_OBJECT_ROLE = "ocr-raw-result"
_INPAINT_CLEANED_OBJECT_ROLE = "inpaint-cleaned-image"
_INPAINT_RAW_MASK_OBJECT_ROLE = "inpaint-raw-mask"
_INPAINT_FINAL_MASK_OBJECT_ROLE = "inpaint-final-mask"
_INPAINT_BLOCK_STATE_OBJECT_ROLE = "inpaint-block-state"
_RENDER_OUTPUT_OBJECT_ROLE = "render-output"
_FILE_SHA256_CACHE: dict[tuple[str, int, int, int], str] = {}

_INPAINT_BLOCK_STATE_FIELDS = (
    "_hard_box_applied",
    "_hard_box_reason_codes",
    "_legacy_fill_ratio",
    "_rescue_fill_ratio",
    "_hard_box_rescue_roi_xyxy",
    "_hard_box_index",
    "_hard_box_metrics",
    "_legacy_mask_pixel_count",
    "_rescue_mask_pixel_count",
    "_final_mask_pixel_count",
    "block_final_mask_pixel_count",
    "block_mask_iou",
    "block_mask_span_coverage",
    "block_mask_bbox",
    "block_mask_source",
    "block_mask_decision",
    "bubble_panel_mask_pixel_count",
    "bubble_panel_mask_source",
    "_erase_mode",
    "_erase_edit_pixel_count",
    "_erase_protect_pixel_count",
    "_erase_skipped_reason",
    "_mask_policy",
    "mask_decision",
    "mask_reject_reason",
    "mask_strategy",
    "mask_strategy_reason",
    "mask_actual_bbox",
    "mask_actual_pixel_count",
)

_DETECTION_BLOCK_FIELDS = (
    "block_id",
    "xyxy",
    "segm_pts",
    "bubble_xyxy",
    "ctd_roi_xyxy",
    "cleanup_roi_xyxy",
    "mask_roi_xyxy",
    "text_class",
    "detector_origin",
    "detector_text_bbox",
    "detector_provider",
    "angle",
    "tr_origin_point",
    "lines",
    "inpaint_bboxes",
    "line_spacing",
    "alignment",
    "min_font_size",
    "max_font_size",
    "font_color",
    "direction",
    "_render_original_xyxy",
    "_render_bubble_xyxy",
    "_render_area_source",
    "_render_area_xyxy",
)


@dataclass(frozen=True)
class DetectionCheckpointResult:
    blocks: list[TextBlock]
    precomputed_mask_details: dict[str, Any] | None


@dataclass(frozen=True)
class OCRCheckpointResult:
    blocks: list[TextBlock]
    attempt_count: int
    engine_name: str
    page_profile: dict[str, Any]
    raw_results: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class TranslationCheckpointResult:
    translations: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class InpaintCheckpointResult:
    cleaned_image: np.ndarray
    raw_mask: np.ndarray
    final_mask: np.ndarray
    cleanup_stats: dict[str, Any]
    cleaned_object_sha256: str
    cleaned_decoded_sha256: str
    block_states: list[dict[str, Any]]


@dataclass(frozen=True)
class RenderCheckpointResult:
    output_path: str
    output_root: str
    output_sha256: str
    output_bytes: bytes | None
    output_exists: bool


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return {
            "dtype": str(value.dtype),
            "shape": [int(item) for item in value.shape],
            "data": value.tolist(),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def decoded_image_sha256(image: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode("ascii", errors="strict"))
    digest.update(b"\0")
    digest.update(
        ",".join(str(int(item)) for item in contiguous.shape).encode(
            "ascii",
            errors="strict",
        )
    )
    digest.update(b"\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def project_checkpoint_page_key(main_page: Any, image_path: str) -> str:
    image_files = list(getattr(main_page, "image_files", []) or [])
    try:
        page_index = image_files.index(image_path)
    except ValueError:
        normalized = os.path.normcase(os.path.abspath(str(image_path)))
        return f"path:{hashlib.sha256(normalized.encode('utf-8')).hexdigest()}"
    return f"page:{page_index:08d}"


def _project_checkpoint_enabled(main_page: Any) -> bool:
    settings_page = getattr(main_page, "settings_page", None)
    getter = getattr(settings_page, "get_project_checkpoint_settings", None)
    if callable(getter):
        try:
            return bool((getter() or {}).get("enabled", False))
        except Exception:
            logger.debug(
                "Failed to read project checkpoint settings; disabling this run.",
                exc_info=True,
            )
            return False
    checkbox = getattr(
        getattr(settings_page, "ui", None),
        "project_checkpoint_enabled_checkbox",
        None,
    )
    return bool(checkbox is not None and checkbox.isChecked())


def open_project_stage_checkpoint_store(
    main_page: Any,
    *,
    initialize: bool,
) -> ProjectCheckpointStore | None:
    if not _project_checkpoint_enabled(main_page):
        return None
    if (
        getattr(
            main_page,
            "project_checkpoint_reference_persisted",
            True,
        )
        is False
    ):
        logger.info(
            "Project stage checkpoints remain disabled until the project is "
            "saved with its checkpoint reference."
        )
        return None
    project_file = str(getattr(main_page, "project_file", "") or "")
    project_kind = str(
        getattr(main_page, "project_kind", PROJECT_KIND_SINGLE)
        or PROJECT_KIND_SINGLE
    )
    if not project_file or project_kind != PROJECT_KIND_SINGLE:
        return None
    try:
        reference = checkpoint_reference_for_save(
            getattr(main_page, "project_checkpoint_reference", None),
            project_file,
        )
        main_page.project_checkpoint_reference = reference.to_dict()
        store = ProjectCheckpointStore(
            project_file,
            reference,
            enabled=True,
        )
        if initialize and not store.ensure_initialized():
            return None
        return store
    except (OSError, ProjectCheckpointError, TypeError, ValueError) as exc:
        logger.warning(
            "Project stage checkpoints are unavailable for this run; processing "
            "will continue normally. reason=%s",
            exc,
        )
        return None


def invalidate_project_page_checkpoints(
    main_page: Any,
    image_path: str,
    *,
    stage: str = "ocr",
) -> int:
    """Invalidate one page without creating a new sidecar.

    Manual edits must also invalidate records while the feature checkbox is
    temporarily off; otherwise re-enabling it could revive stale output.
    """

    project_file = str(getattr(main_page, "project_file", "") or "")
    project_kind = str(
        getattr(main_page, "project_kind", PROJECT_KIND_SINGLE)
        or PROJECT_KIND_SINGLE
    )
    if not project_file or project_kind != PROJECT_KIND_SINGLE:
        return 0
    try:
        reference = normalize_checkpoint_reference(
            getattr(main_page, "project_checkpoint_reference", None),
            project_file,
            create_if_missing=False,
        )
        if reference is None:
            return 0
        store = ProjectCheckpointStore(
            project_file,
            reference,
            enabled=True,
            timeout_sec=0.05,
        )
        if not store.db_path.is_file():
            return 0
        return store.invalidate(
            page_key=project_checkpoint_page_key(main_page, image_path),
            stage=stage,
        )
    except (OSError, ProjectCheckpointError, TypeError, ValueError):
        logger.warning(
            "Failed to invalidate project checkpoints for %s; existing project "
            "data was preserved.",
            os.path.basename(str(image_path)),
            exc_info=True,
        )
        return 0


def invalidate_current_project_page_checkpoints(
    main_page: Any,
    *,
    stage: str = "ocr",
) -> int:
    image_files = list(getattr(main_page, "image_files", []) or [])
    if not image_files:
        return 0
    try:
        index = int(getattr(main_page, "curr_img_idx", 0) or 0)
    except (TypeError, ValueError):
        index = 0
    if index < 0 or index >= len(image_files):
        return 0
    return invalidate_project_page_checkpoints(
        main_page,
        image_files[index],
        stage=stage,
    )


def _registered_model_sha256(model_id: ModelID) -> str:
    spec = ModelDownloader.registry.get(model_id)
    if spec is None or not spec.sha256:
        return ""
    value = str(spec.sha256[0] or "").strip().lower()
    return value if len(value) == 64 else ""


def build_detection_identity(
    settings_page: Any,
    *,
    source_lang_english: str,
) -> dict[str, Any] | None:
    detector = str(
        settings_page.get_tool_selection("detector") or "RT-DETR-v2"
    )
    if detector != "RT-DETR-v2":
        return None
    device = resolve_device(settings_page.is_gpu_enabled(), backend="onnx")
    return {
        "schema_version": PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION,
        "detector": detector,
        "engine": "RTDetrV2ONNXDetection",
        "backend": "onnx",
        "device": device,
        "runtime": {
            "onnxruntime_version": str(getattr(ort, "__version__", "")),
            "available_providers": list(ort.get_available_providers()),
            "selected_providers": _json_safe(get_providers(device)),
        },
        "model": {
            "id": ModelID.RTDETR_V2_ONNX.value,
            "sha256": _registered_model_sha256(ModelID.RTDETR_V2_ONNX),
        },
        "font_model": {
            "id": ModelID.FONT_DETECTOR_ONNX.value,
            "sha256": _registered_model_sha256(ModelID.FONT_DETECTOR_ONNX),
        },
        "confidence_threshold": 0.3,
        "slicer": {
            "height_to_width_ratio_threshold": 3.5,
            "target_slice_ratio": 3.0,
            "overlap_height_ratio": 0.2,
            "min_slice_height_ratio": 0.7,
            "merge_iou_threshold": 0.2,
            "duplicate_iou_threshold": 0.5,
            "merge_y_distance_threshold": 0.1,
            "containment_threshold": 0.85,
        },
        "preprocess_schema": DETECTION_PREPROCESS_SCHEMA_VERSION,
        "postprocess_schema": DETECTION_POSTPROCESS_SCHEMA_VERSION,
        "font_schema": DETECTION_FONT_SCHEMA_VERSION,
        "render_area_schema": DETECTION_RENDER_AREA_SCHEMA_VERSION,
        "sort_schema": DETECTION_SORT_SCHEMA_VERSION,
        "right_to_left": source_lang_english == "Japanese",
        "mask_schema": DETECTION_MASK_SCHEMA_VERSION,
    }


def build_detection_fingerprint(
    *,
    source_sha256: str,
    identity: Mapping[str, Any],
) -> str:
    return canonical_sha256(
        {
            "stage": "detection",
            "source_decoded_content_sha256": source_sha256,
            "identity": dict(identity),
        }
    )


def detection_block_record(block: TextBlock) -> dict[str, Any]:
    return {
        field_name: copy.deepcopy(getattr(block, field_name))
        for field_name in _DETECTION_BLOCK_FIELDS
        if hasattr(block, field_name)
    }


def detection_structure_signature(blocks: list[TextBlock]) -> str:
    return canonical_sha256(
        [detection_block_record(block) for block in blocks]
    )


def _mask_signature(value: Any) -> str:
    return canonical_sha256(
        {
            "mask_schema": DETECTION_MASK_SCHEMA_VERSION,
            "value": value,
        }
    )


def _stringify_mapping_keys(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _stringify_mapping_keys(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_stringify_mapping_keys(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_stringify_mapping_keys(item) for item in value)
    return value


def _pack(value: Any) -> bytes:
    return msgpack.packb(
        _stringify_mapping_keys(value),
        default=ProjectEncoder().encode,
        use_bin_type=True,
    )


def _unpack(payload: bytes) -> Any:
    return msgpack.unpackb(
        payload,
        object_hook=ProjectDecoder().decode,
        raw=False,
        strict_map_key=True,
    )


def _block_from_detection_record(value: Any) -> TextBlock:
    if not isinstance(value, Mapping):
        raise ValueError("Detection checkpoint block must be an object.")
    unexpected = set(value) - set(_DETECTION_BLOCK_FIELDS)
    if unexpected:
        raise ValueError("Detection checkpoint block has unexpected fields.")
    block_id = str(value.get("block_id", "") or "")
    if not block_id:
        raise ValueError("Detection checkpoint block is missing its stable ID.")
    block = TextBlock(block_id=block_id)
    for field_name, field_value in value.items():
        setattr(block, field_name, copy.deepcopy(field_value))
    coordinates = np.asarray(getattr(block, "xyxy", None))
    if coordinates.shape != (4,) or not np.isfinite(coordinates).all():
        raise ValueError("Detection checkpoint block has invalid coordinates.")
    if not isinstance(getattr(block, "text_class", ""), str):
        raise ValueError("Detection checkpoint block has an invalid class.")
    return block


def record_detection_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    source_sha256: str,
    identity: Mapping[str, Any],
    blocks: list[TextBlock],
    precomputed_mask_details: dict[str, Any] | None,
) -> bool:
    if store is None or not store.available:
        return False
    block_signature = detection_structure_signature(blocks)
    mask_signature = _mask_signature(precomputed_mask_details)
    blob = _pack(
        {
            "schema_version": PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION,
            "blocks": [detection_block_record(block) for block in blocks],
            "precomputed_mask_details": copy.deepcopy(
                precomputed_mask_details
            ),
        }
    )
    object_hash = store.put_object(blob)
    if object_hash is None:
        return False
    return store.record_stage(
        page_key,
        "detection",
        fingerprint,
        payload={
            "schema_version": PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION,
            "source_decoded_content_sha256": source_sha256,
            "identity_sha256": canonical_sha256(dict(identity)),
            "block_count": len(blocks),
            "block_signature": block_signature,
            "mask_signature": mask_signature,
        },
        objects={_DETECTION_OBJECT_ROLE: object_hash},
    )


def lookup_detection_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    source_sha256: str,
    identity: Mapping[str, Any],
) -> DetectionCheckpointResult | None:
    if store is None:
        return None
    hit = store.lookup_stage(page_key, "detection", fingerprint)
    if hit is None:
        return None
    try:
        payload = dict(hit.payload)
        if (
            int(payload.get("schema_version", 0))
            != PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION
            or payload.get("source_decoded_content_sha256") != source_sha256
            or payload.get("identity_sha256")
            != canonical_sha256(dict(identity))
        ):
            return None
        object_hash = hit.objects.get(_DETECTION_OBJECT_ROLE)
        raw = store.read_object(object_hash) if object_hash else None
        if raw is None:
            return None
        decoded = _unpack(raw)
        if not isinstance(decoded, Mapping):
            raise ValueError("Detection checkpoint result must be an object.")
        if (
            int(decoded.get("schema_version", 0))
            != PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION
        ):
            return None
        raw_blocks = decoded.get("blocks")
        if not isinstance(raw_blocks, list):
            raise ValueError("Detection checkpoint blocks must be a list.")
        blocks = [_block_from_detection_record(item) for item in raw_blocks]
        block_ids = [str(block.block_id) for block in blocks]
        if len(set(block_ids)) != len(block_ids):
            raise ValueError("Detection checkpoint block IDs must be unique.")
        mask_details = decoded.get("precomputed_mask_details")
        if mask_details is not None and not isinstance(mask_details, dict):
            raise ValueError("Detection checkpoint mask details must be an object.")
        if (
            int(payload.get("block_count", -1)) != len(blocks)
            or payload.get("block_signature")
            != detection_structure_signature(blocks)
            or payload.get("mask_signature") != _mask_signature(mask_details)
        ):
            raise ValueError("Detection checkpoint integrity metadata does not match.")
        return DetectionCheckpointResult(
            blocks=blocks,
            precomputed_mask_details=copy.deepcopy(mask_details),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        msgpack.ExtraData,
        msgpack.FormatError,
        msgpack.StackError,
    ):
        logger.warning(
            "Invalid detection checkpoint ignored for %s; detection will be "
            "recomputed.",
            page_key,
            exc_info=True,
        )
        return None


def build_project_ocr_identity(
    *,
    detection_fingerprint: str,
    runtime_identity: Mapping[str, Any],
    policy: Mapping[str, Any],
    paddle_settings: Mapping[str, Any],
    source_lang_english: str,
) -> dict[str, Any]:
    identity = {
        "schema_version": PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION,
        "detection_fingerprint": detection_fingerprint,
        "runtime": dict(runtime_identity),
        "engine": str(policy.get("primary_ocr_engine", "")),
        "normalized_ocr_mode": str(policy.get("normalized_ocr_mode", "")),
        "source_language": source_lang_english,
        "max_new_tokens": int(paddle_settings.get("max_new_tokens", 0) or 0),
        "prettify_markdown": bool(
            paddle_settings.get("prettify_markdown", False)
        ),
        "visualize": bool(paddle_settings.get("visualize", False)),
        "postprocess_schema": OCR_POSTPROCESS_SCHEMA_VERSION,
    }
    if str(policy.get("primary_ocr_engine", "")) == (
        "PaddleOCR VL Spotting"
    ):
        identity["max_completion_tokens"] = int(
            paddle_settings.get("max_completion_tokens", 0) or 0
        )
        identity["request_timeout_sec"] = int(
            paddle_settings.get("request_timeout_sec", 0) or 0
        )
    return identity


def build_project_ocr_fingerprint(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "stage": "ocr",
            "identity": dict(identity),
        }
    )


def record_ocr_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    blocks: list[TextBlock],
    raw_results: Mapping[str, Mapping[str, Any]],
    attempt_count: int,
    engine_name: str,
    page_profile: Mapping[str, Any] | None,
) -> bool:
    if store is None or not store.available:
        return False
    retained_ids: list[str] = []
    ordered_raw: list[dict[str, Any]] = []
    for block in blocks:
        block_id = str(getattr(block, "block_id", "") or "")
        raw_result = raw_results.get(block_id)
        if not block_id or not isinstance(raw_result, Mapping):
            return False
        normalized = copy.deepcopy(dict(raw_result))
        validate_raw_ocr_result(normalized)
        retained_ids.append(block_id)
        ordered_raw.append(
            {
                "block_id": block_id,
                "result": normalized,
            }
        )
    blob_value = {
        "schema_version": PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION,
        "retained_block_ids": retained_ids,
        "raw_results": ordered_raw,
        "attempt_count": max(0, int(attempt_count)),
        "engine_name": str(engine_name or ""),
        "page_profile": _json_safe(dict(page_profile or {})),
    }
    blob = _pack(blob_value)
    object_hash = store.put_object(blob)
    if object_hash is None:
        return False
    return store.record_stage(
        page_key,
        "ocr",
        fingerprint,
        payload={
            "schema_version": PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": canonical_sha256(dict(identity)),
            "retained_block_ids": retained_ids,
            "raw_result_signature": canonical_sha256(ordered_raw),
        },
        objects={_OCR_OBJECT_ROLE: object_hash},
    )


def lookup_ocr_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    detection_blocks: list[TextBlock],
) -> OCRCheckpointResult | None:
    if store is None:
        return None
    hit = store.lookup_stage(page_key, "ocr", fingerprint)
    if hit is None:
        return None
    try:
        metadata = dict(hit.payload)
        if (
            int(metadata.get("schema_version", 0))
            != PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION
            or metadata.get("identity_sha256")
            != canonical_sha256(dict(identity))
        ):
            return None
        object_hash = hit.objects.get(_OCR_OBJECT_ROLE)
        raw = store.read_object(object_hash) if object_hash else None
        if raw is None:
            return None
        decoded = _unpack(raw)
        if not isinstance(decoded, Mapping):
            raise ValueError("OCR checkpoint result must be an object.")
        if (
            int(decoded.get("schema_version", 0))
            != PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION
        ):
            return None
        retained_ids = decoded.get("retained_block_ids")
        raw_items = decoded.get("raw_results")
        if not isinstance(retained_ids, list) or not isinstance(raw_items, list):
            raise ValueError("OCR checkpoint result lists are missing.")
        if metadata.get("retained_block_ids") != retained_ids:
            raise ValueError("OCR checkpoint retained block order does not match.")
        if metadata.get("raw_result_signature") != canonical_sha256(raw_items):
            raise ValueError("OCR checkpoint raw result signature does not match.")
        detection_by_id = {
            str(getattr(block, "block_id", "") or ""): block
            for block in detection_blocks
        }
        staged_results: list[tuple[str, TextBlock, dict[str, Any]]] = []
        if len(raw_items) != len(retained_ids):
            raise ValueError("OCR checkpoint result count does not match.")
        for expected_id, item in zip(retained_ids, raw_items):
            if not isinstance(item, Mapping):
                raise ValueError("OCR checkpoint raw result must be an object.")
            block_id = str(item.get("block_id", "") or "")
            result = item.get("result")
            if block_id != str(expected_id) or block_id not in detection_by_id:
                raise ValueError("OCR checkpoint block identity does not match detection.")
            if not isinstance(result, Mapping):
                raise ValueError("OCR checkpoint payload is invalid.")
            normalized = copy.deepcopy(dict(result))
            validate_raw_ocr_result(normalized)
            block = detection_by_id[block_id]
            staged_results.append((block_id, block, normalized))

        restored_blocks: list[TextBlock] = []
        raw_results: dict[str, dict[str, Any]] = {}
        for block_id, block, normalized in staged_results:
            apply_raw_ocr_result(block, normalized)
            restored_blocks.append(block)
            raw_results[block_id] = normalized
        page_profile = decoded.get("page_profile")
        if not isinstance(page_profile, Mapping):
            page_profile = {}
        return OCRCheckpointResult(
            blocks=restored_blocks,
            attempt_count=max(0, int(decoded.get("attempt_count", 0) or 0)),
            engine_name=str(decoded.get("engine_name", "") or ""),
            page_profile=copy.deepcopy(dict(page_profile)),
            raw_results=raw_results,
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        msgpack.ExtraData,
        msgpack.FormatError,
        msgpack.StackError,
    ):
        logger.warning(
            "Invalid OCR checkpoint ignored for %s; OCR will continue through "
            "the next cache layer.",
            page_key,
            exc_info=True,
        )
        return None


def _packed_sha256(value: Any) -> str:
    return hashlib.sha256(_pack(value)).hexdigest()


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_sha256_file(path: str | os.PathLike[str]) -> str:
    normalized = os.path.abspath(os.fspath(path))
    stat_result = os.stat(normalized)
    cache_key = (
        normalized,
        int(stat_result.st_size),
        int(stat_result.st_mtime_ns),
        int(stat_result.st_ctime_ns),
    )
    cached = _FILE_SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = _sha256_file(normalized)
    stale_keys = [
        key for key in _FILE_SHA256_CACHE if key[0] == normalized
    ]
    for key in stale_keys:
        _FILE_SHA256_CACHE.pop(key, None)
    _FILE_SHA256_CACHE[cache_key] = digest
    return digest


def _translation_snapshot(blocks: list[TextBlock]) -> list[dict[str, Any]]:
    snapshot: list[dict[str, Any]] = []
    for block in blocks:
        block_id = str(getattr(block, "block_id", "") or "")
        snapshot.append(
            {
                "block_id": block_id,
                "source_text": str(getattr(block, "text", "") or ""),
                "translation": copy.deepcopy(
                    getattr(block, "translation", "")
                ),
                "rich_text": copy.deepcopy(getattr(block, "rich_text", "")),
                "source_lang": str(
                    getattr(block, "source_lang", "") or ""
                ),
                "target_lang": str(
                    getattr(block, "target_lang", "") or ""
                ),
                "repetition_guard": copy.deepcopy(
                    getattr(block, "_translation_repetition_guard", None)
                ),
            }
        )
    return snapshot


def snapshot_project_translations(
    blocks: list[TextBlock] | None,
) -> list[dict[str, Any]]:
    """Capture .ctpr-owned translation state without writing it to sidecar."""

    return _translation_snapshot(list(blocks or []))


def translation_state_signature(
    snapshot_or_blocks: list[dict[str, Any]] | list[TextBlock],
) -> str:
    values = list(snapshot_or_blocks or [])
    if values and isinstance(values[0], TextBlock):
        values = _translation_snapshot(values)  # type: ignore[arg-type]
    return _packed_sha256(
        {
            "schema": TRANSLATION_STATE_SCHEMA_VERSION,
            "blocks": values,
        }
    )


def build_translation_identity(
    *,
    ocr_fingerprint: str,
    source_lang: str,
    target_lang: str,
    extra_context: str,
    translator_key: str,
    translator_engine: str,
    translator_settings: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    dictionary_fingerprint: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_TRANSLATION_CHECKPOINT_SCHEMA_VERSION,
        "ocr_fingerprint": str(ocr_fingerprint or ""),
        "source_lang": str(source_lang or ""),
        "target_lang": str(target_lang or ""),
        "extra_context": str(extra_context or ""),
        "translator_key": str(translator_key or ""),
        "translator_engine": str(translator_engine or ""),
        "translator_settings": _json_safe(dict(translator_settings)),
        "runtime": _json_safe(dict(runtime_identity)),
        "dictionary_fingerprint": str(dictionary_fingerprint or ""),
        "state_schema": TRANSLATION_STATE_SCHEMA_VERSION,
    }


def build_translation_fingerprint(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "stage": "translation",
            "identity": dict(identity),
        }
    )


def record_translation_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    blocks: list[TextBlock],
) -> bool:
    if store is None or not store.available:
        return False
    snapshot = _translation_snapshot(blocks)
    block_ids = [str(item.get("block_id", "") or "") for item in snapshot]
    translations = [item.get("translation") for item in snapshot]
    if (
        not snapshot
        or any(not block_id for block_id in block_ids)
        or len(set(block_ids)) != len(block_ids)
        or any(
            not isinstance(translation, str) or not translation.strip()
            for translation in translations
        )
    ):
        return False
    return store.record_stage(
        page_key,
        "translation",
        fingerprint,
        payload={
            "schema_version": PROJECT_TRANSLATION_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": canonical_sha256(dict(identity)),
            "block_ids": block_ids,
            "translation_state_signature": translation_state_signature(
                snapshot
            ),
        },
        objects={},
    )


def lookup_translation_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    current_blocks: list[TextBlock],
    project_snapshot: list[dict[str, Any]] | None,
) -> TranslationCheckpointResult | None:
    if store is None or not project_snapshot:
        return None
    hit = store.lookup_stage(page_key, "translation", fingerprint)
    if hit is None:
        return None
    try:
        payload = dict(hit.payload)
        if (
            int(payload.get("schema_version", 0))
            != PROJECT_TRANSLATION_CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity_sha256")
            != canonical_sha256(dict(identity))
            or hit.objects
        ):
            return None
        current_ids = [
            str(getattr(block, "block_id", "") or "")
            for block in current_blocks
        ]
        snapshot_ids = [
            str(item.get("block_id", "") or "")
            for item in project_snapshot
            if isinstance(item, Mapping)
        ]
        if (
            not current_ids
            or payload.get("block_ids") != current_ids
            or snapshot_ids != current_ids
            or len(set(current_ids)) != len(current_ids)
            or payload.get("translation_state_signature")
            != translation_state_signature(project_snapshot)
        ):
            return None

        restored: dict[str, dict[str, Any]] = {}
        for block, snapshot in zip(current_blocks, project_snapshot):
            if not isinstance(snapshot, Mapping):
                return None
            if str(snapshot.get("source_text", "") or "") != str(
                getattr(block, "text", "") or ""
            ):
                return None
            translation = snapshot.get("translation")
            if not isinstance(translation, str) or not translation.strip():
                return None
            restored[str(getattr(block, "block_id", "") or "")] = {
                "translation": translation,
                "rich_text": copy.deepcopy(snapshot.get("rich_text", "")),
                "source_lang": str(snapshot.get("source_lang", "") or ""),
                "target_lang": str(snapshot.get("target_lang", "") or ""),
                "repetition_guard": copy.deepcopy(
                    snapshot.get("repetition_guard")
                ),
            }
        return TranslationCheckpointResult(translations=restored)
    except (KeyError, TypeError, ValueError):
        logger.warning(
            "Invalid translation checkpoint ignored for %s; translation will "
            "be recomputed.",
            page_key,
            exc_info=True,
        )
        return None


def apply_translation_checkpoint(
    blocks: list[TextBlock],
    result: TranslationCheckpointResult,
) -> None:
    for block in blocks:
        block_id = str(getattr(block, "block_id", "") or "")
        state = result.translations.get(block_id)
        if state is None:
            raise ValueError("Translation checkpoint block is missing.")
        block.translation = str(state.get("translation", "") or "")
        block.rich_text = copy.deepcopy(state.get("rich_text", ""))
        block.source_lang = str(state.get("source_lang", "") or "")
        block.target_lang = str(state.get("target_lang", "") or "")
        guard = state.get("repetition_guard")
        if isinstance(guard, Mapping):
            setattr(block, "_translation_repetition_guard", dict(guard))
        elif hasattr(block, "_translation_repetition_guard"):
            delattr(block, "_translation_repetition_guard")


def registered_inpainter_model_identity(
    inpainter_key: str,
    backend: str,
) -> dict[str, Any]:
    key = str(inpainter_key or "")
    backend_name = str(backend or "")
    model_id: ModelID | None = None
    if key == "AOT":
        model_id = (
            ModelID.AOT_ONNX
            if backend_name.lower() == "onnx"
            else ModelID.AOT_TORCH
        )
    elif key == "lama_large_512px":
        model_id = ModelID.LAMA_LARGE_512PX
    elif key == "lama_mpe":
        model_id = ModelID.LAMA_MPE
    elif key == "MI-GAN":
        model_id = (
            ModelID.MIGAN_PIPELINE_ONNX
            if backend_name.lower() == "onnx"
            else ModelID.MIGAN_JIT
        )
    if model_id is None:
        return {
            "id": "",
            "files": [],
            "declared_digests": [],
            "file_identities": [],
        }
    spec = ModelDownloader.registry.get(model_id)
    if spec is None:
        return {
            "id": model_id.value,
            "files": [],
            "declared_digests": [],
            "file_identities": [],
        }
    file_identities: list[dict[str, str]] = []
    for index, remote_name in enumerate(spec.files or []):
        local_name = (
            spec.save_as.get(remote_name, remote_name)
            if spec.save_as
            else remote_name
        )
        declared = (
            str(spec.sha256[index] or "").strip().lower()
            if index < len(spec.sha256 or [])
            else ""
        )
        actual_sha256 = (
            declared
            if re.fullmatch(r"[0-9a-f]{64}", declared)
            else ""
        )
        if not actual_sha256:
            local_path = os.path.join(spec.save_dir, local_name)
            try:
                if os.path.isfile(local_path):
                    actual_sha256 = _cached_sha256_file(local_path)
            except OSError:
                actual_sha256 = ""
        file_identities.append(
            {
                "name": str(local_name),
                "declared_digest": declared,
                "sha256": actual_sha256,
            }
        )
    return {
        "id": model_id.value,
        "files": list(spec.files or []),
        "declared_digests": [
            str(value or "").lower() for value in (spec.sha256 or [])
        ],
        "file_identities": file_identities,
    }


def _block_inpaint_record(block: TextBlock) -> dict[str, Any]:
    return {
        **detection_block_record(block),
        "text": str(getattr(block, "text", "") or ""),
        "ocr_status": str(getattr(block, "ocr_status", "") or ""),
        "ocr_empty_reason": str(
            getattr(block, "ocr_empty_reason", "") or ""
        ),
        "semantic_role": str(
            getattr(block, "semantic_role", "") or ""
        ),
        "processing_action": str(
            getattr(block, "processing_action", "") or ""
        ),
        "compound_group_id": str(
            getattr(block, "compound_group_id", "") or ""
        ),
        "mask_strategy": str(
            getattr(block, "mask_strategy", "") or ""
        ),
    }


def snapshot_inpaint_block_state(
    blocks: list[TextBlock] | None,
) -> list[dict[str, Any]]:
    """Capture only mask/inpaint-owned block fields.

    Translation and user-authored render state are deliberately excluded so an
    inpaint hit cannot revive stale text or formatting.
    """

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for block in list(blocks or []):
        block_id = str(getattr(block, "block_id", "") or "")
        if not block_id or block_id in seen:
            raise ValueError(
                "Inpaint checkpoint block identities must be non-empty and "
                "unique."
            )
        seen.add(block_id)
        records.append(
            {
                "block_id": block_id,
                "attributes": {
                    field_name: copy.deepcopy(getattr(block, field_name))
                    for field_name in _INPAINT_BLOCK_STATE_FIELDS
                    if hasattr(block, field_name)
                },
            }
        )
    return records


def restore_inpaint_block_state(
    blocks: list[TextBlock],
    block_states: list[dict[str, Any]],
) -> None:
    """Restore validated mask/inpaint metadata without touching user text."""

    current = list(blocks or [])
    staged: list[tuple[TextBlock, dict[str, Any]]] = []
    if len(current) != len(block_states):
        raise ValueError("Inpaint checkpoint block state count does not match.")
    allowed = set(_INPAINT_BLOCK_STATE_FIELDS)
    for block, record in zip(current, block_states):
        if not isinstance(record, Mapping):
            raise ValueError("Inpaint checkpoint block state must be an object.")
        block_id = str(record.get("block_id", "") or "")
        if block_id != str(getattr(block, "block_id", "") or ""):
            raise ValueError("Inpaint checkpoint block order does not match.")
        attributes = record.get("attributes")
        if not isinstance(attributes, Mapping):
            raise ValueError(
                "Inpaint checkpoint block attributes must be an object."
            )
        unexpected = set(attributes) - allowed
        if unexpected:
            raise ValueError(
                "Inpaint checkpoint block attributes contain unexpected "
                "fields."
            )
        staged.append((block, copy.deepcopy(dict(attributes))))

    for block, attributes in staged:
        for field_name in _INPAINT_BLOCK_STATE_FIELDS:
            if field_name in attributes:
                setattr(block, field_name, attributes[field_name])
            elif hasattr(block, field_name):
                delattr(block, field_name)


def _brush_stroke_signature(strokes: list[dict[str, Any]] | None) -> str:
    return _packed_sha256(
        {
            "schema": "project-brush-strokes-v1",
            "strokes": copy.deepcopy(list(strokes or [])),
        }
    )


def build_inpaint_identity(
    *,
    source_sha256: str,
    detection_fingerprint: str,
    ocr_fingerprint: str,
    blocks: list[TextBlock],
    brush_strokes: list[dict[str, Any]] | None,
    runtime: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    hd_strategy: Mapping[str, Any],
    mask_settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION,
        "source_decoded_content_sha256": str(source_sha256 or ""),
        "detection_fingerprint": str(detection_fingerprint or ""),
        "ocr_fingerprint": str(ocr_fingerprint or ""),
        "ordered_blocks_sha256": canonical_sha256(
            [_block_inpaint_record(block) for block in blocks]
        ),
        "brush_strokes_sha256": _brush_stroke_signature(brush_strokes),
        "runtime": _json_safe(dict(runtime)),
        "model": _json_safe(dict(model_identity)),
        "hd_strategy": _json_safe(dict(hd_strategy)),
        "mask_settings": _json_safe(dict(mask_settings)),
        "positive_claim_model": {
            "id": ModelID.CTD_POSITIVE_CLAIM_ONNX.value,
            "sha256": _registered_model_sha256(
                ModelID.CTD_POSITIVE_CLAIM_ONNX
            ),
        },
        "ocr_processing_contract_schema": (
            OCR_PROCESSING_CONTRACT_SCHEMA_VERSION
        ),
        "input_schema": INPAINT_INPUT_SCHEMA_VERSION,
        "cleanup_schema": INPAINT_CLEANUP_SCHEMA_VERSION,
        "artifact_schema": INPAINT_ARTIFACT_SCHEMA_VERSION,
        "runtime_contract": INPAINT_RUNTIME_CONTRACT_VERSION,
        "retry_policy": INPAINT_RETRY_POLICY_VERSION,
    }


def build_inpaint_fingerprint(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "stage": "inpaint",
            "identity": dict(identity),
        }
    )


def build_skipped_stage_fingerprint(
    *,
    stage: str,
    source_sha256: str,
    detection_fingerprint: str,
    reason: str,
) -> str:
    """Identify a deterministic no-op stage without storing a fake artifact."""

    normalized_stage = str(stage or "").strip()
    if normalized_stage not in {"translation", "inpaint"}:
        raise ValueError("Only translation and inpaint stages may be skipped.")
    schema_version = (
        PROJECT_TRANSLATION_CHECKPOINT_SCHEMA_VERSION
        if normalized_stage == "translation"
        else PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION
    )
    return canonical_sha256(
        {
            "stage": normalized_stage,
            "schema_version": schema_version,
            "outcome": "skipped",
            "reason": str(reason or ""),
            "source_decoded_content_sha256": str(source_sha256 or ""),
            "detection_fingerprint": str(detection_fingerprint or ""),
        }
    )


def _pack_array_artifact(kind: str, value: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(value)
    raw = contiguous.tobytes(order="C")
    return _pack(
        {
            "schema_version": PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION,
            "artifact_schema": INPAINT_ARTIFACT_SCHEMA_VERSION,
            "kind": kind,
            "dtype": str(contiguous.dtype),
            "shape": [int(item) for item in contiguous.shape],
            "compression": "zlib",
            "data": zlib.compress(raw, level=3),
        }
    )


def _unpack_array_artifact(
    payload: bytes,
    *,
    kind: str,
    expected_shape: tuple[int, ...],
    mask: bool,
) -> np.ndarray:
    decoded = _unpack(payload)
    if not isinstance(decoded, Mapping):
        raise ValueError("Inpaint checkpoint artifact must be an object.")
    if (
        int(decoded.get("schema_version", 0))
        != PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION
        or decoded.get("artifact_schema") != INPAINT_ARTIFACT_SCHEMA_VERSION
        or decoded.get("kind") != kind
    ):
        raise ValueError("Inpaint checkpoint artifact schema does not match.")
    shape = tuple(int(item) for item in decoded.get("shape", []))
    if shape != expected_shape:
        raise ValueError("Inpaint checkpoint artifact shape does not match.")
    if decoded.get("dtype") != "uint8":
        raise ValueError("Inpaint checkpoint artifact dtype must be uint8.")
    if decoded.get("compression") != "zlib":
        raise ValueError("Inpaint checkpoint artifact compression does not match.")
    compressed = decoded.get("data")
    if not isinstance(compressed, bytes):
        raise ValueError("Inpaint checkpoint artifact data is missing.")
    expected_bytes = int(np.prod(expected_shape, dtype=np.int64))
    decompressor = zlib.decompressobj()
    raw = decompressor.decompress(compressed, expected_bytes + 1)
    if (
        len(raw) != expected_bytes
        or not decompressor.eof
        or decompressor.unconsumed_tail
        or decompressor.unused_data
    ):
        raise ValueError("Inpaint checkpoint artifact compressed size is invalid.")
    value = np.frombuffer(raw, dtype=np.uint8).reshape(expected_shape).copy()
    if mask and value.ndim != 2:
        raise ValueError("Inpaint checkpoint mask must be two-dimensional.")
    if not mask and (value.ndim != 3 or value.shape[2] not in (3, 4)):
        raise ValueError("Inpaint checkpoint image must be RGB or RGBA.")
    return value


def _compact_cleanup_stats(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    stats = dict(value or {})
    return {
        "applied": bool(stats.get("applied", False)),
        "component_count": max(
            0, int(stats.get("component_count", 0) or 0)
        ),
        "block_count": max(0, int(stats.get("block_count", 0) or 0)),
        "duplicate_bubble_inner_fill_applied": bool(
            stats.get("duplicate_bubble_inner_fill_applied", False)
        ),
        "duplicate_bubble_inner_fill_pixel_count": max(
            0,
            int(
                stats.get(
                    "duplicate_bubble_inner_fill_pixel_count",
                    0,
                )
                or 0
            ),
        ),
    }


def record_inpaint_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    blocks: list[TextBlock],
    cleaned_image: np.ndarray,
    raw_mask: np.ndarray,
    final_mask: np.ndarray,
    cleanup_stats: Mapping[str, Any] | None,
    cleaned_decoded_sha256: str | None = None,
) -> bool:
    if store is None or not store.available:
        return False
    cleaned = np.ascontiguousarray(cleaned_image)
    raw = np.ascontiguousarray(raw_mask)
    final = np.ascontiguousarray(final_mask)
    if (
        cleaned.dtype != np.uint8
        or cleaned.ndim != 3
        or cleaned.shape[2] not in (3, 4)
        or raw.dtype != np.uint8
        or final.dtype != np.uint8
        or raw.shape != cleaned.shape[:2]
        or final.shape != cleaned.shape[:2]
    ):
        return False
    try:
        block_states = snapshot_inpaint_block_state(blocks)
    except (TypeError, ValueError):
        return False
    if not block_states:
        return False
    objects: dict[str, str] = {}
    for role, kind, value in (
        (_INPAINT_CLEANED_OBJECT_ROLE, "cleaned-image", cleaned),
        (_INPAINT_RAW_MASK_OBJECT_ROLE, "raw-mask", raw),
        (_INPAINT_FINAL_MASK_OBJECT_ROLE, "final-mask", final),
    ):
        object_hash = store.put_object(_pack_array_artifact(kind, value))
        if object_hash is None:
            return False
        objects[role] = object_hash
    block_state_blob = _pack(
        {
            "schema_version": PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION,
            "state_schema": INPAINT_BLOCK_STATE_SCHEMA_VERSION,
            "blocks": block_states,
        }
    )
    block_state_hash = store.put_object(block_state_blob)
    if block_state_hash is None:
        return False
    objects[_INPAINT_BLOCK_STATE_OBJECT_ROLE] = block_state_hash
    cleaned_sha256 = str(cleaned_decoded_sha256 or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", cleaned_sha256):
        cleaned_sha256 = decoded_image_sha256(cleaned)
    compact_stats = _compact_cleanup_stats(cleanup_stats)
    return store.record_stage(
        page_key,
        "inpaint",
        fingerprint,
        payload={
            "schema_version": PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": canonical_sha256(dict(identity)),
            "cleaned_shape": [int(item) for item in cleaned.shape],
            "mask_shape": [int(item) for item in final.shape],
            "cleaned_sha256": cleaned_sha256,
            "raw_mask_sha256": decoded_image_sha256(raw),
            "final_mask_sha256": decoded_image_sha256(final),
            "cleanup_stats": compact_stats,
            "block_ids": [
                str(item.get("block_id", "") or "")
                for item in block_states
            ],
            "block_state_sha256": block_state_hash,
        },
        objects=objects,
    )


def lookup_inpaint_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    source_shape: tuple[int, ...],
    current_blocks: list[TextBlock],
) -> InpaintCheckpointResult | None:
    if store is None:
        return None
    hit = store.lookup_stage(page_key, "inpaint", fingerprint)
    if hit is None:
        return None
    try:
        payload = dict(hit.payload)
        if (
            int(payload.get("schema_version", 0))
            != PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity_sha256")
            != canonical_sha256(dict(identity))
        ):
            return None
        cleaned_shape = tuple(
            int(item) for item in payload.get("cleaned_shape", [])
        )
        mask_shape = tuple(
            int(item) for item in payload.get("mask_shape", [])
        )
        if (
            cleaned_shape != tuple(int(item) for item in source_shape)
            or mask_shape != cleaned_shape[:2]
        ):
            return None

        def read(role: str, kind: str, shape: tuple[int, ...], mask: bool):
            object_hash = hit.objects.get(role, "")
            raw = store.read_object(object_hash) if object_hash else None
            if raw is None:
                raise ValueError("Inpaint checkpoint object is missing.")
            return (
                _unpack_array_artifact(
                    raw,
                    kind=kind,
                    expected_shape=shape,
                    mask=mask,
                ),
                object_hash,
            )

        cleaned, cleaned_hash = read(
            _INPAINT_CLEANED_OBJECT_ROLE,
            "cleaned-image",
            cleaned_shape,
            False,
        )
        raw_mask, _ = read(
            _INPAINT_RAW_MASK_OBJECT_ROLE,
            "raw-mask",
            mask_shape,
            True,
        )
        final_mask, _ = read(
            _INPAINT_FINAL_MASK_OBJECT_ROLE,
            "final-mask",
            mask_shape,
            True,
        )
        block_state_hash = str(
            hit.objects.get(_INPAINT_BLOCK_STATE_OBJECT_ROLE, "") or ""
        )
        block_state_raw = (
            store.read_object(block_state_hash)
            if block_state_hash
            else None
        )
        if (
            block_state_raw is None
            or payload.get("block_state_sha256") != block_state_hash
        ):
            raise ValueError("Inpaint checkpoint block state is missing.")
        block_state_payload = _unpack(block_state_raw)
        if (
            not isinstance(block_state_payload, Mapping)
            or int(block_state_payload.get("schema_version", 0))
            != PROJECT_INPAINT_CHECKPOINT_SCHEMA_VERSION
            or block_state_payload.get("state_schema")
            != INPAINT_BLOCK_STATE_SCHEMA_VERSION
        ):
            raise ValueError(
                "Inpaint checkpoint block state schema does not match."
            )
        block_states = block_state_payload.get("blocks")
        if not isinstance(block_states, list):
            raise ValueError("Inpaint checkpoint block states are missing.")
        expected_block_ids = [
            str(getattr(block, "block_id", "") or "")
            for block in current_blocks
        ]
        restored_block_ids = [
            str(item.get("block_id", "") or "")
            if isinstance(item, Mapping)
            else ""
            for item in block_states
        ]
        if (
            not expected_block_ids
            or len(set(expected_block_ids)) != len(expected_block_ids)
            or payload.get("block_ids") != restored_block_ids
            or restored_block_ids != expected_block_ids
        ):
            raise ValueError(
                "Inpaint checkpoint block state identities do not match."
            )
        # Validate the complete state before returning a hit. Applying it to
        # disposable copies keeps lookup side-effect free.
        restore_inpaint_block_state(
            [block.deep_copy() for block in current_blocks],
            block_states,
        )
        if (
            payload.get("cleaned_sha256") != decoded_image_sha256(cleaned)
            or payload.get("raw_mask_sha256")
            != decoded_image_sha256(raw_mask)
            or payload.get("final_mask_sha256")
            != decoded_image_sha256(final_mask)
        ):
            raise ValueError("Inpaint checkpoint artifact digest does not match.")
        cleanup_stats = payload.get("cleanup_stats")
        if not isinstance(cleanup_stats, Mapping):
            cleanup_stats = {}
        return InpaintCheckpointResult(
            cleaned_image=cleaned,
            raw_mask=raw_mask,
            final_mask=final_mask,
            cleanup_stats=_compact_cleanup_stats(cleanup_stats),
            cleaned_object_sha256=cleaned_hash,
            cleaned_decoded_sha256=str(payload["cleaned_sha256"]),
            block_states=copy.deepcopy(block_states),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
        msgpack.ExtraData,
        msgpack.FormatError,
        msgpack.StackError,
        zlib.error,
    ):
        logger.warning(
            "Invalid inpaint checkpoint ignored for %s; inpainting will be "
            "recomputed.",
            page_key,
            exc_info=True,
        )
        return None


def viewer_render_state_signature(viewer_state: Mapping[str, Any] | None) -> str:
    value = dict(viewer_state or {})
    return _packed_sha256(
        {
            "schema": "viewer-text-items-state-v1",
            "text_items_state": copy.deepcopy(
                value.get("text_items_state", [])
            ),
        }
    )


def snapshot_project_render_blocks(
    blocks: list[TextBlock] | None,
) -> list[TextBlock]:
    return [block.deep_copy() for block in list(blocks or [])]


def render_block_state_signature(blocks: list[TextBlock] | None) -> str:
    return _packed_sha256(
        {
            "schema": "render-block-state-v1",
            "blocks": [
                copy.deepcopy(getattr(block, "__dict__", {}))
                for block in list(blocks or [])
            ],
        }
    )


_WINDOWS_LOCALIZED_FONT_ALIASES = {
    "맑은 고딕": ("malgun gothic",),
    "굴림": ("gulim",),
    "굴림체": ("gulimche",),
    "돋움": ("dotum",),
    "돋움체": ("dotumche",),
    "바탕": ("batang",),
    "바탕체": ("batangche",),
}


@lru_cache(maxsize=1)
def _windows_font_catalog() -> tuple[tuple[str, str, str, int, int], ...]:
    if platform.system() != "Windows":
        return ()
    try:
        import winreg

        fonts_root = Path(
            os.environ.get("WINDIR", r"C:\Windows")
        ) / "Fonts"
        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
        )
        records: list[tuple[str, str, str, int, int]] = []
        try:
            value_count = int(winreg.QueryInfoKey(key)[1])
            for index in range(value_count):
                display_name, raw_path, _value_type = winreg.EnumValue(
                    key,
                    index,
                )
                candidate = Path(str(raw_path or ""))
                if not candidate.is_absolute():
                    candidate = fonts_root / candidate
                try:
                    stat_result = candidate.stat()
                except OSError:
                    continue
                records.append(
                    (
                        str(display_name or ""),
                        candidate.name,
                        str(candidate),
                        int(stat_result.st_size),
                        int(stat_result.st_mtime_ns),
                    )
                )
        finally:
            winreg.CloseKey(key)
        records.sort(
            key=lambda item: (
                item[0].casefold(),
                item[1].casefold(),
            )
        )
        return tuple(records)
    except Exception:
        logger.debug(
            "Unable to read the Windows font catalog.",
            exc_info=True,
        )
        return ()


def _windows_font_program_sha256(family: str) -> str:
    normalized = str(family or "").strip().casefold()
    if not normalized:
        return ""
    aliases = {
        normalized,
        *(
            value.casefold()
            for value in _WINDOWS_LOCALIZED_FONT_ALIASES.get(
                normalized,
                (),
            )
        ),
    }
    matches: list[dict[str, str]] = []
    for display_name, file_name, file_path, _size, _mtime_ns in (
        _windows_font_catalog()
    ):
        display_key = display_name.casefold()
        if not any(alias in display_key for alias in aliases):
            continue
        try:
            matches.append(
                {
                    "file_name": file_name,
                    "sha256": _sha256_file(file_path),
                }
            )
        except OSError:
            continue
    matches.sort(key=lambda item: item["file_name"].casefold())
    return canonical_sha256(matches) if matches else ""


@lru_cache(maxsize=64)
def _stable_system_font_identity(family: str) -> dict[str, Any]:
    requested_family = str(family or "").strip()
    try:
        from PySide6.QtCore import qVersion
        from PySide6.QtGui import QFont, QFontDatabase, QFontInfo

        available_families = sorted(
            {
                str(value or "").strip()
                for value in QFontDatabase.families()
                if str(value or "").strip()
            },
            key=str.casefold,
        )
        canonical_by_name = {
            value.casefold(): value for value in available_families
        }
        resolved_family = canonical_by_name.get(
            requested_family.casefold(),
            "",
        )
        if not resolved_family:
            resolved_family = str(
                QFontInfo(QFont(requested_family, 32)).family() or ""
            ).strip()
        fallback_family = str(
            QFontDatabase.systemFont(
                QFontDatabase.SystemFont.GeneralFont
            ).family()
            or ""
        ).strip()
        if not resolved_family:
            resolved_family = fallback_family or requested_family

        windows_catalog = _windows_font_catalog()
        catalog_contract = [
            {
                "display_name": display_name,
                "file_name": file_name,
                "size": size,
                "mtime_ns": mtime_ns,
            }
            for (
                display_name,
                file_name,
                _file_path,
                size,
                mtime_ns,
            ) in windows_catalog
        ]
        return {
            "family": requested_family,
            "resolved_family": resolved_family,
            "fallback_family": fallback_family,
            "file_name": "",
            "file_sha256": "",
            "program_sha256": _windows_font_program_sha256(
                resolved_family
            ),
            "fallback_program_sha256": (
                _windows_font_program_sha256(fallback_family)
                if fallback_family
                else ""
            ),
            "font_catalog_sha256": canonical_sha256(
                catalog_contract or available_families
            ),
            "qt_version": str(qVersion() or ""),
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
        }
    except Exception:
        logger.debug(
            "Unable to fingerprint the selected system font contract.",
            exc_info=True,
        )
        return {
            "family": requested_family,
            "resolved_family": "",
            "fallback_family": "",
            "file_name": "",
            "file_sha256": "",
            "program_sha256": "",
            "fallback_program_sha256": "",
            "font_catalog_sha256": "",
            "qt_version": "",
            "platform": {
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
            },
        }


def resolve_font_identity(
    main_page: Any,
    font_family: str,
) -> dict[str, Any]:
    family = str(font_family or "").strip()
    candidates: list[str] = []
    if os.path.isfile(family):
        candidates.append(os.path.abspath(family))
    path_map = getattr(main_page, "_custom_font_path_to_family", {})
    if isinstance(path_map, Mapping):
        for path, mapped_family in path_map.items():
            if str(mapped_family or "").casefold() == family.casefold():
                candidates.append(os.path.abspath(str(path)))
    for candidate in candidates:
        try:
            if os.path.isfile(candidate):
                return {
                    "family": family,
                    "file_name": os.path.basename(candidate),
                    "file_sha256": _sha256_file(candidate),
                    "program_sha256": "",
                }
        except OSError:
            continue
    return copy.deepcopy(_stable_system_font_identity(family))


def build_render_identity(
    *,
    source_sha256: str,
    translation_fingerprint: str,
    inpaint_fingerprint: str,
    inpaint_artifact_sha256: str,
    blocks: list[TextBlock],
    render_settings: Mapping[str, Any],
    export_settings: Mapping[str, Any],
    font_identity: Mapping[str, Any],
    target_language_code: str,
    output_base_root: str,
) -> dict[str, Any]:
    return {
        "schema_version": PROJECT_RENDER_CHECKPOINT_SCHEMA_VERSION,
        "source_decoded_content_sha256": str(source_sha256 or ""),
        "translation_fingerprint": str(translation_fingerprint or ""),
        "inpaint_fingerprint": str(inpaint_fingerprint or ""),
        "inpaint_artifact_sha256": str(
            inpaint_artifact_sha256 or ""
        ),
        "ordered_translation_sha256": canonical_sha256(
            [
                {
                    "block_id": str(getattr(block, "block_id", "") or ""),
                    "source_text": str(getattr(block, "text", "") or ""),
                    "translation": str(
                        getattr(block, "translation", "") or ""
                    ),
                    "rich_text": copy.deepcopy(
                        getattr(block, "rich_text", "")
                    ),
                }
                for block in blocks
            ]
        ),
        "render_settings": _json_safe(dict(render_settings)),
        "export_settings": _json_safe(
            {
                key: export_settings[key]
                for key in sorted(_RENDER_EXPORT_IDENTITY_KEYS)
                if key in export_settings
            }
        ),
        "font": _json_safe(dict(font_identity)),
        "target_language_code": str(target_language_code or ""),
        "output_base_root": os.path.normcase(
            os.path.abspath(str(output_base_root or ""))
        ),
        "input_schema": RENDER_INPUT_SCHEMA_VERSION,
        "sanitizer_schema": RENDER_SANITIZER_SCHEMA_VERSION,
        "output_schema": RENDER_OUTPUT_SCHEMA_VERSION,
    }


def build_render_fingerprint(identity: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "stage": "render",
            "identity": dict(identity),
        }
    )


def _reserved_output_root_matches(
    recorded_root: str,
    current_base_root: str,
) -> bool:
    recorded = Path(os.path.abspath(recorded_root))
    current = Path(os.path.abspath(current_base_root))
    if os.path.normcase(str(recorded.parent)) != os.path.normcase(
        str(current.parent)
    ):
        return False
    recorded_name = os.path.normcase(recorded.name)
    current_name = os.path.normcase(current.name)
    return bool(
        recorded_name == current_name
        or re.fullmatch(
            rf"{re.escape(current_name)}_[0-9]{{3}}",
            recorded_name,
        )
    )


def _path_within(path: str, root: str) -> bool:
    try:
        lexical_match = os.path.commonpath(
            [os.path.abspath(path), os.path.abspath(root)]
        ) == os.path.abspath(root)
        real_match = os.path.commonpath(
            [os.path.realpath(path), os.path.realpath(root)]
        ) == os.path.realpath(root)
        return lexical_match and real_match
    except (OSError, ValueError):
        return False


def _path_has_symlink_component(path: str, root: str) -> bool:
    try:
        root_path = Path(os.path.abspath(root))
        path_value = Path(os.path.abspath(path))
        relative = path_value.relative_to(root_path)
        cursor = root_path
        if cursor.exists() and cursor.is_symlink():
            return True
        for part in relative.parts:
            cursor = cursor / part
            if cursor.exists() and cursor.is_symlink():
                return True
        return False
    except (OSError, ValueError):
        return True


def record_render_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    blocks: list[TextBlock],
    viewer_state: Mapping[str, Any],
    output_path: str,
    output_root: str,
) -> bool:
    if store is None or not store.available:
        return False
    normalized_path = os.path.abspath(str(output_path or ""))
    normalized_root = os.path.abspath(str(output_root or ""))
    if (
        not normalized_path
        or not normalized_root
        or not _path_within(normalized_path, normalized_root)
        or _path_has_symlink_component(normalized_path, normalized_root)
        or not os.path.isfile(normalized_path)
    ):
        return False
    try:
        output_bytes = Path(normalized_path).read_bytes()
        block_state_signature = render_block_state_signature(blocks)
    except OSError:
        return False
    except (TypeError, ValueError, msgpack.PackException):
        logger.warning(
            "Render checkpoint block state could not be serialized; the "
            "render output remains valid but will not be cached.",
            exc_info=True,
        )
        return False
    object_hash = store.put_object(output_bytes)
    if object_hash is None:
        return False
    return store.record_stage(
        page_key,
        "render",
        fingerprint,
        payload={
            "schema_version": PROJECT_RENDER_CHECKPOINT_SCHEMA_VERSION,
            "identity_sha256": canonical_sha256(dict(identity)),
            "viewer_state_signature": viewer_render_state_signature(
                viewer_state
            ),
            "block_ids": [
                str(getattr(block, "block_id", "") or "")
                for block in blocks
            ],
            "render_block_state_signature": block_state_signature,
            "output_path": normalized_path,
            "output_root": normalized_root,
            "output_sha256": object_hash,
            "output_size": len(output_bytes),
        },
        objects={_RENDER_OUTPUT_OBJECT_ROLE: object_hash},
    )


def lookup_render_checkpoint(
    store: ProjectCheckpointStore | None,
    *,
    page_key: str,
    fingerprint: str,
    identity: Mapping[str, Any],
    project_blocks: list[TextBlock] | None,
    project_viewer_state: Mapping[str, Any] | None,
    current_output_base_root: str,
) -> RenderCheckpointResult | None:
    if store is None or project_viewer_state is None:
        return None
    hit = store.lookup_stage(page_key, "render", fingerprint)
    if hit is None:
        return None
    try:
        payload = dict(hit.payload)
        output_path = os.path.abspath(
            str(payload.get("output_path", "") or "")
        )
        output_root = os.path.abspath(
            str(payload.get("output_root", "") or "")
        )
        object_hash = str(
            hit.objects.get(_RENDER_OUTPUT_OBJECT_ROLE, "") or ""
        )
        if (
            int(payload.get("schema_version", 0))
            != PROJECT_RENDER_CHECKPOINT_SCHEMA_VERSION
            or payload.get("identity_sha256")
            != canonical_sha256(dict(identity))
            or payload.get("viewer_state_signature")
            != viewer_render_state_signature(project_viewer_state)
            or payload.get("block_ids")
            != [
                str(getattr(block, "block_id", "") or "")
                for block in list(project_blocks or [])
            ]
            or payload.get("output_sha256") != object_hash
            or not _reserved_output_root_matches(
                output_root,
                current_output_base_root,
            )
            or not _path_within(output_path, output_root)
            or _path_has_symlink_component(output_path, output_root)
        ):
            return None
        # The full TextBlock __dict__ contains post-inpaint and render
        # diagnostics that are not render inputs and may be populated at
        # different points during project save/load. The stable render
        # identity, ordered block IDs, and persisted viewer state above are
        # the cache guards. Keep the full signature in the record for
        # diagnostics, but do not turn volatile metadata into a false miss.
        if os.path.exists(output_path):
            if (
                not os.path.isfile(output_path)
                or _sha256_file(output_path) != object_hash
            ):
                # Never overwrite an existing mismatched user file.
                return None
            return RenderCheckpointResult(
                output_path=output_path,
                output_root=output_root,
                output_sha256=object_hash,
                output_bytes=None,
                output_exists=True,
            )
        object_bytes = store.read_object(object_hash)
        if (
            object_bytes is None
            or len(object_bytes) != int(payload.get("output_size", -1))
            or hashlib.sha256(object_bytes).hexdigest() != object_hash
        ):
            return None
        return RenderCheckpointResult(
            output_path=output_path,
            output_root=output_root,
            output_sha256=object_hash,
            output_bytes=object_bytes,
            output_exists=False,
        )
    except (KeyError, OSError, TypeError, ValueError):
        logger.warning(
            "Invalid render checkpoint ignored for %s; rendering will be "
            "recomputed.",
            page_key,
            exc_info=True,
        )
        return None


def materialize_render_checkpoint_output(
    result: RenderCheckpointResult,
) -> str:
    if result.output_exists:
        return result.output_path
    if result.output_bytes is None:
        raise ValueError("Render checkpoint output bytes are missing.")
    if (
        hashlib.sha256(result.output_bytes).hexdigest()
        != result.output_sha256
    ):
        raise ValueError("Render checkpoint output digest does not match.")
    path = Path(result.output_path)
    if _path_has_symlink_component(
        result.output_path,
        result.output_root,
    ):
        raise OSError(
            "Render checkpoint output path contains a symbolic link."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    if _path_has_symlink_component(
        result.output_path,
        result.output_root,
    ):
        raise OSError(
            "Render checkpoint output path contains a symbolic link."
        )
    if path.exists():
        raise FileExistsError(
            "Render checkpoint refuses to overwrite an existing output."
        )
    temporary = path.with_name(f".{path.name}.partial-{os.getpid()}")
    try:
        with open(temporary, "xb") as handle:
            handle.write(result.output_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    if _sha256_file(path) != result.output_sha256:
        raise OSError("Materialized render checkpoint failed verification.")
    return str(path)
