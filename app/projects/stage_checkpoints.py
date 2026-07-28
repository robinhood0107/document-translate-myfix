from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import copy
import hashlib
import json
import logging
import os

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
from modules.utils.device import get_providers, resolve_device
from modules.utils.download import ModelDownloader, ModelID
from modules.utils.textblock import TextBlock


logger = logging.getLogger(__name__)


PROJECT_DETECTION_CHECKPOINT_SCHEMA_VERSION = 1
PROJECT_OCR_CHECKPOINT_SCHEMA_VERSION = 1
DETECTION_PREPROCESS_SCHEMA_VERSION = "rtdetr-v2-rgb-640-f32-v1"
DETECTION_POSTPROCESS_SCHEMA_VERSION = "comic-text-bubble-blocks-v1"
DETECTION_SORT_SCHEMA_VERSION = "sort-blk-list-v1"
DETECTION_MASK_SCHEMA_VERSION = "precomputed-mask-details-v1"
DETECTION_RENDER_AREA_SCHEMA_VERSION = "detected-bubble-render-area-v1"
DETECTION_FONT_SCHEMA_VERSION = "font-onnx-512-cv-color-v1"
OCR_POSTPROCESS_SCHEMA_VERSION = "quality-retry-drop-guards-v1"

_DETECTION_OBJECT_ROLE = "detection-result"
_OCR_OBJECT_ROLE = "ocr-raw-result"

_DETECTION_BLOCK_FIELDS = (
    "block_id",
    "xyxy",
    "segm_pts",
    "bubble_xyxy",
    "ctd_roi_xyxy",
    "cleanup_roi_xyxy",
    "mask_roi_xyxy",
    "text_class",
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
    return {
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
