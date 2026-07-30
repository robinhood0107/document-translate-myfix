from __future__ import annotations

import base64
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
import requests

from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.debug_artifacts import (
    active_debug_runtime_directory,
    append_active_raw_response,
    atomic_debug_file,
    atomic_debug_json,
    sanitize_debug_component,
)
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_EMPTY_AFTER_RETRY,
    OCR_STATUS_OK,
    OCR_STATUS_OK_AFTER_RETRY,
    ensure_three_channel,
    set_block_ocr_crop_diagnostics,
    set_block_ocr_diagnostics,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.textblock import TextBlock, ensure_text_block_id

from .base import OCREngine
from .mangalmm_llamacpp_runtime_contract import MANGALMM_MODEL_NAME
from .mangalmm_response_contract import (
    MANGALMM_RESPONSE_SCHEMA_VERSION,
    MangaLMMResponseContractError,
    parse_mangalmm_response,
)
from .persistent_cache import canonical_sha256
from .result_contract import (
    OCR_STRATEGY_MANGALMM_FULL_PAGE,
    finalize_ocr_processing_contracts,
    initialize_ocr_result_contract,
)
from .selection import normalize_ocr_mode


logger = logging.getLogger(__name__)

SERVICE_NAME = "MangaLMM"
SETTINGS_PAGE_NAME = "MangaLMM Settings"
DEFAULT_MANGALMM_SERVER_URL = "http://127.0.0.1:28081/v1"
DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS = 4096
DEFAULT_MANGALMM_PARALLEL_WORKERS = 1
DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC = 60
DEFAULT_MANGALMM_SAFE_RESIZE = True
DEFAULT_MANGALMM_MAX_PIXELS = 2_116_800
DEFAULT_MANGALMM_MAX_LONG_SIDE = 1728
DEFAULT_MANGALMM_STANDARD_SHORT_SIDE = 1224
DEFAULT_MANGALMM_DENSE_BLOCK_COUNT = 24
DEFAULT_MANGALMM_DENSE_SMALL_BLOCK_RATIO = 0.55
DEFAULT_MANGALMM_DENSE_TEXT_COVER_RATIO = 0.18
DEFAULT_MANGALMM_SMALL_BLOCK_AREA_RATIO = 0.008
DEFAULT_MANGALMM_TEMPERATURE = 0.1
DEFAULT_MANGALMM_TOP_K = 1
DEFAULT_MANGALMM_TOP_P = 0.001
DEFAULT_MANGALMM_MIN_P = 0.0
DEFAULT_MANGALMM_REPEAT_PENALTY = 1.05
DEFAULT_MANGALMM_REPEAT_LAST_N = 0
DEFAULT_MANGALMM_PRESENCE_PENALTY = 0.0
DEFAULT_MANGALMM_FREQUENCY_PENALTY = 0.0
DEFAULT_MANGALMM_PNG_COMPRESSION = 1
DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT = 96


@dataclass(slots=True)
class RequestUnit:
    bbox_xyxy: tuple[int, int, int, int]
    unit_kind: str


@dataclass(slots=True)
class ResizePlan:
    profile: str
    original_shape: tuple[int, int]
    request_shape: tuple[int, int]
    base_scale: float
    scale_x: float
    scale_y: float
    max_completion_tokens: int
    block_count: int
    small_block_ratio: float
    text_cover_ratio: float


@dataclass(slots=True)
class AttemptSpec:
    index: int
    resize_plan: ResizePlan
    prompt_mode: str
    prompt_text: str
    attempt_kind: str


@dataclass(slots=True)
class OCRRegion:
    bbox_xyxy: list[int]
    bbox_xyxy_float: list[float]
    text: str
    unit_bbox_xyxy: list[int]
    unit_kind: str
    unit_resize_scale: float
    edge_distance: float
    normalized_text: str
    raw_text: str = ""
    response_bbox_2d: list[float] | None = None
    scale_x: float = 1.0
    scale_y: float = 1.0
    request_shape: list[int] | None = None
    resize_profile: str = "standard"


class MangaLMMOCREngine(OCREngine):
    MAX_COMPLETION_TOKENS_RANGE = (64, 4096)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_ENDPOINT_SUFFIX = "/chat/completions"
    STANDARD_PROMPT = (
        "Please perform OCR on this full manga page. Return one top-level JSON "
        'array only. Every item must contain "bbox_2d" as [x1, y1, x2, y2] '
        'and "text_content" as the exact recognized Japanese text. Do not '
        "translate, summarize, merge neighboring regions, or add commentary."
    )
    DENSE_PROMPT = STANDARD_PROMPT

    def __init__(self) -> None:
        self.server_url = DEFAULT_MANGALMM_SERVER_URL
        self.max_completion_tokens = DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS
        self.parallel_workers = DEFAULT_MANGALMM_PARALLEL_WORKERS
        self.request_timeout_sec = DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC
        self.raw_response_logging = False
        self.safe_resize = DEFAULT_MANGALMM_SAFE_RESIZE
        self.max_pixels = DEFAULT_MANGALMM_MAX_PIXELS
        self.max_long_side = DEFAULT_MANGALMM_MAX_LONG_SIDE
        self.temperature = DEFAULT_MANGALMM_TEMPERATURE
        self.top_k = DEFAULT_MANGALMM_TOP_K
        self.top_p = DEFAULT_MANGALMM_TOP_P
        self.min_p = DEFAULT_MANGALMM_MIN_P
        self.repeat_penalty = DEFAULT_MANGALMM_REPEAT_PENALTY
        self.repeat_last_n = DEFAULT_MANGALMM_REPEAT_LAST_N
        self.presence_penalty = DEFAULT_MANGALMM_PRESENCE_PENALTY
        self.frequency_penalty = DEFAULT_MANGALMM_FREQUENCY_PENALTY
        self.selected_ocr_mode = ""
        self.source_lang_english = ""
        self.contract_mode = "direct_manual"
        self.debug_root: Path | None = None
        self._debug_root_from_env = False
        self.debug_export_limit = DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT
        self._debug_export_counter = count(1)
        self._debug_export_lock = Lock()
        self.last_request_metadata: dict[str, object] = {}
        self.last_page_regions: list[dict[str, object]] = []
        self.last_attempt_history: list[dict[str, object]] = []
        self.last_shadow_regions: list[dict[str, object]] = []
        self.last_merge_split_diagnostics: list[dict[str, object]] = []
        self.last_transport_metadata: dict[str, object] = {}

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_mangalmm_ocr_settings()
        selected_mode_raw = kwargs.get("selected_ocr_mode")
        if selected_mode_raw is None and hasattr(settings, "get_tool_selection"):
            try:
                selected_mode_raw = settings.get_tool_selection("ocr")
            except Exception:
                selected_mode_raw = ""
        self.selected_ocr_mode = normalize_ocr_mode(selected_mode_raw)
        self.source_lang_english = str(kwargs.get("source_lang_english", "") or "")
        self.contract_mode = "direct_manual"
        self.server_url = config.get("server_url", DEFAULT_MANGALMM_SERVER_URL) or DEFAULT_MANGALMM_SERVER_URL
        self.max_completion_tokens = self._clamp_int(
            config.get("max_completion_tokens", DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS),
            DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS,
            self.MAX_COMPLETION_TOKENS_RANGE,
        )
        self.parallel_workers = self._clamp_int(
            config.get("parallel_workers", DEFAULT_MANGALMM_PARALLEL_WORKERS),
            DEFAULT_MANGALMM_PARALLEL_WORKERS,
            self.PARALLEL_WORKERS_RANGE,
        )
        self.request_timeout_sec = max(
            1,
            self._clamp_int(
                config.get("request_timeout_sec", DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC),
                DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC,
                (1, 3600),
            ),
        )
        self.raw_response_logging = bool(config.get("raw_response_logging", False))
        self.safe_resize = bool(config.get("safe_resize", DEFAULT_MANGALMM_SAFE_RESIZE))
        self.max_pixels = max(
            100_000,
            self._clamp_int(
                config.get("max_pixels", DEFAULT_MANGALMM_MAX_PIXELS),
                DEFAULT_MANGALMM_MAX_PIXELS,
                (100_000, 8_000_000),
            ),
        )
        self.max_long_side = max(
            512,
            self._clamp_int(
                config.get("max_long_side", DEFAULT_MANGALMM_MAX_LONG_SIDE),
                DEFAULT_MANGALMM_MAX_LONG_SIDE,
                (512, 4096),
            ),
        )
        self.temperature = self._read_env_float(
            "CT_MANGALMM_TEMPERATURE",
            config.get("temperature", DEFAULT_MANGALMM_TEMPERATURE),
            DEFAULT_MANGALMM_TEMPERATURE,
            minimum=0.0,
            maximum=2.0,
        )
        self.top_k = self._read_env_int(
            "CT_MANGALMM_TOP_K",
            config.get("top_k", DEFAULT_MANGALMM_TOP_K),
            DEFAULT_MANGALMM_TOP_K,
            bounds=(1, 256),
        )
        self.top_p = self._read_env_float(
            "CT_MANGALMM_TOP_P",
            config.get("top_p", DEFAULT_MANGALMM_TOP_P),
            DEFAULT_MANGALMM_TOP_P,
            minimum=0.0,
            maximum=1.0,
        )
        self.min_p = self._read_env_float(
            "CT_MANGALMM_MIN_P",
            config.get("min_p", DEFAULT_MANGALMM_MIN_P),
            DEFAULT_MANGALMM_MIN_P,
            minimum=0.0,
            maximum=1.0,
        )
        self.repeat_penalty = self._read_env_float(
            "CT_MANGALMM_REPEAT_PENALTY",
            config.get("repeat_penalty", DEFAULT_MANGALMM_REPEAT_PENALTY),
            DEFAULT_MANGALMM_REPEAT_PENALTY,
            minimum=0.0,
            maximum=2.0,
        )
        self.repeat_last_n = self._read_env_int(
            "CT_MANGALMM_REPEAT_LAST_N",
            config.get("repeat_last_n", DEFAULT_MANGALMM_REPEAT_LAST_N),
            DEFAULT_MANGALMM_REPEAT_LAST_N,
            bounds=(0, 4096),
        )
        self.presence_penalty = self._read_env_float(
            "CT_MANGALMM_PRESENCE_PENALTY",
            config.get("presence_penalty", DEFAULT_MANGALMM_PRESENCE_PENALTY),
            DEFAULT_MANGALMM_PRESENCE_PENALTY,
            minimum=-2.0,
            maximum=2.0,
        )
        self.frequency_penalty = self._read_env_float(
            "CT_MANGALMM_FREQUENCY_PENALTY",
            config.get("frequency_penalty", DEFAULT_MANGALMM_FREQUENCY_PENALTY),
            DEFAULT_MANGALMM_FREQUENCY_PENALTY,
            minimum=-2.0,
            maximum=2.0,
        )
        debug_root_raw = os.getenv("CT_MANGALMM_DEBUG_ROOT", "").strip()
        self._debug_root_from_env = bool(debug_root_raw)
        self.debug_root = Path(debug_root_raw) if debug_root_raw else None
        self.debug_export_limit = self._read_env_int(
            "CT_MANGALMM_DEBUG_EXPORT_LIMIT",
            DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT,
            DEFAULT_MANGALMM_DEBUG_EXPORT_LIMIT,
            bounds=(1, 4096),
        )

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        blocks = list(blk_list or [])
        self.last_page_regions = []
        self.last_attempt_history = []
        self.last_shadow_regions = []
        self.last_merge_split_diagnostics = []
        self.last_transport_metadata = {}

        image = ensure_three_channel(img)
        for block in blocks:
            initialize_ocr_result_contract(
                block,
                strategy=OCR_STRATEGY_MANGALMM_FULL_PAGE,
                model_identity=MANGALMM_MODEL_NAME,
                runtime_identity=self._runtime_identity_fingerprint(),
            )
            block.merge_split_diagnostics = {
                key: value
                for key, value in dict(block.merge_split_diagnostics).items()
                if not str(key).startswith("mangalmm_")
            }
            block.ocr_geometry_provenance = {
                **dict(block.ocr_geometry_provenance),
                "contract_schema_version": MANGALMM_RESPONSE_SCHEMA_VERSION,
                "ocr_strategy": OCR_STRATEGY_MANGALMM_FULL_PAGE,
                "detector_geometry_authoritative": True,
                "detector_text_bbox": self._coords_or_none(
                    getattr(block, "xyxy", None)
                ),
                "detector_bubble_bbox": self._coords_or_none(
                    getattr(block, "bubble_xyxy", None)
                ),
                "ocr_geometry_source": "mangalmm_bbox_2d_full_page",
            }
        started_at = time.perf_counter()
        page_unit = self._build_request_units(image.shape)[0]
        attempt_specs = self._build_attempt_specs(image.shape, blocks)
        assignments: dict[int, list[dict[str, object]]] = {index: [] for index in range(len(blocks))}
        final_attempt: AttemptSpec | None = None
        quality: dict[str, object] = {}

        for attempt_spec in attempt_specs:
            attempt_result = self._request_regions_for_attempt(image, page_unit, attempt_spec)
            attempt_assignments = self._assign_regions_to_blocks(attempt_result["regions"], blocks)
            matched_region_count = sum(len(items) for items in attempt_assignments.values())
            matched_block_count = sum(1 for items in attempt_assignments.values() if items)
            attempt_entry = {
                **attempt_result["metadata"],
                "attempt_index": int(attempt_spec.index),
                "attempt_kind": attempt_spec.attempt_kind,
                "attempt_profile": attempt_spec.resize_plan.profile,
                "parsed_region_count": int(attempt_result["parsed_region_count"]),
                "mapped_region_count": int(attempt_result["mapped_region_count"]),
                "matched_region_count": int(matched_region_count),
                "matched_block_count": int(matched_block_count),
                "raw_response_length": len(str(attempt_result["raw_text"] or "")),
            }
            failure_reason = ""
            if int(attempt_result["parsed_region_count"]) <= 0:
                failure_reason = "no_valid_regions"
            elif int(attempt_result["mapped_region_count"]) <= 0:
                failure_reason = "mapped_regions_out_of_bounds"
            elif matched_block_count <= 0:
                failure_reason = "no_block_match"
            attempt_entry["failure_reason"] = failure_reason
            attempt_entry["final_status"] = "success" if matched_block_count > 0 else "failure"
            self.last_attempt_history.append(attempt_entry)

            if matched_block_count > 0:
                assignments = attempt_assignments
                final_attempt = attempt_spec
                success_status = OCR_STATUS_OK if attempt_spec.index == 0 else OCR_STATUS_OK_AFTER_RETRY
                empty_status = OCR_STATUS_EMPTY_INITIAL if attempt_spec.index == 0 else OCR_STATUS_EMPTY_AFTER_RETRY
                quality = self._apply_assignments_to_blocks(
                    blocks,
                    assignments,
                    attempt_count=attempt_spec.index + 1,
                    success_status=success_status,
                    empty_status=empty_status,
                    page_bbox=page_unit.bbox_xyxy,
                    resize_plan=attempt_spec.resize_plan,
                )
                break

            if failure_reason == "no_block_match":
                self._export_debug_artifact(
                    blk=None,
                    failure_reason="MangaLMM OCR regions did not match any detected block.",
                    response_kind="no_block_match",
                    raw_text=str(attempt_result["raw_text"] or ""),
                    crop_bbox=page_unit.bbox_xyxy,
                    crop_source=page_unit.unit_kind,
                    resize_plan=attempt_spec.resize_plan,
                    crop_image=attempt_result["crop_image"],
                    request_image=attempt_result["request_image"],
                    analysis={
                        **attempt_result["analysis"],
                        "attempt_index": int(attempt_spec.index),
                        "attempt_kind": attempt_spec.attempt_kind,
                        "prompt_mode": attempt_spec.prompt_mode,
                        "matched_region_count": int(matched_region_count),
                        "matched_block_count": int(matched_block_count),
                    },
                    unit_bbox=page_unit.bbox_xyxy,
                    unit_kind=page_unit.unit_kind,
                )

        if final_attempt is None:
            terminal_attempt = attempt_specs[-1]
            quality = self._apply_assignments_to_blocks(
                blocks,
                assignments,
                attempt_count=len(attempt_specs),
                success_status=OCR_STATUS_OK_AFTER_RETRY if len(attempt_specs) > 1 else OCR_STATUS_OK,
                empty_status=OCR_STATUS_EMPTY_INITIAL,
                page_bbox=page_unit.bbox_xyxy,
                resize_plan=terminal_attempt.resize_plan,
            )
            self.last_request_metadata = {
                **self.last_request_metadata,
                "attempts": list(self.last_attempt_history),
                "attempt_count": len(self.last_attempt_history),
                "retry_count": max(0, len(self.last_attempt_history) - 1),
                "contract_mode": self.contract_mode,
                "final_status": "failure",
                "failure_reason": str(self.last_attempt_history[-1].get("failure_reason", "") or "no_valid_regions"),
                "matched_region_count": 0,
                "matched_block_count": 0,
                "mapped_region_count": len(self.last_page_regions),
                "shadow_region_count": len(self.last_shadow_regions),
                "merge_split_diagnostic_count": len(
                    self.last_merge_split_diagnostics
                ),
                "non_empty_block_count": int(quality.get("non_empty", 0) or 0),
                "raw_response_length": len(str(self.last_request_metadata.get("raw_response", "") or "")),
            }
        else:
            self.last_request_metadata = {
                **self.last_request_metadata,
                "attempts": list(self.last_attempt_history),
                "attempt_count": len(self.last_attempt_history),
                "retry_count": max(0, len(self.last_attempt_history) - 1),
                "contract_mode": self.contract_mode,
                "final_status": "success",
                "failure_reason": "",
                "matched_region_count": sum(len(items) for items in assignments.values()),
                "matched_block_count": sum(1 for items in assignments.values() if items),
                "mapped_region_count": len(self.last_page_regions),
                "shadow_region_count": len(self.last_shadow_regions),
                "merge_split_diagnostic_count": len(
                    self.last_merge_split_diagnostics
                ),
                "non_empty_block_count": int(quality.get("non_empty", 0) or 0),
                "raw_response_length": len(str(self.last_request_metadata.get("raw_response", "") or "")),
            }

        logger.info(
            "mangalmm_full_page_ocr complete: blocks=%d regions=%d profile=%s request_shape=%s attempts=%d status=%s quality=%s elapsed_ms=%.1f",
            len(blocks),
            len(self.last_page_regions),
            self.last_request_metadata.get("resize_profile", ""),
            self.last_request_metadata.get("request_shape", []),
            len(self.last_attempt_history),
            self.last_request_metadata.get("final_status", "unknown"),
            quality.get("reason", "") or "ok",
            (time.perf_counter() - started_at) * 1000.0,
        )
        self.last_request_metadata["processing_contract"] = (
            finalize_ocr_processing_contracts(blocks)
        )
        return blk_list

    def _build_request_units(self, image_shape: tuple[int, ...]) -> list[RequestUnit]:
        image_h, image_w = image_shape[:2]
        return [RequestUnit((0, 0, int(image_w), int(image_h)), "page_full")]

    def _plan_page_request(self, image_shape: tuple[int, ...], blk_list: list[TextBlock]) -> ResizePlan:
        profile, block_count, small_block_ratio, text_cover_ratio = self._select_resize_profile(image_shape, blk_list)
        return self._build_resize_plan(
            image_shape,
            profile=profile,
            block_count=block_count,
            small_block_ratio=small_block_ratio,
            text_cover_ratio=text_cover_ratio,
        )

    def _build_resize_plan(
        self,
        image_shape: tuple[int, ...],
        *,
        profile: str,
        block_count: int,
        small_block_ratio: float,
        text_cover_ratio: float,
        max_completion_tokens_override: int | None = None,
    ) -> ResizePlan:
        image_h, image_w = image_shape[:2]
        standard_short_side = DEFAULT_MANGALMM_STANDARD_SHORT_SIDE
        standard_pixel_cap = self.max_pixels
        standard_long_side = self.max_long_side
        short_side_cap = standard_short_side
        long_side_cap = standard_long_side
        pixel_cap = standard_pixel_cap

        short_side = float(min(image_w, image_h))
        long_side = float(max(image_w, image_h))
        page_area = float(max(1, image_w * image_h))
        if self.safe_resize:
            base_scale = min(
                1.0,
                float(short_side_cap) / short_side if short_side > 0 else 1.0,
                float(long_side_cap) / long_side if long_side > 0 else 1.0,
                math.sqrt(float(pixel_cap) / page_area),
            )
        else:
            base_scale = 1.0

        request_w = max(1, int(round(image_w * base_scale)))
        request_h = max(1, int(round(image_h * base_scale)))
        scale_x = request_w / float(max(1, image_w))
        scale_y = request_h / float(max(1, image_h))
        request_tokens = (
            int(max_completion_tokens_override)
            if max_completion_tokens_override is not None
            else int(self.max_completion_tokens)
        )
        return ResizePlan(
            profile=profile,
            original_shape=(int(image_h), int(image_w)),
            request_shape=(int(request_h), int(request_w)),
            base_scale=float(base_scale),
            scale_x=float(scale_x),
            scale_y=float(scale_y),
            max_completion_tokens=int(request_tokens),
            block_count=int(block_count),
            small_block_ratio=float(small_block_ratio),
            text_cover_ratio=float(text_cover_ratio),
        )

    def _build_attempt_specs(
        self,
        image_shape: tuple[int, ...],
        blk_list: list[TextBlock],
    ) -> list[AttemptSpec]:
        initial_plan = self._plan_page_request(image_shape, blk_list)
        initial_prompt_mode, initial_prompt_text = self._prompt_for_resize_plan(initial_plan)
        attempts = [
            AttemptSpec(
                index=0,
                resize_plan=initial_plan,
                prompt_mode=initial_prompt_mode,
                prompt_text=initial_prompt_text,
                attempt_kind="primary",
            )
        ]
        return attempts

    def _select_resize_profile(
        self,
        image_shape: tuple[int, ...],
        blk_list: list[TextBlock],
    ) -> tuple[str, int, float, float]:
        page_h, page_w = image_shape[:2]
        page_area = float(max(1, page_h * page_w))
        areas: list[float] = []
        for blk in blk_list or []:
            box = self._normalize_box(getattr(blk, "xyxy", None))
            if box is None:
                continue
            areas.append(float(self._bbox_area(box)))

        block_count = len(areas)
        if block_count <= 0:
            return "standard", 0, 0.0, 0.0

        small_block_limit = page_area * DEFAULT_MANGALMM_SMALL_BLOCK_AREA_RATIO
        small_block_count = sum(1 for area in areas if area <= small_block_limit)
        small_block_ratio = small_block_count / float(block_count)
        text_cover_ratio = sum(areas) / page_area
        is_dense = (
            block_count >= DEFAULT_MANGALMM_DENSE_BLOCK_COUNT
            or text_cover_ratio >= DEFAULT_MANGALMM_DENSE_TEXT_COVER_RATIO
            or (
                block_count >= 12
                and small_block_ratio >= DEFAULT_MANGALMM_DENSE_SMALL_BLOCK_RATIO
            )
        )
        return ("dense" if is_dense else "standard", block_count, small_block_ratio, text_cover_ratio)

    def _request_regions_for_attempt(
        self,
        image: np.ndarray,
        unit: RequestUnit,
        attempt_spec: AttemptSpec,
    ) -> dict[str, object]:
        resize_plan = attempt_spec.resize_plan
        crop = self._crop_image(image, unit.bbox_xyxy)
        if crop is None:
            self.last_request_metadata = {
                "error": "Invalid OCR page bounds.",
                "resize_profile": resize_plan.profile,
                "attempt_profile": resize_plan.profile,
                "prompt_mode": attempt_spec.prompt_mode,
                "attempt_index": int(attempt_spec.index),
                "attempt_kind": attempt_spec.attempt_kind,
                "raw_response_length": 0,
                "parsed_region_count": 0,
                "mapped_region_count": 0,
            }
            self._export_debug_artifact(
                blk=None,
                failure_reason="Invalid OCR page bounds.",
                response_kind="invalid_crop",
                raw_text="",
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_plan=resize_plan,
                crop_image=None,
                request_image=None,
                analysis={
                    "attempt_index": int(attempt_spec.index),
                    "attempt_kind": attempt_spec.attempt_kind,
                    "prompt_mode": attempt_spec.prompt_mode,
                },
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            self.last_page_regions = []
            return {
                "regions": [],
                "analysis": {
                    "response_kind": "invalid_crop",
                    "payload_type": "invalid_crop",
                    "raw_length": 0,
                },
                "raw_text": "",
                "crop_image": None,
                "request_image": None,
                "parsed_region_count": 0,
                "mapped_region_count": 0,
                "metadata": dict(self.last_request_metadata),
            }

        request_image = self._resize_for_request(crop, resize_plan)
        raw_text = ""
        analysis: dict[str, object]
        self.last_transport_metadata = {}
        try:
            raw_text = self._request_response_text(
                request_image,
                max_completion_tokens=resize_plan.max_completion_tokens,
                prompt_text=attempt_spec.prompt_text,
            )
            finish_reason = str(
                self.last_transport_metadata.get("finish_reason", "") or ""
            ).strip().lower()
            if finish_reason == "length":
                analysis = {
                    "regions": [],
                    "response_kind": "truncated:finish_reason_length",
                    "payload_type": "invalid",
                    "raw_length": len(raw_text),
                    "parser_error_code": "finish_reason_length",
                    "parser_error": (
                        "MangaLMM response reached the completion-token "
                        "limit and was rejected."
                    ),
                }
            else:
                analysis = self._analyze_region_payload(raw_text)
        except (LocalServiceConnectionError, LocalServiceResponseError) as exc:
            self.last_request_metadata = {
                "original_shape": list(resize_plan.original_shape),
                "request_shape": list(resize_plan.request_shape),
                "resize_profile": resize_plan.profile,
                "attempt_profile": resize_plan.profile,
                "prompt_mode": attempt_spec.prompt_mode,
                "attempt_index": int(attempt_spec.index),
                "attempt_kind": attempt_spec.attempt_kind,
                "scale_x": resize_plan.scale_x,
                "scale_y": resize_plan.scale_y,
                "base_scale": resize_plan.base_scale,
                "max_completion_tokens": resize_plan.max_completion_tokens,
                "block_count": resize_plan.block_count,
                "small_block_ratio": resize_plan.small_block_ratio,
                "text_cover_ratio": resize_plan.text_cover_ratio,
                "error": str(exc),
                "raw_response": raw_text,
                "raw_response_length": len(str(raw_text or "")),
                "parsed_region_count": 0,
                "mapped_region_count": 0,
            }
            self._export_debug_artifact(
                blk=None,
                failure_reason=str(exc),
                response_kind="request_error",
                raw_text=raw_text,
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_plan=resize_plan,
                crop_image=crop,
                request_image=request_image,
                analysis={
                    "error": str(exc),
                    "attempt_index": int(attempt_spec.index),
                    "attempt_kind": attempt_spec.attempt_kind,
                    "prompt_mode": attempt_spec.prompt_mode,
                },
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            raise

        self.last_request_metadata = {
            "original_shape": list(resize_plan.original_shape),
            "request_shape": list(resize_plan.request_shape),
            "resize_profile": resize_plan.profile,
            "attempt_profile": resize_plan.profile,
            "prompt_mode": attempt_spec.prompt_mode,
            "attempt_index": int(attempt_spec.index),
            "attempt_kind": attempt_spec.attempt_kind,
            "scale_x": resize_plan.scale_x,
            "scale_y": resize_plan.scale_y,
            "base_scale": resize_plan.base_scale,
            "max_completion_tokens": resize_plan.max_completion_tokens,
            "block_count": resize_plan.block_count,
            "small_block_ratio": resize_plan.small_block_ratio,
            "text_cover_ratio": resize_plan.text_cover_ratio,
            "response_kind": str(analysis.get("response_kind", "") or "unknown"),
            "payload_type": str(analysis.get("payload_type", "") or "unknown"),
            "parser_error_code": str(
                analysis.get("parser_error_code", "") or ""
            ),
            "parser_error": str(analysis.get("parser_error", "") or ""),
            "finish_reason": str(
                self.last_transport_metadata.get("finish_reason", "") or ""
            ),
            "prompt_tokens": int(
                self.last_transport_metadata.get("prompt_tokens", 0) or 0
            ),
            "completion_tokens": int(
                self.last_transport_metadata.get("completion_tokens", 0)
                or 0
            ),
            "total_tokens": int(
                self.last_transport_metadata.get("total_tokens", 0) or 0
            ),
            "region_count": len(analysis.get("regions", []) or []),
            "raw_response": raw_text,
            "raw_response_length": len(raw_text),
        }

        regions = analysis["regions"]
        if not regions:
            response_kind = str(analysis.get("response_kind", "") or "unknown")
            self._export_debug_artifact(
                blk=None,
                failure_reason="MangaLMM returned no valid OCR regions.",
                response_kind=response_kind,
                raw_text=raw_text,
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_plan=resize_plan,
                crop_image=crop,
                request_image=request_image,
                analysis={
                    **analysis,
                    "attempt_index": int(attempt_spec.index),
                    "attempt_kind": attempt_spec.attempt_kind,
                    "prompt_mode": attempt_spec.prompt_mode,
                },
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            self.last_page_regions = []
            return {
                "regions": [],
                "analysis": analysis,
                "raw_text": raw_text,
                "crop_image": crop,
                "request_image": request_image,
                "parsed_region_count": 0,
                "mapped_region_count": 0,
                "metadata": dict(self.last_request_metadata),
            }

        mapped = self._map_regions_to_page_coords(
            regions,
            unit.bbox_xyxy,
            crop.shape[:2],
            resize_plan,
            unit.unit_kind,
        )
        self.last_page_regions = [
            {
                "bbox_xyxy": list(region.bbox_xyxy),
                "bbox_xyxy_float": list(region.bbox_xyxy_float),
                "text": region.text,
                "raw_text": region.raw_text,
                "unit_bbox_xyxy": list(region.unit_bbox_xyxy),
                "unit_kind": region.unit_kind,
                "unit_resize_scale": float(region.unit_resize_scale),
                "response_bbox_2d": list(region.response_bbox_2d or []),
                "scale_x": float(region.scale_x),
                "scale_y": float(region.scale_y),
                "request_shape": list(region.request_shape or []),
                "resize_profile": region.resize_profile,
            }
            for region in mapped
        ]
        if not mapped:
            self.last_request_metadata["parsed_region_count"] = int(len(regions))
            self.last_request_metadata["mapped_region_count"] = 0
            response_kind = str(analysis.get("response_kind", "") or "unknown")
            self._export_debug_artifact(
                blk=None,
                failure_reason="MangaLMM returned OCR regions outside the page bounds.",
                response_kind=response_kind,
                raw_text=raw_text,
                crop_bbox=unit.bbox_xyxy,
                crop_source=unit.unit_kind,
                resize_plan=resize_plan,
                crop_image=crop,
                request_image=request_image,
                analysis={
                    **analysis,
                    "attempt_index": int(attempt_spec.index),
                    "attempt_kind": attempt_spec.attempt_kind,
                    "prompt_mode": attempt_spec.prompt_mode,
                },
                unit_bbox=unit.bbox_xyxy,
                unit_kind=unit.unit_kind,
            )
            return {
                "regions": [],
                "analysis": analysis,
                "raw_text": raw_text,
                "crop_image": crop,
                "request_image": request_image,
                "parsed_region_count": len(regions),
                "mapped_region_count": 0,
                "metadata": dict(self.last_request_metadata),
            }
        self.last_request_metadata["parsed_region_count"] = int(len(regions))
        self.last_request_metadata["mapped_region_count"] = int(len(mapped))
        self._export_debug_artifact(
            blk=None,
            failure_reason="",
            response_kind=str(analysis.get("response_kind", "") or "json_array"),
            raw_text=raw_text,
            crop_bbox=unit.bbox_xyxy,
            crop_source=unit.unit_kind,
            resize_plan=resize_plan,
            crop_image=crop,
            request_image=request_image,
            analysis={
                **analysis,
                "attempt_index": int(attempt_spec.index),
                "attempt_kind": attempt_spec.attempt_kind,
                "prompt_mode": attempt_spec.prompt_mode,
                "mapped_region_count": len(mapped),
            },
            unit_bbox=unit.bbox_xyxy,
            unit_kind=unit.unit_kind,
        )
        return {
            "regions": mapped,
            "analysis": analysis,
            "raw_text": raw_text,
            "crop_image": crop,
            "request_image": request_image,
            "parsed_region_count": len(regions),
            "mapped_region_count": len(mapped),
            "metadata": dict(self.last_request_metadata),
        }

    def _assign_regions_to_blocks(
        self,
        regions: list[OCRRegion],
        blk_list: list[TextBlock],
    ) -> dict[int, list[dict[str, object]]]:
        self.last_shadow_regions = []
        self.last_merge_split_diagnostics = []
        assignments: dict[int, list[dict[str, object]]] = {
            index: [] for index in range(len(blk_list))
        }
        tentative: dict[int, list[dict[str, object]]] = {
            index: [] for index in range(len(blk_list))
        }
        for block in blk_list:
            initialize_ocr_result_contract(
                block,
                strategy=OCR_STRATEGY_MANGALMM_FULL_PAGE,
                model_identity=MANGALMM_MODEL_NAME,
                runtime_identity=self._runtime_identity_fingerprint(),
            )
        for region in regions:
            candidates = self._candidate_block_matches(region, blk_list)
            if not candidates:
                self.last_shadow_regions.append(
                    self._serialize_shadow_region(
                        region,
                        reason="manga_only_unmatched",
                    )
                )
                continue
            precision_candidates = [
                candidate
                for candidate in candidates
                if (
                    bool(candidate[1]["center_in_precision"])
                    or float(candidate[1]["precision_cover"]) >= 0.20
                )
            ]
            ambiguous_candidates = (
                precision_candidates
                if len(precision_candidates) > 1
                else (
                    candidates
                    if not precision_candidates and len(candidates) > 1
                    else []
                )
            )
            if ambiguous_candidates:
                block_ids = [
                    ensure_text_block_id(blk_list[index])
                    for index, _metrics in ambiguous_candidates
                ]
                diagnostic = {
                    "kind": "one_region_multiple_blocks",
                    "region_bbox_xyxy": list(region.bbox_xyxy),
                    "candidate_block_ids": block_ids,
                }
                self.last_merge_split_diagnostics.append(diagnostic)
                self.last_shadow_regions.append(
                    self._serialize_shadow_region(
                        region,
                        reason="one_region_multiple_blocks",
                        candidate_block_ids=block_ids,
                    )
                )
                for block_index, _metrics in ambiguous_candidates:
                    block = blk_list[block_index]
                    block.merge_split_diagnostics = {
                        **dict(
                            getattr(block, "merge_split_diagnostics", {}) or {}
                        ),
                        "mangalmm_one_region_multiple_blocks": diagnostic,
                    }
                continue

            selected = (
                precision_candidates[0]
                if len(precision_candidates) == 1
                else candidates[0]
            )
            blk_index, match_metrics = selected
            tentative[blk_index].append(
                {
                    "region": region,
                    "metrics": match_metrics,
                }
            )

        for blk_index, items in tentative.items():
            if not items:
                continue
            unique_items = self._dedupe_exact_region_assignments(items)
            if len(unique_items) == 1:
                assignments[blk_index] = unique_items
                continue

            block = blk_list[blk_index]
            block_id = ensure_text_block_id(block)
            diagnostic = {
                "kind": "multiple_regions_one_block",
                "block_id": block_id,
                "region_bboxes_xyxy": [
                    list(item["region"].bbox_xyxy)
                    for item in unique_items
                ],
                "region_count": len(unique_items),
            }
            self.last_merge_split_diagnostics.append(diagnostic)
            block.merge_split_diagnostics = {
                **dict(
                    getattr(block, "merge_split_diagnostics", {}) or {}
                ),
                "mangalmm_multiple_regions_one_block": diagnostic,
            }
            for item in unique_items:
                self.last_shadow_regions.append(
                    self._serialize_shadow_region(
                        item["region"],
                        reason="multiple_regions_one_block",
                        candidate_block_ids=[block_id],
                    )
                )
        return assignments

    @staticmethod
    def _dedupe_exact_region_assignments(
        items: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        unique: list[dict[str, object]] = []
        seen: set[tuple[object, ...]] = set()
        for item in items:
            region: OCRRegion = item["region"]
            key = (
                tuple(float(value) for value in region.bbox_xyxy_float),
                tuple(float(value) for value in region.response_bbox_2d or []),
                str(region.raw_text or region.text),
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(item)
        return unique

    @staticmethod
    def _serialize_shadow_region(
        region: OCRRegion,
        *,
        reason: str,
        candidate_block_ids: list[str] | None = None,
    ) -> dict[str, object]:
        return {
            "reason": str(reason),
            "bbox_xyxy": list(region.bbox_xyxy),
            "bbox_xyxy_float": list(region.bbox_xyxy_float),
            "response_bbox_2d": list(region.response_bbox_2d or []),
            "text": region.text,
            "raw_text": region.raw_text,
            "candidate_block_ids": list(candidate_block_ids or []),
        }

    def _apply_assignments_to_blocks(
        self,
        blk_list: list[TextBlock],
        assignments: dict[int, list[dict[str, object]]],
        *,
        attempt_count: int,
        success_status: str,
        empty_status: str,
        page_bbox: tuple[int, int, int, int],
        resize_plan: ResizePlan,
    ) -> dict:
        page_bbox_list = [int(value) for value in page_bbox]
        for index, blk in enumerate(blk_list):
            items = assignments.get(index, [])
            if not items:
                self._mark_empty(
                    blk,
                    "MangaLMM full-page OCR did not match any OCR region to this block.",
                    attempt_count=attempt_count,
                    status=empty_status,
                    crop_bbox=page_bbox_list,
                    crop_source="page_full",
                    resize_scale=resize_plan.base_scale,
                )
                continue

            unique_items = self._dedupe_exact_region_assignments(items)
            if len(unique_items) != 1:
                block_id = ensure_text_block_id(blk)
                diagnostic = {
                    "kind": "multiple_regions_one_block",
                    "block_id": block_id,
                    "region_count": len(unique_items),
                }
                self.last_merge_split_diagnostics.append(diagnostic)
                blk.merge_split_diagnostics = {
                    **dict(
                        getattr(blk, "merge_split_diagnostics", {}) or {}
                    ),
                    "mangalmm_multiple_regions_one_block": diagnostic,
                }
                self._mark_empty(
                    blk,
                    "MangaLMM returned multiple distinct OCR regions for one "
                    "detector block; automatic text concatenation was refused.",
                    attempt_count=attempt_count,
                    status=empty_status,
                    crop_bbox=page_bbox_list,
                    crop_source="page_full",
                    resize_scale=resize_plan.base_scale,
                )
                continue

            source_item = unique_items[0]
            source_region: OCRRegion = source_item["region"]
            final_text = str(source_region.text or "").strip()
            final_raw_text = str(
                source_region.raw_text or source_region.text or ""
            ).strip()
            blk.texts = [final_text]
            blk.text = final_text
            blk.ocr_regions = [
                self._serialize_region_assignment(item["region"], item["metrics"])
                for item in unique_items
            ]
            blk.ocr_geometry_provenance = {
                **dict(
                    getattr(blk, "ocr_geometry_provenance", {}) or {}
                ),
                "matched_region_count": len(unique_items),
                "matched_region_bboxes_xyxy": [
                    list(item["region"].bbox_xyxy)
                    for item in unique_items
                ],
                "detector_geometry_authoritative": True,
            }
            blk.ocr_crop_bbox = list(source_region.unit_bbox_xyxy)
            blk.ocr_resize_scale = float(source_region.unit_resize_scale)
            set_block_ocr_crop_diagnostics(
                blk,
                effective_crop_xyxy=source_region.unit_bbox_xyxy,
                crop_source=source_region.unit_kind,
            )
            set_block_ocr_diagnostics(
                blk,
                text=final_text,
                confidence=0.0,
                status=success_status,
                empty_reason="",
                attempt_count=attempt_count,
                raw_text=final_raw_text,
                sanitized_text=final_text,
            )

        return summarize_ocr_quality(blk_list)

    @staticmethod
    def _serialize_region_assignment(
        region: OCRRegion,
        metrics: dict[str, float | bool],
    ) -> dict[str, object]:
        return {
            "bbox_xyxy": list(region.bbox_xyxy),
            "bbox_xyxy_float": list(region.bbox_xyxy_float),
            "text": region.text,
            "raw_text": region.raw_text,
            "unit_bbox_xyxy": list(region.unit_bbox_xyxy),
            "unit_kind": region.unit_kind,
            "unit_resize_scale": float(region.unit_resize_scale),
            "edge_distance": float(region.edge_distance),
            "normalized_text": region.normalized_text,
            "response_bbox_2d": list(region.response_bbox_2d or []),
            "scale_x": float(region.scale_x),
            "scale_y": float(region.scale_y),
            "request_shape": list(region.request_shape or []),
            "resize_profile": region.resize_profile,
            "match_metrics": {
                "ownership_cover": float(metrics["ownership_cover"]),
                "precision_cover": float(metrics["precision_cover"]),
                "ownership_iou": float(metrics["ownership_iou"]),
                "center_in_ownership": bool(metrics["center_in_ownership"]),
                "center_in_precision": bool(metrics["center_in_precision"]),
                "center_distance_norm": float(metrics["center_distance_norm"]),
                "precision_area": int(metrics["precision_area"]),
            },
        }

    def _select_best_block(
        self,
        region: OCRRegion,
        blk_list: list[TextBlock],
    ) -> tuple[int, dict[str, float | bool]] | None:
        candidates = self._candidate_block_matches(region, blk_list)
        return candidates[0] if candidates else None

    def _candidate_block_matches(
        self,
        region: OCRRegion,
        blk_list: list[TextBlock],
    ) -> list[tuple[int, dict[str, float | bool]]]:
        region_box = tuple(region.bbox_xyxy)
        candidates: list[tuple[tuple[float, int, float, float, float, int], int, dict[str, float | bool]]] = []
        for index, blk in enumerate(blk_list):
            metrics = self._compute_match_metrics(region_box, blk)
            if metrics is None:
                continue
            sort_key = (
                -float(metrics["ownership_cover"]),
                -int(bool(metrics["center_in_ownership"])),
                -float(metrics["precision_cover"]),
                -float(metrics["ownership_iou"]),
                float(metrics["center_distance_norm"]),
                int(metrics["precision_area"]),
            )
            candidates.append((sort_key, index, metrics))

        if not candidates:
            return []

        candidates.sort(key=lambda item: item[0])
        return [
            (index, metrics)
            for _sort_key, index, metrics in candidates
            if self._accept_match(blk_list[index], metrics)
        ]

    def _compute_match_metrics(
        self,
        region_box: tuple[int, int, int, int],
        blk: TextBlock,
    ) -> dict[str, float | bool] | None:
        support_box = self._support_box(blk)
        ownership_box = self._ownership_box(blk)
        precision_box = self._precision_box(blk)
        if support_box is None or ownership_box is None or precision_box is None:
            return None

        region_area = self._bbox_area(region_box)
        if region_area <= 0:
            return None

        support_intersection = self._intersection_area(region_box, support_box)
        center = self._bbox_center(region_box)
        center_in_support = self._point_in_box(center, support_box)
        if not center_in_support and (support_intersection / float(region_area)) < 0.10:
            return None

        ownership_cover = self._intersection_area(region_box, ownership_box) / float(region_area)
        precision_cover = self._intersection_area(region_box, precision_box) / float(region_area)
        ownership_iou = self._bbox_iou(region_box, ownership_box)
        center_in_ownership = self._point_in_box(center, ownership_box)
        center_in_precision = self._point_in_box(center, precision_box)
        precision_center = self._bbox_center(precision_box)
        precision_diag = max(1.0, self._bbox_diagonal(precision_box))
        center_distance_norm = self._distance(center, precision_center) / precision_diag

        return {
            "ownership_cover": ownership_cover,
            "precision_cover": precision_cover,
            "ownership_iou": ownership_iou,
            "center_in_ownership": center_in_ownership,
            "center_in_precision": center_in_precision,
            "center_distance_norm": center_distance_norm,
            "precision_area": self._bbox_area(precision_box),
        }

    def _accept_match(self, blk: TextBlock, metrics: dict[str, float | bool]) -> bool:
        bubble_box = getattr(blk, "bubble_xyxy", None)
        ownership_cover = float(metrics["ownership_cover"])
        precision_cover = float(metrics["precision_cover"])
        ownership_iou = float(metrics["ownership_iou"])
        center_in_ownership = bool(metrics["center_in_ownership"])
        center_in_precision = bool(metrics["center_in_precision"])
        if bubble_box is not None:
            return ownership_cover >= 0.35 or (center_in_ownership and precision_cover >= 0.20)
        return precision_cover >= 0.35 or (center_in_precision and ownership_iou >= 0.10)

    def _resize_for_request(self, image: np.ndarray, resize_plan: ResizePlan) -> np.ndarray:
        request_h, request_w = resize_plan.request_shape
        orig_h, orig_w = resize_plan.original_shape
        if request_h == orig_h and request_w == orig_w:
            return image
        return cv2.resize(image, (request_w, request_h), interpolation=cv2.INTER_AREA)

    def _map_regions_to_page_coords(
        self,
        regions: list[dict[str, object]],
        unit_bbox: tuple[int, int, int, int],
        unit_shape: tuple[int, int],
        resize_plan: ResizePlan,
        unit_kind: str,
    ) -> list[OCRRegion]:
        unit_x1, unit_y1, unit_x2, unit_y2 = [int(v) for v in unit_bbox]
        unit_h, unit_w = unit_shape
        mapped: list[OCRRegion] = []

        for region in regions:
            bbox = region.get("bbox_2d")
            text = str(region.get("text_content", "") or "").strip()
            raw_text = str(region.get("raw_text_content", text) or "").strip()
            if not isinstance(bbox, list) or len(bbox) != 4 or not text:
                continue
            x1_f, y1_f, x2_f, y2_f = [float(value) for value in bbox]
            resp_x1 = min(x1_f, x2_f)
            resp_y1 = min(y1_f, y2_f)
            resp_x2 = max(x1_f, x2_f)
            resp_y2 = max(y1_f, y2_f)

            local_x1 = max(0.0, min(resp_x1 / resize_plan.scale_x, float(unit_w)))
            local_y1 = max(0.0, min(resp_y1 / resize_plan.scale_y, float(unit_h)))
            local_x2 = max(0.0, min(resp_x2 / resize_plan.scale_x, float(unit_w)))
            local_y2 = max(0.0, min(resp_y2 / resize_plan.scale_y, float(unit_h)))
            if local_x2 <= local_x1 or local_y2 <= local_y1:
                continue

            page_x1_f = max(float(unit_x1), min(float(unit_x1) + local_x1, float(unit_x2)))
            page_y1_f = max(float(unit_y1), min(float(unit_y1) + local_y1, float(unit_y2)))
            page_x2_f = max(float(unit_x1), min(float(unit_x1) + local_x2, float(unit_x2)))
            page_y2_f = max(float(unit_y1), min(float(unit_y1) + local_y2, float(unit_y2)))
            if page_x2_f <= page_x1_f or page_y2_f <= page_y1_f:
                continue

            page_x1 = max(unit_x1, min(int(round(page_x1_f)), unit_x2))
            page_y1 = max(unit_y1, min(int(round(page_y1_f)), unit_y2))
            page_x2 = max(unit_x1, min(int(round(page_x2_f)), unit_x2))
            page_y2 = max(unit_y1, min(int(round(page_y2_f)), unit_y2))
            if page_x2 <= page_x1 or page_y2 <= page_y1:
                continue

            edge_distance = self._edge_distance_to_box((page_x1, page_y1, page_x2, page_y2), unit_bbox)
            mapped.append(
                OCRRegion(
                    bbox_xyxy=[page_x1, page_y1, page_x2, page_y2],
                    bbox_xyxy_float=[page_x1_f, page_y1_f, page_x2_f, page_y2_f],
                    text=text,
                    unit_bbox_xyxy=[unit_x1, unit_y1, unit_x2, unit_y2],
                    unit_kind=unit_kind,
                    unit_resize_scale=float(resize_plan.base_scale),
                    edge_distance=edge_distance,
                    normalized_text=self._normalize_text_key(text),
                    raw_text=raw_text,
                    response_bbox_2d=[resp_x1, resp_y1, resp_x2, resp_y2],
                    scale_x=float(resize_plan.scale_x),
                    scale_y=float(resize_plan.scale_y),
                    request_shape=[int(value) for value in resize_plan.request_shape],
                    resize_profile=resize_plan.profile,
                )
            )

        return mapped

    def _prompt_for_resize_plan(self, resize_plan: ResizePlan) -> tuple[str, str]:
        if resize_plan.profile == "dense":
            return ("dense_grounding_json", self.DENSE_PROMPT)
        return ("standard_grounding", self.STANDARD_PROMPT)

    def _request_response_text(
        self,
        image: np.ndarray,
        *,
        max_completion_tokens: int,
        prompt_text: str,
    ) -> str:
        data_url = self._image_data_url(image)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "min_p": self.min_p,
            "repeat_penalty": self.repeat_penalty,
            "repeat_last_n": self.repeat_last_n,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "max_completion_tokens": int(max_completion_tokens),
        }
        data = self._send_request(payload)
        self.last_transport_metadata = (
            self._extract_transport_metadata(data)
        )
        response_text = self._extract_text_from_response(data)
        return response_text

    @staticmethod
    def _extract_transport_metadata(data: dict) -> dict[str, object]:
        choices = data.get("choices")
        first_choice = (
            choices[0]
            if isinstance(choices, list)
            and choices
            and isinstance(choices[0], dict)
            else {}
        )
        usage = data.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        def token_count(key: str) -> int:
            try:
                return max(0, int(usage.get(key, 0) or 0))
            except (TypeError, ValueError):
                return 0

        return {
            "finish_reason": str(
                first_choice.get("finish_reason", "") or ""
            ),
            "prompt_tokens": token_count("prompt_tokens"),
            "completion_tokens": token_count("completion_tokens"),
            "total_tokens": token_count("total_tokens"),
        }

    def _send_request(self, payload: dict) -> dict:
        try:
            response = requests.post(
                self._chat_completions_url(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self._requests_timeout(),
            )
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                f"Unable to reach the local {SERVICE_NAME} service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        if response.status_code != 200:
            message = f"{SERVICE_NAME} service returned HTTP {response.status_code}."
            try:
                payload_json = response.json()
            except ValueError:
                payload_json = None
            if isinstance(payload_json, dict):
                error = payload_json.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message", "") or "").strip()
                    if detail:
                        message = detail
            raise LocalServiceResponseError(
                message,
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned invalid JSON.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        if not isinstance(data, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service response must be a JSON object.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        if self.raw_response_logging:
            append_active_raw_response(
                "mangalmm",
                data,
                kind="response_json",
            )
        return data

    def _requests_timeout(self):
        return float(self.request_timeout_sec)

    def _extract_text_from_response(self, data: dict) -> str:
        error = data.get("error")
        if isinstance(error, dict):
            error_message = str(error.get("message", "") or f"Unknown {SERVICE_NAME} error.")
            raise LocalServiceResponseError(
                error_message,
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} response did not include choices.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} response choice must be a JSON object.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} response did not include a message payload.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()

        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "") or ""))
            return "\n".join(part for part in parts if part).strip()

        return str(content or "").strip()

    def _parse_region_payload(self, text: str) -> list[dict[str, object]]:
        return self._analyze_region_payload(text)["regions"]

    def _analyze_region_payload(self, text: str) -> dict[str, object]:
        raw = str(text or "").strip()
        try:
            parsed = parse_mangalmm_response(raw)
        except MangaLMMResponseContractError as exc:
            return {
                "regions": [],
                "response_kind": f"parser_error:{exc.code}",
                "payload_type": "invalid",
                "raw_length": len(raw),
                "parser_error_code": exc.code,
                "parser_error": str(exc),
            }
        return {
            "regions": [dict(region) for region in parsed.regions],
            "response_kind": parsed.response_kind,
            "payload_type": parsed.payload_type,
            "raw_length": len(raw),
            "parser_error_code": "",
            "parser_error": "",
        }

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(v) for v in bbox]
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

    def _image_data_url(self, image: np.ndarray) -> str:
        image_bytes = self._encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/png;base64,{image_b64}"

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(
            ".png",
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), int(DEFAULT_MANGALMM_PNG_COMPRESSION)],
        )
        if not success:
            raise LocalServiceResponseError(
                f"Failed to encode OCR image for {SERVICE_NAME}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        return encoded.tobytes()

    def _chat_completions_url(self) -> str:
        base = (self.server_url or DEFAULT_MANGALMM_SERVER_URL).rstrip("/")
        if base.endswith(self.REQUEST_ENDPOINT_SUFFIX):
            return base
        return f"{base}{self.REQUEST_ENDPOINT_SUFFIX}"

    def _runtime_identity_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "contract_schema_version": MANGALMM_RESPONSE_SCHEMA_VERSION,
                "endpoint": self._chat_completions_url(),
                "model": MANGALMM_MODEL_NAME,
            }
        )

    def _mark_empty(
        self,
        blk: TextBlock,
        reason: str,
        *,
        attempt_count: int,
        status: str,
        crop_bbox: list[int] | None = None,
        crop_source: str = "",
        resize_scale: float = 1.0,
    ) -> None:
        blk.texts = []
        blk.text = ""
        blk.ocr_regions = []
        blk.ocr_crop_bbox = list(crop_bbox) if crop_bbox is not None else None
        blk.ocr_resize_scale = float(resize_scale or 1.0)
        if crop_bbox is not None and crop_source:
            set_block_ocr_crop_diagnostics(
                blk,
                effective_crop_xyxy=crop_bbox,
                crop_source=crop_source,
            )
        set_block_ocr_diagnostics(
            blk,
            text="",
            confidence=0.0,
            status=status,
            empty_reason=reason,
            attempt_count=attempt_count,
            raw_text="",
            sanitized_text="",
        )

    def _ownership_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        bubble = getattr(blk, "bubble_xyxy", None)
        if bubble is not None:
            return self._normalize_box(bubble)
        return self._normalize_box(getattr(blk, "xyxy", None))

    def _precision_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        return self._normalize_box(getattr(blk, "xyxy", None))

    def _support_box(self, blk: TextBlock) -> tuple[int, int, int, int] | None:
        ownership = self._ownership_box(blk)
        precision = self._precision_box(blk)
        if ownership is None:
            return precision
        if precision is None:
            return ownership
        return (
            min(ownership[0], precision[0]),
            min(ownership[1], precision[1]),
            max(ownership[2], precision[2]),
            max(ownership[3], precision[3]),
        )

    @staticmethod
    def _normalize_box(box) -> tuple[int, int, int, int] | None:
        if box is None:
            return None
        if len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    @staticmethod
    def _bbox_area(box: tuple[int, int, int, int]) -> int:
        width = max(0, int(box[2]) - int(box[0]))
        height = max(0, int(box[3]) - int(box[1]))
        return width * height

    @staticmethod
    def _intersection_area(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
        x1 = max(int(a[0]), int(b[0]))
        y1 = max(int(a[1]), int(b[1]))
        x2 = min(int(a[2]), int(b[2]))
        y2 = min(int(a[3]), int(b[3]))
        if x2 <= x1 or y2 <= y1:
            return 0
        return (x2 - x1) * (y2 - y1)

    def _bbox_iou(self, a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
        inter = self._intersection_area(a, b)
        if inter <= 0:
            return 0.0
        union = self._bbox_area(a) + self._bbox_area(b) - inter
        return inter / float(max(union, 1))

    @staticmethod
    def _bbox_center(box: tuple[int, int, int, int]) -> tuple[float, float]:
        return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)

    @staticmethod
    def _point_in_box(point: tuple[float, float], box: tuple[int, int, int, int]) -> bool:
        return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]

    @staticmethod
    def _bbox_diagonal(box: tuple[int, int, int, int]) -> float:
        return math.hypot(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    @staticmethod
    def _edge_distance_to_box(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> float:
        left = max(0.0, float(inner[0]) - float(outer[0]))
        top = max(0.0, float(inner[1]) - float(outer[1]))
        right = max(0.0, float(outer[2]) - float(inner[2]))
        bottom = max(0.0, float(outer[3]) - float(inner[3]))
        return min(left, top, right, bottom)

    @staticmethod
    def _normalize_text_key(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "")).strip()

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))

    @staticmethod
    def _read_env_int(
        env_name: str,
        config_value,
        default: int,
        *,
        bounds: tuple[int, int],
    ) -> int:
        raw = os.getenv(env_name, "").strip()
        candidate = raw if raw else config_value
        try:
            parsed = int(candidate)
        except (TypeError, ValueError):
            parsed = default
        low, high = bounds
        return max(low, min(parsed, high))

    @staticmethod
    def _read_env_float(
        env_name: str,
        config_value,
        default: float,
        *,
        minimum: float,
        maximum: float,
    ) -> float:
        raw = os.getenv(env_name, "").strip()
        candidate = raw if raw else config_value
        try:
            parsed = float(candidate)
        except (TypeError, ValueError):
            parsed = default
        return max(minimum, min(parsed, maximum))

    def _export_debug_artifact(
        self,
        *,
        blk: TextBlock | None,
        failure_reason: str,
        response_kind: str,
        raw_text: str,
        crop_bbox,
        crop_source: str,
        resize_plan: ResizePlan,
        crop_image: np.ndarray | None,
        request_image: np.ndarray | None,
        analysis: dict[str, object],
        unit_bbox=None,
        unit_kind: str = "",
    ) -> None:
        debug_root = self.debug_root
        if self._debug_root_from_env:
            active_root = active_debug_runtime_directory(
                "mangalmm-engine"
            )
            debug_root = Path(active_root) if active_root else None
        if debug_root is None:
            return
        with self._debug_export_lock:
            export_index = next(self._debug_export_counter)
        if export_index > self.debug_export_limit:
            return

        safe_kind = sanitize_debug_component(
            response_kind,
            fallback="response",
        )
        artifact_dir = debug_root / f"{export_index:04d}_{safe_kind}"
        metadata = {
            "failure_reason": failure_reason,
            "response_kind": response_kind,
            "raw_text": str(raw_text or ""),
            "crop_bbox": self._coords_or_none(crop_bbox),
            "crop_source": crop_source,
            "resize_profile": resize_plan.profile,
            "base_scale": float(resize_plan.base_scale),
            "scale_x": float(resize_plan.scale_x),
            "scale_y": float(resize_plan.scale_y),
            "original_shape": list(resize_plan.original_shape),
            "request_shape": list(resize_plan.request_shape),
            "max_completion_tokens": int(resize_plan.max_completion_tokens),
            "block_count": int(resize_plan.block_count),
            "small_block_ratio": float(resize_plan.small_block_ratio),
            "text_cover_ratio": float(resize_plan.text_cover_ratio),
            "crop_shape": self._image_shape_or_none(crop_image),
            "request_image_shape": self._image_shape_or_none(request_image),
            "block_xyxy": self._coords_or_none(getattr(blk, "xyxy", None)) if blk is not None else None,
            "bubble_xyxy": self._coords_or_none(getattr(blk, "bubble_xyxy", None)) if blk is not None else None,
            "source_lang": str(getattr(blk, "source_lang", "") or "") if blk is not None else "",
            "direction": str(getattr(blk, "direction", "") or "") if blk is not None else "",
            "unit_bbox": self._coords_or_none(unit_bbox),
            "unit_kind": unit_kind,
            "analysis": analysis,
        }
        try:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            atomic_debug_json(
                artifact_dir,
                "meta.json",
                metadata,
            )
            if crop_image is not None and crop_image.size > 0:
                atomic_debug_file(
                    artifact_dir,
                    "crop.png",
                    lambda path: cv2.imwrite(path, crop_image),
                )
            if request_image is not None and request_image.size > 0:
                atomic_debug_file(
                    artifact_dir,
                    "request.png",
                    lambda path: cv2.imwrite(path, request_image),
                )
        except Exception:
            logger.warning(
                "MangaLMM diagnostic artifact export failed open.",
                exc_info=True,
            )

    @staticmethod
    def _coords_or_none(value) -> list[int] | None:
        if value is None:
            return None
        try:
            coords = [int(float(item)) for item in value]
        except Exception:
            return None
        return coords if len(coords) == 4 else None

    @staticmethod
    def _image_shape_or_none(image: np.ndarray | None) -> list[int] | None:
        if image is None:
            return None
        try:
            height, width = image.shape[:2]
        except Exception:
            return None
        return [int(height), int(width)]
