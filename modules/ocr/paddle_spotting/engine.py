from __future__ import annotations

import base64
import logging
import threading
import time
from collections import Counter
from contextlib import nullcontext
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from modules.ocr.base import OCREngine
from modules.ocr.persistent_cache import canonical_sha256
from modules.ocr.common.result_contract import (
    OCR_STRATEGY_PADDLE_SPOTTING,
    PROCESSING_ACTION_REVIEW,
    SEMANTIC_ROLE_AMBIGUOUS,
    assign_ocr_processing_contract,
    finalize_ocr_processing_contracts,
    initialize_ocr_result_contract,
)
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    OperationCancelledError,
)
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_AFTER_RETRY,
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    OCR_STATUS_OK_AFTER_RETRY,
    OCR_EMPTY_REASON_SPOTTING_UNMATCHED,
    ensure_three_channel,
    set_block_ocr_crop_diagnostics,
    set_block_ocr_diagnostics,
)
from modules.utils.ocr_quality import summarize_ocr_quality
from modules.utils.textblock import TextBlock, ensure_text_block_id

from .reconciliation import SpottingGeometryResult, assign_spotting_regions
from .image_policy import (
    PADDLE_SPOTTING_OFFICIAL_IMAGE_MAX_PIXELS,
    preprocess_spotting_image,
)
from .response_parser import (
    PADDLE_SPOTTING_RESPONSE_SCHEMA_VERSION,
    PaddleSpottingParsedResponse,
    PaddleSpottingResponseContractError,
    parse_paddle_spotting_response,
)


logger = logging.getLogger(__name__)

SERVICE_NAME = "PaddleOCR VL Spotting"
SETTINGS_PAGE_NAME = "PaddleOCR VL Spotting Settings"


class PaddleOCRVLSpottingEngine(OCREngine):
    """Official full-page PaddleOCR-VL Spotting route.

    This route intentionally does not reuse the crop OCR endpoint, projector,
    cache identity, or fallback behavior. Detector geometry remains
    authoritative after native Spotting lines are mapped to blocks.
    """

    DEFAULT_SERVER_URL = (
        "http://127.0.0.1:18002/v1/chat/completions"
    )
    MODEL_IDENTITY = "PaddleOCR-VL-1.6-Spotting"
    OFFICIAL_PROMPT = "Spotting:"
    OFFICIAL_IMAGE_MAX_PIXELS = PADDLE_SPOTTING_OFFICIAL_IMAGE_MAX_PIXELS
    DEFAULT_MAX_COMPLETION_TOKENS = 3000
    DEFAULT_REQUEST_TIMEOUT_SEC = 360
    LOW_RES_DOUBLE_THRESHOLD = 1500
    PNG_COMPRESSION = 3
    PRIMARY_REPEAT_PENALTY = 1.0
    PRIMARY_REPEAT_LAST_N = 64
    RECOVERY_REPEAT_PENALTY = 1.15
    RECOVERY_REPEAT_LAST_N = 4096
    RESPONSE_SCHEMA_VERSION = PADDLE_SPOTTING_RESPONSE_SCHEMA_VERSION
    CACHE_IDENTITY_VERSION = "paddle_spotting_full_page_v2"
    RECONCILIATION_SCHEMA_VERSION = 2

    def __init__(self) -> None:
        self.server_url = self.DEFAULT_SERVER_URL
        self.max_completion_tokens = self.DEFAULT_MAX_COMPLETION_TOKENS
        self.request_timeout_sec = self.DEFAULT_REQUEST_TIMEOUT_SEC
        self.cancel_checker: Callable[[], bool] | None = None
        self.last_page_profile: dict[str, Any] = {}
        self.last_raw_regions: list[dict[str, Any]] = []
        self.last_shadow_regions: list[dict[str, Any]] = []
        self.last_ambiguous_regions: list[dict[str, Any]] = []
        self._session = requests.Session()
        self._request_lock = threading.Lock()
        self._inference_lease_factory: Callable[[], Any] | None = None

    def set_inference_lease_factory(
        self,
        factory: Callable[[], Any] | None,
    ) -> None:
        """Install a short Router inference lease for the HTTP call only."""

        self._inference_lease_factory = factory

    def _inference_lease(self) -> Any:
        factory = self._inference_lease_factory
        return factory() if callable(factory) else nullcontext()

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_paddleocr_vl_spotting_settings()
        self.server_url = (
            str(config.get("server_url", self.DEFAULT_SERVER_URL) or "")
            .strip()
            or self.DEFAULT_SERVER_URL
        )
        self.max_completion_tokens = self._clamp_int(
            config.get(
                "max_completion_tokens",
                self.DEFAULT_MAX_COMPLETION_TOKENS,
            ),
            default=self.DEFAULT_MAX_COMPLETION_TOKENS,
            minimum=512,
            maximum=4096,
        )
        self.request_timeout_sec = self._clamp_int(
            config.get(
                "request_timeout_sec",
                self.DEFAULT_REQUEST_TIMEOUT_SEC,
            ),
            default=self.DEFAULT_REQUEST_TIMEOUT_SEC,
            minimum=30,
            maximum=600,
        )
        self.last_page_profile = {}
        self.last_raw_regions = []
        self.last_shadow_regions = []
        self.last_ambiguous_regions = []

    def set_cancel_checker(
        self,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        self.cancel_checker = cancel_checker

    def _is_cancelled(self) -> bool:
        try:
            return bool(self.cancel_checker and self.cancel_checker())
        except Exception:
            return False

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            raise OperationCancelledError(
                "Cancelled while running PaddleOCR VL Spotting."
            )

    def process_image(
        self,
        img: np.ndarray,
        blk_list: list[TextBlock],
    ) -> list[TextBlock]:
        self._raise_if_cancelled()
        image = ensure_three_channel(img)
        image_height, image_width = image.shape[:2]
        runtime_identity = self._runtime_identity_fingerprint()
        for block in blk_list:
            initialize_ocr_result_contract(
                block,
                strategy=OCR_STRATEGY_PADDLE_SPOTTING,
                model_identity=self.MODEL_IDENTITY,
                runtime_identity=runtime_identity,
            )

        started = time.perf_counter()
        request_image, preprocess = self._preprocess_image(image)
        encoded_started = time.perf_counter()
        image_bytes = self._encode_png(request_image)
        encode_ms = (time.perf_counter() - encoded_started) * 1000.0
        data_url = (
            "data:image/png;base64,"
            + base64.b64encode(image_bytes).decode("ascii")
        )

        attempts: list[dict[str, Any]] = []
        primary = self._run_attempt(
            data_url,
            repeat_penalty=self.PRIMARY_REPEAT_PENALTY,
            repeat_last_n=self.PRIMARY_REPEAT_LAST_N,
            attempt_index=0,
        )
        attempts.append(primary)
        selected = primary
        if self._attempt_requires_recovery(primary):
            recovery = self._run_attempt(
                data_url,
                repeat_penalty=self.RECOVERY_REPEAT_PENALTY,
                repeat_last_n=self.RECOVERY_REPEAT_LAST_N,
                attempt_index=1,
            )
            attempts.append(recovery)
            selected = self._select_attempt(primary, recovery)

        parsed = selected.get("parsed")
        if not isinstance(parsed, PaddleSpottingParsedResponse):
            error = str(
                selected.get("parser_error")
                or selected.get("failure_reason")
                or "PaddleOCR-VL Spotting returned no parseable regions."
            )
            raise LocalServiceResponseError(
                error,
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        geometry = assign_spotting_regions(
            parsed.regions,
            blk_list,
            image_width=image_width,
            image_height=image_height,
        )
        self._apply_geometry(
            blk_list,
            geometry,
            attempt_count=len(attempts),
            image_width=image_width,
            image_height=image_height,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        quality = summarize_ocr_quality(blk_list)

        self.last_raw_regions = [
            {
                "text": region.text,
                "normalized_points": [
                    list(point) for point in region.normalized_points
                ],
                "source_line": region.source_line,
            }
            for region in parsed.regions
        ]
        self.last_shadow_regions = [
            {
                "text": region.text,
                "bbox_xyxy": list(region.bbox_xyxy),
                "normalized_points": [
                    list(point) for point in region.normalized_points
                ],
                "reason": "spotting_region_unmatched",
                "semantic_role": SEMANTIC_ROLE_AMBIGUOUS,
                "processing_action": PROCESSING_ACTION_REVIEW,
            }
            for region in geometry.unmatched_regions
        ]
        self.last_ambiguous_regions = [
            dict(item) for item in geometry.ambiguous_regions
        ]
        mapped_block_count = sum(
            bool(items) for items in geometry.assignments.values()
        )
        block_status_counts = Counter(
            str(item.get("status", "missing") or "missing")
            for item in geometry.block_diagnostics.values()
        )
        relation_type_counts = Counter(
            str(item.get("relation_type", "unknown") or "unknown")
            for item in geometry.relation_components
        )
        self.last_page_profile = {
            "schema_version": 2,
            "strategy": OCR_STRATEGY_PADDLE_SPOTTING,
            "official_contract": {
                "prompt": self.OFFICIAL_PROMPT,
                "image_max_pixels": self.OFFICIAL_IMAGE_MAX_PIXELS,
                "special_tokens": True,
                "response_schema_version": self.RESPONSE_SCHEMA_VERSION,
            },
            "preprocess": preprocess,
            "encode_ms": round(encode_ms, 3),
            "request_bytes": len(image_bytes),
            "attempt_count": len(attempts),
            "attempts": [
                {
                    key: value
                    for key, value in attempt.items()
                    if key not in {"parsed", "response"}
                }
                for attempt in attempts
            ],
            "selected_attempt": int(selected["attempt_index"]),
            "raw_region_count": len(parsed.regions),
            "duplicate_region_count": parsed.duplicate_region_count,
            "mapped_block_count": mapped_block_count,
            "unmapped_block_count": max(0, len(blk_list) - mapped_block_count),
            "unmatched_region_count": len(geometry.unmatched_regions),
            "ambiguous_region_count": len(geometry.ambiguous_regions),
            "pure_spotting": {
                "raw_region_count": len(parsed.regions),
                "duplicate_region_count": parsed.duplicate_region_count,
                "unmatched_region_count": len(geometry.unmatched_regions),
            },
            "detector_assisted_reconciliation": {
                "schema_version": self.RECONCILIATION_SCHEMA_VERSION,
                "block_status_counts": dict(sorted(block_status_counts.items())),
                "relation_type_counts": dict(
                    sorted(relation_type_counts.items())
                ),
                "ambiguous_block_count": len(
                    geometry.ambiguous_block_indices
                ),
                "components": [
                    dict(item) for item in geometry.relation_components
                ],
            },
            "quality": quality,
            "elapsed_ms": round(elapsed_ms, 3),
            "processing_contract": finalize_ocr_processing_contracts(
                blk_list
            ),
        }
        return blk_list

    def _preprocess_image(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        return preprocess_spotting_image(
            image,
            low_resolution_threshold=self.LOW_RES_DOUBLE_THRESHOLD,
        )

    def _run_attempt(
        self,
        data_url: str,
        *,
        repeat_penalty: float,
        repeat_last_n: int,
        attempt_index: int,
    ) -> dict[str, Any]:
        payload = {
            "model": self.MODEL_IDENTITY,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.OFFICIAL_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                }
            ],
            "temperature": 0,
            "seed": 42,
            "max_tokens": self.max_completion_tokens,
            "repeat_penalty": repeat_penalty,
            "repeat_last_n": repeat_last_n,
            "stream": False,
        }
        self._raise_if_cancelled()
        started = time.perf_counter()
        response = self._send_request(payload)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        content, finish_reason = self._extract_content(response)
        parser_error = ""
        parsed: PaddleSpottingParsedResponse | None = None
        try:
            parsed = parse_paddle_spotting_response(content)
        except PaddleSpottingResponseContractError as exc:
            parser_error = f"{exc.code}: {exc}"
        repetition_detected = self._has_repetition(content)
        usage = response.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return {
            "attempt_index": int(attempt_index),
            "repeat_penalty": float(repeat_penalty),
            "repeat_last_n": int(repeat_last_n),
            "finish_reason": finish_reason,
            "elapsed_ms": round(elapsed_ms, 3),
            "content_chars": len(content),
            "region_count": len(parsed.regions) if parsed else 0,
            "duplicate_region_count": (
                parsed.duplicate_region_count if parsed else 0
            ),
            "parser_error": parser_error,
            "repetition_detected": repetition_detected,
            "prompt_tokens": self._int_value(usage.get("prompt_tokens")),
            "completion_tokens": self._int_value(
                usage.get("completion_tokens")
            ),
            "total_tokens": self._int_value(usage.get("total_tokens")),
            "parsed": parsed,
            "response": response,
        }

    def _send_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            with self._request_lock:
                with self._inference_lease():
                    response = self._session.post(
                        self._chat_completions_url(),
                        json=payload,
                        headers={"Content-Type": "application/json"},
                        timeout=float(self.request_timeout_sec),
                    )
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                "Unable to reach the local PaddleOCR VL Spotting service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc
        if response.status_code != 200:
            detail = ""
            try:
                error_payload = response.json()
            except ValueError:
                error_payload = None
            if isinstance(error_payload, dict):
                error = error_payload.get("error")
                if isinstance(error, dict):
                    detail = str(error.get("message", "") or "").strip()
            raise LocalServiceResponseError(
                detail
                or (
                    "PaddleOCR VL Spotting service returned HTTP "
                    f"{response.status_code}."
                ),
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        try:
            data = response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                "PaddleOCR VL Spotting service returned invalid JSON.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc
        if not isinstance(data, dict):
            raise LocalServiceResponseError(
                "PaddleOCR VL Spotting response must be a JSON object.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        return data

    @staticmethod
    def _extract_content(response: dict[str, Any]) -> tuple[str, str]:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LocalServiceResponseError(
                "PaddleOCR VL Spotting response did not include choices.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        choice = choices[0]
        if not isinstance(choice, dict):
            raise LocalServiceResponseError(
                "PaddleOCR VL Spotting response choice is invalid.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        message = choice.get("message")
        if not isinstance(message, dict):
            raise LocalServiceResponseError(
                "PaddleOCR VL Spotting response has no message payload.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", "") or "")
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
        return (
            str(content or "").strip(),
            str(choice.get("finish_reason", "") or "").strip().lower(),
        )

    @classmethod
    def _attempt_requires_recovery(cls, attempt: dict[str, Any]) -> bool:
        return bool(
            attempt.get("parsed") is None
            or str(attempt.get("finish_reason") or "") == "length"
            or attempt.get("repetition_detected")
        )

    @classmethod
    def _select_attempt(
        cls,
        primary: dict[str, Any],
        recovery: dict[str, Any],
    ) -> dict[str, Any]:
        def score(attempt: dict[str, Any]) -> tuple[int, int, int]:
            parsed = attempt.get("parsed")
            complete = (
                isinstance(parsed, PaddleSpottingParsedResponse)
                and str(attempt.get("finish_reason") or "") != "length"
                and not bool(attempt.get("repetition_detected"))
            )
            return (
                int(complete),
                int(attempt.get("region_count", 0) or 0),
                -int(attempt.get("attempt_index", 0) or 0),
            )

        return max((primary, recovery), key=score)

    @staticmethod
    def _has_repetition(content: str) -> bool:
        lines = [
            line.strip()
            for line in str(content or "").replace("</s>", "").splitlines()
            if line.strip()
        ]
        if len(lines) < 8:
            return False
        counts = Counter(lines)
        if max(counts.values(), default=0) >= 4:
            return True
        if len(lines) >= 12 and len(counts) / len(lines) < 0.60:
            return True
        tail = lines[-8:]
        return len(set(tail)) <= 3

    def _apply_geometry(
        self,
        blocks: list[TextBlock],
        geometry: SpottingGeometryResult,
        *,
        attempt_count: int,
        image_width: int,
        image_height: int,
    ) -> None:
        status_ok = (
            OCR_STATUS_OK_AFTER_RETRY
            if attempt_count > 1
            else OCR_STATUS_OK
        )
        status_empty = (
            OCR_STATUS_EMPTY_AFTER_RETRY
            if attempt_count > 1
            else OCR_STATUS_EMPTY_INITIAL
        )
        page_bbox = [0, 0, int(image_width), int(image_height)]
        for block_index, block in enumerate(blocks):
            items = geometry.assignments.get(block_index, ())
            reconciliation = dict(
                geometry.block_diagnostics.get(block_index, {})
            )
            reconciliation_status = str(
                reconciliation.get("status", "missing") or "missing"
            )
            block.merge_split_diagnostics = {
                "schema_version": self.RECONCILIATION_SCHEMA_VERSION,
                "strategy": OCR_STRATEGY_PADDLE_SPOTTING,
                **reconciliation,
            }
            if not items:
                block.text = ""
                block.texts = []
                block.ocr_regions = []
                block.ocr_crop_bbox = list(page_bbox)
                block.ocr_resize_scale = 1.0
                assign_ocr_processing_contract(
                    block,
                    semantic_role=SEMANTIC_ROLE_AMBIGUOUS,
                    processing_action=PROCESSING_ACTION_REVIEW,
                    decision_source=(
                        "paddle_spotting_ambiguous"
                        if reconciliation_status == "ambiguous"
                        else "paddle_spotting_unmatched"
                    ),
                    reasons=(
                        "spotting_relation_ambiguous"
                        if reconciliation_status == "ambiguous"
                        else "spotting_region_unmatched",
                    ),
                )
                set_block_ocr_crop_diagnostics(
                    block,
                    effective_crop_xyxy=page_bbox,
                    crop_source="page_full_spotting",
                )
                set_block_ocr_diagnostics(
                    block,
                    text="",
                    confidence=0.0,
                    status=status_empty,
                    empty_reason=OCR_EMPTY_REASON_SPOTTING_UNMATCHED,
                    attempt_count=attempt_count,
                    raw_text="",
                    sanitized_text="",
                )
                continue

            texts = [item.region.text.strip() for item in items if item.region.text.strip()]
            final_text = "\n".join(texts).strip()
            block.text = final_text
            block.texts = texts
            block.ocr_regions = [
                {
                    "text": item.region.text,
                    "normalized_points": [
                        list(point)
                        for point in item.region.normalized_points
                    ],
                    "points": [list(point) for point in item.region.points],
                    "bbox_xyxy": list(item.region.bbox_xyxy),
                    "source_line": item.region.source_line,
                    "match_metrics": {
                        "region_coverage": item.region_coverage,
                        "block_coverage": item.block_coverage,
                        "iou": item.iou,
                        "center_inside": item.center_inside,
                    },
                }
                for item in items
            ]
            block.ocr_geometry_provenance = {
                **dict(
                    getattr(block, "ocr_geometry_provenance", {}) or {}
                ),
                "source": "paddle_spotting_normalized_quad",
                "detector_geometry_authoritative": True,
                "matched_region_count": len(items),
                "detector_block_id": ensure_text_block_id(block),
                "reconciliation_schema_version": (
                    self.RECONCILIATION_SCHEMA_VERSION
                ),
                "reconciliation_status": reconciliation_status,
                "detector_coverage": float(
                    reconciliation.get("detector_coverage", 0.0) or 0.0
                ),
            }
            block.ocr_crop_bbox = list(page_bbox)
            block.ocr_resize_scale = 1.0
            set_block_ocr_crop_diagnostics(
                block,
                effective_crop_xyxy=page_bbox,
                crop_source="page_full_spotting",
            )
            set_block_ocr_diagnostics(
                block,
                text=final_text,
                confidence=0.0,
                status=status_ok,
                empty_reason="",
                attempt_count=attempt_count,
                raw_text=final_text,
                sanitized_text=final_text,
            )

    def _chat_completions_url(self) -> str:
        parsed = urlparse(self.server_url)
        normalized_path = parsed.path.rstrip("/")
        if normalized_path.endswith("/v1/chat/completions"):
            return parsed._replace(
                path=normalized_path,
                params="",
                query="",
                fragment="",
            ).geturl()
        base = self.server_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def _runtime_identity_fingerprint(self) -> str:
        return canonical_sha256(
            {
                "cache_identity_version": self.CACHE_IDENTITY_VERSION,
                "endpoint": self._chat_completions_url(),
                "model": self.MODEL_IDENTITY,
                "prompt": self.OFFICIAL_PROMPT,
                "image_max_pixels": self.OFFICIAL_IMAGE_MAX_PIXELS,
                "special_tokens": True,
                "response_schema_version": self.RESPONSE_SCHEMA_VERSION,
            }
        )

    @classmethod
    def _encode_png(cls, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(
            ".png",
            image,
            [int(cv2.IMWRITE_PNG_COMPRESSION), cls.PNG_COMPRESSION],
        )
        if not success:
            raise LocalServiceResponseError(
                "Failed to encode a full-page image for PaddleOCR VL Spotting.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        return encoded.tobytes()

    @staticmethod
    def _clamp_int(
        value: Any,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return int(default)
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _int_value(value: Any) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0
