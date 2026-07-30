from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import re
import threading
import time
import unicodedata
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
    OperationCancelledError,
)
from modules.utils.gpu_metrics import query_gpu_metrics_cached
from modules.utils.ocr_debug import (
    OCR_EMPTY_REASON_NON_TEXT_RESPONSE,
    OCR_EMPTY_REASON_LAYOUT_SCHEMA_LABELS,
    OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE,
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    ensure_three_channel,
    resolve_block_crop_bbox,
    set_block_ocr_crop_diagnostics,
    set_block_ocr_diagnostics,
)
from modules.utils.text_normalization import (
    DECORATIVE_NOISE_GLYPHS,
    normalize_decorative_ocr_text,
)
from modules.utils.textblock import TextBlock

from .base import OCREngine
from .persistent_cache import (
    OCRPersistentResultCache,
    OCRResultCacheRecord,
    apply_raw_ocr_result,
    canonical_json,
    canonical_sha256,
    snapshot_raw_ocr_result,
)
from .result_contract import (
    OCR_STRATEGY_PADDLE_CROP,
    initialize_ocr_result_contract,
)


logger = logging.getLogger(__name__)


@dataclass
class PaddleOCRPersistentCachePlan:
    blocks: list[TextBlock]
    jobs: list[dict[str, Any]]
    page_profile: dict[str, Any]
    cache_enabled: bool
    deferred_outcomes: list[tuple[TextBlock, dict[str, Any]]] = field(
        default_factory=list
    )
    lookup_disabled: bool = False
    hit_count: int = 0
    miss_count: int = 0
    runtime_jobs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def requires_runtime(self) -> bool:
        return bool(self.runtime_jobs)

    @property
    def all_hit(self) -> bool:
        return (
            self.cache_enabled
            and not self.lookup_disabled
            and self.hit_count > 0
            and self.miss_count == 0
        )


class PaddleOCRVLEngine(OCREngine):
    PERFORMANCE_TELEMETRY_SCHEMA_VERSION = 1
    DEFAULT_SERVER_URL = "http://127.0.0.1:28118/layout-parsing"
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_PARALLEL_WORKERS = 8
    DEFAULT_SCHEDULER_MODE = "fixed_area_desc"
    MAX_NEW_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_TIMEOUT_SECONDS = 60
    REQUEST_RETRY_TOTAL_ATTEMPTS = 3
    REQUEST_RETRY_BACKOFF_SECONDS = (0.5, 1.5)
    TRANSIENT_HTTP_STATUS_CODES = frozenset({500, 502, 503, 504})
    TEXT_EXPANSION_RATIO = 0.03
    LARGE_CROP_RATIO_THRESHOLD = 0.02
    MEDIUM_CROP_RATIO_THRESHOLD = 0.008
    ALLOWED_SCHEDULER_MODES = frozenset({"fixed", "fixed_area_desc", "auto_v1"})
    MODEL_IDENTITY = "PaddleOCR-VL-1.6-0.9B"
    OCR_CACHE_VERSION = "paddleocr_vl_text_first_guard_v3"
    PERSISTENT_CACHE_KEY_SCHEMA_VERSION = 2
    CROP_SCHEMA_VERSION = "paddle_text_first_bubble_clamp_v2"
    ENCODER_SCHEMA_VERSION = "opencv_jpeg_default_v1"
    PARSER_SCHEMA_VERSION = "paddle_layout_response_v1"
    SANITIZER_SCHEMA_VERSION = "paddle_text_normalizer_v2"
    GUARD_SCHEMA_VERSION = OCR_CACHE_VERSION
    LAYOUT_SCHEMA_TOKENS = frozenset(
        {
            "number",
            "footnote",
            "header",
            "header_image",
            "footer",
            "footer_image",
            "aside_text",
            "ocr",
        }
    )
    STRONG_LAYOUT_SCHEMA_TOKENS = frozenset({"footnote", "header_image", "footer_image", "aside_text"})
    LAYOUT_SCHEMA_EMPTY_REASON = OCR_EMPTY_REASON_LAYOUT_SCHEMA_LABELS
    TEXT_FREE_NO_VISUAL_EVIDENCE_REASON = OCR_EMPTY_REASON_TEXT_FREE_NO_VISUAL_EVIDENCE
    NON_TEXT_RESPONSE_REASON = OCR_EMPTY_REASON_NON_TEXT_RESPONSE
    NON_TEXT_RESPONSE_PATTERNS = (
        re.compile(r"\b(?:no|not any|without)\s+(?:readable\s+)?(?:text|words?|letters?)\b", re.IGNORECASE),
        re.compile(r"\b(?:does not|doesn't)\s+contain\s+(?:any\s+)?(?:readable\s+)?(?:text|words?|letters?)\b", re.IGNORECASE),
        re.compile(r"\b(?:cannot|can't|unable to|not able to)\s+(?:read|recognize|determine|identify|extract)\b", re.IGNORECASE),
        re.compile(r"\b(?:image|picture|photo|crop).*\b(?:blurry|blurred|unclear|low[- ]resolution|too small|not clear)\b", re.IGNORECASE),
        re.compile(r"\b(?:blurry|blurred|unclear|too small|low[- ]resolution).*\b(?:image|text|words?|letters?)\b", re.IGNORECASE),
        re.compile(r"(?:이미지|사진).*(?:흐릿|불분명|해상도|작아|글자|텍스트|문자|판독|확인)", re.IGNORECASE),
        re.compile(r"(?:글자|텍스트|문자).*(?:없|보이지|확인.*어렵|판독.*어렵)", re.IGNORECASE),
        re.compile(r"(?:읽|인식|판독|확인).*(?:어렵|불가|없)", re.IGNORECASE),
    )
    TEXT_FREE_WATERMARK_PATTERNS = (
        re.compile(r"(?:https?://|www\.|@[\w.-]+|[\w.-]+\.(?:com|cc|ai|net|org|jp|kr|io|me)\b)", re.IGNORECASE),
        re.compile(
            r"\b(?:patreon|fanbox|pixiv|fantia|gumroad|ko[- ]?fi|subscribestar|twitter|instagram|discord|telegram|"
            r"scan(?:lation|lator)?|translated\s+by|translation\s+by|translator|typeset|redraw|proofread|commission|"
            r"gpt[0-9a-z.-]*|gemini|claude)\b",
            re.IGNORECASE,
        ),
    )
    TEXT_FREE_CATALOG_MARKER_PATTERN = re.compile(
        r"(?:[卷巻][之]?[一二三四五六七八九十百千0-9]+|\b(?:vol(?:ume)?|chapter|ch\.|part|page|p\.)\s*[#:\-]?\s*\d+\b)",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self.server_url = self.DEFAULT_SERVER_URL
        self.prettify_markdown = False
        self.visualize = False
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS
        self.parallel_workers = self.DEFAULT_PARALLEL_WORKERS
        self.scheduler_mode = self.DEFAULT_SCHEDULER_MODE
        self.last_page_profile: dict[str, Any] = {}
        self._supports_max_new_tokens = True
        self.cancel_checker: Callable[[], bool] | None = None
        self._request_telemetry_context = threading.local()

    def set_cancel_checker(self, cancel_checker: Callable[[], bool] | None) -> None:
        self.cancel_checker = cancel_checker

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_paddleocr_vl_settings()
        self.server_url = config.get("server_url", self.DEFAULT_SERVER_URL) or self.DEFAULT_SERVER_URL
        self.prettify_markdown = bool(config.get("prettify_markdown", False))
        self.visualize = bool(config.get("visualize", False))
        self.max_new_tokens = self._clamp_int(
            config.get("max_new_tokens", self.DEFAULT_MAX_NEW_TOKENS),
            self.DEFAULT_MAX_NEW_TOKENS,
            self.MAX_NEW_TOKENS_RANGE,
        )
        self.parallel_workers = self._clamp_int(
            config.get("parallel_workers", self.DEFAULT_PARALLEL_WORKERS),
            self.DEFAULT_PARALLEL_WORKERS,
            self.PARALLEL_WORKERS_RANGE,
        )
        self.scheduler_mode = self._resolve_scheduler_mode(settings)
        self.last_page_profile = {}
        self._supports_max_new_tokens = True

    def _is_cancelled(self) -> bool:
        try:
            return bool(self.cancel_checker and self.cancel_checker())
        except Exception:
            return False

    def _raise_if_cancelled(self) -> None:
        if self._is_cancelled():
            raise OperationCancelledError("Cancelled while running PaddleOCR VL.")

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        jobs: list[dict[str, Any]] = []
        page_height = int(img.shape[0]) if img is not None and len(img.shape) >= 2 else 0
        page_width = int(img.shape[1]) if img is not None and len(img.shape) >= 2 else 0
        page_area = max(page_height * page_width, 1)
        for job_index, blk in enumerate(blk_list or []):
            self._raise_if_cancelled()
            bbox = self._resolve_bbox(blk, img)
            if bbox is None:
                self._mark_empty(blk, "Invalid OCR crop bounds.")
                continue
            x1, y1, x2, y2 = bbox
            crop_area_px = max(0, (x2 - x1) * (y2 - y1))
            crop_area_ratio = crop_area_px / float(page_area)
            request_record = {
                "job_index": int(job_index),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_source": str(
                    getattr(blk, "ocr_crop_source", "") or ""
                ),
                "crop_area_px": int(crop_area_px),
                "crop_area_ratio": round(float(crop_area_ratio), 6),
                "enqueue_ts": None,
                "start_ts": None,
                "end_ts": None,
                "elapsed_ms": None,
                "queue_wait_ms": 0.0,
                "crop_ms": 0.0,
                "text_guard_ms": 0.0,
                "encode_ms": 0.0,
                "base64_ms": 0.0,
                "payload_build_ms": 0.0,
                "logical_request_count": 0,
                "http_attempt_count": 0,
                "http_retry_count": 0,
                "compatibility_retry_count": 0,
                "retry_backoff_ms": 0.0,
                "request_wall_ms": 0.0,
                "response_decode_ms": 0.0,
                "parse_sanitize_ms": 0.0,
                "request_bytes": 0,
                "base64_chars": 0,
                "status": "pending",
            }
            jobs.append(
                {
                    "job_index": int(job_index),
                    "block": blk,
                    "bbox": bbox,
                    "crop_area_px": int(crop_area_px),
                    "crop_area_ratio": float(crop_area_ratio),
                    "request_record": request_record,
                }
            )

        job_stats = self._summarize_jobs(jobs)
        gpu_metrics = self._resolve_gpu_metrics_for_scheduler()
        ordered_jobs = self._order_jobs(jobs)
        worker_count = self._resolve_worker_count(ordered_jobs, job_stats, gpu_metrics)
        self.last_page_profile = self._build_page_profile(
            ordered_jobs,
            page_width=page_width,
            page_height=page_height,
            job_stats=job_stats,
            worker_count=worker_count,
            gpu_metrics=gpu_metrics,
        )
        if not jobs:
            self._finalize_performance_profile()
            return blk_list

        logger.info(
            "paddleocr_vl start: blocks=%d workers=%d max_new_tokens=%d endpoint=%s scheduler=%s",
            len(ordered_jobs),
            worker_count,
            self.max_new_tokens,
            self.server_url,
            self.scheduler_mode,
        )

        started_at = time.perf_counter()
        self.last_page_profile["started_at"] = time.time()
        try:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            future_map = {}
            try:
                for job in ordered_jobs:
                    self._raise_if_cancelled()
                    job["request_record"]["enqueue_ts"] = time.time()
                    job["enqueue_perf"] = time.perf_counter()
                    future = executor.submit(self._process_job, img, job)
                    future_map[future] = job

                pending = set(future_map)
                while pending:
                    self._raise_if_cancelled()
                    done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                self._raise_if_cancelled()
            except Exception:
                for future in future_map:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
            self.last_page_profile["page_status"] = "ok"
        except Exception as exc:
            self.last_page_profile["page_status"] = "error"
            self.last_page_profile["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.last_page_profile["elapsed_ms"] = round(elapsed_ms, 3)
            self.last_page_profile["completed_at"] = time.time()
            self._finalize_performance_profile()

        logger.info(
            "paddleocr_vl complete: blocks=%d elapsed_ms=%.1f scheduler=%s",
            len(ordered_jobs),
            self.last_page_profile.get("elapsed_ms", 0.0),
            self.scheduler_mode,
        )
        return blk_list

    def prepare_persistent_cache(
        self,
        img: np.ndarray,
        blk_list: list[TextBlock],
        store: OCRPersistentResultCache,
        runtime_identity: dict[str, Any],
        *,
        lookup: bool = True,
    ) -> PaddleOCRPersistentCachePlan:
        """Prepare exact crop identities and restore hits before runtime startup."""

        jobs: list[dict[str, Any]] = []
        deferred_outcomes: list[tuple[TextBlock, dict[str, Any]]] = []
        page_height = int(img.shape[0]) if img is not None and len(img.shape) >= 2 else 0
        page_width = int(img.shape[1]) if img is not None and len(img.shape) >= 2 else 0
        page_area = max(page_height * page_width, 1)
        for job_index, blk in enumerate(blk_list or []):
            self._raise_if_cancelled()
            initialize_ocr_result_contract(
                blk,
                strategy=OCR_STRATEGY_PADDLE_CROP,
                model_identity=str(
                    runtime_identity.get("model_name")
                    or self.MODEL_IDENTITY
                ),
                runtime_identity=canonical_sha256(runtime_identity),
            )
            bbox = self._resolve_bbox(blk, img)
            if bbox is None:
                outcome_block = copy.copy(blk)
                self._mark_empty(outcome_block, "Invalid OCR crop bounds.")
                deferred_outcomes.append(
                    (blk, snapshot_raw_ocr_result(outcome_block))
                )
                continue
            x1, y1, x2, y2 = bbox
            crop_area_px = max(0, (x2 - x1) * (y2 - y1))
            crop_area_ratio = crop_area_px / float(page_area)
            record = {
                "job_index": int(job_index),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "crop_source": str(
                    getattr(blk, "ocr_crop_source", "") or ""
                ),
                "crop_area_px": int(crop_area_px),
                "crop_area_ratio": round(float(crop_area_ratio), 6),
                "enqueue_ts": None,
                "start_ts": None,
                "end_ts": None,
                "elapsed_ms": None,
                "queue_wait_ms": 0.0,
                "crop_ms": 0.0,
                "text_guard_ms": 0.0,
                "encode_ms": 0.0,
                "base64_ms": 0.0,
                "payload_build_ms": 0.0,
                "logical_request_count": 0,
                "http_attempt_count": 0,
                "http_retry_count": 0,
                "compatibility_retry_count": 0,
                "retry_backoff_ms": 0.0,
                "request_wall_ms": 0.0,
                "response_decode_ms": 0.0,
                "parse_sanitize_ms": 0.0,
                "request_bytes": 0,
                "base64_chars": 0,
                "cache_lookup": "pending",
                "status": "pending",
            }
            job = {
                "job_index": int(job_index),
                "block": blk,
                "bbox": bbox,
                "crop_area_px": int(crop_area_px),
                "crop_area_ratio": float(crop_area_ratio),
                "request_record": record,
            }
            crop_started_at = time.perf_counter()
            crop = self._crop_image(img, bbox)
            record["crop_ms"] = round(
                (time.perf_counter() - crop_started_at) * 1000.0,
                3,
            )
            if crop is None:
                outcome_block = copy.copy(blk)
                self._mark_empty(outcome_block, "Invalid OCR crop bounds.")
                job["prepared_outcome"] = snapshot_raw_ocr_result(outcome_block)
                record["status"] = "crop_invalid"
                jobs.append(job)
                continue

            guard_started_at = time.perf_counter()
            should_skip, evidence = self._should_skip_text_free_crop(blk, crop)
            record["text_guard_ms"] = round(
                (time.perf_counter() - guard_started_at) * 1000.0,
                3,
            )
            encode_started_at = time.perf_counter()
            image_bytes = self._encode_image(crop)
            record["encode_ms"] = round(
                (time.perf_counter() - encode_started_at) * 1000.0,
                3,
            )
            contiguous_crop = np.ascontiguousarray(crop)
            block_xyxy = [
                int(float(value))
                for value in getattr(blk, "xyxy", bbox)
            ]
            bubble_xyxy = getattr(blk, "bubble_xyxy", None)
            cache_identity = {
                "cache_key_schema_version": self.PERSISTENT_CACHE_KEY_SCHEMA_VERSION,
                "runtime": dict(runtime_identity),
                "raw_crop_pixel_sha256": hashlib.sha256(
                    contiguous_crop.tobytes(order="C")
                ).hexdigest(),
                "request_jpeg_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "crop_shape": [int(value) for value in contiguous_crop.shape],
                "crop_dtype": str(contiguous_crop.dtype),
                "block_xyxy": block_xyxy,
                "text_class": str(getattr(blk, "text_class", "") or ""),
                "bubble_present": bubble_xyxy is not None,
                "bubble_xyxy": (
                    [int(float(value)) for value in bubble_xyxy]
                    if bubble_xyxy is not None
                    else None
                ),
                "text_free_without_bubble": self._is_text_free_without_bubble(blk),
                "text_guard_evidence": evidence,
                "source_language": str(getattr(blk, "source_lang", "") or ""),
                "max_new_tokens": int(self.max_new_tokens),
                "prettify_markdown": bool(self.prettify_markdown),
                "visualize": bool(self.visualize),
                "crop_schema_version": self.CROP_SCHEMA_VERSION,
                "encoder_schema_version": self.ENCODER_SCHEMA_VERSION,
                "parser_schema_version": self.PARSER_SCHEMA_VERSION,
                "sanitizer_schema_version": self.SANITIZER_SCHEMA_VERSION,
                "guard_schema_version": self.GUARD_SCHEMA_VERSION,
            }
            job.update(
                {
                    "prepared_crop": crop,
                    "prepared_image_bytes": image_bytes,
                    "guard_should_skip": bool(should_skip),
                    "guard_evidence": evidence,
                    "cache_identity": cache_identity,
                    "cache_identity_json": canonical_json(cache_identity),
                    "cache_key": canonical_sha256(cache_identity),
                }
            )
            jobs.append(job)

        job_stats = self._summarize_jobs(jobs)
        gpu_metrics = self._resolve_gpu_metrics_for_scheduler()
        ordered_jobs = self._order_jobs(jobs)
        worker_count = self._resolve_worker_count(ordered_jobs, job_stats, gpu_metrics)
        profile = self._build_page_profile(
            ordered_jobs,
            page_width=page_width,
            page_height=page_height,
            job_stats=job_stats,
            worker_count=worker_count,
            gpu_metrics=gpu_metrics,
        )
        plan = PaddleOCRPersistentCachePlan(
            blocks=blk_list,
            jobs=ordered_jobs,
            page_profile=profile,
            cache_enabled=True,
            deferred_outcomes=deferred_outcomes,
        )
        keyed_jobs = [job for job in ordered_jobs if job.get("cache_key")]
        lookups = (
            store.lookup_many(job["cache_key"] for job in keyed_jobs)
            if lookup
            else {}
        )
        for job in keyed_jobs:
            self._raise_if_cancelled()
            record = job["request_record"]
            result = lookups.get(job["cache_key"]) if lookup else None
            if result is not None and result.disabled:
                plan.lookup_disabled = True
            if result is not None and result.hit:
                job["prepared_outcome"] = dict(result.result or {})
                record["cache_lookup"] = "hit"
                record["status"] = "cache_hit"
                plan.hit_count += 1
                job.pop("prepared_crop", None)
                job.pop("prepared_image_bytes", None)
                continue

            record["cache_lookup"] = "disabled" if plan.lookup_disabled else "miss"
            plan.miss_count += 1
            job["cache_lookup_miss"] = True
            if bool(job.get("guard_should_skip")):
                evidence = dict(job.get("guard_evidence") or {})
                outcome_block = copy.copy(job["block"])
                self._mark_empty(
                    outcome_block,
                    self.TEXT_FREE_NO_VISUAL_EVIDENCE_REASON,
                )
                reject_reason = str(
                    evidence.get("reject_reason") or "no_visual_text_evidence"
                )
                outcome_block.ocr_reject_reason = reject_reason
                job["prepared_outcome"] = snapshot_raw_ocr_result(outcome_block)
                record["status"] = "rejected_no_text_evidence"
                record["text_evidence"] = evidence
                record["non_text_reason"] = reject_reason
                job.pop("prepared_crop", None)
                job.pop("prepared_image_bytes", None)
                continue
            plan.runtime_jobs.append(job)

        plan.page_profile["persistent_cache"] = {
            "enabled": True,
            "lookup_disabled": bool(plan.lookup_disabled),
            "hit_count": int(plan.hit_count),
            "miss_count": int(plan.miss_count),
            "runtime_miss_count": len(plan.runtime_jobs),
        }
        return plan

    def process_persistent_cache_plan(
        self,
        plan: PaddleOCRPersistentCachePlan,
    ) -> list[TextBlock]:
        """Execute only exact-cache misses from a prepared page plan."""

        self.last_page_profile = plan.page_profile
        if not plan.runtime_jobs:
            self._raise_if_cancelled()
            self._commit_persistent_plan_outcomes(plan, {})
            self.last_page_profile["page_status"] = "ok"
            self.last_page_profile["elapsed_ms"] = 0.0
            self.last_page_profile["completed_at"] = time.time()
            self._finalize_performance_profile()
            return plan.blocks

        worker_count = max(
            1,
            min(
                int(self.last_page_profile.get("chosen_workers", self.parallel_workers)),
                len(plan.runtime_jobs),
            ),
        )
        logger.info(
            "paddleocr_vl persistent-cache misses: blocks=%d hits=%d workers=%d",
            len(plan.runtime_jobs),
            plan.hit_count,
            worker_count,
        )
        started_at = time.perf_counter()
        self.last_page_profile["started_at"] = time.time()
        try:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            future_map = {}
            try:
                for job in plan.runtime_jobs:
                    self._raise_if_cancelled()
                    job["request_record"]["enqueue_ts"] = time.time()
                    job["enqueue_perf"] = time.perf_counter()
                    future = executor.submit(self._process_prepared_job, job)
                    future_map[future] = job
                pending = set(future_map)
                outcomes: dict[int, dict[str, Any]] = {}
                while pending:
                    self._raise_if_cancelled()
                    done, pending = wait(
                        pending,
                        timeout=0.2,
                        return_when=FIRST_COMPLETED,
                    )
                    for future in done:
                        job = future_map[future]
                        outcomes[int(job["job_index"])] = future.result()
                self._raise_if_cancelled()
                self._commit_persistent_plan_outcomes(plan, outcomes)
            except Exception:
                for future in future_map:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                raise
            else:
                executor.shutdown(wait=True)
            self.last_page_profile["page_status"] = "ok"
        except Exception as exc:
            self.last_page_profile["page_status"] = "error"
            self.last_page_profile["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.last_page_profile["elapsed_ms"] = round(elapsed_ms, 3)
            self.last_page_profile["completed_at"] = time.time()
            self._finalize_performance_profile()
        return plan.blocks

    @staticmethod
    def _commit_persistent_plan_outcomes(
        plan: PaddleOCRPersistentCachePlan,
        runtime_outcomes: dict[int, dict[str, Any]],
    ) -> None:
        for block, outcome in plan.deferred_outcomes:
            apply_raw_ocr_result(block, outcome)
        for job in plan.jobs:
            job_index = int(job["job_index"])
            outcome = runtime_outcomes.get(job_index)
            if outcome is None:
                prepared = job.get("prepared_outcome")
                if isinstance(prepared, dict):
                    outcome = prepared
            if outcome is not None:
                apply_raw_ocr_result(job["block"], outcome)

    def build_persistent_cache_records(
        self,
        plan: PaddleOCRPersistentCachePlan,
    ) -> list[OCRResultCacheRecord]:
        if str(plan.page_profile.get("page_status", "") or "") != "ok":
            return []
        records: list[OCRResultCacheRecord] = []
        for job in plan.jobs:
            if not job.get("cache_lookup_miss") or not job.get("cache_key"):
                continue
            status = str(job.get("request_record", {}).get("status", "") or "")
            if not status or status == "pending" or status.startswith("error:"):
                continue
            records.append(
                OCRResultCacheRecord(
                    cache_key=str(job["cache_key"]),
                    identity_json=str(job["cache_identity_json"]),
                    result_json=canonical_json(
                        snapshot_raw_ocr_result(job["block"])
                    ),
                )
            )
        return records

    def _process_prepared_job(self, job: dict[str, Any]) -> dict[str, Any]:
        record = job["request_record"]
        started_at_perf = time.perf_counter()
        enqueue_perf = job.get("enqueue_perf")
        if isinstance(enqueue_perf, (int, float)):
            record["queue_wait_ms"] = round(
                max(0.0, started_at_perf - float(enqueue_perf)) * 1000.0,
                3,
        )
        record["start_ts"] = time.time()
        blk = job["block"]
        outcome_block = copy.copy(blk)
        crop = job.get("prepared_crop")
        image_bytes = job.get("prepared_image_bytes")
        try:
            self._raise_if_cancelled()
            if not isinstance(crop, np.ndarray) or not isinstance(
                image_bytes, (bytes, bytearray)
            ):
                raise LocalServiceResponseError(
                    "Prepared PaddleOCR-VL cache miss has no request crop.",
                    service_name="PaddleOCR VL",
                    settings_page_name="PaddleOCR VL Settings",
                )
            self._request_telemetry_context.record = record
            try:
                raw_text = self._request_ocr_text_from_encoded(bytes(image_bytes))
            finally:
                self._request_telemetry_context.record = None
            self._raise_if_cancelled()
            parse_started_at = time.perf_counter()
            cleaned = self._normalize_output_text(raw_text)
            non_text_reason = (
                self._classify_non_text_response(blk, cleaned, crop)
                if cleaned
                else ""
            )
            self._add_request_metric(
                record,
                "parse_sanitize_ms",
                (time.perf_counter() - parse_started_at) * 1000.0,
            )
            if cleaned and self._is_layout_schema_only_text(cleaned):
                self._mark_empty(
                    outcome_block,
                    self.LAYOUT_SCHEMA_EMPTY_REASON,
                    raw_text=raw_text,
                )
                record["status"] = "schema_only"
            elif cleaned and non_text_reason:
                self._mark_empty(
                    outcome_block,
                    self.NON_TEXT_RESPONSE_REASON,
                    raw_text=raw_text,
                )
                outcome_block.ocr_reject_reason = non_text_reason
                record["status"] = "rejected_non_text_response"
                record["non_text_reason"] = non_text_reason
            elif cleaned:
                set_block_ocr_diagnostics(
                    outcome_block,
                    text=cleaned,
                    confidence=0.0,
                    status=OCR_STATUS_OK,
                    empty_reason="",
                    attempt_count=1,
                    raw_text=raw_text,
                    sanitized_text=cleaned,
                )
                record["status"] = "ok"
            else:
                self._mark_empty(
                    outcome_block,
                    "PaddleOCR VL returned no usable text.",
                    raw_text=raw_text,
                )
                record["status"] = "empty"
            return snapshot_raw_ocr_result(outcome_block)
        except Exception as exc:
            record["status"] = f"error:{type(exc).__name__}"
            record["error"] = str(exc)
            raise
        finally:
            job.pop("prepared_crop", None)
            job.pop("prepared_image_bytes", None)
            record["end_ts"] = time.time()
            record["elapsed_ms"] = round(
                (time.perf_counter() - started_at_perf) * 1000.0,
                3,
            )

    def _process_job(self, img: np.ndarray, job: dict[str, Any]) -> None:
        record = job["request_record"]
        started_at_perf = time.perf_counter()
        enqueue_perf = job.get("enqueue_perf")
        if isinstance(enqueue_perf, (int, float)):
            record["queue_wait_ms"] = round(
                max(0.0, started_at_perf - float(enqueue_perf)) * 1000.0,
                3,
            )
        record["start_ts"] = time.time()
        bbox = job["bbox"]
        blk = job["block"]
        try:
            self._raise_if_cancelled()
            crop_started_at = time.perf_counter()
            crop = self._crop_image(img, bbox)
            record["crop_ms"] = round(
                (time.perf_counter() - crop_started_at) * 1000.0,
                3,
            )
            if crop is None:
                self._mark_empty(blk, "Invalid OCR crop bounds.")
                record["status"] = "crop_invalid"
                return

            guard_started_at = time.perf_counter()
            should_skip, evidence = self._should_skip_text_free_crop(blk, crop)
            record["text_guard_ms"] = round(
                (time.perf_counter() - guard_started_at) * 1000.0,
                3,
            )
            if should_skip:
                self._mark_empty(blk, self.TEXT_FREE_NO_VISUAL_EVIDENCE_REASON)
                reject_reason = str(evidence.get("reject_reason") or "no_visual_text_evidence")
                blk.ocr_reject_reason = reject_reason
                record["status"] = "rejected_no_text_evidence"
                record["text_evidence"] = evidence
                record["non_text_reason"] = reject_reason
                return

            self._request_telemetry_context.record = record
            try:
                raw_text = self._request_ocr_text(crop)
            finally:
                self._request_telemetry_context.record = None
            self._raise_if_cancelled()
            parse_started_at = time.perf_counter()
            cleaned = self._normalize_output_text(raw_text)
            non_text_reason = self._classify_non_text_response(blk, cleaned, crop) if cleaned else ""
            self._add_request_metric(
                record,
                "parse_sanitize_ms",
                (time.perf_counter() - parse_started_at) * 1000.0,
            )
            if cleaned and self._is_layout_schema_only_text(cleaned):
                self._mark_empty(blk, self.LAYOUT_SCHEMA_EMPTY_REASON, raw_text=raw_text)
                record["status"] = "schema_only"
            elif cleaned and non_text_reason:
                self._mark_empty(blk, self.NON_TEXT_RESPONSE_REASON, raw_text=raw_text)
                blk.ocr_reject_reason = non_text_reason
                record["status"] = "rejected_non_text_response"
                record["non_text_reason"] = non_text_reason
            elif cleaned:
                set_block_ocr_diagnostics(
                    blk,
                    text=cleaned,
                    confidence=0.0,
                    status=OCR_STATUS_OK,
                    empty_reason="",
                    attempt_count=1,
                    raw_text=raw_text,
                    sanitized_text=cleaned,
                )
                record["status"] = "ok"
            else:
                self._mark_empty(blk, "PaddleOCR VL returned no usable text.", raw_text=raw_text)
                record["status"] = "empty"
        except Exception as exc:
            record["status"] = f"error:{type(exc).__name__}"
            record["error"] = str(exc)
            raise
        finally:
            record["end_ts"] = time.time()
            record["elapsed_ms"] = round((time.perf_counter() - started_at_perf) * 1000.0, 3)

    def _resolve_scheduler_mode(self, settings) -> str:
        env_mode = str(os.environ.get("CT_PADDLEOCR_VL_SCHEDULER_MODE", "") or "").strip().lower()
        if env_mode:
            if env_mode in self.ALLOWED_SCHEDULER_MODES:
                return env_mode
            logger.warning("Ignoring invalid CT_PADDLEOCR_VL_SCHEDULER_MODE=%s", env_mode)

        generic = settings.get_ocr_generic_settings() if hasattr(settings, "get_ocr_generic_settings") else {}
        if isinstance(generic, dict):
            generic_mode = str(generic.get("paddleocr_vl_scheduler_mode", "") or "").strip().lower()
            if generic_mode in self.ALLOWED_SCHEDULER_MODES:
                return generic_mode
        return self.DEFAULT_SCHEDULER_MODE

    def _resolve_gpu_metrics_for_scheduler(self) -> dict[str, Any]:
        if self.scheduler_mode != "auto_v1":
            return {}
        if not self._is_local_server():
            return {}
        return query_gpu_metrics_cached(ttl_sec=1.0)

    def _is_local_server(self) -> bool:
        parsed = urlparse(str(self.server_url or ""))
        host = str(parsed.hostname or "").strip().lower()
        return host in {"127.0.0.1", "localhost", "::1"}

    def _order_jobs(self, jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if self.scheduler_mode == "fixed":
            return list(jobs)
        return sorted(
            jobs,
            key=lambda item: (-int(item["crop_area_px"]), int(item["job_index"])),
        )

    def _resolve_worker_count(
        self,
        jobs: list[dict[str, Any]],
        job_stats: dict[str, float],
        gpu_metrics: dict[str, Any],
    ) -> int:
        if not jobs:
            return 0
        cap = min(self.parallel_workers, len(jobs))
        if self.scheduler_mode != "auto_v1":
            return cap

        primary = gpu_metrics.get("primary") if isinstance(gpu_metrics, dict) else None
        if isinstance(primary, dict):
            memory_free_mb = self._safe_int(primary.get("memory_free_mb"), default=0)
            gpu_util_percent = self._safe_int(primary.get("gpu_util_percent"), default=0)
            if memory_free_mb < 2500:
                adjusted_workers = 1
            elif memory_free_mb < 4500:
                adjusted_workers = 2
            elif memory_free_mb < 6500:
                adjusted_workers = 3
            elif memory_free_mb < 9000:
                adjusted_workers = 4
            else:
                adjusted_workers = 5

            if job_stats["p90_area_ratio"] >= 0.03:
                adjusted_workers -= 2
            elif job_stats["p90_area_ratio"] >= 0.02:
                adjusted_workers -= 1
            if job_stats["large_crop_ratio"] >= 0.35:
                adjusted_workers -= 1
            if self.max_new_tokens >= 1024:
                adjusted_workers -= 1
            if gpu_util_percent >= 85:
                adjusted_workers -= 1
        else:
            if job_stats["p90_area_ratio"] >= 0.03:
                adjusted_workers = min(cap, 2)
            elif job_stats["p90_area_ratio"] >= 0.02:
                adjusted_workers = min(cap, 3)
            else:
                adjusted_workers = min(cap, 4)

        return max(1, min(cap, adjusted_workers))

    def _summarize_jobs(self, jobs: list[dict[str, Any]]) -> dict[str, float]:
        ratios = [float(job["crop_area_ratio"]) for job in jobs]
        if not ratios:
            return {
                "p50_area_ratio": 0.0,
                "p90_area_ratio": 0.0,
                "large_crop_ratio": 0.0,
            }
        large_count = sum(1 for ratio in ratios if ratio >= self.LARGE_CROP_RATIO_THRESHOLD)
        return {
            "p50_area_ratio": self._percentile(ratios, 0.50),
            "p90_area_ratio": self._percentile(ratios, 0.90),
            "large_crop_ratio": large_count / float(len(ratios)),
        }

    def _build_page_profile(
        self,
        jobs: list[dict[str, Any]],
        *,
        page_width: int,
        page_height: int,
        job_stats: dict[str, float],
        worker_count: int,
        gpu_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        request_records = [job["request_record"] for job in jobs]
        return {
            "scheduler_mode": self.scheduler_mode,
            "requested_cap": int(self.parallel_workers),
            "chosen_workers": int(worker_count),
            "block_count": len(jobs),
            "page_width": int(page_width),
            "page_height": int(page_height),
            "job_order": "original" if self.scheduler_mode == "fixed" else "area_desc",
            "max_new_tokens": int(self.max_new_tokens),
            "local_server": self._is_local_server(),
            "p50_area_ratio": round(job_stats.get("p50_area_ratio", 0.0), 6),
            "p90_area_ratio": round(job_stats.get("p90_area_ratio", 0.0), 6),
            "large_crop_ratio": round(job_stats.get("large_crop_ratio", 0.0), 6),
            "gpu_metrics": self._serialize_gpu_metrics(gpu_metrics),
            "request_records": request_records,
        }

    def _finalize_performance_profile(self) -> None:
        records = self.last_page_profile.get("request_records")
        request_records = records if isinstance(records, list) else []
        additive_fields = (
            "queue_wait_ms",
            "crop_ms",
            "text_guard_ms",
            "encode_ms",
            "base64_ms",
            "payload_build_ms",
            "logical_request_count",
            "http_attempt_count",
            "http_retry_count",
            "compatibility_retry_count",
            "retry_backoff_ms",
            "request_wall_ms",
            "response_decode_ms",
            "parse_sanitize_ms",
            "request_bytes",
            "base64_chars",
        )
        performance: dict[str, int | float] = {
            "schema_version": self.PERFORMANCE_TELEMETRY_SCHEMA_VERSION,
            "job_count": len(request_records),
        }
        integer_fields = {
            "logical_request_count",
            "http_attempt_count",
            "http_retry_count",
            "compatibility_retry_count",
            "request_bytes",
            "base64_chars",
        }
        for field_name in additive_fields:
            total = 0.0
            for record in request_records:
                if not isinstance(record, dict):
                    continue
                value = record.get(field_name, 0)
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                total += float(value)
            performance[field_name] = (
                int(round(total))
                if field_name in integer_fields
                else round(total, 3)
            )
        persistent_cache = self.last_page_profile.get("persistent_cache")
        if isinstance(persistent_cache, dict):
            performance["persistent_cache_hit_count"] = int(
                persistent_cache.get("hit_count", 0) or 0
            )
            performance["persistent_cache_miss_count"] = int(
                persistent_cache.get("miss_count", 0) or 0
            )
            performance["persistent_cache_runtime_miss_count"] = int(
                persistent_cache.get("runtime_miss_count", 0) or 0
            )
            performance["persistent_cache_disabled_count"] = int(
                bool(persistent_cache.get("lookup_disabled", False))
            )
        self.last_page_profile["performance"] = performance

    @staticmethod
    def _add_request_metric(
        record: dict[str, Any] | None,
        key: str,
        value: int | float,
    ) -> None:
        if not isinstance(record, dict):
            return
        current = record.get(key, 0)
        if isinstance(current, bool) or not isinstance(current, (int, float)):
            current = 0
        record[key] = current + value

    def _current_request_record(self) -> dict[str, Any] | None:
        record = getattr(self._request_telemetry_context, "record", None)
        return record if isinstance(record, dict) else None

    def _serialize_gpu_metrics(self, gpu_metrics: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(gpu_metrics, dict):
            return {}
        primary = gpu_metrics.get("primary")
        payload = {
            "available": bool(gpu_metrics.get("available", False)),
            "gpu_count": self._safe_int(gpu_metrics.get("gpu_count"), default=0),
            "sampled_at": gpu_metrics.get("sampled_at"),
        }
        if isinstance(primary, dict):
            payload["primary"] = {
                "index": self._safe_int(primary.get("index"), default=0),
                "name": str(primary.get("name", "") or ""),
                "memory_total_mb": self._safe_int(primary.get("memory_total_mb"), default=0),
                "memory_used_mb": self._safe_int(primary.get("memory_used_mb"), default=0),
                "memory_free_mb": self._safe_int(primary.get("memory_free_mb"), default=0),
                "gpu_util_percent": self._safe_int(primary.get("gpu_util_percent"), default=0),
                "memory_util_percent": self._safe_int(primary.get("memory_util_percent"), default=0),
            }
        return payload

    @staticmethod
    def _percentile(values: list[float], ratio: float) -> float:
        ordered = sorted(float(value) for value in values)
        if not ordered:
            return 0.0
        if len(ordered) == 1:
            return ordered[0]
        rank = max(0.0, min(1.0, float(ratio))) * (len(ordered) - 1)
        low = int(rank)
        high = min(len(ordered) - 1, low + 1)
        if low == high:
            return ordered[low]
        fraction = rank - low
        return ordered[low] + (ordered[high] - ordered[low]) * fraction

    @staticmethod
    def _safe_int(value, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _request_ocr_text(self, image: np.ndarray) -> str:
        record = self._current_request_record()
        started_at = time.perf_counter()
        image_bytes = self._encode_image(image)
        self._add_request_metric(
            record,
            "encode_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return self._request_ocr_text_from_encoded(image_bytes)

    def _request_ocr_text_from_encoded(self, image_bytes: bytes) -> str:
        record = self._current_request_record()
        self._add_request_metric(record, "logical_request_count", 1)
        self._add_request_metric(record, "request_bytes", len(image_bytes))
        started_at = time.perf_counter()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        self._add_request_metric(
            record,
            "base64_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        self._add_request_metric(record, "base64_chars", len(image_b64))
        started_at = time.perf_counter()
        payload = {
            "file": image_b64,
            "fileType": 1,
            "prettifyMarkdown": self.prettify_markdown,
            "visualize": self.visualize,
        }
        if self._supports_max_new_tokens:
            payload["maxNewTokens"] = self.max_new_tokens
        self._add_request_metric(
            record,
            "payload_build_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )

        data = self._send_request(payload, telemetry_record=record)
        if self._supports_max_new_tokens and self._response_rejected_max_new_tokens(data):
            self._raise_if_cancelled()
            self._supports_max_new_tokens = False
            payload.pop("maxNewTokens", None)
            self._add_request_metric(record, "compatibility_retry_count", 1)
            data = self._send_request(payload, telemetry_record=record)

        error_code = data.get("errorCode")
        if error_code not in (None, 0):
            error_msg = str(data.get("errorMsg", "") or "Unknown PaddleOCR VL error.")
            raise LocalServiceResponseError(
                error_msg,
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )

        started_at = time.perf_counter()
        text = self._extract_text_from_response(data)
        self._add_request_metric(
            record,
            "parse_sanitize_ms",
            (time.perf_counter() - started_at) * 1000.0,
        )
        return text

    def _send_request(
        self,
        payload: dict,
        *,
        telemetry_record: dict[str, Any] | None = None,
    ) -> dict:
        record = telemetry_record or self._current_request_record()
        last_response = None
        for attempt_index in range(self.REQUEST_RETRY_TOTAL_ATTEMPTS):
            request_started_at = time.perf_counter()
            self._add_request_metric(record, "http_attempt_count", 1)
            try:
                response = requests.post(
                    self.server_url,
                    json=payload,
                    timeout=self.REQUEST_TIMEOUT_SECONDS,
                )
            except requests.exceptions.RequestException as exc:
                self._add_request_metric(
                    record,
                    "request_wall_ms",
                    (time.perf_counter() - request_started_at) * 1000.0,
                )
                if self._should_retry_request(attempt_index):
                    self._add_request_metric(record, "http_retry_count", 1)
                    backoff_started_at = time.perf_counter()
                    self._sleep_before_retry(attempt_index, exc)
                    self._add_request_metric(
                        record,
                        "retry_backoff_ms",
                        (time.perf_counter() - backoff_started_at) * 1000.0,
                    )
                    continue
                raise LocalServiceConnectionError(
                    "Unable to reach the local PaddleOCR VL service.",
                    service_name="PaddleOCR VL",
                    settings_page_name="PaddleOCR VL Settings",
                ) from exc
            self._add_request_metric(
                record,
                "request_wall_ms",
                (time.perf_counter() - request_started_at) * 1000.0,
            )

            if response.status_code == 200:
                return self._decode_response_json(
                    response,
                    telemetry_record=record,
                )

            last_response = response
            if payload.get("maxNewTokens") is not None:
                self._raise_if_cancelled()
                legacy_payload = dict(payload)
                legacy_payload.pop("maxNewTokens", None)
                self._add_request_metric(record, "compatibility_retry_count", 1)
                request_started_at = time.perf_counter()
                self._add_request_metric(record, "http_attempt_count", 1)
                try:
                    legacy_response = requests.post(
                        self.server_url,
                        json=legacy_payload,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.RequestException as exc:
                    self._add_request_metric(
                        record,
                        "request_wall_ms",
                        (time.perf_counter() - request_started_at) * 1000.0,
                    )
                    if self._should_retry_request(attempt_index):
                        self._add_request_metric(record, "http_retry_count", 1)
                        backoff_started_at = time.perf_counter()
                        self._sleep_before_retry(attempt_index, exc)
                        self._add_request_metric(
                            record,
                            "retry_backoff_ms",
                            (time.perf_counter() - backoff_started_at) * 1000.0,
                        )
                        continue
                    raise LocalServiceConnectionError(
                        "Unable to reach the local PaddleOCR VL service.",
                        service_name="PaddleOCR VL",
                        settings_page_name="PaddleOCR VL Settings",
                    ) from exc
                self._add_request_metric(
                    record,
                    "request_wall_ms",
                    (time.perf_counter() - request_started_at) * 1000.0,
                )
                if legacy_response.status_code == 200:
                    self._supports_max_new_tokens = False
                    return self._decode_response_json(
                        legacy_response,
                        telemetry_record=record,
                    )
                if response.status_code not in self.TRANSIENT_HTTP_STATUS_CODES:
                    last_response = legacy_response

            if (
                last_response is not None
                and last_response.status_code in self.TRANSIENT_HTTP_STATUS_CODES
                and self._should_retry_request(attempt_index)
            ):
                self._add_request_metric(record, "http_retry_count", 1)
                backoff_started_at = time.perf_counter()
                self._sleep_before_retry(attempt_index, last_response)
                self._add_request_metric(
                    record,
                    "retry_backoff_ms",
                    (time.perf_counter() - backoff_started_at) * 1000.0,
                )
                continue
            break

        response = last_response
        status_code = response.status_code if response is not None else 500
        raise LocalServiceResponseError(
            f"PaddleOCR VL service returned HTTP {status_code}.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )

    def _should_retry_request(self, attempt_index: int) -> bool:
        return attempt_index < max(1, int(self.REQUEST_RETRY_TOTAL_ATTEMPTS)) - 1

    def _sleep_before_retry(self, attempt_index: int, reason: object) -> None:
        delay = self.REQUEST_RETRY_BACKOFF_SECONDS[
            min(attempt_index, len(self.REQUEST_RETRY_BACKOFF_SECONDS) - 1)
        ]
        logger.warning(
            "paddleocr_vl transient request failure; retrying in %.1fs: %s",
            delay,
            reason,
        )
        deadline = time.perf_counter() + max(0.0, float(delay))
        while True:
            self._raise_if_cancelled()
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            time.sleep(min(0.2, remaining))

    def _decode_response_json(
        self,
        response,
        *,
        telemetry_record: dict[str, Any] | None = None,
    ) -> dict:
        started_at = time.perf_counter()
        try:
            return response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                "PaddleOCR VL service returned invalid JSON.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            ) from exc
        finally:
            self._add_request_metric(
                telemetry_record or self._current_request_record(),
                "response_decode_ms",
                (time.perf_counter() - started_at) * 1000.0,
            )

    def _extract_text_from_response(self, data: dict) -> str:
        result = data.get("result", data)
        if not isinstance(result, dict):
            return ""

        layout_results = result.get("layoutParsingResults")
        if isinstance(layout_results, list):
            for item in layout_results:
                text = self._extract_text_from_layout_item(item)
                if text:
                    return text

        return self._extract_text_from_layout_item(result)

    def _extract_text_from_layout_item(self, item: dict) -> str:
        if not isinstance(item, dict):
            return ""

        markdown = item.get("markdown")
        if isinstance(markdown, dict):
            markdown_text = self._markdown_to_text(markdown.get("text", ""))
            if markdown_text:
                return markdown_text

        pruned = item.get("prunedResult")
        if pruned is not None:
            extracted = self._extract_texts_from_pruned(pruned)
            if extracted:
                return self._normalize_output_text("\n".join(extracted))

        raw_text = item.get("text")
        if isinstance(raw_text, str):
            return self._normalize_output_text(raw_text)

        return ""

    def _extract_texts_from_pruned(self, node) -> list[str]:
        texts: list[str] = []

        def walk(value) -> None:
            if value is None:
                return
            if isinstance(value, dict):
                dict_text = value.get("text")
                dict_texts = value.get("texts")
                if isinstance(dict_text, str):
                    cleaned = dict_text.strip()
                    if cleaned:
                        texts.append(cleaned)
                if isinstance(dict_texts, list):
                    combined = "".join(str(part) for part in dict_texts).strip()
                    if combined:
                        texts.append(combined)
                elif isinstance(dict_texts, str):
                    cleaned = dict_texts.strip()
                    if cleaned:
                        texts.append(cleaned)
                for child in value.values():
                    walk(child)
                return
            if isinstance(value, list):
                for child in value:
                    walk(child)
                return
            if isinstance(value, str):
                cleaned = value.strip()
                if cleaned:
                    texts.append(cleaned)

        walk(node)
        return texts

    def _resolve_bbox(self, blk: TextBlock, image: np.ndarray) -> tuple[int, int, int, int] | None:
        initialize_ocr_result_contract(
            blk,
            strategy=OCR_STRATEGY_PADDLE_CROP,
            model_identity=str(
                getattr(blk, "ocr_model_identity", "")
                or self.MODEL_IDENTITY
            ),
        )
        bbox, crop_source = resolve_block_crop_bbox(
            blk,
            image.shape,
            x_ratio=self.TEXT_EXPANSION_RATIO,
            y_ratio=self.TEXT_EXPANSION_RATIO,
            bubble_as_clamp=True,
            fallback_to_bubble=True,
        )
        set_block_ocr_crop_diagnostics(
            blk,
            effective_crop_xyxy=bbox,
            crop_source=crop_source,
        )
        blk.ocr_geometry_provenance = {
            "strategy": "text_first_bubble_clamp",
            "crop_source": crop_source,
            "text_bbox": self._coords_or_none(getattr(blk, "xyxy", None)),
            "bubble_bbox": self._coords_or_none(
                getattr(blk, "bubble_xyxy", None)
            ),
            "effective_crop_bbox": (
                [int(value) for value in bbox] if bbox is not None else None
            ),
            "text_expansion_ratio": float(self.TEXT_EXPANSION_RATIO),
        }
        if bbox is None:
            return None
        return bbox

    @staticmethod
    def _coords_or_none(value) -> list[float] | None:
        if value is None:
            return None
        try:
            coords = [float(item) for item in value]
        except (TypeError, ValueError):
            return None
        return coords if len(coords) == 4 else None

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

    @staticmethod
    def _is_text_free_without_bubble(blk: TextBlock) -> bool:
        return (
            str(getattr(blk, "text_class", "") or "") == "text_free"
            and getattr(blk, "bubble_xyxy", None) is None
        )

    def _should_skip_text_free_crop(self, blk: TextBlock, crop: np.ndarray) -> tuple[bool, dict[str, Any]]:
        if not self._is_text_free_without_bubble(blk):
            return False, {}
        evidence = self._analyze_text_free_crop(crop)
        if self._looks_like_saturated_texture_without_text_mask(evidence):
            evidence = dict(evidence)
            evidence["reject_reason"] = "saturated_texture_without_text_mask"
            return True, evidence
        if not self._has_minimum_text_free_evidence(evidence):
            evidence = dict(evidence)
            evidence["reject_reason"] = "no_visual_text_evidence"
            return True, evidence
        return False, evidence

    def _analyze_text_free_crop(self, crop: np.ndarray) -> dict[str, Any]:
        try:
            image = ensure_three_channel(crop)
            gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
            hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2HSV)
        except Exception:
            return {
                "width": 0,
                "height": 0,
                "contrast_std": 0.0,
                "contrast_spread": 0.0,
                "edge_density": 0.0,
                "transition_density": 0.0,
                "component_count": 0,
                "component_area_ratio": 0.0,
                "saturated_ratio": 0.0,
                "warm_saturated_ratio": 0.0,
                "low_saturation_ratio": 0.0,
            }

        h, w = gray.shape[:2]
        area = max(1, int(h) * int(w))
        if h <= 0 or w <= 0:
            return {
                "width": int(w),
                "height": int(h),
                "contrast_std": 0.0,
                "contrast_spread": 0.0,
                "edge_density": 0.0,
                "transition_density": 0.0,
                "component_count": 0,
                "component_area_ratio": 0.0,
                "saturated_ratio": 0.0,
                "warm_saturated_ratio": 0.0,
                "low_saturation_ratio": 0.0,
            }

        p5, p95 = np.percentile(gray, [5, 95])
        contrast_std = float(np.std(gray))
        contrast_spread = float(p95 - p5)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        saturated = saturation >= 70
        warm = ((hue <= 34) | (hue >= 168)) & saturated & (value >= 80)
        low_saturation = saturation <= 35
        saturated_ratio = float(np.count_nonzero(saturated)) / float(area)
        warm_saturated_ratio = float(np.count_nonzero(warm)) / float(area)
        low_saturation_ratio = float(np.count_nonzero(low_saturation)) / float(area)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(area)
        gray_i = gray.astype(np.int16)
        horizontal_pairs = max(0, h * max(0, w - 1))
        vertical_pairs = max(0, max(0, h - 1) * w)
        transition_count = 0
        if w > 1:
            transition_count += int(np.count_nonzero(np.abs(gray_i[:, 1:] - gray_i[:, :-1]) >= 28))
        if h > 1:
            transition_count += int(np.count_nonzero(np.abs(gray_i[1:, :] - gray_i[:-1, :]) >= 28))
        transition_density = float(transition_count) / float(max(1, horizontal_pairs + vertical_pairs))

        median = float(np.median(gray))
        dark_threshold = min(150.0, median - max(18.0, contrast_std * 0.45))
        bright_threshold = max(105.0, median + max(18.0, contrast_std * 0.45))
        dark = (gray <= dark_threshold) | (gray <= 70)
        bright = (gray >= bright_threshold) & (gray >= 185)
        candidate = np.where(dark | bright, 255, 0).astype(np.uint8)
        if candidate.size:
            candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)

        component_count = 0
        component_area = 0
        max_component_area = 0
        try:
            labels_count, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                (candidate > 0).astype(np.uint8),
                8,
                cv2.CV_32S,
            )
        except Exception:
            labels_count = 0
            stats = np.zeros((0, 5), dtype=np.int32)

        for label_idx in range(1, labels_count):
            x, y, comp_w, comp_h, comp_area = [int(v) for v in stats[label_idx]]
            del x, y
            if comp_area < 3 or comp_w <= 0 or comp_h <= 0:
                continue
            bbox_area = max(1, comp_w * comp_h)
            fill_ratio = float(comp_area) / float(bbox_area)
            aspect = float(max(comp_w, comp_h)) / float(max(1, min(comp_w, comp_h)))
            if bbox_area > int(round(area * 0.35)) or comp_area > int(round(area * 0.22)):
                continue
            if max(comp_w, comp_h) >= max(16, int(round(min(h, w) * 0.30))) and min(comp_w, comp_h) <= 4 and aspect >= 8.0:
                continue
            if fill_ratio > 0.92 and bbox_area > 24:
                continue
            if comp_w < 2 and comp_h < 2:
                continue
            component_count += 1
            component_area += comp_area
            max_component_area = max(max_component_area, comp_area)

        return {
            "width": int(w),
            "height": int(h),
            "contrast_std": round(contrast_std, 4),
            "contrast_spread": round(contrast_spread, 4),
            "edge_density": round(edge_density, 6),
            "transition_density": round(transition_density, 6),
            "component_count": int(component_count),
            "component_area_ratio": round(float(component_area) / float(area), 6),
            "max_component_area_ratio": round(float(max_component_area) / float(area), 6),
            "saturated_ratio": round(saturated_ratio, 6),
            "warm_saturated_ratio": round(warm_saturated_ratio, 6),
            "low_saturation_ratio": round(low_saturation_ratio, 6),
        }

    @staticmethod
    def _has_minimum_text_free_evidence(evidence: dict[str, Any]) -> bool:
        edge_density = float(evidence.get("edge_density", 0.0) or 0.0)
        transition_density = float(evidence.get("transition_density", 0.0) or 0.0)
        component_count = int(evidence.get("component_count", 0) or 0)
        component_area_ratio = float(evidence.get("component_area_ratio", 0.0) or 0.0)
        contrast_std = float(evidence.get("contrast_std", 0.0) or 0.0)
        contrast_spread = float(evidence.get("contrast_spread", 0.0) or 0.0)

        if contrast_spread < 22.0 and contrast_std < 10.0:
            return False
        if transition_density < 0.003 and edge_density < 0.035:
            return False
        if transition_density < 0.001 and component_area_ratio < 0.001:
            return False
        if edge_density < 0.010 and transition_density < 0.004:
            return False
        if component_count <= 0 and edge_density < 0.018 and transition_density < 0.008:
            return False
        if component_count <= 1 and component_area_ratio < 0.002 and transition_density < 0.010:
            return False
        return True

    def _is_non_text_response(self, blk: TextBlock, text: str, crop: np.ndarray) -> bool:
        return bool(self._classify_non_text_response(blk, text, crop))

    def _classify_non_text_response(self, blk: TextBlock, text: str, crop: np.ndarray) -> str:
        if self._looks_like_non_text_response(text):
            return "model_non_text_response"
        if not self._is_text_free_without_bubble(blk):
            return ""
        non_target_reason = self._classify_non_target_text_free_response(blk, text, crop)
        if non_target_reason:
            return non_target_reason
        if self._looks_like_low_signal_text_free_hallucination(text, crop):
            return "low_signal_text_free_hallucination"
        return ""

    @classmethod
    def _looks_like_non_text_response(cls, text: str) -> bool:
        cleaned = str(text or "").strip()
        if not cleaned:
            return False
        compact = re.sub(r"\s+", " ", cleaned)
        return any(pattern.search(compact) for pattern in cls.NON_TEXT_RESPONSE_PATTERNS)

    def _looks_like_low_signal_text_free_hallucination(self, text: str, crop: np.ndarray) -> bool:
        compact = re.sub(r"\s+", " ", str(text or "").strip())
        if not compact or len(compact) > 48:
            return False
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]", compact):
            return False
        if any(char.isdigit() for char in compact):
            return False
        letters = re.findall(r"[A-Za-z]+", compact)
        if not (2 <= len(letters) <= 5):
            return False
        lowercase_count = sum(1 for token in letters if token.islower())
        if lowercase_count / float(len(letters)) < 0.80:
            return False
        stopwords = {"a", "an", "and", "are", "as", "at", "for", "from", "in", "is", "it", "of", "on", "the", "to", "with"}
        if not any(token.lower() in stopwords for token in letters):
            return False

        evidence = self._analyze_text_free_crop(crop)
        if self._has_strong_text_free_typography_evidence(evidence):
            return False
        return True

    def _looks_like_non_target_text_free_response(self, blk: TextBlock, text: str, crop: np.ndarray) -> bool:
        return bool(self._classify_non_target_text_free_response(blk, text, crop))

    def _classify_non_target_text_free_response(self, blk: TextBlock, text: str, crop: np.ndarray) -> str:
        compact = re.sub(r"\s+", " ", str(text or "").strip())
        if not compact:
            return ""
        if self._looks_like_text_free_watermark_or_credit(compact):
            return "watermark_or_credit"
        if self._looks_like_symbol_only_ocr_text(compact):
            return "symbol_only"
        if self._looks_like_book_spine_or_index_hallucination(blk, compact):
            return "book_spine_or_index"

        evidence = self._analyze_text_free_crop(crop)
        if self._looks_like_numeric_warm_texture_hallucination(compact, evidence):
            return "numeric_warm_texture"
        if self._looks_like_short_saturated_texture_hallucination(compact, evidence):
            return "short_saturated_texture"
        if self._looks_like_source_script_mismatch(blk, compact, evidence):
            return "source_script_mismatch"
        return ""

    @classmethod
    def _looks_like_text_free_watermark_or_credit(cls, text: str) -> bool:
        return any(pattern.search(text) for pattern in cls.TEXT_FREE_WATERMARK_PATTERNS)

    @staticmethod
    def _looks_like_symbol_only_ocr_text(text: str) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        if not compact:
            return False
        semantic_count = sum(1 for ch in compact if unicodedata.category(ch)[0] in {"L", "N"})
        if semantic_count:
            return False
        return len(compact) <= 16

    @classmethod
    def _looks_like_book_spine_or_index_hallucination(cls, blk: TextBlock, text: str) -> bool:
        compact = str(text or "")
        catalog_hits = len(cls.TEXT_FREE_CATALOG_MARKER_PATTERN.findall(compact))
        if catalog_hits >= 3:
            return True
        non_empty_lines = [line.strip() for line in re.split(r"[\r\n/]+", compact) if line.strip()]
        cjk_count = len(re.findall(r"[\u3400-\u9fff]", compact))
        catalog_like_list = len(non_empty_lines) >= 5 and cjk_count >= 12

        try:
            x1, y1, x2, y2 = [float(v) for v in getattr(blk, "xyxy", (0, 0, 0, 0))]
        except Exception:
            return False
        width = max(1.0, x2 - x1)
        height = max(1.0, y2 - y1)
        edge_touching = x1 <= 6.0 or y1 <= 6.0
        long_edge_strip = edge_touching and height >= 500.0 and height >= width * 1.8
        if long_edge_strip and (catalog_hits >= 2 or catalog_like_list):
            return True
        if len(compact) >= 80 and catalog_like_list and catalog_hits >= 1:
            return True
        return False

    @staticmethod
    def _looks_like_numeric_warm_texture_hallucination(text: str, evidence: dict[str, Any]) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        if not re.fullmatch(r"[\d:.,/\-~年月日년월일]+", compact):
            return False
        if not re.search(r"\d{3,}", compact):
            return False
        warm_ratio = float(evidence.get("warm_saturated_ratio", 0.0) or 0.0)
        saturated_ratio = float(evidence.get("saturated_ratio", 0.0) or 0.0)
        low_saturation_ratio = float(evidence.get("low_saturation_ratio", 0.0) or 0.0)
        return warm_ratio >= 0.035 and saturated_ratio >= 0.08 and low_saturation_ratio < 0.88

    @staticmethod
    def _looks_like_saturated_texture_without_text_mask(evidence: dict[str, Any]) -> bool:
        saturated_ratio = float(evidence.get("saturated_ratio", 0.0) or 0.0)
        low_saturation_ratio = float(evidence.get("low_saturation_ratio", 0.0) or 0.0)
        edge_density = float(evidence.get("edge_density", 0.0) or 0.0)
        transition_density = float(evidence.get("transition_density", 0.0) or 0.0)
        component_count = int(evidence.get("component_count", 0) or 0)
        component_area_ratio = float(evidence.get("component_area_ratio", 0.0) or 0.0)
        contrast_spread = float(evidence.get("contrast_spread", 0.0) or 0.0)

        if saturated_ratio < 0.35 or low_saturation_ratio > 0.35 or contrast_spread < 70.0:
            return False
        fragmented_mask = component_count >= 10 and edge_density >= 0.055 and component_area_ratio <= 0.13
        dense_noise_mask = component_count >= 10 and transition_density >= 0.050 and component_area_ratio <= 0.13
        low_text_area_mask = component_count >= 20 and component_area_ratio <= 0.030 and edge_density >= 0.055
        return fragmented_mask or dense_noise_mask or low_text_area_mask

    @staticmethod
    def _looks_like_short_saturated_texture_hallucination(text: str, evidence: dict[str, Any]) -> bool:
        compact = re.sub(r"\s+", "", str(text or ""))
        if not compact or len(compact) > 4:
            return False
        if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7a3]", compact):
            return False
        numeric_token = bool(re.fullmatch(r"(?:\d{2,4}|[A-Za-z]?\d{2,4}|\d{1,4}[A-Za-z]?)", compact))
        single_letter_token = bool(re.fullmatch(r"[A-Za-z]", compact))
        if not numeric_token and not single_letter_token:
            return False

        saturated_ratio = float(evidence.get("saturated_ratio", 0.0) or 0.0)
        low_saturation_ratio = float(evidence.get("low_saturation_ratio", 0.0) or 0.0)
        edge_density = float(evidence.get("edge_density", 0.0) or 0.0)
        transition_density = float(evidence.get("transition_density", 0.0) or 0.0)
        component_count = int(evidence.get("component_count", 0) or 0)
        component_area_ratio = float(evidence.get("component_area_ratio", 0.0) or 0.0)
        contrast_spread = float(evidence.get("contrast_spread", 0.0) or 0.0)

        saturated_texture = saturated_ratio >= 0.35 and low_saturation_ratio <= 0.35
        noisy_edges = edge_density >= 0.055 and component_count >= 8 and component_area_ratio >= 0.020
        noisy_transitions = transition_density >= 0.050 and component_count >= 8
        high_numeric_texture = (
            numeric_token
            and edge_density >= 0.055
            and contrast_spread >= 85.0
            and (transition_density >= 0.004 or component_count >= 8)
        )
        high_contrast_icon = single_letter_token and contrast_spread >= 120.0 and edge_density >= 0.060
        return saturated_texture and (noisy_edges or noisy_transitions or high_numeric_texture or high_contrast_icon)

    def _looks_like_source_script_mismatch(self, blk: TextBlock, text: str, evidence: dict[str, Any]) -> bool:
        counts = self._script_counts(text)
        unsupported = self._unsupported_script_count_for_source(blk, counts)
        if unsupported <= 0:
            return False
        semantic = sum(counts.values())
        if semantic <= 0:
            return False
        unsupported_ratio = unsupported / float(semantic)
        if unsupported < 3:
            return False
        if unsupported_ratio < 0.55:
            return False

        if self._has_strong_text_free_typography_evidence(evidence):
            return False
        return True

    @staticmethod
    def _script_counts(text: str) -> dict[str, int]:
        counts = {
            "latin": 0,
            "kana": 0,
            "cjk": 0,
            "hangul": 0,
            "tibetan": 0,
            "digit": 0,
            "other_letter": 0,
        }
        for ch in str(text or ""):
            codepoint = ord(ch)
            if "0" <= ch <= "9":
                counts["digit"] += 1
            elif "A" <= ch <= "Z" or "a" <= ch <= "z":
                counts["latin"] += 1
            elif 0x3040 <= codepoint <= 0x30FF:
                counts["kana"] += 1
            elif 0x3400 <= codepoint <= 0x9FFF:
                counts["cjk"] += 1
            elif 0xAC00 <= codepoint <= 0xD7A3:
                counts["hangul"] += 1
            elif 0x0F00 <= codepoint <= 0x0FFF:
                counts["tibetan"] += 1
            elif unicodedata.category(ch).startswith("L"):
                counts["other_letter"] += 1
        return counts

    @staticmethod
    def _unsupported_script_count_for_source(blk: TextBlock, counts: dict[str, int]) -> int:
        source_lang = str(getattr(blk, "source_lang", "") or "").strip().lower()
        if source_lang in {"ja", "jpn", "japanese"}:
            supported = {"latin", "kana", "cjk", "digit"}
        elif source_lang in {"zh", "zh-cn", "zh-tw", "chi", "chinese"}:
            supported = {"latin", "cjk", "digit"}
        elif source_lang in {"ko", "kor", "korean"}:
            supported = {"latin", "hangul", "digit"}
        else:
            supported = {"latin", "digit"}
        return sum(value for key, value in counts.items() if key not in supported)

    @staticmethod
    def _has_strong_text_free_typography_evidence(evidence: dict[str, Any]) -> bool:
        edge_density = float(evidence.get("edge_density", 0.0) or 0.0)
        transition_density = float(evidence.get("transition_density", 0.0) or 0.0)
        component_count = int(evidence.get("component_count", 0) or 0)
        component_area_ratio = float(evidence.get("component_area_ratio", 0.0) or 0.0)
        warm_saturated_ratio = float(evidence.get("warm_saturated_ratio", 0.0) or 0.0)
        saturated_ratio = float(evidence.get("saturated_ratio", 0.0) or 0.0)
        low_saturation_ratio = float(evidence.get("low_saturation_ratio", 0.0) or 0.0)
        if warm_saturated_ratio >= 0.10 and saturated_ratio >= 0.18 and low_saturation_ratio < 0.75:
            return False
        if component_count >= 8 and component_area_ratio >= 0.012 and transition_density >= 0.012:
            return True
        if edge_density >= 0.045 and transition_density >= 0.035 and component_area_ratio >= 0.020:
            return True
        return False

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise LocalServiceResponseError(
                "Failed to encode OCR crop for PaddleOCR VL.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )
            
        return encoded.tobytes()

    def _mark_empty(self, blk: TextBlock, reason: str, raw_text: str = "") -> None:
        set_block_ocr_diagnostics(
            blk,
            text="",
            confidence=0.0,
            status=OCR_STATUS_EMPTY_INITIAL,
            empty_reason=reason,
            attempt_count=1,
            raw_text=raw_text,
            sanitized_text="",
        )

    def _response_rejected_max_new_tokens(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        error_msg = str(data.get("errorMsg", "") or "")
        return "maxNewTokens" in error_msg

    def _normalize_output_text(self, text: str) -> str:
        if not text:
            return ""
        normalized = normalize_decorative_ocr_text(
            text,
            glyphs=DECORATIVE_NOISE_GLYPHS,
        )
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @classmethod
    def _is_layout_schema_only_text(cls, text: str) -> bool:
        if not text:
            return False

        tokens = []
        for raw_token in re.split(r"\s+", text.strip().lower()):
            token = raw_token.strip("`*_#-:;,.[](){}<>")
            if token:
                tokens.append(token)

        if len(tokens) < 3:
            return False
        if any(re.search(r"[^a-z_]", token) for token in tokens):
            return False

        token_set = set(tokens)
        if not token_set.issubset(cls.LAYOUT_SCHEMA_TOKENS):
            return False
        return bool(token_set & cls.STRONG_LAYOUT_SCHEMA_TOKENS)

    def _markdown_to_text(self, text: str) -> str:
        if not text:
            return ""

        cleaned = re.sub(r"!\[[^\]]*\]\([^\)]*\)", "", text)
        cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
        cleaned = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"(\*\*|__)(.*?)\1", r"\2", cleaned)
        cleaned = re.sub(r"(\*|_)(.*?)\1", r"\2", cleaned)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"<[^>]+>", "", cleaned)
        return self._normalize_output_text(cleaned)

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))
