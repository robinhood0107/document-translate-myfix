from __future__ import annotations

import base64
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
import requests

from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.gpu_metrics import query_gpu_metrics_cached
from modules.utils.ocr_debug import OCR_STATUS_EMPTY_INITIAL, OCR_STATUS_OK, ensure_three_channel, expand_bbox, set_block_ocr_diagnostics
from modules.utils.text_normalization import (
    DECORATIVE_NOISE_GLYPHS,
    normalize_decorative_ocr_text,
)
from modules.utils.textblock import TextBlock

from .base import OCREngine


logger = logging.getLogger(__name__)


class PaddleOCRVLEngine(OCREngine):
    DEFAULT_SERVER_URL = "http://127.0.0.1:28118/layout-parsing"
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_PARALLEL_WORKERS = 8
    DEFAULT_SCHEDULER_MODE = "fixed"
    MAX_NEW_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_TIMEOUT_SECONDS = 60
    TEXT_EXPANSION_RATIO = 0.05
    LARGE_CROP_RATIO_THRESHOLD = 0.02
    MEDIUM_CROP_RATIO_THRESHOLD = 0.008
    ALLOWED_SCHEDULER_MODES = frozenset({"fixed", "fixed_area_desc", "auto_v1"})

    def __init__(self) -> None:
        self.server_url = self.DEFAULT_SERVER_URL
        self.prettify_markdown = False
        self.visualize = False
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS
        self.parallel_workers = self.DEFAULT_PARALLEL_WORKERS
        self.scheduler_mode = self.DEFAULT_SCHEDULER_MODE
        self.last_page_profile: dict[str, Any] = {}
        self._supports_max_new_tokens = True

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

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        jobs: list[dict[str, Any]] = []
        page_height = int(img.shape[0]) if img is not None and len(img.shape) >= 2 else 0
        page_width = int(img.shape[1]) if img is not None and len(img.shape) >= 2 else 0
        page_area = max(page_height * page_width, 1)
        for job_index, blk in enumerate(blk_list or []):
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
                "crop_area_px": int(crop_area_px),
                "crop_area_ratio": round(float(crop_area_ratio), 6),
                "enqueue_ts": None,
                "start_ts": None,
                "end_ts": None,
                "elapsed_ms": None,
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
            if worker_count <= 1:
                for job in ordered_jobs:
                    job["request_record"]["enqueue_ts"] = time.time()
                    self._process_job(img, job)
            else:
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    future_map = {}
                    for job in ordered_jobs:
                        job["request_record"]["enqueue_ts"] = time.time()
                        future = executor.submit(self._process_job, img, job)
                        future_map[future] = job
                    try:
                        for future in as_completed(future_map):
                            future.result()
                    except Exception:
                        for future in future_map:
                            future.cancel()
                        raise
            self.last_page_profile["page_status"] = "ok"
        except Exception as exc:
            self.last_page_profile["page_status"] = "error"
            self.last_page_profile["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            elapsed_ms = (time.perf_counter() - started_at) * 1000.0
            self.last_page_profile["elapsed_ms"] = round(elapsed_ms, 3)
            self.last_page_profile["completed_at"] = time.time()

        logger.info(
            "paddleocr_vl complete: blocks=%d elapsed_ms=%.1f scheduler=%s",
            len(ordered_jobs),
            self.last_page_profile.get("elapsed_ms", 0.0),
            self.scheduler_mode,
        )
        return blk_list

    def _process_job(self, img: np.ndarray, job: dict[str, Any]) -> None:
        record = job["request_record"]
        started_at_perf = time.perf_counter()
        record["start_ts"] = time.time()
        bbox = job["bbox"]
        blk = job["block"]
        try:
            crop = self._crop_image(img, bbox)
            if crop is None:
                self._mark_empty(blk, "Invalid OCR crop bounds.")
                record["status"] = "crop_invalid"
                return

            raw_text = self._request_ocr_text(crop)
            cleaned = self._normalize_output_text(raw_text)
            if cleaned:
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
        image_bytes = self._encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "file": image_b64,
            "fileType": 1,
            "prettifyMarkdown": self.prettify_markdown,
            "visualize": self.visualize,
        }
        if self._supports_max_new_tokens:
            payload["maxNewTokens"] = self.max_new_tokens

        data = self._send_request(payload)
        if self._supports_max_new_tokens and self._response_rejected_max_new_tokens(data):
            self._supports_max_new_tokens = False
            payload.pop("maxNewTokens", None)
            data = self._send_request(payload)

        error_code = data.get("errorCode")
        if error_code not in (None, 0):
            error_msg = str(data.get("errorMsg", "") or "Unknown PaddleOCR VL error.")
            raise LocalServiceResponseError(
                error_msg,
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )

        return self._extract_text_from_response(data)

    def _send_request(self, payload: dict) -> dict:
        try:
            response = requests.post(
                self.server_url,
                json=payload,
                timeout=self.REQUEST_TIMEOUT_SECONDS,
            )
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                "Unable to reach the local PaddleOCR VL service.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            ) from exc

        if response.status_code != 200:
            if payload.get("maxNewTokens") is not None:
                legacy_payload = dict(payload)
                legacy_payload.pop("maxNewTokens", None)
                try:
                    legacy_response = requests.post(
                        self.server_url,
                        json=legacy_payload,
                        timeout=self.REQUEST_TIMEOUT_SECONDS,
                    )
                except requests.exceptions.RequestException as exc:
                    raise LocalServiceConnectionError(
                        "Unable to reach the local PaddleOCR VL service.",
                        service_name="PaddleOCR VL",
                        settings_page_name="PaddleOCR VL Settings",
                    ) from exc
                if legacy_response.status_code == 200:
                    response = legacy_response
                    self._supports_max_new_tokens = False

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"PaddleOCR VL service returned HTTP {response.status_code}.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            )

        try:
            return response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                "PaddleOCR VL service returned invalid JSON.",
                service_name="PaddleOCR VL",
                settings_page_name="PaddleOCR VL Settings",
            ) from exc

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
        source_bbox = getattr(blk, "bubble_xyxy", None)
        if source_bbox is not None:
            bbox = expand_bbox(source_bbox, image.shape)
        else:
            text_bbox = getattr(blk, "xyxy", None)
            if text_bbox is None:
                return None
            bbox = expand_bbox(
                text_bbox,
                image.shape,
                x_ratio=self.TEXT_EXPANSION_RATIO,
                y_ratio=self.TEXT_EXPANSION_RATIO,
            )

        x1, y1, x2, y2 = bbox
        if x2 <= x1 or y2 <= y1:
            return None
        return x1, y1, x2, y2

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

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
