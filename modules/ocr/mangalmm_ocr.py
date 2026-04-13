from __future__ import annotations

import base64
import json
import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.language_utils import is_no_space_lang
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    ensure_three_channel,
    resolve_block_crop_bbox,
    set_block_ocr_crop_diagnostics,
    set_block_ocr_diagnostics,
)
from modules.utils.textblock import TextBlock, sort_textblock_rectangles

from .base import OCREngine


logger = logging.getLogger(__name__)

SERVICE_NAME = "MangaLMM"
SETTINGS_PAGE_NAME = "MangaLMM Settings"
DEFAULT_MANGALMM_SERVER_URL = "http://127.0.0.1:28081/v1"
DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS = 256
DEFAULT_MANGALMM_PARALLEL_WORKERS = 1
DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC = 60
DEFAULT_MANGALMM_SAFE_RESIZE = True
DEFAULT_MANGALMM_MAX_PIXELS = 1_200_000
DEFAULT_MANGALMM_MAX_LONG_SIDE = 1280
DEFAULT_MANGALMM_TEMPERATURE = 0.0
DEFAULT_MANGALMM_TOP_K = 1


class MangaLMMOCREngine(OCREngine):
    MAX_COMPLETION_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_ENDPOINT_SUFFIX = "/chat/completions"
    PROMPT = (
        'Please perform OCR on this image and output only a JSON array of recognized text regions, '
        'where each item has "bbox_2d" and "text_content". Do not translate. Do not explain. '
        "Do not add markdown or code fences."
    )
    TEXT_EXPANSION_RATIO = 0.06

    def __init__(self) -> None:
        self.server_url = DEFAULT_MANGALMM_SERVER_URL
        self.max_completion_tokens = DEFAULT_MANGALMM_MAX_COMPLETION_TOKENS
        self.parallel_workers = DEFAULT_MANGALMM_PARALLEL_WORKERS
        self.request_timeout_sec = DEFAULT_MANGALMM_REQUEST_TIMEOUT_SEC
        self.raw_response_logging = False
        self.safe_resize = DEFAULT_MANGALMM_SAFE_RESIZE
        self.max_pixels = DEFAULT_MANGALMM_MAX_PIXELS
        self.max_long_side = DEFAULT_MANGALMM_MAX_LONG_SIDE

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_mangalmm_ocr_settings()
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
                (100_000, 4_000_000),
            ),
        )
        self.max_long_side = max(
            256,
            self._clamp_int(
                config.get("max_long_side", DEFAULT_MANGALMM_MAX_LONG_SIDE),
                DEFAULT_MANGALMM_MAX_LONG_SIDE,
                (256, 4096),
            ),
        )

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        jobs: list[tuple[TextBlock, tuple[int, int, int, int], str]] = []
        for blk in blk_list or []:
            bbox, crop_source = resolve_block_crop_bbox(
                blk,
                img.shape,
                x_ratio=self.TEXT_EXPANSION_RATIO,
                y_ratio=self.TEXT_EXPANSION_RATIO,
                bubble_as_clamp=True,
                fallback_to_bubble=True,
            )
            if bbox is None:
                self._mark_empty(blk, "Invalid OCR crop bounds.")
                continue
            set_block_ocr_crop_diagnostics(
                blk,
                effective_crop_xyxy=bbox,
                crop_source=crop_source,
            )
            jobs.append((blk, bbox, crop_source))

        if not jobs:
            return blk_list

        worker_count = min(self.parallel_workers, len(jobs))
        logger.info(
            "mangalmm_ocr start: blocks=%d workers=%d max_completion_tokens=%d endpoint=%s",
            len(jobs),
            worker_count,
            self.max_completion_tokens,
            self._chat_completions_url(),
        )

        started_at = time.perf_counter()
        if worker_count <= 1:
            for blk, bbox, crop_source in jobs:
                self._process_block(img, blk, bbox, crop_source)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self._process_block, img, blk, bbox, crop_source): (blk, bbox)
                    for blk, bbox, crop_source in jobs
                }
                try:
                    for future in as_completed(future_map):
                        future.result()
                except Exception:
                    for future in future_map:
                        future.cancel()
                    raise

        logger.info(
            "mangalmm_ocr complete: blocks=%d elapsed_ms=%.1f",
            len(jobs),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return blk_list

    def _process_block(
        self,
        image: np.ndarray,
        blk: TextBlock,
        bbox: tuple[int, int, int, int],
        crop_source: str,
    ) -> None:
        crop = self._crop_image(image, bbox)
        if crop is None:
            self._mark_empty(blk, "Invalid OCR crop bounds.")
            return

        request_image, scale = self._resize_for_request(crop)
        raw_text = self._request_response_text(request_image)
        parsed_regions = self._parse_region_payload(raw_text)

        if not parsed_regions:
            self._mark_empty(blk, "MangaLMM returned no valid OCR regions.", raw_text=raw_text)
            blk.ocr_crop_bbox = [int(v) for v in bbox]
            blk.ocr_resize_scale = float(scale)
            blk.ocr_regions = []
            return

        mapped_regions = self._map_regions_to_original_bbox(parsed_regions, bbox, crop.shape[:2], scale)
        if not mapped_regions:
            self._mark_empty(blk, "MangaLMM returned OCR regions outside the crop bounds.", raw_text=raw_text)
            blk.ocr_crop_bbox = [int(v) for v in bbox]
            blk.ocr_resize_scale = float(scale)
            blk.ocr_regions = []
            return

        sorted_entries = sort_textblock_rectangles(
            [
                (tuple(region["bbox_xyxy"]), region["text"])
                for region in mapped_regions
                if str(region.get("text", "")).strip()
            ],
            blk.source_lang_direction,
        )
        texts = [text for _bbox, text in sorted_entries if str(text or "").strip()]
        if is_no_space_lang(getattr(blk, "source_lang", "")):
            final_text = "".join(texts)
        else:
            final_text = " ".join(texts)

        blk.texts = texts
        blk.text = final_text
        blk.ocr_regions = mapped_regions
        blk.ocr_crop_bbox = [int(v) for v in bbox]
        blk.ocr_resize_scale = float(scale)
        set_block_ocr_crop_diagnostics(
            blk,
            effective_crop_xyxy=bbox,
            crop_source=crop_source,
        )
        set_block_ocr_diagnostics(
            blk,
            text=final_text,
            confidence=0.0,
            status=OCR_STATUS_OK,
            empty_reason="",
            attempt_count=1,
            raw_text=raw_text,
            sanitized_text=final_text,
        )

    def _resize_for_request(self, crop: np.ndarray) -> tuple[np.ndarray, float]:
        request_image = crop
        scale = 1.0
        if not self.safe_resize:
            return request_image, scale

        height, width = crop.shape[:2]
        if height <= 0 or width <= 0:
            return request_image, scale

        area = width * height
        long_side = max(width, height)
        if area <= self.max_pixels and long_side <= self.max_long_side:
            return request_image, scale

        scale = min(
            1.0,
            math.sqrt(self.max_pixels / float(area)) if area > 0 else 1.0,
            self.max_long_side / float(long_side) if long_side > 0 else 1.0,
        )
        if scale >= 1.0:
            return request_image, 1.0

        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        request_image = cv2.resize(crop, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
        return request_image, float(scale)

    def _map_regions_to_original_bbox(
        self,
        regions: list[dict[str, object]],
        crop_bbox: tuple[int, int, int, int],
        crop_shape: tuple[int, int],
        scale: float,
    ) -> list[dict[str, object]]:
        crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in crop_bbox]
        crop_h, crop_w = crop_shape
        inverse_scale = 1.0 / scale if scale > 0 else 1.0
        mapped: list[dict[str, object]] = []

        for region in regions:
            bbox = region.get("bbox_2d")
            text = str(region.get("text_content", "") or "").strip()
            if not isinstance(bbox, list) or len(bbox) != 4 or not text:
                continue
            x1_f, y1_f, x2_f, y2_f = [float(value) for value in bbox]
            local_x1 = max(0.0, min(x1_f, x2_f) * inverse_scale)
            local_y1 = max(0.0, min(y1_f, y2_f) * inverse_scale)
            local_x2 = min(float(crop_w), max(x1_f, x2_f) * inverse_scale)
            local_y2 = min(float(crop_h), max(y1_f, y2_f) * inverse_scale)

            if local_x2 <= local_x1 or local_y2 <= local_y1:
                continue

            page_x1 = int(round(crop_x1 + local_x1))
            page_y1 = int(round(crop_y1 + local_y1))
            page_x2 = int(round(crop_x1 + local_x2))
            page_y2 = int(round(crop_y1 + local_y2))

            page_x1 = max(crop_x1, min(page_x1, crop_x2))
            page_y1 = max(crop_y1, min(page_y1, crop_y2))
            page_x2 = max(crop_x1, min(page_x2, crop_x2))
            page_y2 = max(crop_y1, min(page_y2, crop_y2))
            if page_x2 <= page_x1 or page_y2 <= page_y1:
                continue

            mapped.append(
                {
                    "bbox_xyxy": [page_x1, page_y1, page_x2, page_y2],
                    "text": text,
                }
            )

        return mapped

    def _request_response_text(self, image: np.ndarray) -> str:
        data_url = self._image_data_url(image)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": DEFAULT_MANGALMM_TEMPERATURE,
            "top_k": DEFAULT_MANGALMM_TOP_K,
            "max_completion_tokens": self.max_completion_tokens,
        }
        data = self._send_request(payload)
        response_text = self._extract_text_from_response(data)
        if self.raw_response_logging:
            logger.info("mangalmm_ocr raw response text: %s", response_text)
        return response_text

    def _send_request(self, payload: dict) -> dict:
        try:
            response = requests.post(
                self._chat_completions_url(),
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.request_timeout_sec,
            )
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                f"Unable to reach the local {SERVICE_NAME} service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned HTTP {response.status_code}.",
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

        if self.raw_response_logging:
            logger.info("mangalmm_ocr raw response json: %s", data)
        return data

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

        message = choices[0].get("message")
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
        raw = str(text or "").strip()
        if not raw:
            return []

        candidates = [raw]
        unfenced = self._strip_code_fences(raw)
        if unfenced and unfenced not in candidates:
            candidates.append(unfenced)

        for candidate in candidates:
            decoded = self._extract_first_json_array(candidate)
            normalized = self._normalize_region_list(decoded)
            if normalized:
                return normalized
        return []

    def _extract_first_json_array(self, text: str) -> object:
        decoder = json.JSONDecoder()
        start = text.find("[")
        while start != -1:
            try:
                payload, _end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                start = text.find("[", start + 1)
                continue
            if isinstance(payload, list):
                return payload
            start = text.find("[", start + 1)
        return None

    def _normalize_region_list(self, payload: object) -> list[dict[str, object]]:
        if not isinstance(payload, list):
            return []

        normalized: list[dict[str, object]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox_2d")
            text = str(item.get("text_content", "") or "").strip()
            if not isinstance(bbox, list) or len(bbox) != 4 or not text:
                continue
            try:
                coords = [float(value) for value in bbox]
            except (TypeError, ValueError):
                continue
            normalized.append(
                {
                    "bbox_2d": coords,
                    "text_content": text,
                }
            )
        return normalized

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        stripped = str(text or "").strip()
        if not stripped.startswith("```"):
            return stripped
        stripped = re.sub(r"^\s*```(?:json)?\s*", "", stripped, count=1, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```\s*$", "", stripped, count=1)
        return stripped.strip()

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

    def _image_data_url(self, image: np.ndarray) -> str:
        image_bytes = self._encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{image_b64}"

    def _encode_image(self, image: np.ndarray) -> bytes:
        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            raise LocalServiceResponseError(
                f"Failed to encode OCR crop for {SERVICE_NAME}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )
        return encoded.tobytes()

    def _chat_completions_url(self) -> str:
        base = (self.server_url or DEFAULT_MANGALMM_SERVER_URL).rstrip("/")
        if base.endswith(self.REQUEST_ENDPOINT_SUFFIX):
            return base
        return f"{base}{self.REQUEST_ENDPOINT_SUFFIX}"

    def _mark_empty(self, blk: TextBlock, reason: str, *, raw_text: str = "") -> None:
        blk.texts = []
        blk.ocr_regions = []
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

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))
