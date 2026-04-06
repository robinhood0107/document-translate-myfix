from __future__ import annotations

import base64
import logging
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import requests

from modules.detection.utils.content import detect_content_in_bbox
from modules.utils.exceptions import (
    LocalServiceConnectionError,
    LocalServiceResponseError,
)
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_AFTER_RETRY,
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_OK,
    OCR_STATUS_OK_AFTER_RETRY,
    ensure_three_channel,
    expand_bbox,
    set_block_ocr_diagnostics,
)
from modules.utils.textblock import TextBlock

from .base import OCREngine


logger = logging.getLogger(__name__)

SERVICE_NAME = "PaddleOCR VL"
SETTINGS_PAGE_NAME = "PaddleOCR VL Settings"
DEFAULT_DIRECT_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"


class PaddleOCRVLEngine(OCREngine):
    DEFAULT_SERVER_URL = "http://127.0.0.1:28118/layout-parsing"
    DEFAULT_MAX_NEW_TOKENS = 256
    DEFAULT_PARALLEL_WORKERS = 2
    MAX_NEW_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_TIMEOUT_RANGE = (1, 3600)
    REQUEST_TIMEOUT_SECONDS = 60
    TEXT_EXPANSION_RATIO = 0.05
    CHAT_COMPLETIONS_SUFFIX = "/chat/completions"
    MODELS_SUFFIX = "/models"
    LAYOUT_SUFFIX = "/layout-parsing"
    DIRECT_PROMPT = "OCR:"
    DIRECT_TOKEN_LADDER = (3, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
    LEADING_ARTIFACT_PATTERN = re.compile(r"^\s*[・•·▪◦●○◎◇◆#*\-]+\s*")
    PLACEHOLDER_PHRASES = (
        "the quick brown fox jumps over the lazy dog.",
    )
    STRUCTURED_OUTPUT_PATTERNS = (
        r"^\s*```",
        r"^\s*[\{\[][^\n]*[\}\]]\s*$",
        r"^\s*<\s*(?:html|body|div|span|table|tr|td|th)\b",
        r"^\s*\|.+\|\s*$",
    )
    STRUCTURED_OUTPUT_FIELD_HINTS = (
        '"bbox"',
        '"box"',
        '"coord"',
        '"coords"',
        '"points"',
        '"x1"',
        '"y1"',
        '"x2"',
        '"y2"',
        '"markdown"',
        '"html"',
    )
    MAX_NEW_TOKENS_ERROR_PATTERNS = (
        "maxNewTokens",
    )
    MAX_TOKENS_ERROR_PATTERNS = (
        "max_tokens",
        "max token",
    )

    def __init__(self) -> None:
        self.server_url = self.DEFAULT_SERVER_URL
        self.prettify_markdown = False
        self.visualize = False
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS
        self.parallel_workers = self.DEFAULT_PARALLEL_WORKERS
        self.request_timeout_seconds = self.REQUEST_TIMEOUT_SECONDS
        self.raw_response_logging = False
        self._request_mode = "layout"
        self._supports_layout_max_new_tokens = True
        self._supports_direct_max_tokens = True
        self._model_id: str | None = None
        self._session_local = threading.local()

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_paddleocr_vl_settings()
        raw_server_url = config.get("server_url", self.DEFAULT_SERVER_URL) or self.DEFAULT_SERVER_URL
        self._request_mode = self._detect_request_mode(raw_server_url)
        if self._request_mode == "direct":
            self.server_url = self._normalize_direct_server_url(raw_server_url)
        else:
            self.server_url = self._normalize_layout_server_url(raw_server_url)

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
        self.request_timeout_seconds = self._clamp_int(
            config.get("request_timeout_sec", self.REQUEST_TIMEOUT_SECONDS),
            self.REQUEST_TIMEOUT_SECONDS,
            self.REQUEST_TIMEOUT_RANGE,
        )
        self.raw_response_logging = bool(config.get("raw_response_logging", False))
        self._supports_layout_max_new_tokens = True
        self._supports_direct_max_tokens = True
        self._model_id = None
        self._session_local = threading.local()

    def process_image(self, img: np.ndarray, blk_list: list[TextBlock]) -> list[TextBlock]:
        jobs: list[tuple[TextBlock, tuple[int, int, int, int]]] = []
        for blk in blk_list or []:
            bbox = self._resolve_bbox(blk, img)
            if bbox is None:
                self._mark_empty(blk, "Invalid OCR crop bounds.")
                continue
            jobs.append((blk, bbox))

        if not jobs:
            return blk_list

        worker_count = min(self.parallel_workers, len(jobs))
        logger.info(
            "paddleocr_vl start: mode=%s blocks=%d workers=%d endpoint=%s",
            self._request_mode,
            len(jobs),
            worker_count,
            self.server_url,
        )

        started_at = time.perf_counter()
        if worker_count <= 1:
            for blk, bbox in jobs:
                self._process_block(img, blk, bbox)
        else:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_map = {
                    executor.submit(self._process_block, img, blk, bbox): (blk, bbox)
                    for blk, bbox in jobs
                }
                try:
                    for future in as_completed(future_map):
                        future.result()
                except Exception:
                    for future in future_map:
                        future.cancel()
                    raise

        logger.info(
            "paddleocr_vl complete: mode=%s blocks=%d elapsed_ms=%.1f",
            self._request_mode,
            len(jobs),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return blk_list

    def _process_block(self, img: np.ndarray, blk: TextBlock, bbox: tuple[int, int, int, int]) -> None:
        crop = self._crop_image(img, bbox)
        if crop is None:
            self._mark_empty(blk, "Invalid OCR crop bounds.")
            return

        if self._request_mode == "direct":
            self._process_direct_block(crop, blk)
            return

        text = self._request_layout_text(crop)
        cleaned = self._normalize_output_text(text)
        if cleaned:
            set_block_ocr_diagnostics(
                blk,
                text=cleaned,
                confidence=0.0,
                status=OCR_STATUS_OK,
                empty_reason="",
                attempt_count=1,
            )
        else:
            self._mark_empty(blk, "PaddleOCR VL returned no text.")

    def _process_direct_block(self, crop: np.ndarray, blk: TextBlock) -> None:
        metrics = self._content_metrics(crop)
        primary_max_tokens = self._select_direct_max_tokens(crop)
        primary_text = self._normalize_output_text(
            self._request_direct_text(crop, max_tokens=primary_max_tokens)
        )
        primary_issue = self._classify_direct_retry_reason(primary_text, metrics=metrics)
        artifact_retry_crop = None
        if primary_text and self.LEADING_ARTIFACT_PATTERN.match(primary_text):
            artifact_retry_crop = self._build_artifact_trim_crop(crop)

        if primary_issue is None:
            if self._should_try_tiny_budget_probe(primary_text, metrics):
                retry_text = self._normalize_output_text(self._request_direct_text(crop, max_tokens=3))
                retry_issue = self._classify_direct_retry_reason(retry_text, metrics=metrics)
                if (
                    retry_issue is None
                    and retry_text
                    and len(retry_text) < len(primary_text)
                    and primary_text.startswith(retry_text)
                ):
                    set_block_ocr_diagnostics(
                        blk,
                        text=retry_text,
                        confidence=0.0,
                        status=OCR_STATUS_OK_AFTER_RETRY,
                        empty_reason="",
                        attempt_count=2,
                    )
                    return

            if self._should_try_high_contrast_retry(primary_text, crop, metrics):
                contrast_retry_crop = self._build_high_contrast_retry_crop(crop)
                if contrast_retry_crop is not None:
                    retry_text = self._normalize_output_text(
                        self._request_direct_text(
                            contrast_retry_crop,
                            max_tokens=min(primary_max_tokens, 12),
                        )
                    )
                    retry_issue = self._classify_direct_retry_reason(retry_text, metrics=metrics)
                    if (
                        retry_issue is None
                        and retry_text
                        and len(retry_text) > len(primary_text)
                    ):
                        set_block_ocr_diagnostics(
                            blk,
                            text=retry_text,
                            confidence=0.0,
                            status=OCR_STATUS_OK_AFTER_RETRY,
                            empty_reason="",
                            attempt_count=2,
                        )
                        return

            if artifact_retry_crop is not None:
                retry_text = self._normalize_output_text(
                    self._request_direct_text(
                        artifact_retry_crop,
                        max_tokens=self._retry_direct_max_tokens(primary_max_tokens, issue="artifact"),
                    )
                )
                retry_issue = self._classify_direct_retry_reason(retry_text, metrics=metrics)
                if retry_issue is None and self._artifact_retry_is_preferred(primary_text, retry_text):
                    set_block_ocr_diagnostics(
                        blk,
                        text=retry_text,
                        confidence=0.0,
                        status=OCR_STATUS_OK_AFTER_RETRY,
                        empty_reason="",
                        attempt_count=2,
                    )
                    return

            set_block_ocr_diagnostics(
                blk,
                text=primary_text,
                confidence=0.0,
                status=OCR_STATUS_OK,
                empty_reason="",
                attempt_count=1,
            )
            return

        retry_crop = artifact_retry_crop if artifact_retry_crop is not None else crop
        retry_max_tokens = self._retry_direct_max_tokens(
            primary_max_tokens,
            issue=primary_issue,
            metrics=metrics,
        )
        retry_text = self._normalize_output_text(
            self._request_direct_text(
                retry_crop,
                max_tokens=retry_max_tokens,
            )
        )
        retry_issue = self._classify_direct_retry_reason(retry_text, metrics=metrics)
        if retry_issue is None:
            set_block_ocr_diagnostics(
                blk,
                text=retry_text,
                confidence=0.0,
                status=OCR_STATUS_OK_AFTER_RETRY,
                empty_reason="",
                attempt_count=2,
            )
            return

        if retry_issue == "repetition":
            recovered_text, extra_attempts = self._recover_from_repetition_backoff(
                retry_crop,
                start_tokens=retry_max_tokens,
                metrics=metrics,
            )
            if recovered_text:
                set_block_ocr_diagnostics(
                    blk,
                    text=recovered_text,
                    confidence=0.0,
                    status=OCR_STATUS_OK_AFTER_RETRY,
                    empty_reason="",
                    attempt_count=2 + extra_attempts,
                )
                return

        self._mark_empty(
            blk,
            self._empty_reason_for(primary_issue, retry_issue),
            attempt_count=2,
        )

    def _request_layout_text(self, image: np.ndarray) -> str:
        image_bytes = self._encode_image(image)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        payload = {
            "file": image_b64,
            "fileType": 1,
            "prettifyMarkdown": self.prettify_markdown,
            "visualize": self.visualize,
        }
        if self._supports_layout_max_new_tokens:
            payload["maxNewTokens"] = self.max_new_tokens

        data = self._send_layout_request(payload)
        if self._supports_layout_max_new_tokens and self._response_rejected_max_new_tokens(data):
            self._supports_layout_max_new_tokens = False
            payload.pop("maxNewTokens", None)
            data = self._send_layout_request(payload)

        error_code = data.get("errorCode")
        if error_code not in (None, 0):
            error_msg = str(data.get("errorMsg", "") or "Unknown PaddleOCR VL error.")
            raise LocalServiceResponseError(
                error_msg,
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        return self._extract_text_from_layout_response(data)

    def _request_direct_text(self, image: np.ndarray, *, max_tokens: int | None) -> str:
        payload = {
            "model": self._discover_model_id(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": self._image_data_url(image)}},
                        {"type": "text", "text": self.DIRECT_PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        if self._supports_direct_max_tokens and max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        data = self._send_direct_request(payload)
        return self._extract_text_from_direct_response(data)

    def _send_layout_request(self, payload: dict) -> dict:
        response, data = self._post_json(self.server_url, payload)
        if response.status_code != 200 and payload.get("maxNewTokens") is not None:
            legacy_payload = dict(payload)
            legacy_payload.pop("maxNewTokens", None)
            legacy_response, legacy_data = self._post_json(self.server_url, legacy_payload)
            if legacy_response.status_code == 200:
                response = legacy_response
                data = legacy_data
                self._supports_layout_max_new_tokens = False

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned HTTP {response.status_code}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        if not isinstance(data, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned invalid JSON.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        if self.raw_response_logging:
            logger.info("paddleocr_vl raw layout response json: %s", data)
        return data

    def _send_direct_request(self, payload: dict) -> dict:
        response, data = self._post_json(self._chat_completions_url(), payload)
        if payload.get("max_tokens") is not None and self._max_tokens_was_rejected(response, data):
            self._supports_direct_max_tokens = False
            retry_payload = dict(payload)
            retry_payload.pop("max_tokens", None)
            response, data = self._post_json(self._chat_completions_url(), retry_payload)

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned HTTP {response.status_code}.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        if not isinstance(data, dict):
            raise LocalServiceResponseError(
                f"{SERVICE_NAME} service returned invalid JSON.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            )

        if self.raw_response_logging:
            logger.info("paddleocr_vl raw direct response json: %s", data)
        return data

    def _post_json(self, url: str, payload: dict) -> tuple[requests.Response, dict | None]:
        session = self._get_session()
        try:
            response = session.post(url, json=payload, timeout=self.request_timeout_seconds)
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                f"Unable to reach the local {SERVICE_NAME} service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        try:
            data = response.json()
        except ValueError:
            data = None
        return response, data

    def _get_json(self, url: str) -> tuple[requests.Response, dict | None]:
        session = self._get_session()
        try:
            response = session.get(url, timeout=self.request_timeout_seconds)
        except requests.exceptions.RequestException as exc:
            raise LocalServiceConnectionError(
                f"Unable to reach the local {SERVICE_NAME} service.",
                service_name=SERVICE_NAME,
                settings_page_name=SETTINGS_PAGE_NAME,
            ) from exc

        try:
            data = response.json()
        except ValueError:
            data = None
        return response, data

    def _extract_text_from_layout_response(self, data: dict) -> str:
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

    def _extract_text_from_direct_response(self, data: dict) -> str:
        error_message = self._extract_error_message(data)
        if error_message:
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

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content_text = self._extract_content_text(message.get("content"))
        if self.raw_response_logging and content_text:
            logger.info("paddleocr_vl raw direct content: %s", content_text)
        return content_text

    def _extract_content_text(self, content) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str) and text_value.strip():
                    parts.append(text_value)
            return "\n".join(parts)
        return ""

    def _discover_model_id(self) -> str:
        if self._model_id:
            return self._model_id

        response, data = self._get_json(self._models_url())
        if response.status_code == 200 and isinstance(data, dict):
            models = data.get("data")
            if isinstance(models, list):
                for item in models:
                    if isinstance(item, dict):
                        model_id = str(item.get("id", "") or "").strip()
                        if model_id:
                            self._model_id = model_id
                            return self._model_id

        self._model_id = DEFAULT_DIRECT_MODEL_NAME
        return self._model_id

    def _resolve_bbox(self, blk: TextBlock, image: np.ndarray) -> tuple[int, int, int, int] | None:
        if self._request_mode == "direct":
            text_bbox = getattr(blk, "xyxy", None)
            if text_bbox is not None:
                x_ratio, y_ratio = self._direct_text_expansion_ratios(text_bbox)
                bbox = expand_bbox(
                    text_bbox,
                    image.shape,
                    x_ratio=x_ratio,
                    y_ratio=y_ratio,
                )
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2

            bubble_bbox = getattr(blk, "bubble_xyxy", None)
            if bubble_bbox is not None:
                bbox = expand_bbox(bubble_bbox, image.shape)
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    return x1, y1, x2, y2
            return None

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

    def _direct_text_expansion_ratios(self, text_bbox) -> tuple[float, float]:
        if text_bbox is None or len(text_bbox) != 4:
            return self.TEXT_EXPANSION_RATIO, self.TEXT_EXPANSION_RATIO

        x1, y1, x2, y2 = (int(value) for value in text_bbox)
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)

        # Tighten the crop slightly for tall narrow vertical SFX, but keep
        # ultra-thin glyph columns on the default expansion path.
        if height >= 600 and 100 <= width <= 120 and height >= width * 5:
            return 0.02, 0.02
        if height >= 450 and 110 <= width <= 140 and height >= width * 4:
            return 0.02, 0.02
        return self.TEXT_EXPANSION_RATIO, self.TEXT_EXPANSION_RATIO

    def _crop_image(self, image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        x1, y1, x2, y2 = bbox
        crop = image[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None
        return ensure_three_channel(crop)

    def _build_artifact_trim_crop(self, crop: np.ndarray) -> np.ndarray | None:
        content_bboxes = detect_content_in_bbox(crop, min_area=10, margin=1)
        if content_bboxes.shape[0] < 2:
            return None

        bboxes = np.asarray(content_bboxes, dtype=int)
        order = np.argsort(bboxes[:, 1], kind="stable")
        bboxes = bboxes[order]

        top_min = int(bboxes[0, 1])
        top_group_mask = bboxes[:, 1] <= top_min + 3
        top_group = bboxes[top_group_mask]
        remaining = bboxes[~top_group_mask]
        if top_group.size == 0 or remaining.size == 0:
            return None

        top_y2 = int(np.max(top_group[:, 3]))
        main_y1 = int(np.min(remaining[:, 1]))
        gap = main_y1 - top_y2
        if gap < max(2, int(round(crop.shape[0] * 0.04))):
            return None

        top_height = int(np.max(top_group[:, 3]) - np.min(top_group[:, 1]))
        if top_height > max(8, int(round(crop.shape[0] * 0.2))):
            return None

        top_area = int(np.sum((top_group[:, 2] - top_group[:, 0]) * (top_group[:, 3] - top_group[:, 1])))
        main_area = int(np.sum((remaining[:, 2] - remaining[:, 0]) * (remaining[:, 3] - remaining[:, 1])))
        if main_area <= 0 or top_area >= max(12, int(main_area * 0.12)):
            return None

        if top_y2 >= int(round(crop.shape[0] * 0.35)):
            return None

        trim_y = min(crop.shape[0] - 1, top_y2 + 1)
        trimmed = crop[trim_y:, :]
        if trimmed.size == 0 or trimmed.shape[0] < max(8, int(round(crop.shape[0] * 0.4))):
            return None

        return ensure_three_channel(trimmed)

    def _build_high_contrast_retry_crop(self, crop: np.ndarray) -> np.ndarray | None:
        grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(grayscale, 180, 255, cv2.THRESH_BINARY)
        if binary is None or binary.size == 0:
            return None
        contrast = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return ensure_three_channel(contrast)

    def _artifact_retry_is_preferred(self, primary_text: str, retry_text: str) -> bool:
        if not primary_text or not retry_text or primary_text == retry_text:
            return False
        if not self.LEADING_ARTIFACT_PATTERN.match(primary_text):
            return False
        if self.LEADING_ARTIFACT_PATTERN.match(retry_text):
            return False

        primary_compare = self._normalize_for_artifact_compare(primary_text)
        retry_compare = self._normalize_for_artifact_compare(retry_text)
        if not primary_compare or not retry_compare:
            return False

        primary_without_artifact = self._normalize_for_artifact_compare(
            self.LEADING_ARTIFACT_PATTERN.sub("", primary_text, count=1)
        )
        if not primary_without_artifact:
            return False

        return retry_compare == primary_without_artifact == primary_compare

    def _select_direct_max_tokens(self, crop: np.ndarray) -> int:
        metrics = self._content_metrics(crop)
        height = metrics["height"]
        width = metrics["width"]
        component_count = metrics["component_count"]
        density = metrics["density"]
        area = int(height * width)
        longest = max(height, width)

        if 90 <= width <= 120 and height <= 140 and component_count <= 6 and density >= 0.75:
            return 3
        if height <= 90 and component_count <= 30 and density <= 0.12:
            return 6
        if width >= 300 and height >= 300 and component_count <= 45 and 0.45 <= density <= 0.65:
            return 6
        if height >= 500 and height > width and 80 <= component_count <= 100:
            return 256
        if area <= 12_000 or longest <= 96:
            return 12
        if area <= 28_000 or longest <= 160:
            return 24
        if area <= 72_000:
            return 48
        if area <= 160_000:
            return 96
        return 192

    def _retry_direct_max_tokens(
        self,
        primary_max_tokens: int,
        issue: str,
        *,
        metrics: dict[str, float | int] | None = None,
    ) -> int:
        ladder = list(self.DIRECT_TOKEN_LADDER)
        if primary_max_tokens not in ladder:
            ladder.append(primary_max_tokens)
            ladder.sort()
        index = ladder.index(primary_max_tokens)
        if issue == "empty":
            return ladder[min(index + 1, len(ladder) - 1)]
        if issue == "repetition":
            component_count = int((metrics or {}).get("component_count", 0) or 0)
            if component_count >= 80:
                return ladder[min(index + 1, len(ladder) - 1)]
            return ladder[max(index - 2, 0)]
        if issue in {"placeholder", "structured", "artifact"}:
            return ladder[max(index - 1, 0)]
        return primary_max_tokens

    def _recover_from_repetition_backoff(
        self,
        crop: np.ndarray,
        *,
        start_tokens: int,
        metrics: dict[str, float | int],
    ) -> tuple[str, int]:
        attempts = 0
        for token_budget in reversed(self.DIRECT_TOKEN_LADDER):
            if token_budget >= start_tokens:
                continue
            attempts += 1
            candidate_text = self._normalize_output_text(
                self._request_direct_text(crop, max_tokens=token_budget)
            )
            candidate_issue = self._classify_direct_retry_reason(candidate_text, metrics=metrics)
            if candidate_issue is None and candidate_text:
                return candidate_text, attempts
        return "", attempts

    def _classify_direct_retry_reason(
        self,
        text: str,
        *,
        metrics: dict[str, float | int] | None = None,
    ) -> str | None:
        if not text:
            return "empty"

        lowered = text.lower()
        if any(placeholder in lowered for placeholder in self.PLACEHOLDER_PHRASES):
            return "placeholder"
        if self._looks_structured_output(text):
            return "structured"
        if self._looks_repetition_failure(text, metrics=metrics):
            return "repetition"
        return None

    def _looks_structured_output(self, text: str) -> bool:
        stripped = (text or "").strip()
        if not stripped:
            return False

        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in self.STRUCTURED_OUTPUT_PATTERNS):
            return True

        lowered = stripped.lower()
        if any(hint in lowered for hint in self.STRUCTURED_OUTPUT_FIELD_HINTS):
            digit_count = sum(char.isdigit() for char in stripped)
            punctuation_count = sum(stripped.count(char) for char in "[]{}(),:")
            if digit_count >= 4 and punctuation_count >= 4:
                return True

        return False

    def _looks_repetition_failure(
        self,
        text: str,
        *,
        metrics: dict[str, float | int] | None = None,
    ) -> bool:
        stripped = self._normalize_output_text(text)
        if not stripped:
            return False

        component_count = int((metrics or {}).get("component_count", 0) or 0)
        width = int((metrics or {}).get("width", 0) or 0)
        height = int((metrics or {}).get("height", 0) or 0)
        tall_narrow_crop = height >= 500 and width <= 140

        normalized_lines = [
            re.sub(r"\s+", "", line)
            for line in stripped.splitlines()
            if re.sub(r"\s+", "", line)
        ]
        if normalized_lines:
            line_run = 1
            for idx in range(1, len(normalized_lines)):
                if normalized_lines[idx] == normalized_lines[idx - 1] and not self._is_punctuation_only(
                    normalized_lines[idx]
                ):
                    line_run += 1
                    repeat_threshold = 2 if tall_narrow_crop and len(normalized_lines[idx]) <= 4 else 4
                    if line_run >= repeat_threshold:
                        return True
                else:
                    line_run = 1

        compact = re.sub(r"\s+", "", stripped)
        if len(compact) < 24:
            return False

        max_unit = min(6, len(compact) // 8)
        if max_unit < 2:
            return False

        for unit_len in range(2, max_unit + 1):
            for start in range(0, len(compact) - unit_len * 8 + 1):
                unit = compact[start : start + unit_len]
                if self._is_punctuation_only(unit):
                    continue
                count = 1
                pos = start + unit_len
                while pos + unit_len <= len(compact) and compact[pos : pos + unit_len] == unit:
                    count += 1
                    pos += unit_len
                if count < 8:
                    continue
                span = unit_len * count
                if span / float(len(compact)) >= 0.6:
                    if start > 0 and component_count >= 70 and height >= 450 and len(set(unit)) == 1:
                        continue
                    return True

        return False

    def _should_try_tiny_budget_probe(self, text: str, metrics: dict[str, float | int]) -> bool:
        if not text or len(text) < 2 or len(text) > 3:
            return False
        return (
            90 <= int(metrics["width"]) <= 120
            and int(metrics["height"]) <= 140
            and int(metrics["component_count"]) <= 6
            and float(metrics["density"]) >= 0.75
        )

    def _should_try_high_contrast_retry(
        self,
        text: str,
        crop: np.ndarray,
        metrics: dict[str, float | int],
    ) -> bool:
        if not text or len(text) != 1:
            return False
        if self._is_punctuation_only(text):
            return False
        return crop.shape[0] * crop.shape[1] >= 50_000 and int(metrics["component_count"]) <= 12

    def _content_metrics(self, crop: np.ndarray) -> dict[str, float | int]:
        height, width = crop.shape[:2]
        area = int(height * width)
        content_bboxes = detect_content_in_bbox(crop, min_area=10, margin=1)
        if content_bboxes.size == 0:
            return {
                "width": width,
                "height": height,
                "component_count": 0,
                "density": 0.0,
            }

        bboxes = np.asarray(content_bboxes, dtype=int)
        component_area = int(np.sum((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])))
        density = (component_area / float(area)) if area > 0 else 0.0
        return {
            "width": width,
            "height": height,
            "component_count": int(len(bboxes)),
            "density": float(density),
        }

    def _empty_reason_for(self, primary_issue: str | None, retry_issue: str | None) -> str:
        issue = retry_issue or primary_issue or "empty"
        if issue == "placeholder":
            return f"{SERVICE_NAME} returned placeholder text instead of OCR output."
        if issue == "structured":
            return f"{SERVICE_NAME} returned structured output instead of raw OCR text."
        if issue == "repetition":
            return f"{SERVICE_NAME} returned repeated text instead of stable OCR output."
        return f"{SERVICE_NAME} returned no usable OCR text."

    def _get_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            pool_size = max(4, self.parallel_workers)
            adapter = requests.adapters.HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            session.headers.update({"Content-Type": "application/json"})
            self._session_local.session = session
        return session

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
        return f"{self.server_url.rstrip('/')}{self.CHAT_COMPLETIONS_SUFFIX}"

    def _models_url(self) -> str:
        return f"{self.server_url.rstrip('/')}{self.MODELS_SUFFIX}"

    def _response_rejected_max_new_tokens(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        error_msg = str(data.get("errorMsg", "") or "")
        return any(pattern in error_msg for pattern in self.MAX_NEW_TOKENS_ERROR_PATTERNS)

    def _max_tokens_was_rejected(self, response: requests.Response, data: dict | None) -> bool:
        if any(pattern in (response.text or "").lower() for pattern in self.MAX_TOKENS_ERROR_PATTERNS):
            return True
        error_message = self._extract_error_message(data or {})
        if not error_message:
            return False
        lowered = error_message.lower()
        return any(pattern in lowered for pattern in self.MAX_TOKENS_ERROR_PATTERNS)

    def _extract_error_message(self, data: dict) -> str:
        if not isinstance(data, dict):
            return ""

        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message", "") or "").strip()
        if isinstance(error, str):
            return error.strip()

        detail = data.get("detail")
        if isinstance(detail, str):
            return detail.strip()
        if isinstance(detail, dict):
            return str(detail.get("message", "") or "").strip()

        message = data.get("message")
        if isinstance(message, str):
            return message.strip()

        return ""

    def _mark_empty(self, blk: TextBlock, reason: str, attempt_count: int = 1) -> None:
        status = OCR_STATUS_EMPTY_AFTER_RETRY if attempt_count > 1 else OCR_STATUS_EMPTY_INITIAL
        set_block_ocr_diagnostics(
            blk,
            text="",
            confidence=0.0,
            status=status,
            empty_reason=reason,
            attempt_count=attempt_count,
        )

    def _normalize_output_text(self, text: str) -> str:
        return self._normalize_simple_output_text(text)

    @staticmethod
    def _normalize_simple_output_text(text: str) -> str:
        if not text:
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
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
    def _normalize_for_artifact_compare(text: str) -> str:
        stripped = PaddleOCRVLEngine.LEADING_ARTIFACT_PATTERN.sub(
            "",
            PaddleOCRVLEngine._normalize_simple_output_text(text),
            count=1,
        )
        return re.sub(r"\s+", "", stripped)

    @staticmethod
    def _is_punctuation_only(text: str) -> bool:
        stripped = re.sub(r"\s+", "", text or "")
        if not stripped:
            return True
        return all(not ch.isalnum() and not ("\u3040" <= ch <= "\u30ff") and not ("\u4e00" <= ch <= "\u9fff") for ch in stripped)

    @classmethod
    def _detect_request_mode(cls, url: str) -> str:
        lowered = (url or "").strip().rstrip("/").lower()
        if lowered.endswith(cls.LAYOUT_SUFFIX):
            return "layout"
        if lowered.endswith("/v1") or lowered.endswith(cls.CHAT_COMPLETIONS_SUFFIX) or lowered.endswith(cls.MODELS_SUFFIX):
            return "direct"
        return "layout"

    @classmethod
    def _normalize_layout_server_url(cls, url: str) -> str:
        base = (url or cls.DEFAULT_SERVER_URL).strip().rstrip("/")
        for suffix in (cls.CHAT_COMPLETIONS_SUFFIX, cls.MODELS_SUFFIX, "/v1"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if not base.endswith(cls.LAYOUT_SUFFIX):
            base = f"{base}{cls.LAYOUT_SUFFIX}"
        return base

    @classmethod
    def _normalize_direct_server_url(cls, url: str) -> str:
        base = (url or cls.DEFAULT_SERVER_URL).strip().rstrip("/")
        for suffix in (cls.CHAT_COMPLETIONS_SUFFIX, cls.MODELS_SUFFIX, cls.LAYOUT_SUFFIX):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        if not base.endswith("/v1"):
            base = f"{base}/v1"
        return base

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))
