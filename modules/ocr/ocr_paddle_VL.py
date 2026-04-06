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
DEFAULT_PADDLEOCR_VL_SERVER_URL = "http://127.0.0.1:8118/v1"
DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS = 2
DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC = 60
DEFAULT_PADDLEOCR_VL_MODEL_NAME = "PaddleOCR-VL-1.5-0.9B"
DEFAULT_PADDLEOCR_VL_MAX_TOKENS = 192
RETRY_PADDLEOCR_VL_MAX_TOKENS = 256


class PaddleOCRVLEngine(OCREngine):
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_TIMEOUT_RANGE = (1, 3600)
    REQUEST_ENDPOINT_SUFFIX = "/chat/completions"
    MODELS_ENDPOINT_SUFFIX = "/models"
    PROMPT = "OCR:"
    TEXT_EXPANSION_RATIO = 0.05
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
    MAX_TOKENS_ERROR_PATTERNS = (
        "max_tokens",
        "max token",
    )

    def __init__(self) -> None:
        self.server_url = DEFAULT_PADDLEOCR_VL_SERVER_URL
        self.parallel_workers = DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS
        self.request_timeout_sec = DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC
        self.raw_response_logging = False
        self._supports_max_tokens = True
        self._model_id: str | None = None
        self._session_local = threading.local()

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_paddleocr_vl_settings()
        self.server_url = self._normalize_server_url(
            config.get("server_url", DEFAULT_PADDLEOCR_VL_SERVER_URL) or DEFAULT_PADDLEOCR_VL_SERVER_URL
        )
        self.parallel_workers = self._clamp_int(
            config.get("parallel_workers", DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS),
            DEFAULT_PADDLEOCR_VL_PARALLEL_WORKERS,
            self.PARALLEL_WORKERS_RANGE,
        )
        self.request_timeout_sec = self._clamp_int(
            config.get("request_timeout_sec", DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC),
            DEFAULT_PADDLEOCR_VL_REQUEST_TIMEOUT_SEC,
            self.REQUEST_TIMEOUT_RANGE,
        )
        self.raw_response_logging = bool(config.get("raw_response_logging", False))
        self._supports_max_tokens = True
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
            "paddleocr_vl start: blocks=%d workers=%d endpoint=%s",
            len(jobs),
            worker_count,
            self._chat_completions_url(),
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
            "paddleocr_vl complete: blocks=%d elapsed_ms=%.1f",
            len(jobs),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return blk_list

    def _process_block(self, img: np.ndarray, blk: TextBlock, bbox: tuple[int, int, int, int]) -> None:
        crop = self._crop_image(img, bbox)
        if crop is None:
            self._mark_empty(blk, "Invalid OCR crop bounds.")
            return

        primary_text = self._normalize_output_text(
            self._request_ocr_text(crop, max_tokens=DEFAULT_PADDLEOCR_VL_MAX_TOKENS)
        )
        primary_issue = self._classify_retry_reason(primary_text)
        artifact_retry_crop = None
        if primary_text and self.LEADING_ARTIFACT_PATTERN.match(primary_text):
            artifact_retry_crop = self._build_artifact_trim_crop(crop)

        if primary_issue is None:
            if artifact_retry_crop is not None:
                retry_text = self._normalize_output_text(
                    self._request_ocr_text(artifact_retry_crop, max_tokens=RETRY_PADDLEOCR_VL_MAX_TOKENS)
                )
                retry_issue = self._classify_retry_reason(retry_text)
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
        retry_text = self._normalize_output_text(
            self._request_ocr_text(retry_crop, max_tokens=RETRY_PADDLEOCR_VL_MAX_TOKENS)
        )
        retry_issue = self._classify_retry_reason(retry_text)
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

        self._mark_empty(
            blk,
            self._empty_reason_for(primary_issue, retry_issue),
            attempt_count=2,
        )

    def _request_ocr_text(self, image: np.ndarray, *, max_tokens: int | None) -> str:
        data_url = self._image_data_url(image)
        payload = {
            "model": self._discover_model_id(),
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": self.PROMPT},
                    ],
                }
            ],
            "temperature": 0.0,
        }
        if self._supports_max_tokens and max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)

        data = self._send_request(payload)
        return self._extract_text_from_response(data)

    def _send_request(self, payload: dict) -> dict:
        response, data = self._post_json(self._chat_completions_url(), payload)

        if payload.get("max_tokens") is not None and self._max_tokens_was_rejected(response, data):
            self._supports_max_tokens = False
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
            logger.info("paddleocr_vl raw response json: %s", data)

        return data

    def _post_json(self, url: str, payload: dict) -> tuple[requests.Response, dict | None]:
        session = self._get_session()
        try:
            response = session.post(url, json=payload, timeout=self.request_timeout_sec)
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

    def _extract_text_from_response(self, data: dict) -> str:
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
            logger.info("paddleocr_vl raw content: %s", content_text)
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

        models_url = self._models_url()
        session = self._get_session()
        try:
            response = session.get(models_url, timeout=self.request_timeout_sec)
        except requests.exceptions.RequestException as exc:
            logger.warning("paddleocr_vl model autodiscovery failed: %s", exc)
            self._model_id = DEFAULT_PADDLEOCR_VL_MODEL_NAME
            return self._model_id

        try:
            data = response.json()
        except ValueError:
            data = None

        if response.status_code == 200 and isinstance(data, dict):
            models = data.get("data")
            if isinstance(models, list):
                for item in models:
                    if isinstance(item, dict):
                        model_id = str(item.get("id", "") or "").strip()
                        if model_id:
                            self._model_id = model_id
                            return self._model_id

        self._model_id = DEFAULT_PADDLEOCR_VL_MODEL_NAME
        return self._model_id

    def _resolve_bbox(self, blk: TextBlock, image: np.ndarray) -> tuple[int, int, int, int] | None:
        text_bbox = getattr(blk, "xyxy", None)
        if text_bbox is not None:
            bbox = expand_bbox(
                text_bbox,
                image.shape,
                x_ratio=self.TEXT_EXPANSION_RATIO,
                y_ratio=self.TEXT_EXPANSION_RATIO,
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

    def _classify_retry_reason(self, text: str) -> str | None:
        if not text:
            return "empty"

        lowered = text.lower()
        if any(placeholder in lowered for placeholder in self.PLACEHOLDER_PHRASES):
            return "placeholder"

        if self._looks_structured_output(text):
            return "structured"

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

    def _empty_reason_for(self, primary_issue: str | None, retry_issue: str | None) -> str:
        issue = retry_issue or primary_issue or "empty"
        if issue == "placeholder":
            return f"{SERVICE_NAME} returned placeholder text instead of OCR output."
        if issue == "structured":
            return f"{SERVICE_NAME} returned structured output instead of raw OCR text."
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
        return f"{self.server_url.rstrip('/')}{self.REQUEST_ENDPOINT_SUFFIX}"

    def _models_url(self) -> str:
        return f"{self.server_url.rstrip('/')}{self.MODELS_ENDPOINT_SUFFIX}"

    def _max_tokens_was_rejected(self, response: requests.Response, data: dict | None) -> bool:
        if response is None and data is None:
            return False

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
        if not text:
            return ""
        normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        normalized = re.sub(r"\n{3,}", "\n\n", normalized)
        return normalized.strip()

    @classmethod
    def _normalize_for_artifact_compare(cls, text: str) -> str:
        stripped = cls.LEADING_ARTIFACT_PATTERN.sub("", cls._normalize_output_text(text), count=1)
        return re.sub(r"\s+", "", stripped)

    @classmethod
    def _normalize_server_url(cls, url: str) -> str:
        base = (url or DEFAULT_PADDLEOCR_VL_SERVER_URL).strip().rstrip("/")
        for suffix in (cls.REQUEST_ENDPOINT_SUFFIX, cls.MODELS_ENDPOINT_SUFFIX, "/layout-parsing"):
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
