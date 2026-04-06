from __future__ import annotations

import base64
import logging
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
from modules.utils.ocr_debug import (
    OCR_STATUS_EMPTY_INITIAL,
    OCR_STATUS_EMPTY_AFTER_RETRY,
    OCR_STATUS_OK,
    ensure_three_channel,
    expand_bbox,
    set_block_ocr_diagnostics,
)
from modules.utils.textblock import TextBlock

from .base import OCREngine


logger = logging.getLogger(__name__)

SERVICE_NAME = "HunyuanOCR"
SETTINGS_PAGE_NAME = "HunyuanOCR Settings"
DEFAULT_HUNYUAN_SERVER_URL = "http://127.0.0.1:28080/v1"
DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS = 256
DEFAULT_HUNYUAN_PARALLEL_WORKERS = 2
DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC = 60
DEFAULT_HUNYUAN_TOP_K = 1
DEFAULT_HUNYUAN_REPETITION_PENALTY = 1.0


class HunyuanOCREngine(OCREngine):
    MAX_COMPLETION_TOKENS_RANGE = (64, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_ENDPOINT_SUFFIX = "/chat/completions"
    PROMPT = "提取图中的文字。"
    STRICT_PROMPT = (
        "只返回图片中的原文文字本身。"
        "不要描述图片，不要解释，不要翻译，不要总结，不要补全，"
        "不要添加前缀、引号、坐标、项目符号或Markdown。"
        "保持原有换行。"
        "没有可识别文字就输出空字符串。"
    )
    TEXT_EXPANSION_RATIO = 0.05
    TOP_K = DEFAULT_HUNYUAN_TOP_K
    REPETITION_PENALTY = DEFAULT_HUNYUAN_REPETITION_PENALTY
    DESCRIPTIVE_PREFIX_PATTERNS = (
        r"^\s*(?:the\s+image|this\s+image|image\s+shows|the\s+text\s+in\s+the\s+image|text\s+in\s+the\s+image)\b",
        r"^\s*(?:图片|圖片|图像|圖像|图中|圖中|画面中|畫面中|这张图片|這張圖片)",
        r"^\s*(?:从图中|從圖中|从图片|從圖片|图中可以看到|圖中可以看到|可以看到|可以看出)",
        r"^\s*(?:この画像|この図|この漫畫|このイラスト|画像の|図の)",
        r"^\s*(?:要理解|下面是|以下是|内容如下|內容如下)",
        r"^\s*#{1,6}\s",
    )
    DESCRIPTIVE_PHRASES = (
        "图片中的文字",
        "圖片中的文字",
        "图像中的文字",
        "圖像中的文字",
        "图中是",
        "圖中是",
        "从图中",
        "從圖中",
        "图中可以看到",
        "圖中可以看到",
        "文字是",
        "这张图片",
        "這張圖片",
        "画面中",
        "畫面中",
        "この画像",
        "この図",
        "the text in the image",
        "text in the image",
        "the image shows",
        "this image",
        "to understand this",
    )
    DESCRIPTIVE_WRAPPER_PATTERNS = (
        r"^\s*(?:the\s+text\s+in\s+the\s+image|text\s+in\s+the\s+image)(?:\s+is)?\s*[:：]\s*",
        r"^\s*(?:图片中的文字|圖片中的文字|图像中的文字|圖像中的文字)(?:是|為)?\s*[:：]?\s*",
        r"^\s*(?:图像中的文字如下|圖片中的文字如下|图像中的文字是|圖片中的文字是)\s*[:：]?\s*",
        r"^\s*(?:从图中可以看到|從圖中可以看到|图中可以看到|圖中可以看到)(?:[，,]\s*)?(?:文字是|內容是)?\s*[:：]?\s*",
        r"^\s*(?:可以看到|可以看出)(?:[，,]\s*)?(?:文字是|內容是)?\s*[:：]?\s*",
        r"^\s*(?:画像内の文字|画像の文字|図中の文字)(?:は|：|:)?\s*",
    )
    PROMPT_ECHO_PHRASES = (
        "提取图中的文字",
        "只返回图片中的原文文字本身",
        "不要描述图片",
        "保持原有换行",
        "没有可识别文字就输出空字符串",
    )
    STRUCTURED_OUTPUT_PATTERNS = (
        r"^\s*```",
        r"^\s*[\{\[][\s\S]*[\}\]]\s*$",
        r"^\s*<\s*(?:table|tr|td|th|div|span|html|body)\b",
        r"^\s*\|.+\|\s*$",
    )
    STRUCTURED_OUTPUT_PHRASES = (
        '"text"',
        '"bbox"',
        '"box"',
        '"points"',
        '"coord"',
        '"coords"',
        '"x1"',
        '"y1"',
        '"x2"',
        '"y2"',
        '"markdown"',
        '"html"',
    )
    IGNORED_QUOTED_TEXTS = {
        "text",
        "bbox",
        "box",
        "points",
        "coord",
        "coords",
        "markdown",
        "html",
        "latex",
        "x1",
        "y1",
        "x2",
        "y2",
    }
    QUOTED_TEXT_PATTERN = re.compile(r"[「『“\"']([^「」『』“”\"'\n]{1,200})[」』”\"']")

    def __init__(self) -> None:
        self.server_url = DEFAULT_HUNYUAN_SERVER_URL
        self.max_completion_tokens = DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS
        self.parallel_workers = DEFAULT_HUNYUAN_PARALLEL_WORKERS
        self.request_timeout_sec = DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC
        self.raw_response_logging = False

    def initialize(self, settings, **kwargs) -> None:
        config = settings.get_hunyuan_ocr_settings()
        self.server_url = config.get("server_url", DEFAULT_HUNYUAN_SERVER_URL) or DEFAULT_HUNYUAN_SERVER_URL
        self.max_completion_tokens = self._clamp_int(
            config.get("max_completion_tokens", DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS),
            DEFAULT_HUNYUAN_MAX_COMPLETION_TOKENS,
            self.MAX_COMPLETION_TOKENS_RANGE,
        )
        self.parallel_workers = self._clamp_int(
            config.get("parallel_workers", DEFAULT_HUNYUAN_PARALLEL_WORKERS),
            DEFAULT_HUNYUAN_PARALLEL_WORKERS,
            self.PARALLEL_WORKERS_RANGE,
        )
        self.request_timeout_sec = max(
            1,
            self._clamp_int(
                config.get("request_timeout_sec", DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC),
                DEFAULT_HUNYUAN_REQUEST_TIMEOUT_SEC,
                (1, 3600),
            ),
        )
        self.raw_response_logging = bool(config.get("raw_response_logging", False))

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
            "hunyuan_ocr start: blocks=%d workers=%d max_completion_tokens=%d endpoint=%s",
            len(jobs),
            worker_count,
            self.max_completion_tokens,
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
            "hunyuan_ocr complete: blocks=%d elapsed_ms=%.1f",
            len(jobs),
            (time.perf_counter() - started_at) * 1000.0,
        )
        return blk_list

    def _process_block(self, img: np.ndarray, blk: TextBlock, bbox: tuple[int, int, int, int]) -> None:
        crop = self._crop_image(img, bbox)
        if crop is None:
            self._mark_empty(blk, "Invalid OCR crop bounds.")
            return

        text, descriptive_filtered, request_attempt_count = self._request_ocr_text(crop)
        cleaned = text
        if cleaned:
            set_block_ocr_diagnostics(
                blk,
                text=cleaned,
                confidence=0.0,
                status=OCR_STATUS_OK,
                empty_reason="",
                attempt_count=request_attempt_count,
            )
        else:
            if descriptive_filtered:
                self._mark_empty(
                    blk,
                    "HunyuanOCR returned non-OCR output instead of raw OCR text.",
                    attempt_count=request_attempt_count,
                )
            else:
                self._mark_empty(
                    blk,
                    "HunyuanOCR returned no text.",
                    attempt_count=request_attempt_count,
                )

    def _request_ocr_text(self, image: np.ndarray) -> tuple[str, bool, int]:
        first_text = self._request_ocr_text_for_prompt(image, self.PROMPT)
        first_prepared = self._prepare_output_text(first_text)
        if not self._looks_bad_output(first_prepared):
            return first_prepared, False, 1

        logger.warning("hunyuan_ocr non-ocr response detected; retrying with strict OCR prompt")

        retry_text = self._request_ocr_text_for_prompt(image, self.STRICT_PROMPT)
        retry_prepared = self._prepare_output_text(retry_text)
        if not self._looks_bad_output(retry_prepared):
            return retry_prepared, True, 2

        salvaged = self._salvage_text_from_descriptive_output(retry_prepared)
        if not salvaged:
            salvaged = self._salvage_text_from_descriptive_output(first_prepared)
        if salvaged and not self._looks_bad_output(salvaged):
            logger.warning("hunyuan_ocr salvaged OCR text from non-ocr response")
            return salvaged, True, 2

        return "", True, 2

    def _request_ocr_text_for_prompt(self, image: np.ndarray, prompt: str) -> str:
        data_url = self._image_data_url(image)
        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_completion_tokens": self.max_completion_tokens,
            "top_k": self.TOP_K,
            "repetition_penalty": self.REPETITION_PENALTY,
        }

        data = self._send_request(payload)
        return self._extract_text_from_response(data)

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
            logger.info("hunyuan_ocr raw response json: %s", data)
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

        choice = choices[0] or {}
        message = choice.get("message") or {}
        content_text = self._extract_content_text(message.get("content"))

        if self.raw_response_logging and content_text:
            logger.info("hunyuan_ocr raw content: %s", content_text)

        return content_text

    def _prepare_output_text(self, text: str) -> str:
        return self._normalize_output_text(self._clean_repeated_substrings(text))

    def _looks_descriptive(self, text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        for pattern in self.DESCRIPTIVE_PREFIX_PATTERNS:
            if re.match(pattern, lowered, flags=re.IGNORECASE):
                return True

        lowered_lines = [line.strip().lower() for line in stripped.splitlines() if line.strip()]
        if any(
            line.startswith(("-", "*", "###", "1.", "2.", "3."))
            and any(phrase in line for phrase in self.DESCRIPTIVE_PHRASES)
            for line in lowered_lines
        ):
            return True

        return any(phrase in lowered for phrase in self.DESCRIPTIVE_PHRASES)

    def _looks_prompt_echo(self, text: str) -> bool:
        if not text:
            return False
        lowered = text.strip().lower()
        return any(phrase in lowered for phrase in self.PROMPT_ECHO_PHRASES)

    def _looks_structured_output(self, text: str) -> bool:
        if not text:
            return False

        stripped = text.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        if any(re.match(pattern, stripped, flags=re.IGNORECASE) for pattern in self.STRUCTURED_OUTPUT_PATTERNS):
            return True

        if any(phrase in lowered for phrase in self.STRUCTURED_OUTPUT_PHRASES):
            return True

        coordinate_markers = ("bbox", "box", "coord", "coords", "points", "x1", "y1", "x2", "y2")
        if any(marker in lowered for marker in coordinate_markers):
            digit_count = sum(char.isdigit() for char in stripped)
            punctuation_count = sum(stripped.count(char) for char in "[]{}(),:")
            if digit_count >= 4 and punctuation_count >= 4:
                return True

        return False

    def _looks_bad_output(self, text: str) -> bool:
        return (
            self._looks_descriptive(text)
            or self._looks_prompt_echo(text)
            or self._looks_structured_output(text)
        )

    def _strip_descriptive_wrappers(self, text: str) -> str:
        stripped = text.strip()
        if not stripped:
            return ""

        updated = stripped
        changed = True
        while changed and updated:
            changed = False
            for pattern in self.DESCRIPTIVE_WRAPPER_PATTERNS:
                new_value, count = re.subn(pattern, "", updated, count=1, flags=re.IGNORECASE)
                if count:
                    updated = new_value.strip()
                    changed = True
        return self._prepare_output_text(updated)

    def _extract_quoted_text(self, text: str) -> str:
        matches = self.QUOTED_TEXT_PATTERN.findall(text or "")
        cleaned_parts: list[str] = []
        seen: set[str] = set()
        for match in matches:
            cleaned = self._prepare_output_text(match)
            lowered = cleaned.lower()
            if lowered in self.IGNORED_QUOTED_TEXTS:
                continue
            if cleaned.isdigit():
                continue
            if not cleaned or cleaned in seen:
                continue
            seen.add(cleaned)
            cleaned_parts.append(cleaned)
        return "\n".join(cleaned_parts).strip()

    def _salvage_text_from_descriptive_output(self, text: str) -> str:
        stripped = self._strip_descriptive_wrappers(text)
        if stripped and not self._looks_bad_output(stripped):
            return stripped

        if self._looks_structured_output(text):
            return ""

        quoted = self._extract_quoted_text(text)
        if quoted and not self._looks_bad_output(quoted):
            return quoted

        return ""

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
                    continue
                if item.get("type") == "text":
                    item_text = item.get("text")
                    if isinstance(item_text, str) and item_text.strip():
                        parts.append(item_text)
            return "\n".join(parts)

        return ""

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
        base = (self.server_url or DEFAULT_HUNYUAN_SERVER_URL).rstrip("/")
        if base.endswith(self.REQUEST_ENDPOINT_SUFFIX):
            return base
        return f"{base}{self.REQUEST_ENDPOINT_SUFFIX}"

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

    @staticmethod
    def _clean_repeated_substrings(text: str) -> str:
        total_length = len(text or "")
        if total_length < 8000:
            return text

        for length in range(2, total_length // 10 + 1):
            candidate = text[-length:]
            count = 0
            index = total_length - length

            while index >= 0 and text[index:index + length] == candidate:
                count += 1
                index -= length

            if count >= 10:
                return text[: total_length - length * (count - 1)]

        return text

    @staticmethod
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))
