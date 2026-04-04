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
from modules.utils.ocr_debug import OCR_STATUS_EMPTY_INITIAL, OCR_STATUS_OK, ensure_three_channel, expand_bbox, set_block_ocr_diagnostics
from modules.utils.textblock import TextBlock

from .base import OCREngine


logger = logging.getLogger(__name__)


class PaddleOCRVLEngine(OCREngine):
    DEFAULT_SERVER_URL = "http://127.0.0.1:28118/layout-parsing"
    DEFAULT_MAX_NEW_TOKENS = 1024
    DEFAULT_PARALLEL_WORKERS = 8
    MAX_NEW_TOKENS_RANGE = (128, 2048)
    PARALLEL_WORKERS_RANGE = (1, 8)
    REQUEST_TIMEOUT_SECONDS = 60
    TEXT_EXPANSION_RATIO = 0.05

    def __init__(self) -> None:
        self.server_url = self.DEFAULT_SERVER_URL
        self.prettify_markdown = False
        self.visualize = False
        self.max_new_tokens = self.DEFAULT_MAX_NEW_TOKENS
        self.parallel_workers = self.DEFAULT_PARALLEL_WORKERS
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
        self._supports_max_new_tokens = True

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
            "paddleocr_vl start: blocks=%d workers=%d max_new_tokens=%d endpoint=%s",
            len(jobs),
            worker_count,
            self.max_new_tokens,
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

        text = self._request_ocr_text(crop)
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
            raise LocalServiceResponseError(error_msg)

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
                "Unable to reach the local PaddleOCR VL service."
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
                        "Unable to reach the local PaddleOCR VL service."
                    ) from exc
                if legacy_response.status_code == 200:
                    response = legacy_response
                    self._supports_max_new_tokens = False

        if response.status_code != 200:
            raise LocalServiceResponseError(
                f"PaddleOCR VL service returned HTTP {response.status_code}."
            )

        try:
            return response.json()
        except ValueError as exc:
            raise LocalServiceResponseError(
                "PaddleOCR VL service returned invalid JSON."
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
            raise LocalServiceResponseError("Failed to encode OCR crop for PaddleOCR VL.")
        return encoded.tobytes()

    def _mark_empty(self, blk: TextBlock, reason: str) -> None:
        set_block_ocr_diagnostics(
            blk,
            text="",
            confidence=0.0,
            status=OCR_STATUS_EMPTY_INITIAL,
            empty_reason=reason,
            attempt_count=1,
        )

    def _response_rejected_max_new_tokens(self, data: dict) -> bool:
        if not isinstance(data, dict):
            return False
        error_msg = str(data.get("errorMsg", "") or "")
        return "maxNewTokens" in error_msg

    def _normalize_output_text(self, text: str) -> str:
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
    def _clamp_int(value, default: int, bounds: tuple[int, int]) -> int:
        low, high = bounds
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = default
        return max(low, min(parsed, high))
