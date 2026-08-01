"""Wire contracts for Paddle crop OCR transports.

The bundled runtime uses the official PaddleOCR-VL GGUF ``OCR:`` prompt
directly through llama.cpp.  Custom endpoints that still expose PaddleX's
``/layout-parsing`` contract remain supported by the engine as unmanaged
compatibility endpoints.
"""

from __future__ import annotations

import base64
import re
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np

from modules.utils.exceptions import LocalServiceResponseError

from .response_parser import normalize_output_text


DEFAULT_PADDLE_DIRECT_SERVER_URL = (
    "http://127.0.0.1:18000/v1/chat/completions"
)
PADDLE_DIRECT_MODEL_ALIAS = "PaddleOCR-VL-1.6-0.9B"
PADDLE_DIRECT_PROMPT = "OCR:"
PADDLE_DIRECT_TRANSPORT_SCHEMA_VERSION = 1
PADDLE_DIRECT_PARSER_SCHEMA_VERSION = "paddle_llamacpp_chat_response_v1"


def is_direct_llama_cpp_endpoint(server_url: str) -> bool:
    """Return whether *server_url* exposes llama.cpp chat completions."""

    path = urlparse(str(server_url or "").strip()).path.rstrip("/").lower()
    return path.endswith("/v1/chat/completions")


def direct_transport_identity() -> dict[str, Any]:
    """Return the stable request/response contract used in fingerprints."""

    return {
        "schema_version": PADDLE_DIRECT_TRANSPORT_SCHEMA_VERSION,
        "api_path": "/v1/chat/completions",
        "model_alias": PADDLE_DIRECT_MODEL_ALIAS,
        "prompt": PADDLE_DIRECT_PROMPT,
        "image_media_type": "image/png",
        "content_order": ["image_url", "text"],
        "response_schema": "openai_chat_completions",
        "normalizer": "paddlex_relay_compatible_v1",
    }


def encoded_product_jpeg_to_png(image_bytes: bytes) -> bytes:
    """Re-encode the exact product JPEG crop as the official PNG input.

    The JPEG decode is intentional.  It preserves the already validated
    relay-to-direct image contract instead of silently changing model input.
    """

    decoded = cv2.imdecode(
        np.frombuffer(bytes(image_bytes), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    if decoded is None:
        raise LocalServiceResponseError(
            "Unable to decode the product JPEG for direct Paddle OCR.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    encoded, png = cv2.imencode(".png", decoded)
    if not encoded:
        raise LocalServiceResponseError(
            "Unable to encode the official direct Paddle PNG.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    return bytes(png.tobytes())


def build_direct_ocr_payload(
    image_png: bytes,
    *,
    max_tokens: int,
) -> tuple[dict[str, Any], int]:
    """Build the official image-first ``OCR:`` llama.cpp request."""

    image_b64 = base64.b64encode(bytes(image_png)).decode("ascii")
    return (
        {
            "model": PADDLE_DIRECT_MODEL_ALIAS,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64," + image_b64
                            },
                        },
                        {"type": "text", "text": PADDLE_DIRECT_PROMPT},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": int(max_tokens),
            "stream": False,
        },
        len(image_b64),
    )


def extract_direct_ocr_text(payload: Any) -> str:
    """Parse and normalize one non-streaming llama.cpp chat response."""

    if not isinstance(payload, dict):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response must be an object.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response did not include choices.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response choice is invalid.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
    if finish_reason == "length":
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp OCR response was truncated.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    message = choice.get("message")
    if not isinstance(message, dict):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response has no message payload.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )
    content = message.get("content", "")
    if isinstance(content, list):
        content = "\n".join(
            str(item.get("text", "") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    elif not isinstance(content, str):
        raise LocalServiceResponseError(
            "Direct Paddle llama.cpp response content has an invalid type.",
            service_name="PaddleOCR VL",
            settings_page_name="PaddleOCR VL Settings",
        )

    # PaddleX converted single newlines into separate markdown paragraphs.
    # Reproduce that behavior before the shared product normalizer so the
    # direct route stays byte-for-byte compatible with the validated relay.
    paragraph_lines = re.sub(r"(?<!\n)\n(?!\n)", "\n\n", content.strip())
    return normalize_output_text(paragraph_lines)


__all__ = [
    "DEFAULT_PADDLE_DIRECT_SERVER_URL",
    "PADDLE_DIRECT_MODEL_ALIAS",
    "PADDLE_DIRECT_PARSER_SCHEMA_VERSION",
    "PADDLE_DIRECT_PROMPT",
    "PADDLE_DIRECT_TRANSPORT_SCHEMA_VERSION",
    "build_direct_ocr_payload",
    "direct_transport_identity",
    "encoded_product_jpeg_to_png",
    "extract_direct_ocr_text",
    "is_direct_llama_cpp_endpoint",
]
