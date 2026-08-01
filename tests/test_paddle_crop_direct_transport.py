from __future__ import annotations

import base64
import unittest
from unittest import mock

import cv2
import numpy as np

from modules.ocr.paddle_crop.transport import (
    DEFAULT_PADDLE_DIRECT_SERVER_URL,
    PADDLE_DIRECT_MODEL_ALIAS,
    build_direct_ocr_payload,
    direct_transport_identity,
    encoded_product_jpeg_to_png,
    extract_direct_ocr_text,
    is_direct_llama_cpp_endpoint,
)
from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine
from modules.utils.exceptions import LocalServiceResponseError


class PaddleCropDirectTransportTests(unittest.TestCase):
    def test_managed_endpoint_is_direct_chat_completions(self) -> None:
        self.assertTrue(
            is_direct_llama_cpp_endpoint(DEFAULT_PADDLE_DIRECT_SERVER_URL)
        )
        self.assertFalse(
            is_direct_llama_cpp_endpoint(
                "http://127.0.0.1:28118/layout-parsing"
            )
        )

    def test_product_jpeg_is_reencoded_as_png(self) -> None:
        source = np.full((32, 64, 3), 255, dtype=np.uint8)
        cv2.putText(
            source,
            "OCR",
            (2, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        ok, jpeg = cv2.imencode(".jpg", source)
        self.assertTrue(ok)

        png = encoded_product_jpeg_to_png(jpeg.tobytes())

        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        decoded = cv2.imdecode(
            np.frombuffer(png, dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        self.assertEqual(decoded.shape, source.shape)

    def test_payload_uses_official_image_first_ocr_contract(self) -> None:
        payload, base64_chars = build_direct_ocr_payload(
            b"png-bytes",
            max_tokens=768,
        )

        self.assertEqual(payload["model"], PADDLE_DIRECT_MODEL_ALIAS)
        self.assertEqual(payload["max_tokens"], 768)
        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["image_url", "text"])
        self.assertEqual(content[1]["text"], "OCR:")
        encoded = content[0]["image_url"]["url"].split(",", 1)[1]
        self.assertEqual(base64.b64decode(encoded), b"png-bytes")
        self.assertEqual(base64_chars, len(encoded))

    def test_response_matches_validated_relay_normalization(self) -> None:
        result = extract_direct_ocr_text(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": "一行目\n二行目"},
                    }
                ]
            }
        )

        self.assertEqual(result, "一行目\n\n二行目")

    def test_truncated_response_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            LocalServiceResponseError,
            "truncated",
        ):
            extract_direct_ocr_text(
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "partial"},
                        }
                    ]
                }
            )

    def test_identity_includes_prompt_and_media_contract(self) -> None:
        identity = direct_transport_identity()

        self.assertEqual(identity["prompt"], "OCR:")
        self.assertEqual(identity["image_media_type"], "image/png")
        self.assertEqual(identity["content_order"], ["image_url", "text"])

    def test_engine_default_routes_encoded_crop_to_direct_contract(self) -> None:
        source = np.full((24, 48, 3), 255, dtype=np.uint8)
        ok, jpeg = cv2.imencode(".jpg", source)
        self.assertTrue(ok)
        engine = PaddleOCRVLEngine()
        response = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": "テスト"},
                }
            ]
        }

        with mock.patch.object(
            engine,
            "_send_request",
            return_value=response,
        ) as send:
            result = engine._request_ocr_text_from_encoded(jpeg.tobytes())

        self.assertEqual(result, "テスト")
        payload = send.call_args.args[0]
        content = payload["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text"])
        self.assertEqual(content[1]["text"], "OCR:")


if __name__ == "__main__":
    unittest.main()
