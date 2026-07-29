from __future__ import annotations

import unittest
from unittest import mock

from modules.ocr.hunyuan_ocr import HunyuanOCREngine
from modules.ocr.mangalmm_ocr import MangaLMMOCREngine
from modules.translation.llm.custom_local_gemma import (
    CustomLocalGemmaTranslation,
)


class _Response:
    status_code = 200
    text = ""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        return None


class RawResponseDebugRoutingTests(unittest.TestCase):
    def test_gemma_raw_response_uses_active_debug_sink(self) -> None:
        payload = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"translation":"ok"}'},
                }
            ],
            "usage": {},
        }
        engine = CustomLocalGemmaTranslation()
        engine.raw_response_logging = True
        with mock.patch(
            "modules.translation.llm.custom_local_gemma.requests.post",
            return_value=_Response(payload),
        ), mock.patch(
            "modules.translation.llm.custom_local_gemma."
            "append_active_raw_response",
            return_value=True,
        ) as append:
            result = engine._request_translation("system", "user")

        self.assertEqual(result, payload)
        append.assert_called_once_with(
            "gemma",
            payload,
            kind="response_json",
        )

    def test_hunyuan_raw_response_uses_active_debug_sink(self) -> None:
        payload = {
            "choices": [{"message": {"content": "ok"}}],
        }
        engine = HunyuanOCREngine()
        engine.raw_response_logging = True
        with mock.patch(
            "modules.ocr.hunyuan_ocr.requests.post",
            return_value=_Response(payload),
        ), mock.patch(
            "modules.ocr.hunyuan_ocr.append_active_raw_response",
            return_value=True,
        ) as append:
            result = engine._send_request({"messages": []})

        self.assertEqual(result, payload)
        append.assert_called_once_with(
            "hunyuanocr",
            payload,
            kind="response_json",
        )

    def test_mangalmm_raw_response_uses_active_debug_sink(self) -> None:
        payload = {
            "choices": [{"message": {"content": "[]"}}],
        }
        engine = MangaLMMOCREngine()
        engine.raw_response_logging = True
        with mock.patch(
            "modules.ocr.mangalmm_ocr.requests.post",
            return_value=_Response(payload),
        ), mock.patch(
            "modules.ocr.mangalmm_ocr.append_active_raw_response",
            return_value=True,
        ) as append:
            result = engine._send_request({"messages": []})

        self.assertEqual(result, payload)
        append.assert_called_once_with(
            "mangalmm",
            payload,
            kind="response_json",
        )


if __name__ == "__main__":
    unittest.main()
