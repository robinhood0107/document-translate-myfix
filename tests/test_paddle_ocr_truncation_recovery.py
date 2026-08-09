from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.ocr.paddle_crop.engine import PaddleOCRVLEngine  # noqa: E402
from modules.ocr.paddle_crop.transport import (  # noqa: E402
    PaddleDirectOcrTruncatedError,
    extract_direct_ocr_text,
)
from modules.utils.exceptions import LocalServiceResponseError  # noqa: E402


def _response(text: str, finish_reason: str) -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ]
    }


class TruncationSignalTests(unittest.TestCase):
    def test_a_truncated_response_raises_the_dedicated_type(self) -> None:
        with self.assertRaises(PaddleDirectOcrTruncatedError):
            extract_direct_ocr_text(_response("half a bub", "length"))

    def test_the_dedicated_type_is_still_a_response_error(self) -> None:
        # 기존에 이 오류를 잡던 코드가 계속 동작해야 한다.
        self.assertTrue(
            issubclass(PaddleDirectOcrTruncatedError, LocalServiceResponseError)
        )

    def test_a_complete_response_is_unaffected(self) -> None:
        self.assertEqual(
            extract_direct_ocr_text(_response("hello", "stop")),
            "hello",
        )


class _StubEngine:
    """`_request_direct_ocr_text_from_encoded` 만 실제 구현으로 돌린다."""

    TRUNCATION_RETRY_MULTIPLIER = PaddleOCRVLEngine.TRUNCATION_RETRY_MULTIPLIER
    TRUNCATION_RETRY_MAX_TOKENS = PaddleOCRVLEngine.TRUNCATION_RETRY_MAX_TOKENS

    _request_direct_ocr_text_from_encoded = (
        PaddleOCRVLEngine._request_direct_ocr_text_from_encoded
    )

    def __init__(self, responses: list[dict]) -> None:
        self.max_new_tokens = 1024
        self._responses = list(responses)
        self.requested_max_tokens: list[int] = []
        self.record: dict = {}

    def _current_request_record(self) -> dict:
        return self.record

    def _add_request_metric(self, record, name, value) -> None:
        record[name] = record.get(name, 0) + value

    def _send_request(self, payload, *, telemetry_record=None) -> dict:
        self.requested_max_tokens.append(int(payload["max_tokens"]))
        return self._responses.pop(0)


class TruncationRetryTests(unittest.TestCase):
    def setUp(self) -> None:
        # 실제 인코딩을 타지 않도록 최소 PNG 바이트를 대신 쓴다.
        self._patched = []

    def _run(self, responses: list[dict]) -> _StubEngine:
        import modules.ocr.paddle_crop.engine as engine_module

        original = engine_module.encoded_product_jpeg_to_png
        engine_module.encoded_product_jpeg_to_png = lambda data: b"png-bytes"
        try:
            engine = _StubEngine(responses)
            engine.result = engine._request_direct_ocr_text_from_encoded(b"jpeg")
            return engine
        finally:
            engine_module.encoded_product_jpeg_to_png = original

    def test_a_truncated_first_attempt_retries_with_a_larger_budget(self) -> None:
        engine = self._run(
            [_response("cut off", "length"), _response("full text", "stop")]
        )

        self.assertEqual(engine.result, "full text")
        self.assertEqual(engine.requested_max_tokens, [1024, 3072])
        self.assertEqual(engine.record.get("truncation_retry_count"), 1)

    def test_a_complete_first_attempt_does_not_retry(self) -> None:
        engine = self._run([_response("all of it", "stop")])

        self.assertEqual(engine.result, "all of it")
        self.assertEqual(engine.requested_max_tokens, [1024])
        self.assertNotIn("truncation_retry_count", engine.record)

    def test_the_retry_budget_is_capped(self) -> None:
        import modules.ocr.paddle_crop.engine as engine_module

        original = engine_module.encoded_product_jpeg_to_png
        engine_module.encoded_product_jpeg_to_png = lambda data: b"png-bytes"
        try:
            engine = _StubEngine(
                [_response("cut", "length"), _response("done", "stop")]
            )
            engine.max_new_tokens = 4096
            with self.assertRaises(PaddleDirectOcrTruncatedError):
                engine._request_direct_ocr_text_from_encoded(b"jpeg")
        finally:
            engine_module.encoded_product_jpeg_to_png = original

        # 3배는 12288 이지만 상한이 4096 이다. 이미 상한이면 같은 한도로 다시
        # 물어봐야 소용이 없으므로 재시도하지 않는다.
        self.assertEqual(engine.requested_max_tokens, [4096])

    def test_a_still_truncated_retry_raises_the_dedicated_type(self) -> None:
        with self.assertRaises(PaddleDirectOcrTruncatedError):
            self._run([_response("cut", "length"), _response("cut", "length")])


class BlockScopeTests(unittest.TestCase):
    def test_a_truncated_block_is_emptied_instead_of_failing_the_page(self) -> None:
        # 말풍선 하나의 잘림이 페이지 전체를 출력에서 지우던 동작을 막는다.
        import inspect

        source = inspect.getsource(PaddleOCRVLEngine._process_job)
        self.assertIn("except PaddleDirectOcrTruncatedError:", source)
        handler = source.index("except PaddleDirectOcrTruncatedError:")
        marked = source.index("TRUNCATED_OCR_REASON", handler)
        returned = source.index("return", marked)
        self.assertLess(handler, marked)
        self.assertLess(marked, returned)


if __name__ == "__main__":
    unittest.main()
