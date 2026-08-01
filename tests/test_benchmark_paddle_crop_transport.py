from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import benchmark_paddle_crop_transport as transport
from modules.utils.exceptions import LocalServiceResponseError


class PaddleCropTransportBenchmarkTests(unittest.TestCase):
    def test_direct_payload_uses_official_crop_contract(self) -> None:
        payload = transport._direct_payload("abc", max_tokens=768)

        self.assertEqual(payload["model"], "PaddleOCR-VL-1.6-0.9B")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["max_tokens"], 768)
        self.assertFalse(payload["stream"])
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "OCR:"})
        self.assertEqual(
            content[1],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,abc"},
            },
        )
        self.assertNotIn("special", payload)

    def test_extract_chat_content_accepts_text_parts(self) -> None:
        text, finish = transport._extract_chat_content(
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"type": "text", "text": "第一行"},
                                {"type": "text", "text": "第二行"},
                            ]
                        },
                    }
                ]
            }
        )

        self.assertEqual(text, "第一行\n第二行")
        self.assertEqual(finish, "stop")

    def test_extract_chat_content_rejects_missing_choices(self) -> None:
        with self.assertRaises(LocalServiceResponseError):
            transport._extract_chat_content({})

    def test_unknown_transport_is_rejected_before_start(self) -> None:
        with self.assertRaises(transport.TransportContractError):
            transport._start_runtime("vllm", timeout_sec=1)

    def test_selected_result_pages_reads_locked_subset(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps({"page_ids": ["p_017", "p_016"]}),
                encoding="utf-8",
            )

            self.assertEqual(
                transport._selected_result_pages(root),
                {"p_016", "p_017"},
            )


if __name__ == "__main__":
    unittest.main()
