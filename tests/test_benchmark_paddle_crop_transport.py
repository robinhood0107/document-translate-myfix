from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

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
        self.assertEqual(
            content[0],
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,abc"},
            },
        )
        self.assertEqual(content[1], {"type": "text", "text": "OCR:"})
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

    def test_relay_compatible_text_preserves_paragraph_spacing(self) -> None:
        self.assertEqual(
            transport._relay_compatible_text("first\nsecond\n\nthird....."),
            "first\n\nsecond\n\nthird...",
        )

    def test_unknown_transport_is_rejected_before_start(self) -> None:
        with self.assertRaises(transport.TransportContractError):
            transport._start_runtime("vllm", timeout_sec=1)

    def test_runtime_snapshot_rejects_changed_image(self) -> None:
        snapshot = {
            "containers": [
                {
                    "name": "paddleocr-llamacpp",
                    "image_id": "sha256:changed",
                    "state": "running",
                    "runtime_labels": {},
                    "command": [],
                    "mounts": [],
                }
            ]
        }

        with self.assertRaises(transport.TransportContractError):
            transport._validate_runtime_snapshot(snapshot, transport="direct")

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

    def test_empty_page_filter_means_full_corpus(self) -> None:
        pages = [{"page_id": "094"}, {"page_id": "095"}]

        self.assertEqual(
            transport._select_manifest_pages(pages, []),
            pages,
        )

    def test_legacy_summary_infers_result_page_directories(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "summary.json").write_text(
                json.dumps({"schema_version": 1}),
                encoding="utf-8",
            )
            for page_id in ("094", "095"):
                page = root / page_id
                page.mkdir()
                (page / "result.json").write_text("{}", encoding="utf-8")

            self.assertEqual(
                transport._selected_result_pages(root),
                {"094", "095"},
            )

    def test_improvement_percent(self) -> None:
        self.assertAlmostEqual(
            transport._improvement_percent(100.0, 25.0),
            75.0,
        )
        self.assertIsNone(transport._improvement_percent(0.0, 0.0))

    def test_build_manifest_from_page_snapshots(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "page.png"
            self.assertTrue(
                cv2.imwrite(str(source), np.zeros((40, 60, 3), dtype=np.uint8))
            )
            snapshots = root / "page_snapshots.json"
            snapshots.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "image_path": str(source),
                                "image_stem": "page",
                                "blocks": [
                                    {
                                        "xyxy": [5, 6, 30, 20],
                                        "bubble_xyxy": [2, 3, 40, 30],
                                        "text_class": "text_bubble",
                                    }
                                ],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = transport.build_manifest_from_snapshots(
                snapshots_path=snapshots,
                output_dir=root / "manifest",
                suite_id="test-suite",
                language="en",
                split="development",
            )

            self.assertEqual(result["page_count"], 1)
            self.assertEqual(result["block_count"], 1)
            manifest = json.loads(
                (root / "manifest" / "corpus-manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["pages"][0]["language"], "en")

    def test_compare_exact_transports_accepts_identical_blocks(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            block = {
                "block_id": "detector-0000",
                "text": "same",
                "ocr_status": "ok",
            }
            for name, wall in (("baseline", 10.0), ("candidate", 5.0)):
                run = root / name
                page = run / "page"
                page.mkdir(parents=True)
                (run / "summary.json").write_text(
                    json.dumps(
                        {
                            "page_ids": ["page"],
                            "wall_seconds": wall,
                        }
                    ),
                    encoding="utf-8",
                )
                (page / "result.json").write_text(
                    json.dumps({"blocks": [block]}),
                    encoding="utf-8",
                )

            result = transport.compare_exact_transports(
                baseline_dir=root / "baseline",
                candidate_dir=root / "candidate",
                output_dir=root / "comparison",
            )

            self.assertEqual(result["block_contract_changed_count"], 0)
            self.assertEqual(result["normalized_text_changed_count"], 0)
            self.assertEqual(result["improvement_percent"]["wall_seconds"], 50.0)


if __name__ == "__main__":
    unittest.main()
