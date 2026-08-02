from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_pipeline  # noqa: E402


class BenchmarkPipelineTurbo4LedgerTests(unittest.TestCase):
    def test_private_ocr_contract_uses_block_order_not_generated_ids(self) -> None:
        first_blocks = [
            SimpleNamespace(block_id="first-a"),
            SimpleNamespace(block_id="first-b"),
        ]
        second_blocks = [
            SimpleNamespace(block_id="second-a"),
            SimpleNamespace(block_id="second-b"),
        ]
        first = benchmark_pipeline._canonical_ocr_raw_results(
            first_blocks,
            {
                "first-a": {"text": "alpha", "status": "ok"},
                "first-b": {"text": "beta", "status": "ok"},
            },
        )
        second = benchmark_pipeline._canonical_ocr_raw_results(
            second_blocks,
            {
                "second-a": {"text": "alpha", "status": "ok"},
                "second-b": {"text": "beta", "status": "ok"},
            },
        )

        self.assertEqual(first, second)
        self.assertEqual(
            benchmark_pipeline._snapshot_contract_sha256(first),
            benchmark_pipeline._snapshot_contract_sha256(second),
        )

    def test_http_experiment_records_the_sent_seeded_payload(self) -> None:
        from modules.translation.llm import custom_local_gemma as gemma_module

        original_gemma = gemma_module.requests
        response = SimpleNamespace(
            status_code=200,
            text="",
            json=lambda: {"choices": [{"message": {"content": "ok"}}]},
        )
        payload = {"model": "test"}
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            record_path = output_dir / "gemma-records.jsonl"
            with mock.patch.object(
                original_gemma,
                "post",
                return_value=response,
            ) as post, mock.patch.dict(
                os.environ,
                {"CT_BENCH_OUTPUT_DIR": str(output_dir)},
                clear=False,
            ):
                with benchmark_pipeline._benchmark_http_clients(
                    {
                        "gemma_seed": 20260801,
                        "gemma_request_record_path": str(record_path),
                    }
                ):
                    gemma_module.requests.post(
                        "http://127.0.0.1:18080/v1/chat/completions",
                        json=payload,
                    )
            rows = [
                json.loads(line)
                for line in record_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(post.call_args.kwargs["json"]["seed"], 20260801)
        self.assertEqual(rows[0]["request"]["seed"], 20260801)
        self.assertEqual(rows[0]["attempt_index"], 0)
        self.assertEqual(rows[0]["status_code"], 200)
        self.assertEqual(
            rows[1],
            {"attempt_count": 1, "record_type": "summary", "write_error": ""},
        )
        self.assertNotIn("seed", payload)
        self.assertIs(gemma_module.requests, original_gemma)
