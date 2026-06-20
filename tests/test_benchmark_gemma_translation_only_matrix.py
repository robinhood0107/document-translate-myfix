from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_gemma_translation_only_matrix as matrix  # noqa: E402


class GemmaTranslationOnlyMatrixTests(unittest.TestCase):
    def test_coverage_reports_low_context_overflow_without_raw_text(self) -> None:
        chunks = [
            {"id": "short", "texts": ["Short line."], "total_chars": 11, "max_block_chars": 11},
            {"id": "long", "texts": ["A" * 2400], "total_chars": 2400, "max_block_chars": 2400},
        ]

        report = matrix.coverage_report(chunks, max_completion_tokens=512)
        payload = json.dumps(report, ensure_ascii=False)

        self.assertGreater(report["4096"]["direct_percent"], report["768"]["direct_percent"])
        self.assertNotIn("Short line.", payload)
        self.assertNotIn("A" * 100, payload)

    def test_rank_translation_profiles_ignores_low_gpu_free_memory(self) -> None:
        ranked = matrix.rank_translation_profiles(
            [
                {
                    "profile": "slow-more-vram",
                    "status": "passed",
                    "translation_total_elapsed_sec": 10.0,
                    "warm_p50_sec": 2.0,
                    "fallback_count": 0,
                    "gpu_after": {"primary": {"memory_free_mb": 900}},
                },
                {
                    "profile": "fast-low-vram",
                    "status": "passed",
                    "translation_total_elapsed_sec": 3.0,
                    "warm_p50_sec": 1.5,
                    "fallback_count": 4,
                    "gpu_after": {"primary": {"memory_free_mb": 2}},
                },
            ]
        )

        self.assertEqual(ranked[0]["profile"], "fast-low-vram")
        self.assertEqual(ranked[0]["min_gpu_free_mb"], 2)

    def test_split_long_text_for_ctx_produces_ordered_segments(self) -> None:
        text = "First sentence. " * 200
        parts = matrix.split_long_text_for_ctx(text, ctx_size=768, max_completion_tokens=128)

        self.assertGreater(len(parts), 1)
        self.assertEqual("".join(parts), text)

    def test_select_benchmark_chunks_keeps_long_tail_examples(self) -> None:
        chunks = [
            {"id": f"chunk-{index}", "texts": [str(index)], "total_chars": index, "max_block_chars": index}
            for index in range(1, 21)
        ]

        selected = matrix.select_benchmark_chunks(chunks, max_chunks=8)

        self.assertLessEqual(len(selected), 8)
        self.assertIn(20, {item["total_chars"] for item in selected})

    def test_page_snapshot_chunks_extract_text_without_paths_in_summary(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "page_snapshots.json"
            path.write_text(
                json.dumps(
                    {
                        "pages": [
                            {
                                "image_path": "C:/ExampleWorkspace/private/page.png",
                                "blocks": [{"text": "Sensitive line"}, {"text": ""}, {"text": "Second"}],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            chunks = matrix.iter_page_snapshot_chunks(path, chunk_size=2)
            report = matrix.coverage_report(chunks, max_completion_tokens=512)
            payload = json.dumps(report, ensure_ascii=False)

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["block_count"], 2)
        self.assertNotIn("Sensitive line", payload)
        self.assertNotIn("ExampleWorkspace", payload)


if __name__ == "__main__":
    unittest.main()
