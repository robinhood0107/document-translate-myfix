from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_stage_batched_speed_matrix as matrix  # noqa: E402


class StageBatchedSpeedMatrixTests(unittest.TestCase):
    def test_build_profile_command_includes_speed_profile_and_inputs(self) -> None:
        command = matrix.build_profile_command(
            profile="ctx3072-fast",
            input_paths=[Path("example_source_chapter.cbz")],
            run_dir=Path("example_run"),
            source_lang="English",
            target_lang="Korean",
            ocr_mode="optimal-plus",
            archive_format="cbz",
            archive_image_format="png",
        )

        self.assertIn("--speed-profile", command)
        self.assertIn("ctx3072-fast", command)
        self.assertIn("--ocr-mode", command)
        self.assertIn("optimal-plus", command)
        self.assertIn("--input", command)
        self.assertIn("example_source_chapter.cbz", command)

    def test_safe_summary_removes_paths_and_keeps_runtime_numbers(self) -> None:
        cleaned = matrix.safe_summary(
            {
                "image_paths": ["C:/ExampleWorkspace/private/page.png"],
                "final_archive_path": "C:/ExampleWorkspace/private/result.cbz",
                "page_failed_count": 0,
                "final_archive_page_count": 3,
                "archive_compression_level": 0,
                "runtime_reuse_mode": "signature",
                "gemma_runtime_overrides": {"context_size": 3072},
            }
        )

        payload = json.dumps(cleaned, ensure_ascii=False)
        self.assertNotIn("ExampleWorkspace", payload)
        self.assertEqual(cleaned["page_failed_count"], 0)
        self.assertEqual(cleaned["gemma_runtime_overrides"]["context_size"], 3072)

    def test_matrix_report_omits_raw_text(self) -> None:
        report = matrix.render_matrix_report(
            [
                {
                    "profile": "ctx3072-gpu24-extreme",
                    "status": "passed",
                    "elapsed_sec": 12.34,
                    "summary": {
                        "page_failed_count": 0,
                        "final_archive_page_count": 2,
                        "archive_compression_level": 0,
                        "runtime_reuse_mode": "signature",
                        "gemma_runtime_overrides": {"context_size": 3072, "n_gpu_layers": 24},
                    },
                }
            ]
        )

        self.assertIn("ctx3072-gpu24-extreme", report)
        self.assertIn("3072", report)
        self.assertIn("24", report)
        self.assertNotIn("sensitive source text", report)

    def test_anonymize_input_paths_hashes_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "example_source_chapter.cbz"
            source.write_text("not an archive for this unit test", encoding="utf-8")
            anonymized = matrix.anonymize_input_paths([source])

        self.assertEqual(anonymized[0]["suffix"], ".cbz")
        self.assertTrue(anonymized[0]["exists"])
        self.assertNotIn("example_source_chapter", json.dumps(anonymized))
        self.assertEqual(len(anonymized[0]["path_hash"]), 64)

    def test_status_marks_page_failure_as_failed_even_with_zero_returncode(self) -> None:
        self.assertEqual(
            matrix.profile_status(
                returncode=0,
                summary={"page_failed_count": 1},
                allow_failure=False,
            ),
            "failed",
        )
        self.assertEqual(
            matrix.profile_status(
                returncode=0,
                summary={"page_failed_count": 1},
                allow_failure=True,
            ),
            "shadow_failed",
        )


if __name__ == "__main__":
    unittest.main()
