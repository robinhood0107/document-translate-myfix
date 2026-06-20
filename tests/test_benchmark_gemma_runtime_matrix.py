from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_gemma_runtime_matrix as matrix  # noqa: E402


class GemmaRuntimeMatrixTests(unittest.TestCase):
    def test_profile_runtime_overrides_include_extreme_gpu_layers(self) -> None:
        overrides = matrix.profile_runtime_overrides("ctx2560-gpu25-danger")

        self.assertEqual(overrides["context_size"], 2560)
        self.assertEqual(overrides["n_gpu_layers"], 25)

    def test_profile_runtime_overrides_include_low_context_profiles(self) -> None:
        overrides = matrix.profile_runtime_overrides("ctx1280-gpu23-shadow")

        self.assertEqual(overrides["context_size"], 1280)
        self.assertEqual(overrides["n_gpu_layers"], 23)

    def test_profile_runtime_overrides_include_server_tuning_flags(self) -> None:
        batch = matrix.profile_runtime_overrides("ctx2048-batch1024")
        self.assertEqual(batch["batch_size"], 1024)
        self.assertEqual(batch["ubatch_size"], 512)

        flash = matrix.profile_runtime_overrides("ctx2048-flash-attn")
        self.assertTrue(flash["flash_attn"])

        kv = matrix.profile_runtime_overrides("ctx2048-q8-kv")
        self.assertEqual(kv["cache_type_k"], "q8_0")
        self.assertEqual(kv["cache_type_v"], "q8_0")

    def test_chat_payload_uses_synthetic_prompt_only(self) -> None:
        payload = matrix.build_chat_payload("gemma-4-26B-IQ4_NL.gguf", max_tokens=96)
        serialized = json.dumps(payload, ensure_ascii=False)

        self.assertIn("gemma-4-26B-IQ4_NL.gguf", serialized)
        self.assertIn("This is a short benchmark sentence.", serialized)
        self.assertNotIn("C:/", serialized)
        self.assertNotIn("/mnt/c/" + "Users", serialized)
        self.assertEqual(payload["max_tokens"], 96)

    def test_report_omits_raw_payload_content(self) -> None:
        report = matrix.render_report(
            [
                {
                    "profile": "ctx3072-gpu24-extreme",
                    "status": "passed",
                    "health_elapsed_sec": 1.2,
                    "gemma_runtime_overrides": {"context_size": 3072, "n_gpu_layers": 24},
                    "requests": [{"status": "passed", "elapsed_sec": 2.0, "completion_tps": 12.5}],
                    "gpu_after": {"primary": {"memory_free_mb": 120}},
                }
            ]
        )

        self.assertIn("ctx3072-gpu24-extreme", report)
        self.assertIn("12.5", report)
        self.assertNotIn("This is a short benchmark sentence.", report)

    def test_ranking_uses_elapsed_before_gpu_free_memory(self) -> None:
        ranked = matrix.rank_successful_profiles(
            [
                {
                    "profile": "slow-with-more-vram",
                    "status": "passed",
                    "requests": [{"status": "passed", "elapsed_sec": 9.0, "completion_tps": 10}],
                    "gpu_after": {"primary": {"memory_free_mb": 500}},
                },
                {
                    "profile": "fast-with-low-vram",
                    "status": "passed",
                    "requests": [{"status": "passed", "elapsed_sec": 2.0, "completion_tps": 30}],
                    "gpu_after": {"primary": {"memory_free_mb": 5}},
                },
            ]
        )

        self.assertEqual(ranked[0]["profile"], "fast-with-low-vram")
        self.assertEqual(ranked[0]["min_gpu_free_mb"], 5)


if __name__ == "__main__":
    unittest.main()
