from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import benchmark_common  # noqa: E402


class BenchmarkGemmaRuntimeStagingTests(unittest.TestCase):
    def test_stage_gemma_runtime_writes_extreme_server_options(self) -> None:
        preset = {
            "gemma": {
                "context_size": 2048,
                "threads": 14,
                "n_gpu_layers": 24,
                "n_parallel": 2,
                "batch_size": 1024,
                "ubatch_size": 512,
                "cache_type_k": "q8_0",
                "cache_type_v": "q8_0",
                "flash_attn": True,
                "no_warmup": True,
            }
        }
        with tempfile.TemporaryDirectory() as tmp:
            staged = benchmark_common._stage_gemma_runtime(preset, Path(tmp))
            compose = yaml.safe_load(Path(staged["compose_path"]).read_text(encoding="utf-8"))

        command = compose["services"]["gemma-local-server"]["command"]
        for expected in (
            "-c",
            "2048",
            "-t",
            "14",
            "--n-gpu-layers",
            "24",
            "-np",
            "2",
            "-b",
            "1024",
            "-ub",
            "512",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--flash-attn",
            "--no-warmup",
        ):
            self.assertIn(expected, command)
        self.assertNotIn("--cache-type-k", command)
        self.assertNotIn("--cache-type-v", command)


if __name__ == "__main__":
    unittest.main()
