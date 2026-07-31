from __future__ import annotations

import unittest
from pathlib import Path

from modules.ocr.paddleocr_vl_spotting.runtime_contract import (
    DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
    DEFAULT_PADDLE_SPOTTING_READY_MANIFEST,
    PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
    PADDLE_SPOTTING_MMPROJ_NAME,
    PADDLE_SPOTTING_MODEL_SPECS,
)


ROOT = Path(__file__).resolve().parents[1]


class PaddleSpottingPreparationScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = (
            ROOT
            / "scripts"
            / "prepare_paddleocr_spotting_llamacpp_runtime.ps1"
        ).read_text(encoding="utf-8")
        cls.compose = (
            ROOT
            / "paddleocr_vl_spotting_docker_files"
            / "docker-compose.yaml"
        ).read_text(encoding="utf-8")

    def test_preparation_uses_only_the_dedicated_spotting_contract(
        self,
    ) -> None:
        self.assertIn(DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME, self.script)
        self.assertIn(DEFAULT_PADDLE_SPOTTING_READY_MANIFEST, self.script)
        self.assertIn(str(PADDLE_SPOTTING_IMAGE_MAX_PIXELS), self.script)
        self.assertIn(PADDLE_SPOTTING_MMPROJ_NAME, self.script)
        self.assertIn(
            PADDLE_SPOTTING_MODEL_SPECS[PADDLE_SPOTTING_MMPROJ_NAME][
                "sha256"
            ],
            self.script,
        )
        self.assertIn("'--special'", self.script)

    def test_preparation_preserves_crop_projector_and_stops_normally(
        self,
    ) -> None:
        self.assertIn(
            "The crop OCR projector must never be modified in place.",
            (
                ROOT
                / "scripts"
                / "derive_paddleocr_spotting_mmproj.py"
            ).read_text(encoding="utf-8"),
        )
        self.assertIn("'stop', '--timeout'", self.script)
        self.assertNotIn("'down'", self.script)
        self.assertNotIn("docker compose down", self.script.lower())
        self.assertIn(
            "[string]$Manifest.source_image_id -ne $ImageId",
            self.script,
        )

    def test_gpu_preflight_uses_total_device_memory(self) -> None:
        self.assertIn("'--query-gpu=memory.used'", self.script)
        self.assertIn("[int]$MaximumBackgroundGpuMiB = 2048", self.script)
        self.assertNotIn("--query-compute-apps=used_memory", self.script)

    def test_compose_is_a_separate_single_server_with_special_tokens(
        self,
    ) -> None:
        self.assertIn("paddleocr-spotting-llamacpp:", self.compose)
        self.assertIn(DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME, self.compose)
        self.assertIn(PADDLE_SPOTTING_MMPROJ_NAME, self.compose)
        self.assertIn("- --special", self.compose)
        self.assertNotIn("paddlex --serve", self.compose)
        self.assertNotIn("paddleocr-server", self.compose)


if __name__ == "__main__":
    unittest.main()
