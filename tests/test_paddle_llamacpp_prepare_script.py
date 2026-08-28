from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from modules.ocr.paddle_llamacpp_runtime_contract import (
    DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
    PADDLE_LLAMA_MMPROJ_NAME,
    PADDLE_LLAMA_MODEL_NAME,
    PADDLE_LLAMA_MODEL_SPECS,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_paddleocr_llamacpp_runtime.ps1"
COMPOSE_FILE = ROOT / "paddleocr_vl_docker_files" / "docker-compose.yaml"


class PaddleLlamaPrepareScriptTests(unittest.TestCase):
    def test_ready_manifest_is_published_after_copy_verify_and_smoke(self) -> None:
        script = PREPARE_SCRIPT.read_text(encoding="utf-8")

        reset_index = script.index(
            'rm -f "/models/$READY_MANIFEST" '
            '"/models/.${READY_MANIFEST}.partial"'
        )
        copy_index = script.index('partial="/models/.${TARGET_FILE}.partial"')
        verify_index = script.index("$VerifiedFiles = @()", copy_index)
        smoke_index = script.index("$SmokeContainer =")
        manifest_index = script.index("$Manifest = [ordered]@{")
        publish_index = script.index(
            'partial="/models/.${READY_MANIFEST}.partial"'
        )

        self.assertLess(reset_index, copy_index)
        self.assertLess(copy_index, verify_index)
        self.assertLess(verify_index, smoke_index)
        self.assertLess(smoke_index, manifest_index)
        self.assertLess(manifest_index, publish_index)
        self.assertIn('mv -f "$partial" "$target"', script)
        self.assertNotIn("'down'", script)

    def test_script_pins_exact_files_volume_and_image(self) -> None:
        script = PREPARE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(DEFAULT_PADDLE_LLAMA_MODEL_VOLUME, script)
        self.assertIn("Get-ManagedLlamaCppImagePolicy", script)
        self.assertIn("Resolve-ManagedLlamaCppImageRef", script)
        for name, spec in PADDLE_LLAMA_MODEL_SPECS.items():
            self.assertIn(name, script)
            self.assertIn(str(spec["bytes"]), script)
            self.assertIn(str(spec["sha256"]), script)

    def test_compose_uses_llama_backend_and_read_only_external_volume(self) -> None:
        compose = yaml.safe_load(COMPOSE_FILE.read_text(encoding="utf-8"))
        services = compose["services"]
        self.assertEqual(
            set(services),
            {"paddleocr-llamacpp"},
        )
        llama = services["paddleocr-llamacpp"]
        command = [str(value) for value in llama["command"]]
        self.assertIn(
            (
                "/models/${PADDLEOCR_LLAMA_MODEL_FILE:-"
                f"{PADDLE_LLAMA_MODEL_NAME}}}"
            ),
            command,
        )
        self.assertIn(
            (
                "/models/${PADDLEOCR_LLAMA_MMPROJ_FILE:-"
                f"{PADDLE_LLAMA_MMPROJ_NAME}}}"
            ),
            command,
        )
        self.assertIn("--sleep-idle-seconds", command)
        volume = compose["volumes"]["paddleocr-llamacpp-models"]
        self.assertTrue(volume["external"])
        self.assertIn(DEFAULT_PADDLE_LLAMA_MODEL_VOLUME, volume["name"])
        self.assertTrue(
            any(
                str(mount).endswith(":/models:ro")
                for mount in llama["volumes"]
            )
        )

    def test_prepare_smoke_calls_direct_chat_completions(self) -> None:
        script = PREPARE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn("/v1/chat/completions", script)
        self.assertIn("OCR:", script)


if __name__ == "__main__":
    unittest.main()
