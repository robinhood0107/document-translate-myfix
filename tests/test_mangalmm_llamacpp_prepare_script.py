from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from modules.ocr.mangalmm_llamacpp_runtime_contract import (
    DEFAULT_MANGALMM_LLAMA_CPP_IMAGE,
    DEFAULT_MANGALMM_MODEL_VOLUME,
    MANGALMM_MMPROJ_NAME,
    MANGALMM_MODEL_NAME,
    MANGALMM_MODEL_SPECS,
)


ROOT = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = ROOT / "scripts" / "prepare_mangalmm_llamacpp_runtime.ps1"
COMPOSE_FILE = ROOT / "mangalmm_docker_files" / "docker-compose.yaml"


class MangaLMMPrepareScriptTests(unittest.TestCase):
    def test_ready_manifest_is_published_after_copy_verify_and_cuda_smoke(
        self,
    ) -> None:
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
        self.assertIn("'--gpus', 'all'", script)
        self.assertNotIn("'down'", script)

    def test_script_pins_exact_files_volume_and_image(self) -> None:
        script = PREPARE_SCRIPT.read_text(encoding="utf-8")

        self.assertIn(DEFAULT_MANGALMM_MODEL_VOLUME, script)
        image_repository, image_digest = DEFAULT_MANGALMM_LLAMA_CPP_IMAGE.split(
            "@",
            1,
        )
        self.assertIn(f"{image_repository}@sha256:", script)
        self.assertIn(image_digest.removeprefix("sha256:"), script)
        for name, spec in MANGALMM_MODEL_SPECS.items():
            self.assertIn(name, script)
            self.assertIn(str(spec["bytes"]), script)
            self.assertIn(str(spec["sha256"]), script)

    def test_compose_uses_read_only_external_volume_and_pinned_runtime(
        self,
    ) -> None:
        compose_text = COMPOSE_FILE.read_text(encoding="utf-8")
        compose = yaml.safe_load(compose_text)
        self.assertEqual(set(compose["services"]), {"mangalmm-local-server"})

        service = compose["services"]["mangalmm-local-server"]
        command = [str(value) for value in service["command"]]
        self.assertIn(
            (
                "/models/${MANGALMM_MODEL_FILE:-"
                f"{MANGALMM_MODEL_NAME}}}"
            ),
            command,
        )
        self.assertIn(
            (
                "/models/${MANGALMM_MMPROJ_FILE:-"
                f"{MANGALMM_MMPROJ_NAME}}}"
            ),
            command,
        )
        self.assertIn("--cache-ram", command)
        self.assertEqual(service["pull_policy"], "missing")
        self.assertNotIn("../testmodel", compose_text)
        self.assertTrue(
            any(
                str(mount).endswith(":/models:ro")
                for mount in service["volumes"]
            )
        )

        volume = compose["volumes"]["mangalmm-models"]
        self.assertTrue(volume["external"])
        self.assertIn(DEFAULT_MANGALMM_MODEL_VOLUME, volume["name"])


if __name__ == "__main__":
    unittest.main()
