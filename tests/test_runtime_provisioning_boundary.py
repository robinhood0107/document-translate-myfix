from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager


ROOT = Path(__file__).resolve().parents[1]


class RuntimeProvisioningBoundaryTests(unittest.TestCase):
    def test_application_runtime_managers_never_pull_or_prepare(self) -> None:
        for manager in (LocalOCRRuntimeManager, LocalGemmaRuntimeManager):
            source = inspect.getsource(manager)
            self.assertNotIn('"docker", "pull"', source)
            self.assertNotIn("run_managed_runtime_preparation", source)
            self.assertNotIn("allow_download=True", source)

    def test_every_product_compose_uses_never_pull_policy(self) -> None:
        compose_files = (
            ROOT / "docker-compose.yaml",
            ROOT / "hunyuanocr_docker_files" / "docker-compose.yaml",
            ROOT / "paddleocr_vl_docker_files" / "docker-compose.yaml",
            ROOT / "paddleocr_vl_spotting_docker_files" / "docker-compose.yaml",
            ROOT / "mangalmm_docker_files" / "docker-compose.yaml",
        )
        for path in compose_files:
            text = path.read_text(encoding="utf-8")
            self.assertIn("pull_policy:", text)
            self.assertNotIn("pull_policy: missing", text)
