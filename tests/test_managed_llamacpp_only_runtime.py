from __future__ import annotations

import json
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.ui.settings.settings_page import (  # noqa: E402
    migrate_managed_runtime_to_llamacpp,
)
from modules.ocr.local_runtime import LocalOCRRuntimeManager  # noqa: E402
from modules.ocr.managed_backend_policy import (  # noqa: E402
    MANAGED_LLAMA_CPP_MIGRATION_VERSION,
    MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY,
    MANAGED_LOCAL_INFERENCE_BACKEND,
    find_vllm_process_commands,
    sanitize_managed_runtime_environment,
)
from scripts.retire_legacy_vllm_runtime import (  # noqa: E402
    DEFAULT_MANIFEST,
    LegacyRuntimeRetirementError,
    resolve_manifest,
    retire_manifest,
)
from scripts.verify_managed_llamacpp_runtime import (  # noqa: E402
    verify_static_contracts,
)


class _FakeQSettings:
    def __init__(self, values: dict[str, object]) -> None:
        self.values = dict(values)
        self.writes: list[tuple[str, object]] = []
        self.sync_count = 0

    def value(self, key: str, default=None, type=None):
        value = self.values.get(key, default)
        if type is None or value is None:
            return value
        return type(value)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value
        self.writes.append((key, value))

    def sync(self) -> None:
        self.sync_count += 1


class ManagedLlamaCppOnlyRuntimeTests(unittest.TestCase):
    def test_legacy_backend_migration_preserves_endpoint_and_other_values(
        self,
    ) -> None:
        settings = _FakeQSettings(
            {
                "paddleocr_vl/backend": "vLLM",
                "paddleocr_vl/server_url": "http://example.test/custom",
                "paddleocr_vl/max_new_tokens": 768,
            }
        )

        changed = migrate_managed_runtime_to_llamacpp(settings)

        self.assertTrue(changed)
        self.assertEqual(
            settings.values["paddleocr_vl/backend"],
            MANAGED_LOCAL_INFERENCE_BACKEND,
        )
        self.assertEqual(
            settings.values["paddleocr_vl/server_url"],
            "http://example.test/custom",
        )
        self.assertEqual(settings.values["paddleocr_vl/max_new_tokens"], 768)
        self.assertEqual(
            settings.values[MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY],
            MANAGED_LLAMA_CPP_MIGRATION_VERSION,
        )
        self.assertEqual(settings.sync_count, 1)

    def test_completed_migration_does_not_overwrite_later_values(self) -> None:
        settings = _FakeQSettings(
            {
                MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY: (
                    MANAGED_LLAMA_CPP_MIGRATION_VERSION
                ),
                "paddleocr_vl/backend": "vllm",
            }
        )

        self.assertFalse(migrate_managed_runtime_to_llamacpp(settings))
        self.assertEqual(settings.values["paddleocr_vl/backend"], "vllm")
        self.assertEqual(settings.writes, [])
        self.assertEqual(settings.sync_count, 0)

    def test_new_install_records_migration_without_backend_override(self) -> None:
        settings = _FakeQSettings({})

        self.assertFalse(migrate_managed_runtime_to_llamacpp(settings))
        self.assertEqual(
            settings.values[MANAGED_LLAMA_CPP_MIGRATION_VERSION_KEY],
            MANAGED_LLAMA_CPP_MIGRATION_VERSION,
        )
        self.assertNotIn("paddleocr_vl/backend", settings.values)

    def test_managed_environment_drops_only_retired_vllm_controls(self) -> None:
        environment, ignored = sanitize_managed_runtime_environment(
            {
                "PADDLEOCR_BACKEND": "vllm",
                "PADDLEOCR_VLLM_IMAGE": "legacy-image",
                "PADDLEOCR_LLAMA_CTX_SIZE": "4096",
                "UNRELATED_BACKEND": "vllm",
            }
        )

        self.assertEqual(
            set(ignored),
            {"PADDLEOCR_BACKEND", "PADDLEOCR_VLLM_IMAGE"},
        )
        self.assertEqual(environment["PADDLEOCR_LLAMA_CTX_SIZE"], "4096")
        self.assertEqual(environment["UNRELATED_BACKEND"], "vllm")

    def test_runtime_environment_warning_does_not_forward_vllm_values(
        self,
    ) -> None:
        manager = LocalOCRRuntimeManager()
        with mock.patch.dict(
            os.environ,
            {
                "PADDLEOCR_BACKEND": "vllm",
                "PADDLEOCR_VLLM_CONFIG": "legacy.yml",
            },
            clear=False,
        ), mock.patch.object(
            manager,
            "_paddle_runtime_contract",
        ) as contract:
            contract.return_value.compose_environment.return_value = {}
            with self.assertLogs("modules.ocr.local_runtime", level="WARNING"):
                environment = manager._build_env("PaddleOCR VL")

        self.assertNotIn("PADDLEOCR_BACKEND", environment)
        self.assertNotIn("PADDLEOCR_VLLM_CONFIG", environment)

    def test_process_guard_detects_vllm_but_not_vendor_image_names(self) -> None:
        self.assertEqual(
            find_vllm_process_commands(
                "ARGS\npython -m vllm.entrypoints.openai.api_server\n"
            ),
            ["python -m vllm.entrypoints.openai.api_server"],
        )
        self.assertEqual(
            find_vllm_process_commands(
                "ARGS\npaddlex --serve --pipeline /tmp/pipeline_conf.yaml\n"
            ),
            [],
        )

    def test_all_active_compose_contracts_are_llamacpp_only(self) -> None:
        result = verify_static_contracts()

        self.assertEqual(result["paddle_backend"], "llama-cpp-server")
        self.assertIn("paddleocr-llamacpp", result["services"])
        self.assertNotIn("paddleocr-vllm", result["services"])

    def test_retirement_manifest_dry_run_targets_only_owned_container(
        self,
    ) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        specification = manifest["containers"][0]
        inspected = {
            "Id": "sha256:legacy-container-id",
            "Config": {
                "Image": specification["expected_image"],
                "Labels": {specification["required_label"]: "legacy"},
            },
            "State": {"Running": True},
        }

        with mock.patch(
            "scripts.retire_legacy_vllm_runtime._inspect_container",
            return_value=inspected,
        ), mock.patch(
            "scripts.retire_legacy_vllm_runtime._docker"
        ) as docker:
            actions = retire_manifest(DEFAULT_MANIFEST, execute=False)

        self.assertEqual(
            actions,
            [
                {
                    "container": "paddleocr-vllm",
                    "container_id": "sha256:legacy-container-id",
                    "status": "would-remove",
                }
            ],
        )
        docker.assert_not_called()

    def test_retirement_refuses_container_with_unexpected_image(self) -> None:
        inspected = {
            "Id": "sha256:unexpected-container-id",
            "Config": {
                "Image": "unrelated/image:latest",
                "Labels": {
                    "com.comictranslate.paddleocr-runtime-fingerprint": "x"
                },
            },
            "State": {"Running": False},
        }

        with mock.patch(
            "scripts.retire_legacy_vllm_runtime._inspect_container",
            return_value=inspected,
        ):
            with self.assertRaises(LegacyRuntimeRetirementError):
                retire_manifest(DEFAULT_MANIFEST, execute=False)

    def test_execution_requires_resolved_immutable_identity(self) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        specification = manifest["containers"][0]
        inspected = {
            "Id": "sha256:legacy-container-id",
            "Config": {
                "Image": specification["expected_image"],
                "Labels": {specification["required_label"]: "legacy"},
            },
            "State": {"Running": False},
        }

        with mock.patch(
            "scripts.retire_legacy_vllm_runtime._inspect_container",
            return_value=inspected,
        ), self.assertRaises(LegacyRuntimeRetirementError):
            retire_manifest(DEFAULT_MANIFEST, execute=True)

    def test_resolved_manifest_captures_id_and_label_value(self) -> None:
        manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8"))
        specification = manifest["containers"][0]
        inspected = {
            "Id": "sha256:legacy-container-id",
            "Config": {
                "Image": specification["expected_image"],
                "Labels": {specification["required_label"]: "legacy"},
            },
            "State": {"Running": False},
        }

        with mock.patch(
            "scripts.retire_legacy_vllm_runtime._inspect_container",
            return_value=inspected,
        ):
            resolved = resolve_manifest(DEFAULT_MANIFEST)

        self.assertEqual(
            resolved["containers"][0]["container_id"],
            "sha256:legacy-container-id",
        )
        self.assertEqual(
            resolved["containers"][0]["expected_label_value"],
            "legacy",
        )
        self.assertIn("source_manifest_sha256", resolved)


if __name__ == "__main__":
    unittest.main()
