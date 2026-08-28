from __future__ import annotations

import argparse
import contextlib
import io
import unittest
from unittest import mock

from scripts import windows_install_state
from modules.utils import windows_installation


class WindowsInstallStateTests(unittest.TestCase):
    def _payload(self) -> dict:
        return {
            "schema_version": 1,
            "runtime": "cuda13",
            "provisioned_tier": "core",
            "requirements": {"path": "requirements-cuda13.txt", "sha256": "a" * 64},
            "application_models": [
                {
                    "id": model_id,
                    "files": [
                        {
                            "path": f"models/{model_id}/model.bin",
                            "bytes": 1,
                            "mtime_ns": 1,
                        }
                    ],
                }
                for model_id in sorted(windows_installation.CORE_APPLICATION_MODEL_IDS)
            ],
            "llama_image": {
                "ref": "ghcr.io/ggml-org/llama.cpp:server-cuda",
                "id": "sha256:compatible",
                "required_cuda": "12.8",
            },
            "managed_runtimes": [
                {"name": "hunyuan-volume", "runtime_name": "HunyuanOCR-llama.cpp"},
                {"name": "paddle-volume", "runtime_name": "PaddleOCR-VL-llama.cpp"},
                {"name": "gemma-volume", "runtime_name": "Gemma"},
            ],
        }

    def test_preflight_exports_the_setup_selected_fallback_image(self) -> None:
        args = argparse.Namespace(
            runtime="cuda13",
            requirements="requirements-cuda13.txt",
            emit_cmd=True,
        )
        output = io.StringIO()
        with (
            mock.patch.object(windows_install_state, "_read_state", return_value=self._payload()),
            mock.patch.object(
                windows_install_state,
                "_requirements_record",
                return_value=self._payload()["requirements"],
            ),
            mock.patch.object(windows_install_state, "_validate_application_models"),
            mock.patch.object(windows_install_state, "_validate_docker_state"),
            contextlib.redirect_stdout(output),
        ):
            self.assertEqual(windows_install_state.command_preflight(args), 0)
        self.assertIn(
            "LLAMA_CPP_IMAGE=ghcr.io/ggml-org/llama.cpp:server-cuda",
            output.getvalue(),
        )
        self.assertIn("COMIC_MODEL_DOWNLOAD_POLICY=forbid", output.getvalue())
        self.assertIn("COMIC_WINDOWS_INSTALL_TIER=core", output.getvalue())

    def test_full_preflight_validates_the_full_application_profile(self) -> None:
        payload = self._payload()
        payload["provisioned_tier"] = "full"
        args = argparse.Namespace(
            runtime="cuda13",
            requirements="requirements-cuda13.txt",
            emit_cmd=False,
        )
        with (
            mock.patch.object(windows_install_state, "_read_state", return_value=payload),
            mock.patch.object(
                windows_install_state,
                "_requirements_record",
                return_value=payload["requirements"],
            ),
            mock.patch.object(
                windows_install_state,
                "_validate_application_models",
            ) as validate_models,
            mock.patch.object(windows_install_state, "_validate_docker_state"),
        ):
            windows_install_state.command_preflight(args)

        validate_models.assert_called_once_with(
            payload,
            profile="full",
            allow_digest_fallback=True,
        )

    def test_image_identity_drift_fails_instead_of_repairing(self) -> None:
        args = argparse.Namespace(
            runtime="cuda13",
            requirements="requirements-cuda13.txt",
            emit_cmd=False,
        )
        with (
            mock.patch.object(windows_install_state, "_read_state", return_value=self._payload()),
            mock.patch.object(
                windows_install_state,
                "_requirements_record",
                return_value=self._payload()["requirements"],
            ),
            mock.patch.object(windows_install_state, "_validate_application_models"),
            mock.patch.object(
                windows_install_state,
                "_validate_docker_state",
                side_effect=windows_install_state.InstallStateError("image drift"),
            ),
        ):
            with self.assertRaisesRegex(windows_install_state.InstallStateError, "image drift"):
                windows_install_state.command_preflight(args)

    def test_core_tier_rejects_optional_selection_before_page_work(self) -> None:
        settings = mock.Mock()
        settings.get_tool_selection.side_effect = lambda key: {
            "ocr": "mangalmm",
            "inpainter": "lama_large_512px",
            "translator": "Custom Local Server(Gemma)",
        }.get(key, "")
        payload = self._payload()
        with (
            mock.patch.object(windows_installation, "active_windows_runtime", return_value="cuda13"),
            mock.patch.object(
                windows_installation,
                "active_windows_install_state",
                return_value=payload,
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "setup_full"):
                windows_installation.assert_selected_windows_models_installed(
                    settings,
                    "Japanese",
                )

    def test_full_tier_string_does_not_hide_missing_optional_model_records(self) -> None:
        settings = mock.Mock()
        settings.get_tool_selection.side_effect = lambda key: {
            "ocr": "default",
            "inpainter": "AOT",
            "translator": "Custom Local Server(Gemma)",
        }.get(key, "")
        payload = self._payload()
        payload["provisioned_tier"] = "full"
        payload["managed_runtimes"].extend(
            [
                {"name": "manga-volume", "runtime_name": "MangaLMM-llama.cpp"},
                {
                    "name": "spotting-volume",
                    "runtime_name": "PaddleOCR-VL-Spotting-llama.cpp",
                },
            ]
        )
        with (
            mock.patch.object(windows_installation, "active_windows_runtime", return_value="cuda13"),
            mock.patch.object(
                windows_installation,
                "active_windows_install_state",
                return_value=payload,
            ),
            self.assertRaisesRegex(RuntimeError, "setup_full"),
        ):
            windows_installation.assert_selected_windows_models_installed(
                settings,
                "Japanese",
            )

    def test_full_runtime_record_set_rejects_a_missing_optional_volume(self) -> None:
        payload = self._payload()
        payload["provisioned_tier"] = "full"
        payload["managed_runtimes"].append(
            {"name": "manga-volume", "runtime_name": "MangaLMM-llama.cpp"}
        )

        with self.assertRaisesRegex(
            windows_install_state.InstallStateError,
            "PaddleOCR-VL-Spotting",
        ):
            windows_install_state._validate_managed_runtime_records(
                payload,
                tier="full",
            )
