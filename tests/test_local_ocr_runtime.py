from __future__ import annotations

import subprocess
import unittest
from unittest import mock
from urllib.error import HTTPError

from modules.ocr.local_runtime import (
    MANGALMM_LLAMA_CPP_IMAGE_REF,
    PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST,
    PADDLEOCR_LLAMA_CPP_IMAGE_REF,
    PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_DIGEST,
    PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF,
    LocalOCRRuntimeManager,
)
from modules.ocr.mangalmm_llamacpp_runtime_contract import (
    DEFAULT_MANGALMM_MODEL_VOLUME,
    DEFAULT_MANGALMM_READY_MANIFEST,
    MangaLMMRuntimeContract,
)
from modules.ocr.paddle_llamacpp_runtime_contract import (
    DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
    DEFAULT_PADDLE_LLAMA_READY_MANIFEST,
    PADDLE_LLAMA_MMPROJ_NAME,
    PADDLE_LLAMA_MODEL_NAME,
    PADDLE_LLAMA_MODEL_SPECS,
    PaddleLlamaRuntimeContract,
)
from modules.ocr.paddleocr_vl_spotting.runtime_contract import (
    DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
    DEFAULT_PADDLE_SPOTTING_READY_MANIFEST,
    PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
    PADDLE_SPOTTING_MMPROJ_NAME,
    PADDLE_SPOTTING_MODEL_NAME,
    PADDLE_SPOTTING_MODEL_SPECS,
    PaddleSpottingRuntimeContract,
)
from modules.utils.exceptions import LocalServiceSetupError, OperationCancelledError
from modules.utils.llama_cpp_runtime import DEFAULT_LLAMA_CPP_IMAGE


class _DummySettingsPage:
    def __init__(
        self,
        *,
        paddle_url: str = (
            "http://127.0.0.1:18000/v1/chat/completions"
        ),
        hunyuan_url: str = "http://127.0.0.1:28080/v1",
        mangalmm_url: str = "http://127.0.0.1:28081/v1",
        spotting_url: str = (
            "http://127.0.0.1:18002/v1/chat/completions"
        ),
    ) -> None:
        self._paddle_url = paddle_url
        self._hunyuan_url = hunyuan_url
        self._mangalmm_url = mangalmm_url
        self._spotting_url = spotting_url

    def get_paddleocr_vl_settings(self) -> dict:
        return {"server_url": self._paddle_url}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"server_url": self._hunyuan_url}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {"server_url": self._mangalmm_url}

    def get_paddleocr_vl_spotting_settings(self) -> dict:
        return {"server_url": self._spotting_url}


def _paddle_contract() -> PaddleLlamaRuntimeContract:
    return PaddleLlamaRuntimeContract(
        volume_name=DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
        ready_manifest_name=DEFAULT_PADDLE_LLAMA_READY_MANIFEST,
        ready_manifest_sha256="a" * 64,
        preparation_version=1,
        llama_image_ref=PADDLEOCR_LLAMA_CPP_IMAGE_REF,
        llama_image_id="sha256:llama-image-id",
        compose_file_sha256="compose",
        command_sha256="command",
        fingerprint="runtime",
        command=("--example",),
        runtime_options={"PADDLEOCR_LLAMA_SLEEP_IDLE_SECONDS": "5"},
    )


def _mangalmm_contract() -> MangaLMMRuntimeContract:
    return MangaLMMRuntimeContract(
        volume_name=DEFAULT_MANGALMM_MODEL_VOLUME,
        ready_manifest_name=DEFAULT_MANGALMM_READY_MANIFEST,
        ready_manifest_sha256="b" * 64,
        preparation_version=2,
        llama_image_ref=MANGALMM_LLAMA_CPP_IMAGE_REF,
        llama_image_id="sha256:mangalmm-image-id",
        compose_file_sha256="mangalmm-compose",
        command_sha256="mangalmm-command",
        fingerprint="mangalmm-runtime",
        command=("--example",),
        runtime_options={},
    )


def _paddle_spotting_contract() -> PaddleSpottingRuntimeContract:
    return PaddleSpottingRuntimeContract(
        volume_name=DEFAULT_PADDLE_SPOTTING_MODEL_VOLUME,
        ready_manifest_name=DEFAULT_PADDLE_SPOTTING_READY_MANIFEST,
        ready_manifest_sha256="c" * 64,
        preparation_version=2,
        llama_image_ref=PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_REF,
        llama_image_id="sha256:spotting-image-id",
        compose_file_sha256="spotting-compose",
        command_sha256="spotting-command",
        fingerprint="spotting-runtime",
        command=("--special",),
        runtime_options={},
    )


class LocalOCRRuntimeManagerTests(unittest.TestCase):
    def test_spotting_cache_identity_is_separate_from_crop_contract(
        self,
    ) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        contract = _paddle_spotting_contract()

        with mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=["paddleocr-spotting-llamacpp"],
        ), mock.patch.object(
            manager,
            "_managed_containers_match_contract",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_paddle_spotting_runtime_contract",
            return_value=contract,
        ):
            identity = manager.get_ocr_cache_identity(
                "PaddleOCR VL Spotting",
                settings_page,
            )

        self.assertEqual(identity["strategy"], "paddle_spotting_full_page")
        self.assertEqual(
            identity["clip.vision.image_max_pixels"],
            PADDLE_SPOTTING_IMAGE_MAX_PIXELS,
        )
        self.assertTrue(identity["special_tokens"])
        self.assertEqual(identity["model_file"], PADDLE_SPOTTING_MODEL_NAME)
        self.assertEqual(identity["mmproj_file"], PADDLE_SPOTTING_MMPROJ_NAME)
        self.assertEqual(
            identity["mmproj_sha256"],
            PADDLE_SPOTTING_MODEL_SPECS[
                PADDLE_SPOTTING_MMPROJ_NAME
            ]["sha256"],
        )
        self.assertEqual(
            identity["llama_image_digest"],
            PADDLEOCR_SPOTTING_LLAMA_CPP_IMAGE_DIGEST,
        )
        self.assertNotEqual(
            identity["model_volume"],
            DEFAULT_PADDLE_LLAMA_MODEL_VOLUME,
        )

    def test_windows_rejects_paddle_container_created_by_wsl_compose(self) -> None:
        manager = LocalOCRRuntimeManager()
        contract = _paddle_contract()
        inspected = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="runtime|sha256:llama-image-id|Ubuntu\n",
            stderr="",
        )

        with mock.patch.object(
            manager,
            "_paddle_runtime_contract",
            return_value=contract,
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=inspected,
        ), mock.patch(
            "modules.ocr.local_runtime.os.name",
            "nt",
        ):
            self.assertFalse(
                manager._paddle_containers_match_contract(
                    ["paddleocr-llamacpp"]
                )
            )

    def test_windows_accepts_paddle_container_created_by_windows_compose(self) -> None:
        manager = LocalOCRRuntimeManager()
        contract = _paddle_contract()
        inspected = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="runtime|sha256:llama-image-id|\n",
            stderr="",
        )

        with mock.patch.object(
            manager,
            "_paddle_runtime_contract",
            return_value=contract,
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=inspected,
        ), mock.patch(
            "modules.ocr.local_runtime.os.name",
            "nt",
        ):
            self.assertTrue(
                manager._paddle_containers_match_contract(
                    ["paddleocr-llamacpp"]
                )
            )

    def test_windows_rejects_mangalmm_container_created_by_wsl_compose(
        self,
    ) -> None:
        manager = LocalOCRRuntimeManager()
        inspected = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "mangalmm-runtime|sha256:mangalmm-image-id|Ubuntu\n"
            ),
            stderr="",
        )

        with mock.patch.object(
            manager,
            "_mangalmm_runtime_contract",
            return_value=_mangalmm_contract(),
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=inspected,
        ), mock.patch(
            "modules.ocr.local_runtime.os.name",
            "nt",
        ):
            self.assertFalse(
                manager._mangalmm_containers_match_contract(
                    ["mangalmm-local-server"]
                )
            )

    def test_windows_accepts_exact_mangalmm_container(self) -> None:
        manager = LocalOCRRuntimeManager()
        inspected = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="mangalmm-runtime|sha256:mangalmm-image-id|\n",
            stderr="",
        )

        with mock.patch.object(
            manager,
            "_mangalmm_runtime_contract",
            return_value=_mangalmm_contract(),
        ), mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=inspected,
        ), mock.patch(
            "modules.ocr.local_runtime.os.name",
            "nt",
        ):
            self.assertTrue(
                manager._mangalmm_containers_match_contract(
                    ["mangalmm-local-server"]
                )
            )

    def test_paddle_cache_identity_is_managed_only(self) -> None:
        manager = LocalOCRRuntimeManager()
        unmanaged = _DummySettingsPage(
            paddle_url="http://192.168.0.20:28118/layout-parsing"
        )

        with mock.patch.object(
            manager,
            "_inspect_docker_image_id",
        ) as inspect_image:
            self.assertIsNone(
                manager.get_ocr_cache_identity("PaddleOCR VL", unmanaged)
            )

        inspect_image.assert_not_called()

    def test_paddle_cache_identity_includes_runtime_contract(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        contract = _paddle_contract()

        with mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=["paddleocr-llamacpp"],
        ), mock.patch.object(
            manager,
            "_paddle_containers_match_contract",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_paddle_runtime_contract",
            return_value=contract,
        ):
            identity = manager.get_ocr_cache_identity(
                "PaddleOCR VL",
                settings_page,
            )

        self.assertEqual(identity["identity_schema_version"], 3)
        self.assertEqual(
            identity["llama_image_ref"],
            PADDLEOCR_LLAMA_CPP_IMAGE_REF,
        )
        self.assertEqual(
            identity["llama_image_digest"],
            PADDLEOCR_LLAMA_CPP_IMAGE_DIGEST,
        )
        self.assertEqual(identity["llama_image_id"], "sha256:llama-image-id")
        self.assertEqual(identity["model_file"], PADDLE_LLAMA_MODEL_NAME)
        self.assertEqual(identity["mmproj_file"], PADDLE_LLAMA_MMPROJ_NAME)
        self.assertEqual(
            identity["model_sha256"],
            PADDLE_LLAMA_MODEL_SPECS[PADDLE_LLAMA_MODEL_NAME]["sha256"],
        )
        self.assertEqual(identity["runtime_fingerprint"], "runtime")
        self.assertEqual(identity["command_sha256"], "command")
        self.assertEqual(identity["transport"]["prompt"], "OCR:")

    def test_paddle_cache_identity_requires_exact_managed_containers(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=[],
        ), mock.patch.object(
            manager,
            "_inspect_docker_image_id",
        ) as inspect_image:
            identity = manager.get_ocr_cache_identity(
                "PaddleOCR VL",
                settings_page,
            )

        self.assertIsNone(identity)
        inspect_image.assert_not_called()

    def test_stale_paddle_containers_are_force_recreated(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(manager, "validate_engine"), mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=["paddleocr-llamacpp"],
        ), mock.patch.object(
            manager,
            "_paddle_containers_match_contract",
            return_value=False,
        ), mock.patch.object(
            manager,
            "_wait_for_health",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose:
            manager.ensure_engine("PaddleOCR VL", settings_page)

        run_compose.assert_called_once_with(
            "PaddleOCR VL",
            "up",
            "-d",
            "--force-recreate",
            step_name="force-recreate",
        )
        self.assertEqual(manager._active_engine, "PaddleOCR VL")

    def test_stale_mangalmm_container_is_force_recreated(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(manager, "validate_engine"), mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=["mangalmm-local-server"],
        ), mock.patch.object(
            manager,
            "_mangalmm_containers_match_contract",
            return_value=False,
        ), mock.patch.object(
            manager,
            "_wait_for_health",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose:
            manager.ensure_engine("MangaLMM", settings_page)

        run_compose.assert_called_once_with(
            "MangaLMM",
            "up",
            "-d",
            "--force-recreate",
            step_name="force-recreate",
        )
        self.assertEqual(manager._active_engine, "MangaLMM")

    def test_exact_stopped_mangalmm_container_is_started_without_compose_up(
        self,
    ) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(manager, "validate_engine"), mock.patch.object(
            manager,
            "_present_managed_container_names",
            return_value=["mangalmm-local-server"],
        ), mock.patch.object(
            manager,
            "_mangalmm_containers_match_contract",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_probe_health_state",
            return_value="unavailable",
        ), mock.patch.object(
            manager,
            "_start_existing_managed_containers",
        ) as start_existing, mock.patch.object(
            manager,
            "_wait_for_health",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose:
            manager.ensure_engine("MangaLMM", settings_page)

        start_existing.assert_called_once_with(
            "MangaLMM",
            ["mangalmm-local-server"],
        )
        run_compose.assert_not_called()
        self.assertEqual(manager._active_engine, "MangaLMM")

    def test_mangalmm_handoff_stops_runtime_before_next_gpu_stage(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "MangaLMM"

        with mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose, mock.patch.object(
            manager,
            "_running_managed_container_names",
            return_value=[],
        ):
            manager.release_for_handoff()

        run_compose.assert_called_once_with(
            "MangaLMM",
            "stop",
            "--timeout",
            "10",
            step_name="stop",
        )
        self.assertIsNone(manager._active_engine)

    def test_shutdown_stops_and_preserves_active_engine_containers(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "PaddleOCR VL"
        manager._readiness_cache.add(
            (
                "PaddleOCR VL",
                "http://127.0.0.1:18000/v1/chat/completions",
                "managed",
            )
        )

        with mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(
                 manager,
                 "_running_managed_container_names",
                 return_value=[],
             ):
            manager.shutdown()

        run_compose.assert_called_once_with(
            "PaddleOCR VL",
            "stop",
            "--timeout",
            "10",
            step_name="stop",
        )
        self.assertIsNone(manager._active_engine)
        self.assertFalse(manager._readiness_cache)

    def test_paddle_handoff_preserves_sleeping_containers(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "PaddleOCR VL"

        with mock.patch.object(
            manager,
            "_wait_for_paddle_llama_sleep",
            return_value=True,
        ) as wait_for_sleep, mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose:
            first_report = manager.release_for_handoff()
            second_report = manager.release_for_handoff()

        wait_for_sleep.assert_called_once_with()
        run_compose.assert_not_called()
        self.assertEqual(manager._active_engine, "PaddleOCR VL")
        self.assertTrue(manager._paddle_idle_released)
        self.assertEqual(
            first_report,
            {"runtime_state": "sleeping", "gpu_release_expected": True},
        )
        self.assertEqual(
            second_report,
            {"runtime_state": "sleeping", "gpu_release_expected": False},
        )

    def test_paddle_handoff_stops_when_sleep_is_not_confirmed(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "PaddleOCR VL"

        with mock.patch.object(
            manager,
            "_wait_for_paddle_llama_sleep",
            return_value=False,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose, mock.patch.object(
            manager,
            "_running_managed_container_names",
            return_value=[],
        ):
            report = manager.release_for_handoff()

        run_compose.assert_called_once_with(
            "PaddleOCR VL",
            "stop",
            "--timeout",
            "10",
            step_name="stop",
        )
        self.assertIsNone(manager._active_engine)
        self.assertFalse(manager._paddle_idle_released)
        self.assertEqual(
            report,
            {"runtime_state": "stopped", "gpu_release_expected": True},
        )

    def test_paddle_handoff_adopts_exact_running_containers(self) -> None:
        manager = LocalOCRRuntimeManager()

        with mock.patch.object(
            manager,
            "_running_managed_container_names",
            return_value=["paddleocr-llamacpp"],
        ), mock.patch.object(
            manager,
            "_paddle_containers_match_contract",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_wait_for_paddle_llama_sleep",
            return_value=True,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose:
            manager.release_for_handoff()

        run_compose.assert_not_called()
        self.assertEqual(manager._active_engine, "PaddleOCR VL")
        self.assertTrue(manager._paddle_idle_released)

    def test_paddle_handoff_stops_stale_running_containers(self) -> None:
        manager = LocalOCRRuntimeManager()

        with mock.patch.object(
            manager,
            "_running_managed_container_names",
            side_effect=[
                ["paddleocr-llamacpp"],
                [],
            ],
        ), mock.patch.object(
            manager,
            "_paddle_containers_match_contract",
            return_value=False,
        ), mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose, mock.patch(
            "modules.ocr.local_runtime.time.monotonic",
            side_effect=[0.0, 3.1],
        ):
            manager.release_for_handoff()

        run_compose.assert_called_once_with(
            "PaddleOCR VL",
            "stop",
            "--timeout",
            "10",
            step_name="stop",
        )
        self.assertIsNone(manager._active_engine)
        self.assertFalse(manager._paddle_idle_released)

    def test_shutdown_preserves_active_engine_until_stop_retry_succeeds(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "HunyuanOCR"
        manager._readiness_cache.add(
            ("HunyuanOCR", "http://127.0.0.1:28080/v1", "managed")
        )
        failure = LocalServiceSetupError(
            "stop failed",
            service_name="HunyuanOCR",
            settings_page_name="HunyuanOCR Settings",
        )

        with mock.patch.object(
            manager,
            "_run_compose",
            side_effect=[failure, None],
        ) as run_compose, mock.patch.object(
            manager,
            "_running_managed_container_names",
            return_value=[],
        ):
            with self.assertRaises(LocalServiceSetupError):
                manager.shutdown()
            self.assertEqual(manager._active_engine, "HunyuanOCR")
            manager.shutdown()

        self.assertIsNone(manager._active_engine)
        self.assertFalse(manager._readiness_cache)
        self.assertEqual(run_compose.call_count, 2)

    def test_cancelled_startup_can_stop_containers_started_by_compose(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        cancelled = OperationCancelledError("cancelled during health wait")

        with mock.patch.object(manager, "validate_engine"), \
             mock.patch.object(manager, "_probe_health_state", return_value="unavailable"), \
             mock.patch.object(manager, "_present_managed_container_names", return_value=[]), \
             mock.patch.object(manager, "_wait_for_health", side_effect=cancelled), \
             mock.patch.object(manager, "_run_compose") as run_compose, \
             mock.patch.object(
                 manager,
                 "_running_managed_container_names",
                 return_value=[],
             ):
            with self.assertRaises(OperationCancelledError):
                manager.ensure_engine("PaddleOCR VL", settings_page)
            manager.shutdown()

        self.assertEqual(
            run_compose.call_args_list,
            [
                mock.call("PaddleOCR VL", "up", "-d", step_name="up"),
                mock.call(
                    "PaddleOCR VL",
                    "stop",
                    "--timeout",
                    "10",
                    step_name="stop",
                ),
            ],
        )

    def test_cancelled_startup_cleanup_catches_late_ocr_container_start(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._managed_start_attempted_engine = "PaddleOCR VL"
        manager._active_engine = None

        with mock.patch.object(
            manager,
            "_run_compose",
        ) as run_compose, mock.patch.object(
            manager,
            "_running_managed_container_names",
            side_effect=[[], ["paddleocr-llamacpp"], []],
        ), mock.patch(
            "modules.ocr.local_runtime.time.monotonic",
            side_effect=[0.0, 0.5, 3.1],
        ), mock.patch(
            "modules.ocr.local_runtime.time.sleep",
        ):
            manager._stop_engine("PaddleOCR VL")

        self.assertEqual(run_compose.call_count, 2)

    def test_running_container_check_treats_missing_container_as_stopped(self) -> None:
        manager = LocalOCRRuntimeManager()
        missing = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Error: No such object: paddleocr-server",
        )

        with mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=missing,
        ):
            self.assertEqual(
                manager._running_managed_container_names("PaddleOCR VL"),
                [],
            )

    def test_running_container_check_rejects_docker_inspect_failure(self) -> None:
        manager = LocalOCRRuntimeManager()
        unavailable = subprocess.CompletedProcess(
            args=[],
            returncode=1,
            stdout="",
            stderr="Cannot connect to the Docker daemon",
        )

        with mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=unavailable,
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "could not verify",
        ):
            manager._running_managed_container_names("PaddleOCR VL")

    def test_running_container_check_rejects_invalid_inspect_state(self) -> None:
        manager = LocalOCRRuntimeManager()
        invalid = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="unknown\n",
            stderr="",
        )

        with mock.patch(
            "modules.utils.llama_cpp_runtime.run_docker_command",
            return_value=invalid,
        ), self.assertRaisesRegex(
            LocalServiceSetupError,
            "Unexpected docker inspect state",
        ):
            manager._running_managed_container_names("PaddleOCR VL")

    def test_default_urls_are_managed(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        self.assertTrue(manager.should_manage_engine("PaddleOCR VL", settings_page))
        self.assertTrue(manager.should_manage_engine("HunyuanOCR", settings_page))
        self.assertTrue(manager.should_manage_engine("MangaLMM", settings_page))

    def test_custom_urls_are_not_managed(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage(
            paddle_url="http://192.168.0.10:28118/layout-parsing",
            hunyuan_url="http://127.0.0.1:38080/v1",
            mangalmm_url="http://127.0.0.1:38081/v1",
        )
        self.assertFalse(manager.should_manage_engine("PaddleOCR VL", settings_page))
        self.assertFalse(manager.should_manage_engine("HunyuanOCR", settings_page))
        self.assertFalse(manager.should_manage_engine("MangaLMM", settings_page))

    def test_hunyuan_env_defaults_are_applied(self) -> None:
        manager = LocalOCRRuntimeManager()
        env = manager._build_env("HunyuanOCR")
        self.assertEqual(env["LLAMA_CPP_IMAGE"], DEFAULT_LLAMA_CPP_IMAGE)
        self.assertEqual(env["LLAMA_N_GPU_LAYERS"], "80")

    def test_validate_engine_requires_docker_for_managed_mode(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        with mock.patch.object(
            manager,
            "_resolve_compose_command",
            side_effect=LocalServiceSetupError("docker missing", service_name="HunyuanOCR", settings_page_name="HunyuanOCR Settings"),
        ):
            with self.assertRaises(LocalServiceSetupError):
                manager.validate_engine("HunyuanOCR", settings_page)

    def test_probe_managed_engine_returns_healthy_without_compose_side_effects(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        with mock.patch.object(manager, "_wait_for_health", return_value=True) as wait_for_health, \
             mock.patch.object(manager, "_run_compose") as run_compose:
            result = manager.probe_managed_engine("PaddleOCR VL", settings_page)

        self.assertEqual(result, "healthy")
        wait_for_health.assert_called_once()
        self.assertEqual(
            wait_for_health.call_args.args[0],
            ("http://127.0.0.1:18000/health",),
        )
        run_compose.assert_not_called()

    def test_probe_managed_engine_returns_unavailable_without_compose_side_effects(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        with mock.patch.object(manager, "_wait_for_health", return_value=False) as wait_for_health, \
             mock.patch.object(manager, "_run_compose") as run_compose:
            result = manager.probe_managed_engine("HunyuanOCR", settings_page)

        self.assertEqual(result, "unavailable")
        wait_for_health.assert_called_once()
        run_compose.assert_not_called()

    def test_paddle_health_is_unavailable_when_llama_is_not_ready(self) -> None:
        manager = LocalOCRRuntimeManager()
        with mock.patch(
            "modules.ocr.local_runtime.urlopen",
            side_effect=OSError("connection refused"),
        ):
            state = manager._probe_health_state(
                ("http://127.0.0.1:18000/health",)
            )

        self.assertEqual(state, "unavailable")

    def test_paddle_health_is_healthy_when_llama_is_ready(self) -> None:
        manager = LocalOCRRuntimeManager()
        llama_response = mock.MagicMock()
        llama_response.__enter__.return_value.status = 200

        with mock.patch(
            "modules.ocr.local_runtime.urlopen",
            return_value=llama_response,
        ):
            state = manager._probe_health_state(
                ("http://127.0.0.1:18000/health",)
            )

        self.assertEqual(state, "healthy")

    def test_health_probe_treats_http_404_as_unavailable(self) -> None:
        manager = LocalOCRRuntimeManager()
        error = HTTPError(
            "http://127.0.0.1:18000/health",
            404,
            "Not Found",
            hdrs=None,
            fp=None,
        )

        with mock.patch(
            "modules.ocr.local_runtime.urlopen",
            side_effect=error,
        ):
            state = manager._probe_single_health_state(
                "http://127.0.0.1:18000/health"
            )

        self.assertEqual(state, "unavailable")

    def test_health_probe_treats_http_503_as_loading(self) -> None:
        manager = LocalOCRRuntimeManager()
        error = HTTPError(
            "http://127.0.0.1:18000/health",
            503,
            "Service Unavailable",
            hdrs=None,
            fp=None,
        )

        with mock.patch(
            "modules.ocr.local_runtime.urlopen",
            side_effect=error,
        ):
            state = manager._probe_single_health_state(
                "http://127.0.0.1:18000/health"
            )

        self.assertEqual(state, "loading")

    def test_wait_for_health_waits_until_every_configured_endpoint_is_ready(self) -> None:
        manager = LocalOCRRuntimeManager()
        health_urls = ("http://127.0.0.1:18000/health",)

        with mock.patch.object(
            manager,
            "_probe_health_state",
            side_effect=["loading", "healthy"],
        ) as probe_health, mock.patch("modules.ocr.local_runtime.time.sleep"):
            ready = manager._wait_for_health(
                health_urls,
                timeout_sec=2,
                engine_key="PaddleOCR VL",
                step_key="health_wait",
                message="waiting",
            )

        self.assertTrue(ready)
        self.assertEqual(
            probe_health.call_args_list,
            [mock.call(health_urls), mock.call(health_urls)],
        )

    def test_ensure_engine_waits_for_loading_runtime_before_compose_restart(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="loading"), \
             mock.patch.object(manager, "_wait_for_health", return_value=True) as wait_for_health, \
             mock.patch.object(manager, "_run_compose") as run_compose:
            manager.ensure_engine("MangaLMM", settings_page)

        wait_for_health.assert_called_once()
        run_compose.assert_not_called()

    def test_ensure_engine_starts_existing_managed_containers_before_compose_up(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        events: list[dict] = []

        with mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="unavailable"), \
             mock.patch.object(
                 manager,
                 "_present_managed_container_names",
                 return_value=["paddleocr-llamacpp"],
                 create=True,
             ) as existing_containers, \
             mock.patch.object(
                 manager,
                 "_paddle_containers_match_contract",
                 return_value=True,
             ), \
             mock.patch.object(manager, "_start_existing_managed_containers", create=True) as start_existing, \
             mock.patch.object(manager, "_wait_for_health", return_value=True) as wait_for_health, \
             mock.patch.object(
                 manager,
                 "_run_compose",
                 side_effect=AssertionError("existing managed containers should be started before compose up"),
             ):
            manager.ensure_engine("PaddleOCR VL", settings_page, progress_callback=events.append)

        existing_containers.assert_called_once_with("PaddleOCR VL")
        start_existing.assert_called_once_with(
            "PaddleOCR VL",
            ["paddleocr-llamacpp"],
        )
        wait_for_health.assert_called_once()
        self.assertTrue(any(event.get("step_key") == "container_start" for event in events))

    def test_ensure_engine_reuses_readiness_cache_for_same_engine_url(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        events: list[dict] = []

        with mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="healthy") as probe_health, \
             mock.patch.object(manager, "_run_compose") as run_compose:
            manager.ensure_engine("PaddleOCR VL", settings_page, progress_callback=events.append)
            manager.ensure_engine("PaddleOCR VL", settings_page, progress_callback=events.append)

        self.assertEqual(probe_health.call_count, 2)
        run_compose.assert_not_called()
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_ensure_engine_cache_misses_when_managed_url_changes(self) -> None:
        manager = LocalOCRRuntimeManager()
        first_settings = _DummySettingsPage(
            paddle_url="http://127.0.0.1:18000/v1/chat/completions"
        )
        second_settings = _DummySettingsPage(
            paddle_url="http://127.0.0.1:18000/v1/alternate"
        )

        with mock.patch.object(manager, "should_manage_engine", return_value=True), \
             mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="healthy") as probe_health:
            manager.ensure_engine("PaddleOCR VL", first_settings)
            manager.ensure_engine("PaddleOCR VL", second_settings)

        self.assertEqual(probe_health.call_count, 2)

    def test_ensure_engine_cache_misses_and_stops_active_runtime_when_engine_changes(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()

        with mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="healthy") as probe_health, \
             mock.patch.object(manager, "_stop_engine") as stop_engine:
            manager.ensure_engine("PaddleOCR VL", settings_page)
            manager.ensure_engine("HunyuanOCR", settings_page)

        self.assertEqual(probe_health.call_count, 2)
        stop_engine.assert_called_once_with("PaddleOCR VL")

    def test_cancelled_ensure_engine_does_not_seed_readiness_cache(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        cancel = mock.Mock(side_effect=[True, False])

        with mock.patch.object(manager, "validate_engine", return_value=None), \
             mock.patch.object(manager, "_probe_health_state", return_value="healthy") as probe_health:
            with self.assertRaises(OperationCancelledError):
                manager.ensure_engine("PaddleOCR VL", settings_page, cancel_checker=cancel)
            manager.ensure_engine("PaddleOCR VL", settings_page, cancel_checker=cancel)

        self.assertEqual(probe_health.call_count, 1)


if __name__ == "__main__":
    unittest.main()
