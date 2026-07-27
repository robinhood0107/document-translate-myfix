from __future__ import annotations

import unittest
from unittest import mock

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.utils.exceptions import LocalServiceSetupError, OperationCancelledError
from modules.utils.llama_cpp_runtime import DEFAULT_LLAMA_CPP_IMAGE


class _DummySettingsPage:
    def __init__(
        self,
        *,
        paddle_url: str = "http://127.0.0.1:28118/layout-parsing",
        hunyuan_url: str = "http://127.0.0.1:28080/v1",
        mangalmm_url: str = "http://127.0.0.1:28081/v1",
    ) -> None:
        self._paddle_url = paddle_url
        self._hunyuan_url = hunyuan_url
        self._mangalmm_url = mangalmm_url

    def get_paddleocr_vl_settings(self) -> dict:
        return {"server_url": self._paddle_url}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"server_url": self._hunyuan_url}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {"server_url": self._mangalmm_url}


class LocalOCRRuntimeManagerTests(unittest.TestCase):
    def test_shutdown_stops_and_preserves_active_engine_containers(self) -> None:
        manager = LocalOCRRuntimeManager()
        manager._active_engine = "PaddleOCR VL"
        manager._readiness_cache.add(
            ("PaddleOCR VL", "http://127.0.0.1:28118/layout-parsing", "managed")
        )

        with mock.patch.object(manager, "_run_compose") as run_compose:
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

    def test_shutdown_clears_active_engine_when_stop_fails(self) -> None:
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

        with mock.patch.object(manager, "_run_compose", side_effect=failure), \
             self.assertLogs("modules.ocr.local_runtime", level="WARNING"):
            manager.shutdown()

        self.assertIsNone(manager._active_engine)
        self.assertFalse(manager._readiness_cache)

    def test_cancelled_startup_can_stop_containers_started_by_compose(self) -> None:
        manager = LocalOCRRuntimeManager()
        settings_page = _DummySettingsPage()
        cancelled = OperationCancelledError("cancelled during health wait")

        with mock.patch.object(manager, "validate_engine"), \
             mock.patch.object(manager, "_probe_health_state", return_value="unavailable"), \
             mock.patch.object(manager, "_existing_managed_container_names", return_value=[]), \
             mock.patch.object(manager, "_wait_for_health", side_effect=cancelled), \
             mock.patch.object(manager, "_run_compose") as run_compose:
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
                 "_existing_managed_container_names",
                 return_value=["paddleocr-vllm", "paddleocr-server"],
                 create=True,
             ) as existing_containers, \
             mock.patch.object(manager, "_start_existing_managed_containers", create=True) as start_existing, \
             mock.patch.object(manager, "_wait_for_health", return_value=True) as wait_for_health, \
             mock.patch.object(
                 manager,
                 "_run_compose",
                 side_effect=AssertionError("existing managed containers should be started before compose up"),
             ):
            manager.ensure_engine("PaddleOCR VL", settings_page, progress_callback=events.append)

        existing_containers.assert_called_once_with("PaddleOCR VL")
        start_existing.assert_called_once_with("PaddleOCR VL", ["paddleocr-vllm", "paddleocr-server"])
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

        self.assertEqual(probe_health.call_count, 1)
        run_compose.assert_not_called()
        self.assertTrue(any(event.get("readiness_cache_hit") for event in events))

    def test_ensure_engine_cache_misses_when_managed_url_changes(self) -> None:
        manager = LocalOCRRuntimeManager()
        first_settings = _DummySettingsPage(paddle_url="http://127.0.0.1:28118/layout-parsing")
        second_settings = _DummySettingsPage(paddle_url="http://127.0.0.1:28118/alternate")

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
