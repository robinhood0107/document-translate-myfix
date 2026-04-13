from __future__ import annotations

import unittest
from unittest import mock

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.utils.exceptions import LocalServiceSetupError
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


if __name__ == "__main__":
    unittest.main()
