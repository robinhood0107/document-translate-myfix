from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.translation.local_runtime import LocalGemmaRuntimeManager
from modules.utils.exceptions import LocalServiceSetupError
from modules.utils.pipeline_config import validate_ocr, validate_translator


class _FakeSettingsPage:
    class ui:
        value_mappings = {"Custom Local Server(Gemma)": "Custom Local Server(Gemma)"}

        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(self, server_url: str = "http://127.0.0.1:28118/layout-parsing") -> None:
        self._server_url = server_url
        self._translator = "Custom Local Server(Gemma)"
        self._ocr = "Optimal (HunyuanOCR / PaddleOCR VL)"

    def get_tool_selection(self, key: str) -> str:
        if key == "ocr":
            return self._ocr
        if key == "translator":
            return self._translator
        raise AssertionError(key)

    def get_workflow_mode(self) -> str:
        return "legacy_page_pipeline"

    def get_paddleocr_vl_settings(self) -> dict:
        return {"server_url": self._server_url}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28080/v1"}

    def get_mangalmm_ocr_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28081/v1"}

    def get_all_settings(self) -> dict:
        return {"tools": {"translator": self._translator}}

    def get_credentials(self, provider_name: str) -> dict:
        if provider_name == "Custom Local Server(Gemma)":
            return {
                "api_url": "http://127.0.0.1:18080/v1",
                "model": "gemma-4-26b-a4b-it-heretic.q3_k_m.gguf",
            }
        return {}


class _FakeCombo:
    def currentText(self) -> str:
        return "Japanese"


class _FakeBatchReportCtrl:
    def __init__(self) -> None:
        self.entries: list[tuple[str, str]] = []

    def register_preflight_error(self, title: str, details: str = "") -> None:
        self.entries.append((title, details))


class _FakeMain:
    def __init__(self) -> None:
        self.settings_page = _FakeSettingsPage()
        self.s_combo = _FakeCombo()
        self.lang_mapping = {"Japanese": "Japanese"}
        self.batch_report_ctrl = _FakeBatchReportCtrl()
        self.t_combo = _FakeCombo()


class PipelineConfigRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_validate_ocr_initializes_runtime_manager_when_missing(self) -> None:
        main = _FakeMain()
        self.assertFalse(hasattr(main, "local_ocr_runtime_manager"))

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", return_value=None) as validate_engine, \
             mock.patch.object(LocalOCRRuntimeManager, "preflight_cache_key", return_value="HunyuanOCR|http://127.0.0.1:28080/v1") as preflight_cache_key, \
             mock.patch.object(LocalOCRRuntimeManager, "probe_managed_engine", return_value="healthy") as probe_managed_engine:
            result = validate_ocr(main, source_lang="Japanese")

        self.assertTrue(result)
        self.assertTrue(hasattr(main, "local_ocr_runtime_manager"))
        self.assertIsInstance(main.local_ocr_runtime_manager, LocalOCRRuntimeManager)
        validate_engine.assert_called_once()
        preflight_cache_key.assert_called_once()
        probe_managed_engine.assert_called_once()

    def test_validate_ocr_registers_preflight_error_for_local_runtime_failure(self) -> None:
        main = _FakeMain()
        failure = LocalServiceSetupError(
            "No such image: local/llama.cpp:server-cuda-b8672",
            service_name="HunyuanOCR",
            settings_page_name="HunyuanOCR Settings",
        )

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", side_effect=failure), \
             mock.patch("app.ui.messages.Messages.show_local_service_error", return_value=None):
            result = validate_ocr(main, source_lang="Japanese")

        self.assertFalse(result)
        self.assertEqual(len(main.batch_report_ctrl.entries), 1)
        title, details = main.batch_report_ctrl.entries[0]
        self.assertIn("runtime setup failed", title)
        self.assertIn("No such image", details)

    def test_validate_ocr_health_miss_does_not_fail_preflight(self) -> None:
        main = _FakeMain()

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", return_value=None), \
             mock.patch.object(LocalOCRRuntimeManager, "preflight_cache_key", return_value="HunyuanOCR|http://127.0.0.1:28080/v1"), \
             mock.patch.object(LocalOCRRuntimeManager, "probe_managed_engine", return_value="unavailable") as probe_managed_engine:
            result = validate_ocr(main, source_lang="Japanese")

        self.assertTrue(result)
        self.assertEqual(main.batch_report_ctrl.entries, [])
        probe_managed_engine.assert_called_once()

    def test_validate_ocr_supports_direct_mangalmm_runtime(self) -> None:
        main = _FakeMain()
        main.settings_page._ocr = "MangaLMM"

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", return_value=None) as validate_engine, \
             mock.patch.object(LocalOCRRuntimeManager, "preflight_cache_key", return_value="MangaLMM|http://127.0.0.1:28081/v1") as preflight_cache_key, \
             mock.patch.object(LocalOCRRuntimeManager, "probe_managed_engine", return_value="healthy") as probe_managed_engine:
            result = validate_ocr(main, source_lang="Japanese")

        self.assertTrue(result)
        validate_engine.assert_called_once()
        preflight_cache_key.assert_called_once()
        probe_managed_engine.assert_called_once()

    def test_validate_ocr_reuses_preflight_cache_for_same_engine_and_url(self) -> None:
        main = _FakeMain()
        preflight_cache: dict[str, str] = {}

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", return_value=None), \
             mock.patch.object(LocalOCRRuntimeManager, "preflight_cache_key", return_value="HunyuanOCR|http://127.0.0.1:28080/v1"), \
             mock.patch.object(LocalOCRRuntimeManager, "probe_managed_engine", return_value="healthy") as probe_managed_engine:
            first = validate_ocr(main, source_lang="Japanese", preflight_cache=preflight_cache)
            second = validate_ocr(main, source_lang="Japanese", preflight_cache=preflight_cache)

        self.assertTrue(first)
        self.assertTrue(second)
        self.assertEqual(preflight_cache, {"HunyuanOCR|http://127.0.0.1:28080/v1": "healthy"})
        probe_managed_engine.assert_called_once()

    def test_validate_translator_initializes_runtime_manager_when_missing(self) -> None:
        main = _FakeMain()
        self.assertFalse(hasattr(main, "local_translation_runtime_manager"))

        with mock.patch.object(LocalGemmaRuntimeManager, "validate_server", return_value=None) as validate_server:
            result = validate_translator(main, "English")

        self.assertTrue(result)
        self.assertTrue(hasattr(main, "local_translation_runtime_manager"))
        self.assertIsInstance(main.local_translation_runtime_manager, LocalGemmaRuntimeManager)
        validate_server.assert_called_once()

    def test_validate_translator_registers_preflight_error_for_runtime_failure(self) -> None:
        main = _FakeMain()
        failure = LocalServiceSetupError(
            "Docker compose up failed",
            service_name="Gemma",
            settings_page_name="Gemma Local Server Settings",
        )

        with mock.patch.object(LocalGemmaRuntimeManager, "validate_server", side_effect=failure), \
             mock.patch("app.ui.messages.Messages.show_local_service_error", return_value=None):
            result = validate_translator(main, "English")

        self.assertFalse(result)
        self.assertEqual(len(main.batch_report_ctrl.entries), 1)
        title, details = main.batch_report_ctrl.entries[0]
        self.assertIn("Gemma", title)
        self.assertIn("Docker compose up failed", details)

    def test_gemma_runtime_env_uses_configured_model_filename(self) -> None:
        settings = _FakeSettingsPage()
        settings.get_credentials = lambda provider_name: {
            "api_url": "http://127.0.0.1:18080/v1",
            "model": "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf",
        } if provider_name == "Custom Local Server(Gemma)" else {}
        manager = LocalGemmaRuntimeManager()
        self.assertEqual(manager._build_env("gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf")["LLAMA_MODEL_FILE"], "gemma-4-26B-A4B-it-UD-Q2_K_XL.gguf")


if __name__ == "__main__":
    unittest.main()
