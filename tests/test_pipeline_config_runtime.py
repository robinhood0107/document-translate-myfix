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

    def __init__(
        self,
        server_url: str = (
            "http://127.0.0.1:18000/v1/chat/completions"
        ),
    ) -> None:
        self._server_url = server_url
        self._translator = "Custom Local Server(Gemma)"
        self._ocr = "Optimal (HunyuanOCR / PaddleOCR VL)"
        self._hunyuan_url = "http://127.0.0.1:28080/v1"

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
        return {"server_url": self._hunyuan_url}

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

    def test_validate_ocr_never_touches_docker_on_the_ui_thread(self) -> None:
        """사전 검사는 즉시 읽을 수 있는 설정만 본다.

        런타임 준비 상태 확인(compose, 이미지, 준비 볼륨 계약)은 모두 Docker 호출이고
        파이프라인의 ensure_engine 이 어차피 수행한다. 사전 검사에서 하면 진행 표시
        없이 UI 가 멈추고, 페이지가 많은 아카이브에서는 창이 응답을 잃는다.
        """

        main = _FakeMain()

        with mock.patch.object(
            LocalOCRRuntimeManager, "validate_engine"
        ) as validate_engine, mock.patch.object(
            LocalOCRRuntimeManager, "probe_managed_engine"
        ) as probe_managed_engine:
            self.assertTrue(validate_ocr(main, source_lang="Japanese"))

        validate_engine.assert_not_called()
        probe_managed_engine.assert_not_called()
        self.assertEqual(main.batch_report_ctrl.entries, [])

    def test_validate_ocr_still_rejects_an_empty_server_url(self) -> None:
        """설정만으로 알 수 있는 결함은 계속 즉시 막는다."""

        main = _FakeMain()
        main.settings_page._server_url = ""

        with mock.patch(
            "app.ui.messages.Messages.show_missing_local_service_config_error",
            return_value=None,
        ):
            self.assertFalse(validate_ocr(main, source_lang="Japanese"))

        self.assertEqual(len(main.batch_report_ctrl.entries), 1)
        title, details = main.batch_report_ctrl.entries[0]
        self.assertIn("settings missing", title)
        self.assertIn("Server URL", details)

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
