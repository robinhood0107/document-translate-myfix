from __future__ import annotations

import os
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from modules.ocr.local_runtime import LocalOCRRuntimeManager
from modules.utils.exceptions import LocalServiceSetupError
from modules.utils.pipeline_config import validate_ocr


class _FakeSettingsPage:
    class ui:
        @staticmethod
        def tr(value: str) -> str:
            return value

    def __init__(self, server_url: str = "http://127.0.0.1:28118/layout-parsing") -> None:
        self._server_url = server_url

    def get_tool_selection(self, key: str) -> str:
        assert key == "ocr"
        return "Optimal (HunyuanOCR / PaddleOCR VL)"

    def get_paddleocr_vl_settings(self) -> dict:
        return {"server_url": self._server_url}

    def get_hunyuan_ocr_settings(self) -> dict:
        return {"server_url": "http://127.0.0.1:28080/v1"}


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


class PipelineConfigRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_validate_ocr_initializes_runtime_manager_when_missing(self) -> None:
        main = _FakeMain()
        self.assertFalse(hasattr(main, "local_ocr_runtime_manager"))

        with mock.patch.object(LocalOCRRuntimeManager, "validate_engine", return_value=None) as validate_engine:
            result = validate_ocr(main, source_lang="Japanese")

        self.assertTrue(result)
        self.assertTrue(hasattr(main, "local_ocr_runtime_manager"))
        self.assertIsInstance(main.local_ocr_runtime_manager, LocalOCRRuntimeManager)
        validate_engine.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
