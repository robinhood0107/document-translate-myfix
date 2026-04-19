from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets

from app.controllers.batch_report import BatchReportController


class _FakeButton:
    def __init__(self) -> None:
        self.enabled: bool | None = None

    def setEnabled(self, value: bool) -> None:
        self.enabled = bool(value)


class _FakeRadio:
    def isChecked(self) -> bool:
        return True


class _FakeMain:
    def __init__(self) -> None:
        self.batch_report_button = _FakeButton()
        self.retry_failed_button = _FakeButton()
        self.one_page_auto_button = _FakeButton()
        self.automatic_radio = _FakeRadio()
        self._batch_active = False
        self.curr_img_idx = 0
        self.image_files = ["/tmp/page-001.png"]
        self.image_states = {"/tmp/page-001.png": {"skip": False}}


class BatchReportRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_preflight_errors_are_preserved_in_project_payload(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")
        ctrl.register_preflight_error("HunyuanOCR runtime setup failed", "No such image")
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        self.assertEqual(len(finalized["preflight_errors"]), 1)
        self.assertTrue(main.batch_report_button.enabled)

        payload = ctrl.export_latest_report_for_project()
        self.assertIsNotNone(payload)
        self.assertEqual(payload["preflight_errors"][0]["title"], "HunyuanOCR runtime setup failed")
        self.assertEqual(payload["preflight_errors"][0]["details"], "No such image")

        imported_main = _FakeMain()
        imported = BatchReportController(imported_main)
        imported.import_latest_report_from_project(payload)
        reexported = imported.export_latest_report_for_project()
        self.assertIsNotNone(reexported)
        self.assertEqual(reexported["preflight_errors"], payload["preflight_errors"])
        self.assertTrue(imported_main.batch_report_button.enabled)


if __name__ == "__main__":
    unittest.main()
