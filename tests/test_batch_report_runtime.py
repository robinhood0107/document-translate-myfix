from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.controllers.batch_report import BatchReportController
from app.controllers.image import ImageStateController


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

    def test_preflight_warnings_are_deduplicated_and_preserved(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")
        ctrl.register_preflight_warning(
            "PDF import memory limit applied",
            "Pages: 3. Requested/applied sizes: 3: 30000×20000 → 12247×8165.",
        )
        ctrl.register_preflight_warning(
            "PDF import memory limit applied",
            "Pages: 3. Requested/applied sizes: 3: 30000×20000 → 12247×8165.",
        )
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        self.assertEqual(len(finalized["preflight_warnings"]), 1)
        payload = ctrl.export_latest_report_for_project()
        imported = BatchReportController(_FakeMain())
        imported.import_latest_report_from_project(payload)
        self.assertEqual(
            imported.export_latest_report_for_project()["preflight_warnings"],
            payload["preflight_warnings"],
        )

    def test_compiled_korean_pdf_warning_is_available(self) -> None:
        translator = QtCore.QTranslator()
        qm_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "resources",
            "translations",
            "compiled",
            "ct_ko.qm",
        )
        self.assertTrue(translator.load(qm_path))
        self._app.installTranslator(translator)
        try:
            translated = QtCore.QCoreApplication.translate(
                "PdfImport", "PDF import memory limit applied"
            )
        finally:
            self._app.removeTranslator(translator)

        self.assertEqual(translated, "PDF 가져오기 메모리 한도 적용")

    def test_inpaint_skip_preserves_stage_and_boundary_reason(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="batch")

        ctrl.register_batch_skip(
            "/tmp/page-001.png",
            "inpaint",
            "IndexError: index 2160 is out of bounds for axis 0 with size 2160\n\nTraceback...",
        )
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        entry = finalized["skipped_entries"][0]
        self.assertEqual(entry["stages"], ["inpaint"])
        self.assertIn("index 2160", entry["errors"][0])
        self.assertIn("Inpainting failed", entry["reasons"][0])
        self.assertIn("Mask boundary exceeded image bounds", entry["reasons"][0])

    def test_legacy_translation_skip_reason_maps_to_webtoon_translation(self) -> None:
        main = _FakeMain()
        ctrl = BatchReportController(main)
        ctrl.start_batch_report(["/tmp/page-001.png"], run_type="webtoon_batch")

        ctrl.register_batch_skip("/tmp/page-001.png", "Translation", "translator timed out")
        finalized = ctrl.finalize_batch_report(False)

        self.assertIsNotNone(finalized)
        entry = finalized["skipped_entries"][0]
        self.assertEqual(entry["stages"], ["Translation"])
        self.assertIn("Webtoon translation chunk failed", entry["reasons"][0])
        self.assertNotIn("Page processing failed", entry["reasons"][0])

    def test_legacy_translation_skip_reason_message_is_not_generic(self) -> None:
        ctrl = object.__new__(ImageStateController)

        message = ctrl._build_skip_message(
            "/tmp/page-001.png",
            "Translation",
            "translator timed out",
        )

        self.assertIn("Could not translate webtoon chunk", message)
        self.assertNotIn("Page processing failed", message)


if __name__ == "__main__":
    unittest.main()
