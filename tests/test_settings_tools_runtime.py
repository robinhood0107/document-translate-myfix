from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtCore, QtWidgets

from app.ui.settings.settings_page import SettingsPage


class _Signal:
    def connect(self, *_args, **_kwargs) -> None:
        return None


class _FakeUpdateChecker:
    def __init__(self) -> None:
        self.update_available = _Signal()
        self.up_to_date = _Signal()
        self.error_occurred = _Signal()
        self.download_progress = _Signal()
        self.download_finished = _Signal()


class SettingsToolsRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        QtCore.QSettings.setDefaultFormat(QtCore.QSettings.Format.IniFormat)
        QtCore.QSettings.setPath(
            QtCore.QSettings.Format.IniFormat,
            QtCore.QSettings.Scope.UserScope,
            self._temp_dir.name,
        )
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.clear()
        settings.sync()

    def tearDown(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.clear()
        settings.sync()
        self._temp_dir.cleanup()

    def _make_page(self) -> SettingsPage:
        with mock.patch("app.ui.settings.settings_page.UpdateChecker", _FakeUpdateChecker):
            page = SettingsPage()
        self.addCleanup(page.deleteLater)
        return page

    def test_default_mode_uses_legacy_bbox_and_source_lama(self) -> None:
        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(page.ui.mask_inpaint_mode_combo.currentText(), "RT-DETR-v2 + Legacy BBox + Source LaMa")
        self.assertEqual(page.ui.mask_refiner_combo.currentText(), "legacy_bbox")
        self.assertFalse(page.ui.mask_refiner_combo.isEnabled())
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")
        self.assertFalse(page.ui.inpainter_combo.isEnabled())
        self.assertEqual(page.get_tool_selection("detector"), "RT-DETR-v2")
        self.assertFalse(page.ui.detector_combo.isEnabled())
        self.assertFalse(page.ui.keep_existing_lines_checkbox.isChecked())
        self.assertTrue(page.ui.keep_existing_lines_checkbox.isHidden())
        self.assertTrue(page.ui.ctd_settings_widget.isHidden())

    def test_source_parity_mode_forces_ctd_and_disables_detector(self) -> None:
        page = self._make_page()
        idx = page.ui.mask_inpaint_mode_combo.findText("Source Parity CTD/LaMa")
        page.ui.mask_inpaint_mode_combo.setCurrentIndex(idx)
        page.ui.tools_page._update_mask_inpaint_mode_widgets(idx)

        self.assertEqual(page.get_mask_inpaint_mode(), "source_parity_ctd_lama")
        self.assertEqual(page.ui.mask_refiner_combo.currentText(), "ctd")
        self.assertFalse(page.ui.mask_refiner_combo.isEnabled())
        self.assertFalse(page.ui.detector_combo.isEnabled())
        self.assertFalse(page.ui.ctd_settings_widget.isHidden())
        self.assertFalse(page.ui.keep_existing_lines_checkbox.isChecked())
        self.assertTrue(page.ui.keep_existing_lines_checkbox.isHidden())

    def test_old_hybrid_mode_value_migrates_to_new_legacy_bbox_mode(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.setValue("tools/detector", "RT-DETR-v2")
        settings.setValue("tools/inpainter", "LaMa")
        settings.setValue("tools/mask_refiner_settings/mask_inpaint_mode", "rtdetr_source_ctd_lama")
        settings.setValue("tools/mask_refiner_settings/mask_refiner", "ctd")
        settings.sync()

        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(page.ui.mask_inpaint_mode_combo.currentText(), "RT-DETR-v2 + Legacy BBox + Source LaMa")
        self.assertEqual(page.ui.mask_refiner_combo.currentText(), "legacy_bbox")
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")
        self.assertFalse(page.ui.keep_existing_lines_checkbox.isChecked())


if __name__ == "__main__":
    unittest.main()
