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

    def test_tools_page_exposes_ctd_and_torch_inpainter_controls(self) -> None:
        page = self._make_page()

        self.assertIsNotNone(page.ui.mask_refiner_combo)
        self.assertEqual(page.ui.mask_refiner_combo.currentText(), "ctd")
        self.assertTrue(page.ui.keep_existing_lines_checkbox.isChecked())
        self.assertEqual(page.ui.ctd_detect_size_combo.currentText(), "1280")
        self.assertEqual(page.ui.ctd_det_rearrange_max_batches_combo.currentText(), "4")
        self.assertEqual(page.ui.ctd_device_combo.currentText(), "cuda")
        self.assertEqual(page.ui.inpainter_combo.currentText(), "AOT")
        self.assertEqual(page.ui.inpainter_device_combo.currentText(), "cuda")

    def test_save_and_load_round_trip_preserves_ctd_and_runtime_settings(self) -> None:
        page = self._make_page()
        page.ui.mask_refiner_combo.setCurrentIndex(page.ui.mask_refiner_combo.findText("ctd"))
        page.ui.keep_existing_lines_checkbox.setChecked(False)
        page.ui.ctd_detect_size_combo.setCurrentIndex(page.ui.ctd_detect_size_combo.findText("1536"))
        page.ui.ctd_det_rearrange_max_batches_combo.setCurrentIndex(page.ui.ctd_det_rearrange_max_batches_combo.findText("8"))
        page.ui.ctd_device_combo.setCurrentIndex(page.ui.ctd_device_combo.findText("cuda"))
        page.ui.ctd_font_size_multiplier_spinbox.setValue(1.35)
        page.ui.ctd_font_size_max_spinbox.setValue(256)
        page.ui.ctd_font_size_min_spinbox.setValue(12)
        page.ui.ctd_mask_dilate_size_spinbox.setValue(3)
        page.ui.inpainter_combo.setCurrentIndex(page.ui.inpainter_combo.findText("lama_large_512px"))
        page.ui.inpainter_size_combo.setCurrentIndex(page.ui.inpainter_size_combo.findText("1536"))
        page.ui.inpainter_device_combo.setCurrentIndex(page.ui.inpainter_device_combo.findText("cuda"))
        page.ui.inpainter_precision_combo.setCurrentIndex(page.ui.inpainter_precision_combo.findText("bf16"))
        page.save_settings()

        reloaded = self._make_page()
        reloaded.load_settings()

        self.assertEqual(reloaded.ui.mask_refiner_combo.currentText(), "ctd")
        self.assertFalse(reloaded.ui.keep_existing_lines_checkbox.isChecked())
        self.assertEqual(reloaded.ui.ctd_detect_size_combo.currentText(), "1536")
        self.assertEqual(reloaded.ui.ctd_det_rearrange_max_batches_combo.currentText(), "8")
        self.assertEqual(reloaded.ui.ctd_device_combo.currentText(), "cuda")
        self.assertAlmostEqual(reloaded.ui.ctd_font_size_multiplier_spinbox.value(), 1.35, places=2)
        self.assertEqual(reloaded.ui.ctd_font_size_max_spinbox.value(), 256)
        self.assertEqual(reloaded.ui.ctd_font_size_min_spinbox.value(), 12)
        self.assertEqual(reloaded.ui.ctd_mask_dilate_size_spinbox.value(), 3)
        self.assertEqual(reloaded.ui.inpainter_combo.currentText(), "lama_large_512px")
        self.assertEqual(reloaded.ui.inpainter_size_combo.currentText(), "1536")
        self.assertEqual(reloaded.ui.inpainter_device_combo.currentText(), "cuda")
        self.assertEqual(reloaded.ui.inpainter_precision_combo.currentText(), "bf16")

    def test_legacy_lama_setting_migrates_to_lama_large_512px(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.setValue("tools/inpainter", "LaMa")
        settings.sync()

        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.ui.inpainter_combo.currentText(), "lama_large_512px")
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")


if __name__ == "__main__":
    unittest.main()
