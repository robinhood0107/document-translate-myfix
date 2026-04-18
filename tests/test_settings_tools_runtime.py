from __future__ import annotations

import os
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
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
        patchers = [
            mock.patch("app.ui.settings.settings_page.UpdateChecker", _FakeUpdateChecker),
            mock.patch("app.ui.settings.notifications_page.get_music_dir", return_value=Path(self._temp_dir.name)),
            mock.patch("app.ui.settings.notifications_page.list_music_wav_files", return_value=["notify.wav"]),
        ]
        stack = ExitStack()
        for patcher in patchers:
            stack.enter_context(patcher)
        self.addCleanup(stack.close)
        page = SettingsPage()
        self.addCleanup(page.deleteLater)
        return page

    def test_default_mode_uses_ctd_mask_refiner_and_source_lama(self) -> None:
        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(
            page.ui.tools_page.automatic_runtime_value_label.text(),
            "RT-DETR-v2 + Legacy BBox Rescue + Source LaMa",
        )
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")
        self.assertFalse(page.ui.inpainter_combo.isEnabled())
        self.assertEqual(page.get_tool_selection("detector"), "RT-DETR-v2")
        self.assertFalse(page.ui.detector_combo.isEnabled())
        self.assertFalse(hasattr(page.ui, "mask_inpaint_mode_combo"))
        self.assertFalse(hasattr(page.ui, "mask_refiner_combo"))
        self.assertFalse(hasattr(page.ui, "ctd_settings_widget"))
        self.assertEqual(page.get_mask_refiner_settings()["mask_refiner"], "ctd")
        self.assertTrue(page.get_mask_refiner_settings()["keep_existing_lines"])

    def test_mask_refiner_settings_round_trip_preserves_ctd_defaults(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.setValue("tools/detector", "RT-DETR-v2")
        settings.setValue("tools/inpainter", "LaMa")
        settings.setValue("tools/mask_refiner_settings/mask_inpaint_mode", "rtdetr_source_ctd_lama")
        settings.setValue("tools/mask_refiner_settings/mask_refiner", "ctd")
        settings.setValue("tools/mask_refiner_settings/keep_existing_lines", True)
        settings.sync()

        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(
            page.ui.tools_page.automatic_runtime_value_label.text(),
            "RT-DETR-v2 + Legacy BBox Rescue + Source LaMa",
        )
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")
        self.assertEqual(page.get_mask_refiner_settings()["mask_refiner"], "ctd")
        self.assertTrue(page.get_mask_refiner_settings()["keep_existing_lines"])
        page.save_settings()
        self.assertEqual(
            settings.value("tools/mask_refiner_settings/mask_inpaint_mode", "", type=str),
            "rtdetr_legacy_bbox_source_lama",
        )
        self.assertEqual(
            settings.value("tools/mask_refiner_settings/mask_refiner", "", type=str),
            "ctd",
        )
        self.assertTrue(settings.value("tools/mask_refiner_settings/keep_existing_lines", False, type=bool))

    def test_notification_settings_round_trip(self) -> None:
        page = self._make_page()
        page.load_settings()

        self.assertTrue(page.ui.notifications_page.enable_completion_sound_checkbox.isChecked())
        note_labels = page.ui.notifications_page.findChildren(QtWidgets.QLabel)
        joined_notes = "\n".join(label.text() for label in note_labels)
        self.assertIn("repository music folder", joined_notes)
        page.ui.notifications_page.enable_completion_sound_checkbox.setChecked(False)
        combo = page.ui.notifications_page.completion_sound_combo
        combo.setCurrentIndex(1)
        page.save_settings()

        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        self.assertFalse(settings.value("notifications/enable_completion_sound", True, type=bool))
        self.assertEqual(settings.value("notifications/completion_sound_mode", "", type=str), "file")
        self.assertEqual(settings.value("notifications/completion_sound_file", "", type=str), "notify.wav")

    def test_mangalmm_settings_round_trip(self) -> None:
        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_tool_selection("ocr"), "paddleocr_vl")
        page.ui.mangalmm_ocr_server_url_input.setText("http://127.0.0.1:28081/v1")
        page.ui.mangalmm_ocr_max_completion_tokens_spinbox.setValue(320)
        page.ui.mangalmm_ocr_parallel_workers_spinbox.setValue(1)
        page.ui.mangalmm_ocr_request_timeout_spinbox.setValue(75)
        page.ui.mangalmm_ocr_raw_response_logging_checkbox.setChecked(True)
        page.ui.mangalmm_ocr_safe_resize_checkbox.setChecked(False)
        page.ui.mangalmm_ocr_max_pixels_spinbox.setValue(1500000)
        page.ui.mangalmm_ocr_max_long_side_spinbox.setValue(1408)
        page._set_ocr_mode("best_local_plus")
        page.save_settings()

        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        self.assertEqual(settings.value("tools/ocr", "", type=str), "best_local_plus")
        self.assertEqual(settings.value("mangalmm_ocr/server_url", "", type=str), "http://127.0.0.1:28081/v1")
        self.assertEqual(settings.value("mangalmm_ocr/max_completion_tokens", 0, type=int), 320)
        self.assertEqual(settings.value("mangalmm_ocr/parallel_workers", 0, type=int), 1)
        self.assertEqual(settings.value("mangalmm_ocr/request_timeout_sec", 0, type=int), 75)
        self.assertTrue(settings.value("mangalmm_ocr/raw_response_logging", False, type=bool))
        self.assertFalse(settings.value("mangalmm_ocr/safe_resize", True, type=bool))
        self.assertEqual(settings.value("mangalmm_ocr/max_pixels", 0, type=int), 1500000)
        self.assertEqual(settings.value("mangalmm_ocr/max_long_side", 0, type=int), 1408)


if __name__ == "__main__":
    unittest.main()
