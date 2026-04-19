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

    def test_default_mode_uses_stage_batched_and_ctd_mask_refiner(self) -> None:
        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(
            page.ui.tools_page.automatic_runtime_value_label.text(),
            "RT-DETR-v2 + CTD Line Protect + Source LaMa",
        )
        self.assertEqual(page.get_workflow_mode(), "stage_batched_pipeline")
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
        settings.setValue("tools/workflow_mode", "legacy_page_pipeline")
        settings.setValue("tools/mask_refiner_settings/mask_inpaint_mode", "rtdetr_source_ctd_lama")
        settings.setValue("tools/mask_refiner_settings/mask_refiner", "ctd")
        settings.setValue("tools/mask_refiner_settings/keep_existing_lines", True)
        settings.sync()

        page = self._make_page()
        page.load_settings()

        self.assertEqual(page.get_mask_inpaint_mode(), "rtdetr_legacy_bbox_source_lama")
        self.assertEqual(
            page.ui.tools_page.automatic_runtime_value_label.text(),
            "RT-DETR-v2 + CTD Line Protect + Source LaMa",
        )
        self.assertEqual(page.get_workflow_mode(), "legacy_page_pipeline")
        self.assertEqual(page.get_tool_selection("inpainter"), "lama_large_512px")
        self.assertEqual(page.get_mask_refiner_settings()["mask_refiner"], "ctd")
        self.assertTrue(page.get_mask_refiner_settings()["keep_existing_lines"])
        page.save_settings()
        self.assertEqual(
            settings.value("tools/workflow_mode", "", type=str),
            "legacy_page_pipeline",
        )
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
        page.ui.notifications_page.enable_ntfy_checkbox.setChecked(True)
        page.ui.notifications_page.ntfy_server_url_input.setText("https://ntfy.example.com")
        page.ui.notifications_page.ntfy_topic_input.setText("comic-translate")
        page.ui.notifications_page.ntfy_access_token_input.setText("secret-token")
        page.ui.notifications_page.ntfy_failure_checkbox.setChecked(False)
        page.ui.notifications_page.ntfy_cancelled_checkbox.setChecked(True)
        page.ui.notifications_page.ntfy_timeout_spinbox.setValue(12)
        page.save_settings()

        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        self.assertFalse(settings.value("notifications/enable_completion_sound", True, type=bool))
        self.assertEqual(settings.value("notifications/completion_sound_mode", "", type=str), "file")
        self.assertEqual(settings.value("notifications/completion_sound_file", "", type=str), "notify.wav")
        self.assertTrue(settings.value("notifications/enable_ntfy_notifications", False, type=bool))
        self.assertEqual(
            settings.value("notifications/ntfy_server_url", "", type=str),
            "https://ntfy.example.com",
        )
        self.assertEqual(
            settings.value("notifications/ntfy_topic", "", type=str),
            "comic-translate",
        )
        self.assertEqual(
            settings.value("notifications/ntfy_access_token", "", type=str),
            "secret-token",
        )
        self.assertTrue(settings.value("notifications/ntfy_send_success", False, type=bool))
        self.assertFalse(settings.value("notifications/ntfy_send_failure", True, type=bool))
        self.assertTrue(settings.value("notifications/ntfy_send_cancelled", False, type=bool))
        self.assertEqual(settings.value("notifications/ntfy_timeout_sec", 0, type=int), 12)

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
        page._set_ocr_mode("mangalmm")
        page.save_settings()

        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        self.assertEqual(settings.value("tools/ocr", "", type=str), "mangalmm")
        self.assertEqual(settings.value("mangalmm_ocr/server_url", "", type=str), "http://127.0.0.1:28081/v1")
        self.assertEqual(settings.value("mangalmm_ocr/max_completion_tokens", 0, type=int), 320)
        self.assertEqual(settings.value("mangalmm_ocr/parallel_workers", 0, type=int), 1)
        self.assertEqual(settings.value("mangalmm_ocr/request_timeout_sec", 0, type=int), 75)
        self.assertTrue(settings.value("mangalmm_ocr/raw_response_logging", False, type=bool))
        self.assertFalse(settings.value("mangalmm_ocr/safe_resize", True, type=bool))
        self.assertEqual(settings.value("mangalmm_ocr/max_pixels", 0, type=int), 1500000)
        self.assertEqual(settings.value("mangalmm_ocr/max_long_side", 0, type=int), 1408)

    def test_legacy_optimal_plus_setting_loads_as_optimal(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.setValue("tools/ocr", "best_local_plus")
        settings.sync()

        page = self._make_page()
        page.load_settings()

        combo_items = [page.ui.ocr_combo.itemText(index) for index in range(page.ui.ocr_combo.count())]
        self.assertNotIn("Optimal+ (HunyuanOCR / MangaLMM / PaddleOCR VL)", combo_items)
        self.assertEqual(page.get_tool_selection("ocr"), "best_local")
        self.assertEqual(page.get_ocr_mode_label(), "Optimal (HunyuanOCR / PaddleOCR VL)")

        page.save_settings()
        self.assertEqual(settings.value("tools/ocr", "", type=str), "best_local")

    def test_series_settings_round_trip(self) -> None:
        settings = QtCore.QSettings("ComicLabs", "ComicTranslate")
        settings.setValue("series/queue_failure_policy", "retry")
        settings.setValue("series/retry_count", 2)
        settings.setValue("series/retry_delay_sec", 15)
        settings.setValue("series/auto_open_failed_child", False)
        settings.setValue("series/resume_from_first_incomplete", False)
        settings.setValue("series/return_to_series_after_completion", True)
        settings.sync()

        page = self._make_page()
        page.load_settings()

        series_settings = page.get_series_settings()
        self.assertEqual(series_settings["queue_failure_policy"], "retry")
        self.assertEqual(series_settings["retry_count"], 2)
        self.assertEqual(series_settings["retry_delay_sec"], 15)
        self.assertFalse(series_settings["auto_open_failed_child"])
        self.assertFalse(series_settings["resume_from_first_incomplete"])
        self.assertTrue(series_settings["return_to_series_after_completion"])

        page.save_settings()
        self.assertEqual(settings.value("series/queue_failure_policy", "", type=str), "retry")
        self.assertEqual(settings.value("series/retry_count", 0, type=int), 2)
        self.assertEqual(settings.value("series/retry_delay_sec", 0, type=int), 15)
        self.assertFalse(settings.value("series/auto_open_failed_child", True, type=bool))
        self.assertFalse(settings.value("series/resume_from_first_incomplete", True, type=bool))
        self.assertTrue(settings.value("series/return_to_series_after_completion", False, type=bool))


if __name__ == "__main__":
    unittest.main()
