from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from app.ui.settings.settings_page import SettingsPage


class _Check:
    def __init__(self, checked: bool) -> None:
        self._checked = checked

    def isChecked(self) -> bool:
        return self._checked


class _Spin:
    def __init__(self, value: int) -> None:
        self._value = value

    def value(self) -> int:
        return self._value


class _Text:
    def __init__(self, value: str) -> None:
        self._value = value

    def text(self) -> str:
        return self._value


class _Combo:
    def __init__(self, value: str) -> None:
        self._value = value

    def currentData(self):
        return self._value


class _FakeQSettings:
    def beginGroup(self, _name: str) -> None:
        return None

    def value(self, key: str, default=None, type=None):
        if key == "project_autosave_enabled":
            return False
        return default

    def endGroup(self) -> None:
        return None


class _FakePage:
    def __init__(self) -> None:
        self.ui = SimpleNamespace(
            raw_text_checkbox=_Check(True),
            translated_text_checkbox=_Check(False),
            inpainted_image_checkbox=_Check(True),
            detector_overlay_checkbox=_Check(True),
            raw_mask_checkbox=_Check(True),
            mask_overlay_checkbox=_Check(False),
            cleanup_mask_delta_checkbox=_Check(True),
            debug_metadata_checkbox=_Check(False),
            automatic_output_target_combo=_Combo("single_archive"),
            automatic_output_image_format_combo=_Combo("webp"),
            automatic_output_archive_format_combo=_Combo("zip"),
            automatic_output_archive_image_format_combo=_Combo("jpg"),
            automatic_output_archive_level_spinbox=_Spin(7),
            project_autosave_interval_spinbox=_Spin(5),
            project_autosave_folder_input=_Text(""),
        )
        self.auto_export_source_txt_checkbox = _Check(True)
        self.auto_export_source_md_checkbox = _Check(False)
        self.auto_export_translation_txt_checkbox = _Check(True)
        self.auto_export_translation_md_checkbox = _Check(False)

    def window(self):
        return self


class ExportSettingsTests(unittest.TestCase):
    def test_get_export_settings_includes_debug_flags(self) -> None:
        page = _FakePage()
        with mock.patch("app.ui.settings.settings_page.QSettings", return_value=_FakeQSettings()):
            with mock.patch(
                "app.ui.settings.settings_page.get_default_project_autosave_dir",
                return_value="/tmp/projects",
            ):
                settings = SettingsPage.get_export_settings(page)

        self.assertTrue(settings["export_raw_text"])
        self.assertTrue(settings["export_inpainted_image"])
        self.assertTrue(settings["export_detector_overlay"])
        self.assertTrue(settings["export_raw_mask"])
        self.assertFalse(settings["export_mask_overlay"])
        self.assertTrue(settings["export_cleanup_mask_delta"])
        self.assertFalse(settings["export_debug_metadata"])
        self.assertEqual(settings["automatic_output_target"], "single_archive")
        self.assertEqual(settings["automatic_output_image_format"], "webp")
        self.assertEqual(settings["automatic_output_archive_format"], "zip")
        self.assertEqual(settings["automatic_output_archive_image_format"], "jpg")
        self.assertEqual(settings["automatic_output_archive_compression_level"], 7)
        self.assertEqual(settings["project_autosave_interval_min"], 5)
        self.assertEqual(settings["project_autosave_folder"], "/tmp/projects")
        self.assertTrue(settings["auto_export_source_txt"])
        self.assertFalse(settings["auto_export_source_md"])
        self.assertTrue(settings["auto_export_translation_txt"])
        self.assertFalse(settings["auto_export_translation_md"])


if __name__ == "__main__":
    unittest.main()
