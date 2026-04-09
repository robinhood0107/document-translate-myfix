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
            project_autosave_interval_spinbox=_Spin(5),
            project_autosave_folder_input=_Text(""),
        )

    def window(self):
        return None


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
        self.assertEqual(settings["project_autosave_interval_min"], 5)
        self.assertEqual(settings["project_autosave_folder"], "/tmp/projects")


if __name__ == "__main__":
    unittest.main()
