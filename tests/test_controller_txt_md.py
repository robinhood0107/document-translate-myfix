from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from controller import ComicTranslate


class _FakeSettingsPage:
    def __init__(self, export_settings: dict) -> None:
        self._export_settings = export_settings

    def get_export_settings(self) -> dict:
        return dict(self._export_settings)


class ControllerTxtMdTests(unittest.TestCase):
    def test_run_txt_md_auto_exports_writes_only_selected_targets(self) -> None:
        controller = SimpleNamespace()
        controller.settings_page = _FakeSettingsPage(
            {
                "auto_export_source_txt": True,
                "auto_export_source_md": False,
                "auto_export_translation_txt": False,
                "auto_export_translation_md": True,
            }
        )
        controller._ensure_txt_md_ready = mock.Mock(return_value=True)
        controller._write_txt_md_exchange = mock.Mock()
        controller._txt_md_auto_save_path = mock.Mock(
            side_effect=lambda target, suffix: f"/tmp/{target}{suffix}"
        )
        controller.tr = lambda text: text

        ComicTranslate._run_txt_md_auto_exports(controller, ["page-a.png"])

        controller._ensure_txt_md_ready.assert_called_once_with()
        self.assertEqual(
            controller._write_txt_md_exchange.call_args_list,
            [
                mock.call("source", ".txt", ["page-a.png"], save_path="/tmp/source.txt"),
                mock.call("translation", ".md", ["page-a.png"], save_path="/tmp/translation.md"),
            ],
        )

    def test_run_txt_md_auto_exports_stops_when_ready_check_fails(self) -> None:
        controller = SimpleNamespace()
        controller.settings_page = _FakeSettingsPage(
            {
                "auto_export_source_txt": True,
                "auto_export_source_md": True,
                "auto_export_translation_txt": True,
                "auto_export_translation_md": True,
            }
        )
        controller._ensure_txt_md_ready = mock.Mock(return_value=False)
        controller._write_txt_md_exchange = mock.Mock()
        controller._txt_md_auto_save_path = mock.Mock()
        controller.tr = lambda text: text

        ComicTranslate._run_txt_md_auto_exports(controller, ["page-a.png"])

        controller._write_txt_md_exchange.assert_not_called()


if __name__ == "__main__":
    unittest.main()
